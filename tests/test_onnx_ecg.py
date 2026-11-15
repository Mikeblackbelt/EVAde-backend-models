"""Tests for the deep ECG serving adapter, on a tiny synthetic model exported with the real export.py (no datasets).

A small resnet1d_wang with random weights and random BatchNorm statistics is saved as seed0/best.pt in a fake run
folder, exported with evade.training.deep.export (ONNX, and TorchScript by forcing the fallback), loaded with
EcgDeepModel and compared with the training code's own sliding-window predict().
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from unittest import mock

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("scipy")
pytest.importorskip("sklearn")      # evade.training.deep.train imports it

from evade.models.onnx_ecg import (EcgDeepModel, EcgInputError, ModelIntegrityError, canonical_lead)  # noqa: E402
from evade.training.deep import export as EX  # noqa: E402
from evade.training.deep.models import XResNet1d, build  # noqa: E402
from evade.training.deep.train import predict as train_predict  # noqa: E402

LEADS = ["I", "II", "III", "AVR", "AVL", "AVF", "V1", "V2", "V3", "V4", "V5", "V6"]
CLASSES = ["NORM", "MI", "STTC", "CD", "HYP"]
FS, CROP, STRIDE, REC = 100, 250, 125, 1000
ARCH = "resnet1d_wang"
REPO = Path(__file__).resolve().parents[1]


def _tiny_build(arch, c_in, n_out):
    """build() with xresnet1d50 shrunk to one block per stage: same ops as xresnet1d101, fast to export on CPU."""
    return XResNet1d([1, 1, 1, 1], c_in, n_out) if arch == "xresnet1d50" else build(arch, c_in, n_out)


def _fake_run(root: Path, arch: str = ARCH) -> tuple[Path, torch.nn.Module, np.ndarray, np.ndarray]:
    torch.manual_seed(0)
    model = _tiny_build(arch, len(LEADS), len(CLASSES))
    g = torch.Generator().manual_seed(1)
    with torch.no_grad():                         # non-trivial BatchNorm so the parity check means something
        for mod in model.modules():
            if isinstance(mod, torch.nn.BatchNorm1d):
                mod.running_mean.copy_(0.1 * torch.randn(mod.num_features, generator=g))
                mod.running_var.copy_(0.5 + torch.rand(mod.num_features, generator=g))
                mod.weight.copy_(0.5 + torch.rand(mod.num_features, generator=g))
                mod.bias.copy_(0.1 * torch.randn(mod.num_features, generator=g))
    model.eval()
    rng = np.random.default_rng(0)
    mean = (0.02 * rng.standard_normal(len(LEADS))).astype(np.float32)
    std = (0.1 + 0.2 * rng.random(len(LEADS))).astype(np.float32)
    run = root / "run"
    (run / "seed0").mkdir(parents=True)
    torch.save(model.state_dict(), run / "seed0" / "best.pt")
    ci = {"lo": 0.5, "hi": 0.6, "n_boot": 10, "unit": "patient", "n_patients": 10}
    metrics = {"task": "ecg_ptbxl_superclass", "arch": arch, "leads": LEADS, "classes": CLASSES,
               "protocol": "synthetic test", "n": {"train": 1, "val": 1, "test": 1}, "seeds": [0],
               "selected_seed": 0, "selected": {"test_macro_auroc": 0.5, "ci95": ci},
               "ensemble": {"test_macro_auroc": 0.5, "ci95": ci}, "baseline_beta_eva6_macro_auroc": 0.867,
               "normalization": {"mean": mean.tolist(), "std": std.tolist()},
               "window": {"fs_hz": FS, "crop_samples": CROP, "stride_samples": STRIDE, "record_samples": REC},
               "smoke": True}
    (run / "metrics.json").write_text(json.dumps(metrics))
    return run, model, mean, std


def _export(run: Path, out: Path, arch: str = ARCH) -> dict:
    argv = ["export", "--run", str(run), "--arch", arch, "--out", str(out), "--model-id", "EVA-TEST"]
    with mock.patch.object(sys, "argv", argv), mock.patch.dict(os.environ, {"EVADE_CODE_COMMIT": "test"}),             mock.patch.object(EX, "build", _tiny_build):
        assert EX.main() == 0
    return json.loads((out / "manifest.json").read_text())


@pytest.fixture(scope="module")
def exported(tmp_path_factory):
    pytest.importorskip("onnxruntime")
    pytest.importorskip("onnx")
    root = tmp_path_factory.mktemp("ecgdeep")
    run, model, mean, std = _fake_run(root)
    man = _export(run, root / "onnx")
    return {"dir": root / "onnx", "manifest": man, "model": model, "mean": mean, "std": std, "root": root, "run": run}


@pytest.fixture(scope="module")
def exported_ts(exported):
    out = exported["root"] / "ts"

    def boom(*a, **k):
        raise RuntimeError("forced: no ONNX exporter")
    with mock.patch.object(torch.onnx, "export", boom):
        man = _export(exported["run"], out)
    return {"dir": out, "manifest": man}


@pytest.fixture(scope="module")
def exported_xresnet(tmp_path_factory):
    """XResNet1d (one block per stage): the served xresnet1d101's stem, bottleneck blocks, odd-length ceil-mode
    avg-pool shortcuts and head, exported through export.py under the arch name xresnet1d50."""
    pytest.importorskip("onnxruntime")
    pytest.importorskip("onnx")
    root = tmp_path_factory.mktemp("ecgdeep_x")
    run, model, mean, std = _fake_run(root, "xresnet1d50")
    man = _export(run, root / "onnx", "xresnet1d50")
    return {"dir": root / "onnx", "manifest": man, "model": model, "mean": mean, "std": std}


def _ecg(seconds=10.0, fs=FS, seed=0) -> np.ndarray:
    """Band-limited synthetic 12-lead signal in mV (sum of sines below 20 Hz), defined analytically at any fs."""
    rng = np.random.default_rng(seed)
    t = np.arange(int(round(seconds * fs))) / fs
    f = rng.uniform(0.5, 20.0, (len(LEADS), 6))
    a = rng.uniform(0.05, 0.4, (len(LEADS), 6))
    ph = rng.uniform(0, 2 * np.pi, (len(LEADS), 6))
    return (a[:, :, None] * np.sin(2 * np.pi * f[:, :, None] * t + ph[:, :, None])).sum(1).astype(np.float32)


def _torch_reference(model, x_mv: np.ndarray, mean, std) -> np.ndarray:
    xn = (x_mv - mean[:, None]) / std[:, None]
    return train_predict(model, torch.from_numpy(xn[None].astype(np.float32)), CROP, STRIDE)[0]


def _copy(src: Path, dst: Path) -> Path:
    shutil.copytree(src, dst)
    return dst


# ---- export contract and parity ---------------------------------------------------------------------------------

def test_export_writes_onnx_and_manifest_fields(exported):
    man = exported["manifest"]
    assert man["format"] == "onnx" and man["weights"] == "model.onnx"   # not the silent TorchScript fallback
    assert set(man["files_sha256"]) == {"model.onnx", "feature_names.json", "metrics.json"}
    assert man["input"]["normalization"]["type"] == "per_lead_zscore"
    assert man["input"]["tensor_shape"] == ["batch", 12, CROP]


def test_onnx_matches_torch_within_1e4(exported):
    m = EcgDeepModel(exported["dir"])
    assert m.format == "onnx"
    for seed in range(3):
        x = _ecg(seed=seed)
        out = m.predict(x, FS, LEADS)
        ref = _torch_reference(exported["model"], x, exported["mean"], exported["std"])
        got = np.array([out.probabilities[c] for c in CLASSES])
        assert out.n_windows == len(range(0, REC - CROP + 1, STRIDE)) == 7
        np.testing.assert_allclose(got, ref, atol=1e-4, rtol=0)
        assert 0.0 < got.min() and got.max() < 1.0


def test_xresnet_onnx_matches_torch_within_1e4(exported_xresnet):
    e = exported_xresnet
    assert e["manifest"]["format"] == "onnx" and e["manifest"]["arch"] == "xresnet1d50"
    m = EcgDeepModel(e["dir"])
    x = _ecg(seed=7)
    got = np.array([m.predict(x, FS, LEADS).probabilities[c] for c in CLASSES])
    np.testing.assert_allclose(got, _torch_reference(e["model"], x, e["mean"], e["std"]), atol=1e-4, rtol=0)


def test_torchscript_fallback_loads_and_matches(exported, exported_ts):
    assert exported_ts["manifest"]["format"] == "torchscript"
    m = EcgDeepModel(exported_ts["dir"])
    x = _ecg(seed=5)
    got = np.array([m.predict(x, FS, LEADS).probabilities[c] for c in CLASSES])
    ref = _torch_reference(exported["model"], x, exported["mean"], exported["std"])
    np.testing.assert_allclose(got, ref, atol=1e-4, rtol=0)


def test_result_dict_is_review_flag_not_diagnosis(exported):
    d = EcgDeepModel(exported["dir"]).predict(_ecg(), FS, LEADS).to_dict()
    assert set(d["probabilities"]) == set(CLASSES)
    assert d["flags"] is None                                   # export.py writes no operating points yet
    codes = {n["code"] for n in d["quality"]["notes"]}
    assert {"draft_model", "no_operating_points"} <= codes
    assert d["model"]["model_id"] == "EVA-TEST" and d["model"]["status"] == "draft"
    assert "not a diagnosis" in d["disclaimer"].lower()
    json.dumps(d)                                               # JSON-serialisable for the API


# ---- integrity ------------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("victim", ["model.onnx", "metrics.json", "feature_names.json"])
def test_sha_mismatch_refused(exported, tmp_path, victim):
    d = _copy(exported["dir"], tmp_path / "m")
    with open(d / victim, "ab") as f:
        f.write(b" ")
    with pytest.raises(ModelIntegrityError, match="sha256 mismatch"):
        EcgDeepModel(d)


def test_missing_file_and_unsafe_name_refused(exported, tmp_path):
    d = _copy(exported["dir"], tmp_path / "m")
    (d / "metrics.json").unlink()
    with pytest.raises(ModelIntegrityError, match="missing"):
        EcgDeepModel(d)
    d2 = _copy(exported["dir"], tmp_path / "m2")
    man = json.loads((d2 / "manifest.json").read_text())
    man["files_sha256"]["../outside.json"] = "0" * 64
    (d2 / "manifest.json").write_text(json.dumps(man))
    with pytest.raises(ModelIntegrityError, match="unsafe"):
        EcgDeepModel(d2)


def test_pinned_manifest_sha(exported):
    good = EcgDeepModel(exported["dir"]).manifest_sha256
    EcgDeepModel(exported["dir"], expected_manifest_sha256=good)
    with pytest.raises(ModelIntegrityError, match="pinned"):
        EcgDeepModel(exported["dir"], expected_manifest_sha256="0" * 64)


def test_operating_points_give_flags(exported, tmp_path):
    d = _copy(exported["dir"], tmp_path / "m")
    man = json.loads((d / "manifest.json").read_text())
    man["output"]["operating_points"] = {"NORM": 0.0, "MI": {"threshold": 1.0, "target": "test"}}
    (d / "manifest.json").write_text(json.dumps(man))           # manifest is not in files_sha256
    out = EcgDeepModel(d).predict(_ecg(), FS, LEADS)
    assert out.flags == {"NORM": True, "MI": False}


# ---- input handling -------------------------------------------------------------------------------------------------

def test_wrong_lead_count_error(exported):
    m = EcgDeepModel(exported["dir"])
    with pytest.raises(EcgInputError, match="expected 12 leads"):
        m.predict(_ecg()[:8], FS)
    with pytest.raises(EcgInputError, match="missing leads"):
        m.predict(_ecg()[:11], FS, LEADS[:11])
    with pytest.raises(EcgInputError, match="lead names given"):
        m.predict(_ecg(), FS, LEADS[:11])


def test_lead_aliases_and_reordering(exported):
    m = EcgDeepModel(exported["dir"])
    x = _ecg(seed=2)
    base = m.predict(x, FS, LEADS).probabilities
    names = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
    assert m.predict(x, FS, names).probabilities == base
    assert canonical_lead("avr") == canonical_lead("AVR") == canonical_lead("aVR") == "AVR"
    assert canonical_lead("DII") == canonical_lead("lead II") == "II"
    perm = np.random.default_rng(0).permutation(12)
    out = m.predict(x[perm], FS, [names[i] for i in perm])
    assert out.info["lead_order"] == "reordered"
    np.testing.assert_allclose([out.probabilities[c] for c in CLASSES], [base[c] for c in CLASSES], atol=1e-6)
    with pytest.raises(EcgInputError, match="duplicate"):
        m.predict(x, FS, ["I"] * 12)
    noted = m.predict(x, FS)
    assert noted.info["lead_order"] == "assumed"


def test_resampling_from_500hz(exported):
    m = EcgDeepModel(exported["dir"])
    x100, x500 = _ecg(fs=100, seed=3), _ecg(fs=500, seed=3)
    assert x500.shape == (12, 5000)
    p100, p500 = m.preprocess(x100, 100, LEADS), m.preprocess(x500, 500.0, LEADS)
    assert p500.x.shape == p100.x.shape == (12, REC)
    assert p500.info["samples_after_resample"] == REC and p500.info["padded_samples"] == 0
    assert any(n.code == "resampled" for n in p500.notes) and not any(n.code == "resampled" for n in p100.notes)
    inner = slice(50, REC - 50)                                 # polyphase edge effects stay near the ends
    err = np.abs(p500.x[:, inner] - p100.x[:, inner]).max() / np.abs(p100.x[:, inner]).max()
    assert err < 0.01, err
    a = m.predict(x100, 100, LEADS).probabilities
    b = m.predict(x500, 500, LEADS).probabilities
    assert max(abs(a[c] - b[c]) for c in CLASSES) < 0.02


def test_length_policies(exported):
    m = EcgDeepModel(exported["dir"])
    short, long_ = _ecg(seconds=8.0), _ecg(seconds=12.0)
    out = m.predict(short, FS, LEADS)
    assert out.info["padded_samples"] == 200 and any(n.code == "padded" and n.severity == "warning" for n in out.notes)
    assert out.n_windows == 7 and not out.quality_ok
    v = m.predict(short, FS, LEADS, length_policy="valid")
    assert v.n_windows == len(range(0, 800 - CROP + 1, STRIDE)) and any(n.code == "short_record" for n in v.notes)
    c = m.predict(long_, FS, LEADS)
    assert c.info["cropped_samples"] == 200 and any(n.code == "cropped" for n in c.notes)
    with pytest.raises(EcgInputError, match="expected 1000 samples"):
        m.predict(short, FS, LEADS, length_policy="strict")
    with pytest.raises(EcgInputError, match="at least"):
        m.predict(_ecg(seconds=2.0), FS, LEADS)


def test_bad_values_and_quality_notes(exported):
    m = EcgDeepModel(exported["dir"])
    x = _ecg()
    bad = x.copy(); bad[0, 10] = np.nan
    with pytest.raises(EcgInputError, match="non-finite"):
        m.predict(bad, FS, LEADS)
    with pytest.raises(EcgInputError, match="2-D"):
        m.predict(x[0], FS)
    with pytest.raises(EcgInputError, match="fs_hz"):
        m.predict(x, 0, LEADS)
    flat = x.copy(); flat[3] = 0.0
    codes = {n.code for n in m.predict(flat, FS, LEADS).notes}
    assert "flat_leads" in codes
    uv = m.predict(x * 1000.0, FS, LEADS)                       # microvolts passed as if mV
    assert any(n.code == "high_amplitude" for n in uv.notes)
    same = m.predict(x * 1000.0, FS, LEADS, unit="uV").probabilities
    base = m.predict(x, FS, LEADS).probabilities
    assert max(abs(same[c] - base[c]) for c in CLASSES) < 1e-5
    clip = x.copy(); clip[1] = np.clip(clip[1], -0.1, 0.1)
    assert any(n.code == "possible_clipping" for n in m.predict(clip, FS, LEADS).notes)


def test_input_rate_bounds(exported):
    m = EcgDeepModel(exported["dir"])
    for fs in (49.9, 1000.1, 20000.0):
        with pytest.raises(EcgInputError, match="between 50 and 1000 Hz"):
            m.preprocess(_ecg(fs=100), fs, LEADS)
    for fs in (50, 1000):                                      # the bounds themselves are accepted
        p = m.preprocess(_ecg(fs=fs, seed=4), fs, LEADS)
        assert p.x.shape == (12, REC) and p.info["padded_samples"] == 0
    with pytest.raises(EcgInputError, match="factors of at most 100"):
        m.preprocess(_ecg(fs=50.5), 50.5, LEADS)               # 100/50.5 needs up=196 with down<=100
    assert m._ratio(1000.0) == (1, 10) and m._ratio(50.0) == (2, 1)


def test_input_length_and_row_caps(exported):
    m = EcgDeepModel(exported["dir"])
    x60 = _ecg(seconds=60.0, fs=500, seed=5)                    # exactly 60 s: accepted, cropped before resampling
    p = m.preprocess(x60, 500, LEADS)
    assert p.x.shape == (12, REC) and p.info["cropped_samples"] == 6000 - REC
    assert p.info["samples_after_resample"] == REC + 200        # the central 10 s plus 1 s each side, not 60 s
    assert [n.code for n in p.notes].count("cropped") == 1 and not any(n.code == "padded" for n in p.notes)
    over = np.concatenate([x60, x60[:, :1]], axis=1)
    with pytest.raises(EcgInputError, match="at most 60 s"):
        m.preprocess(over, 500, LEADS)
    with pytest.raises(EcgInputError, match="at most 60 s"):
        m.preprocess(_ecg(seconds=61.0, fs=100), 100, LEADS, length_policy="strict")
    with pytest.raises(EcgInputError, match="expected 1000 samples"):
        m.preprocess(_ecg(seconds=30.0, fs=500), 500, LEADS, length_policy="strict")   # strict is not pre-cropped
    with pytest.raises(EcgInputError, match="at most 64 leads"):
        m.preprocess(np.zeros((65, 1000), np.float32), 100, [f"X{i}" for i in range(65)])


def test_precrop_matches_resampling_the_whole_record(exported):
    """Cropping a long record before resampling gives the same model input as resampling all of it first."""
    from scipy.signal import resample_poly
    m = EcgDeepModel(exported["dir"])
    x = _ecg(seconds=30.0, fs=500, seed=6)
    p = m.preprocess(x, 500, LEADS)
    full = resample_poly(x.astype(np.float64), 1, 5, axis=1)
    off = (full.shape[1] - REC) // 2
    ref = (full[:, off:off + REC] - exported["mean"][:, None]) / exported["std"][:, None]
    assert np.abs(p.x - ref).max() / np.abs(ref).max() < 1e-3
    assert p.info["cropped_samples"] == 3000 - REC and p.info["samples_in"] == 15000


def test_module_imports_no_fastapi_or_torch():
    code = ("import sys, evade.models.onnx_ecg; "
            "bad = [m for m in ('fastapi', 'torch', 'onnxruntime', 'starlette') if m in sys.modules]; "
            "assert not bad, bad")
    subprocess.run([sys.executable, "-c", code], check=True, cwd=REPO)
