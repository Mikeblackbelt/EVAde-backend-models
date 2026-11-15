"""Deep sleep staging on synthetic Sleep-EDFx-like EDF files (no real data): EDF reading, labels and trimming,
metrics, then the whole pipeline on CPU: pack, a 1-epoch smoke train (kfold, beta5, final), export and ONNX parity,
plus a SIGUSR1-style interrupt and resume."""
import json
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from evade.training.deep import sleep_pack as SP  # noqa: E402
from evade.training.deep import sleep_train as ST  # noqa: E402

FS = 100
STAGE_TEXT = {0: "Sleep stage W", 1: "Sleep stage 1", 2: "Sleep stage 2", 3: "Sleep stage 3", 4: "Sleep stage R"}


# ---------------------------------------------------------------- a small EDF / EDF+ writer for the tests
def _f(v, w):
    s = str(v)
    assert len(s) <= w, (s, w)
    return s.ljust(w).encode("latin-1")


def write_edf(path: Path, signals: list[dict], record_s: float, n_records: int, tals: list[bytes] | None = None):
    """signals: dicts with label, fs, data (float, physical), pmin, pmax. tals: annotation TALs for record 0."""
    sigs = list(signals)
    ann_bytes = 0
    if tals is not None:
        body = b"".join(tals)
        ann_bytes = len(body) + 64
        ann_bytes += ann_bytes % 2
        sigs.append({"label": "EDF Annotations", "fs": None, "pmin": -1, "pmax": 1, "unit": ""})
    ns = len(sigs)
    nsamp = [int(round(s["fs"] * record_s)) if s["fs"] else ann_bytes // 2 for s in sigs]
    h = (_f("0", 8) + _f("X X X X", 80) + _f("Startdate 01-JAN-1989 X X X", 80) + _f("01.01.89", 8)
         + _f("00.00.00", 8) + _f(256 * (ns + 1), 8) + _f("EDF+C" if tals is not None else "", 44)
         + _f(n_records, 8) + _f(int(record_s) if float(record_s).is_integer() else record_s, 8) + _f(ns, 4))
    h += b"".join(_f(s["label"], 16) for s in sigs) + b"".join(_f("", 80) for _ in sigs)
    h += b"".join(_f(s.get("unit", "uV"), 8) for s in sigs)
    h += b"".join(_f(s["pmin"], 8) for s in sigs) + b"".join(_f(s["pmax"], 8) for s in sigs)
    h += b"".join(_f(-32768, 8) for _ in sigs) + b"".join(_f(32767, 8) for _ in sigs)
    h += b"".join(_f("", 80) for _ in sigs) + b"".join(_f(n, 8) for n in nsamp) + b"".join(_f("", 32) for _ in sigs)
    out = [h]
    for r in range(n_records):
        for s, n in zip(sigs, nsamp):
            if s["fs"] is None:
                tal = f"+{r * record_s:g}\x14\x14\x00".encode() + (b"".join(tals) if r == 0 else b"")
                out.append(tal.ljust(n * 2, b"\x00"))
            else:
                x = np.asarray(s["data"][r * n:(r + 1) * n], float)
                dig = np.round((x - s["pmin"]) / (s["pmax"] - s["pmin"]) * 65535 - 32768)
                out.append(np.clip(dig, -32768, 32767).astype("<i2").tobytes())
    path.write_bytes(b"".join(out))


def tal(onset, dur, text):
    return f"+{onset:g}\x15{dur:g}\x14{text}\x14\x00".encode()


def synth_night(rng, stages: np.ndarray):
    """Two EEG channels whose spectrum depends on the stage (so a model can learn something)."""
    freq = {0: 10.0, 1: 6.0, 2: 13.0, 3: 1.0, 4: 7.5}
    amp = {0: 20.0, 1: 30.0, 2: 25.0, 3: 80.0, 4: 20.0}
    t = np.arange(3000) / FS
    x = []
    for s in stages:
        s = max(int(s), 0)
        base = amp[s] * np.sin(2 * np.pi * freq[s] * t + rng.uniform(0, 6.3))
        x.append(np.stack([base + 5 * rng.standard_normal(3000), 0.6 * base + 5 * rng.standard_normal(3000)]))
    return np.concatenate(x, axis=1)            # (2, n*3000)


def write_night(folder: Path, name7: str, rng, pre=65, sleep=40, post=65):
    sleep_stages = np.repeat([1, 2, 3, 2, 4, 2, 3, 4], int(np.ceil(sleep / 8)))[:sleep]
    stages = np.concatenate([np.zeros(pre, int), sleep_stages, np.zeros(post, int)])
    sig = synth_night(rng, stages)
    n_rec = len(stages)
    marker = np.zeros(n_rec * 30)                        # a 1 Hz channel interleaved with the EEG
    write_edf(folder / f"{name7}0-PSG.edf",
              [{"label": "EEG Fpz-Cz", "fs": FS, "data": sig[0], "pmin": -500, "pmax": 500},
               {"label": "EEG Pz-Oz", "fs": FS, "data": sig[1], "pmin": -500, "pmax": 500},
               {"label": "Event marker", "fs": 1, "data": marker, "pmin": -1, "pmax": 1}], 30, n_rec)
    tals, i = [], 0
    while i < len(stages):                                # run-length annotations, as the real hypnograms
        j = i
        while j < len(stages) and stages[j] == stages[i]:
            j += 1
        tals.append(tal(i * 30, (j - i) * 30, STAGE_TEXT[int(stages[i])]))
        i = j
    tals.append(tal((pre + 5) * 30, 30, "Movement time"))  # overwrites one sleep epoch: must be dropped
    tals.append(tal(len(stages) * 30, 3000, "Sleep stage ?"))
    write_edf(folder / f"{name7}C-Hypnogram.edf", [], 0, 1, tals)
    return stages


@pytest.fixture(scope="module")
def raw(tmp_path_factory):
    d = tmp_path_factory.mktemp("sleep_raw") / "sleep-cassette"
    d.mkdir()
    rng = np.random.default_rng(0)
    # 6 nights, 5 subjects: SC400 has two nights (they must stay in one fold); SC405 is not a beta subject
    for name in ["SC4001E", "SC4002E", "SC4011E", "SC4021E", "SC4031E", "SC4051E"]:
        write_night(d, name, rng)
    return d.parent


# ---------------------------------------------------------------- EDF reading and labels
def test_edf_signals_and_annotations_roundtrip(tmp_path):
    rng = np.random.default_rng(1)
    stages = write_night(tmp_path, "SC4991E", rng, pre=3, sleep=8, post=3)
    sig, fs = SP.read_signals(tmp_path / "SC4991E0-PSG.edf", ["EEG Fpz-Cz", "EEG Pz-Oz"])
    ref = synth_night(np.random.default_rng(1), stages)    # same rng sequence as write_night
    assert fs == 100.0 and sig.shape == ref.shape
    assert np.abs(sig - ref).max() < 0.02                   # 1000 uV / 65535 quantization
    on, du, de = SP.read_annotations(tmp_path / "SC4991EC-Hypnogram.edf")
    assert de[0] == "Sleep stage W" and on[0] == 0 and du[0] == 90
    assert "Movement time" in de and de[-1] == "Sleep stage ?"


def test_edf_reader_matches_mne(tmp_path):
    mne = pytest.importorskip("mne")
    rng = np.random.default_rng(2)
    write_night(tmp_path, "SC4992E", rng, pre=3, sleep=8, post=3)
    psg = tmp_path / "SC4992E0-PSG.edf"
    raw = mne.io.read_raw_edf(psg, include=["EEG Fpz-Cz", "EEG Pz-Oz"], preload=True, verbose="error")
    ours, _ = SP.read_signals(psg, ["EEG Fpz-Cz", "EEG Pz-Oz"])
    assert np.allclose(raw.get_data() * 1e6, ours, atol=1e-3)
    ann = mne.read_annotations(tmp_path / "SC4992EC-Hypnogram.edf")
    on, du, de = SP.read_annotations(tmp_path / "SC4992EC-Hypnogram.edf")
    # mne sorts by onset; ours keeps file order (the overwrite rule depends on it), so compare sorted
    theirs = sorted(zip(np.round(ann.onset, 3), np.round(ann.duration, 3), ann.description))
    assert theirs == sorted(zip(np.round(on, 3), np.round(du, 3), de))


def test_labels_and_trim_match_beta():
    eeg = pytest.importorskip("evade.preprocessing.eeg")
    on = np.array([0, 3000, 3600, 4500, 9000, 9030, 12000.0, 12600])
    du = np.array([3000, 600, 900, 4500, 30, 2970, 600.0, 900])
    de = ["Sleep stage W", "Sleep stage 1", "Sleep stage 4", "Sleep stage R", "Movement time", "Sleep stage 3",
          "Sleep stage ?", "Sleep stage W"]
    ours = SP.hypnogram_to_epochs(on, du, de, 450)
    assert np.array_equal(ours, eeg.hypnogram_to_epochs(on, du, de, 450))
    keep = SP.trim_and_drop(ours)
    asleep = np.flatnonzero((ours >= 1) & (ours <= 4))
    assert keep[0] == asleep[0] - 60 and keep[-1] == min(asleep[-1] + 60, 449)
    assert (ours[keep] >= 0).all() and 300 not in keep      # the movement epoch (9000 s) is dropped


def test_scores_match_sklearn():
    from sklearn.metrics import cohen_kappa_score, f1_score
    rng = np.random.default_rng(3)
    y = rng.integers(0, 5, 500)
    p = np.where(rng.random(500) < 0.6, y, rng.integers(0, 4, 500))   # never predicts REM in the noise part
    s = ST.scores(ST.confusion(y, p))
    assert s["kappa"] == pytest.approx(cohen_kappa_score(y, p))
    assert s["macro_f1"] == pytest.approx(f1_score(y, p, average="macro"))
    ci = ST.subject_bootstrap(np.stack([ST.confusion(y[i::5], p[i::5]) for i in range(5)]), n_boot=200)
    assert ci["kappa"]["lo"] <= s["kappa"] <= ci["kappa"]["hi"]


# ---------------------------------------------------------------- in-model normalization (the serving rule)
def test_epoch_robust_z_matches_numpy_and_needs_no_night():
    from evade.training.deep.sleep_models import build, epoch_robust_z
    rng = np.random.default_rng(4)
    x = (rng.standard_normal((6, 2, 3000)) * 30 + 5).astype(np.float32)
    x[2] = 0.0                                                    # flat epoch: floor 0.1, no NaN
    med = np.median(x, axis=2, keepdims=True)
    mad = np.median(np.abs(x - med), axis=2, keepdims=True)
    ref = np.clip((x - med) / np.maximum(1.4826 * mad, 0.1), -20, 20)
    got = epoch_robust_z(torch.from_numpy(x)).numpy()
    assert np.allclose(got, ref, atol=1e-4) and np.isfinite(got).all()
    # an epoch is encoded the same whether it comes alone, in a chunk, or scaled by a gain (unit-free)
    m = build("cnn_bigru", 2).eval()
    with torch.no_grad():
        full = m.encode(torch.from_numpy(x))
        alone = m.encode(torch.from_numpy(x[4:5]))
        gain = m.encode(torch.from_numpy(x[4:5] * 3.0 + 10.0))
    assert torch.allclose(full[4:5], alone, atol=1e-5) and torch.allclose(alone, gain, atol=1e-4)


# ---------------------------------------------------------------- end to end
def _train(npz, run, protocol, *extra):
    return ST.main(["--data", str(npz), "--run", str(run), "--protocol", protocol, "--seq-len", "10", "--bs", "4",
                    "--eval-stride", "5", "--n-boot", "100", "--resume", *extra])


def test_pipeline_end_to_end(raw, tmp_path):
    npz = tmp_path / "sleep_sc.npz"
    assert SP.main(["--src", str(raw), "--out", str(npz)]) == 0
    z = np.load(npz, allow_pickle=False)
    assert z["X"].dtype == np.float16 and z["X"].shape[1:] == (2, 3000)
    assert len(set(z["night"])) == 6 and len(set(z["subject"])) == 5
    # 65 W + 40 sleep + 65 W -> 60 + 40 + 60 kept, minus the movement epoch
    assert (z["night"] == "SC4001E").sum() == 159
    assert set(np.unique(z["y"])) <= {0, 1, 2, 3, 4}

    run = tmp_path / "run"
    assert _train(npz, run / "kfold", "kfold", "--folds", "3", "--smoke") == 0
    assert _train(npz, run / "beta5", "beta5", "--val-frac", "0", "--smoke") == 0
    assert _train(npz, run / "final", "final", "--smoke") == 0
    m = json.loads((run / "kfold" / "metrics.json").read_text())
    oof = m["out_of_fold"]
    for k in ("kappa", "macro_f1", "accuracy"):
        assert k in oof["pooled"] and {"lo", "hi"} <= set(oof["ci95"][k])
    assert set(oof["pooled"]["per_class_f1"]) == set(SP.CLASSES)
    assert m["baseline_beta"]["eva3_hgb"]["subject_mean_kappa"] == 0.669
    assert m["protocol"]["splits_sha256"] and (run / "kfold" / "splits.csv").exists()
    assert json.loads((run / "beta5" / "metrics.json").read_text())["protocol"]["subjects"][0] == "SC400"

    from evade.training.deep import sleep_export as SE
    out = run / "export" / "sleep_deep"
    assert SE.main(["--run", str(run), "--out", str(out), "--model-id", "EVA-99"]) == 0
    man = json.loads((out / "manifest.json").read_text())
    import hashlib
    for f, h in man["files_sha256"].items():
        assert hashlib.sha256((out / f).read_bytes()).hexdigest() == h
    # the fields the API's sleep_deep adapter reads (api-model-stubs: evade/models/deep_tasks.py SleepAdapter,
    # evade/models/manifest.py validate), with the values it accepts
    assert man["task"] == "sleep_deep" and man["status"] == "draft" and man["format"] in ("onnx", "torchscript")
    inp, outp = man["input"], man["output"]
    assert inp["kind"] == "tensor" and inp["modality"] == "eeg" and inp["fs_hz"] == 100
    assert inp["channels"] == ["Fpz-Cz", "Pz-Oz"] and inp["tensor_shape"] == ["batch", "seq", 2, 3000]
    assert inp["normalization"]["type"] == "none"                      # the API sends raw uV
    assert inp["normalization"]["in_model"]["type"] == "per_epoch_channel_robust_z"
    assert outp["type"] in ("binary", "multiclass", "multilabel", "regression")
    assert outp["activation"] == "softmax" and outp["classes"] == SP.CLASSES
    assert outp["operating_points"]
    try:
        import onnxruntime  # noqa: F401
        has_ort = True
    except ImportError:
        has_ort = False
    if man["format"] == "onnx" and has_ort:
        assert man["onnx_parity"]["checked"] and man["onnx_parity"]["ok"], man["onnx_parity"]


def test_interrupt_and_resume(raw, tmp_path, monkeypatch):
    npz = tmp_path / "sleep_sc.npz"
    SP.main(["--src", str(raw), "--out", str(npz), "--limit", "3"])   # SC400, SC401: one train, one val subject
    run = tmp_path / "final"
    real_predict = ST.predict

    def predict_then_signal(*args, **kw):        # SIGUSR1 arrives during the epoch-0 validation
        ST.STOP["flag"] = True
        return real_predict(*args, **kw)

    monkeypatch.setattr(ST, "predict", predict_then_signal)
    args = ["--epochs", "2", "--patience", "0", "--seq-len", "10"]
    with pytest.raises(SystemExit) as e:
        _train(npz, run, "final", *args)
    assert e.value.code == 99 and (run / "final" / "last.pt").exists()
    ck = torch.load(run / "final" / "last.pt", weights_only=True)
    assert ck["epoch"] == 0
    monkeypatch.setattr(ST, "predict", real_predict)
    ST.STOP["flag"] = False
    assert _train(npz, run, "final", *args) == 0
    res = json.loads((run / "final" / "result.json").read_text())
    assert res["epochs_run"] == 2 and not (run / "final" / "last.pt").exists()
