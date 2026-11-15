"""End-to-end CPU check of the deep ECG pipeline on synthetic data (no datasets):
synthetic PTB-XL-like WFDB folder -> ptbxl_pack -> 1-epoch train (resnet1d_wang) -> export (ONNX) -> EcgDeepModel.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("pandas")
pytest.importorskip("sklearn")
pytest.importorskip("onnxruntime")
pytest.importorskip("onnx")

from evade.models.onnx_ecg import EcgDeepModel  # noqa: E402
from evade.training.deep import export as EX  # noqa: E402
from evade.training.deep import ptbxl_pack as PK  # noqa: E402
from evade.training.deep import train as TR  # noqa: E402
from evade.training.deep.models import build  # noqa: E402

N_PER_FOLD = 8
CODES = ["NORM", "IMI", "NDT", "CLBBB", "LVH"]          # one diagnostic code per superclass
CLASS_OF = dict(zip(CODES, PK.SUPER))


def _write_wfdb(path: Path, x_mv: np.ndarray, gain: float = 1000.0) -> None:
    """Write a 12-lead format-16 WFDB record (header + .dat), as in PTB-XL records100."""
    name = path.name
    raw = np.round(x_mv * gain).astype("<i2")                # (12, 1000) -> interleaved (1000, 12)
    raw.T.tofile(path.with_suffix(".dat"))
    lines = [f"{name} 12 100 1000"]
    for lead in PK.LEADS:
        lines.append(f"{name}.dat 16 {gain:g}(0)/mV 16 0 0 0 0 {lead}")
    path.with_suffix(".hea").write_text("\n".join(lines) + "\n")


def _synthetic_ptbxl(src: Path) -> None:
    rng = np.random.default_rng(0)
    rows, t = [], np.arange(1000) / 100.0
    rec = src / "records100" / "00000"
    rec.mkdir(parents=True)
    ecg_id = 0
    for fold in range(1, 11):
        for _ in range(N_PER_FOLD):
            ecg_id += 1
            k = int(rng.integers(len(CODES)))
            f = 1.0 + 0.4 * k                                 # class-dependent rhythm so training has a signal
            x = 0.5 * np.sin(2 * np.pi * f * t)[None] + 0.05 * rng.standard_normal((12, 1000))
            stem = f"{ecg_id:05d}_lr"
            _write_wfdb(rec / stem, x.astype(np.float32))
            rows.append({"ecg_id": ecg_id, "patient_id": 1000 + ecg_id // 2, "strat_fold": fold,
                         "scp_codes": repr({CODES[k]: 100.0}), "filename_lr": f"records100/00000/{stem}"})
    rows.append({"ecg_id": ecg_id + 1, "patient_id": 1, "strat_fold": 1, "scp_codes": repr({"SR": 0.0}),
                 "filename_lr": "records100/00000/missing_lr"})   # no diagnostic superclass: dropped by the packer
    import pandas as pd
    pd.DataFrame(rows).set_index("ecg_id").to_csv(src / "ptbxl_database.csv")
    st = pd.DataFrame({"diagnostic": [1] * len(CODES) + [0], "diagnostic_class": list(PK.SUPER) + [None]},
                      index=CODES + ["SR"])
    st.to_csv(src / "scp_statements.csv")


def test_pack_train_export_serve_on_cpu(tmp_path):
    src, npz, run, out = tmp_path / "ptbxl", tmp_path / "ptbxl100.npz", tmp_path / "run", tmp_path / "export"
    _synthetic_ptbxl(src)
    with mock.patch.object(sys, "argv", ["pack", "--src", str(src), "--out", str(npz)]):
        assert PK.main() == 0
    z = np.load(npz, allow_pickle=False)
    assert z["X"].shape == (10 * N_PER_FOLD, 12, 1000) and z["X"].dtype == np.float16
    rec = PK.read_record(next((src / "records100" / "00000").glob("*.hea")))
    np.testing.assert_allclose(z["X"][0].astype(np.float32), rec, atol=2e-3)

    argv = ["train", "--data", str(npz), "--arch", "resnet1d_wang", "--seeds", "0", "--epochs", "1", "--bs", "16",
            "--run", str(run), "--n-boot", "20"]
    fake_signal = SimpleNamespace(SIGUSR1=None, signal=lambda *a: None)   # SIGUSR1 does not exist on Windows
    with mock.patch.object(sys, "argv", argv), mock.patch.object(TR, "signal", fake_signal):
        assert TR.main() == 0
    m = json.loads((run / "metrics.json").read_text())
    assert m["selected_seed"] == 0 and m["window"]["crop_samples"] == 250
    assert m["selected"]["ci95"]["unit"] == "patient" and m["baseline_beta_eva6_macro_auroc"] == 0.867
    assert (run / "seed0" / "best.pt").exists() and not (run / "seed0" / "last.pt").exists()

    with mock.patch.object(sys, "argv", ["export", "--run", str(run), "--arch", "resnet1d_wang", "--out", str(out)]), \
            mock.patch.dict(os.environ, {"EVADE_CODE_COMMIT": "test"}):
        assert EX.main() == 0
    man = json.loads((out / "manifest.json").read_text())
    assert man["format"] == "onnx" and man["headline"]["beta_eva6"] == 0.867

    served = EcgDeepModel(out)
    x = z["X"][-1].astype(np.float32)                         # a test-fold record, in mV
    res = served.predict(x, 100, PK.LEADS)
    model = build("resnet1d_wang", 12, 5)
    model.load_state_dict(torch.load(run / "seed0" / "best.pt", map_location="cpu", weights_only=True))
    mean, std = np.asarray(m["normalization"]["mean"]), np.asarray(m["normalization"]["std"])
    xn = ((x - mean[:, None]) / std[:, None]).astype(np.float32)
    ref = TR.predict(model, torch.from_numpy(xn[None]), 250, 125)[0]
    np.testing.assert_allclose([res.probabilities[c] for c in PK.SUPER], ref, atol=1e-4, rtol=0)
