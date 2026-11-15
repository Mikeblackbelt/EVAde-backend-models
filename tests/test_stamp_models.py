"""scripts/stamp_models_for_push.py: every pushed folder gets a registry manifest that fetch_models accepts, and the
space-weather folder still loads after conversion."""
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
pytest.importorskip("sklearn")

import stamp_models_for_push as STAMP  # noqa: E402
from evade.models import manifest as MF  # noqa: E402
from evade.models.fetch import _files_of  # noqa: E402
from evade.spaceweather import model as SW  # noqa: E402
from tests.test_spaceweather import write_tiny_model  # noqa: E402


def legacy_folder(root, name):
    from sklearn.linear_model import LogisticRegression
    import joblib
    import numpy as np
    d = root / name
    d.mkdir(parents=True)
    joblib.dump(LogisticRegression().fit(np.random.default_rng(0).normal(size=(40, 3)), [0, 1] * 20), d / "model.joblib")
    (d / "feature_names.json").write_text(json.dumps(["a", "b", "c"]))
    (d / "metrics.json").write_text("{}")
    (d / "features.parquet").write_text("not copied")
    return d


def test_stamp_legacy_and_spaceweather_folders(tmp_path):
    src = tmp_path / "src"
    sw = write_tiny_model(src)                       # original space-weather layout (sha256 per file)
    (sw / "feature_names.json").write_text((sw / "feature_names.json").read_text())
    man = json.loads((sw / "manifest.json").read_text())
    man["sha256"]["feature_names.json"] = SW.sha256(sw / "feature_names.json")
    (sw / "manifest.json").write_text(json.dumps(man))
    ecg = legacy_folder(src, "ecg")
    out = tmp_path / "push"
    assert STAMP.main(["--out", str(out), "--ecg", str(ecg), "--spaceweather", str(sw)]) == 0
    for task, mid in (("ecg", "EVA-6"), ("spaceweather", "EVA-8")):
        m = json.loads((out / task / "manifest.json").read_text())
        MF.validate(m)
        assert m["model_id"] == mid and m["task"] == task
        assert set(_files_of(m)) >= {"model.joblib", "feature_names.json"}
    assert not (out / "ecg" / "features.parquet").exists()
    art = SW.load_forecaster(out)                    # converted layout still loads, hash-checked
    assert art["model_id"] == "EVA-8" and art["sha256_verified"]


def test_unassigned_folder_is_refused(tmp_path):
    d = tmp_path / "src" / "ecg_deep"
    d.mkdir(parents=True)
    (d / "model.onnx").write_bytes(b"onnx")
    (d / "manifest.json").write_text(json.dumps({"model_id": "UNASSIGNED", "task": "ecg_deep", "status": "draft",
                                                 "format": "onnx", "weights": "model.onnx", "sha256": "0" * 64,
                                                 "input": {"kind": "tensor", "modality": "ecg",
                                                           "tensor_shape": ["batch", 12, 250]},
                                                 "output": {"type": "multilabel", "classes": ["NORM"]},
                                                 "code_commit": "x", "trained_at": "x", "data": "x"}))
    with pytest.raises(MF.ManifestError):
        STAMP.main(["--out", str(tmp_path / "push"), "--ecg-deep", str(d)])
    assert STAMP.main(["--out", str(tmp_path / "push"), "--ecg-deep", str(d), "--id", "ecg_deep=EVA-13"]) == 0
