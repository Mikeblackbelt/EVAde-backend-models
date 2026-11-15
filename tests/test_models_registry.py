"""Model artifact contract: manifest validation, sha256 checks, draft gating, legacy folders, training export, R2 sync."""
import json
import os
from argparse import Namespace

import numpy as np
import pytest

pytest.importorskip("sklearn")
joblib = pytest.importorskip("joblib")
from sklearn.linear_model import LogisticRegression  # noqa: E402

from evade.models import manifest as MF  # noqa: E402
from evade.models import registry as R  # noqa: E402
from evade.models.export import restamp, write_manifest  # noqa: E402
from evade.models.fetch import pull, push  # noqa: E402
from evade.models.registry import ModelRegistry  # noqa: E402
from evade.storage.cloudflare import R2Blobs  # noqa: E402
from tests.fakes import FakeS3  # noqa: E402

FEATS = ["kp_index", "proton_flux_10mev", "solar_wind_speed"]


def make_folder(root, task="spaceweather", model_id="EVA-99", status="draft", **kw):
    d = root / task
    d.mkdir(parents=True)
    rng = np.random.default_rng(0)
    X = rng.standard_normal((60, len(FEATS)))
    joblib.dump(LogisticRegression().fit(X, (X[:, 0] > 0).astype(int)), d / "model.joblib")
    (d / "feature_names.json").write_text(json.dumps(FEATS))
    (d / "metrics.json").write_text(json.dumps({"auroc": 0.9}))
    write_manifest(d, task=task, model_id=model_id, status=status,
                   input_spec={"kind": "features", "modality": "space_weather", "features": "feature_names.json"},
                   output_spec={"type": "binary", "classes": ["elevated_radiation"], "thresholds": {"review": 0.5}},
                   data={"datasets": ["synthetic test data"]}, headline={"auroc": 0.9}, commit="abc123", **kw)
    return d


def test_manifest_folder_loads_generically(tmp_path):
    make_folder(tmp_path)
    reg = ModelRegistry(tmp_path, allow_draft=True)
    e = reg.scan()["spaceweather"]
    assert e.available and e.sha256_verified and not e.legacy, e.reason
    served = reg.served()["spaceweather"]
    assert served["model_id"] == "EVA-99" and served["features"] == FEATS and served["metrics"] == {"auroc": 0.9}
    assert served["model"].predict_proba(np.zeros((1, 3))).shape == (1, 2)
    m = served["manifest"]
    assert set(m["files"]) == {"feature_names.json", "metrics.json"} and m["code_commit"] == "abc123"


def test_draft_needs_flag_and_validated_does_not(tmp_path):
    d = make_folder(tmp_path)
    e = ModelRegistry(tmp_path, allow_draft=False).scan()["spaceweather"]
    assert not e.available and "EVADE_ALLOW_DRAFT" in e.reason and e.model is None
    restamp(d, status="validated")
    assert ModelRegistry(tmp_path, allow_draft=False).scan()["spaceweather"].available


def test_sha_mismatch_refuses_to_load(tmp_path):
    d = make_folder(tmp_path)
    with open(d / "model.joblib", "ab") as f:
        f.write(b"tampered")
    e = ModelRegistry(tmp_path, allow_draft=True).scan()["spaceweather"]
    assert not e.available and "sha256 mismatch" in e.reason and e.model is None


def test_sidecar_file_tamper_detected(tmp_path):
    d = make_folder(tmp_path)
    (d / "feature_names.json").write_text(json.dumps(FEATS[::-1]))
    e = ModelRegistry(tmp_path, allow_draft=True).scan()["spaceweather"]
    assert not e.available and "feature_names.json" in e.reason


def test_unsafe_weights_path_and_broken_manifest_do_not_crash(tmp_path):
    d = make_folder(tmp_path)
    m = json.loads((d / "manifest.json").read_text())
    m["weights"] = "../model.joblib"
    (d / "manifest.json").write_text(json.dumps(m))
    (tmp_path / "broken").mkdir()
    (tmp_path / "broken" / "manifest.json").write_text("{not json")
    reg = ModelRegistry(tmp_path, allow_draft=True)
    ents = reg.scan()
    assert not ents["spaceweather"].available and "bare file name" in ents["spaceweather"].reason
    assert not ents["broken"].available and reg.served() == {}
    assert {x["task"] for x in reg.listing()} == {"spaceweather", "broken"}


def test_unassigned_id_listed_not_served_until_restamped(tmp_path):
    d = make_folder(tmp_path, model_id=None)
    assert json.loads((d / "manifest.json").read_text())["model_id"] == MF.UNASSIGNED
    e = ModelRegistry(tmp_path, allow_draft=True).scan()["spaceweather"]
    assert not e.available and "not assigned" in e.reason
    restamp(d, model_id="EVA-98")
    assert ModelRegistry(tmp_path, allow_draft=True).scan()["spaceweather"].available


def test_missing_runtime_is_unavailable_not_a_crash(tmp_path, monkeypatch):
    d = tmp_path / "eegnet"
    d.mkdir()
    (d / "model.onnx").write_bytes(b"\x08\x01not-a-real-onnx")
    write_manifest(d, task="eegnet", model_id="EVA-97", fmt="onnx", weights="model.onnx",
                   input_spec={"kind": "tensor", "modality": "eeg", "fs_hz": 250.0, "window_s": 4.0,
                               "tensor_shape": [1, 18, 1000]},
                   output_spec={"type": "binary", "classes": ["seizure_suspect"]}, data="test")
    monkeypatch.setattr(R, "runtime_available", lambda fmt: fmt == "joblib")
    e = ModelRegistry(tmp_path, allow_draft=True).scan()["eegnet"]
    assert not e.available and "onnxruntime" in e.reason


def test_legacy_beta_folders_still_load(tmp_path):
    for task in ("seizure", "sleep"):
        d = tmp_path / task
        d.mkdir()
        joblib.dump(LogisticRegression().fit(np.eye(4), [0, 1, 0, 1]), d / "model.joblib")
        (d / "feature_names.json").write_text(json.dumps(["a", "b", "c", "d"]))
        (d / "metrics.json").write_text(json.dumps({"leave_one_subject_out_chbmit": {"hgb": {"threshold_at_1pct_fpr": 0.9}}}))
    reg = ModelRegistry(tmp_path, allow_draft=True)
    ents = reg.scan()
    assert "sleep" not in ents                       # no manifest and not a legacy served task: ignored as before
    e = ents["seizure"]
    assert e.available and e.legacy and not e.sha256_verified
    assert reg.served()["seizure"]["model_id"] == "EVA-2" and reg.served()["seizure"]["status"] == "draft"
    assert ModelRegistry(tmp_path, allow_draft=False).scan()["seizure"].available is False


def test_training_writes_a_servable_folder(tmp_path):
    pytest.importorskip("pandas")
    from evade.training.train import _write
    rng = np.random.default_rng(1)
    X = rng.standard_normal((40, 4))
    model = {k: LogisticRegression().fit(X, (X[:, i] > 0).astype(int)) for i, k in enumerate(["NORM", "MI"])}
    metrics = {"hgb": {"macro_auroc": np.float64(0.8), "macro_auroc_ci95": [0.7, 0.9]}, "split": "test", "n_train": 40}
    _write(tmp_path, "ecg", metrics, model, ["f1", "f2", "f3", "f4"], Namespace(model_id="EVA-96", smoke=True))
    m = json.loads((tmp_path / "ecg" / "manifest.json").read_text())
    assert m["model_id"] == "EVA-96" and m["input"]["fs_hz"] == 100.0 and m["output"]["type"] == "multilabel"
    assert m["headline"]["macro_auroc_test_fold"] == 0.8 and m["data"]["smoke"] is True
    served = ModelRegistry(tmp_path, allow_draft=True).scan()["ecg"]
    assert served.available and served.sha256_verified and set(served.model) == {"NORM", "MI"}


def test_push_then_pull_verifies_hashes(tmp_path):
    src = make_folder(tmp_path / "src")
    s3 = FakeS3()
    blobs = R2Blobs("bkt", s3)
    assert push(blobs, src) == "models/spaceweather/"
    assert s3.keys("models/spaceweather/")[-1] == "models/spaceweather/model.joblib"
    dest = tmp_path / "dest"
    assert pull(blobs, dest) == {"spaceweather": "fetched"}
    assert ModelRegistry(dest, allow_draft=True).scan()["spaceweather"].available
    assert pull(blobs, dest) == {"spaceweather": "up_to_date"}
    # a corrupted object in R2: the pull fails and the served folder is left as it was
    s3.objects[("bkt", "models/spaceweather/model.joblib")] += b"x"
    restamp(src, status="validated")                 # remote manifest changes, so a re-download is attempted
    blobs.put("models/spaceweather/manifest.json", (src / "manifest.json").read_bytes(), "application/json")
    res = pull(blobs, dest)
    assert res["spaceweather"].startswith("failed") and "sha256 mismatch" in res["spaceweather"]
    assert ModelRegistry(dest, allow_draft=True).scan()["spaceweather"].available
    assert not [p for p in dest.iterdir() if p.name.startswith(".")]


def test_api_lists_unavailable_models(tmp_path, monkeypatch):
    pytest.importorskip("fastapi")
    os.environ.setdefault("EVADE_API_TOKEN", "test-token")
    from fastapi.testclient import TestClient
    from evade.api import app as A
    make_folder(tmp_path)
    reg = ModelRegistry(tmp_path, allow_draft=False)
    reg.scan()
    monkeypatch.setattr(A.STATE, "registry", reg)
    r = TestClient(A.app).get("/v1/models", headers={"Authorization": f"Bearer {os.environ['EVADE_API_TOKEN']}"})
    d = r.json()["data"]
    assert r.status_code == 200 and d["unavailable"][0]["task"] == "spaceweather"
    assert "EVADE_ALLOW_DRAFT" in d["unavailable"][0]["reason"]
    reg2 = ModelRegistry(tmp_path, allow_draft=True)
    reg2.scan()
    monkeypatch.setattr(A.STATE, "registry", reg2)
    monkeypatch.setitem(A.STATE.models, "spaceweather", reg2.served()["spaceweather"])
    d = TestClient(A.app).get("/v1/models", headers={"Authorization": f"Bearer {os.environ['EVADE_API_TOKEN']}"}).json()["data"]
    row = [m for m in d["models"] if m["task"] == "spaceweather"][0]
    assert row["model_id"] == "EVA-99" and row["headline"] == {"auroc": 0.9} and row["input"]["modality"] == "space_weather"
