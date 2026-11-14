"""API contract tests with synthetic signals (no datasets, no trained models needed)."""
import os
from datetime import datetime, timezone

import numpy as np
import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")
os.environ["EVADE_API_TOKEN"] = "test-token"
os.environ.pop("EVADE_MODEL_DIR", None)

from fastapi.testclient import TestClient  # noqa: E402

from evade.api.app import app  # noqa: E402

H = {"Authorization": "Bearer test-token"}
C = TestClient(app)
NOW = datetime.now(timezone.utc).isoformat()


def ecg_window(hr=60.0, secs=20.0, fs=250.0, seed=0):
    rng = np.random.default_rng(seed)
    t = np.arange(int(secs * fs)) / fs
    x = 0.02 * rng.standard_normal(t.size)
    for b in np.arange(0.5, secs, 60.0 / hr):
        x += np.exp(-0.5 * ((t - b) / 0.012) ** 2)
    return {"modality": "ecg", "t0_utc": NOW, "fs_hz": fs, "unit": "mv", "channels": ["II"], "samples": [x.round(4).tolist()]}


def new_user():
    r = C.post("/v1/users", json={"consent": True}, headers=H)
    assert r.status_code == 200
    return r.json()["data"]["user_id"]


def test_health_open_and_auth_required():
    assert C.get("/v1/health").status_code == 200
    assert C.post("/v1/users", json={"consent": True}).status_code == 401
    assert C.post("/v1/users", json={"consent": True}, headers={"Authorization": "Bearer wrong"}).status_code == 401


def test_unknown_user_404():
    r = C.post("/v1/predict/ecg", json={"user_id": "nope", "window": ecg_window()}, headers=H)
    assert r.status_code == 404 and r.json()["error"]["code"] == "user_not_found"


def test_validation_errors():
    uid = new_user()
    w = ecg_window(); w["samples"] = [[0.0] * 100]
    r = C.post("/v1/predict/ecg", json={"user_id": uid, "window": w}, headers=H)
    assert r.status_code == 422 and r.json()["error"]["code"] == "bad_shape"
    w = ecg_window(); w["t0_utc"] = "2999-01-01T00:00:00Z"
    assert C.post("/v1/predict/ecg", json={"user_id": uid, "window": w}, headers=H).json()["error"]["code"] == "future_timestamp"


def test_ecg_baseline_then_deviation_flags():
    uid = new_user()
    base = [ecg_window(60, seed=i) for i in range(30)]
    for i in range(0, 30, 10):
        r = C.post("/v1/baseline/ecg", json={"user_id": uid, "windows": base[i:i + 10], "mode": "demo"}, headers=H)
        assert r.status_code == 200
    hr = [b for b in r.json()["data"]["baselines"] if b["metric"] == "hr_bpm"][0]
    assert hr["status"] == "provisional" and abs(hr["center"] - 60) < 2
    r = C.post("/v1/predict/ecg", json={"user_id": uid, "window": ecg_window(110, seed=99)}, headers=H).json()["data"]
    assert 105 < r["features"]["hr_bpm"] < 115
    hrc = [c for c in r["comparison"] if c["metric"] == "hr_bpm"][0]
    assert hrc["level"] == "red" and hrc["confidence"] == "demo" and hrc["component_score"] > 0.4


def test_users_are_isolated():
    a, b = new_user(), new_user()
    C.post("/v1/baseline/ecg", json={"user_id": a, "windows": [ecg_window(60, seed=i) for i in range(30)]}, headers=H)
    rb = C.get(f"/v1/users/{b}/baseline", headers=H).json()["data"]["baselines"]
    assert rb == []


def test_risk_worked_example_b():
    uid = new_user()
    body = {"user_id": uid, "domain_inputs": {"cardiac": {"s": 0.2}, "behavioral": {"s": 0.65}, "neuro": {"s": 0.25},
                                              "radiation": {"s": 0.4}, "bone": {"s": 0.25}},
            "logistics_floor_days": 0.5, "r_7d_ago": 0.539}
    d = C.post("/v1/predict/risk", json=body, headers=H).json()["data"]
    assert d["risk"]["band"] == "high" and abs(d["risk"]["r_agg"] - 0.659) < 0.002
    assert d["mdtr"]["display"] == "16 days" and d["mdtr"]["driver"] == "trend"


def test_risk_withheld_without_logistics():
    uid = new_user()
    d = C.post("/v1/predict/risk", json={"user_id": uid}, headers=H).json()["data"]
    assert d["mdtr"]["status"] == "withheld"


def test_radiation_and_spe_override_feeds_risk():
    uid = new_user()
    d = C.post("/v1/predict/radiation-risk", json={"user_id": uid, "t_utc": NOW, "cumulative_dose_msv": 300,
                                                   "dose_equivalent_rate_msv_per_day": 1.8, "spe_dose_msv": 260}, headers=H).json()["data"]
    assert d["spe_override"] and d["s"] == 1.0 and abs(d["t_rad_days"] - 152.8) < 0.1
    r = C.post("/v1/predict/risk", json={"user_id": uid, "logistics_floor_days": 0.5}, headers=H).json()["data"]
    assert "O5" in r["risk"]["override_ids"] and r["risk"]["r_disp"] >= 0.8 and r["mdtr"]["driver"] == "acute"


def test_phq9_item9_escalates():
    uid = new_user()
    d = C.post("/v1/input/instrument", json={"user_id": uid, "instrument": "phq9", "t_utc": NOW,
                                             "answers": [1, 1, 1, 0, 0, 0, 0, 0, 1]}, headers=H).json()["data"]
    assert d["escalation"]["override"] == "O1" and d["severity_band"] == "minimal"


def test_eeg_endpoint_runs_without_model():
    uid = new_user()
    fs, n = 250.0, 2500
    rng = np.random.default_rng(1)
    names = ["FP1-F7", "F7-T7", "T7-P7", "P7-O1", "FP1-F3", "F3-C3", "C3-P3", "P3-O1", "FP2-F4", "F4-C4", "C4-P4",
             "P4-O2", "FP2-F8", "F8-T8", "T8-P8", "P8-O2", "FZ-CZ", "CZ-PZ"]
    w = {"modality": "eeg", "t0_utc": NOW, "fs_hz": fs, "unit": "uv", "channels": names,
         "samples": (20 * rng.standard_normal((18, n))).round(3).tolist()}
    r = C.post("/v1/predict/eeg", json={"user_id": uid, "window": w}, headers=H)
    assert r.status_code == 200, r.text
    d = r.json()["data"]
    assert d["channels_mapped"] == 18 and "theta_beta" in d["features"] and "seizure" not in d


def test_device_bind_ingest_dedupe_and_isolation():
    a, b = new_user(), new_user()
    assert C.post("/v1/devices/bind", json={"user_id": a, "device_id": "polar-123", "kind": "hr_strap"}, headers=H).status_code == 200
    r = C.post("/v1/devices/bind", json={"user_id": b, "device_id": "polar-123", "kind": "hr_strap"}, headers=H)
    assert r.status_code == 409 and r.json()["error"]["code"] == "device_already_bound"
    rd = {"metric": "hr_bpm", "value": 61.0, "unit": "bpm", "t_utc": NOW, "modality": "ecg"}
    d = C.post("/v1/ingest/batch", json={"user_id": a, "device_id": "polar-123", "seq": 0, "readings": [rd]}, headers=H).json()["data"]
    assert d["accepted"] == 1 and d["next_seq_expected"] == 1
    d = C.post("/v1/ingest/batch", json={"user_id": a, "device_id": "polar-123", "seq": 0, "readings": [rd]}, headers=H).json()["data"]
    assert d["duplicates"] == 1 and d["accepted"] == 0
    d = C.post("/v1/ingest/batch", json={"user_id": a, "device_id": "polar-123", "seq": 3, "readings": [rd]}, headers=H).json()["data"]
    assert d["gap_before"] == 2
    assert C.post("/v1/ingest/batch", json={"user_id": b, "device_id": "polar-123", "seq": 9, "readings": [rd]}, headers=H).status_code == 404
