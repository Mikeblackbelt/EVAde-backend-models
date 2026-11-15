"""Deep model endpoints: mock mode before the weights exist, switch-over to real TorchScript, state_dict and ONNX
models built inside the tests, sha256 refusal, mock isolation from risk state, and input limits."""
import hashlib
import io
import json

import numpy as np
import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from tests.conftest import KEY_HEADER as H  # noqa: E402  (X-EVAde-Key, checked by the global middleware)
from tests.test_api import C, NOW, new_user  # noqa: E402  (shared client, user helper)

from evade.api import app as A  # noqa: E402
from evade.api import deep_models as D  # noqa: E402
from evade.models import deep as DM  # noqa: E402
from evade.models.deep import DeepModelRegistry  # noqa: E402
from evade.models.deep_tasks import ECG_LEADS, SEIZURE_CHANNELS, SLEEP_CHANNELS  # noqa: E402

PTBXL_LEADS = ["I", "II", "III", "AVR", "AVL", "AVF", "V1", "V2", "V3", "V4", "V5", "V6"]
STAGES = ["W", "N1", "N2", "N3", "REM"]


# ---------------- inputs ----------------
def ecg_body(uid, seed=0, fs=250.0, leads=None):
    rng = np.random.default_rng(seed)
    n = int(10 * fs)
    t = np.arange(n) / fs
    x = 0.05 * rng.standard_normal((12, n)) + 0.3 * np.sin(2 * np.pi * 1.1 * t)[None] * rng.uniform(0.5, 1.5, (12, 1))
    return {"user_id": uid, "fs_hz": fs, "leads": leads or list(ECG_LEADS), "samples": x.round(4).tolist()}


def sleep_body(uid, n_epochs=3, seed=0):
    rng = np.random.default_rng(seed)
    x = 20 * rng.standard_normal((2, n_epochs * 3000))
    return {"user_id": uid, "fs_hz": 100, "channels": list(SLEEP_CHANNELS), "samples": x.round(2).tolist()}


def eeg_body(uid, secs=8.0, fs=250.0, seed=0):
    rng = np.random.default_rng(seed)
    x = 20 * rng.standard_normal((18, int(secs * fs)))
    return {"user_id": uid, "fs_hz": fs, "channels": list(SEIZURE_CHANNELS), "samples": x.round(2).tolist()}


def post(path, body):
    return C.post(path, json=body, headers=H)


def give_eeg_baseline(monkeypatch, uid):
    monkeypatch.setitem(A.STATE.eeg_feats, uid, {"rows": [], "med": np.zeros(3), "mad": np.ones(3)})


@pytest.fixture
def mock_registry(tmp_path):
    prev = D.registry()
    D.set_registry(DeepModelRegistry(tmp_path / "empty", allow_draft=True).scan())
    yield
    D.set_registry(prev)


def use_dir(root, allow_draft=True):
    return D.set_registry(DeepModelRegistry(root, allow_draft=allow_draft).scan())


# ---------------- model folders ----------------
def sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def write_folder(root, task, weights_name, weights: bytes, manifest: dict, side: dict | None = None,
                 files_key="files_sha256"):
    d = root / task
    d.mkdir(parents=True, exist_ok=True)
    (d / weights_name).write_bytes(weights)
    files = {weights_name: sha(weights)}
    for name, content in (side or {}).items():
        (d / name).write_bytes(content)
        files[name] = sha(content)
    m = {"model_id": "EVA-901", "task": task, "status": "draft", "weights": weights_name, "sha256": sha(weights),
         files_key: files if files_key == "files_sha256" else {k: v for k, v in files.items() if k != weights_name},
         "code_commit": "test", "trained_at": "2026-11-14T00:00:00+00:00", "data": {"dataset": "synthetic test"},
         **manifest}
    (d / "manifest.json").write_text(json.dumps(m))
    return d


def onnx_ecg_bytes(W, b):
    onnx = pytest.importorskip("onnx")
    from onnx import TensorProto, helper, numpy_helper
    g = helper.make_graph(
        [helper.make_node("ReduceMean", ["x"], ["m"], axes=[2], keepdims=0),
         helper.make_node("MatMul", ["m", "W"], ["mw"]), helper.make_node("Add", ["mw", "b"], ["logits"])],
        "tiny_ecg", [helper.make_tensor_value_info("x", TensorProto.FLOAT, ["batch", 12, 250])],
        [helper.make_tensor_value_info("logits", TensorProto.FLOAT, ["batch", 5])],
        [numpy_helper.from_array(W.astype(np.float32), "W"), numpy_helper.from_array(b.astype(np.float32), "b")])
    model = helper.make_model(g, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 7
    onnx.checker.check_model(model)
    return model.SerializeToString()


def ecg_manifest(fmt="onnx", **kw):
    return {"format": fmt, "input": {"kind": "tensor", "modality": "ecg", "fs_hz": 100, "leads": PTBXL_LEADS,
                                     "tensor_shape": ["batch", 12, 250], "record_samples": 1000,
                                     "window_stride_samples": 125, "unit": "mV",
                                     "normalization": {"type": "per_lead_zscore", "mean": [0.01] * 12,
                                                       "std": [0.2] * 12}},
            "output": {"type": "multilabel", "classes": ["NORM", "MI", "STTC", "CD", "HYP"], "activation": "sigmoid"},
            **kw}


def torch_bytes(module, script=True):
    torch = pytest.importorskip("torch")
    buf = io.BytesIO()
    if script:
        torch.jit.save(torch.jit.script(module), buf)
    else:
        torch.save(module.state_dict(), buf)
    return buf.getvalue()


def tiny_modules():
    torch = pytest.importorskip("torch")
    from torch import nn

    class SleepEpoch(nn.Module):                 # (B, 2, 3000) -> (B, 5)
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(2, 5)

        def forward(self, x):
            return self.fc(x.std(dim=-1))

    class SleepSeq(nn.Module):                   # (B, L, 2, 3000) -> (B, L, 5)
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(2, 5)

        def forward(self, x):
            return self.fc(x.std(dim=-1))

    class Seizure(nn.Module):                    # (B, 18, 1000) -> (B, 1)
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(18, 1)

        def forward(self, x):
            return self.fc(x.std(dim=-1))

    class EcgNet(nn.Module):                     # (B, 12, 250) -> (B, 5)
        def __init__(self, c_in=12, n_out=5):
            super().__init__()
            self.fc = nn.Linear(c_in, n_out)

        def forward(self, x):
            return self.fc(x.mean(dim=-1))

    torch.manual_seed(0)
    return SleepEpoch(), SleepSeq(), Seizure(), EcgNet


def sleep_manifest(seq=None, fmt="torch"):
    shape = ["batch", 2, 3000] if seq is None else ["batch", seq, 2, 3000]
    return {"format": fmt, "input": {"kind": "tensor", "modality": "eeg", "fs_hz": 100, "channels": ["FPZ-CZ", "PZ-OZ"],
                                     "tensor_shape": shape, "normalization": "per_night_zscore"},
            "output": {"type": "multiclass", "classes": STAGES, "activation": "softmax"}}


def seizure_manifest(ops=True, norm="per_window_zscore", fmt="torchscript"):
    out = {"type": "binary", "classes": ["seizure_suspect"], "activation": "sigmoid"}
    if ops:
        out["operating_points"] = [{"name": "review", "target_fpr": 0.05, "threshold": 0.30},
                                   {"name": "suspect_seizure", "target_fpr": 0.01, "threshold": 0.60}]
    return {"format": fmt, "input": {"kind": "tensor", "modality": "eeg", "fs_hz": 250, "window_s": 4.0, "step_s": 2.0,
                                     "channels": SEIZURE_CHANNELS, "tensor_shape": ["batch", 18, 1000],
                                     "normalization": norm},
            "output": out}


def keyset(d, depth=2):
    """Keys of a response's data, two levels down (lists: the first element), for mock/model schema parity."""
    out = set()
    for k, v in d.items():
        out.add(k)
        if depth > 1 and isinstance(v, dict) and k not in ("probabilities", "thresholds", "minutes_per_stage"):
            out |= {f"{k}.{s}" for s in keyset(v, depth - 1)}
        if depth > 1 and isinstance(v, list) and v and isinstance(v[0], dict):
            out |= {f"{k}[].{s}" for s in keyset(v[0], depth - 1)}
    return out


# ---------------- mock mode ----------------
def test_mock_ecg_schema_and_determinism(mock_registry):
    uid = new_user()
    r1, r2 = post("/v1/predict/ecg-deep", ecg_body(uid)), post("/v1/predict/ecg-deep", ecg_body(uid))
    assert r1.status_code == 200, r1.text
    j = r1.json()
    d = j["data"]
    assert d["source"] == "mock" and d["model_id"] is None and d["model_state"] == "mock"
    assert D.MOCK_WARNING in j["warnings"] and j["model_ids"] == []
    assert set(d["probabilities"]) == {"NORM", "MI", "STTC", "CD", "HYP"}
    assert all(0.0 <= p <= 1.0 for p in d["probabilities"].values())
    assert all(t["probability"] >= d["thresholds"][t["class"]] for t in d["top_classes"])
    assert d == r2.json()["data"]                                     # same input, same output
    other = post("/v1/predict/ecg-deep", ecg_body(uid, seed=5)).json()["data"]
    assert other["probabilities"] != d["probabilities"]


def test_mock_sleep_hypnogram_is_plausible(mock_registry):
    uid = new_user()
    r = post("/v1/predict/sleep-deep", sleep_body(uid, n_epochs=12))
    assert r.status_code == 200, r.text
    d = r.json()["data"]
    assert d["source"] == "mock" and d["model_id"] is None and len(d["epochs"]) == 12
    for e in d["epochs"]:
        assert set(e["probabilities"]) == set(STAGES) and abs(sum(e["probabilities"].values()) - 1) < 1e-6
        assert e["stage"] == max(e["probabilities"], key=e["probabilities"].get)
    stages = [e["stage"] for e in d["epochs"]]
    assert not any(a == "W" and b == "N3" for a, b in zip(stages, stages[1:]))   # no wake-to-N3 jumps
    s = d["summary"]
    assert s["n_epochs"] == 12 and s["recording_min"] == 6.0
    assert abs(sum(s["minutes_per_stage"].values()) - 6.0) < 1e-9
    assert s["total_sleep_time_min"] == 6.0 - s["minutes_per_stage"]["W"]
    assert d == post("/v1/predict/sleep-deep", sleep_body(uid, n_epochs=12)).json()["data"]


def test_mock_seizure_needs_baseline_then_windows(mock_registry, monkeypatch):
    uid = new_user()
    d = post("/v1/predict/seizure-deep", eeg_body(uid)).json()["data"]
    assert d["status"] == "needs_baseline" and d["needs_baseline"] and d["flag"] is None and d["windows"] == []
    assert d["source"] == "mock"
    give_eeg_baseline(monkeypatch, uid)
    r = post("/v1/predict/seizure-deep", eeg_body(uid))
    d = r.json()["data"]
    assert d["status"] == "ok" and not d["needs_baseline"] and len(d["windows"]) == 3      # 8 s: 4 s windows, 2 s step
    assert d["flag"] in ("none", "review", "suspect_seizure") and d["model_id"] is None
    assert all(0 <= w["probability"] <= 1 for w in d["windows"])
    assert D.MOCK_WARNING in r.json()["warnings"]
    assert d == post("/v1/predict/seizure-deep", eeg_body(uid)).json()["data"]


def test_mock_never_touches_risk_flags_overrides_or_baselines(mock_registry, monkeypatch):
    uid = new_user()
    give_eeg_baseline(monkeypatch, uid)
    # make the mock seizure score cross every threshold: it must still not reach any state
    monkeypatch.setattr(D.ADAPTERS["seizure_deep"], "mock", lambda x, fs, n: np.full(n, 0.99))
    risk_before = post("/v1/predict/risk", {"user_id": uid}).json()["data"]["risk"]
    snap = lambda: (json.dumps(A.STATE.latest.get(uid), default=str, sort_keys=True), sorted(A.STATE.overrides.get(uid, ())),
                    json.dumps(A.STATE.baselines.for_user(uid), default=str, sort_keys=True),
                    A.STATE.eeg_feats[uid]["med"].tolist(), len(A.STATE.risk_history.get(uid, [])))
    before = snap()
    calls = []
    monkeypatch.setattr(A, "remember", lambda *a, **k: calls.append("remember") or [])
    monkeypatch.setattr(A, "persist_soft", lambda *a, **k: calls.append("persist_soft"))
    d = post("/v1/predict/seizure-deep", eeg_body(uid)).json()["data"]
    assert d["flag"] == "suspect_seizure" and d["source"] == "mock"
    post("/v1/predict/ecg-deep", ecg_body(uid))
    post("/v1/predict/sleep-deep", sleep_body(uid))
    assert calls == [] and snap() == before
    monkeypatch.undo()
    risk_after = post("/v1/predict/risk", {"user_id": uid}).json()["data"]["risk"]
    assert risk_after["r_agg"] == risk_before["r_agg"] and risk_after["flag_domains"] == risk_before["flag_domains"]
    # the guard itself: a mock output is refused before anything is written
    with pytest.raises(D.MockOutputError):
        D.feed_shadow_risk(A, uid, {**d, "source": "mock"}, [])
    with pytest.raises(D.MockOutputError):
        D.feed_shadow_risk(A, uid, {**d, "source": "model", "model_id": None}, [])


def test_models_listing_shows_mock_state(mock_registry):
    d = C.get("/v1/models", headers=H).json()["data"]
    states = {m["task"]: m for m in d["deep_models"]}
    assert set(states) == {"ecg_deep", "sleep_deep", "seizure_deep"}
    assert all(m["state"] == "mock" and m["source"] == "mock" and m["model_id"] is None for m in states.values())


def test_auth_and_unknown_user(mock_registry):
    assert C.post("/v1/predict/ecg-deep", json=ecg_body("x")).status_code == 401
    assert C.post("/v1/models/reload").status_code == 401
    bad = {"X-EVAde-Key": "wrong-key-0123456789abcdef-0123456789"}
    assert C.post("/v1/predict/sleep-deep", json={}, headers=bad).status_code == 401
    assert C.post("/v1/models/reload", headers=bad).status_code == 401
    r = post("/v1/predict/ecg-deep", ecg_body("00000000-0000-4000-8000-000000000000"))
    assert r.status_code == 404 and r.json()["error"]["code"] == "user_not_found"
    assert _err(post("/v1/predict/ecg-deep", ecg_body("no-such-user"))) == (422, "validation_error")   # UUIDs only


# ---------------- real models ----------------
def test_onnx_ecg_switches_to_model_mode(tmp_path, mock_registry):
    pytest.importorskip("onnxruntime")
    uid = new_user()
    mock = post("/v1/predict/ecg-deep", ecg_body(uid)).json()["data"]
    rng = np.random.default_rng(1)
    W, b = rng.standard_normal((12, 5)), rng.standard_normal(5)
    write_folder(tmp_path, "ecg_deep", "model.onnx", onnx_ecg_bytes(W, b), ecg_manifest(),
                 side={"feature_names.json": json.dumps(PTBXL_LEADS).encode(), "metrics.json": b"{}"})
    reg = use_dir(tmp_path)
    assert reg.get("ecg_deep").available, reg.get("ecg_deep").reason
    body = ecg_body(uid)
    r = post("/v1/predict/ecg-deep", body)
    j = r.json()
    d = j["data"]
    assert d["source"] == "model" and d["model_id"] == "EVA-901" and d["model_state"] == "available"
    assert j["model_ids"] == ["EVA-901"] and "draft model output" in j["warnings"] and D.MOCK_WARNING not in j["warnings"]
    assert keyset(d) == keyset(mock)
    # same numbers as the export protocol computed by hand: resample, z-score, 250-sample crops at stride 125
    from evade.preprocessing.signal import resample
    x = resample(np.asarray(body["samples"]), 250.0, 100.0)[:, :1000]
    x = (x - 0.01) / 0.2
    crops = np.stack([x[:, s:s + 250] for s in range(0, 751, 125)])
    p = (1 / (1 + np.exp(-(crops.mean(-1) @ W + b)))).mean(0)
    assert np.allclose([d["probabilities"][c] for c in ["NORM", "MI", "STTC", "CD", "HYP"]], p, atol=1e-3)
    listing = {m["task"]: m for m in C.get("/v1/models", headers=H).json()["data"]["deep_models"]}
    assert listing["ecg_deep"]["state"] == "available" and listing["ecg_deep"]["sha256_verified"]


def test_ecg_lead_order_and_per_lead_stats_follow_the_manifest(tmp_path, mock_registry):
    pytest.importorskip("onnxruntime")
    rng = np.random.default_rng(3)
    W, b = rng.standard_normal((12, 5)), rng.standard_normal(5)
    mu, sd = rng.normal(0, 0.05, 12), rng.uniform(0.1, 0.4, 12)
    perm = rng.permutation(12)                    # model input row i is canonical lead perm[i]

    def folder(root, leads, mean, std, w):
        m = ecg_manifest()
        m["input"] = {**m["input"], "leads": leads,
                      "normalization": {"type": "per_lead_zscore", "mean": list(mean), "std": list(std)}}
        write_folder(root, "ecg_deep", "model.onnx", onnx_ecg_bytes(w, b), m)

    folder(tmp_path / "a", PTBXL_LEADS, mu, sd, W)
    folder(tmp_path / "b", [PTBXL_LEADS[p] for p in perm], mu[perm], sd[perm], W[perm])
    uid = new_user()
    body = ecg_body(uid, seed=7)
    use_dir(tmp_path / "a")
    pa = post("/v1/predict/ecg-deep", body).json()["data"]["probabilities"]
    use_dir(tmp_path / "b")
    shuffled = dict(body, leads=body["leads"][::-1], samples=body["samples"][::-1])   # request order differs too
    pb = post("/v1/predict/ecg-deep", shuffled).json()["data"]
    assert pb["source"] == "model" and pb["probabilities"] == pa
    assert len(set(pa.values())) > 1


def test_garbage_manifest_is_listed_not_a_500(tmp_path, mock_registry):
    d = tmp_path / "sleep_deep"
    d.mkdir()
    (d / "model.pt").write_bytes(b"x")
    for bad in ('{"output": "x", "format": "torch"}', '[1, 2]', '{not json', '{"model_id": "EVA-1"}'):
        (d / "manifest.json").write_text(bad)
        use_dir(tmp_path)
        r = C.get("/v1/models", headers=H)
        assert r.status_code == 200
        row = {m["task"]: m for m in r.json()["data"]["deep_models"]}["sleep_deep"]
        assert row["state"] == "unavailable" and row["reason_code"] == "manifest_invalid" and row["output"] is None
        assert post("/v1/predict/sleep-deep", sleep_body(new_user())).json()["data"]["source"] == "mock"


def test_ecg_thresholds_from_manifest_never_invented():
    from evade.models.deep_tasks import EcgAdapter
    cls = ["NORM", "MI", "STTC", "CD", "HYP"]
    assert EcgAdapter.manifest_thresholds({"operating_points": {c: {"threshold": 0.3} for c in cls}}) == dict.fromkeys(cls, 0.3)
    assert EcgAdapter.manifest_thresholds({"thresholds": {c.lower(): 0.4 for c in cls}}) == dict.fromkeys(cls, 0.4)
    assert EcgAdapter.manifest_thresholds({"thresholds": {"review": 0.2}}) == dict.fromkeys(cls, 0.2)
    assert EcgAdapter.manifest_thresholds({"operating_points": {"NORM": 0.3}}) is None      # partial: not used


def test_export_manifest_without_model_id_is_unavailable(tmp_path, mock_registry):
    pytest.importorskip("onnxruntime")
    write_folder(tmp_path, "ecg_deep", "model.onnx", onnx_ecg_bytes(np.zeros((12, 5)), np.zeros(5)),
                 ecg_manifest(model_id="UNASSIGNED"))
    e = use_dir(tmp_path).get("ecg_deep")
    assert e.state == "unavailable" and e.reason_code == "model_id_unassigned"
    d = post("/v1/predict/ecg-deep", ecg_body(new_user())).json()
    assert d["data"]["source"] == "mock" and any("unavailable" in w for w in d["warnings"])


def test_torchscript_sleep_and_seizure_switch_to_model_mode(tmp_path, mock_registry, monkeypatch):
    sleep_m, _, seizure_m, _ = tiny_modules()
    uid = new_user()
    give_eeg_baseline(monkeypatch, uid)
    mock_sleep = post("/v1/predict/sleep-deep", sleep_body(uid)).json()["data"]
    mock_sz = post("/v1/predict/seizure-deep", eeg_body(uid)).json()["data"]
    write_folder(tmp_path, "sleep_deep", "model.pt", torch_bytes(sleep_m), sleep_manifest(fmt="torch"))
    write_folder(tmp_path, "seizure_deep", "model.ts", torch_bytes(seizure_m), seizure_manifest(), files_key="files")
    reg = use_dir(tmp_path)
    assert reg.get("sleep_deep").available and reg.get("sleep_deep").runner.kind == "torchscript", reg.get("sleep_deep").reason
    assert reg.get("seizure_deep").available, reg.get("seizure_deep").reason

    d = post("/v1/predict/sleep-deep", sleep_body(uid)).json()["data"]
    assert d["source"] == "model" and d["model_id"] == "EVA-901" and len(d["epochs"]) == 3
    assert keyset(d) == keyset(mock_sleep)
    assert all(abs(sum(e["probabilities"].values()) - 1) < 1e-6 for e in d["epochs"])

    d = post("/v1/predict/seizure-deep", eeg_body(uid)).json()["data"]
    assert d["source"] == "model" and d["status"] == "ok" and len(d["windows"]) == 3
    assert d["thresholds"] == {"review_5pct_fpr": 0.3, "suspect_1pct_fpr": 0.6}
    assert keyset(d) == keyset(mock_sz)
    shadow = A.STATE.latest[uid][("eeg", "seizure_deep_probability")]      # real output: shadow input only
    assert shadow["shadow"] is True and shadow["domain"] == "neuro" and shadow["value"] == d["max_probability"]


def test_sleep_sequence_model_chunks_and_pads(tmp_path, mock_registry):
    _, seq_m, _, _ = tiny_modules()
    write_folder(tmp_path, "sleep_deep", "model.pt", torch_bytes(seq_m), sleep_manifest(seq=3))
    assert use_dir(tmp_path).get("sleep_deep").available
    d = post("/v1/predict/sleep-deep", sleep_body(new_user(), n_epochs=4)).json()["data"]
    assert d["source"] == "model" and [e["index"] for e in d["epochs"]] == [0, 1, 2, 3]


def test_state_dict_with_registered_arch(tmp_path, mock_registry, monkeypatch):
    pytest.importorskip("torch")
    _, _, _, EcgNet = tiny_modules()
    monkeypatch.setitem(DM.ARCHS, "tiny_ecg", lambda c_in, n_out: EcgNet(c_in, n_out))
    write_folder(tmp_path, "ecg_deep", "model.pt", torch_bytes(EcgNet(), script=False),
                 ecg_manifest(fmt="torch", arch="tiny_ecg"))
    e = use_dir(tmp_path).get("ecg_deep")
    assert e.available and e.runner.kind == "torch_state_dict", e.reason
    assert post("/v1/predict/ecg-deep", ecg_body(new_user())).json()["data"]["source"] == "model"
    monkeypatch.delitem(DM.ARCHS, "tiny_ecg")
    e = use_dir(tmp_path).get("ecg_deep")
    assert e.state == "unavailable" and e.reason_code == "arch_unknown"


def test_sha_mismatch_is_unavailable_and_never_loaded(tmp_path, mock_registry, monkeypatch):
    sleep_m, _, _, _ = tiny_modules()
    d = write_folder(tmp_path, "sleep_deep", "model.pt", torch_bytes(sleep_m), sleep_manifest())
    (d / "model.pt").write_bytes((d / "model.pt").read_bytes() + b"tampered")
    loaded = []
    monkeypatch.setattr(DM, "_load_torchscript", lambda data: loaded.append(1))
    monkeypatch.setattr(DM, "_load_state_dict", lambda *a: loaded.append(1))
    e = use_dir(tmp_path).get("sleep_deep")
    assert e.state == "unavailable" and e.reason_code == "sha256_mismatch" and loaded == [] and e.runner is None
    r = post("/v1/predict/sleep-deep", sleep_body(new_user())).json()
    assert r["data"]["source"] == "mock" and r["data"]["model_state"] == "unavailable"
    listing = {m["task"]: m for m in C.get("/v1/models", headers=H).json()["data"]["deep_models"]}
    assert listing["sleep_deep"]["reason_code"] == "sha256_mismatch" and "refusing" in listing["sleep_deep"]["reason"]
    assert str(tmp_path) not in json.dumps(listing)                     # no paths in the API


def test_side_file_sha_mismatch(tmp_path, mock_registry):
    pytest.importorskip("onnxruntime")
    d = write_folder(tmp_path, "ecg_deep", "model.onnx", onnx_ecg_bytes(np.zeros((12, 5)), np.zeros(5)), ecg_manifest(),
                     side={"metrics.json": b"{}"})
    (d / "metrics.json").write_text('{"changed": 1}')
    assert use_dir(tmp_path).get("ecg_deep").reason_code == "sha256_mismatch"


def test_draft_gate_runtime_missing_and_unsupported_specs(tmp_path, mock_registry, monkeypatch):
    sleep_m, _, seizure_m, _ = tiny_modules()
    write_folder(tmp_path, "sleep_deep", "model.pt", torch_bytes(sleep_m), sleep_manifest())
    assert use_dir(tmp_path, allow_draft=False).get("sleep_deep").reason_code == "draft_not_allowed"
    monkeypatch.setattr(DM, "runtime_available", lambda fmt: False)
    assert use_dir(tmp_path).get("sleep_deep").reason_code == "runtime_missing"
    monkeypatch.undo()
    write_folder(tmp_path, "seizure_deep", "model.ts", torch_bytes(seizure_m), seizure_manifest(ops=False))
    assert use_dir(tmp_path).get("seizure_deep").reason_code == "spec_unsupported"      # no default thresholds
    write_folder(tmp_path, "seizure_deep", "model.ts", torch_bytes(seizure_m),
                 seizure_manifest(norm="per-person per-channel median and 1.4826*MAD from baseline windows"))
    assert use_dir(tmp_path).get("seizure_deep").reason_code == "spec_unsupported"


def test_reload_endpoint_picks_up_new_weights(tmp_path, mock_registry, monkeypatch):
    sleep_m, _, _, _ = tiny_modules()
    monkeypatch.setenv("EVADE_MODEL_DIR", str(tmp_path))
    monkeypatch.setenv("EVADE_ALLOW_DRAFT", "1")
    r = C.post("/v1/models/reload", headers=H)
    assert r.status_code == 200, r.text
    assert {m["task"]: m["state"] for m in r.json()["data"]["deep_models"]}["sleep_deep"] == "mock"
    write_folder(tmp_path, "sleep_deep", "model.pt", torch_bytes(sleep_m), sleep_manifest())
    r = C.post("/v1/models/reload", headers=H)
    assert {m["task"]: m["state"] for m in r.json()["data"]["deep_models"]}["sleep_deep"] == "available"
    assert post("/v1/predict/sleep-deep", sleep_body(new_user())).json()["data"]["source"] == "model"


# ---------------- validation ----------------
def _err(r):
    return r.status_code, r.json()["error"]["code"]


def test_oversized_inputs_get_422_without_echoing_values(mock_registry):
    uid = new_user()
    b = ecg_body(uid, fs=100.0)
    b["leads"] = b["leads"] + ["V7"]
    b["samples"] = b["samples"] + [[123.456] * 1000]
    r = post("/v1/predict/ecg-deep", b)
    assert r.status_code == 422 and "123.456" not in r.text
    b = ecg_body(uid, fs=100.0)
    b["samples"] = [[0] * 10_501 for _ in range(12)]          # too many samples per lead, body under 1 MB
    assert _err(post("/v1/predict/ecg-deep", b)) == (422, "validation_error")
    b = eeg_body(uid, secs=4.0)
    b["channels"] = b["channels"] + ["FZ-PZ"]
    b["samples"] = b["samples"] + [[123.456] * 1000]
    r = post("/v1/predict/seizure-deep", b)
    assert r.status_code == 422 and "123.456" not in r.text
    big = {"user_id": uid, "fs_hz": 100, "channels": list(SLEEP_CHANNELS), "samples": [[0] * (21 * 3000)] * 2}
    r = post("/v1/predict/sleep-deep", big)                  # 21 epochs, still under the 1 MB body cap
    assert r.status_code == 422, r.status_code


def test_shape_and_value_errors(mock_registry):
    uid = new_user()
    b = ecg_body(uid, fs=100.0)
    b["samples"] = [row[:500] for row in b["samples"]]
    assert _err(post("/v1/predict/ecg-deep", b)) == (422, "bad_shape")
    b = ecg_body(uid, fs=100.0, leads=["I"] * 12)
    assert _err(post("/v1/predict/ecg-deep", b)) == (422, "bad_shape")
    b = sleep_body(uid)
    b["samples"] = [row[:4000] for row in b["samples"]]
    assert _err(post("/v1/predict/sleep-deep", b)) == (422, "bad_shape")
    b = eeg_body(uid, secs=3.0)
    assert _err(post("/v1/predict/seizure-deep", b)) == (422, "bad_shape")
    b = eeg_body(uid)
    raw = json.dumps(b).replace(str(b["samples"][0][0]), "NaN", 1)
    r = C.post("/v1/predict/seizure-deep", content=raw, headers={**H, "Content-Type": "application/json"})
    assert _err(r) == (422, "nan_in_signal")
    b = ecg_body(uid)
    b["t0_utc"] = "2999-01-01T00:00:00Z"
    assert _err(post("/v1/predict/ecg-deep", b)) == (422, "future_timestamp")
    b = ecg_body(uid, leads=[c.upper() for c in reversed(ECG_LEADS)])   # any order and case is accepted
    b["t0_utc"] = NOW
    assert post("/v1/predict/ecg-deep", b).status_code == 200


def test_ecg_deep_input_bounds(mock_registry):
    """The deep-serving bounds (50-1000 Hz, polyphase factors <= 100, 60 s, 64 rows) hold on the one served ECG path."""
    from evade.models.deep_tasks import EcgAdapter, ecg_resample_ratio
    assert ecg_resample_ratio(100.0, 100.0) is None
    assert ecg_resample_ratio(250.0, 100.0) == (2, 5)
    assert ecg_resample_ratio(500.0, 100.0) == (1, 5)
    for fs_in, fs_out in ((99.7, 100.0), (49.0, 100.0), (100.0, 1001.0), (997.0, 100.0)):
        with pytest.raises(ValueError):
            ecg_resample_ratio(fs_in, fs_out)
    with pytest.raises(ValueError):                         # 61 s or 65 rows never reach the resampler
        EcgAdapter.to_record(np.zeros((12, 6100)), 100.0, 100.0, 1000, np.zeros(12))
    with pytest.raises(ValueError):
        EcgAdapter.to_record(np.zeros((65, 1000)), 100.0, 100.0, 1000, np.zeros(65))
    uid = new_user()
    r = post("/v1/predict/ecg-deep", ecg_body(uid, fs=99.7))   # 1000/997 needs factors over 100: refused, also in mock
    assert _err(r) == (422, "bad_shape") and r.json()["error"]["field"] == "fs_hz" and "99.7" not in r.text
    assert _err(post("/v1/predict/ecg-deep", ecg_body(uid, fs=40.0))) == (422, "validation_error")
    assert post("/v1/predict/ecg-deep", ecg_body(uid, fs=500.0)).status_code == 200
