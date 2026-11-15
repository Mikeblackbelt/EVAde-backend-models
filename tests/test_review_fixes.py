"""Regression tests for the review of the storage and model-registry branch (findings 1-14). No network, no
credentials: D1 is SQLite behind httpx.MockTransport, R2 is an in-memory S3 double."""
import json
import os
import time
import uuid
from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

pytest.importorskip("httpx")
import httpx  # noqa: E402

from evade.storage import MemoryBackend, StorageConfigError, StorageError, from_env  # noqa: E402
from evade.storage.base import storage_deadline, user_hash  # noqa: E402
from evade.storage.cloudflare import CloudflareBackend, D1Client, D1Error, R2Blobs  # noqa: E402
from tests.conftest import KEY_HEADER  # noqa: E402
from tests.fakes import TOKEN, FakeD1, FakeS3  # noqa: E402

USER_TABLES = ("users", "baselines", "baseline_archive", "user_state", "inputs", "flags", "risk_results", "devices")
NOW = datetime.now(timezone.utc).isoformat()


def backend(fake=None, s3=None, **kw):
    fake = fake or FakeD1()
    s3 = s3 if s3 is not None else FakeS3()
    d1 = D1Client("acct", "db-uuid", TOKEN, transport=fake.transport(), sleep=lambda s: None, **kw)
    return CloudflareBackend(d1, R2Blobs("evade-bucket", s3), store_raw_windows=True), fake, s3


def brec(uid, metric="hr_bpm", bid=None, n=3):
    return {"user_id": uid, "modality": "ecg", "metric": metric, "baseline_id": bid or f"b_{metric}", "mode": "demo",
            "phase": "demo", "status": "building", "values": [60.0 + i for i in range(n)], "center": None,
            "spread": None, "created": time.time()}


# ---------------- API fixture: the app wired to a fresh Cloudflare backend ----------------
@pytest.fixture()
def api(monkeypatch):
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient
    from evade.api import app as A
    from evade.baseline.store import BaselineStore
    st, fake, s3 = backend(breaker_threshold=1000)   # scenarios inject failures back to back; the breaker has its own tests
    monkeypatch.setattr(A.STATE, "storage", st)
    monkeypatch.setattr(A.STATE, "baselines", BaselineStore(st))
    monkeypatch.setattr(A.STATE, "pending_deletes", set())
    monkeypatch.setattr(A.STATE, "missing", A.MissCache())
    monkeypatch.setattr(A.STATE, "state_dirty", set())
    return TestClient(A.app), dict(KEY_HEADER), A, st, fake, s3


def restart(A, fake, s3, monkeypatch):
    """A new process: every in-memory structure is empty, storage (D1 + R2) is kept."""
    from evade.baseline.store import BaselineStore
    st2, _, _ = backend(fake, s3, breaker_threshold=1000)
    for name in ("users", "latest", "radiation", "overrides", "risk_history", "inputs", "devices", "eeg_feats"):
        monkeypatch.setattr(A.STATE, name, {})
    monkeypatch.setattr(A.STATE, "storage", st2)
    monkeypatch.setattr(A.STATE, "baselines", BaselineStore(st2))
    monkeypatch.setattr(A.STATE, "pending_deletes", set())
    monkeypatch.setattr(A.STATE, "missing", A.MissCache())
    monkeypatch.setattr(A.STATE, "state_dirty", set())
    return st2


def new_user(C, H):
    r = C.post("/v1/users", json={"consent": True}, headers=H)
    assert r.status_code == 200, r.text
    return r.json()["data"]["user_id"]


def readings(n, value=60.0, modality="vitals"):
    return [{"metric": "hr_bpm", "value": value + i % 3, "unit": "bpm", "t_utc": NOW, "modality": modality}
            for i in range(n)]


# ---------------- 1. R2 failure during DELETE, restart, retried DELETE ----------------
def test_f1_r2_failure_then_restart_then_retried_delete(api, monkeypatch):
    C, H, A, st, fake, s3 = api
    uid = new_user(C, H)
    st.put_blob(f"exports/{uid}/summary.json", b"{}")
    s3.fail_delete = True
    r = C.delete(f"/v1/users/{uid}", headers=H)
    assert r.status_code == 503 and r.json()["error"]["code"] == "deletion_incomplete"
    assert fake.count("users", uid) == 0 and s3.keys(f"exports/{uid}/")       # rows gone, blob still there
    restart(A, fake, s3, monkeypatch)
    s3.fail_delete = False
    r = C.delete(f"/v1/users/{uid}", headers=H)
    assert r.status_code == 200, r.text
    assert s3.keys(f"exports/{uid}/") == []
    assert C.delete(f"/v1/users/{uid}", headers=H).status_code == 404        # a completed deletion is not redone


def test_f1_hydrate_refuses_tombstoned_user(api, monkeypatch):
    C, H, A, st, fake, s3 = api
    uid = new_user(C, H)
    fake.db.execute("INSERT INTO tombstones (user_hash, deleted_at) VALUES (?, ?)", [user_hash(uid), 1.0])
    fake.db.commit()
    restart(A, fake, s3, monkeypatch)
    assert C.get(f"/v1/users/{uid}/baseline", headers=H).status_code == 404


# ---------------- 2. tombstone enforced in SQL on every user-table write ----------------
WRITES = {
    "_put_state": lambda st, u: st.put_state(u, "latest", [{"metric": "hr_bpm"}]),
    "_add_inputs": lambda st, u: st.add_inputs(u, [{"kind": "reading:vitals", "t_utc": NOW, "meta": {}}]),
    "_add_flags": lambda st, u: st.add_flags(u, [{"t": 1.0, "domain": "cardiac", "level": "red"}]),
    "_add_risk": lambda st, u: st.add_risk(u, 1.0, 0.4, "elevated", {}),
    "_put_baselines": lambda st, u: st.put_baselines([brec(u)]),
    "_put_device": lambda st, u: st.put_device("dev-race", {"user_id": u, "kind": "hr_strap", "last_seq": -1}),
    "_archive_baselines": lambda st, u: st.archive_baselines(u, "ecg", [brec(u, bid="b_arch")]),
}


@pytest.mark.parametrize("method", sorted(WRITES))
def test_f2_delete_between_gate_check_and_write(method):
    st, fake, s3 = backend()
    other, _, _ = backend(fake, s3)                 # another process: st has no in-process tombstone
    uid = "user-race"
    st.put_user(uid, {"created_at": 1.0})
    orig = getattr(st, method)

    def racing(*a, **k):                            # the gate in Storage has passed; DELETE lands now
        other.delete_user(uid)
        return orig(*a, **k)

    setattr(st, method, racing)
    WRITES[method](st, uid)
    for t in USER_TABLES:
        assert fake.count(t, uid) == 0, t


def test_f2_write_in_flight_during_api_delete(api):
    C, H, A, st, fake, s3 = api
    uid = new_user(C, H)
    assert C.post("/v1/baseline/vitals", json={"user_id": uid, "readings": readings(12), "mode": "demo"},
                  headers=H).status_code == 200
    other, _, _ = backend(fake, s3)
    fired = []

    def hook(body):                                 # the DELETE commits just before this request's state write
        sqls = " ".join(s["sql"] for s in body.get("batch", [body]))
        if not fired and "INSERT INTO user_state" in sqls:
            fired.append(1)
            other.delete_user(uid)

    fake.on_request = hook
    r = C.post("/v1/compare", json={"user_id": uid, "readings": readings(1, 95.0)}, headers=H)
    assert fired and r.status_code == 200
    for t in USER_TABLES:
        assert fake.count(t, uid) == 0, t


def test_f2_is_deleted_falls_back_to_d1_and_caches():
    st, fake, s3 = backend()
    st.put_user("u1", {"created_at": 1.0})
    st.delete_user("u1")
    st2, _, _ = backend(fake, s3)
    n = fake.calls
    assert st2.is_deleted("u1") and fake.calls == n + 1
    assert st2.is_deleted("u1") and fake.calls == n + 1          # positive result cached
    assert not st2.is_deleted("u2")


def test_f2_get_device_checks_d1_tombstone():
    st, fake, s3 = backend()
    fake.db.execute("INSERT INTO devices (device_id, user_id, rec, updated_at) VALUES ('d1', 'gone', ?, 1)",
                    [json.dumps({"user_id": "gone", "last_seq": 3})])
    fake.db.execute("INSERT INTO tombstones (user_hash, deleted_at) VALUES (?, 1)", [user_hash("gone")])
    fake.db.commit()
    assert st.get_device("d1") is None


# ---------------- 3. no double-apply on retry after a 503 ----------------
def test_f3_audit_write_failure_is_not_a_503_so_no_retry_doubles(api):
    C, H, A, st, fake, s3 = api
    uid = new_user(C, H)
    fake.fail_with = [503] * 3                      # the input-log write fails after the baseline was updated
    r = C.post("/v1/baseline/vitals", json={"user_id": uid, "readings": readings(5), "mode": "demo"}, headers=H)
    assert r.status_code == 200 and r.json()["warnings"]
    assert [b["n"] for b in r.json()["data"]["baselines"]] == [5]
    fake.fail_with = [503] * 3                      # the risk-result write fails after the history was updated
    r = C.post("/v1/predict/risk", json={"user_id": uid, "domain_inputs": {"cardiac": {"s": 0.2}}}, headers=H)
    assert r.status_code == 200 and r.json()["warnings"] and len(A.STATE.risk_history[uid]) == 1
    fake.fail_with = [503] * 6                      # the latest-state write and the flush after it both fail
    r = C.post("/v1/compare", json={"user_id": uid, "readings": readings(1, 95.0)}, headers=H)
    assert r.status_code == 200 and r.json()["warnings"]
    assert (uid, "latest") in A.STATE.state_dirty
    n = fake.calls
    C.get(f"/v1/users/{uid}/baseline", headers=H)   # no storage work of its own: no flush, no D1 call
    assert fake.calls == n and (uid, "latest") in A.STATE.state_dirty
    # the next request that works with storage flushes the latest state in its own deadline
    C.post("/v1/predict/risk", json={"user_id": uid, "domain_inputs": {"cardiac": {"s": 0.2}}}, headers=H)
    assert fake.count("user_state", uid) == 1 and not A.STATE.state_dirty


def test_f3_unbind_writes_storage_first_and_is_retryable(api):
    C, H, A, st, fake, s3 = api
    uid = new_user(C, H)
    assert C.post("/v1/devices/bind", json={"user_id": uid, "device_id": "d-unb", "kind": "hr_strap"},
                  headers=H).status_code == 200
    fake.fail_with = [503] * 3
    assert C.delete(f"/v1/devices/d-unb?user_id={uid}", headers=H).status_code == 503
    assert "d-unb" in A.STATE.devices and fake.count("devices", uid) == 1
    assert C.delete(f"/v1/devices/d-unb?user_id={uid}", headers=H).status_code == 200
    assert "d-unb" not in A.STATE.devices and fake.count("devices", uid) == 0


def test_f3_unbind_deletes_stored_row_not_in_memory_but_only_for_owner(api):
    C, H, A, st, fake, s3 = api
    a, b = new_user(C, H), new_user(C, H)
    assert C.post("/v1/devices/bind", json={"user_id": a, "device_id": "d-mem", "kind": "hr_strap"},
                  headers=H).status_code == 200
    del A.STATE.devices["d-mem"]                    # binding in storage only
    assert C.delete(f"/v1/devices/d-mem?user_id={b}", headers=H).status_code == 404
    assert fake.count("devices", a) == 1
    assert C.delete(f"/v1/devices/d-mem?user_id={a}", headers=H).status_code == 200
    assert fake.count("devices", a) == 0


def test_f3_bind_failure_rolls_back_memory(api):
    C, H, A, st, fake, s3 = api
    uid = new_user(C, H)
    fake.fail_with = [503] * 3
    assert C.post("/v1/devices/bind", json={"user_id": uid, "device_id": "d-bind", "kind": "hr_strap"},
                  headers=H).status_code == 503
    assert "d-bind" not in A.STATE.devices


def test_f3_reset_archives_in_storage_before_memory(api):
    C, H, A, st, fake, s3 = api
    uid = new_user(C, H)
    C.post("/v1/baseline/vitals", json={"user_id": uid, "readings": readings(4), "mode": "demo"}, headers=H)
    fake.fail_with = [503] * 3
    r = C.post("/v1/baseline/vitals/reset", json={"user_id": uid, "reason": "user_request"}, headers=H)
    assert r.status_code == 503
    assert A.STATE.baselines.for_user(uid)          # memory untouched, so the retry archives the same baseline
    r = C.post("/v1/baseline/vitals/reset", json={"user_id": uid, "reason": "user_request"}, headers=H)
    assert r.status_code == 200 and r.json()["data"]["archived"] == 1
    assert fake.count("baselines", uid) == 0 and fake.count("baseline_archive", uid) == 1


# ---------------- 4. D1 outage: short timeouts, few retries, breaker, deadline, health ----------------
def test_f4_timeouts_and_default_retries():
    fake = FakeD1()
    d1 = D1Client("acct", "db", TOKEN, transport=fake.transport(), sleep=lambda s: None)
    t = d1._client.timeout
    assert t.connect == 3.0 and t.read == 5.0 and d1.retries == 2
    fake.fail_with = [503] * 3
    with pytest.raises(D1Error):
        d1.query("SELECT 1")
    assert fake.calls == 3


def test_f4_retry_after_is_honoured_and_capped_by_deadline():
    fake = FakeD1()
    slept = []
    d1 = D1Client("acct", "db", TOKEN, transport=fake.transport(), sleep=slept.append, backoff_s=0.5)
    fake.fail_with = [(429, {"Retry-After": "2"})]
    assert d1.query("SELECT 1 AS x") == [{"x": 1}] and slept == [2.0]
    slept.clear()
    fake.fail_with = [(503, {"Retry-After": "3"})]
    with storage_deadline(1.0), pytest.raises(D1Error, match="deadline"):
        d1.query("SELECT 1")
    assert slept == []


def test_f4_circuit_breaker_fails_fast_then_half_opens():
    fake = FakeD1()
    now = [1000.0]
    d1 = D1Client("acct", "db", TOKEN, transport=fake.transport(), sleep=lambda s: None, clock=lambda: now[0],
                  breaker_threshold=3, breaker_cooldown_s=30.0)
    fake.fail_with = [httpx.ConnectTimeout("t")] * 3
    with pytest.raises(D1Error):
        d1.query("SELECT 1")
    n = fake.calls
    with pytest.raises(D1Error, match="circuit open"):
        d1.query("SELECT 1")
    assert fake.calls == n                          # no network call while open
    now[0] += 31
    assert d1.query("SELECT 1 AS x") == [{"x": 1}]  # half-open trial succeeds and closes it


def test_f4_request_deadline_stops_retries():
    fake = FakeD1()
    d1 = D1Client("acct", "db", TOKEN, transport=fake.transport(), sleep=lambda s: None)
    with storage_deadline(0.0), pytest.raises(D1Error, match="deadline"):
        d1.query("SELECT 1")
    assert fake.calls == 0


def test_f4_outage_fails_fast_and_health_never_waits_on_d1(api):
    C, H, A, st, fake, s3 = api
    st.d1.breaker_threshold = 5                     # the production default
    uid = new_user(C, H)
    A.STATE.baselines.add(uid, "vitals", "hr_bpm", 60.0)        # dirty: a flush would call D1
    fake.fail_with = [httpx.ConnectTimeout("t")] * 100
    r = C.get(f"/v1/users/{uuid.uuid4()}/baseline", headers=H)
    assert r.status_code == 503
    for i in range(3):
        C.get(f"/v1/users/{uuid.uuid4()}/baseline", headers=H)
    n = fake.calls
    t0 = time.monotonic()
    r = C.get(f"/v1/users/{uuid.uuid4()}/baseline", headers=H)
    assert r.status_code == 503 and fake.calls == n and time.monotonic() - t0 < 1.0     # breaker open
    r = C.get("/v1/health", headers=H)
    assert r.status_code == 200 and fake.calls == n
    import inspect
    assert inspect.iscoroutinefunction(A.health)    # not queued behind a threadpool blocked on D1


# ---------------- 5. default memory mode keeps no history copies ----------------
def test_f5_default_backend_drops_history_and_blobs(monkeypatch):
    monkeypatch.delenv("EVADE_STORAGE", raising=False)
    monkeypatch.setenv("EVADE_STORE_RAW_WINDOWS", "1")
    st = from_env()
    assert isinstance(st, MemoryBackend) and not st.persistent
    st.add_inputs("u", [{"kind": "reading:vitals", "t_utc": NOW, "meta": {}}])
    st.add_flags("u", [{"t": 1.0, "level": "red"}])
    st.add_risk("u", 1.0, 0.3, "low", {})
    assert st.put_raw_window("u", "w", b"x" * 1000) is None
    assert st.list_inputs("u") == [] and st.list_flags("u") == [] and st.risk_history("u") == []
    assert not st.inputs and not st.flags and not st.risk and not st.blobs


# ---------------- 6. separate read-only models bucket ----------------
def test_f6_models_bucket_read_only_for_servers(monkeypatch):
    from evade import storage as S
    seen = []
    monkeypatch.setattr(R2Blobs, "from_credentials",
                        classmethod(lambda cls, acct, bucket, ak, sk, endpoint=None: seen.append((bucket, ak, sk)) or "blobs"))
    for k in list(os.environ):
        if k.startswith("CF_R2"):
            monkeypatch.delenv(k)
    monkeypatch.setenv("CF_ACCOUNT_ID", "acct")
    monkeypatch.setenv("CF_R2_BUCKET", "data-bucket")
    monkeypatch.setenv("CF_R2_ACCESS_KEY_ID", "data-rw-key")
    monkeypatch.setenv("CF_R2_SECRET_ACCESS_KEY", "data-rw-secret")
    monkeypatch.setenv("CF_R2_MODELS_BUCKET", "models-bucket")
    monkeypatch.setenv("CF_R2_MODELS_ACCESS_KEY_ID", "models-ro-key")
    monkeypatch.setenv("CF_R2_MODELS_SECRET_ACCESS_KEY", "models-ro-secret")
    S.models_r2_from_env()
    assert seen[-1] == ("models-bucket", "models-ro-key", "models-ro-secret")
    with pytest.raises(StorageConfigError, match="CF_R2_MODELS_WRITE_ACCESS_KEY_ID"):
        S.models_r2_from_env(write=True)
    monkeypatch.setenv("CF_R2_MODELS_WRITE_ACCESS_KEY_ID", "models-rw-key")
    monkeypatch.setenv("CF_R2_MODELS_WRITE_SECRET_ACCESS_KEY", "models-rw-secret")
    S.models_r2_from_env(write=True)
    assert seen[-1] == ("models-bucket", "models-rw-key", "models-rw-secret")


def test_f6_pull_uses_a_read_only_view():
    pytest.importorskip("joblib")
    from evade.models.fetch import ReadOnlyBlobs
    ro = ReadOnlyBlobs(R2Blobs("bkt", FakeS3()))
    assert not hasattr(ro, "put") and not hasattr(ro, "delete_prefix")
    assert ro.list("models/") == []


# ---------------- 7, 8, 11. legacy pins, read-once loading, generic reasons ----------------
@pytest.fixture()
def sk():
    pytest.importorskip("sklearn")
    joblib = pytest.importorskip("joblib")
    from sklearn.linear_model import LogisticRegression
    return joblib, LogisticRegression


def legacy_folder(root, joblib, LR, task="seizure"):
    d = root / task
    d.mkdir()
    joblib.dump(LR().fit(np.eye(4), [0, 1, 0, 1]), d / "model.joblib")
    (d / "feature_names.json").write_text(json.dumps(["a", "b", "c", "d"]))
    return d


def test_f7_legacy_pin(tmp_path, sk, monkeypatch):
    from evade.models import manifest as MF
    from evade.models.registry import ModelRegistry
    joblib, LR = sk
    d = legacy_folder(tmp_path, joblib, LR)
    monkeypatch.delenv("EVADE_LEGACY_SHA256_SEIZURE", raising=False)
    e = ModelRegistry(tmp_path, allow_draft=True).scan()["seizure"]
    assert e.available and not e.sha256_verified and e.public()["integrity"] == "unverified"
    monkeypatch.setenv("EVADE_LEGACY_SHA256_SEIZURE", "0" * 64)
    e = ModelRegistry(tmp_path, allow_draft=True).scan()["seizure"]
    assert not e.available and e.reason_code == "sha256_mismatch" and e.model is None
    monkeypatch.setenv("EVADE_LEGACY_SHA256_SEIZURE", MF.sha256_file(d / "model.joblib"))
    e = ModelRegistry(tmp_path, allow_draft=True).scan()["seizure"]
    # the pin checks the weights, but feature_names.json is read without a hash, so the entry is not verified
    assert e.available and not e.sha256_verified and e.public()["integrity"] == "unverified"


def test_f7_api_marks_legacy_unverified(tmp_path, sk, monkeypatch, api):
    from evade.models.registry import ModelRegistry
    C, H, A, *_ = api
    joblib, LR = sk
    legacy_folder(tmp_path, joblib, LR)
    monkeypatch.delenv("EVADE_LEGACY_SHA256_SEIZURE", raising=False)
    reg = ModelRegistry(tmp_path, allow_draft=True)
    reg.scan()
    monkeypatch.setattr(A.STATE, "registry", reg)
    monkeypatch.setattr(A.STATE, "models", reg.served())
    row = C.get("/v1/models", headers=H).json()["data"]["models"][0]
    assert row["task"] == "seizure" and row["integrity"] == "unverified" and row["legacy"] is True


def _manifest_folder(root, joblib, LR, obj=None):
    from evade.models.export import write_manifest
    d = root / "spaceweather"
    d.mkdir(parents=True)
    X = np.random.default_rng(0).standard_normal((40, 3))
    joblib.dump(obj if obj is not None else LR().fit(X, (X[:, 0] > 0).astype(int)), d / "model.joblib")
    (d / "feature_names.json").write_text(json.dumps(["a", "b", "c"]))
    write_manifest(d, task="spaceweather", model_id="EVA-99", status="draft",
                   input_spec={"kind": "features", "modality": "space_weather", "features": "feature_names.json"},
                   output_spec={"type": "binary"}, data="test", commit="abc")
    return d


def test_f8_loader_never_called_on_sha_mismatch(tmp_path, sk, monkeypatch):
    from evade.models import registry as R
    joblib, LR = sk
    d = _manifest_folder(tmp_path, joblib, LR)
    calls = []
    real = R.LOADERS["joblib"]
    monkeypatch.setitem(R.LOADERS, "joblib", lambda data: calls.append(data) or real(data))
    with open(d / "model.joblib", "ab") as f:
        f.write(b"tampered")
    e = R.ModelRegistry(tmp_path, allow_draft=True).scan()["spaceweather"]
    assert not e.available and e.reason_code == "sha256_mismatch" and calls == []


def test_f8_loads_the_hashed_bytes_and_type_checks(tmp_path, sk, monkeypatch):
    from evade.models import manifest as MF
    from evade.models import registry as R
    joblib, LR = sk
    d = _manifest_folder(tmp_path, joblib, LR)
    got = []
    real = R.LOADERS["joblib"]
    monkeypatch.setitem(R.LOADERS, "joblib", lambda data: got.append(data) or real(data))
    e = R.ModelRegistry(tmp_path, allow_draft=True).scan()["spaceweather"]
    assert e.available and isinstance(got[0], bytes) and MF.sha256_bytes(got[0]) == e.manifest["sha256"]
    other = tmp_path / "other"
    _manifest_folder(other, joblib, LR, obj={"not": "a model"})
    e = R.ModelRegistry(other, allow_draft=True).scan()["spaceweather"]
    assert not e.available and e.reason_code == "unexpected_model_type"


def test_f11_unavailable_reason_is_generic(tmp_path, sk, monkeypatch, api):
    from evade.models.registry import ModelRegistry
    C, H, A, *_ = api
    joblib, LR = sk
    d = _manifest_folder(tmp_path, joblib, LR)
    with open(d / "model.joblib", "ab") as f:
        f.write(b"x")
    (tmp_path / "broken").mkdir()
    (tmp_path / "broken" / "manifest.json").write_text("{not json")
    reg = ModelRegistry(tmp_path, allow_draft=True)
    reg.scan()
    monkeypatch.setattr(A.STATE, "registry", reg)
    un = {u["task"]: u for u in C.get("/v1/models", headers=H).json()["data"]["unavailable"]}
    assert un["spaceweather"]["reason_code"] == "sha256_mismatch"
    assert un["broken"]["reason_code"] == "manifest_invalid"
    text = json.dumps(un)
    assert "JSONDecode" not in text and "Expecting" not in text and str(tmp_path) not in text
    assert "Error" not in un["broken"]["reason"]


# ---------------- 9, 10. fetch swap rollback, parsed manifest comparison ----------------
def test_f9_failed_swap_restores_the_old_folder(tmp_path, sk, monkeypatch):
    import pathlib
    from evade.models.export import restamp
    from evade.models.fetch import pull, push
    from evade.models.registry import ModelRegistry
    joblib, LR = sk
    src = _manifest_folder(tmp_path / "src", joblib, LR)
    blobs = R2Blobs("bkt", FakeS3())
    push(blobs, src)
    dest = tmp_path / "dest"
    assert pull(blobs, dest) == {"spaceweather": "fetched"}
    restamp(src, status="validated")
    push(blobs, src)
    real = pathlib.Path.rename

    def flaky(self, target):
        if self.name.startswith(".fetch-"):
            raise OSError("injected rename failure")
        return real(self, target)

    monkeypatch.setattr(pathlib.Path, "rename", flaky)
    res = pull(blobs, dest)
    assert res["spaceweather"].startswith("failed")
    monkeypatch.setattr(pathlib.Path, "rename", real)
    e = ModelRegistry(dest, allow_draft=True).scan()["spaceweather"]
    assert e.available and e.manifest["status"] == "draft"          # the old folder is back
    assert not [p for p in dest.iterdir() if p.name.startswith(".")]


def test_f10_hand_written_manifest_is_up_to_date_on_second_pull(tmp_path, sk):
    from evade.models import manifest as MF
    from evade.models.fetch import pull, push
    joblib, LR = sk
    d = tmp_path / "src" / "spaceweather"
    d.mkdir(parents=True)
    X = np.random.default_rng(2).standard_normal((40, 3))
    joblib.dump(LR().fit(X, (X[:, 1] > 0).astype(int)), d / "model.joblib")
    m = {"model_id": "EVA-95", "task": "spaceweather", "status": "draft", "format": "joblib",
         "weights": "model.joblib", "sha256": MF.sha256_file(d / "model.joblib"),
         "input": {"modality": "space_weather"}, "output": {"labels": ["x"]}, "commit": "abc"}
    (d / "manifest.json").write_text(json.dumps(m))
    blobs = R2Blobs("bkt", FakeS3())
    push(blobs, d)
    dest = tmp_path / "dest"
    assert pull(blobs, dest) == {"spaceweather": "fetched"}
    assert pull(blobs, dest) == {"spaceweather": "up_to_date"}


# ---------------- 12. miss cache and bounded history ----------------
def test_f12_unknown_user_misses_are_cached(api):
    C, H, A, st, fake, s3 = api
    n = fake.calls
    nobody = str(uuid.uuid4())
    assert C.get(f"/v1/users/{nobody}/baseline", headers=H).status_code == 404
    assert C.get(f"/v1/users/{nobody}/baseline", headers=H).status_code == 404
    assert fake.calls == n + 1


def test_f12_hydrated_risk_history_is_bounded_but_keeps_the_7_day_point(api, monkeypatch):
    C, H, A, st, fake, s3 = api
    uid = new_user(C, H)
    now = time.time()
    st.add_risk(uid, now - 8 * 86400, 0.5, "elevated", {})
    for i in range(150):
        st.add_risk(uid, now - 3600 + i, 0.3, "low", {})
    restart(A, fake, s3, monkeypatch)
    assert C.get(f"/v1/users/{uid}/baseline", headers=H).status_code == 200
    hist = A.STATE.risk_history[uid]
    assert len(hist) <= 100 + 170 and len(hist) < 151 and hist[0] == (pytest.approx(now - 8 * 86400), 0.5)


def _r7(hist, now):
    old = [x for x in hist if now - x[0] >= 7 * 86400]
    return old[-1] if old else None


def test_f12_bounded_history_still_answers_the_7_day_lookup_later():
    from evade.storage.base import bound_history
    now = 1_000_000_000.0
    rng = np.random.default_rng(3)
    full = sorted((now - float(s), 0.5) for s in rng.uniform(0, 10 * 86400, 2000))
    b = bound_history(full, 100, now)
    assert len(b) <= 100 + 170
    for later in (0.0, 3600.0, 2 * 86400.0, 6.9 * 86400):         # the lookup as the bounded points age
        exact, got = _r7(full, now + later), _r7(b, now + later)
        assert got is not None and exact[0] - got[0] <= 3600.0


def test_f12_in_memory_history_is_bounded_and_keeps_the_trend_input(api, monkeypatch):
    """Many risk calls: the in-memory history stays bounded, and every 7-day trend input matches the one an
    untrimmed history would give, to within an hour."""
    C, H, A, st, fake, s3 = api
    monkeypatch.setattr(A, "persist_soft", lambda *a, **k: None)     # memory behaviour only; faster
    uid = new_user(C, H)
    clock = [time.time() - 9 * 86400]
    monkeypatch.setattr(time, "time", lambda: clock[0])
    seen = []
    real = A.mdtr
    monkeypatch.setattr(A, "mdtr", lambda *a, **k: seen.append(k.get("r_7d_ago")) or real(*a, **k))
    shadow, lens, n = [], [], 1200
    for i in range(n):
        r = C.post("/v1/predict/risk", json={"user_id": uid, "domain_inputs": {"cardiac": {"s": (i % 97) / 100}},
                                             "logistics_floor_days": 0.5}, headers=H)
        assert r.status_code == 200
        exact = _r7(shadow, clock[0])                  # the lookup on the full history, before this call's point
        got = seen[-1]
        assert (exact is None) == (got is None), i
        if exact is not None:
            match = [t for t, v in shadow if v == got and exact[0] - 3600.0 <= t <= exact[0]]
            assert match, (i, exact, got)
        shadow.append((clock[0], A.STATE.risk_history[uid][-1][1]))
        lens.append(len(A.STATE.risk_history[uid]))
        clock[0] += 9 * 86400 / n                      # about one call every 11 minutes over 9 days
    assert max(lens) <= A.RISK_MAX and len(shadow) == n and seen[-1] is not None


# ---------------- 13. device last_seq never rewinds ----------------
def test_f13_device_seq_is_monotonic_and_update_never_inserts():
    st, fake, s3 = backend()
    st.put_user("u", {"created_at": 1.0})
    st.put_device("d", {"user_id": "u", "kind": "hr_strap", "last_seq": 5})
    st.update_device("d", {"user_id": "u", "kind": "hr_strap", "last_seq": 3})
    assert st.get_device("d")["last_seq"] == 5
    st.update_device("d", {"user_id": "u", "kind": "hr_strap", "last_seq": 12})
    assert st.get_device("d")["last_seq"] == 12
    st.update_device("d", {"user_id": "u", "kind": "hr_strap", "last_seq": 9})
    assert st.get_device("d")["last_seq"] == 12
    st.delete_device("d")
    st.update_device("d", {"user_id": "u", "kind": "hr_strap", "last_seq": 20})
    assert st.get_device("d") is None


# ---------------- 14. a reset baseline cannot come back through a stale flush ----------------
@pytest.mark.parametrize("persisted_first", [False, True])
def test_f14_stale_flush_after_reset_is_ignored(persisted_first):
    from evade.baseline.store import BaselineStore
    st, fake, s3 = backend()
    st.put_user("u", {"created_at": 1.0})
    bs = BaselineStore(st)
    bs.add("u", "ecg", "hr_bpm", 60.0)
    if persisted_first:
        bs.flush()
        bs.add("u", "ecg", "hr_bpm", 61.0)
    orig = st.put_baselines

    def racing(recs):                               # the flush has its snapshot; the reset lands first
        bs.reset("u", "ecg")
        return orig(recs)

    st.put_baselines = racing
    bs.flush()
    assert fake.count("baselines", "u") == 0 and fake.count("baseline_archive", "u") == 1
    st.put_baselines = orig
    bs.add("u", "ecg", "hr_bpm", 70.0)              # a new baseline after the reset is stored normally
    bs.flush()
    assert fake.count("baselines", "u") == 1
