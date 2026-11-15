"""Regression tests for the residual gaps found by the adversarial verification of the storage branch (items 1-12).
Rebuilt from the verifier's descriptions. No network: D1 is SQLite behind httpx.MockTransport, R2 an S3 double."""
import json
import time
import uuid
from datetime import datetime, timezone

import numpy as np
import pytest

pytest.importorskip("httpx")
import httpx  # noqa: E402

from evade.baseline.store import BaselineStore  # noqa: E402
from evade.storage import StorageConfigError, StorageError  # noqa: E402
from evade.storage import base as B  # noqa: E402
from evade.storage.cloudflare import D1Client, D1Error, R2Blobs, check_schema  # noqa: E402
from tests.fakes import TOKEN, FakeD1, FakeS3  # noqa: E402
from tests.test_review_fixes import USER_TABLES, api, backend, new_user, readings, restart  # noqa: E402,F401

NOW = datetime.now(timezone.utc).isoformat()


def ecg_window(hr=60.0, secs=10.0, fs=250.0, seed=0):
    rng = np.random.default_rng(seed)
    t = np.arange(int(secs * fs)) / fs
    x = 0.02 * rng.standard_normal(t.size)
    for b in np.arange(0.5, secs, 60.0 / hr):
        x += np.exp(-0.5 * ((t - b) / 0.012) ** 2)
    return {"modality": "ecg", "t0_utc": NOW, "fs_hz": fs, "unit": "mv", "channels": ["II"],
            "samples": [x.round(4).tolist()]}


# ---------------- 1. finding 14: a baseline created and flushed during a reset is not lost ----------------
def test_1_baseline_created_and_flushed_between_snapshot_and_archive_survives():
    st, fake, s3 = backend()
    st.put_user("u", {"created_at": 1.0})
    bs = BaselineStore(st)
    bs.add("u", "ecg", "hr_bpm", 60.0)
    bs.flush()
    orig = st.archive_baselines

    def racing(user_id, modality, records=None):   # the reset has its snapshot; a new metric is created and flushed
        bs.add("u", "ecg", "rmssd_ms", 40.0)
        bs.flush()
        return orig(user_id, modality, records)

    st.archive_baselines = racing
    assert bs.reset("u", "ecg") == 1
    st.archive_baselines = orig
    assert {b["metric"] for b in bs.for_user("u")} == {"rmssd_ms"}
    assert {b["metric"] for b in st.get_baselines("u")} == {"rmssd_ms"}    # before the fix: [] (lost on restart)
    bs.add("u", "ecg", "rmssd_ms", 41.0)
    bs.flush()                                      # later flushes of it are not dropped as "archived"
    st2, _, _ = backend(fake, s3)                   # restart
    rows = st2.get_baselines("u")
    assert [(r["metric"], r["values"]) for r in rows] == [("rmssd_ms", [40.0, 41.0])]
    assert fake.count("baseline_archive", "u") == 1


def test_1_archived_id_of_another_user_does_not_block_a_live_baseline():
    from tests.test_review_fixes import brec
    st, fake, s3 = backend()
    for u in ("ua", "ub"):
        st.put_user(u, {"created_at": 1.0})
    st.archive_baselines("ua", "ecg", [brec("ua", bid="b_ecg_deadbeef")])
    st.put_baselines([brec("ub", bid="b_ecg_deadbeef")])      # forced id collision across users
    assert fake.count("baselines", "ub") == 1


def test_1_reset_whose_archive_committed_but_response_was_lost_settles_memory():
    fake = FakeD1()
    lost = []

    def handler(request):
        resp = fake.handler(request)
        body = json.loads(request.content)
        if not lost and "batch" in body and "baseline_archive" in json.dumps(body):
            lost.append(1)
            raise httpx.ReadTimeout("response lost after commit")
        return resp

    from evade.storage.cloudflare import CloudflareBackend
    d1 = D1Client("acct", "db", TOKEN, transport=httpx.MockTransport(handler), sleep=lambda s: None, retries=0)
    st = CloudflareBackend(d1, R2Blobs("bkt", FakeS3()))
    st.put_user("u", {"created_at": 1.0})
    bs = BaselineStore(st)
    bs.add("u", "ecg", "hr_bpm", 60.0)
    bs.flush()
    assert bs.reset("u", "ecg") == 1 and lost        # the archive is checked, so memory follows storage
    assert bs.for_user("u") == [] and fake.count("baselines", "u") == 0 and fake.count("baseline_archive", "u") == 1


# ---------------- 2. schema version checked at startup ----------------
def test_2_missing_migration_2_refuses_cloudflare_storage(monkeypatch):
    from evade import storage as S
    fake = FakeD1()
    d1 = D1Client("acct", "db", TOKEN, transport=fake.transport(), sleep=lambda s: None)
    assert check_schema(d1) == 2
    fake.db.execute("DELETE FROM schema_migrations WHERE version = 2")
    fake.db.execute("DROP TABLE pending_deletions")
    fake.db.commit()
    with pytest.raises(StorageConfigError, match=r"migration 2 \(pending_deletions\)"):
        check_schema(d1)
    monkeypatch.setenv("EVADE_STORAGE", "cloudflare")
    monkeypatch.setattr(S, "d1_from_env", lambda: d1)
    with pytest.raises(StorageConfigError, match="python -m evade.storage.migrate"):
        S.from_env()
    fake.db.execute("DROP TABLE schema_migrations")
    fake.db.commit()
    with pytest.raises(StorageConfigError, match="schema missing"):
        check_schema(d1)
    fake.fail_with = [403]                          # a bad token is not reported as a schema problem
    with pytest.raises(D1Error, match="HTTP 403"):
        check_schema(d1)


# ---------------- 4. one deadline per request; no flush without storage work ----------------
def test_4_no_flush_for_requests_without_storage_work(api):
    C, H, A, st, fake, s3 = api
    uid = new_user(C, H)
    A.STATE.baselines.add(uid, "vitals", "hr_bpm", 60.0)        # dirty, as if left by another request
    n = fake.calls
    assert C.get("/v1/models", headers=H).status_code == 200
    assert C.get("/v1/models").status_code == 401
    assert C.get(f"/v1/users/{uid}/baseline", headers=H).status_code == 200     # held in memory
    assert C.get("/v1/no-such-path", headers=H).status_code == 404
    assert fake.calls == n and A.STATE.baselines.dirty


def test_4_flush_shares_the_request_deadline(api):
    C, H, A, st, fake, s3 = api
    uid = new_user(C, H)
    A.STATE.baselines.add(uid, "vitals", "hr_bpm", 60.0)
    seen = []
    fake.on_request = lambda body: seen.append(B._DEADLINE.get())
    assert C.post("/v1/compare", json={"user_id": uid, "readings": readings(1)}, headers=H).status_code == 200
    assert len(seen) >= 2 and seen[0] is not None and len(set(seen)) == 1   # endpoint writes and flush: one deadline
    assert not A.STATE.baselines.dirty


def test_4_confirm_on_an_in_memory_user_is_flushed(api):
    C, H, A, st, fake, s3 = api
    uid = new_user(C, H)
    assert C.post("/v1/baseline/vitals", json={"user_id": uid, "readings": readings(12), "mode": "demo"},
                  headers=H).status_code == 200
    assert C.post("/v1/baseline/vitals/confirm", json={"user_id": uid}, headers=H).json()["data"]["confirmed"] == 1
    assert [r["status"] for r in st.get_baselines(uid)] == ["confirmed"]


# ---------------- 5. DELETE uses the miss cache, without breaking the retried deletion ----------------
def test_5_delete_of_unknown_user_uses_the_miss_cache(api):
    C, H, A, st, fake, s3 = api
    uid = str(uuid.uuid4())
    n = fake.calls
    assert C.delete(f"/v1/users/{uid}", headers=H).status_code == 404
    assert fake.calls == n + 1                      # one storage request
    assert C.delete(f"/v1/users/{uid}", headers=H).status_code == 404
    assert C.get(f"/v1/users/{uid}/baseline", headers=H).status_code == 404
    assert fake.calls == n + 1                      # cached


def test_5_get_between_failed_delete_and_retry_does_not_cache_a_miss(api, monkeypatch):
    C, H, A, st, fake, s3 = api
    uid = new_user(C, H)
    st.put_blob(f"exports/{uid}/summary.json", b"{}")
    s3.fail_delete = True
    assert C.delete(f"/v1/users/{uid}", headers=H).status_code == 503
    restart(A, fake, s3, monkeypatch)
    s3.fail_delete = False
    assert C.get(f"/v1/users/{uid}/baseline", headers=H).status_code == 404
    assert C.delete(f"/v1/users/{uid}", headers=H).status_code == 200      # still finishes the deletion
    assert s3.keys(f"exports/{uid}/") == []
    assert C.delete(f"/v1/users/{uid}", headers=H).status_code == 404


# ---------------- 6. R2 calls bounded by timeouts, attempts and the deadline ----------------
def test_6_r2_client_attempts_timeouts_and_deadline():
    pytest.importorskip("boto3")
    r2 = R2Blobs.from_credentials("acct", "bkt", "test-access-key", "test-secret-key", endpoint="http://127.0.0.1:9")
    cfg = r2._s3.meta.config
    assert cfg.read_timeout <= 5 and cfg.connect_timeout <= 3 and cfg.retries["total_max_attempts"] == 2
    attempts = []
    r2._s3.meta.events.register("before-send.s3", lambda **kw: attempts.append(1))
    with pytest.raises(StorageError, match="R2 put failed"):
        r2.put("raw/u/w.json", b"{}", "application/json")      # nothing listens on port 9
    assert len(attempts) == 2                       # first try plus one retry, no more

    attempts.clear()
    r2._s3.meta.events.register("before-send.s3", lambda **kw: time.sleep(0.4) if not attempts[1:] else None)
    with B.storage_deadline(0.3), pytest.raises(StorageError, match="deadline"):
        r2.put("raw/u/w.json", b"{}", "application/json")
    assert len(attempts) == 1                       # the retry was not started after the deadline


# ---------------- 7. half-open breaker lets one trial through ----------------
def test_7_half_open_breaker_allows_a_single_trial():
    fake = FakeD1()
    now = [1000.0]
    d1 = D1Client("acct", "db", TOKEN, transport=fake.transport(), sleep=lambda s: None, clock=lambda: now[0],
                  breaker_threshold=2, breaker_cooldown_s=30.0, retries=0)
    fake.fail_with = [httpx.ConnectTimeout("t")] * 2
    for _ in range(2):
        with pytest.raises(D1Error):
            d1.query("SELECT 1")
    now[0] += 31
    inner = []

    def concurrent(body):                           # another caller while the trial is in flight
        if not inner:
            inner.append(1)
            with pytest.raises(D1Error, match="half-open"):
                d1.query("SELECT 2")

    fake.on_request = concurrent
    assert d1.query("SELECT 1 AS x") == [{"x": 1}] and inner
    fake.on_request = None
    assert d1.query("SELECT 1 AS x") == [{"x": 1}]  # closed again: everyone goes through
    fake.fail_with = [httpx.ConnectTimeout("t")] * 3
    for _ in range(2):
        with pytest.raises(D1Error):
            d1.query("SELECT 1")
    now[0] += 31
    with pytest.raises(D1Error):                    # the trial fails ...
        d1.query("SELECT 1")
    n = fake.calls
    with pytest.raises(D1Error, match="circuit open"):   # ... and the circuit is open again
        d1.query("SELECT 1")
    assert fake.calls == n


# ---------------- 8. models bucket fails closed ----------------
def test_8_models_bucket_fails_closed_without_its_own_bucket(monkeypatch):
    import os
    from evade import storage as S
    seen = []
    monkeypatch.setattr(R2Blobs, "from_credentials",
                        classmethod(lambda cls, acct, bucket, ak, sk, endpoint=None: seen.append((bucket, ak)) or "b"))
    for k in list(os.environ):
        if k.startswith("CF_R2") or k == "EVADE_ALLOW_SHARED_MODELS_BUCKET":
            monkeypatch.delenv(k)
    monkeypatch.setenv("CF_ACCOUNT_ID", "acct")
    monkeypatch.setenv("CF_R2_BUCKET", "data-bucket")
    monkeypatch.setenv("CF_R2_ACCESS_KEY_ID", "data-rw-key")
    monkeypatch.setenv("CF_R2_SECRET_ACCESS_KEY", "data-rw-secret")
    with pytest.raises(StorageConfigError, match="CF_R2_MODELS_BUCKET") as ei:
        S.models_r2_from_env()
    assert "data-rw" not in str(ei.value) and seen == []
    monkeypatch.setenv("EVADE_ALLOW_SHARED_MODELS_BUCKET", "1")
    S.models_r2_from_env()
    assert seen == [("data-bucket", "data-rw-key")]


# ---------------- 9. every file the loader reads is hashed, or the entry is unverified ----------------
def test_9_unhashed_sidecar_marks_the_entry_unverified(tmp_path):
    pytest.importorskip("sklearn")
    joblib = pytest.importorskip("joblib")
    from sklearn.linear_model import LogisticRegression
    from evade.models.registry import ModelRegistry
    from tests.test_review_fixes import _manifest_folder
    d = _manifest_folder(tmp_path, joblib, LogisticRegression)
    e = ModelRegistry(tmp_path, allow_draft=True).scan()["spaceweather"]
    assert e.available and e.sha256_verified
    (d / "metrics.json").write_text(json.dumps({"auroc": 0.99}))      # added after stamping: not in manifest.files
    e = ModelRegistry(tmp_path, allow_draft=True).scan()["spaceweather"]
    assert e.available and not e.sha256_verified and e.public()["integrity"] == "unverified"
    (d / "feature_names.json").write_text(json.dumps(["c", "b", "a"]))   # listed in manifest.files: refused
    e = ModelRegistry(tmp_path, allow_draft=True).scan()["spaceweather"]
    assert not e.available and e.reason_code == "sha256_mismatch"


# ---------------- 10. no memory residue for a user deleted mid-request ----------------
@pytest.mark.parametrize("path,body", [
    ("/v1/input/instrument", {"instrument": "phq9", "t_utc": NOW, "answers": [0, 0, 0, 0, 0, 0, 0, 0, 2]}),
    ("/v1/devices/bind", {"device_id": "dev-race-10", "kind": "hr_strap"}),
    ("/v1/input/self-report", {"t_utc": NOW, "sleep_quality": 2, "stress": 4, "fatigue": 4, "mood": 2}),
])
def test_10_concurrent_delete_leaves_no_memory_state(api, monkeypatch, path, body):
    C, H, A, st, fake, s3 = api
    uid = new_user(C, H)
    real = A.user

    def racing(u):                                  # the DELETE lands after the user check, before the lock
        rec = real(u)
        A.delete_user(u)
        return rec

    monkeypatch.setattr(A, "user", racing)
    r = C.post(path, json={**body, "user_id": uid}, headers=H)
    monkeypatch.setattr(A, "user", real)
    assert r.status_code in (200, 404)
    assert uid not in A.STATE.users and uid not in A.STATE.overrides and uid not in A.STATE.inputs
    assert not [k for k, v in A.STATE.devices.items() if v["user_id"] == uid]
    for t in USER_TABLES:
        assert fake.count(t, uid) == 0, t


# ---------------- 12. a raw-window PUT racing a DELETE is cleaned up ----------------
def test_12_raw_window_put_landing_after_delete_is_removed(api):
    C, H, A, st, fake, s3 = api
    uid = new_user(C, H)
    other, _, _ = backend(fake, s3)                 # another process deletes the user
    orig = s3.put_object

    def racing(**kw):                               # the DELETE lists and clears raw/<uid>/ before this PUT lands
        if kw["Key"].startswith(f"raw/{uid}/"):
            other.delete_user(uid)
        return orig(**kw)

    s3.put_object = racing
    r = C.post("/v1/compare", json={"user_id": uid, "windows": [ecg_window()]}, headers=H)
    s3.put_object = orig
    assert r.status_code == 200
    assert s3.keys(f"raw/{uid}/") == []             # before the fix: an orphan blob that no DELETE would remove
    for t in USER_TABLES:
        assert fake.count(t, uid) == 0, t
