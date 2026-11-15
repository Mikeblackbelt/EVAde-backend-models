"""Storage layer: memory backend, Cloudflare backend against a fake D1 (httpx.MockTransport + SQLite) and a fake S3
client, and the API wired to it. No network, no credentials."""
import logging
import os
import time

import pytest

pytest.importorskip("httpx")
import httpx  # noqa: E402

from evade.storage import MemoryBackend, StorageConfigError, from_env  # noqa: E402
from evade.storage.cloudflare import CloudflareBackend, D1Client, D1Error, R2Blobs  # noqa: E402
from tests.fakes import TOKEN, FakeD1, FakeS3  # noqa: E402


def cf_backend(fake=None, s3=None, raw=True, **kw):
    fake = fake or FakeD1()
    d1 = D1Client("acct", "db-uuid", TOKEN, transport=fake.transport(), sleep=lambda s: None, **kw)
    return CloudflareBackend(d1, R2Blobs("evade-bucket", s3 if s3 is not None else FakeS3()), store_raw_windows=raw), fake


def baseline(uid, metric, n=3):
    return {"user_id": uid, "modality": "ecg", "metric": metric, "baseline_id": f"b_{metric}", "mode": "demo",
            "phase": "demo", "status": "building", "values": [60.0 + i for i in range(n)], "center": None,
            "spread": None, "created": time.time()}


def exercise(st):
    """Write one of everything for two users; delete one; check the other is untouched."""
    a, b = "user-a", "user-b"
    for u in (a, b):
        st.put_user(u, {"created_at": 1.0, "cohort": "demo", "age_band": None, "sex": None})
        st.put_baselines([baseline(u, "hr_bpm"), baseline(u, "rmssd_ms")])
        st.put_state(u, "overrides", ["O5"])
        st.put_state(u, "radiation", {"domain": {"s": 0.4, "q": 1.0, "s_validated": None}, "cumulative": 300.0, "rate": 1.8})
        st.add_inputs(u, [{"kind": "window:ecg", "t_utc": "2026-11-14T00:00:00+00:00", "meta": {"fs_hz": 250.0}}])
        st.add_flags(u, [{"t": 2.0, "domain": "cardiac", "metric": "hr_bpm", "modality": "ecg", "level": "red", "z": 3.1,
                          "component_score": 0.73, "confidence": "demo"}])
        st.add_risk(u, 3.0, 0.42, "elevated", {"r_disp": 0.45, "mdtr": {"display": "55 days"}})
        st.put_device(f"dev-{u}", {"user_id": u, "kind": "hr_strap", "last_seq": 4})
        assert st.put_raw_window(u, "w1", b'{"samples": [[0.1]]}') == f"raw/{u}/w1.json"
        st.put_blob(f"exports/{u}/summary.json", b"{}")
    st.put_blob("models/seizure/manifest.json", b"{}")

    assert st.get_user(a)["cohort"] == "demo"
    assert {r["metric"] for r in st.get_baselines(a)} == {"hr_bpm", "rmssd_ms"}
    assert st.get_baselines(a)[0]["values"] == [60.0, 61.0, 62.0]
    assert st.get_state(a)["overrides"] == ["O5"] and st.get_state(a)["radiation"]["cumulative"] == 300.0
    assert st.list_inputs(a)[0]["meta"]["fs_hz"] == 250.0
    assert st.list_flags(a)[0]["level"] == "red"
    assert st.risk_history(a) == [(3.0, 0.42)]
    assert st.get_device(f"dev-{a}")["last_seq"] == 4 and list(st.devices_for_user(a)) == [f"dev-{a}"]

    st.archive_baselines(a, "ecg")
    assert st.get_baselines(a) == []
    st.put_baselines([baseline(a, "hr_bpm")])

    out = st.delete_user(a)
    assert out["blobs"] == 2
    assert st.get_user(a) is None and st.get_baselines(a) == [] and st.get_state(a) == {}
    assert st.list_inputs(a) == [] and st.list_flags(a) == [] and st.risk_history(a) == []
    assert st.get_device(f"dev-{a}") is None and st.devices_for_user(a) == {}
    assert st.list_blobs(f"raw/{a}/") == [] and st.list_blobs(f"exports/{a}/") == []
    # a late write for the deleted user is dropped
    st.put_baselines([baseline(a, "hr_bpm")]); st.put_user(a, {"created_at": 1.0}); st.put_state(a, "overrides", ["O1"])
    assert st.get_baselines(a) == [] and st.get_user(a) is None and st.get_state(a) == {}
    # the other user and the model blobs are untouched
    assert st.get_user(b) is not None and len(st.get_baselines(b)) == 2 and st.risk_history(b) == [(3.0, 0.42)]
    assert st.list_blobs(f"raw/{b}/") == [f"raw/{b}/w1.json"] and st.list_blobs("models/") == ["models/seizure/manifest.json"]


def test_memory_backend_roundtrip_and_delete():
    exercise(MemoryBackend(store_raw_windows=True))


def test_cloudflare_backend_roundtrip_and_delete():
    st, fake = cf_backend()
    exercise(st)
    for t in ("users", "baselines", "baseline_archive", "user_state", "inputs", "flags", "risk_results", "devices"):
        assert fake.count(t, "user-a") == 0, t
    assert fake.count("users", "user-b") == 1
    assert fake.db.execute("SELECT COUNT(*) FROM tombstones").fetchone()[0] == 1


def test_tombstone_survives_restart():
    st, fake = cf_backend()
    st.put_user("u1", {"created_at": 1.0})
    st.delete_user("u1")
    st2, _ = cf_backend(fake)                     # new process: no in-memory tombstone
    assert st2.is_tombstoned("u1")
    st2.put_user("u1", {"created_at": 2.0})       # blocked in SQL by the tombstone
    assert st2.get_user("u1") is None


def test_raw_windows_off_by_default():
    st, _ = cf_backend(raw=False)
    assert st.put_raw_window("u", "w", b"x") is None
    assert MemoryBackend().put_raw_window("u", "w", b"x") is None


def test_d1_parameter_chunking():
    st, fake = cf_backend()
    st.put_baselines([baseline("u", f"m{i}") for i in range(30)])     # 12 columns x 30 rows = 360 parameters
    assert len(st.get_baselines("u")) == 30 and fake.max_params_seen <= 100


def test_d1_retries_then_succeeds():
    fake = FakeD1()
    slept = []
    d1 = D1Client("acct", "db", TOKEN, transport=fake.transport(), sleep=slept.append, backoff_s=0.5)
    fake.fail_with = [503, httpx.ConnectError("boom"), 429]
    assert d1.query("SELECT 1 AS x") == [{"x": 1}]
    assert slept == [0.5, 1.0, 2.0]


def test_d1_gives_up_and_does_not_retry_client_errors():
    fake = FakeD1()
    d1 = D1Client("acct", "db", TOKEN, transport=fake.transport(), sleep=lambda s: None, retries=2)
    fake.fail_with = [503, 503, 503]
    with pytest.raises(D1Error, match="unavailable after 3 attempts"):
        d1.query("SELECT 1")
    fake.fail_with = [403, 200]
    with pytest.raises(D1Error, match="HTTP 403"):
        d1.query("SELECT 1")
    assert fake.fail_with == [200]                  # no retry after a 4xx
    fake.fail_with = []
    with pytest.raises(D1Error, match="no such table"):
        d1.query("SELECT * FROM nope")


def test_d1_success_false_in_200_is_an_error():
    def handler(request):
        return httpx.Response(200, json={"success": False, "errors": [{"code": 1, "message": "bad"}], "result": []})
    d1 = D1Client("acct", "db", TOKEN, transport=httpx.MockTransport(handler), sleep=lambda s: None)
    with pytest.raises(D1Error, match="bad"):
        d1.query("SELECT 1")


def test_secrets_never_logged_or_shown(caplog, monkeypatch):
    caplog.set_level(logging.DEBUG)
    fake = FakeD1()
    st, _ = cf_backend(fake, retries=1)
    fake.fail_with = [500, httpx.ConnectError("x")]
    with pytest.raises(D1Error) as ei:
        st.get_user("u")
    assert TOKEN not in caplog.text and TOKEN not in str(ei.value)
    assert TOKEN not in repr(st) and TOKEN not in repr(st.d1)
    for k in ("CF_ACCOUNT_ID", "CF_D1_DATABASE_ID", "CF_R2_BUCKET", "CF_R2_ACCESS_KEY_ID", "CF_R2_SECRET_ACCESS_KEY"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("EVADE_STORAGE", "cloudflare")
    monkeypatch.setenv("CF_API_TOKEN", "should-not-appear-ABC")
    with pytest.raises(StorageConfigError) as ei:
        from_env()
    assert "CF_ACCOUNT_ID" in str(ei.value) and "should-not-appear-ABC" not in str(ei.value)


def test_from_env_defaults_to_memory(monkeypatch):
    monkeypatch.delenv("EVADE_STORAGE", raising=False)
    assert isinstance(from_env(), MemoryBackend)
    monkeypatch.setenv("EVADE_STORAGE", "postgres")
    with pytest.raises(StorageConfigError):
        from_env()


def test_r2_pagination_and_delete():
    s3 = FakeS3()
    r2 = R2Blobs("bkt", s3)
    for i in range(5):
        r2.put(f"raw/u/{i}.json", b"x", "application/json")
    r2.put("raw/v/0.json", b"y", "application/json")
    assert len(r2.list("raw/u/")) == 5
    assert r2.delete_prefix("raw/u/") == 5 and s3.keys() == ["raw/v/0.json"]
    assert r2.get("raw/u/0.json") is None and r2.get("raw/v/0.json") == b"y"


def test_schema_applies_statement_by_statement():
    fake = FakeD1()
    d1 = D1Client("acct", "db", TOKEN, transport=fake.transport(), sleep=lambda s: None)
    from evade.storage.cloudflare import SCHEMA_PATH
    assert d1.execute_script(SCHEMA_PATH.read_text(encoding="utf-8")) > 10      # idempotent on an existing schema
    assert all("batch" not in r for r in fake.requests)


# ---------------- the API wired to the Cloudflare backend ----------------
@pytest.fixture()
def api_cf(monkeypatch):
    pytest.importorskip("fastapi")
    os.environ.setdefault("EVADE_API_TOKEN", "test-token")
    from fastapi.testclient import TestClient
    from evade.api import app as A
    from evade.baseline.store import BaselineStore
    st, fake = cf_backend()
    monkeypatch.setattr(A.STATE, "storage", st)
    monkeypatch.setattr(A.STATE, "baselines", BaselineStore(st))
    h = {"Authorization": f"Bearer {os.environ['EVADE_API_TOKEN']}"}
    return TestClient(A.app), h, A, st, fake


def _forget(A, uid):
    """Simulate a restart: this process forgets the user; storage keeps it."""
    for d in (A.STATE.users, A.STATE.latest, A.STATE.radiation, A.STATE.overrides, A.STATE.risk_history, A.STATE.eeg_feats):
        d.pop(uid, None)
    for k in [k for k, v in A.STATE.devices.items() if v["user_id"] == uid]:
        del A.STATE.devices[k]
    from evade.baseline.store import BaselineStore
    A.STATE.baselines = BaselineStore(A.STATE.storage)


def test_api_persists_and_rehydrates_then_deletes(api_cf):
    from datetime import datetime, timezone
    C, H, A, st, fake = api_cf
    now = datetime.now(timezone.utc).isoformat()
    uid = C.post("/v1/users", json={"consent": True}, headers=H).json()["data"]["user_id"]
    rd = [{"metric": "hr_bpm", "value": 60.0 + i % 3, "unit": "bpm", "t_utc": now, "modality": "vitals"} for i in range(12)]
    assert C.post("/v1/baseline/vitals", json={"user_id": uid, "readings": rd, "mode": "demo"}, headers=H).status_code == 200
    assert C.post("/v1/predict/radiation-risk", json={"user_id": uid, "t_utc": now, "cumulative_dose_msv": 300,
                                                      "dose_equivalent_rate_msv_per_day": 1.8, "spe_dose_msv": 260},
                  headers=H).status_code == 200
    assert C.post("/v1/devices/bind", json={"user_id": uid, "device_id": "cf-dev-1", "kind": "hr_strap"}, headers=H).status_code == 200
    r = C.post("/v1/compare", json={"user_id": uid, "readings": [{**rd[0], "value": 95.0}]}, headers=H).json()["data"]
    assert r["flags"]
    assert fake.count("baselines", uid) == 1 and fake.count("flags", uid) >= 1 and fake.count("inputs", uid) >= 13

    _forget(A, uid)
    b = C.get(f"/v1/users/{uid}/baseline", headers=H).json()["data"]["baselines"]
    assert b and b[0]["metric"] == "hr_bpm" and b[0]["status"] == "provisional"
    assert "cf-dev-1" in A.STATE.devices and "O5" in A.STATE.overrides[uid]
    rr = C.post("/v1/predict/risk", json={"user_id": uid, "logistics_floor_days": 0.5}, headers=H).json()["data"]
    assert "O5" in rr["risk"]["override_ids"] and fake.count("risk_results", uid) == 1

    # another user cannot take the stored binding even after a restart
    _forget(A, uid)
    other = C.post("/v1/users", json={"consent": True}, headers=H).json()["data"]["user_id"]
    r = C.post("/v1/devices/bind", json={"user_id": other, "device_id": "cf-dev-1", "kind": "hr_strap"}, headers=H)
    assert r.status_code == 409

    assert C.delete(f"/v1/users/{uid}", headers=H).status_code == 200
    for t in ("users", "baselines", "user_state", "inputs", "flags", "risk_results", "devices"):
        assert fake.count(t, uid) == 0, t
    assert C.delete(f"/v1/users/{uid}", headers=H).status_code == 404
    assert C.get(f"/v1/users/{uid}/baseline", headers=H).status_code == 404


def test_api_delete_is_retryable_when_storage_fails(api_cf):
    C, H, A, st, fake = api_cf
    uid = C.post("/v1/users", json={"consent": True}, headers=H).json()["data"]["user_id"]
    fake.fail_with = [503] * 4                      # exhausts the 3 retries
    r = C.delete(f"/v1/users/{uid}", headers=H)
    assert r.status_code == 503 and r.json()["error"]["code"] == "deletion_incomplete"
    assert fake.count("users", uid) == 1
    assert C.get(f"/v1/users/{uid}/baseline", headers=H).status_code == 404
    assert C.delete(f"/v1/users/{uid}", headers=H).status_code == 200
    assert fake.count("users", uid) == 0


def test_api_storage_outage_returns_503(api_cf):
    C, H, A, st, fake = api_cf
    fake.fail_with = [503] * 4
    r = C.post("/v1/users", json={"consent": True}, headers=H)
    assert r.status_code == 503 and r.json()["error"]["code"] == "storage_unavailable"
    assert TOKEN not in r.text
