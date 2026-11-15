"""The user's security rules: a key on every request (constant time, from the environment), no CORS, no docs by default,
HTTPS enforcement option, no health data or IDs in logs, bounded inputs and one error shape."""
import json
import logging
import os
import subprocess
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")
from fastapi.testclient import TestClient  # noqa: E402

from evade.api import app as A  # noqa: E402
from tests.conftest import KEY_HEADER, TEST_API_KEY  # noqa: E402
from tests.test_review_fixes import api, new_user  # noqa: E402,F401

ROOT = Path(__file__).resolve().parents[1]
C = TestClient(A.app)
H = KEY_HEADER
NOW = datetime.now(timezone.utc).isoformat()
SHAPE = {"code", "message", "field"}


def is_error(r, status, code=None):
    assert r.status_code == status, r.text
    body = r.json()
    assert set(body) == {"error"} and set(body["error"]) == SHAPE, body
    if code:
        assert body["error"]["code"] == code, body
    return body["error"]


# ---------------- auth ----------------
@pytest.mark.parametrize("method,path", [("GET", "/v1/health"), ("GET", "/v1/models"), ("POST", "/v1/users"),
                                         ("DELETE", f"/v1/users/{uuid.uuid4()}"), ("GET", "/no/such/path"),
                                         ("GET", "/docs"), ("GET", "/openapi.json"), ("OPTIONS", "/v1/users")])
def test_every_path_needs_the_key(method, path):
    for headers in ({}, {"X-EVAde-Key": "wrong"}, {"X-EVAde-Key": TEST_API_KEY[:-1]},
                    {"X-EVAde-Key": TEST_API_KEY + "x"}, {"Authorization": f"Bearer {TEST_API_KEY}"}):
        is_error(C.request(method, path, headers=headers), 401, "unauthenticated")


def test_right_key_works_and_header_name_is_case_insensitive():
    assert C.get("/v1/health", headers=H).status_code == 200
    assert C.get("/v1/health", headers={"x-evade-key": TEST_API_KEY}).status_code == 200


def test_a_new_route_cannot_skip_the_key():
    @A.app.get("/v1/added-later")
    def added():
        return {"ok": True}
    try:
        is_error(C.get("/v1/added-later"), 401)
        assert C.get("/v1/added-later", headers=H).json() == {"ok": True}
    finally:
        A.app.router.routes[:] = [r for r in A.app.router.routes if getattr(r, "path", "") != "/v1/added-later"]


def test_key_check_is_constant_time_on_bytes(monkeypatch):
    calls = []
    real = A.hmac.compare_digest
    monkeypatch.setattr(A.hmac, "compare_digest", lambda a, b: calls.append((type(a), type(b))) or real(a, b))
    C.get("/v1/health", headers=H)
    assert calls == [(bytes, bytes)]


def test_startup_refuses_missing_or_short_key(monkeypatch):
    monkeypatch.delenv("EVADE_API_KEY")
    with pytest.raises(RuntimeError, match="EVADE_API_KEY is not set"):
        A.require_api_key()
    monkeypatch.setenv("EVADE_API_KEY", "s3cret-but-short")
    with pytest.raises(RuntimeError, match="at least 32") as ei:
        A.require_api_key()
    assert "s3cret" not in str(ei.value)
    is_error(C.get("/v1/health", headers={"X-EVAde-Key": "s3cret-but-short"}), 401)    # never accepted


def _run(code: str, **env):
    e = {k: v for k, v in os.environ.items() if not k.startswith("EVADE_")}
    e.update(PYTHONDONTWRITEBYTECODE="1", **env)
    return subprocess.run([sys.executable, "-c", code], cwd=ROOT, env=e, capture_output=True, text=True, timeout=180)


def test_process_does_not_start_without_a_valid_key():
    for env in ({}, {"EVADE_API_KEY": "x" * 31}):
        r = _run("import evade.api.app", **env)
        assert r.returncode != 0 and "refusing to start" in r.stderr
    r = _run("import evade.api.app", EVADE_API_KEY="k" * 32)
    assert r.returncode == 0, r.stderr


def test_docs_off_by_default_and_behind_the_key_when_on():
    for p in ("/docs", "/redoc", "/openapi.json"):
        is_error(C.get(p, headers=H), 404, "not_found")
    code = ("from fastapi.testclient import TestClient; from evade.api.app import app; c = TestClient(app); "
            "k = {'X-EVAde-Key': 'k' * 40}; "
            "print(c.get('/openapi.json').status_code, c.get('/openapi.json', headers=k).status_code, "
            "c.get('/docs').status_code, c.get('/docs', headers=k).status_code)")
    r = _run(code, EVADE_API_KEY="k" * 40, EVADE_ENABLE_DOCS="1")
    assert r.returncode == 0, r.stderr
    assert r.stdout.split() == ["401", "200", "401", "200"]


# ---------------- CORS ----------------
def test_no_cors():
    assert not [m for m in A.app.user_middleware if "CORS" in getattr(m.cls, "__name__", "")]
    origin = {"Origin": "https://attacker.example"}
    r = C.options("/v1/users", headers={**origin, "Access-Control-Request-Method": "POST"})
    assert r.status_code == 401 and "access-control-allow-origin" not in r.headers
    r = C.get("/v1/health", headers={**H, **origin})
    assert r.status_code == 200 and not [h for h in r.headers if h.lower().startswith("access-control")]
    assert "CORS" not in (ROOT / "evade/api/app.py").read_text(encoding="utf-8").split('"""', 2)[2]


# ---------------- HTTPS ----------------
def test_require_https_rejects_forwarded_http(monkeypatch):
    assert C.get("/v1/health", headers={**H, "X-Forwarded-Proto": "http"}).status_code == 200    # off by default
    monkeypatch.setenv("EVADE_REQUIRE_HTTPS", "1")
    is_error(C.get("/v1/health", headers={**H, "X-Forwarded-Proto": "http"}), 403, "https_required")
    is_error(C.get("/v1/health", headers={"X-Forwarded-Proto": "http"}), 403, "https_required")
    assert C.get("/v1/health", headers={**H, "X-Forwarded-Proto": "https"}).status_code == 200


def test_dockerfile_disables_access_log():
    cmd = [ln for ln in (ROOT / "Dockerfile").read_text(encoding="utf-8").splitlines() if ln.startswith("CMD")][0]
    assert "--no-access-log" in cmd


# ---------------- error shape and validation ----------------
def test_one_error_shape_everywhere():
    uid = new_user(C, H)
    is_error(C.get("/v1/health"), 401, "unauthenticated")
    is_error(C.get("/v1/nothing-here", headers=H), 404, "not_found")
    is_error(C.put("/v1/health", headers=H, content=b""), 405, "method_not_allowed")
    is_error(C.post("/v1/users", headers={**H, "Content-Type": "application/json"}, content=iter([b"{}"])), 411)
    is_error(C.post("/v1/users", headers={**H, "Content-Type": "application/json", "Content-Length": "2000000"},
                    content=b"{}"), 413, "payload_too_large")
    is_error(C.post("/v1/users", headers={**H, "Content-Type": "application/json"}, content=b"{not json"), 422)
    e = is_error(C.post("/v1/users", json={"consent": False}, headers=H), 422, "validation_error")
    assert e["field"] == "consent"
    is_error(C.get(f"/v1/users/{uuid.uuid4()}/baseline", headers=H), 404, "user_not_found")
    is_error(C.get(f"/v1/devices/no-such-device?user_id={uid}", headers=H), 404, "device_not_found")


def test_storage_error_has_the_shape(api):
    Cs, Hs, A2, st, fake, s3 = api
    fake.fail_with = [503] * 3
    is_error(Cs.post("/v1/users", json={"consent": True}, headers=Hs), 503, "storage_unavailable")


@pytest.mark.parametrize("path,body,field", [
    ("/v1/compare", {"user_id": "not-a-uuid"}, "user_id"),
    ("/v1/compare", {"user_id": str(uuid.uuid4()).upper()}, "user_id"),
    ("/v1/devices/bind", {"device_id": "bad id with spaces", "kind": "hr_strap"}, "device_id"),
    ("/v1/devices/bind", {"device_id": "x" * 65, "kind": "hr_strap"}, "device_id"),
    ("/v1/devices/bind", {"device_id": "dev-1", "kind": "hr_strap", "vendor": "v" * 41}, "vendor"),
    ("/v1/compare", {"readings": [{"metric": "HR<script>", "value": 1, "unit": "bpm", "t_utc": NOW}]},
     "readings.0.metric"),
    ("/v1/compare", {"readings": [{"metric": "hr_bpm", "value": 1, "unit": "b" * 17, "t_utc": NOW}]},
     "readings.0.unit"),
    ("/v1/compare", {"readings": [{"metric": "hr_bpm", "value": 1e9, "unit": "bpm", "t_utc": NOW}]},
     "readings.0.value"),
    ("/v1/compare", {"readings": [{"metric": "hr_bpm", "value": 1, "unit": "bpm", "t_utc": NOW}] * 501}, "readings"),
    ("/v1/predict/radiation-risk", {"t_utc": NOW, "cumulative_dose_msv": 1, "dose_equivalent_rate_msv_per_day": 1,
                                    "seu_count_last_hour": 10**9}, "seu_count_last_hour"),
    ("/v1/input/instrument", {"instrument": "gad7", "t_utc": NOW, "answers": [0, 0, 0, 0, 0, 0, 7]}, "answers.6"),
    ("/v1/input/instrument", {"instrument": "gad7", "t_utc": NOW, "answers": [0] * 50}, "answers"),
    ("/v1/predict/ecg", {"window": {"modality": "ecg", "t0_utc": NOW, "fs_hz": 100, "channels": ["c" * 17],
                                    "samples": [[0.0] * 300]}}, "window.channels.0"),
    ("/v1/ingest/batch", {"device_id": "dev-1", "seq": 2**60}, "seq"),
    ("/v1/users", {"consent": True, "cohort": "c" * 33}, "cohort"),
])
def test_inputs_are_bounded(path, body, field):
    uid = str(uuid.uuid4())
    if path != "/v1/users":
        body = {"user_id": uid, **body}
    e = is_error(C.post(path, json=body, headers=H), 422)
    assert e["field"] == field, e


def test_nan_and_infinity_are_rejected():
    uid = new_user(C, H)
    h = {**H, "Content-Type": "application/json"}
    for path, raw in (("/v1/predict/risk", '{"user_id": "%s", "domain_inputs": {"cardiac": {"s": NaN}}}'),
                      ("/v1/compare", '{"user_id": "%s", "readings": [{"metric": "hr_bpm", "value": Infinity, '
                                      '"unit": "bpm", "t_utc": "2026-01-01T00:00:00Z"}]}'),
                      ("/v1/compare", '{"user_id": "%s", "readings": [{"metric": "hr_bpm", "value": 60, '
                                      '"unit": "bpm", "t_utc": "2026-01-01T00:00:00Z", "quality": NaN}]}')):
        is_error(C.post(path, content=(raw % uid).encode(), headers=h), 422)


def test_websocket_needs_the_key():
    from starlette.websockets import WebSocket, WebSocketDisconnect

    @A.app.websocket("/v1/ws-test")
    async def ws(websocket: WebSocket):
        await websocket.accept()
        await websocket.send_text("hello")
        await websocket.close()
    try:
        with pytest.raises(WebSocketDisconnect) as ei:
            with C.websocket_connect("/v1/ws-test") as w:
                w.receive_text()
        assert ei.value.code == 1008
        with C.websocket_connect("/v1/ws-test", headers=H) as w:
            assert w.receive_text() == "hello"
    finally:
        A.app.router.routes[:] = [r for r in A.app.router.routes if getattr(r, "path", "") != "/v1/ws-test"]


def test_path_and_query_ids_are_validated():
    is_error(C.get("/v1/users/../../etc/baseline", headers=H), 404)
    assert is_error(C.get("/v1/users/not-a-uuid/baseline", headers=H), 422)["field"] == "user_id"
    assert is_error(C.delete("/v1/users/1234", headers=H), 422)["field"] == "user_id"
    assert is_error(C.get("/v1/actions?user_id=x", headers=H), 422)["field"] == "user_id"
    assert is_error(C.get(f"/v1/devices/a%20b?user_id={uuid.uuid4()}", headers=H), 422)["field"] == "device_id"


def test_validation_errors_do_not_echo_values():
    secret = "SECRET-VALUE-7731"
    for body in ({"consent": True, "cohort": secret * 5},
                 {"consent": secret},
                 {"consent": True, "age_band": secret}):
        r = C.post("/v1/users", json=body, headers=H)
        assert r.status_code == 422 and secret not in r.text
    r = C.post("/v1/predict/risk", json={"user_id": str(uuid.uuid4()), "domain_inputs": {secret: {"s": 0.5}}},
               headers=H)
    assert r.status_code == 422 and secret not in r.text


# ---------------- logs ----------------
MARK_VALUE = 87.654321
MARK_COHORT = "cohortmarker"
MARK_DEVICE = "devmarker-991"


def _all_log_text(records) -> str:
    out = []
    for rec in records:
        msg = rec.getMessage()
        if "testserver" in msg:                     # the test client's own request log, not the service's
            continue
        out.append(f"{rec.name} {msg} {rec.args!r} {rec.exc_text or ''}")
    return "\n".join(out)


def test_logs_hold_no_health_data_ids_or_header_values(api, caplog):
    caplog.set_level(logging.DEBUG)
    Cs, Hs, A2, st, fake, s3 = api
    for name in ("botocore", "httpx", "httpcore"):
        assert logging.getLogger(name).level >= logging.WARNING
    uid = Cs.post("/v1/users", json={"consent": True, "cohort": MARK_COHORT}, headers=Hs).json()["data"]["user_id"]
    reading = {"metric": "hr_bpm", "value": MARK_VALUE, "unit": "bpm", "t_utc": NOW, "modality": "vitals"}
    ok = [
        ("/v1/baseline/vitals", {"user_id": uid, "readings": [reading] * 12, "mode": "demo"}),
        ("/v1/compare", {"user_id": uid, "readings": [{**reading, "value": 143.21}]}),
        ("/v1/input/self-report", {"user_id": uid, "t_utc": NOW, "sleep_quality": 1, "stress": 5, "fatigue": 5,
                                   "mood": 1}),
        ("/v1/input/instrument", {"user_id": uid, "instrument": "phq9", "t_utc": NOW,
                                  "answers": [3, 3, 3, 3, 3, 3, 3, 3, 3]}),
        ("/v1/devices/bind", {"user_id": uid, "device_id": MARK_DEVICE, "kind": "hr_strap"}),
        ("/v1/predict/risk", {"user_id": uid, "domain_inputs": {"behavioral": {"s": 0.77}}}),
    ]
    for path, body in ok:
        assert Cs.post(path, json=body, headers=Hs).status_code == 200, path
    # validation errors
    Cs.post("/v1/compare", json={"user_id": uid, "readings": [{**reading, "unit": "u" * 40}]}, headers=Hs)
    Cs.post("/v1/users", json={"consent": True, "cohort": MARK_COHORT * 9}, headers=Hs)
    Cs.post("/v1/input/self-report", json={"user_id": uid, "t_utc": NOW, "sleep_quality": 9, "stress": 5,
                                           "fatigue": 5, "mood": 1}, headers=Hs)
    # storage errors: best-effort warnings, a 503, a failed deletion
    fake.fail_with = [503] * 3
    Cs.post("/v1/compare", json={"user_id": uid, "readings": [reading]}, headers=Hs)
    fake.fail_with = [503] * 3
    assert Cs.post(f"/v1/devices/bind", json={"user_id": uid, "device_id": "devmarker-992", "kind": "hr_strap"},
                   headers=Hs).status_code == 503
    fake.fail_with = [503] * 3
    assert Cs.delete(f"/v1/users/{uid}", headers=Hs).status_code == 503
    assert Cs.delete(f"/v1/users/{uid}", headers=Hs).status_code == 200
    Cs.get("/v1/models", headers={"X-EVAde-Key": "wrong-key-value-123"})
    text = _all_log_text(caplog.records)
    assert caplog.records and "storage" in text      # the scenario did log, so the checks below mean something
    for needle in (uid, uid[:8], user_hash_prefix(uid), str(MARK_VALUE), "143.21", MARK_COHORT, MARK_DEVICE,
                   "devmarker-992", "0.77", TEST_API_KEY, "wrong-key-value-123", "[3, 3, 3"):
        assert needle not in text, needle


def user_hash_prefix(uid):
    from evade.storage.base import user_hash
    return user_hash(uid)[:16]


def test_unhandled_error_logs_only_the_type(api, caplog, monkeypatch):
    caplog.set_level(logging.DEBUG)
    Cs, Hs, A2, st, fake, s3 = api
    uid = new_user(Cs, Hs)

    def boom(*a, **k):
        raise ValueError(f"value {MARK_VALUE} for {uid}")

    monkeypatch.setattr(A2, "domain_scores", boom)
    r = Cs.post("/v1/compare", json={"user_id": uid, "readings": []}, headers=Hs)
    is_error(r, 500, "internal_error")
    text = _all_log_text(caplog.records)
    assert "ValueError" in text and uid not in text and str(MARK_VALUE) not in text


def test_transfer_encoding_is_refused_so_the_cap_holds():
    r = C.post("/v1/users", content=b'{"consent": true}',
               headers={**H, "Content-Type": "application/json", "Transfer-Encoding": "chunked", "Content-Length": "17"})
    is_error(r, 411, "length_required")


def test_trailing_slash_is_not_redirected():
    r = C.get("/v1/health/", headers=H, follow_redirects=False)
    assert r.status_code == 404 and "location" not in r.headers


def test_dict_keys_are_masked_in_the_field_path(api):
    client, headers = api[0], api[1]
    uid = new_user(client, headers)
    r = client.post("/v1/predict/risk", headers=headers,
                    json={"user_id": uid, "domain_inputs": {"SECRET_VALUE_7731": {"s": 0.5}}})
    assert r.status_code == 422 and "SECRET_VALUE_7731" not in r.text


@pytest.mark.parametrize("key", ["K" * 32 + " ", " " + "K" * 32, "é" * 32, "K" * 16 + "	" + "K" * 16])
def test_unsendable_keys_are_refused(key):
    assert A.api_key_problem(key) is not None


def test_dockerfile_trusts_the_proxy_scheme():
    text = (ROOT / "Dockerfile").read_text()
    assert "--forwarded-allow-ips" in text
