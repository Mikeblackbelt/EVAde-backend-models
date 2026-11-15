"""User contract for the team's Worker: POST /v1/users with an optional existing user_id (create, idempotent repeat,
409 for a deleted ID) and DELETE /v1/users/{id} (200 once, 404 after; an incomplete deletion is retried with 200).
Memory backend and the Cloudflare backend against the fakes (tests/fakes.py)."""
import uuid

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from tests.conftest import KEY_HEADER as H  # noqa: E402
from tests.test_api import C  # noqa: E402
from tests.test_storage import _forget, api_cf, cf_backend  # noqa: E402,F401  (api_cf is a fixture)


def new_id() -> str:
    return str(uuid.uuid4())


def create(client, uid=None, **attrs):
    body = {"consent": True, **attrs}
    if uid is not None:
        body["user_id"] = uid
    return client.post("/v1/users", json=body, headers=H)


# ---------------- memory backend ----------------
def test_without_user_id_the_server_generates_one():
    r = create(C)
    d = r.json()["data"]
    assert r.status_code == 200 and d["created"] is True
    assert uuid.UUID(d["user_id"]).version == 4 and r.json()["user_id"] == d["user_id"]
    assert create(C).json()["data"]["user_id"] != d["user_id"]


def test_register_under_an_existing_id_is_idempotent():
    uid = new_id()
    r1 = create(C, uid, cohort="crew-a", age_band="30-39")
    assert r1.status_code == 200, r1.text
    d1 = r1.json()["data"]
    assert d1["user_id"] == uid and r1.json()["user_id"] == uid and d1["created"] is True
    r2 = create(C, uid, cohort="other")                       # repeat: same user, no error, attributes unchanged
    d2 = r2.json()["data"]
    assert r2.status_code == 200 and d2 == {**d1, "created": False}
    from evade.api import app as A
    assert A.STATE.users[uid]["cohort"] == "crew-a"
    assert C.get(f"/v1/users/{uid}/baseline", headers=H).status_code == 200   # usable like a generated ID


@pytest.mark.parametrize("bad", ["not-a-uuid", str(uuid.uuid4()).upper(), "{" + str(uuid.uuid4()) + "}",
                                 str(uuid.uuid4()).replace("-", ""), ""])
def test_user_id_must_be_a_lowercase_uuid(bad):
    r = create(C, bad)
    assert r.status_code == 422 and r.json()["error"]["field"] == "user_id"
    assert bad == "" or bad not in r.text                      # the submitted value is not echoed


def test_consent_is_still_required_with_a_user_id():
    r = C.post("/v1/users", json={"consent": False, "user_id": new_id()}, headers=H)
    assert r.status_code == 422


def test_delete_contract_and_deleted_id_is_never_recreated():
    uid = new_id()
    assert create(C, uid).status_code == 200
    r = C.delete(f"/v1/users/{uid}", headers=H)
    assert r.status_code == 200
    d = r.json()["data"]
    assert d["deleted"] is True and "storage" in d and "blobs_removed" in d
    r = C.delete(f"/v1/users/{uid}", headers=H)               # already deleted: 404, the Worker treats it as done
    assert r.status_code == 404 and r.json()["error"]["code"] == "user_not_found"
    r = create(C, uid)                                         # a deleted user is never brought back
    assert r.status_code == 409 and r.json()["error"]["code"] == "user_deleted"
    assert C.get(f"/v1/users/{uid}/baseline", headers=H).status_code == 404
    r = C.delete(f"/v1/users/{new_id()}", headers=H)           # never existed: 404 too
    assert r.status_code == 404 and r.json()["error"]["code"] == "user_not_found"
    assert C.delete("/v1/users/not-a-uuid", headers=H).status_code == 422


def test_user_calls_need_the_key():
    assert C.post("/v1/users", json={"consent": True, "user_id": new_id()}).status_code == 401
    assert C.delete(f"/v1/users/{new_id()}").status_code == 401


# ---------------- Cloudflare backend (fake D1 + fake S3) ----------------
def test_storage_backed_register_persists_and_survives_restart(api_cf, monkeypatch):
    from evade.baseline.store import BaselineStore
    client, _, A, st, fake = api_cf
    uid = new_id()
    r = create(client, uid, cohort="crew-b")
    assert r.status_code == 200 and r.json()["data"]["created"] is True
    assert fake.count("users", uid) == 1
    created_at = r.json()["data"]["created_at"]

    _forget(A, uid)                                            # restart: this process forgets, D1 keeps the row
    r = create(client, uid, cohort="changed")
    assert r.status_code == 200 and r.json()["data"] == {"user_id": uid, "created": False, "created_at": created_at}
    assert fake.count("users", uid) == 1
    assert A.STATE.users[uid]["cohort"] == "crew-b"           # loaded from storage, not overwritten

    assert client.delete(f"/v1/users/{uid}", headers=H).status_code == 200
    assert fake.count("users", uid) == 0
    assert client.delete(f"/v1/users/{uid}", headers=H).status_code == 404

    st2, _ = cf_backend(fake)                                  # new process: only the stored tombstone remains
    monkeypatch.setattr(A.STATE, "storage", st2)
    monkeypatch.setattr(A.STATE, "baselines", BaselineStore(st2))
    _forget(A, uid)
    r = create(client, uid)
    assert r.status_code == 409 and r.json()["error"]["code"] == "user_deleted"
    assert fake.count("users", uid) == 0
    assert client.delete(f"/v1/users/{uid}", headers=H).status_code == 404


def test_storage_backed_unknown_id_after_a_cached_miss(api_cf):
    client, _, A, st, fake = api_cf
    uid = new_id()
    assert client.get(f"/v1/users/{uid}/baseline", headers=H).status_code == 404   # caches the miss
    r = create(client, uid)
    assert r.status_code == 200 and r.json()["data"]["created"] is True and fake.count("users", uid) == 1
    assert client.get(f"/v1/users/{uid}/baseline", headers=H).status_code == 200


def test_incomplete_deletion_blocks_reregistration_and_is_retried(api_cf):
    client, _, A, st, fake = api_cf
    uid = new_id()
    assert create(client, uid).status_code == 200
    fake.fail_with = [503] * 4                                 # exhausts the D1 retries part-way through the delete
    r = client.delete(f"/v1/users/{uid}", headers=H)
    assert r.status_code == 503 and r.json()["error"]["code"] == "deletion_incomplete"
    r = create(client, uid)
    assert r.status_code == 409 and r.json()["error"]["code"] == "user_deleted"
    r = client.delete(f"/v1/users/{uid}", headers=H)           # finding 1: the retry finishes it with 200
    assert r.status_code == 200 and r.json()["data"]["deleted"] is True
    assert fake.count("users", uid) == 0
    assert client.delete(f"/v1/users/{uid}", headers=H).status_code == 404
    assert create(client, uid).status_code == 409
