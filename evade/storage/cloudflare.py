"""Cloudflare backend: D1 (structured state, via the REST API) and R2 (blobs, via the S3-compatible API).

D1:  POST https://api.cloudflare.com/client/v4/accounts/{account_id}/d1/database/{database_id}/query
     body {"sql": ..., "params": [...]} or {"batch": [{"sql", "params"}, ...]}. Parameterized SQL only; at most
     100 bound parameters per statement (D1 limit), so multi-row inserts are chunked.
R2:  https://{account_id}.r2.cloudflarestorage.com with boto3 (optional dependency; only needed for blobs).

Retries: transport errors, 429 and 5xx, with exponential backoff; a 200 body with success=false is an error.
Credentials are held in memory only and never logged, put in exception messages, or shown in repr().
"""
from __future__ import annotations

import json
import logging
import math
import time
import uuid
from pathlib import Path
from typing import Any, Callable, Iterable

from .base import Storage, StorageConfigError, StorageError, user_hash

log = logging.getLogger("evade.storage")
D1_URL = "https://api.cloudflare.com/client/v4/accounts/{account}/d1/database/{db}/query"
MAX_PARAMS = 100
USER_TABLES = ("users", "baselines", "baseline_archive", "user_state", "inputs", "flags", "risk_results", "devices")


def _val(v: Any) -> Any:
    if isinstance(v, bool):
        return int(v)
    if isinstance(v, float) and not math.isfinite(v):
        return None
    if isinstance(v, (dict, list, tuple)):
        return _dumps(v)
    return v


def _json_default(o: Any) -> Any:
    try:
        f = float(o)                          # numpy scalars
        return f if math.isfinite(f) else None
    except (TypeError, ValueError):
        return str(o)


def _clean(x: Any) -> Any:
    """Replace NaN/inf anywhere in a JSON document with None (D1 stores strict JSON)."""
    if isinstance(x, float):
        return x if math.isfinite(x) else None
    if isinstance(x, dict):
        return {str(k): _clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_clean(v) for v in x]
    return x


def _dumps(x: Any) -> str:
    return json.dumps(_clean(x), default=_json_default, allow_nan=False)


class D1Error(StorageError):
    pass


class D1Client:
    def __init__(self, account_id: str, database_id: str, api_token: str, *, transport=None, timeout: float = 10.0,
                 retries: int = 3, backoff_s: float = 0.5, sleep: Callable[[float], None] = time.sleep):
        import httpx
        self._httpx = httpx
        self._url = D1_URL.format(account=account_id, db=database_id)
        self.__token = api_token
        self._client = httpx.Client(timeout=httpx.Timeout(timeout), transport=transport)
        self.retries, self.backoff_s, self._sleep = retries, backoff_s, sleep

    def __repr__(self) -> str:
        return "D1Client(<redacted>)"

    def _post(self, body: dict) -> list[dict]:
        httpx = self._httpx
        last = "no attempt"
        for attempt in range(self.retries + 1):
            if attempt:
                self._sleep(min(self.backoff_s * 2 ** (attempt - 1), 8.0))
            try:
                r = self._client.post(self._url, json=body, headers={"Authorization": f"Bearer {self.__token}"})
            except httpx.TransportError as e:
                last = f"transport error: {type(e).__name__}"
                log.warning("D1 request failed (%s), attempt %d", last, attempt + 1)
                continue
            if r.status_code == 429 or r.status_code >= 500:
                last = f"HTTP {r.status_code}"
                log.warning("D1 request failed (%s), attempt %d", last, attempt + 1)
                continue
            try:
                data = r.json()
            except ValueError:
                raise D1Error(f"D1 returned HTTP {r.status_code} with a non-JSON body") from None
            if r.status_code >= 400 or not data.get("success", False):
                errs = "; ".join(f"{e.get('code')}: {e.get('message')}" for e in data.get("errors", []) or [])
                raise D1Error(f"D1 query failed (HTTP {r.status_code}): {errs or 'no error detail'}")
            res = data.get("result") or []
            for item in res:
                if isinstance(item, dict) and item.get("success") is False:
                    raise D1Error("D1 statement failed")
            return res
        raise D1Error(f"D1 unavailable after {self.retries + 1} attempts ({last})")

    def query(self, sql: str, params: Iterable[Any] = ()) -> list[dict]:
        res = self._post({"sql": sql, "params": [_val(p) for p in params]})
        return (res[0].get("results") or []) if res else []

    def batch(self, stmts: list[tuple[str, Iterable[Any]]]) -> list[list[dict]]:
        if not stmts:
            return []
        res = self._post({"batch": [{"sql": s, "params": [_val(p) for p in ps]} for s, ps in stmts]})
        return [(x.get("results") or []) for x in res]

    def execute_script(self, sql_text: str) -> int:
        """Apply a .sql file one statement at a time (comments stripped). Returns the number of statements."""
        body = "\n".join(line for line in sql_text.splitlines() if not line.strip().startswith("--"))
        stmts = [s.strip() for s in body.split(";") if s.strip()]
        for s in stmts:
            self.query(s)
        return len(stmts)


class R2Blobs:
    def __init__(self, bucket: str, client: Any):
        self.bucket, self._s3 = bucket, client

    def __repr__(self) -> str:
        return f"R2Blobs(bucket={self.bucket!r})"

    @classmethod
    def from_credentials(cls, account_id: str, bucket: str, access_key_id: str, secret_access_key: str,
                         endpoint: str | None = None) -> "R2Blobs":
        try:
            import boto3
            from botocore.config import Config
        except ImportError:
            raise StorageConfigError("boto3 is required for R2 blobs: pip install boto3") from None
        s3 = boto3.client("s3", endpoint_url=endpoint or f"https://{account_id}.r2.cloudflarestorage.com",
                          aws_access_key_id=access_key_id, aws_secret_access_key=secret_access_key, region_name="auto",
                          config=Config(signature_version="s3v4", connect_timeout=5, read_timeout=30,
                                        retries={"max_attempts": 4, "mode": "standard"}))
        return cls(bucket, s3)

    def _call(self, what: str, fn, **kw):
        try:
            return fn(Bucket=self.bucket, **kw)
        except Exception as e:                   # noqa: BLE001 - botocore errors carry no secrets, but keep it short
            code = getattr(e, "response", {}).get("Error", {}).get("Code") if hasattr(e, "response") else None
            if code in ("NoSuchKey", "404"):
                raise KeyError(kw.get("Key")) from None
            raise StorageError(f"R2 {what} failed: {code or type(e).__name__}") from None

    def put(self, key: str, data: bytes, content_type: str) -> None:
        self._call("put", self._s3.put_object, Key=key, Body=data, ContentType=content_type)

    def get(self, key: str) -> bytes | None:
        try:
            r = self._call("get", self._s3.get_object, Key=key)
        except KeyError:
            return None
        return r["Body"].read()

    def list(self, prefix: str) -> list[str]:
        keys, token = [], None
        while True:
            kw = {"Prefix": prefix, **({"ContinuationToken": token} if token else {})}
            r = self._call("list", self._s3.list_objects_v2, **kw)
            keys += [o["Key"] for o in r.get("Contents", []) or []]
            if not r.get("IsTruncated"):
                return keys
            token = r.get("NextContinuationToken")

    def delete_prefix(self, prefix: str) -> int:
        keys = self.list(prefix)
        for i in range(0, len(keys), 1000):
            chunk = keys[i:i + 1000]
            r = self._call("delete", self._s3.delete_objects,
                           Delete={"Objects": [{"Key": k} for k in chunk], "Quiet": True})
            if r.get("Errors"):
                raise StorageError(f"R2 delete failed for {len(r['Errors'])} objects")
        return len(keys)


class CloudflareBackend(Storage):
    name = "cloudflare"
    persistent = True

    def __init__(self, d1: D1Client, blobs: R2Blobs | None = None, store_raw_windows: bool = False):
        super().__init__(store_raw_windows)
        self.d1, self.blobs = d1, blobs

    def __repr__(self) -> str:
        return f"CloudflareBackend(blobs={'yes' if self.blobs else 'no'})"

    def has_blobs(self) -> bool:
        return self.blobs is not None

    def _insert_many(self, sql_head: str, cols: int, rows: list[tuple], tail: str = "") -> None:
        per = max(1, MAX_PARAMS // cols)
        ph = "(" + ",".join("?" * cols) + ")"
        stmts = []
        for i in range(0, len(rows), per):
            chunk = rows[i:i + per]
            stmts.append((f"{sql_head} VALUES {','.join([ph] * len(chunk))} {tail}".strip(),
                          [v for r in chunk for v in r]))
        for j in range(0, len(stmts), 20):           # several statements per HTTP call
            self.d1.batch(stmts[j:j + 20])

    # ---- deletion: tombstone first, then every table ----
    def _delete_user(self, user_id):
        stmts = [("INSERT OR IGNORE INTO tombstones (user_hash, deleted_at) VALUES (?, ?)", [user_hash(user_id), time.time()])]
        stmts += [(f"DELETE FROM {t} WHERE user_id = ?", [user_id]) for t in USER_TABLES]
        self.d1.batch(stmts)
        return {t: "deleted" for t in USER_TABLES}

    def is_tombstoned(self, user_id: str) -> bool:
        return bool(self.d1.query("SELECT 1 AS x FROM tombstones WHERE user_hash = ?", [user_hash(user_id)]))

    # ---- users ----
    def _put_user(self, user_id, attrs):
        a = dict(attrs)
        created = float(a.pop("created_at", time.time()))
        self.d1.query("INSERT INTO users (user_id, created_at, attrs) SELECT ?, ?, ? "
                      "WHERE NOT EXISTS (SELECT 1 FROM tombstones WHERE user_hash = ?) "
                      "ON CONFLICT (user_id) DO UPDATE SET attrs = excluded.attrs",
                      [user_id, created, _dumps(a), user_hash(user_id)])

    def _get_user(self, user_id):
        rows = self.d1.query("SELECT created_at, attrs FROM users WHERE user_id = ?", [user_id])
        if not rows:
            return None
        return {"created_at": rows[0]["created_at"], **json.loads(rows[0]["attrs"] or "{}")}

    # ---- baselines ----
    def _put_baselines(self, records):
        now = time.time()
        rows = [(r["user_id"], r["modality"], r["metric"], r["baseline_id"], r.get("mode"), r.get("phase"), r["status"],
                 _dumps(r.get("values", [])), r.get("center"), r.get("spread"), r.get("created"), now) for r in records]
        self._insert_many("INSERT INTO baselines (user_id, modality, metric, baseline_id, mode, phase, status, vals, "
                          "center, spread, created_at, updated_at)", 12, rows,
                          "ON CONFLICT (user_id, modality, metric) DO UPDATE SET baseline_id = excluded.baseline_id, "
                          "mode = excluded.mode, phase = excluded.phase, status = excluded.status, vals = excluded.vals, "
                          "center = excluded.center, spread = excluded.spread, updated_at = excluded.updated_at")

    def _get_baselines(self, user_id):
        rows = self.d1.query("SELECT modality, metric, baseline_id, mode, phase, status, vals, center, spread, created_at "
                             "FROM baselines WHERE user_id = ?", [user_id])
        return [{"user_id": user_id, "modality": r["modality"], "metric": r["metric"], "baseline_id": r["baseline_id"],
                 "mode": r["mode"], "phase": r["phase"], "status": r["status"], "values": json.loads(r["vals"] or "[]"),
                 "center": r["center"], "spread": r["spread"], "created": r["created_at"]} for r in rows]

    def _archive_baselines(self, user_id, modality):
        self.d1.batch([
            ("INSERT OR IGNORE INTO baseline_archive (baseline_id, user_id, modality, metric, mode, phase, vals, center, "
             "spread, created_at, archived_at) SELECT baseline_id, user_id, modality, metric, mode, phase, vals, center, "
             "spread, created_at, ? FROM baselines WHERE user_id = ? AND modality = ?", [time.time(), user_id, modality]),
            ("DELETE FROM baselines WHERE user_id = ? AND modality = ?", [user_id, modality])])

    # ---- state ----
    def _put_state(self, user_id, key, value):
        self.d1.query("INSERT INTO user_state (user_id, key, value, updated_at) VALUES (?, ?, ?, ?) "
                      "ON CONFLICT (user_id, key) DO UPDATE SET value = excluded.value, updated_at = excluded.updated_at",
                      [user_id, key, _dumps(value), time.time()])

    def _get_state(self, user_id):
        rows = self.d1.query("SELECT key, value FROM user_state WHERE user_id = ?", [user_id])
        return {r["key"]: json.loads(r["value"]) for r in rows}

    # ---- inputs, flags, risk ----
    def _add_inputs(self, user_id, rows):
        now = time.time()
        self._insert_many("INSERT OR IGNORE INTO inputs (input_id, user_id, kind, t_utc, meta, blob_key, created_at)", 7,
                          [(uuid.uuid4().hex, user_id, r["kind"], r.get("t_utc"), _dumps(r.get("meta", {})),
                            r.get("blob_key"), now) for r in rows])

    def _list_inputs(self, user_id):
        rows = self.d1.query("SELECT input_id, kind, t_utc, meta, blob_key FROM inputs WHERE user_id = ? "
                             "ORDER BY created_at", [user_id])
        return [{**r, "meta": json.loads(r["meta"] or "{}")} for r in rows]

    def _add_flags(self, user_id, flags):
        self._insert_many("INSERT OR IGNORE INTO flags (flag_id, user_id, t, domain, metric, modality, level, z, "
                          "component_score, confidence)", 10,
                          [(uuid.uuid4().hex, user_id, f.get("t", time.time()), f.get("domain"), f.get("metric"),
                            f.get("modality"), f.get("level"), f.get("z"), f.get("component_score"), f.get("confidence"))
                           for f in flags])

    def _list_flags(self, user_id):
        return self.d1.query("SELECT t, domain, metric, modality, level, z, component_score, confidence FROM flags "
                             "WHERE user_id = ? ORDER BY t", [user_id])

    def _add_risk(self, user_id, t, r_mdtr, band, summary):
        self.d1.query("INSERT OR IGNORE INTO risk_results (risk_id, user_id, t, r_mdtr, band, summary) "
                      "VALUES (?, ?, ?, ?, ?, ?)", [uuid.uuid4().hex, user_id, t, r_mdtr, band, _dumps(summary)])

    def _risk_history(self, user_id):
        rows = self.d1.query("SELECT t, r_mdtr FROM risk_results WHERE user_id = ? ORDER BY t", [user_id])
        return [(r["t"], r["r_mdtr"]) for r in rows]

    # ---- devices ----
    def _put_device(self, device_id, rec):
        self.d1.query("INSERT INTO devices (device_id, user_id, rec, updated_at) VALUES (?, ?, ?, ?) "
                      "ON CONFLICT (device_id) DO UPDATE SET user_id = excluded.user_id, rec = excluded.rec, "
                      "updated_at = excluded.updated_at", [device_id, rec["user_id"], _dumps(rec), time.time()])

    def _get_device(self, device_id):
        rows = self.d1.query("SELECT rec FROM devices WHERE device_id = ?", [device_id])
        return json.loads(rows[0]["rec"]) if rows else None

    def _delete_device(self, device_id):
        self.d1.query("DELETE FROM devices WHERE device_id = ?", [device_id])

    def _devices_for_user(self, user_id):
        rows = self.d1.query("SELECT device_id, rec FROM devices WHERE user_id = ?", [user_id])
        return {r["device_id"]: json.loads(r["rec"]) for r in rows}

    # ---- blobs ----
    def _need_blobs(self) -> R2Blobs:
        if self.blobs is None:
            raise StorageConfigError("R2 is not configured (CF_R2_BUCKET, CF_R2_ACCESS_KEY_ID, CF_R2_SECRET_ACCESS_KEY)")
        return self.blobs

    def _put_blob(self, key, data, content_type):
        self._need_blobs().put(key, data, content_type)

    def _get_blob(self, key):
        return self._need_blobs().get(key)

    def _list_blobs(self, prefix):
        return self._need_blobs().list(prefix)

    def _delete_prefix(self, prefix):
        return self._need_blobs().delete_prefix(prefix)


SCHEMA_PATH = Path(__file__).with_name("schema.sql")
