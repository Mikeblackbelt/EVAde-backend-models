"""Cloudflare backend: D1 (structured state, via the REST API) and R2 (blobs, via the S3-compatible API).

D1:  POST https://api.cloudflare.com/client/v4/accounts/{account_id}/d1/database/{database_id}/query
     body {"sql": ..., "params": [...]} or {"batch": [{"sql", "params"}, ...]}. Parameterized SQL only; at most
     100 bound parameters per statement (D1 limit), so multi-row inserts are chunked.
R2:  https://{account_id}.r2.cloudflarestorage.com with boto3 (optional dependency; only needed for blobs).

Retries: transport errors, 429 and 5xx, at most 2 retries with exponential backoff or Retry-After; timeouts connect
3 s, read 5 s; a circuit breaker fails fast for 30 s after 5 consecutive failed attempts, then lets exactly one trial
request through (every other caller still fails fast until the trial succeeds or fails); the per-request deadline
(evade.storage.base.storage_deadline) caps the total. A 200 body with success=false is an error.
R2: boto3 with connect 3 s, read 5 s and at most 2 attempts in total; a botocore before-send hook checks the
per-request deadline before every attempt, so a retry never starts after the deadline (one attempt that has already
started can still overrun it by up to its own timeouts).
Schema: from_env() refuses to start unless schema_migrations is at SCHEMA_VERSION (check_schema), naming the
missing migration.
Tombstones: every write to a user table is an INSERT ... SELECT ... WHERE NOT EXISTS (tombstone), or an UPDATE/DELETE
of rows that a deletion already removed, so a write in flight when a DELETE commits cannot recreate data.
Credentials are held in memory only and never logged, put in exception messages, or shown in repr().
"""
from __future__ import annotations

import contextlib
import json
import logging
import math
import numbers
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator

from .base import (WEEK_S, Storage, StorageConfigError, StorageError, bound_history, deadline_remaining,
                   mark_storage_call, user_hash)

log = logging.getLogger("evade.storage")
D1_URL = "https://api.cloudflare.com/client/v4/accounts/{account}/d1/database/{db}/query"
SCHEMA_VERSION = 2            # highest migration in schema.sql that this code needs
MIGRATIONS = {1: "base tables", 2: "pending_deletions"}
ID_CHUNK = 90                 # ids per IN (...) list, leaving room for the other parameters under MAX_PARAMS
MAX_PARAMS = 100
STMTS_PER_CALL = 20
USER_TABLES = ("users", "baselines", "baseline_archive", "user_state", "inputs", "flags", "risk_results", "devices")
NOT_TOMB = "NOT EXISTS (SELECT 1 FROM tombstones WHERE user_hash = ?)"


def _val(v: Any) -> str | None:
    """D1 REST params are documented as strings: send scalars as text (column affinity turns numeric text back into
    REAL/INTEGER on insert and compares it as a number) and keep NULL as null. Where a parameter is compared with an
    expression that has no affinity (json_extract), the SQL casts it explicitly."""
    if v is None:
        return None
    if isinstance(v, bool):
        return "1" if v else "0"
    if isinstance(v, numbers.Integral):
        return str(int(v))
    if isinstance(v, numbers.Real):
        f = float(v)
        return repr(f) if math.isfinite(f) else None
    if isinstance(v, (dict, list, tuple)):
        return _dumps(v)
    return str(v)


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
    def __init__(self, msg: str, transient: bool = True):
        super().__init__(msg)
        self.transient = transient


def _retry_after(r) -> float | None:
    v = r.headers.get("retry-after")
    try:
        return max(0.0, float(v)) if v is not None else None
    except ValueError:
        return None                               # an HTTP date: fall back to the backoff schedule


class D1Client:
    """REST client with short timeouts, at most `retries` retries (honouring Retry-After), a circuit breaker, and
    the per-request deadline from evade.storage.base.storage_deadline."""

    def __init__(self, account_id: str, database_id: str, api_token: str, *, transport=None, timeout: float = 5.0,
                 connect_timeout: float = 3.0, retries: int = 2, backoff_s: float = 0.5, max_retry_after_s: float = 5.0,
                 breaker_threshold: int = 5, breaker_cooldown_s: float = 30.0,
                 sleep: Callable[[float], None] = time.sleep, clock: Callable[[], float] = time.monotonic):
        import httpx
        self._httpx = httpx
        self._url = D1_URL.format(account=account_id, db=database_id)
        self.__token = api_token
        self.timeout, self.connect_timeout = float(timeout), float(connect_timeout)
        self._client = httpx.Client(timeout=httpx.Timeout(self.timeout, connect=self.connect_timeout), transport=transport)
        self.retries, self.backoff_s, self.max_retry_after_s = retries, backoff_s, max_retry_after_s
        self.breaker_threshold, self.breaker_cooldown_s = breaker_threshold, breaker_cooldown_s
        self._sleep, self._clock = sleep, clock
        self._blk = threading.Lock()
        self._fails = 0
        self._open_until = 0.0
        self._trial_until = 0.0                   # half-open: the single trial request holds the slot until then

    def __repr__(self) -> str:
        return "D1Client(<redacted>)"

    # ---- circuit breaker: after `breaker_threshold` consecutive failed attempts, fail fast for the cooldown, then
    # ---- let one trial request through (half-open); its result closes or reopens the circuit ----
    def _breaker_check(self) -> bool:
        """Raises while open, or while half-open with the trial slot taken. Returns True for the trial request."""
        with self._blk:
            now = self._clock()
            if now < self._open_until:
                raise D1Error("D1 circuit open after repeated failures; failing fast")
            if self._fails >= self.breaker_threshold:
                if now < self._trial_until:
                    raise D1Error("D1 circuit half-open, a trial request is in flight; failing fast")
                self._trial_until = now + self.connect_timeout + self.timeout + 1.0   # freed early in a finally
                return True
            return False

    def _release_trial(self) -> None:
        with self._blk:
            self._trial_until = 0.0

    def _breaker_record(self, ok: bool) -> None:
        with self._blk:
            if ok:
                self._fails, self._open_until = 0, 0.0
                return
            self._fails += 1
            if self._fails >= self.breaker_threshold:
                if self._open_until <= self._clock():
                    log.error("D1 circuit opened for %.0f s after %d consecutive failures",
                              self.breaker_cooldown_s, self._fails)
                self._open_until = self._clock() + self.breaker_cooldown_s

    def _post(self, body: dict) -> list[dict]:
        httpx = self._httpx
        last = "no attempt"
        for attempt in range(self.retries + 1):
            trial = self._breaker_check()
            wait = self.backoff_s * 2 ** attempt
            try:
                rem = deadline_remaining()
                if rem is not None and rem <= 0:
                    raise D1Error(f"storage deadline exceeded ({last})")
                kw = {} if rem is None else {"timeout": httpx.Timeout(min(self.timeout, rem),
                                                                       connect=min(self.connect_timeout, rem))}
                mark_storage_call()
                try:
                    r = self._client.post(self._url, json=body, headers={"Authorization": f"Bearer {self.__token}"},
                                          **kw)
                except httpx.TransportError as e:
                    last = f"transport error: {type(e).__name__}"
                    self._breaker_record(False)
                    log.warning("D1 request failed (%s), attempt %d", last, attempt + 1)
                else:
                    if r.status_code == 429 or r.status_code >= 500:
                        last = f"HTTP {r.status_code}"
                        self._breaker_record(False)
                        log.warning("D1 request failed (%s), attempt %d", last, attempt + 1)
                        ra = _retry_after(r)
                        if ra is not None:
                            if ra > self.max_retry_after_s:
                                raise D1Error(f"D1 asked to retry after {ra:.0f} s ({last}); not waiting")
                            wait = max(wait, ra)
                    else:
                        self._breaker_record(True)    # reachable: a 4xx is a request problem, not an outage
                        return self._parse(r)
            finally:
                if trial:                             # every exit frees the slot (deadline, Retry-After, _parse)
                    self._release_trial()
            if attempt == self.retries:
                break
            wait = min(wait, 8.0)
            rem = deadline_remaining()
            if rem is not None and wait >= rem:
                raise D1Error(f"storage deadline exceeded ({last})")
            self._sleep(wait)
        raise D1Error(f"D1 unavailable after {self.retries + 1} attempts ({last})")

    @staticmethod
    def _parse(r) -> list[dict]:
        try:
            data = r.json()
        except ValueError:
            raise D1Error(f"D1 returned HTTP {r.status_code} with a non-JSON body", transient=False) from None
        if r.status_code >= 400 or not data.get("success", False):
            # D1 error messages name tables and constraints, never bound parameter values; capped anyway
            errs = "; ".join(f"{e.get('code')}: {str(e.get('message'))[:160]}" for e in data.get("errors", []) or [])
            raise D1Error(f"D1 query failed (HTTP {r.status_code}): {errs or 'no error detail'}", transient=False)
        res = data.get("result") or []
        for item in res:
            if isinstance(item, dict) and item.get("success") is False:
                raise D1Error("D1 statement failed", transient=False)
        return res

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


R2_CONNECT_TIMEOUT_S, R2_READ_TIMEOUT_S, R2_MAX_ATTEMPTS = 3, 5, 2


def r2_deadline_guard(**_kw) -> None:
    """botocore before-send hook: runs before every HTTP attempt, retries included, in the calling thread."""
    rem = deadline_remaining()
    if rem is not None and rem <= 0:
        raise StorageError("R2 attempt not started: storage deadline exceeded")


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
        # total_max_attempts counts the first call (botocore's max_attempts meaning differs between retry modes)
        s3 = boto3.client("s3", endpoint_url=endpoint or f"https://{account_id}.r2.cloudflarestorage.com",
                          aws_access_key_id=access_key_id, aws_secret_access_key=secret_access_key, region_name="auto",
                          config=Config(signature_version="s3v4", connect_timeout=R2_CONNECT_TIMEOUT_S,
                                        read_timeout=R2_READ_TIMEOUT_S,
                                        retries={"total_max_attempts": R2_MAX_ATTEMPTS, "mode": "standard"}))
        s3.meta.events.register("before-send.s3", r2_deadline_guard)
        return cls(bucket, s3)

    def _call(self, what: str, fn, **kw):
        rem = deadline_remaining()
        if rem is not None and rem <= 0:
            raise StorageError(f"R2 {what} skipped: storage deadline exceeded")
        mark_storage_call()
        try:
            return fn(Bucket=self.bucket, **kw)
        except StorageError:
            raise
        except Exception as e:                   # noqa: BLE001 - botocore errors carry no secrets, but keep it short
            code = getattr(e, "response", {}).get("Error", {}).get("Code") if hasattr(e, "response") else None
            if code in ("NoSuchKey", "404"):
                raise KeyError("no such key") from None      # the key names a user: never in an exception
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
        self.delete_keys(keys)
        return len(keys)

    def delete_keys(self, keys: list[str]) -> None:
        for i in range(0, len(keys), 1000):
            chunk = keys[i:i + 1000]
            r = self._call("delete", self._s3.delete_objects,
                           Delete={"Objects": [{"Key": k} for k in chunk], "Quiet": True})
            if r.get("Errors"):
                raise StorageError(f"R2 delete failed for {len(r['Errors'])} objects")


class CloudflareBackend(Storage):
    name = "cloudflare"
    persistent = True

    def __init__(self, d1: D1Client, blobs: R2Blobs | None = None, store_raw_windows: bool = False):
        super().__init__(store_raw_windows)
        self.d1, self.blobs = d1, blobs
        self._tl = threading.local()             # per-thread write buffer for buffered()

    def __repr__(self) -> str:
        return f"CloudflareBackend(blobs={'yes' if self.blobs else 'no'})"

    def has_blobs(self) -> bool:
        return self.blobs is not None

    # ---- writes: sent now, or collected by buffered() and sent as one batch ----
    def _write(self, stmts: list[tuple[str, list]]) -> None:
        if not stmts:
            return
        buf = getattr(self._tl, "buf", None)
        if buf is not None:
            buf.extend(stmts)
            return
        self._send(stmts)

    def _send(self, stmts: list[tuple[str, list]]) -> None:
        if len(stmts) == 1:
            self.d1.query(*stmts[0])
            return
        for j in range(0, len(stmts), STMTS_PER_CALL):
            self.d1.batch(stmts[j:j + STMTS_PER_CALL])

    @contextlib.contextmanager
    def buffered(self) -> Iterator[None]:
        if getattr(self._tl, "buf", None) is not None:      # nested: the outer block sends
            yield
            return
        self._tl.buf = []
        try:
            yield
            stmts = self._tl.buf
        finally:
            self._tl.buf = None
        self._send(stmts)

    @staticmethod
    def _guarded_rows(head: str, cols: int, rows: list[tuple], user_id: str, tail: str = "",
                      extra_where: str = "") -> list[tuple[str, list]]:
        """Multi-row INSERT ... SELECT ... FROM (VALUES ...) AS v WHERE NOT EXISTS (tombstone) [extra], chunked so each
        statement binds at most MAX_PARAMS parameters (one is the user hash). The VALUES columns are v.column1..N."""
        per = max(1, (MAX_PARAMS - 1) // cols)
        ph = "(" + ",".join("?" * cols) + ")"
        out = []
        for i in range(0, len(rows), per):
            chunk = rows[i:i + per]
            sql = (f"{head} SELECT * FROM (VALUES {','.join([ph] * len(chunk))}) AS v WHERE {NOT_TOMB}"
                   f"{extra_where} {tail}")
            out.append((sql.strip(), [x for r in chunk for x in r] + [user_hash(user_id)]))
        return out

    # ---- deletion: tombstone and pending marker first, then every table; marker cleared after the blobs ----
    def _delete_user(self, user_id):
        h, now = user_hash(user_id), time.time()
        stmts = [("INSERT OR IGNORE INTO tombstones (user_hash, deleted_at) VALUES (?, ?)", [h, now]),
                 ("INSERT OR IGNORE INTO pending_deletions (user_hash, started_at) VALUES (?, ?)", [h, now])]
        stmts += [(f"DELETE FROM {t} WHERE user_id = ?", [user_id]) for t in USER_TABLES]
        if not self.has_blobs():
            stmts.append(("DELETE FROM pending_deletions WHERE user_hash = ?", [h]))
        self.d1.batch(stmts)
        return {t: "deleted" for t in USER_TABLES}

    def _finish_delete(self, user_id):
        if self.has_blobs():
            self.d1.query("DELETE FROM pending_deletions WHERE user_hash = ?", [user_hash(user_id)])

    def is_tombstoned(self, user_id: str) -> bool:
        return bool(self.d1.query("SELECT 1 AS x FROM tombstones WHERE user_hash = ?", [user_hash(user_id)]))

    def deletion_incomplete(self, user_id: str) -> bool:
        return bool(self.d1.query("SELECT 1 AS x FROM pending_deletions WHERE user_hash = ?", [user_hash(user_id)]))

    # ---- users ----
    def _put_user(self, user_id, attrs):
        a = dict(attrs)
        created = float(a.pop("created_at", time.time()))
        self._write([("INSERT INTO users (user_id, created_at, attrs) SELECT ?, ?, ? "
                      f"WHERE {NOT_TOMB} ON CONFLICT (user_id) DO UPDATE SET attrs = excluded.attrs",
                      [user_id, created, _dumps(a), user_hash(user_id)])])

    @staticmethod
    def _user_row(rows):
        if not rows:
            return None
        return {"created_at": rows[0]["created_at"], **json.loads(rows[0]["attrs"] or "{}")}

    def _get_user(self, user_id):
        return self._user_row(self.d1.query("SELECT created_at, attrs FROM users WHERE user_id = ?", [user_id]))

    def load_user(self, user_id: str, risk_keep: int = 100) -> dict | None:
        """One D1 request: tombstone and pending-deletion checks, user, baselines, state, devices, and the risk
        history bounded as in bound_history (last `risk_keep` points, oldest point per hour of the last 7 days,
        newest point older). A deleted user gives None, or {"deletion_incomplete": True} while its marker remains."""
        if self.is_deleted(user_id, check_remote=False):
            return {"deletion_incomplete": True} if self.deletion_incomplete(user_id) else None
        now = time.time()
        week_ago = now - WEEK_S
        res = self.d1.batch([
            ("SELECT 1 AS x FROM tombstones WHERE user_hash = ?", [user_hash(user_id)]),
            ("SELECT created_at, attrs FROM users WHERE user_id = ?", [user_id]),
            (self._BASELINE_SQL, [user_id]),
            ("SELECT key, value FROM user_state WHERE user_id = ?", [user_id]),
            ("SELECT device_id, rec FROM devices WHERE user_id = ?", [user_id]),
            (f"SELECT t, r_mdtr FROM risk_results WHERE user_id = ? ORDER BY t DESC LIMIT {int(risk_keep)}", [user_id]),
            ("SELECT t, r_mdtr FROM risk_results WHERE user_id = ? AND t <= ? ORDER BY t DESC LIMIT 1",
             [user_id, week_ago]),
            ("SELECT MIN(t) AS t, r_mdtr FROM risk_results WHERE user_id = ? AND t > ? "
             "GROUP BY CAST(t / 3600 AS INTEGER)", [user_id, week_ago]),
            ("SELECT 1 AS x FROM pending_deletions WHERE user_hash = ?", [user_hash(user_id)]),
        ])
        if res[0]:
            with self._tlock:
                self._tomb.add(user_id)
            return {"deletion_incomplete": True} if res[8] else None
        rec = self._user_row(res[1])
        if rec is None:
            return None
        hist = [(r["t"], r["r_mdtr"]) for r in res[5] + res[6] + res[7]]
        return {"user": rec, "baselines": self._baseline_rows(user_id, res[2]),
                "state": {r["key"]: json.loads(r["value"]) for r in res[3]},
                "devices": {r["device_id"]: json.loads(r["rec"]) for r in res[4]},
                "risk": bound_history(hist, risk_keep, now)}

    # ---- baselines ----
    _BASELINE_SQL = ("SELECT modality, metric, baseline_id, mode, phase, status, vals, center, spread, created_at "
                     "FROM baselines WHERE user_id = ?")

    def _put_baselines(self, records):
        """Upsert, never for a tombstoned user and never for a baseline_id already archived (a stale flush racing a
        reset is ignored)."""
        now = time.time()
        by_user: dict[str, list] = {}
        for r in records:
            by_user.setdefault(r["user_id"], []).append(
                (r["user_id"], r["modality"], r["metric"], r["baseline_id"], r.get("mode"), r.get("phase"), r["status"],
                 _dumps(r.get("values", [])), r.get("center"), r.get("spread"), r.get("created"), now))
        stmts = []
        for uid, rows in by_user.items():
            stmts += self._guarded_rows(
                "INSERT INTO baselines (user_id, modality, metric, baseline_id, mode, phase, status, vals, center, "
                "spread, created_at, updated_at)", 12, rows, uid,
                "ON CONFLICT (user_id, modality, metric) DO UPDATE SET baseline_id = excluded.baseline_id, "
                "mode = excluded.mode, phase = excluded.phase, status = excluded.status, vals = excluded.vals, "
                "center = excluded.center, spread = excluded.spread, updated_at = excluded.updated_at",
                " AND NOT EXISTS (SELECT 1 FROM baseline_archive a WHERE a.baseline_id = v.column4 "
                "AND a.user_id = v.column1)")
        self._write(stmts)

    @staticmethod
    def _baseline_rows(user_id, rows):
        return [{"user_id": user_id, "modality": r["modality"], "metric": r["metric"], "baseline_id": r["baseline_id"],
                 "mode": r["mode"], "phase": r["phase"], "status": r["status"], "values": json.loads(r["vals"] or "[]"),
                 "center": r["center"], "spread": r["spread"], "created": r["created_at"]} for r in rows]

    def _get_baselines(self, user_id):
        return self._baseline_rows(user_id, self.d1.query(self._BASELINE_SQL, [user_id]))

    _ARCHIVE_COPY = ("INSERT OR IGNORE INTO baseline_archive (baseline_id, user_id, modality, metric, mode, phase, "
                     "vals, center, spread, created_at, archived_at) SELECT baseline_id, user_id, modality, metric, "
                     "mode, phase, vals, center, spread, created_at, ? FROM baselines WHERE user_id = ? AND modality = ?")

    def _archive_baselines(self, user_id, modality, records):
        """records None: archive every stored baseline of the modality. Otherwise archive exactly the snapshot's
        baseline_ids (the in-memory copies first, then the stored rows with those ids) and delete only those rows,
        so a baseline created and flushed after the snapshot stays live in storage, as it does in memory."""
        now = time.time()
        h = user_hash(user_id)
        if records is None:
            self._write([(f"{self._ARCHIVE_COPY} AND {NOT_TOMB}", [now, user_id, modality, h]),
                         ("DELETE FROM baselines WHERE user_id = ? AND modality = ?", [user_id, modality])])
            return
        recs = [r for r in records if r.get("modality", modality) == modality]
        if not recs:
            return
        rows = [(r["baseline_id"], user_id, modality, r["metric"], r.get("mode"), r.get("phase"),
                 _dumps(r.get("values", [])), r.get("center"), r.get("spread"), r.get("created"), now) for r in recs]
        stmts = self._guarded_rows("INSERT OR IGNORE INTO baseline_archive (baseline_id, user_id, modality, metric, "
                                   "mode, phase, vals, center, spread, created_at, archived_at)", 11, rows, user_id)
        ids = sorted({r["baseline_id"] for r in recs})
        for i in range(0, len(ids), ID_CHUNK):
            chunk = ids[i:i + ID_CHUNK]
            inn = ",".join("?" * len(chunk))
            stmts += [(f"{self._ARCHIVE_COPY} AND baseline_id IN ({inn}) AND {NOT_TOMB}",
                       [now, user_id, modality, *chunk, h]),
                      (f"DELETE FROM baselines WHERE user_id = ? AND modality = ? AND baseline_id IN ({inn})",
                       [user_id, modality, *chunk])]
        self._write(stmts)

    def _archived_ids(self, user_id, ids):
        out = set()
        for i in range(0, len(ids), ID_CHUNK):
            chunk = list(ids[i:i + ID_CHUNK])
            inn = ",".join("?" * len(chunk))
            rows = self.d1.query(f"SELECT baseline_id FROM baseline_archive WHERE user_id = ? AND baseline_id IN ({inn})",
                                 [user_id, *chunk])
            out |= {r["baseline_id"] for r in rows}
        return out

    # ---- state ----
    def _put_state(self, user_id, key, value):
        self._write([("INSERT INTO user_state (user_id, key, value, updated_at) SELECT ?, ?, ?, ? "
                      f"WHERE {NOT_TOMB} ON CONFLICT (user_id, key) DO UPDATE SET value = excluded.value, "
                      "updated_at = excluded.updated_at", [user_id, key, _dumps(value), time.time(), user_hash(user_id)])])

    def _get_state(self, user_id):
        rows = self.d1.query("SELECT key, value FROM user_state WHERE user_id = ?", [user_id])
        return {r["key"]: json.loads(r["value"]) for r in rows}

    # ---- inputs, flags, risk ----
    def _add_inputs(self, user_id, rows):
        now = time.time()
        self._write(self._guarded_rows(
            "INSERT OR IGNORE INTO inputs (input_id, user_id, kind, t_utc, meta, blob_key, created_at)", 7,
            [(uuid.uuid4().hex, user_id, r["kind"], r.get("t_utc"), _dumps(r.get("meta", {})), r.get("blob_key"), now)
             for r in rows], user_id))

    def _list_inputs(self, user_id):
        rows = self.d1.query("SELECT input_id, kind, t_utc, meta, blob_key FROM inputs WHERE user_id = ? "
                             "ORDER BY created_at", [user_id])
        return [{**r, "meta": json.loads(r["meta"] or "{}")} for r in rows]

    def _add_flags(self, user_id, flags):
        self._write(self._guarded_rows(
            "INSERT OR IGNORE INTO flags (flag_id, user_id, t, domain, metric, modality, level, z, component_score, "
            "confidence)", 10,
            [(uuid.uuid4().hex, user_id, f.get("t", time.time()), f.get("domain"), f.get("metric"), f.get("modality"),
              f.get("level"), f.get("z"), f.get("component_score"), f.get("confidence")) for f in flags], user_id))

    def _list_flags(self, user_id):
        return self.d1.query("SELECT t, domain, metric, modality, level, z, component_score, confidence FROM flags "
                             "WHERE user_id = ? ORDER BY t", [user_id])

    def _add_risk(self, user_id, t, r_mdtr, band, summary):
        self._write([("INSERT OR IGNORE INTO risk_results (risk_id, user_id, t, r_mdtr, band, summary) "
                      f"SELECT ?, ?, ?, ?, ?, ? WHERE {NOT_TOMB}",
                      [uuid.uuid4().hex, user_id, t, r_mdtr, band, _dumps(summary), user_hash(user_id)])])

    def _risk_history(self, user_id):
        rows = self.d1.query("SELECT t, r_mdtr FROM risk_results WHERE user_id = ? ORDER BY t", [user_id])
        return [(r["t"], r["r_mdtr"]) for r in rows]

    # ---- devices ----
    def _put_device(self, device_id, rec):
        self._write([("INSERT INTO devices (device_id, user_id, rec, updated_at) SELECT ?, ?, ?, ? "
                      f"WHERE {NOT_TOMB} ON CONFLICT (device_id) DO UPDATE SET user_id = excluded.user_id, "
                      "rec = excluded.rec, updated_at = excluded.updated_at",
                      [device_id, rec["user_id"], _dumps(rec), time.time(), user_hash(rec["user_id"])])])

    def _update_device(self, device_id, rec):
        # json_extract has no affinity and params arrive as text, so both sides are cast to INTEGER
        self._write([("UPDATE devices SET rec = ?, updated_at = ? WHERE device_id = ? AND user_id = ? "
                      "AND COALESCE(CAST(json_extract(rec, '$.last_seq') AS INTEGER), -1) < CAST(? AS INTEGER) "
                      f"AND {NOT_TOMB}",
                      [_dumps(rec), time.time(), device_id, rec["user_id"], int(rec.get("last_seq", -1)),
                       user_hash(rec["user_id"])])])

    def _get_device(self, device_id):
        rows = self.d1.query("SELECT rec FROM devices WHERE device_id = ?", [device_id])
        return json.loads(rows[0]["rec"]) if rows else None

    def _delete_device(self, device_id, user_id=None):
        if user_id is None:
            self._write([("DELETE FROM devices WHERE device_id = ?", [device_id])])
        else:
            self._write([("DELETE FROM devices WHERE device_id = ? AND user_id = ?", [device_id, user_id])])

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

    def _delete_keys(self, keys):
        self._need_blobs().delete_keys(keys)


SCHEMA_PATH = Path(__file__).with_name("schema.sql")


def check_schema(d1: D1Client) -> int:
    """Refuse to run against a D1 database that lacks a migration this code needs (without migration 2, every DELETE
    would fail). Returns the applied version. Raises StorageConfigError naming the missing migrations."""
    hint = "apply evade/storage/schema.sql with python -m evade.storage.migrate (it is idempotent)"
    try:
        rows = d1.query("SELECT MAX(version) AS v FROM schema_migrations")
    except D1Error as e:
        if e.transient or "no such table" not in str(e):
            raise                                # an outage, a bad token or a wrong database ID: not a schema issue
        raise StorageConfigError(f"D1 schema missing (no schema_migrations table). Refusing EVADE_STORAGE=cloudflare: "
                                 f"{hint}") from None
    v = int((rows[0].get("v") if rows else None) or 0)
    if v < SCHEMA_VERSION:
        missing = ", ".join(f"migration {n} ({MIGRATIONS[n]})" for n in range(v + 1, SCHEMA_VERSION + 1))
        raise StorageConfigError(f"D1 schema is at version {v}; this code needs {SCHEMA_VERSION}. Missing: {missing}. "
                                 f"Refusing EVADE_STORAGE=cloudflare: {hint}")
    return v
