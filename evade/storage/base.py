"""Storage interface for EVAde service state.

What is stored (all keyed by the pseudonymous user_id; no names or emails ever reach storage):
  users          created_at plus coarse attributes (cohort, age band, sex) given at creation
  baselines      per (user, modality, metric): status, the derived metric values that built it, center, spread;
                 reset baselines move to an archive
  user state     small JSON documents per user: radiation domain, active overrides, latest comparisons,
                 the locked EEG feature baseline (median, MAD)
  inputs         metadata of readings and windows (modality, time, fs, shape, quality, source); raw samples
                 only if EVADE_STORE_RAW_WINDOWS=1, as blobs under raw/<user_id>/
  flags, risk    non-green comparisons and risk results (R, band, overrides, MDTR summary)
  devices        Bluetooth device bindings (device_id -> user, kind, last seq)
  blobs          raw windows, exports (exports/<user_id>/...); model artifacts live in their own bucket
                 (evade/models/README.md)

delete_user writes a tombstone and a pending-deletion marker first, then removes every row and every blob under the
user's prefixes, and clears the marker last. The persistent backend enforces the tombstone inside every write
statement, so a write already in flight when the DELETE commits cannot recreate rows; the in-process set below only
saves a round trip. R2 has no such guard: a raw-window PUT already in flight when the DELETE lists the prefix can
land afterwards, so the API re-checks the tombstone after its PUTs and deletes those keys (best effort; a crash
between the PUT and the re-check can still leave an orphan blob).

Time budget: storage_deadline() sets a per-request deadline (a context variable) that the D1 client honours, so one
request cannot spend more than that on storage, however many calls it makes.
"""
from __future__ import annotations

import contextlib
import contextvars
import hashlib
import threading
import time
from typing import Any, Iterator

USER_BLOB_PREFIXES = ("raw/", "exports/")
STATE_KEYS = ("radiation", "overrides", "latest", "eeg_baseline")
WEEK_S = 7 * 86400

_DEADLINE: contextvars.ContextVar[float | None] = contextvars.ContextVar("evade_storage_deadline", default=None)
# a mutable holder set by the request middleware; backends append to it on every network call, so the middleware
# (whose context is a parent of the endpoint's) can tell whether a request touched storage
_TOUCHED: contextvars.ContextVar[list | None] = contextvars.ContextVar("evade_storage_touched", default=None)


@contextlib.contextmanager
def storage_deadline(seconds: float | None) -> Iterator[list]:
    """Every storage call inside the block must finish within `seconds` (None: no deadline). Yields a list that
    gets one entry per backend network call made inside the block (threads started from it included)."""
    touched: list = []
    tok = _DEADLINE.set(None if seconds is None else time.monotonic() + max(0.0, float(seconds)))
    tok2 = _TOUCHED.set(touched)
    try:
        yield touched
    finally:
        _TOUCHED.reset(tok2)
        _DEADLINE.reset(tok)


def deadline_remaining() -> float | None:
    d = _DEADLINE.get()
    return None if d is None else d - time.monotonic()


def mark_storage_call() -> None:
    t = _TOUCHED.get()
    if t is not None:
        t.append(1)


class StorageError(RuntimeError):
    """A backend call failed after retries. Messages never contain credentials, user IDs or submitted values.
    transient: an outage (timeout, 5xx, deadline, open breaker) rather than a request the backend rejected."""

    transient = True


class StorageConfigError(StorageError):
    pass


def user_hash(user_id: str) -> str:
    return hashlib.sha256(user_id.encode("utf-8")).hexdigest()


def bound_history(hist, keep: int = 100, now: float | None = None) -> list:
    """Risk history (hydrated and in memory), bounded to at most keep + 170 points: the last `keep` (t, r) points, the
    oldest point of every hour of the last 7 days (as they age they answer the 7-day trend lookup: whenever the
    exact answer exists, a kept point from the same hour exists and is at most an hour older), and the newest
    point at least 7 days old (the trend input now). Sorted by time."""
    now = time.time() if now is None else now
    hist = sorted({(float(t), float(r)) for t, r in hist})
    out = set(hist[-keep:]) if keep > 0 else set()
    hourly: dict[int, tuple] = {}
    for t, r in hist:
        if now - t < WEEK_S:
            hourly.setdefault(int(t // 3600), (t, r))   # ascending order: the oldest point of each hour
    out |= set(hourly.values())
    old = [x for x in hist if now - x[0] >= WEEK_S]
    if old:
        out.add(old[-1])
    return sorted(out)


class Storage:
    """Public API with tombstone gating; subclasses implement the _underscore methods."""

    name = "base"
    persistent = False          # memory: False (the app's in-memory state is the source of truth)

    def __init__(self, store_raw_windows: bool = False):
        self.store_raw_windows = store_raw_windows
        self._tomb: set[str] = set()
        self._tlock = threading.Lock()

    # ---- deletion ----
    def is_deleted(self, user_id: str, check_remote: bool = True) -> bool:
        """In-process tombstone, else (persistent backends, check_remote) the stored one; positives are cached.
        Never call with check_remote inside an application lock: it can be a network call."""
        with self._tlock:
            if user_id in self._tomb:
                return True
        if check_remote and self.persistent and self.is_tombstoned(user_id):
            with self._tlock:
                self._tomb.add(user_id)
            return True
        return False

    def _gate(self, user_id: str) -> bool:
        """Cheap pre-check for writes (in-process only); the backend's SQL guard is the authoritative check."""
        with self._tlock:
            return user_id in self._tomb

    def is_tombstoned(self, user_id: str) -> bool:
        return False

    def deletion_incomplete(self, user_id: str) -> bool:
        """True if a delete_user for this user started (tombstone written) but did not finish (blobs may remain)."""
        return False

    def delete_user(self, user_id: str) -> dict:
        """Idempotent: safe to run again after a failure part-way through."""
        with self._tlock:
            self._tomb.add(user_id)              # in-process gate first: concurrent writes are dropped from now on
        counts = self._delete_user(user_id)
        blobs = 0
        if self.has_blobs():
            for p in USER_BLOB_PREFIXES:
                blobs += self._delete_prefix(f"{p}{user_id}/")
        self._finish_delete(user_id)
        return {**counts, "blobs": blobs}

    # ---- users ----
    def put_user(self, user_id: str, attrs: dict, *, if_absent: bool = False) -> None:
        """if_absent: insert only; an existing row is left exactly as it is (register-with-existing-ID path)."""
        if not self._gate(user_id):
            self._put_user(user_id, attrs, if_absent) if if_absent else self._put_user(user_id, attrs)

    def get_user(self, user_id: str) -> dict | None:
        return None if self.is_deleted(user_id) else self._get_user(user_id)

    def load_user(self, user_id: str, risk_keep: int = 100) -> dict | None:
        """Everything the API needs to hydrate one user: {"user", "baselines", "state", "devices", "risk"} with risk
        bounded by bound_history; None if unknown or deleted; {"deletion_incomplete": True} if a deletion started
        but did not finish (the caller must not cache that as a miss: a retried DELETE has to finish it)."""
        if self.is_deleted(user_id):
            return {"deletion_incomplete": True} if self.deletion_incomplete(user_id) else None
        rec = self._get_user(user_id)
        if rec is None:
            return None
        return {"user": rec, "baselines": self._get_baselines(user_id), "state": self._get_state(user_id),
                "devices": self._devices_for_user(user_id),
                "risk": bound_history(self._risk_history(user_id), risk_keep)}

    # ---- baselines ----
    def put_baselines(self, records: list[dict]) -> None:
        recs = [r for r in records if not self._gate(r["user_id"])]
        if recs:
            self._put_baselines(recs)

    def get_baselines(self, user_id: str) -> list[dict]:
        return self._get_baselines(user_id)

    def archive_baselines(self, user_id: str, modality: str, records: list[dict] | None = None) -> None:
        """Move baselines for `modality` to the archive. With `records` (the caller's in-memory snapshot, possibly
        never stored yet), exactly those baseline_ids are archived and removed, so a baseline created after the
        snapshot is left alone; archiving them also blocks a stale flush from writing them back. Without `records`,
        every stored baseline of the modality is archived."""
        if not self._gate(user_id):
            self._archive_baselines(user_id, modality, None if records is None else list(records))

    def archived_baseline_ids(self, user_id: str, ids) -> set:
        """Which of these baseline_ids are in the user's archive (used to settle a reset whose outcome is unknown)."""
        ids = sorted(set(ids))
        return self._archived_ids(user_id, ids) if ids else set()

    # ---- per-user state documents ----
    def put_state(self, user_id: str, key: str, value: Any) -> None:
        if key not in STATE_KEYS:
            raise ValueError(f"unknown state key {key}")
        if not self._gate(user_id):
            self._put_state(user_id, key, value)

    def get_state(self, user_id: str) -> dict:
        return self._get_state(user_id)

    # ---- readings / windows metadata ----
    def add_inputs(self, user_id: str, rows: list[dict]) -> None:
        if rows and not self._gate(user_id):
            self._add_inputs(user_id, rows)

    def list_inputs(self, user_id: str) -> list[dict]:
        return self._list_inputs(user_id)

    def put_raw_window(self, user_id: str, window_id: str, payload: bytes) -> str | None:
        """Store raw samples only when explicitly enabled; returns the blob key or None."""
        if not (self.store_raw_windows and self.has_blobs()) or self._gate(user_id):
            return None
        key = f"raw/{user_id}/{window_id}.json"
        self._put_blob(key, payload, "application/json")
        return key

    def delete_blobs(self, keys: list[str]) -> None:
        if keys and self.has_blobs():
            self._delete_keys(list(keys))

    # ---- flags and risk results ----
    def add_flags(self, user_id: str, flags: list[dict]) -> None:
        if flags and not self._gate(user_id):
            self._add_flags(user_id, flags)

    def list_flags(self, user_id: str) -> list[dict]:
        return self._list_flags(user_id)

    def add_risk(self, user_id: str, t: float, r_mdtr: float, band: str, summary: dict) -> None:
        if not self._gate(user_id):
            self._add_risk(user_id, t, r_mdtr, band, summary)

    def risk_history(self, user_id: str) -> list[tuple[float, float]]:
        return self._risk_history(user_id)

    # ---- device bindings ----
    def put_device(self, device_id: str, rec: dict) -> None:
        """Create or overwrite a binding (bind resets last_seq)."""
        if not self._gate(rec["user_id"]):
            self._put_device(device_id, rec)

    def update_device(self, device_id: str, rec: dict) -> None:
        """Update an existing binding of the same user only if rec.last_seq is higher than the stored one;
        never inserts (an unbound or deleted device stays gone)."""
        if not self._gate(rec["user_id"]):
            self._update_device(device_id, rec)

    def get_device(self, device_id: str) -> dict | None:
        rec = self._get_device(device_id)
        return None if rec is None or self.is_deleted(rec["user_id"]) else rec

    def delete_device(self, device_id: str, user_id: str | None = None) -> None:
        self._delete_device(device_id, user_id)

    def devices_for_user(self, user_id: str) -> dict[str, dict]:
        return self._devices_for_user(user_id)

    # ---- batching ----
    @contextlib.contextmanager
    def buffered(self) -> Iterator[None]:
        """Writes inside the block may be sent together when it exits (one request to the backend)."""
        yield

    # ---- blobs (exports, raw windows) ----
    def has_blobs(self) -> bool:
        return False

    def put_blob(self, key: str, data: bytes, content_type: str = "application/octet-stream") -> None:
        self._put_blob(key, data, content_type)

    def get_blob(self, key: str) -> bytes | None:
        return self._get_blob(key)

    def list_blobs(self, prefix: str) -> list[str]:
        return self._list_blobs(prefix)

    # ---- to implement ----
    def _delete_user(self, user_id): raise NotImplementedError
    def _finish_delete(self, user_id): return None
    def _put_user(self, user_id, attrs, if_absent=False): raise NotImplementedError
    def _get_user(self, user_id): raise NotImplementedError
    def _put_baselines(self, records): raise NotImplementedError
    def _get_baselines(self, user_id): raise NotImplementedError
    def _archive_baselines(self, user_id, modality, records): raise NotImplementedError
    def _archived_ids(self, user_id, ids): raise NotImplementedError
    def _put_state(self, user_id, key, value): raise NotImplementedError
    def _get_state(self, user_id): raise NotImplementedError
    def _add_inputs(self, user_id, rows): raise NotImplementedError
    def _list_inputs(self, user_id): raise NotImplementedError
    def _add_flags(self, user_id, flags): raise NotImplementedError
    def _list_flags(self, user_id): raise NotImplementedError
    def _add_risk(self, user_id, t, r_mdtr, band, summary): raise NotImplementedError
    def _risk_history(self, user_id): raise NotImplementedError
    def _put_device(self, device_id, rec): raise NotImplementedError
    def _update_device(self, device_id, rec): raise NotImplementedError
    def _get_device(self, device_id): raise NotImplementedError
    def _delete_device(self, device_id, user_id): raise NotImplementedError
    def _devices_for_user(self, user_id): raise NotImplementedError
    def _put_blob(self, key, data, content_type): raise StorageConfigError("no blob store configured")
    def _get_blob(self, key): raise StorageConfigError("no blob store configured")
    def _list_blobs(self, prefix): raise StorageConfigError("no blob store configured")
    def _delete_prefix(self, prefix): return 0
    def _delete_keys(self, keys): return None
