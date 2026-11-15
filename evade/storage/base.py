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
  blobs          raw windows, model artifacts (models/<task>/...), exports (exports/<user_id>/...)

delete_user writes a tombstone first, so no later write can recreate the user's data, then removes every row and
every blob under that user's prefixes. Writes for a tombstoned user are dropped silently.
"""
from __future__ import annotations

import hashlib
import threading
from typing import Any

USER_BLOB_PREFIXES = ("raw/", "exports/")
STATE_KEYS = ("radiation", "overrides", "latest", "eeg_baseline")


class StorageError(RuntimeError):
    """A backend call failed after retries. Messages never contain credentials."""


class StorageConfigError(StorageError):
    pass


def user_hash(user_id: str) -> str:
    return hashlib.sha256(user_id.encode("utf-8")).hexdigest()


class Storage:
    """Public API with tombstone gating; subclasses implement the _underscore methods."""

    name = "base"
    persistent = False          # memory: False (the app's in-memory state is the source of truth)

    def __init__(self, store_raw_windows: bool = False):
        self.store_raw_windows = store_raw_windows
        self._tomb: set[str] = set()
        self._tlock = threading.Lock()

    # ---- deletion ----
    def is_deleted(self, user_id: str) -> bool:
        with self._tlock:
            return user_id in self._tomb

    def delete_user(self, user_id: str) -> dict:
        with self._tlock:
            self._tomb.add(user_id)              # in-process gate first: concurrent writes are dropped from now on
        counts = self._delete_user(user_id)
        blobs = 0
        if self.has_blobs():
            for p in USER_BLOB_PREFIXES:
                blobs += self._delete_prefix(f"{p}{user_id}/")
        return {**counts, "blobs": blobs}

    # ---- users ----
    def put_user(self, user_id: str, attrs: dict) -> None:
        if not self.is_deleted(user_id):
            self._put_user(user_id, attrs)

    def get_user(self, user_id: str) -> dict | None:
        return None if self.is_deleted(user_id) else self._get_user(user_id)

    # ---- baselines ----
    def put_baselines(self, records: list[dict]) -> None:
        recs = [r for r in records if not self.is_deleted(r["user_id"])]
        if recs:
            self._put_baselines(recs)

    def get_baselines(self, user_id: str) -> list[dict]:
        return self._get_baselines(user_id)

    def archive_baselines(self, user_id: str, modality: str) -> None:
        if not self.is_deleted(user_id):
            self._archive_baselines(user_id, modality)

    # ---- per-user state documents ----
    def put_state(self, user_id: str, key: str, value: Any) -> None:
        if key not in STATE_KEYS:
            raise ValueError(f"unknown state key {key}")
        if not self.is_deleted(user_id):
            self._put_state(user_id, key, value)

    def get_state(self, user_id: str) -> dict:
        return self._get_state(user_id)

    # ---- readings / windows metadata ----
    def add_inputs(self, user_id: str, rows: list[dict]) -> None:
        if rows and not self.is_deleted(user_id):
            self._add_inputs(user_id, rows)

    def list_inputs(self, user_id: str) -> list[dict]:
        return self._list_inputs(user_id)

    def put_raw_window(self, user_id: str, window_id: str, payload: bytes) -> str | None:
        """Store raw samples only when explicitly enabled; returns the blob key or None."""
        if not (self.store_raw_windows and self.has_blobs()) or self.is_deleted(user_id):
            return None
        key = f"raw/{user_id}/{window_id}.json"
        self._put_blob(key, payload, "application/json")
        return key

    # ---- flags and risk results ----
    def add_flags(self, user_id: str, flags: list[dict]) -> None:
        if flags and not self.is_deleted(user_id):
            self._add_flags(user_id, flags)

    def list_flags(self, user_id: str) -> list[dict]:
        return self._list_flags(user_id)

    def add_risk(self, user_id: str, t: float, r_mdtr: float, band: str, summary: dict) -> None:
        if not self.is_deleted(user_id):
            self._add_risk(user_id, t, r_mdtr, band, summary)

    def risk_history(self, user_id: str) -> list[tuple[float, float]]:
        return self._risk_history(user_id)

    # ---- device bindings ----
    def put_device(self, device_id: str, rec: dict) -> None:
        if not self.is_deleted(rec["user_id"]):
            self._put_device(device_id, rec)

    def get_device(self, device_id: str) -> dict | None:
        rec = self._get_device(device_id)
        return None if rec is None or self.is_deleted(rec["user_id"]) else rec

    def delete_device(self, device_id: str) -> None:
        self._delete_device(device_id)

    def devices_for_user(self, user_id: str) -> dict[str, dict]:
        return self._devices_for_user(user_id)

    # ---- blobs (models, exports, raw windows) ----
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
    def _put_user(self, user_id, attrs): raise NotImplementedError
    def _get_user(self, user_id): raise NotImplementedError
    def _put_baselines(self, records): raise NotImplementedError
    def _get_baselines(self, user_id): raise NotImplementedError
    def _archive_baselines(self, user_id, modality): raise NotImplementedError
    def _put_state(self, user_id, key, value): raise NotImplementedError
    def _get_state(self, user_id): raise NotImplementedError
    def _add_inputs(self, user_id, rows): raise NotImplementedError
    def _list_inputs(self, user_id): raise NotImplementedError
    def _add_flags(self, user_id, flags): raise NotImplementedError
    def _list_flags(self, user_id): raise NotImplementedError
    def _add_risk(self, user_id, t, r_mdtr, band, summary): raise NotImplementedError
    def _risk_history(self, user_id): raise NotImplementedError
    def _put_device(self, device_id, rec): raise NotImplementedError
    def _get_device(self, device_id): raise NotImplementedError
    def _delete_device(self, device_id): raise NotImplementedError
    def _devices_for_user(self, user_id): raise NotImplementedError
    def _put_blob(self, key, data, content_type): raise StorageConfigError("no blob store configured")
    def _get_blob(self, key): raise StorageConfigError("no blob store configured")
    def _list_blobs(self, prefix): raise StorageConfigError("no blob store configured")
    def _delete_prefix(self, prefix): return 0
