"""Service state storage: NullBackend (default, memory only) or CloudflareBackend (D1 + R2). Configuration from env only.

  EVADE_STORAGE              memory (default) | cloudflare
  EVADE_STORAGE_DEADLINE_S   per-request time budget for all storage calls (default 8)
  CF_ACCOUNT_ID              Cloudflare account ID
  CF_API_TOKEN               API token with Account > D1 > Edit
  CF_D1_DATABASE_ID          D1 database UUID
  CF_R2_BUCKET               data bucket for raw windows and exports (optional: without the R2 variables, blobs are off)
  CF_R2_ACCESS_KEY_ID        R2 API token Access Key ID (Object Read & Write on the data bucket only)
  CF_R2_SECRET_ACCESS_KEY    R2 API token Secret Access Key
  CF_R2_ENDPOINT             optional override, e.g. https://<account>.eu.r2.cloudflarestorage.com for EU buckets
  EVADE_STORE_RAW_WINDOWS    "1" to keep raw signal windows in R2 under raw/<user_id>/ (off by default)

Model artifacts (scripts/fetch_models.py), in their own bucket:
  CF_R2_MODELS_BUCKET                    models bucket
  CF_R2_MODELS_ACCESS_KEY_ID / _SECRET_ACCESS_KEY              Object Read only (servers)
  CF_R2_MODELS_WRITE_ACCESS_KEY_ID / _WRITE_SECRET_ACCESS_KEY  Object Read & Write (training machines only)
"""
from __future__ import annotations

import logging
import os

from .base import (Storage, StorageConfigError, StorageError, USER_BLOB_PREFIXES, deadline_remaining,
                   storage_deadline)
from .memory import MemoryBackend, NullBackend

__all__ = ["Storage", "StorageError", "StorageConfigError", "MemoryBackend", "NullBackend", "from_env", "r2_from_env",
           "models_r2_from_env", "USER_BLOB_PREFIXES", "storage_deadline", "deadline_remaining", "deadline_s"]

log = logging.getLogger("evade.storage")


def _need(names: tuple[str, ...]) -> dict:
    missing = [n for n in names if not os.environ.get(n)]
    if missing:
        raise StorageConfigError(f"missing environment variables: {', '.join(missing)}")   # names only, never values
    return {n: os.environ[n] for n in names}


def deadline_s() -> float:
    try:
        return max(0.5, float(os.environ.get("EVADE_STORAGE_DEADLINE_S", "8")))
    except ValueError:
        return 8.0


def r2_from_env(required: bool = False):
    """The data bucket (raw windows, exports) with the API's read-write key."""
    from .cloudflare import R2Blobs
    names = ("CF_ACCOUNT_ID", "CF_R2_BUCKET", "CF_R2_ACCESS_KEY_ID", "CF_R2_SECRET_ACCESS_KEY")
    if not required and not all(os.environ.get(n) for n in names[1:]):
        return None
    e = _need(names)
    return R2Blobs.from_credentials(e["CF_ACCOUNT_ID"], e["CF_R2_BUCKET"], e["CF_R2_ACCESS_KEY_ID"],
                                    e["CF_R2_SECRET_ACCESS_KEY"], os.environ.get("CF_R2_ENDPOINT") or None)


def models_r2_from_env(write: bool = False):
    """The models bucket. Servers read with a read-only key; only training machines hold the write key.
    Without CF_R2_MODELS_BUCKET, falls back to models/ in the data bucket (and warns: that key can rewrite models)."""
    from .cloudflare import R2Blobs
    bucket = os.environ.get("CF_R2_MODELS_BUCKET")
    if not bucket:
        log.warning("CF_R2_MODELS_BUCKET is not set: model artifacts share the data bucket and its read-write key")
        return r2_from_env(required=True)
    pre = "CF_R2_MODELS_WRITE_" if write else "CF_R2_MODELS_"
    names = ("CF_ACCOUNT_ID", f"{pre}ACCESS_KEY_ID", f"{pre}SECRET_ACCESS_KEY")
    e = _need(names)
    return R2Blobs.from_credentials(e["CF_ACCOUNT_ID"], bucket, e[names[1]], e[names[2]],
                                    os.environ.get("CF_R2_ENDPOINT") or None)


def d1_from_env():
    from .cloudflare import D1Client
    e = _need(("CF_ACCOUNT_ID", "CF_API_TOKEN", "CF_D1_DATABASE_ID"))
    return D1Client(e["CF_ACCOUNT_ID"], e["CF_D1_DATABASE_ID"], e["CF_API_TOKEN"])


def from_env() -> Storage:
    kind = os.environ.get("EVADE_STORAGE", "memory").strip().lower()
    raw = os.environ.get("EVADE_STORE_RAW_WINDOWS") == "1"
    if kind == "memory":
        return NullBackend(store_raw_windows=raw)
    if kind == "cloudflare":
        from .cloudflare import CloudflareBackend
        return CloudflareBackend(d1_from_env(), r2_from_env(), store_raw_windows=raw)
    raise StorageConfigError("EVADE_STORAGE must be memory or cloudflare")
