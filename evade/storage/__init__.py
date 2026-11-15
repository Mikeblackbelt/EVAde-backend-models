"""Service state storage: MemoryBackend (default) or CloudflareBackend (D1 + R2). Configuration from env only.

  EVADE_STORAGE              memory (default) | cloudflare
  CF_ACCOUNT_ID              Cloudflare account ID
  CF_API_TOKEN               API token with Account > D1 > Edit
  CF_D1_DATABASE_ID          D1 database UUID
  CF_R2_BUCKET               R2 bucket (optional: without the R2 variables, blobs are disabled)
  CF_R2_ACCESS_KEY_ID        R2 API token Access Key ID (Object Read & Write on that bucket)
  CF_R2_SECRET_ACCESS_KEY    R2 API token Secret Access Key
  CF_R2_ENDPOINT             optional override, e.g. https://<account>.eu.r2.cloudflarestorage.com for EU buckets
  EVADE_STORE_RAW_WINDOWS    "1" to keep raw signal windows in R2 under raw/<user_id>/ (off by default)
"""
from __future__ import annotations

import os

from .base import Storage, StorageConfigError, StorageError, USER_BLOB_PREFIXES
from .memory import MemoryBackend

__all__ = ["Storage", "StorageError", "StorageConfigError", "MemoryBackend", "from_env", "r2_from_env",
           "USER_BLOB_PREFIXES"]


def _need(names: tuple[str, ...]) -> dict:
    missing = [n for n in names if not os.environ.get(n)]
    if missing:
        raise StorageConfigError(f"missing environment variables: {', '.join(missing)}")   # names only, never values
    return {n: os.environ[n] for n in names}


def r2_from_env(required: bool = False):
    from .cloudflare import R2Blobs
    names = ("CF_ACCOUNT_ID", "CF_R2_BUCKET", "CF_R2_ACCESS_KEY_ID", "CF_R2_SECRET_ACCESS_KEY")
    if not required and not all(os.environ.get(n) for n in names[1:]):
        return None
    e = _need(names)
    return R2Blobs.from_credentials(e["CF_ACCOUNT_ID"], e["CF_R2_BUCKET"], e["CF_R2_ACCESS_KEY_ID"],
                                    e["CF_R2_SECRET_ACCESS_KEY"], os.environ.get("CF_R2_ENDPOINT") or None)


def d1_from_env():
    from .cloudflare import D1Client
    e = _need(("CF_ACCOUNT_ID", "CF_API_TOKEN", "CF_D1_DATABASE_ID"))
    return D1Client(e["CF_ACCOUNT_ID"], e["CF_D1_DATABASE_ID"], e["CF_API_TOKEN"])


def from_env() -> Storage:
    kind = os.environ.get("EVADE_STORAGE", "memory").strip().lower()
    raw = os.environ.get("EVADE_STORE_RAW_WINDOWS") == "1"
    if kind == "memory":
        return MemoryBackend(store_raw_windows=raw)
    if kind == "cloudflare":
        from .cloudflare import CloudflareBackend
        return CloudflareBackend(d1_from_env(), r2_from_env(), store_raw_windows=raw)
    raise StorageConfigError("EVADE_STORAGE must be memory or cloudflare")
