"""Sync servable model folders between Cloudflare R2 and EVADE_MODEL_DIR, verifying sha256 on every file.

R2 layout: models/<task>/manifest.json plus the files it names (weights, feature_names.json, metrics.json).
Pull downloads into a temporary folder, checks every hash, then swaps the folder in; a failed check leaves the
current folder untouched. Push uploads the files first and manifest.json last, so a reader never sees a manifest
whose files are missing. The hashes come from the same bucket, so they catch corruption and partial uploads; keep
the bucket's write credentials to the people who train models.
"""
from __future__ import annotations

import hmac
import json
import logging
import re
import shutil
import uuid
from pathlib import Path

from . import manifest as MF

log = logging.getLogger("evade.models")
PREFIX = "models/"
TASK_RE = re.compile(r"^[a-z0-9_]{1,64}$")


def _files_of(m: dict) -> dict[str, str]:
    """Every file a manifest names, with its expected sha256 (weights first)."""
    out = {MF.safe_name(m["weights"], "weights"): m["sha256"]}
    for name, digest in (m.get("files") or {}).items():
        out[MF.safe_name(name, "files key")] = digest
    ref = m["input"].get("features")
    if ref and ref not in out:
        raise MF.ManifestError(f"input.features file {ref} has no sha256 in manifest.files")
    return out


def _local_ok(d: Path, m: dict) -> bool:
    mp = d / MF.MANIFEST
    if not mp.is_file():
        return False
    try:
        if json.loads(mp.read_text(encoding="utf-8")) != m:       # any manifest change means a fresh download
            return False
        return all((d / n).is_file() and hmac.compare_digest(MF.sha256_file(d / n), h) for n, h in _files_of(m).items())
    except (OSError, ValueError):
        return False


def pull(blobs, dest: str | Path, tasks: list[str] | None = None, prefix: str = PREFIX) -> dict[str, str]:
    """blobs: an object with list(prefix) and get(key) (evade.storage.cloudflare.R2Blobs). Returns task -> outcome."""
    dest = Path(dest).expanduser()
    dest.mkdir(parents=True, exist_ok=True)
    found = {}
    for k in blobs.list(prefix):
        parts = k[len(prefix):].split("/")
        if len(parts) == 2 and parts[1] == MF.MANIFEST and TASK_RE.match(parts[0]):
            found[parts[0]] = k
    result = {}
    for task in sorted(found):
        if tasks and task not in tasks:
            continue
        try:
            raw = blobs.get(found[task])
            m = MF.validate(json.loads(raw))
            files = _files_of(m)
            target = dest / task
            if _local_ok(target, m):
                result[task] = "up_to_date"
                continue
            tmp = dest / f".fetch-{task}-{uuid.uuid4().hex[:8]}"
            tmp.mkdir()
            try:
                for name, digest in files.items():
                    data = blobs.get(f"{prefix}{task}/{name}")
                    if data is None:
                        raise MF.ManifestError(f"{name} missing in R2")
                    (tmp / name).write_bytes(data)
                    if not hmac.compare_digest(MF.sha256_file(tmp / name), digest):
                        raise MF.ManifestError(f"sha256 mismatch for {name}")
                (tmp / MF.MANIFEST).write_bytes(raw)
                old = dest / f".old-{task}-{uuid.uuid4().hex[:8]}"
                if target.exists():
                    target.rename(old)
                tmp.rename(target)
                shutil.rmtree(old, ignore_errors=True)
            finally:
                shutil.rmtree(tmp, ignore_errors=True)
            result[task] = "fetched"
        except Exception as e:                       # noqa: BLE001 - report per task, keep going
            log.warning("model %s not fetched: %s", task, e)
            result[task] = f"failed: {e}"
    return result


def push(blobs, task_dir: str | Path, prefix: str = PREFIX) -> str:
    """Upload a local servable folder (hashes checked first). blobs needs put(key, data, content_type)."""
    d = Path(task_dir)
    m = MF.validate(json.loads((d / MF.MANIFEST).read_text(encoding="utf-8")))
    if not TASK_RE.match(d.name):
        raise MF.ManifestError("folder name must match [a-z0-9_]{1,64}")
    files = _files_of(m)
    for name, digest in files.items():
        if not hmac.compare_digest(MF.sha256_file(d / name), digest):
            raise MF.ManifestError(f"local sha256 mismatch for {name}; re-run evade.models.export")
    for name in files:
        blobs.put(f"{prefix}{d.name}/{name}", (d / name).read_bytes(), "application/octet-stream")
    blobs.put(f"{prefix}{d.name}/{MF.MANIFEST}", (d / MF.MANIFEST).read_bytes(), "application/json")
    return f"{prefix}{d.name}/"
