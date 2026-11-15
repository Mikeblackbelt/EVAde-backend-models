"""Sync servable model folders between Cloudflare R2 and EVADE_MODEL_DIR, verifying sha256 on every file.

R2 layout: models/<task>/manifest.json plus the files it names (weights, feature_names.json, metrics.json), in the
models bucket (CF_R2_MODELS_BUCKET; see evade/models/README.md). Servers pull with a read-only key and pull() only
ever sees a read-only view; push() needs the write key, which only training machines hold.

Pull downloads into a temporary folder, checks every hash, then swaps the folder in; a failed check leaves the
current folder untouched, and a failed swap puts the old folder back. A folder whose manifest (compared as parsed
JSON) and file hashes already match is left alone. Push uploads the files first and manifest.json last, so a reader
never sees a manifest whose files are missing. The hashes come from the same bucket, so they catch corruption and
partial uploads, not a compromised bucket.
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


class ReadOnlyBlobs:
    """list/get only: what a serving machine may do with the models bucket."""

    def __init__(self, blobs):
        self._b = blobs

    def __repr__(self) -> str:
        return f"ReadOnlyBlobs({self._b!r})"

    def list(self, prefix: str) -> list[str]:
        return self._b.list(prefix)

    def get(self, key: str) -> bytes | None:
        return self._b.get(key)


def _files_of(m: dict) -> dict[str, str]:
    """Every file a manifest names, with its expected sha256 (weights first)."""
    out = {MF.safe_name(m["weights"], "weights"): m["sha256"]}
    for name, digest in (m.get("files") or {}).items():
        out[MF.safe_name(name, "files key")] = digest
    ref = m["input"].get("features")
    if ref and ref not in out:
        raise MF.ManifestError(f"input.features file {ref} has no sha256 in manifest.files")
    return out


def _local_ok(d: Path, remote_doc: object, m: dict) -> bool:
    """remote_doc: the remote manifest as parsed JSON (not normalized), compared with the local file parsed the same way."""
    mp = d / MF.MANIFEST
    if not mp.is_file():
        return False
    try:
        if json.loads(mp.read_text(encoding="utf-8")) != remote_doc:
            return False
        return all((d / n).is_file() and hmac.compare_digest(MF.sha256_file(d / n), h) for n, h in _files_of(m).items())
    except (OSError, ValueError):
        return False


def _swap(tmp: Path, target: Path, dest: Path, task: str) -> None:
    """Replace target with tmp; if the second rename fails, the old folder is renamed back."""
    old = dest / f".old-{task}-{uuid.uuid4().hex[:8]}"
    had_old = target.exists()
    if had_old:
        target.rename(old)
    try:
        tmp.rename(target)
    except Exception:
        if had_old:
            old.rename(target)
        raise
    if had_old:
        shutil.rmtree(old, ignore_errors=True)


def pull(blobs, dest: str | Path, tasks: list[str] | None = None, prefix: str = PREFIX) -> dict[str, str]:
    """blobs: an object with list(prefix) and get(key) (evade.storage.cloudflare.R2Blobs); only those two are used.
    Returns task -> outcome."""
    blobs = ReadOnlyBlobs(blobs)
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
            doc = json.loads(raw)
            m = MF.validate(doc)
            files = _files_of(m)
            target = dest / task
            if _local_ok(target, doc, m):
                result[task] = "up_to_date"
                continue
            tmp = dest / f".fetch-{task}-{uuid.uuid4().hex[:8]}"
            tmp.mkdir()
            try:
                for name, digest in files.items():
                    data = blobs.get(f"{prefix}{task}/{name}")
                    if data is None:
                        raise MF.ManifestError(f"{name} missing in R2")
                    if not hmac.compare_digest(MF.sha256_bytes(data), digest):
                        raise MF.ManifestError(f"sha256 mismatch for {name}")
                    (tmp / name).write_bytes(data)
                (tmp / MF.MANIFEST).write_bytes(raw)
                _swap(tmp, target, dest, task)
            finally:
                shutil.rmtree(tmp, ignore_errors=True)
            result[task] = "fetched"
        except Exception as e:                       # noqa: BLE001 - report per task, keep going
            log.warning("model %s not fetched: %s", task, e)
            result[task] = f"failed: {e}"
    return result


def push(blobs, task_dir: str | Path, prefix: str = PREFIX) -> str:
    """Upload a local servable folder (hashes checked first). blobs needs put(key, data, content_type): use the
    models bucket's write key (training machines only)."""
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
