"""Model artifact contract: <EVADE_MODEL_DIR>/<task>/manifest.json (see evade/models/README.md).

Pure-stdlib validation and hashing, shared by the registry (serving), export (training) and fetch (R2 sync).
"""
from __future__ import annotations

import hashlib
import re
from pathlib import Path

SCHEMA_VERSION = 1
MANIFEST = "manifest.json"
UNASSIGNED = "UNASSIGNED"           # model_id written by a training run that was not given --model-id
STATUSES = ("draft", "validated")
FORMATS = ("joblib", "onnx", "torchscript")
INPUT_KINDS = ("features", "tensor")
OUTPUT_TYPES = ("binary", "multiclass", "multilabel", "regression")
MODEL_ID_RE = re.compile(r"^EVA-[1-9][0-9]*$")
SHA_RE = re.compile(r"^[0-9a-f]{64}$")
REQUIRED = ("model_id", "task", "status", "format", "weights", "sha256", "input", "output")
RECOMMENDED = ("code_commit", "trained_at", "data")     # missing ones are reported, not fatal
ALIASES = {"code_commit": ("commit",), "data": ("data_used",)}
FEATURE_REF_KEYS = ("features", "feature_names", "feature_list")


class ManifestError(ValueError):
    pass


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def safe_name(name: object, field: str) -> str:
    """A bare file name: no directories, no traversal, no absolute paths."""
    if not isinstance(name, str) or not name or name in (".", "..") or "/" in name or "\\" in name \
            or ":" in name or name.startswith(".") or Path(name).name != name:
        raise ManifestError(f"{field} must be a bare file name inside the model folder")
    return name


def normalize(m: dict) -> dict:
    """Lenient reading of hand-written manifests: field aliases, input.kind inferred from tensor_shape vs a feature
    list reference, output.labels as an alias of output.classes. Returns a new dict; the input is not changed."""
    m = dict(m)
    for k, alts in ALIASES.items():
        if k not in m:
            for a in alts:
                if a in m:
                    m[k] = m[a]
                    break
    if isinstance(m.get("input"), dict):
        inp = dict(m["input"])
        if "features" not in inp:
            for a in FEATURE_REF_KEYS[1:]:
                if isinstance(inp.get(a), str):
                    inp["features"] = inp[a]
                    break
        if "kind" not in inp:
            inp["kind"] = "tensor" if inp.get("tensor_shape") else "features"
        m["input"] = inp
    if isinstance(m.get("output"), dict):
        out = dict(m["output"])
        if "classes" not in out and isinstance(out.get("labels"), list):
            out["classes"] = out["labels"]
        m["output"] = out
    return m


def missing_recommended(m: dict) -> list[str]:
    return [k for k in RECOMMENDED if m.get(k) in (None, "")]


def validate(m: dict, *, allow_unassigned: bool = False) -> dict:
    """Check a manifest dict; return the normalized manifest or raise ManifestError naming the first problem."""
    if not isinstance(m, dict):
        raise ManifestError("manifest must be a JSON object")
    m = normalize(m)
    missing = [k for k in REQUIRED if k not in m]
    if missing:
        raise ManifestError(f"missing fields: {', '.join(missing)}")
    mid = m["model_id"]
    if not (isinstance(mid, str) and (MODEL_ID_RE.match(mid) or (allow_unassigned and mid == UNASSIGNED))):
        raise ManifestError("model_id must be EVA-<n> (a MODEL_REGISTRY.md row)")
    if not isinstance(m["task"], str) or not re.match(r"^[a-z0-9_]{1,64}$", m["task"]):
        raise ManifestError("task must match [a-z0-9_]{1,64}")
    if m["status"] not in STATUSES:
        raise ManifestError(f"status must be one of {STATUSES}")
    if m["format"] not in FORMATS:
        raise ManifestError(f"format must be one of {FORMATS}")
    safe_name(m["weights"], "weights")
    if not isinstance(m["sha256"], str) or not SHA_RE.match(m["sha256"]):
        raise ManifestError("sha256 must be 64 lowercase hex characters")
    for name, digest in (m.get("files") or {}).items():
        safe_name(name, "files key")
        if not isinstance(digest, str) or not SHA_RE.match(digest):
            raise ManifestError(f"files[{name}] must be a sha256 hex digest")
    inp, out = m["input"], m["output"]
    if not isinstance(inp, dict) or inp.get("kind") not in INPUT_KINDS:
        raise ManifestError(f"input.kind must be one of {INPUT_KINDS}")
    if not isinstance(inp.get("modality"), str):
        raise ManifestError("input.modality is required")
    if inp["kind"] == "features":
        if inp.get("features") is not None:          # absent: the registry uses feature_names.json if present
            safe_name(inp["features"], "input.features")
    elif not (isinstance(inp.get("tensor_shape"), list) and inp["tensor_shape"]):
        raise ManifestError("input.tensor_shape is required for kind=tensor")
    if not isinstance(out, dict):
        raise ManifestError("output must be an object")
    if out.get("type") is not None and out["type"] not in OUTPUT_TYPES:
        raise ManifestError(f"output.type must be one of {OUTPUT_TYPES}")
    if out.get("classes") is not None and not isinstance(out["classes"], list):
        raise ManifestError("output.classes (or output.labels) must be a list")
    return m
