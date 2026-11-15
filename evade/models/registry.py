"""Model registry: scan EVADE_MODEL_DIR, validate manifests, verify sha256, load by format.

Every folder <EVADE_MODEL_DIR>/<task>/ with a manifest.json is a candidate. A folder that fails any check is
listed as unavailable with a generic reason code (the detail goes to the log only); nothing here raises at service
start. Draft models load only with EVADE_ALLOW_DRAFT=1. The beta folders seizure/ and ecg/ from before the manifest
contract (model.joblib, no manifest.json) still load, with a manifest synthesized from MODEL_REGISTRY.md (EVA-2,
EVA-6). EVADE_LEGACY_SHA256_SEIZURE / EVADE_LEGACY_SHA256_ECG pins the weights' sha256 (a mismatch refuses to load);
they are served marked "unverified" (re-stamp them with evade.models.export to give them a manifest).

sha256_verified means every file the loader read was hashed: the weights and each file in manifest.files. A sidecar
it reads that is not in manifest.files (feature_names.json, metrics.json) marks the entry unverified.

Every file is read once: the bytes that are hashed are the bytes that are deserialized (no window between the check
and the load), and the loaded object must look like a model (predict / predict_proba, a dict of such, an ONNX
session or a TorchScript module).

Safety: joblib and pickle files run code when loaded. EVADE_MODEL_DIR must hold only artifacts built by
evade.training on a machine we control (or fetched from our own R2 models bucket with scripts/fetch_models.py);
the sha256 check catches corruption and accidental swaps, not a compromised source, and the type check runs after
unpickling, so it is a sanity check, not a defence.
"""
from __future__ import annotations

import hmac
import importlib.util
import io
import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from . import manifest as MF

log = logging.getLogger("evade.models")

RUNTIME_MODULE = {"joblib": "joblib", "onnx": "onnxruntime", "torchscript": "torch"}

# generic, path-free reasons for GET /v1/models; the detailed reason is logged at scan time
REASONS = {
    "manifest_invalid": "manifest.json is unreadable or does not follow the contract (see evade/models/README.md)",
    "model_id_unassigned": "model_id not assigned: add a MODEL_REGISTRY.md row and re-stamp with evade.models.export",
    "draft_not_allowed": "draft model; set EVADE_ALLOW_DRAFT=1 to serve it",
    "runtime_missing": "the runtime for this model format is not installed",
    "file_missing": "a file named by the manifest is missing",
    "sha256_mismatch": "sha256 mismatch; refusing to load",
    "unexpected_model_type": "the file did not load as a supported model type",
    "load_failed": "the model could not be loaded",
}

# pre-contract beta folders: task -> manifest fields that MODEL_REGISTRY.md already fixes
LEGACY = {
    "seizure": {"model_id": "EVA-2", "input": {"kind": "features", "modality": "eeg", "fs_hz": 250.0, "window_s": 4.0,
                                               "step_s": 2.0, "montage": "18-channel bipolar double banana",
                                               "features": "feature_names.json", "normalization": "per_user_robust"},
                "output": {"type": "binary", "classes": ["seizure_suspect"],
                           "operating_points_ref": "metrics.json leave_one_subject_out_chbmit.hgb.operating_points"},
                "data": "CHB-MIT chb01-05, Siena 6 subjects (see MODEL_REGISTRY.md)"},
    "ecg": {"model_id": "EVA-6", "input": {"kind": "features", "modality": "ecg", "fs_hz": 100.0, "window_s": 10.0,
                                           "leads": ["I", "II", "III", "AVR", "AVL", "AVF", "V1", "V2", "V3", "V4", "V5", "V6"],
                                           "features": "feature_names.json"},
            "output": {"type": "multilabel", "classes": ["NORM", "MI", "STTC", "CD", "HYP"],
                       "estimator": "dict of one-vs-rest binary classifiers"},
            "data": "PTB-XL 1.0.3 (see MODEL_REGISTRY.md)"},
}


class ModelLoadError(Exception):
    def __init__(self, code: str, detail: str):
        super().__init__(detail)
        self.code = code


def legacy_pin_var(task: str) -> str:
    return f"EVADE_LEGACY_SHA256_{task.upper()}"


def runtime_available(fmt: str) -> bool:
    mod = RUNTIME_MODULE.get(fmt)
    return bool(mod) and importlib.util.find_spec(mod) is not None


def _load_joblib(data: bytes) -> Any:
    import joblib
    # joblib unpickles code: only artifacts built by evade.training on a machine we control (see module docstring)
    return joblib.load(io.BytesIO(data))


def _load_onnx(data: bytes) -> Any:
    import onnxruntime as ort
    return ort.InferenceSession(data, providers=["CPUExecutionProvider"])


def _load_torchscript(data: bytes) -> Any:
    import torch
    m = torch.jit.load(io.BytesIO(data), map_location="cpu")
    m.eval()
    return m


LOADERS: dict[str, Callable[[bytes], Any]] = {"joblib": _load_joblib, "onnx": _load_onnx,
                                              "torchscript": _load_torchscript}


def _is_estimator(o: Any) -> bool:
    return callable(getattr(o, "predict_proba", None)) or callable(getattr(o, "predict", None))


def model_type_ok(fmt: str, obj: Any) -> bool:
    if fmt == "joblib":
        return _is_estimator(obj) or (isinstance(obj, dict) and bool(obj) and all(_is_estimator(v) for v in obj.values()))
    if fmt == "onnx":
        import onnxruntime as ort
        return isinstance(obj, ort.InferenceSession)
    if fmt == "torchscript":
        import torch
        return isinstance(obj, torch.jit.ScriptModule)
    return False


@dataclass
class ModelEntry:
    task: str
    path: Path
    manifest: dict | None = None
    available: bool = False
    reason: str | None = None            # detail for logs and tests; never returned by the API
    reason_code: str | None = None       # generic code, returned by the API with REASONS[code]
    legacy: bool = False
    sha256_verified: bool = False
    model: Any = None
    features: list[str] | None = None
    metrics: dict = field(default_factory=dict)

    def as_served(self) -> dict:
        """The dict shape the API keeps in STATE.models[task]."""
        m = self.manifest or {}
        return {"model": self.model, "features": self.features, "metrics": self.metrics, "model_id": m.get("model_id"),
                "status": m.get("status", "draft"), "format": m.get("format"), "manifest": m, "legacy": self.legacy,
                "sha256_verified": self.sha256_verified}

    def public(self) -> dict:
        m = self.manifest or {}
        return {"task": self.task, "model_id": m.get("model_id"), "status": m.get("status"), "format": m.get("format"),
                "available": self.available, "reason_code": self.reason_code,
                "reason": REASONS.get(self.reason_code) if self.reason_code else None, "legacy": self.legacy,
                "sha256_verified": self.sha256_verified,
                "integrity": "sha256_verified" if self.sha256_verified else "unverified",
                "input": m.get("input"), "output": m.get("output"), "code_commit": m.get("code_commit"),
                "trained_at": m.get("trained_at"), "data": m.get("data")}


def _inside(folder: Path, name: str) -> Path:
    p = (folder / MF.safe_name(name, "file")).resolve()
    if p.parent != folder.resolve():
        raise MF.ManifestError(f"{name} resolves outside the model folder")
    return p


def _read(folder: Path, name: str) -> bytes:
    p = _inside(folder, name)
    if not p.is_file():
        raise ModelLoadError("file_missing", f"{name} missing")
    return p.read_bytes()


def _verify(data: bytes, digest: str, name: str) -> None:
    if not (isinstance(digest, str) and MF.SHA_RE.match(digest)) or \
            not hmac.compare_digest(MF.sha256_bytes(data), digest):
        raise ModelLoadError("sha256_mismatch", f"sha256 mismatch for {name}; refusing to load")


class ModelRegistry:
    def __init__(self, root: str | Path | None, allow_draft: bool = False):
        self.root = Path(root).expanduser() if root else None
        self.allow_draft = allow_draft
        self.entries: dict[str, ModelEntry] = {}

    @classmethod
    def from_env(cls) -> "ModelRegistry":
        r = cls(os.environ.get("EVADE_MODEL_DIR"), os.environ.get("EVADE_ALLOW_DRAFT") == "1")
        r.scan()
        return r

    def scan(self) -> dict[str, ModelEntry]:
        self.entries = {}
        if not self.root or not self.root.is_dir():
            return self.entries
        for d in sorted(p for p in self.root.iterdir() if p.is_dir() and not p.name.startswith(".")):
            if (d / MF.MANIFEST).exists() or (d.name in LEGACY and (d / "model.joblib").exists()):
                e = ModelEntry(task=d.name, path=d)
                try:
                    self._load(e)
                except ModelLoadError as ex:
                    e.available, e.model, e.reason_code, e.reason = False, None, ex.code, str(ex)
                except (MF.ManifestError, ValueError) as ex:     # includes JSON decoding errors
                    e.available, e.model = False, None
                    e.reason_code, e.reason = "manifest_invalid", f"{type(ex).__name__}: {ex}"
                except Exception as ex:                  # noqa: BLE001 - one bad folder must not stop the service
                    e.available, e.model = False, None
                    e.reason_code, e.reason = "load_failed", f"{type(ex).__name__}: {ex}"
                if not e.available:
                    log.warning("model folder %s unavailable (%s): %s", d.name, e.reason_code, e.reason)
                elif e.legacy and not e.sha256_verified:
                    log.warning("model folder %s is a legacy folder without a pinned sha256 (%s): served as unverified",
                                d.name, legacy_pin_var(d.name))
                self.entries[d.name] = e
        return self.entries

    def _manifest_for(self, e: ModelEntry) -> dict:
        mp = e.path / MF.MANIFEST
        if mp.exists():
            m = json.loads(mp.read_bytes().decode("utf-8"))
            if isinstance(m, dict) and m.get("model_id") == MF.UNASSIGNED:
                e.manifest = m
                raise ModelLoadError("model_id_unassigned",
                                     "model_id not assigned: add a MODEL_REGISTRY.md row, then run "
                                     "python -m evade.models.export --dir <folder> --model-id EVA-<n>")
            return MF.validate(m)
        e.legacy = True
        return {"schema_version": MF.SCHEMA_VERSION, "task": e.task, "status": "draft", "format": "joblib",
                "weights": "model.joblib", "sha256": None, "code_commit": "unknown (pre-manifest beta)",
                "trained_at": None, **LEGACY[e.task]}

    def _load(self, e: ModelEntry) -> None:
        m = self._manifest_for(e)
        e.manifest = m
        if m["task"] != e.task:
            log.warning("model folder %s has manifest task %s; served under the folder name", e.task, m["task"])
        if m["status"] == "draft" and not self.allow_draft:
            raise ModelLoadError("draft_not_allowed", "draft model; set EVADE_ALLOW_DRAFT=1 to serve it")
        if not runtime_available(m["format"]):
            raise ModelLoadError("runtime_missing",
                                 f"runtime for format {m['format']} not installed ({RUNTIME_MODULE[m['format']]})")
        # read every file once; verify the bytes before anything is deserialized, then load those same bytes
        weights = _read(e.path, m["weights"])
        side: dict[str, bytes] = {}
        if not e.legacy:
            _verify(weights, m["sha256"], "the weights file")
            for name, digest in (m.get("files") or {}).items():
                side[name] = _read(e.path, name)
                _verify(side[name], digest, name)
            e.sha256_verified = True
        else:
            pin = os.environ.get(legacy_pin_var(e.task), "").strip().lower()
            m["sha256"] = MF.sha256_bytes(weights)
            if pin:
                _verify(weights, pin, f"the weights file (pinned by {legacy_pin_var(e.task)})")
                e.sha256_verified = True

        unhashed: list[str] = []

        def sidecar(name: str) -> bytes | None:
            """A file the loader reads. Only files hashed in manifest.files (or nothing at all) keep the entry
            verified: reading any other file marks it unverified."""
            if name in side:
                return side[name]
            p = _inside(e.path, name)
            if not p.is_file():
                return None
            unhashed.append(name)
            return p.read_bytes()

        inp = m["input"]
        if inp.get("kind") == "features":
            ref = inp.get("features") or "feature_names.json"
            raw = sidecar(ref)
            if raw is None and inp.get("features"):
                raise ModelLoadError("file_missing", f"{ref} missing")
            e.features = json.loads(raw.decode("utf-8")) if raw is not None else None
        if MF.missing_recommended(m) and not e.legacy:
            log.warning("model folder %s manifest lacks %s", e.task, ", ".join(MF.missing_recommended(m)))
        raw = sidecar("metrics.json")
        e.metrics = json.loads(raw.decode("utf-8")) if raw is not None else {}
        if unhashed and e.sha256_verified:
            e.sha256_verified = False
            log.warning("model folder %s: %s not hashed in manifest.files; served as unverified (re-stamp with "
                        "evade.models.export)", e.task, ", ".join(sorted(set(unhashed))))
        model = LOADERS[m["format"]](weights)
        if not model_type_ok(m["format"], model):
            raise ModelLoadError("unexpected_model_type",
                                 f"{m['weights']} loaded as {type(model).__name__}, not a {m['format']} model")
        e.model = model
        e.available, e.reason, e.reason_code = True, None, None

    def served(self) -> dict[str, dict]:
        return {t: e.as_served() for t, e in self.entries.items() if e.available}

    def listing(self) -> list[dict]:
        return [e.public() for e in self.entries.values()]
