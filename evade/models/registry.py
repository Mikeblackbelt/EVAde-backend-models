"""Model registry: scan EVADE_MODEL_DIR, validate manifests, verify sha256, load by format.

Every folder <EVADE_MODEL_DIR>/<task>/ with a manifest.json is a candidate. A folder that fails any check is
listed as unavailable with a reason; nothing here raises at service start. Draft models load only with
EVADE_ALLOW_DRAFT=1. The beta folders seizure/ and ecg/ from before the manifest contract (model.joblib, no
manifest.json) still load, with a manifest synthesized from MODEL_REGISTRY.md (EVA-2, EVA-6).

Safety: joblib and pickle files run code when loaded. EVADE_MODEL_DIR must hold only artifacts built by
evade.training on a machine we control (or fetched from our own R2 bucket with scripts/fetch_models.py);
the sha256 check catches corruption and accidental swaps, not a compromised source.
"""
from __future__ import annotations

import hmac
import importlib.util
import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from . import manifest as MF

log = logging.getLogger("evade.models")

RUNTIME_MODULE = {"joblib": "joblib", "onnx": "onnxruntime", "torchscript": "torch"}

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


def runtime_available(fmt: str) -> bool:
    mod = RUNTIME_MODULE.get(fmt)
    return bool(mod) and importlib.util.find_spec(mod) is not None


def _load_joblib(path: Path) -> Any:
    import joblib
    # joblib unpickles code: only artifacts built by evade.training on a machine we control (see module docstring)
    return joblib.load(path)


def _load_onnx(path: Path) -> Any:
    import onnxruntime as ort
    return ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])


def _load_torchscript(path: Path) -> Any:
    import torch
    m = torch.jit.load(str(path), map_location="cpu")
    m.eval()
    return m


LOADERS: dict[str, Callable[[Path], Any]] = {"joblib": _load_joblib, "onnx": _load_onnx, "torchscript": _load_torchscript}


@dataclass
class ModelEntry:
    task: str
    path: Path
    manifest: dict | None = None
    available: bool = False
    reason: str | None = None
    legacy: bool = False
    sha256_verified: bool = False
    model: Any = None
    features: list[str] | None = None
    metrics: dict = field(default_factory=dict)

    def as_served(self) -> dict:
        """The dict shape the API keeps in STATE.models[task]."""
        m = self.manifest or {}
        return {"model": self.model, "features": self.features, "metrics": self.metrics, "model_id": m.get("model_id"),
                "status": m.get("status", "draft"), "format": m.get("format"), "manifest": m}

    def public(self) -> dict:
        m = self.manifest or {}
        return {"task": self.task, "model_id": m.get("model_id"), "status": m.get("status"), "format": m.get("format"),
                "available": self.available, "reason": self.reason, "legacy": self.legacy,
                "sha256_verified": self.sha256_verified, "input": m.get("input"), "output": m.get("output"),
                "code_commit": m.get("code_commit"), "trained_at": m.get("trained_at"), "data": m.get("data")}


def _read_json(p: Path) -> Any:
    return json.loads(p.read_text(encoding="utf-8"))


def _inside(folder: Path, name: str) -> Path:
    p = (folder / MF.safe_name(name, "file")).resolve()
    if p.parent != folder.resolve():
        raise MF.ManifestError(f"{name} resolves outside the model folder")
    return p


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
                except Exception as ex:                  # noqa: BLE001 - one bad folder must not stop the service
                    e.available, e.model = False, None
                    e.reason = e.reason or f"{type(ex).__name__}: {ex}"
                if not e.available:
                    log.warning("model folder %s unavailable: %s", d.name, e.reason)
                self.entries[d.name] = e
        return self.entries

    def _manifest_for(self, e: ModelEntry) -> dict:
        mp = e.path / MF.MANIFEST
        if mp.exists():
            m = _read_json(mp)
            if isinstance(m, dict) and m.get("model_id") == MF.UNASSIGNED:
                e.manifest = m
                raise MF.ManifestError("model_id not assigned: add a MODEL_REGISTRY.md row, then run "
                                       "python -m evade.models.export --dir <folder> --model-id EVA-<n>")
            return MF.validate(m)
        legacy = LEGACY[e.task]
        e.legacy = True
        w = e.path / "model.joblib"
        return {"schema_version": MF.SCHEMA_VERSION, "task": e.task, "status": "draft", "format": "joblib",
                "weights": "model.joblib", "sha256": MF.sha256_file(w), "code_commit": "unknown (pre-manifest beta)",
                "trained_at": None, **legacy}

    def _load(self, e: ModelEntry) -> None:
        m = self._manifest_for(e)
        e.manifest = m
        if m["task"] != e.task:
            log.warning("model folder %s has manifest task %s; served under the folder name", e.task, m["task"])
        if m["status"] == "draft" and not self.allow_draft:
            e.reason = "draft model; set EVADE_ALLOW_DRAFT=1 to serve it"
            return
        if not runtime_available(m["format"]):
            e.reason = f"runtime for format {m['format']} not installed ({RUNTIME_MODULE[m['format']]})"
            return
        w = _inside(e.path, m["weights"])
        if not w.is_file():
            e.reason = f"weights file {m['weights']} missing"
            return
        if not e.legacy:                                   # verify every hash before anything is deserialized
            if not hmac.compare_digest(MF.sha256_file(w), m["sha256"]):
                e.reason = "sha256 mismatch for the weights file; refusing to load"
                return
            for name, digest in (m.get("files") or {}).items():
                f = _inside(e.path, name)
                if not f.is_file() or not hmac.compare_digest(MF.sha256_file(f), digest):
                    e.reason = f"sha256 mismatch or missing file: {name}"
                    return
            e.sha256_verified = True
        inp = m["input"]
        if inp.get("kind") == "features":
            ref = inp.get("features") or ("feature_names.json" if (e.path / "feature_names.json").is_file() else None)
            e.features = _read_json(_inside(e.path, ref)) if ref else None
        if MF.missing_recommended(m) and not e.legacy:
            log.warning("model folder %s manifest lacks %s", e.task, ", ".join(MF.missing_recommended(m)))
        mp = e.path / "metrics.json"
        e.metrics = _read_json(mp) if mp.is_file() else {}
        e.model = LOADERS[m["format"]](w)
        e.available, e.reason = True, None

    def served(self) -> dict[str, dict]:
        return {t: e.as_served() for t, e in self.entries.items() if e.available}

    def listing(self) -> list[dict]:
        return [e.public() for e in self.entries.values()]
