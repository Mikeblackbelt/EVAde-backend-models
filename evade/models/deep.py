"""Loader for the deep models (ecg_deep, sleep_deep, seizure_deep): one folder per task under EVADE_MODEL_DIR.

  <EVADE_MODEL_DIR>/<task>/manifest.json + model.pt | model.ts | model.onnx (+ the side files the manifest hashes)

Each task is in one of three states:
  mock         no folder yet: the endpoint returns placeholder outputs (source "mock")
  unavailable  a folder exists but failed a check (reason_code from registry.REASONS or DEEP_REASONS); still mock
  available    the model loaded: the endpoint returns model outputs (source "model")

Checks, in order (the conventions of registry.py): the manifest validates (lenient, as manifest.normalize), the
model_id is assigned, the draft gate (EVADE_ALLOW_DRAFT=1), the runtime is installed, the task adapter supports the
input and output spec, every file's sha256 matches, and only then is anything deserialized. Each file is read once:
the bytes that are hashed are the bytes that are loaded.

Formats:
  onnx              onnxruntime.InferenceSession(bytes), CPU provider
  torchscript       torch.jit.load(BytesIO)
  torch_state_dict  torch.load(BytesIO, weights_only=True) into the architecture named by manifest["arch"]
  torch             either of the two above, told apart by the archive layout (TorchScript archives carry code/)
torch and onnxruntime are optional: without them the task is unavailable (runtime_missing) and stays on mock.
"""
from __future__ import annotations

import io
import json
import logging
import os
import threading
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np

from . import manifest as MF
from .registry import REASONS, ModelLoadError, _inside, _read, _verify, runtime_available

log = logging.getLogger("evade.models.deep")

DEEP_TASKS = ("ecg_deep", "sleep_deep", "seizure_deep")
WEIGHT_CANDIDATES = (("model.pt", "torch"), ("model.ts", "torchscript"), ("model.onnx", "onnx"))
FORMATS = ("onnx", "torchscript", "torch", "torch_state_dict")
RUNTIME = {"onnx": "onnx", "torchscript": "torchscript", "torch": "torchscript", "torch_state_dict": "torchscript"}

DEEP_REASONS = {
    **REASONS,
    "arch_unknown": "state_dict weights name an architecture this service does not know (manifest field arch)",
    "spec_unsupported": "the manifest's input or output spec is not supported by this endpoint "
                        "(for example an unknown normalization or missing operating points)",
}
MOCK_REASON = "no model folder yet: placeholder outputs"

# architecture builders for state_dict weights: name -> builder(c_in, n_out) returning a torch.nn.Module
ARCHS: dict[str, Callable[[int, int], Any]] = {}


def register_arch(name: str, builder: Callable[[int, int], Any]) -> None:
    ARCHS[name] = builder


def build_arch(name: str, c_in: int, n_out: int):
    if name in ARCHS:
        return ARCHS[name](c_in, n_out)
    try:                                       # the training code's architectures, once that branch is merged
        from ..training.deep.models import build   # type: ignore[import-not-found]
    except ImportError:
        build = None
    if build is not None:
        try:
            return build(name, c_in, n_out)
        except ValueError:
            pass
    raise ModelLoadError("arch_unknown", f"unknown architecture {name!r}")


# ---------------- runners: numpy float32 batch in, numpy float32 out ----------------
class OnnxRunner:
    kind = "onnx"

    def __init__(self, data: bytes):
        import onnxruntime as ort
        self.sess = ort.InferenceSession(data, providers=["CPUExecutionProvider"])
        self.input = self.sess.get_inputs()[0].name

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return np.asarray(self.sess.run(None, {self.input: np.ascontiguousarray(x, np.float32)})[0], np.float32)


class TorchRunner:
    def __init__(self, module, kind: str):
        self.module, self.kind = module, kind
        self.lock = threading.Lock()

    def __call__(self, x: np.ndarray) -> np.ndarray:
        import torch
        with self.lock, torch.inference_mode():
            y = self.module(torch.from_numpy(np.ascontiguousarray(x, np.float32)))
        if isinstance(y, (tuple, list)):
            y = y[0]
        return y.detach().cpu().float().numpy()


def _is_torchscript_archive(data: bytes) -> bool:
    try:
        with zipfile.ZipFile(io.BytesIO(data)) as z:
            return any(n.split("/")[1:2] == ["code"] for n in z.namelist())
    except zipfile.BadZipFile:
        return False


def _load_torchscript(data: bytes) -> TorchRunner:
    import torch
    m = torch.jit.load(io.BytesIO(data), map_location="cpu")
    if not isinstance(m, torch.jit.ScriptModule):
        raise ModelLoadError("unexpected_model_type", "weights did not load as a TorchScript module")
    m.eval()
    return TorchRunner(m, "torchscript")


def _load_state_dict(data: bytes, manifest: dict, c_in: int, n_out: int) -> TorchRunner:
    import torch
    arch = manifest.get("arch")
    if not isinstance(arch, str) or not arch:
        raise ModelLoadError("arch_unknown", "state_dict weights need manifest field arch")
    sd = torch.load(io.BytesIO(data), map_location="cpu", weights_only=True)   # tensors only, no code
    if isinstance(sd, dict) and isinstance(sd.get("state_dict"), dict):
        sd = sd["state_dict"]
    if not (isinstance(sd, dict) and sd and all(isinstance(k, str) and isinstance(v, torch.Tensor) for k, v in sd.items())):
        raise ModelLoadError("unexpected_model_type", "weights are not a state_dict of tensors")
    model = build_arch(arch, c_in, n_out)
    model.load_state_dict(sd, strict=True)
    model.eval()
    return TorchRunner(model, "torch_state_dict")


# ---------------- registry ----------------
@dataclass
class DeepEntry:
    task: str
    path: Path | None = None
    manifest: dict | None = None
    state: str = "mock"                   # mock | unavailable | available
    reason_code: str | None = None
    reason: str | None = None             # detail for logs and tests; never returned by the API
    sha256_verified: bool = False
    runner: Any = None
    spec: dict = field(default_factory=dict)   # what the task adapter extracted from the manifest
    metrics: dict = field(default_factory=dict)

    @property
    def available(self) -> bool:
        return self.state == "available"

    def public(self) -> dict:
        m = self.manifest or {}
        return {"task": self.task, "state": self.state,
                "source": "model" if self.available else "mock",
                "model_id": m.get("model_id") if self.available else None,
                "status": m.get("status") if self.available else None,
                "format": m.get("format") if self.manifest else None,
                "runtime": getattr(self.runner, "kind", None),
                "reason_code": self.reason_code,
                "reason": DEEP_REASONS.get(self.reason_code) if self.reason_code else (
                    MOCK_REASON if self.state == "mock" else None),
                "sha256_verified": self.sha256_verified,
                "input": m.get("input") if self.manifest else None,
                "output": m.get("output") if self.manifest else None,
                "headline": m.get("headline") if self.manifest else None,
                "code_commit": m.get("code_commit"), "trained_at": m.get("trained_at")}


def _read_manifest(folder: Path) -> dict:
    m = json.loads((folder / MF.MANIFEST).read_bytes().decode("utf-8"))
    if not isinstance(m, dict):
        raise MF.ManifestError("manifest must be a JSON object")
    m = MF.normalize(m)
    if "weights" not in m:                       # a dropped-in model.pt with a minimal manifest
        for name, fmt in WEIGHT_CANDIDATES:
            if (folder / name).is_file():
                m["weights"] = name
                m.setdefault("format", fmt)
                break
    if "files" not in m and isinstance(m.get("files_sha256"), dict):   # deep export writes files_sha256
        m["files"] = {k: v for k, v in m["files_sha256"].items() if k != m.get("weights")}
    return m


def _validate(m: dict) -> dict:
    """manifest.validate with the deep formats mapped onto the contract's (manifest.py stays unchanged)."""
    fmt = m.get("format")
    if fmt not in FORMATS:
        raise MF.ManifestError(f"format must be one of {FORMATS}")
    v = MF.validate({**m, "format": "onnx" if fmt == "onnx" else "torchscript"}, allow_unassigned=True)
    v["format"] = fmt
    return v


class DeepModelRegistry:
    """Scans only the deep task folders. Built off to the side and swapped in whole, so a reload never exposes a
    half-scanned registry to a request that is running."""

    def __init__(self, root: str | Path | None, allow_draft: bool = False, tasks=DEEP_TASKS,
                 adapters: dict | None = None):
        self.root = Path(root).expanduser() if root else None
        self.allow_draft = allow_draft
        self.tasks = tuple(tasks)
        if adapters is None:
            from .deep_tasks import ADAPTERS
            adapters = ADAPTERS
        self.adapters = adapters
        self.entries: dict[str, DeepEntry] = {t: DeepEntry(task=t) for t in self.tasks}

    @classmethod
    def from_env(cls) -> "DeepModelRegistry":
        return cls(os.environ.get("EVADE_MODEL_DIR"), os.environ.get("EVADE_ALLOW_DRAFT") == "1").scan()

    def scan(self) -> "DeepModelRegistry":
        self.entries = {}
        for t in self.tasks:
            e = DeepEntry(task=t)
            folder = self.root / t if self.root else None
            if folder is not None and folder.is_dir() and (folder / MF.MANIFEST).is_file():
                e.path = folder
                try:
                    self._load(e)
                except ModelLoadError as ex:
                    e.state, e.reason_code, e.reason = "unavailable", ex.code, str(ex)
                except (MF.ManifestError, ValueError, KeyError, TypeError) as ex:
                    e.state, e.reason_code, e.reason = "unavailable", "manifest_invalid", f"{type(ex).__name__}: {ex}"
                except Exception as ex:          # noqa: BLE001 - one bad folder must not stop the service
                    e.state, e.reason_code, e.reason = "unavailable", "load_failed", f"{type(ex).__name__}: {ex}"
                if e.state != "available":
                    e.runner = None
                    log.warning("deep model folder %s unavailable (%s): %s", t, e.reason_code, e.reason)
                else:
                    log.info("deep model %s loaded (%s, %s)", t, e.manifest.get("model_id"), e.runner.kind)
            self.entries[t] = e
        return self

    def _load(self, e: DeepEntry) -> None:
        m = _validate(_read_manifest(e.path))       # an invalid manifest is never kept or shown by the API
        e.manifest = m
        if m["model_id"] == MF.UNASSIGNED:
            raise ModelLoadError("model_id_unassigned", "model_id not assigned: add a MODEL_REGISTRY.md row, then run "
                                 "python -m evade.models.export --dir <folder> --model-id EVA-<n>")
        if m["status"] == "draft" and not self.allow_draft:
            raise ModelLoadError("draft_not_allowed", "draft model; set EVADE_ALLOW_DRAFT=1 to serve it")
        if not runtime_available(RUNTIME[m["format"]]):
            raise ModelLoadError("runtime_missing", f"runtime for format {m['format']} not installed")
        # read every file once and verify it before anything is deserialized
        weights = _read(e.path, m["weights"])
        _verify(weights, m["sha256"], "the weights file")
        side: dict[str, bytes] = {}
        for name, digest in (m.get("files") or {}).items():
            side[name] = _read(e.path, name)
            _verify(side[name], digest, name)
        e.sha256_verified = True

        def sidecar(name: str) -> bytes | None:
            if name in side:
                return side[name]
            p = _inside(e.path, name)
            return p.read_bytes() if p.is_file() else None

        raw = sidecar("metrics.json")
        e.metrics = json.loads(raw.decode("utf-8")) if raw is not None else {}
        adapter = self.adapters[e.task]
        e.spec = adapter.spec_from_manifest(m, sidecar)   # raises ModelLoadError("spec_unsupported", ...)
        fmt = m["format"]
        if fmt == "onnx":
            e.runner = OnnxRunner(weights)
        elif fmt == "torchscript" or (fmt == "torch" and _is_torchscript_archive(weights)):
            e.runner = _load_torchscript(weights)
        else:
            e.runner = _load_state_dict(weights, m, e.spec["c_in"], e.spec["n_out"])
        adapter.smoke(e.spec, e.runner)              # one zero batch: output shape must match the spec
        e.state, e.reason, e.reason_code = "available", None, None

    def get(self, task: str) -> DeepEntry:
        return self.entries.get(task) or DeepEntry(task=task)

    def listing(self) -> list[dict]:
        return [self.get(t).public() for t in self.tasks]
