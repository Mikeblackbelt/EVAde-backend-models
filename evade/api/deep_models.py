"""Deep model endpoints (ecg_deep, sleep_deep, seizure_deep): placeholder outputs until the weights exist.

Each endpoint answers in mock mode (source "mock", model_id null, a placeholder warning) until its folder
<EVADE_MODEL_DIR>/<task>/ holds a manifest.json and weights that pass every check (evade/models/deep.py); then the
same endpoint answers from the model (source "model"). The response schema is the same in both modes.
Folders are scanned at service start; POST /v1/models/reload rescans the deep folders without a restart.

Safety rules kept here:
  - mock outputs never reach the risk engine, flags, overrides or baselines (feed_shadow_risk refuses them);
  - real seizure outputs enter the neuro domain only as a shadow input, as the beta seizure model does;
  - every request has size limits, and validation messages never echo signal values;
  - nothing here logs request bodies or health data.
See evade/api/README_deep_models.md.
"""
from __future__ import annotations

import math
import threading
from datetime import datetime
from typing import Annotated, Literal

import numpy as np
from fastapi import APIRouter
from pydantic import BaseModel, Field, field_validator, model_validator

from ..models.deep import DEEP_TASKS, DeepEntry, DeepModelRegistry
from ..models.deep_tasks import (ADAPTERS, ECG_FS, ECG_LEADS, EPOCH_S, MOCK_SEIZURE_THRESHOLDS, SEIZURE_CHANNELS,
                                 SLEEP_CHANNELS, SLEEP_FS, ecg_resample_ratio)
from ..preprocessing.eeg import norm_bipolar_name
from .schemas import Source, _not_future

MOCK_WARNING = "placeholder output: model not trained yet; not a prediction"
MAX_ECG_ROW = 10_500                  # samples per lead: 10.5 s at up to 1000 Hz
MAX_SLEEP_EPOCHS = 20                 # the 1 MB body cap allows about 20 epochs of 2 x 3000 samples
MAX_SLEEP_ROW = MAX_SLEEP_EPOCHS * int(EPOCH_S * SLEEP_FS)
MAX_SEIZURE_S = 30.0                  # 18 channels x 30 s at 250 Hz is about the 1 MB body cap
MAX_SEIZURE_ROW = int(MAX_SEIZURE_S * 1000)
QUALITY_MIN_SEIZURE = 0.5             # as /v1/predict/eeg
SHADOW_SCORE = {"suspect_seizure": ("red", 0.70), "review": ("yellow", 0.40), "none": ("green", 0.0)}
NOTE = "for human review, not a diagnosis"

_REG: DeepModelRegistry | None = None
_REG_LOCK = threading.Lock()
_RELOAD_LOCK = threading.Lock()


class MockOutputError(RuntimeError):
    """Raised when anything tries to feed a placeholder output into risk, flags, overrides or baselines."""


def registry() -> DeepModelRegistry:
    global _REG
    with _REG_LOCK:
        if _REG is None:
            _REG = DeepModelRegistry.from_env()
        return _REG


def set_registry(reg: DeepModelRegistry) -> DeepModelRegistry:
    """Swap the whole registry at once (reload and tests); running requests keep the entry they already hold."""
    global _REG
    with _REG_LOCK:
        _REG = reg
    return reg


def deep_listing() -> list[dict]:
    return registry().listing()


# ---------------- requests ----------------
def _finite(samples: list[list[float]]) -> None:
    if any(not math.isfinite(v) for row in samples for v in row):
        raise ValueError("nan_in_signal: samples must be finite numbers")


def _rows_equal(samples: list[list[float]]) -> int:
    n = {len(r) for r in samples}
    if len(n) != 1:
        raise ValueError("bad_shape: every channel needs the same number of samples")
    return n.pop()


def _key_ecg(name: str) -> str:
    return name.strip().upper()


class _Base(BaseModel):
    user_id: str = Field(min_length=1, max_length=128)
    t0_utc: datetime | None = None
    source: Source = "device"

    @field_validator("t0_utc")
    @classmethod
    def _time(cls, v):
        return None if v is None else _not_future(v)


class EcgDeepRequest(_Base):
    fs_hz: float = Field(ge=50, le=1000)
    leads: list[Annotated[str, Field(max_length=8)]] = Field(min_length=12, max_length=12)
    samples: list[Annotated[list[float], Field(max_length=MAX_ECG_ROW)]] = Field(
        min_length=12, max_length=12, description="12 leads x samples, 10 s, millivolts")

    @model_validator(mode="after")
    def _shape(self):
        canon = {_key_ecg(c): i for i, c in enumerate(ECG_LEADS)}
        idx = [canon.get(_key_ecg(c), -1) for c in self.leads]
        if sorted(idx) != list(range(12)):
            raise ValueError("bad_shape: leads must be I, II, III, aVR, aVL, aVF, V1-V6, each once")
        n = _rows_equal(self.samples)
        if not 9.5 <= n / self.fs_hz <= 10.5:
            raise ValueError("bad_shape: the ECG must be 10 s long (9.5-10.5 s accepted)")
        _finite(self.samples)
        return self

    def canonical(self) -> np.ndarray:
        canon = {_key_ecg(c): i for i, c in enumerate(ECG_LEADS)}
        x = np.empty((12, len(self.samples[0])), np.float64)
        for name, row in zip(self.leads, self.samples):
            x[canon[_key_ecg(name)]] = row
        return x


class SleepDeepRequest(_Base):
    fs_hz: float = Field(SLEEP_FS, ge=SLEEP_FS, le=SLEEP_FS, description="must be 100")
    channels: list[Annotated[str, Field(max_length=16)]] = Field(list(SLEEP_CHANNELS), min_length=2, max_length=2)
    samples: list[Annotated[list[float], Field(max_length=MAX_SLEEP_ROW)]] = Field(
        min_length=2, max_length=2, description=f"2 channels x (n x 3000) samples, 1 to {MAX_SLEEP_EPOCHS} epochs")

    @model_validator(mode="after")
    def _shape(self):
        if sorted(c.strip().upper() for c in self.channels) != sorted(c.upper() for c in SLEEP_CHANNELS):
            raise ValueError("bad_shape: channels must be Fpz-Cz and Pz-Oz")
        n = _rows_equal(self.samples)
        t = int(EPOCH_S * SLEEP_FS)
        if n == 0 or n % t:
            raise ValueError(f"bad_shape: samples per channel must be a whole number of 30 s epochs ({t} each)")
        _finite(self.samples)
        return self

    def canonical(self) -> np.ndarray:
        pos = {c.upper(): i for i, c in enumerate(SLEEP_CHANNELS)}
        x = np.empty((2, len(self.samples[0])), np.float64)
        for name, row in zip(self.channels, self.samples):
            x[pos[name.strip().upper()]] = row
        return x


class SeizureDeepRequest(_Base):
    fs_hz: float = Field(ge=100, le=1000)
    mains_hz: Literal[50, 60] = 60
    channels: list[Annotated[str, Field(max_length=16)]] = Field(min_length=18, max_length=18)
    samples: list[Annotated[list[float], Field(max_length=MAX_SEIZURE_ROW)]] = Field(
        min_length=18, max_length=18, description="18 bipolar channels x samples, 4-30 s")

    @model_validator(mode="after")
    def _shape(self):
        names = [norm_bipolar_name(c) for c in self.channels]
        if sorted(names) != sorted(SEIZURE_CHANNELS):
            raise ValueError("bad_shape: channels must be the 18 bipolar double-banana pairs (FP1-F7 ... CZ-PZ), each once")
        n = _rows_equal(self.samples)
        if not 4.0 <= n / self.fs_hz <= MAX_SEIZURE_S:
            raise ValueError(f"bad_shape: the EEG window must be 4-{MAX_SEIZURE_S:.0f} s long")
        _finite(self.samples)
        return self

    def canonical(self) -> np.ndarray:
        pos = {c: i for i, c in enumerate(SEIZURE_CHANNELS)}
        x = np.empty((18, len(self.samples[0])), np.float64)
        for name, row in zip(self.channels, self.samples):
            x[pos[norm_bipolar_name(name)]] = row
        return x


# ---------------- response helpers ----------------
def header(task: str, e: DeepEntry) -> dict:
    m = e.manifest or {}
    real = e.available
    return {"task": task, "source": "model" if real else "mock", "model_id": m.get("model_id") if real else None,
            "model_status": m.get("status") if real else None, "model_state": e.state, "note": NOTE}


def warnings_for(e: DeepEntry) -> list[str]:
    if e.available:
        return ["draft model output"] if (e.manifest or {}).get("status") == "draft" else []
    w = [MOCK_WARNING]
    if e.state == "unavailable":
        w.append(f"model folder present but unavailable ({e.reason_code}); see GET /v1/models")
    return w


def model_ids(e: DeepEntry) -> list[str]:
    return [e.manifest["model_id"]] if e.available else []


def feed_shadow_risk(A, user_id: str, data: dict, warn: list[str], input_source: str = "device") -> None:
    """The only path from a deep output into risk state. Refuses anything that is not a real model output, so a
    placeholder can never raise a flag, change R, set an override or touch a baseline. Simulated or replayed
    inputs are labelled synthetic (as in /v1/compare), which makes a risk result provisional."""
    if data.get("source") != "model" or data.get("model_id") is None:
        raise MockOutputError("placeholder outputs never feed the risk engine")
    if data["task"] != "seizure_deep" or data.get("status") != "ok":
        return                                   # only seizure has a beta shadow precedent (neuro domain)
    color, score = SHADOW_SCORE[data["flag"]]
    comp = {"metric": "seizure_deep_probability", "modality": "eeg", "domain": "neuro",
            "value": data["max_probability"], "level": color, "component_score": score, "shadow": True,
            "confidence": "synthetic" if input_source in ("simulated", "replay")
            else "demo" if data.get("model_status") == "draft" else "ok", "unit": "probability",
            "quality_index": data["quality_index"]}
    flags = A.remember(user_id, [comp])
    A.persist_soft(user_id, warn, state_keys=("latest",), flags=flags, endpoint="predict/seizure-deep")


def _app():
    from . import app as A                       # at request time: app imports this module at startup
    return A


# ---------------- router ----------------
def build_router() -> APIRouter:
    """No per-route auth: app.py's global middleware checks X-EVAde-Key on every path, these included, before any
    route runs. Scans the deep model folders (service start)."""
    set_registry(DeepModelRegistry.from_env())
    r = APIRouter(prefix="/v1", tags=["deep models"])

    @r.post("/predict/ecg-deep")
    def predict_ecg_deep(body: EcgDeepRequest):
        A = _app()
        A.user(body.user_id)
        e = registry().get("ecg_deep")
        try:                                     # same bound in mock mode, so a client never sees it change later
            ecg_resample_ratio(body.fs_hz, e.spec["fs"] if e.available else ECG_FS)
        except ValueError as ex:
            raise A.err(422, "bad_shape", str(ex), "fs_hz") from None
        ad, x = ADAPTERS["ecg_deep"], body.canonical()
        core = ad.predict(e.spec, e.runner, x, body.fs_hz) if e.available else ad.mock(x, body.fs_hz)
        data = {**header("ecg_deep", e), **core,
                "input": {"fs_hz": body.fs_hz, "n_samples": x.shape[1], "leads": list(ECG_LEADS),
                          "model_fs_hz": e.spec.get("fs", 100.0) if e.available else 100.0}}
        return A.envelope(body.user_id, data, model_ids(e), warnings_for(e))

    @r.post("/predict/sleep-deep")
    def predict_sleep_deep(body: SleepDeepRequest):
        A = _app()
        A.user(body.user_id)
        e = registry().get("sleep_deep")
        ad, x = ADAPTERS["sleep_deep"], body.canonical()
        core = ad.predict(e.spec, e.runner, x, body.fs_hz) if e.available else ad.mock(x, body.fs_hz)
        data = {**header("sleep_deep", e), **core,
                "input": {"fs_hz": body.fs_hz, "channels": list(SLEEP_CHANNELS), "epoch_s": EPOCH_S}}
        return A.envelope(body.user_id, data, model_ids(e), warnings_for(e))

    @r.post("/predict/seizure-deep")
    def predict_seizure_deep(body: SeizureDeepRequest):
        A = _app()
        A.user(body.user_id)
        e = registry().get("seizure_deep")
        ad, x = ADAPTERS["seizure_deep"], body.canonical()
        spec = e.spec if e.available else None
        wins, starts, q = ad.windows(x, body.fs_hz, float(body.mains_hz), spec)
        thresholds = spec["thresholds"] if spec else dict(MOCK_SEIZURE_THRESHOLDS)
        stats = A.STATE.eeg_feats.get(body.user_id) or {}          # read only: same gate as /v1/predict/eeg
        if q < QUALITY_MIN_SEIZURE:
            status = "low_quality"
        elif stats.get("med") is None:
            status = "needs_baseline"
        else:
            status = "ok"
        if status == "ok":
            p = ad.predict(spec, e.runner, wins) if e.available else ad.mock(x, body.fs_hz, len(wins))
            core = ad.finish(p, starts, thresholds)
        else:
            core = {"windows": [], "max_probability": None, "flag": None,
                    "thresholds": {k: round(v, 4) for k, v in thresholds.items()}, "override_candidate": False}
        data = {**header("seizure_deep", e), "status": status, "quality_index": round(q, 3),
                "needs_baseline": status == "needs_baseline", **core,
                "baseline_hint": f"submit at least {A.EEG_FEAT_MIN} 4 s windows of EEG to /v1/baseline/eeg first"
                if status == "needs_baseline" else None,
                "input": {"fs_hz": body.fs_hz, "channels": list(SEIZURE_CHANNELS), "window_s": 4.0, "step_s": 2.0}}
        warn = warnings_for(e)
        if e.available:                          # never for mock: feed_shadow_risk would refuse it anyway
            feed_shadow_risk(A, body.user_id, data, warn, body.source)
        return A.envelope(body.user_id, data, model_ids(e), warn)

    @r.post("/models/reload")
    def reload_deep_models():
        A = _app()
        if not _RELOAD_LOCK.acquire(blocking=False):
            raise A.err(409, "reload_in_progress", "a reload is already running; retry shortly")
        try:
            reg = set_registry(DeepModelRegistry.from_env())   # built off to the side, then swapped in whole
        finally:
            _RELOAD_LOCK.release()
        return A.envelope(None, {"deep_models": reg.listing(),
                                 "note": "rescans the deep model folders only; other models load at service start"})

    return r


__all__ = ["DEEP_TASKS", "build_router", "deep_listing", "feed_shadow_risk", "registry", "set_registry",
           "MockOutputError", "MOCK_WARNING"]
