"""Per-user baselines and baseline-vs-current comparison (lit review/planning/baseline-comparison.md).

In-memory store for the hackathon beta (state is lost on restart; a database replaces it later).
Robust statistics: center = median, spread = 1.4826 x MAD floored at 15% of |center|; skewed metrics are
log-transformed; only deviation in each metric's bad direction counts toward the level and component score.
"""
from __future__ import annotations

import math
import threading
import time
import uuid
from dataclasses import dataclass, field

import numpy as np

# minimum observations before a baseline locks (standard, demo) per modality
N_NEEDED = {"ecg": (24, 30), "ppg": (24, 30), "eeg": (60, 30), "vitals": (20, 10), "sleep": (7, 3),
            "behavioral": (2, 1), "radiation": (1, 1), "bone": (1, 1)}
LOG_METRICS = {"rmssd_ms", "sdnn_ms", "theta_beta", "rel_beta", "rel_theta", "rel_alpha"}
# +1: high is bad, -1: low is bad, 0: both directions
BAD_DIR = {"hr_bpm": +1, "rmssd_ms": -1, "sdnn_ms": -1, "rr_irregularity": +1, "theta_beta": +1, "rel_beta": +1,
           "sleep_hours": -1, "spo2_pct": -1, "temp_c": 0, "bp_systolic_mmhg": +1, "bp_diastolic_mmhg": +1,
           "glucose_mgdl": 0, "resp_rate_bpm": 0, "stress": +1, "fatigue": +1, "mood": -1, "sleep_quality": -1}
DOMAIN_OF = {"ecg": "cardiac", "ppg": "cardiac", "vitals": "cardiac", "eeg": "neuro", "sleep": "behavioral",
             "behavioral": "behavioral", "radiation": "radiation", "bone": "bone"}
Z_ANCHORS = [(1.0, 0.15), (2.0, 0.40), (3.0, 0.70), (4.0, 1.0)]
Z_CLIP, SPREAD_FLOOR, EW_ALPHA = 5.0, 0.15, 0.02


def component_score(z_bad: float) -> float:
    if z_bad < 1.0:
        return 0.0
    for (x0, y0), (x1, y1) in zip(Z_ANCHORS, Z_ANCHORS[1:]):
        if z_bad <= x1:
            return y0 + (y1 - y0) * (z_bad - x0) / (x1 - x0)
    return 1.0


def level(z_bad: float) -> str:
    return "red" if z_bad >= 2 else "yellow" if z_bad >= 1 else "green"


def _t(metric: str, v: float) -> float:
    return math.log(max(v, 1e-9)) if metric in LOG_METRICS else v


@dataclass
class Baseline:
    baseline_id: str
    user_id: str
    modality: str
    metric: str
    mode: str = "demo"
    phase: str = "demo"
    status: str = "building"            # building -> provisional -> confirmed -> archived
    values: list = field(default_factory=list)
    center: float | None = None
    spread: float | None = None
    created: float = field(default_factory=time.time)

    def n_needed(self) -> int:
        std, demo = N_NEEDED.get(self.modality, (20, 10))
        return demo if self.mode == "demo" else std

    def lock(self) -> None:
        x = np.array([_t(self.metric, v) for v in self.values], dtype=float)
        c = float(np.median(x))
        mad = 1.4826 * float(np.median(np.abs(x - c)))
        floor = SPREAD_FLOOR * abs(c) if self.metric not in LOG_METRICS else SPREAD_FLOOR   # log scale: ~15% change
        self.center, self.spread = c, max(mad, floor, 1e-6)
        self.status = "provisional"

    def public(self) -> dict:
        return {"baseline_id": self.baseline_id, "modality": self.modality, "metric": self.metric, "status": self.status,
                "phase": self.phase, "mode": self.mode, "n": len(self.values), "n_needed": self.n_needed(),
                "center": self.center, "spread": self.spread, "log_scale": self.metric in LOG_METRICS}


class BaselineStore:
    def __init__(self):
        self._b: dict[tuple, Baseline] = {}
        self._archive: list[Baseline] = []
        self._lock = threading.Lock()
        self._deleted: set[str] = set()     # tombstones: never recreate data for a deleted user

    def _get(self, user_id: str, modality: str, metric: str, mode: str = "demo", phase: str = "demo") -> Baseline:
        k = (user_id, modality, metric)
        if user_id in self._deleted:
            return Baseline("deleted", user_id, modality, metric, mode, phase)   # not stored
        if k not in self._b:
            self._b[k] = Baseline(f"b_{modality}_{uuid.uuid4().hex[:8]}", user_id, modality, metric, mode, phase)
        return self._b[k]

    def add(self, user_id: str, modality: str, metric: str, value: float, *, quality: float = 1.0,
            mode: str = "demo", phase: str = "demo") -> Baseline:
        with self._lock:
            b = self._get(user_id, modality, metric, mode, phase)
            if value is None or not math.isfinite(value) or quality < 0.6:
                return b
            if b.status == "building":
                b.values.append(float(value))
                if len(b.values) >= b.n_needed():
                    b.lock()
            return b

    def compare(self, user_id: str, modality: str, metric: str, value: float | None, *, quality: float = 1.0,
                unit: str = "", context: list[str] | None = None, shadow: bool = False) -> dict:
        with self._lock:
            b = self._b.get((user_id, modality, metric))
            out = {"metric": metric, "modality": modality, "domain": DOMAIN_OF.get(modality, "cardiac"), "unit": unit,
                   "value": value, "quality_index": round(quality, 3), "shadow": shadow, "context_used": context or []}
            if value is None or not math.isfinite(value) or quality < 0.6:
                return {**out, "value": None, "z": None, "level": "green", "component_score": 0.0, "confidence": "low_quality"}
            if b is None or b.center is None:
                return {**out, "baseline_id": b.baseline_id if b else None, "z": None, "level": "green",
                        "component_score": 0.0, "confidence": "provisional",
                        "baseline_progress": round(len(b.values) / b.n_needed(), 2) if b else 0.0}
            z = max(-Z_CLIP, min(Z_CLIP, (_t(metric, value) - b.center) / b.spread))
            d = BAD_DIR.get(metric, 0)
            z_bad = abs(z) if d == 0 else max(0.0, d * z)
            lvl = level(z_bad)
            center_raw = math.exp(b.center) if metric in LOG_METRICS else b.center
            # slow update only while green, so the baseline follows adaptation without absorbing a decline
            if lvl == "green" and b.status == "confirmed":
                b.center = (1 - EW_ALPHA) * b.center + EW_ALPHA * _t(metric, value)
            conf = "ok" if b.status == "confirmed" else ("demo" if b.mode == "demo" else "provisional")
            return {**out, "baseline_id": b.baseline_id, "baseline_status": b.status, "baseline_center": round(center_raw, 4),
                    "baseline_spread": round(b.spread, 4), "n_baseline": len(b.values), "z": round(z, 3),
                    "delta_abs": round(value - center_raw, 4),
                    "delta_pct": round(100 * (value - center_raw) / center_raw, 2) if center_raw else None,
                    "level": lvl, "component_score": round(component_score(z_bad), 3), "confidence": conf}

    def confirm(self, user_id: str, modality: str) -> int:
        with self._lock:
            n = 0
            for (u, m, _), b in self._b.items():
                if u == user_id and m == modality and b.status == "provisional":
                    b.status, n = "confirmed", n + 1
            return n

    def reset(self, user_id: str, modality: str, phase: str = "demo") -> int:
        with self._lock:
            keys = [k for k in self._b if k[0] == user_id and k[1] == modality]
            for k in keys:
                b = self._b.pop(k)
                b.status = "archived"
                self._archive.append(b)
            return len(keys)

    def for_user(self, user_id: str) -> list[dict]:
        with self._lock:
            return [b.public() for (u, _, _), b in self._b.items() if u == user_id]

    def delete_user(self, user_id: str) -> None:
        with self._lock:
            self._deleted.add(user_id)
            for k in [k for k in self._b if k[0] == user_id]:
                del self._b[k]
            self._archive = [b for b in self._archive if b.user_id != user_id]


def domain_scores(comparisons: list[dict]) -> dict:
    """Domain score = 0.6 x max + 0.4 x mean of component scores (non-shadow), as in risk-score.md."""
    by: dict[str, list] = {}
    shadow: dict[str, list] = {}
    for c in comparisons:
        (shadow if c.get("shadow") else by).setdefault(c["domain"], []).append(c["component_score"])
    out = {}
    for d in set(by) | set(shadow):
        v = by.get(d, [])
        s_valid = 0.6 * max(v) + 0.4 * (sum(v) / len(v)) if v else 0.0
        allv = v + shadow.get(d, [])
        s_all = 0.6 * max(allv) + 0.4 * (sum(allv) / len(allv)) if allv else 0.0
        out[d] = {"s": round(s_all, 4), "s_validated": round(s_valid, 4)}
    return out
