"""Task adapters for the deep models: manifest spec, preprocessing, inference, post-processing and mock outputs.

Every adapter returns the same keys in mock and in model mode, so the frontend codes against one schema. Mock
outputs are deterministic for the same input (seeded from a sha256 of the samples) and plausible (probabilities in
[0, 1], softmax rows that sum to 1, hypnograms that follow sleep-stage transitions), but they are placeholders, never
predictions. Inputs reach the adapters already validated and in canonical channel order (see evade/api/deep_models.py).
"""
from __future__ import annotations

import hashlib
import json
from typing import Callable

import numpy as np

from ..preprocessing import config
from ..preprocessing.eeg import norm_bipolar_name, preprocess_eeg
from ..preprocessing.signal import make_windows, resample, signal_quality
from .registry import ModelLoadError

ECG_LEADS = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
ECG_CLASSES = ["NORM", "MI", "STTC", "CD", "HYP"]
ECG_FS, ECG_SECONDS = 100.0, 10.0
SLEEP_CHANNELS = ["Fpz-Cz", "Pz-Oz"]
SLEEP_STAGES = ["W", "N1", "N2", "N3", "REM"]
SLEEP_FS, EPOCH_S = 100.0, 30.0
SEIZURE_CHANNELS = list(config.BIPOLAR_NAMES)
SEIZURE_FS = 250.0
DEFAULT_ECG_THRESHOLD = 0.5
MOCK_SEIZURE_THRESHOLDS = {"review_5pct_fpr": 0.5, "suspect_1pct_fpr": 0.8}   # placeholders, not operating points

Sidecar = Callable[[str], "bytes | None"]


def unsupported(detail: str) -> ModelLoadError:
    return ModelLoadError("spec_unsupported", detail)


def input_seed(task: str, x: np.ndarray, fs: float, extra: str = "") -> int:
    """Deterministic seed from the input (hashlib, not hash(): Python's hash is salted per process)."""
    h = hashlib.sha256()
    h.update(f"{task}|{fs:.6f}|{extra}|{x.shape}".encode())
    h.update(np.ascontiguousarray(x, np.float32).tobytes())
    return int.from_bytes(h.digest()[:8], "little")


def sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(z, -60, 60)))


def softmax(z: np.ndarray, axis: int = -1) -> np.ndarray:
    z = z - z.max(axis=axis, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=axis, keepdims=True)


def _names(inp: dict, keys: tuple[str, ...], sidecar: Sidecar) -> list[str] | None:
    for k in keys:
        v = inp.get(k)
        if isinstance(v, list):
            return [str(c) for c in v]
        if isinstance(v, str) and v.endswith(".json"):
            raw = sidecar(v)
            if raw is None:
                raise ModelLoadError("file_missing", f"{v} missing")
            names = json.loads(raw.decode("utf-8"))
            if isinstance(names, list):
                return [str(c) for c in names]
    return None


def _norm_type(inp: dict) -> tuple[str, dict]:
    n = inp.get("normalization")
    if n is None:
        return "none", {}
    if isinstance(n, str):
        return n.strip().lower(), {}
    if isinstance(n, dict) and isinstance(n.get("type"), str):
        return n["type"].strip().lower(), n
    raise unsupported("normalization must be a string or an object with a type")


def _order(model_names: list[str], canonical: list[str], key=str.upper) -> list[int]:
    """Index of each model channel in the canonical order; raises unless it is a permutation."""
    canon = {key(c): i for i, c in enumerate(canonical)}
    idx = [canon.get(key(c), -1) for c in model_names]
    if len(model_names) != len(canonical) or sorted(idx) != list(range(len(canonical))):
        raise unsupported(f"channels must be a permutation of {canonical}")
    return idx


def _shape(inp: dict, ndim: tuple[int, ...]) -> list:
    s = inp.get("tensor_shape")
    if not isinstance(s, list) or len(s) not in ndim:
        raise unsupported(f"tensor_shape must have {' or '.join(map(str, ndim))} dimensions")
    return s


def _check_out(y: np.ndarray, shape: tuple) -> None:
    if tuple(y.shape) != shape:
        raise ModelLoadError("unexpected_model_type", f"model output shape {tuple(y.shape)}, expected {shape}")


# ---------------- ECG (PTB-XL superclass, multilabel) ----------------
class EcgAdapter:
    """Mirrors evade/training/deep/export.py: 100 Hz, 1000 samples, no band-pass, per-lead z-score with the train
    statistics from the manifest, sliding crops (tensor_shape[2] samples, window_stride_samples stride), sigmoid,
    mean over crops."""
    task = "ecg_deep"

    def spec_from_manifest(self, m: dict, sidecar: Sidecar) -> dict:
        inp, out = m["input"], m["output"]
        leads = _names(inp, ("leads", "channels", "features"), sidecar) or [c.upper() for c in ECG_LEADS]
        order = _order(leads, ECG_LEADS)
        shape = _shape(inp, (3,))
        if shape[1] != 12 or not isinstance(shape[2], int) or shape[2] <= 0:
            raise unsupported("tensor_shape must be [batch, 12, crop_samples]")
        fs = float(inp.get("fs_hz", ECG_FS))
        record = int(inp.get("record_samples", round(ECG_SECONDS * fs)))
        crop = int(shape[2])
        stride = int(inp.get("window_stride_samples", crop))
        if not (0 < crop <= record and stride > 0):
            raise unsupported("crop and stride must fit inside the record")
        ntype, nspec = _norm_type(inp)
        if ntype == "none":
            mean, std = np.zeros(12, np.float32), np.ones(12, np.float32)
        elif ntype in ("per_lead_zscore", "zscore"):
            mean, std = np.asarray(nspec.get("mean"), np.float32), np.asarray(nspec.get("std"), np.float32)
            if mean.shape != (12,) or std.shape != (12,) or not np.all(std > 0):
                raise unsupported("per_lead_zscore needs 12 means and 12 positive stds")
        else:
            raise unsupported(f"normalization {ntype} not supported for ecg_deep")
        classes = [str(c).upper() for c in (out.get("classes") or [])]
        cls_idx = _order(classes, ECG_CLASSES)
        act = (out.get("activation") or "sigmoid").lower()
        if act not in ("sigmoid", "probabilities"):
            raise unsupported("ecg_deep output.activation must be sigmoid or probabilities")
        thresholds, src = self.manifest_thresholds(out), "manifest"
        if thresholds is None:                   # export.py writes none yet: labelled, so the UI can tell
            thresholds, src = {c: DEFAULT_ECG_THRESHOLD for c in ECG_CLASSES}, "default"
        return {"c_in": 12, "n_out": 5, "fs": fs, "record": record, "crop": crop, "stride": stride,
                "lead_order": order, "mean": mean, "std": std, "class_order": cls_idx, "activation": act,
                "thresholds": thresholds, "threshold_source": src}

    @staticmethod
    def manifest_thresholds(out: dict) -> dict | None:
        """Per-class thresholds from output.operating_points ({class: t} or {class: {"threshold": t}}, the form the
        deep-serving ECG adapter reads) or output.thresholds ({class: t}, or one "review" value for all)."""
        for key in ("operating_points", "thresholds"):
            ops = out.get(key)
            if not isinstance(ops, dict):
                continue
            up = {str(k).upper(): (v.get("threshold") if isinstance(v, dict) else v) for k, v in ops.items()}
            if all(isinstance(up.get(c), (int, float)) for c in ECG_CLASSES):
                return {c: float(up[c]) for c in ECG_CLASSES}
            if isinstance(up.get("REVIEW"), (int, float)):
                return {c: float(up["REVIEW"]) for c in ECG_CLASSES}
        return None

    def smoke(self, spec: dict, runner) -> None:
        _check_out(runner(np.zeros((2, 12, spec["crop"]), np.float32)), (2, spec["n_out"]))

    @staticmethod
    def to_record(x: np.ndarray, fs_in: float, fs: float, record: int, fill: np.ndarray) -> np.ndarray:
        """Resample, then crop the central record or pad the end with each lead's training mean."""
        y = resample(np.asarray(x, np.float64), fs_in, fs)
        n = y.shape[1]
        if n >= record:
            off = (n - record) // 2
            return y[:, off:off + record]
        return np.concatenate([y, np.repeat(fill[:, None], record - n, axis=1)], axis=1)

    def predict(self, spec: dict, runner, x: np.ndarray, fs_in: float) -> dict:
        canon_mean = np.empty(12)
        canon_mean[spec["lead_order"]] = spec["mean"]          # manifest lead order -> canonical order
        rec = self.to_record(x, fs_in, spec["fs"], spec["record"], canon_mean)[spec["lead_order"]]
        rec = (rec - spec["mean"][:, None]) / spec["std"][:, None]
        starts = range(0, spec["record"] - spec["crop"] + 1, spec["stride"])
        batch = np.stack([rec[:, s:s + spec["crop"]] for s in starts]).astype(np.float32)
        y = runner(batch)
        p = (sigmoid(y) if spec["activation"] == "sigmoid" else np.clip(y, 0, 1)).mean(axis=0)
        probs = {ECG_CLASSES[spec["class_order"][i]]: float(p[i]) for i in range(len(p))}
        return self.finish(probs, spec["thresholds"], spec["threshold_source"], len(batch))

    def mock(self, x: np.ndarray, fs_in: float) -> dict:
        rng = np.random.default_rng(input_seed(self.task, x, fs_in))
        normal = rng.random() < 0.6                     # most records look normal, a few carry one or two findings
        p = {"NORM": rng.beta(8, 2) if normal else rng.beta(1.5, 6)}
        k = rng.choice(4, size=0 if normal else rng.integers(1, 3), replace=False)
        for i, c in enumerate(ECG_CLASSES[1:]):
            p[c] = rng.beta(6, 2) if i in k else rng.beta(1, 9)
        return self.finish({c: float(p[c]) for c in ECG_CLASSES}, {c: DEFAULT_ECG_THRESHOLD for c in ECG_CLASSES},
                           "default", 0)

    @staticmethod
    def finish(probs: dict, thresholds: dict, src: str, n_crops: int) -> dict:
        probs = {c: round(probs[c], 4) for c in ECG_CLASSES}
        top = sorted(((c, p) for c, p in probs.items() if p >= thresholds[c]), key=lambda t: -t[1])
        return {"probabilities": probs, "thresholds": {c: round(thresholds[c], 4) for c in ECG_CLASSES},
                "threshold_source": src,
                "top_classes": [{"class": c, "probability": p} for c, p in top], "n_crops": n_crops}


# ---------------- sleep staging ----------------
STAGE_ALIASES = {"W": "W", "WAKE": "W", "N1": "N1", "N2": "N2", "N3": "N3", "N4": "N3", "R": "REM", "REM": "REM"}
# row: from-stage, column: to-stage (W, N1, N2, N3, REM); a plausible night, used only for mock hypnograms
MOCK_TRANSITIONS = np.array([
    [0.86, 0.11, 0.02, 0.00, 0.01],
    [0.07, 0.58, 0.32, 0.00, 0.03],
    [0.02, 0.04, 0.84, 0.07, 0.03],
    [0.01, 0.01, 0.12, 0.86, 0.00],
    [0.04, 0.05, 0.04, 0.00, 0.87],
])


class SleepAdapter:
    """30 s epochs of Fpz-Cz and Pz-Oz. Model input per epoch [batch, 2, 3000], or per sequence
    [batch, L, 2, 3000] (L an integer: chunks of L epochs, the last one zero-padded; L a name: the whole sequence).
    Normalization: none, or per-channel z-score over the submitted recording (the training plan's per-night rule)."""
    task = "sleep_deep"

    def spec_from_manifest(self, m: dict, sidecar: Sidecar) -> dict:
        inp, out = m["input"], m["output"]
        chans = _names(inp, ("channels", "features"), sidecar) or [c.upper() for c in SLEEP_CHANNELS]
        order = _order(chans, SLEEP_CHANNELS)
        fs = float(inp.get("fs_hz", SLEEP_FS))
        t = int(round(EPOCH_S * fs))
        shape = _shape(inp, (3, 4))
        if shape[-2] != 2 or shape[-1] != t:
            raise unsupported(f"tensor_shape must end with [2, {t}] (30 s epochs at fs_hz)")
        seq = shape[1] if len(shape) == 4 else None
        if seq is not None and isinstance(seq, int) and seq <= 0:
            raise unsupported("sequence length must be positive")
        ntype, _ = _norm_type(inp)
        if ntype in ("per_recording_zscore", "per_night_zscore", "per_channel_per_night_zscore"):
            ntype = "per_recording_zscore"
        elif ntype != "none":
            raise unsupported(f"normalization {ntype} not supported for sleep_deep")
        classes = [STAGE_ALIASES.get(str(c).upper(), str(c)) for c in (out.get("classes") or [])]
        cls_idx = _order(classes, SLEEP_STAGES)
        act = (out.get("activation") or "softmax").lower()
        if act not in ("softmax", "probabilities"):
            raise unsupported("sleep_deep output.activation must be softmax or probabilities")
        return {"c_in": 2, "n_out": 5, "fs": fs, "epoch_samples": t, "channel_order": order,
                "seq": seq if len(shape) == 4 else 0, "norm": ntype, "class_order": cls_idx, "activation": act}

    def smoke(self, spec: dict, runner) -> None:
        t = spec["epoch_samples"]
        if spec["seq"] == 0:
            _check_out(runner(np.zeros((1, 2, t), np.float32)), (1, 5))
        else:
            L = spec["seq"] if isinstance(spec["seq"], int) else 2
            _check_out(runner(np.zeros((1, L, 2, t), np.float32)), (1, L, 5))

    def predict(self, spec: dict, runner, x: np.ndarray, fs_in: float) -> dict:
        y = resample(np.asarray(x, np.float64)[spec["channel_order"]], fs_in, spec["fs"])
        t = spec["epoch_samples"]
        n_ep = y.shape[1] // t
        y = y[:, :n_ep * t]
        if spec["norm"] == "per_recording_zscore":
            y = (y - y.mean(axis=1, keepdims=True)) / np.maximum(y.std(axis=1, keepdims=True), 1e-6)
        ep = y.reshape(2, n_ep, t).transpose(1, 0, 2).astype(np.float32)        # (E, 2, T)
        if spec["seq"] == 0:
            z = runner(ep)
        elif isinstance(spec["seq"], int):
            L = spec["seq"]
            k = -(-n_ep // L)
            pad = np.zeros((k * L - n_ep, 2, t), np.float32)
            z = runner(np.concatenate([ep, pad]).reshape(k, L, 2, t)).reshape(k * L, -1)[:n_ep]
        else:
            z = runner(ep[None]).reshape(n_ep, -1)
        p = softmax(z) if spec["activation"] == "softmax" else z / np.maximum(z.sum(axis=1, keepdims=True), 1e-9)
        canon = np.zeros_like(p)
        canon[:, spec["class_order"]] = p
        return self.finish(canon)

    def mock(self, x: np.ndarray, fs_in: float) -> dict:
        n_ep = x.shape[1] // int(round(EPOCH_S * fs_in))
        rng = np.random.default_rng(input_seed(self.task, x, fs_in))
        s = int(rng.choice(5, p=[0.70, 0.20, 0.10, 0.0, 0.0]))
        stages = []
        for _ in range(n_ep):
            stages.append(s)
            s = int(rng.choice(5, p=MOCK_TRANSITIONS[s]))
        p = np.zeros((n_ep, 5))
        for i, st in enumerate(stages):
            rest = rng.dirichlet(np.ones(4)) * (1.0 - rng.uniform(0.55, 0.9))
            p[i] = np.insert(rest, st, 0.0)
            p[i, st] = 1.0 - rest.sum()
        return self.finish(p)

    @staticmethod
    def finish(p: np.ndarray) -> dict:
        p = np.round(p, 4)
        p[np.arange(len(p)), p.argmax(axis=1)] += 1.0 - p.sum(axis=1)        # rows sum to 1 after rounding
        stages = [SLEEP_STAGES[i] for i in p.argmax(axis=1)]
        epochs = [{"index": i, "t_start_s": i * EPOCH_S, "stage": st,
                   "probabilities": {c: round(float(p[i, j]), 4) for j, c in enumerate(SLEEP_STAGES)}}
                  for i, st in enumerate(stages)]
        mins = {c: stages.count(c) * EPOCH_S / 60.0 for c in SLEEP_STAGES}
        rec = len(stages) * EPOCH_S / 60.0
        tst = rec - mins["W"]
        onset = next((i for i, st in enumerate(stages) if st != "W"), None)
        return {"epochs": epochs, "summary": {
            "n_epochs": len(stages), "recording_min": rec, "total_sleep_time_min": tst,
            "sleep_efficiency": round(tst / rec, 4) if rec else None,
            "sleep_onset_latency_min": onset * EPOCH_S / 60.0 if onset is not None else None,
            "minutes_per_stage": mins,
            "scope": "submitted epochs only; send consecutive chunks and combine them for a night"}}


# ---------------- seizure-suspect windows ----------------
def _op_threshold(out: dict, fpr: float, names: tuple[str, ...]) -> float | None:
    for op in out.get("operating_points") or []:
        if not isinstance(op, dict) or not isinstance(op.get("threshold"), (int, float)):
            continue
        tf = op.get("target_fpr")
        if (isinstance(tf, (int, float)) and abs(tf - fpr) < 1e-9) or op.get("name") in names:
            return float(op["threshold"])
    thr = out.get("thresholds")
    if isinstance(thr, dict):
        for n in names:
            if isinstance(thr.get(n), (int, float)):
                return float(thr[n])
    return None


class SeizureAdapter:
    """18-channel bipolar EEG, band-pass 0.5-45 Hz with a mains notch, resampled to the model rate, 4 s windows with
    a 2 s step (the beta windows), normalization none or per window, sigmoid, centered moving average over
    smooth_windows (default 3, as in the beta). Levels: review above the 5% FPR threshold, suspect_seizure above the
    1% threshold, both from the manifest operating points (no default thresholds). A per-user baseline-window
    normalization is not implemented yet, so a manifest asking for one is listed unavailable (spec_unsupported)."""
    task = "seizure_deep"

    def spec_from_manifest(self, m: dict, sidecar: Sidecar) -> dict:
        inp, out = m["input"], m["output"]
        chans = _names(inp, ("channels", "features"), sidecar) or SEIZURE_CHANNELS
        order = _order([norm_bipolar_name(c) or c for c in chans], SEIZURE_CHANNELS)
        fs = float(inp.get("fs_hz", SEIZURE_FS))
        win_s, step_s = float(inp.get("window_s", config.EEG_WIN_S)), float(inp.get("step_s", config.EEG_STEP_S))
        shape = _shape(inp, (3,))
        if shape[1] != 18 or shape[2] != int(round(win_s * fs)):
            raise unsupported("tensor_shape must be [batch, 18, window_s * fs_hz]")
        ntype, _ = _norm_type(inp)
        if ntype not in ("none", "per_window_zscore", "per_window_robust"):
            raise unsupported(f"normalization {ntype} not supported for seizure_deep")
        t_rev = _op_threshold(out, 0.05, ("review",))
        t_sus = _op_threshold(out, 0.01, ("suspect_seizure", "suspect"))
        if t_rev is None or t_sus is None or not (0.0 <= t_rev <= t_sus <= 1.0):
            raise unsupported("output.operating_points need thresholds at target_fpr 0.05 and 0.01 (review <= suspect)")
        act = (out.get("activation") or "sigmoid").lower()
        if act not in ("sigmoid", "softmax", "probabilities"):
            raise unsupported("seizure_deep output.activation must be sigmoid, softmax or probabilities")
        smooth = int(inp.get("smooth_windows", out.get("smooth_windows", 3)))
        return {"c_in": 18, "n_out": 2 if act == "softmax" else 1, "fs": fs, "win_s": win_s, "step_s": step_s,
                "channel_order": order, "norm": ntype, "activation": act, "smooth": max(smooth, 1),
                "thresholds": {"review_5pct_fpr": t_rev, "suspect_1pct_fpr": t_sus}}

    def smoke(self, spec: dict, runner) -> None:
        y = runner(np.zeros((2, 18, int(round(spec["win_s"] * spec["fs"]))), np.float32))
        if y.shape[0] != 2 or y.reshape(2, -1).shape[1] != spec["n_out"]:
            raise ModelLoadError("unexpected_model_type", f"model output shape {tuple(y.shape)}")

    @staticmethod
    def windows(x: np.ndarray, fs_in: float, mains: float, spec: dict | None):
        """Preprocess and window (same in mock and model mode, so quality and window times match)."""
        fs = spec["fs"] if spec else SEIZURE_FS
        win_s, step_s = (spec["win_s"], spec["step_s"]) if spec else (config.EEG_WIN_S, config.EEG_STEP_S)
        xp, _ = preprocess_eeg(x, fs_in, fs, mains)
        wins, starts = make_windows(xp, fs, win_s, step_s)
        q = float(np.mean([signal_quality(w) for w in wins])) if len(wins) else 0.0
        return wins, starts, q

    def predict(self, spec: dict, runner, wins: np.ndarray) -> np.ndarray:
        w = wins[:, spec["channel_order"]]
        if spec["norm"] == "per_window_zscore":
            w = (w - w.mean(axis=2, keepdims=True)) / np.maximum(w.std(axis=2, keepdims=True), 1e-6)
        elif spec["norm"] == "per_window_robust":
            med = np.median(w, axis=2, keepdims=True)
            w = np.clip((w - med) / np.maximum(1.4826 * np.median(np.abs(w - med), axis=2, keepdims=True), 1e-6), -20, 20)
        y = runner(w.astype(np.float32)).reshape(len(w), -1)
        if spec["activation"] == "softmax":
            p = softmax(y)[:, 1]
        elif spec["activation"] == "sigmoid":
            p = sigmoid(y[:, 0])
        else:
            p = np.clip(y[:, 0], 0, 1)
        return self.smooth(p, spec["smooth"])

    def mock(self, x: np.ndarray, fs_in: float, n_windows: int) -> np.ndarray:
        rng = np.random.default_rng(input_seed(self.task, x, fs_in))
        return self.smooth(rng.beta(1.0, 30.0, size=n_windows), 3)

    @staticmethod
    def smooth(p: np.ndarray, k: int) -> np.ndarray:
        return np.convolve(p, np.ones(k) / k, mode="same") if k > 1 and len(p) >= k else p

    @staticmethod
    def finish(p: np.ndarray, starts: np.ndarray, thresholds: dict) -> dict:
        def level(v):
            return "suspect_seizure" if v > thresholds["suspect_1pct_fpr"] else \
                "review" if v > thresholds["review_5pct_fpr"] else "none"
        pmax = float(p.max()) if len(p) else 0.0
        return {"windows": [{"t_start_s": round(float(s), 3), "probability": round(float(v), 4), "level": level(v)}
                            for s, v in zip(starts, p)],
                "max_probability": round(pmax, 4), "flag": level(pmax),
                "thresholds": {k: round(v, 4) for k, v in thresholds.items()},
                "override_candidate": pmax > thresholds["suspect_1pct_fpr"]}


ADAPTERS = {"ecg_deep": EcgAdapter(), "sleep_deep": SleepAdapter(), "seizure_deep": SeizureAdapter()}
