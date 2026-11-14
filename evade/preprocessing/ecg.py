"""ECG and PPG preprocessing: QRS detection, beat scoring, HRV features, window tables."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import signal as sps

from . import PIPELINE_VERSION, config
from .signal import assign_fold, bandpass, make_windows, notch, resample, signal_quality

# MIT-BIH beat annotation symbols counted as beats (excludes rhythm and non-beat marks)
BEAT_SYMBOLS = set("NLRBAaJSVrFejnE/fQ?")
PACED_RECORDS = {"102", "104", "107", "217"}   # paced beats; analysed with care


def detect_qrs(ecg: np.ndarray, fs: float) -> np.ndarray:
    """Simplified Pan-Tompkins detector. Returns R-peak sample indices.

    Steps: 5-15 Hz band-pass, derivative, squaring, 150 ms moving integration, adaptive threshold
    with a 250 ms refractory period, then refinement to the local maximum of |filtered ECG|
    within +/-50 ms. Baseline detector for phase 0; not a validated clinical algorithm.
    """
    x = np.asarray(ecg, float)
    x = np.nan_to_num(x - np.nanmedian(x))
    f = bandpass(x, fs, 5.0, 15.0, order=2)
    d = np.diff(f, prepend=f[0])
    sq = d ** 2
    win = max(1, int(0.150 * fs))
    integ = np.convolve(sq, np.ones(win) / win, mode="same")
    if not np.any(integ > 0):
        return np.array([], dtype=int)
    # adaptive threshold: fraction of a running high percentile over 8 s blocks
    block = int(8 * fs)
    thr = np.empty_like(integ)
    for s in range(0, len(integ), block):
        seg = integ[s:s + block]
        thr[s:s + block] = 0.35 * np.percentile(seg, 99) if len(seg) else 0
    cand, _ = sps.find_peaks(integ, height=thr, distance=int(0.25 * fs))
    r = int(0.05 * fs)
    refined = []
    for c in cand:
        lo, hi = max(0, c - r), min(len(f), c + r + 1)
        refined.append(lo + int(np.argmax(np.abs(f[lo:hi]))))
    return np.unique(np.array(refined, dtype=int))


def score_beats(ref: np.ndarray, det: np.ndarray, fs: float, tol_s: float = 0.150) -> dict:
    """Beat-level sensitivity and positive predictive value with a matching tolerance (EC57 style)."""
    ref = np.sort(np.asarray(ref, int))
    det = np.sort(np.asarray(det, int))
    tol = int(round(tol_s * fs))
    used = np.zeros(len(det), bool)
    tp = 0
    for r in ref:
        i = np.searchsorted(det, r)
        best, bd = None, tol + 1
        for j in (i - 1, i):
            if 0 <= j < len(det) and not used[j] and abs(det[j] - r) < bd:
                best, bd = j, abs(det[j] - r)
        if best is not None and bd <= tol:
            used[best] = True
            tp += 1
    fn = len(ref) - tp
    fp = len(det) - tp
    return {"tp": tp, "fn": fn, "fp": fp,
            "sensitivity": tp / len(ref) if len(ref) else float("nan"),
            "ppv": tp / len(det) if len(det) else float("nan")}


def hrv_features(rr_s: np.ndarray, rr_min: float = 0.3, rr_max: float = 2.0) -> dict:
    """Time-domain HRV from RR intervals in seconds. Intervals outside [rr_min, rr_max] are dropped.

    Returns hr_bpm, sdnn_ms, rmssd_ms, pnn50, n_rr, valid_frac. NaN where too few intervals.
    """
    rr = np.asarray(rr_s, float)
    ok = (rr >= rr_min) & (rr <= rr_max)
    valid_frac = float(ok.mean()) if rr.size else 0.0
    v = rr[ok]
    out = {"hr_bpm": np.nan, "sdnn_ms": np.nan, "rmssd_ms": np.nan, "pnn50": np.nan,
           "n_rr": int(v.size), "valid_frac": valid_frac}
    if v.size >= 3:
        d = np.diff(v)
        out.update(hr_bpm=float(60.0 / v.mean()), sdnn_ms=float(v.std(ddof=1) * 1000),
                   rmssd_ms=float(np.sqrt(np.mean(d ** 2)) * 1000),
                   pnn50=float(np.mean(np.abs(d) > 0.050)))
    return out


def preprocess_ecg(x: np.ndarray, fs: float, fs_out: float = config.FS_TARGET,
                   mains: float = config.NOTCH_HZ) -> tuple[np.ndarray, float]:
    """Band-pass 0.5-40 Hz, mains notch, resample to fs_out. x is (channels, samples) or (samples,)."""
    x = np.atleast_2d(np.asarray(x, float))
    x = np.nan_to_num(x)
    x = bandpass(x, fs, *config.ECG_BAND)
    x = notch(x, fs, mains)
    return resample(x, fs, fs_out), fs_out


def window_table(x: np.ndarray, fs: float, win_s: float, step_s: float, *, participant_id: str,
                 record_id: str, dataset: str, modality: str, labels: np.ndarray | None = None,
                 t0_utc: pd.Timestamp | None = None) -> tuple[np.ndarray, pd.DataFrame]:
    """Window a preprocessed signal and build the schema table (one row per window)."""
    wins, starts = make_windows(x, fs, win_s, step_s)
    q = np.array([signal_quality(w) for w in wins]) if len(wins) else np.empty(0)
    n = len(wins)
    ts = (pd.Timestamp(t0_utc) + pd.to_timedelta(starts, unit="s")) if t0_utc is not None else pd.NaT
    df = pd.DataFrame({
        "participant_id": participant_id, "record_id": record_id, "dataset": dataset,
        "modality": modality, "fs": fs, "window_index": np.arange(n),
        "window_start_s": starts, "window_length_s": win_s, "timestamp_utc": ts,
        "quality_index": q, "low_quality": q < config.QUALITY_MIN,
        "label": labels if labels is not None else -1,
        "fold": assign_fold(participant_id), "pipeline_version": PIPELINE_VERSION,
    })
    return wins.astype(np.float32), df
