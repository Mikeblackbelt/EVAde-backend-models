"""Signal helpers: filtering, resampling, windowing, quality, robust statistics, subject-wise folds."""
from __future__ import annotations

import hashlib
from fractions import Fraction

import numpy as np
from scipy import signal as sps

from . import config


def bandpass(x: np.ndarray, fs: float, lo: float, hi: float, order: int = 4) -> np.ndarray:
    """Zero-phase Butterworth band-pass along the last axis. hi is clipped below Nyquist."""
    nyq = fs / 2.0
    hi = min(hi, 0.99 * nyq)
    if not 0 < lo < hi:
        raise ValueError(f"bad band {lo}-{hi} Hz for fs={fs}")
    sos = sps.butter(order, [lo, hi], btype="bandpass", fs=fs, output="sos")
    return sps.sosfiltfilt(sos, x, axis=-1)


def notch(x: np.ndarray, fs: float, f0: float = config.NOTCH_HZ, q: float = 30.0) -> np.ndarray:
    """Zero-phase mains notch. Skipped (returns x) if f0 is at or above Nyquist."""
    if f0 >= fs / 2.0:
        return x
    b, a = sps.iirnotch(f0, q, fs=fs)
    return sps.filtfilt(b, a, x, axis=-1)


def resample(x: np.ndarray, fs_in: float, fs_out: float = config.FS_TARGET) -> np.ndarray:
    """Polyphase resample along the last axis. Returns x unchanged if rates are equal."""
    if abs(fs_in - fs_out) < 1e-9:
        return x
    frac = Fraction(fs_out / fs_in).limit_denominator(1000)
    return sps.resample_poly(x, frac.numerator, frac.denominator, axis=-1)


def make_windows(x: np.ndarray, fs: float, win_s: float, step_s: float) -> tuple[np.ndarray, np.ndarray]:
    """Slice (channels, samples) or (samples,) into windows.

    Returns (windows, starts_s) with windows shaped (N, channels, T). Incomplete tail is dropped.
    """
    x = np.atleast_2d(x)
    t = int(round(win_s * fs))
    step = int(round(step_s * fs))
    if t <= 0 or step <= 0:
        raise ValueError("window and step must be positive")
    n = x.shape[-1]
    if n < t:
        return np.empty((0, x.shape[0], t), dtype=x.dtype), np.empty(0)
    idx = np.arange(0, n - t + 1, step)
    wins = np.stack([x[:, i:i + t] for i in idx])
    return wins, idx / fs


def signal_quality(win: np.ndarray, flat_tol: float = 1e-12) -> float:
    """Quality index in [0, 1] for one window (channels, T) or (T,).

    1 minus the worst of: NaN fraction, flat-line fraction (zero first difference),
    saturation fraction (samples at the window extremes), and extreme-amplitude fraction
    (more than 8 robust SDs from the median, penalised only above 10% of samples). Design heuristic, to be calibrated.
    """
    w = np.atleast_2d(win).astype(float)
    nan_frac = float(np.isnan(w).mean())
    w = np.nan_to_num(w, nan=0.0)
    flat = float((np.abs(np.diff(w, axis=-1)) <= flat_tol).mean())
    lo, hi = w.min(axis=-1, keepdims=True), w.max(axis=-1, keepdims=True)
    span = np.maximum(hi - lo, 1e-12)
    sat = float((((w <= lo + 1e-6 * span) | (w >= hi - 1e-6 * span)).mean(axis=-1) > 0.05).mean())
    med = np.median(w, axis=-1, keepdims=True)
    mad = np.median(np.abs(w - med), axis=-1, keepdims=True) * 1.4826
    amp = float((np.abs(w - med) > 8.0 * np.maximum(mad, 1e-12)).mean())
    # amplitude outliers only count beyond 10% of samples, so legitimate QRS peaks are not penalised
    return float(np.clip(1.0 - max(nan_frac, flat, sat, max(0.0, amp - 0.10) * 5.0), 0.0, 1.0))


def robust_stats(x: np.ndarray, floor_frac: float = 0.15) -> tuple[float, float]:
    """Median and 1.4826*MAD with a floor of floor_frac*|median| (matches baseline-comparison.md)."""
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        raise ValueError("no finite values")
    med = float(np.median(x))
    spread = float(np.median(np.abs(x - med)) * 1.4826)
    return med, max(spread, floor_frac * abs(med), 1e-9)


def robust_z(x, center: float, spread: float, clip: float = 5.0):
    return np.clip((np.asarray(x, float) - center) / spread, -clip, clip)


def assign_fold(participant_id: str, n_folds: int = config.N_FOLDS) -> int:
    """Deterministic subject-wise fold from a hash of the participant ID (same person, same fold)."""
    h = hashlib.sha256(str(participant_id).encode()).hexdigest()
    return int(h[:8], 16) % n_folds
