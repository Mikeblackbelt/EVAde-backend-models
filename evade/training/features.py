"""Window feature extraction for the beta models (CPU, vectorized). Design values, not tuned.

All functions take windows shaped (n_windows, n_channels, n_samples) and return a float32 array
(n_windows, n_features) plus the feature names.
"""
from __future__ import annotations

import numpy as np
from scipy.signal import welch
from scipy.stats import kurtosis, skew

EEG_BANDS = {"delta": (0.5, 4.0), "theta": (4.0, 8.0), "alpha": (8.0, 13.0), "beta": (13.0, 30.0), "gamma": (30.0, 45.0)}
SLEEP_BANDS = {"delta": (0.5, 4.0), "theta": (4.0, 8.0), "alpha": (8.0, 11.5), "sigma": (11.5, 16.0), "beta": (16.0, 30.0)}
ECG_BANDS = {"b1": (0.5, 4.0), "b2": (4.0, 10.0), "b3": (10.0, 20.0), "b4": (20.0, 40.0)}
EPS = 1e-12


def band_powers(wins: np.ndarray, fs: float, bands: dict, nperseg: int = 256) -> np.ndarray:
    """Absolute band power, shape (n, ch, n_bands)."""
    nper = min(nperseg, wins.shape[-1])
    f, p = welch(wins, fs=fs, nperseg=nper, axis=-1)
    df = f[1] - f[0]
    out = np.empty(wins.shape[:2] + (len(bands),), dtype=np.float64)
    for k, (lo, hi) in enumerate(bands.values()):
        m = (f >= lo) & (f < hi)
        out[..., k] = p[..., m].sum(axis=-1) * df
    return out


def spectral_features(wins: np.ndarray, fs: float, bands: dict, ch_names: list[str], nperseg: int = 256,
                      batch: int = 512):
    """Per channel: log10 band power, relative band power, line length, log variance; plus channel-mean relative powers.

    Processed in batches of windows so long recordings do not need several GB for the spectra at once.
    """
    if len(wins) > batch:
        parts = [spectral_features(wins[i:i + batch], fs, bands, ch_names, nperseg, batch) for i in range(0, len(wins), batch)]
        return np.vstack([p[0] for p in parts]), parts[0][1]
    wins = np.asarray(wins, dtype=np.float64)
    n, ch, _ = wins.shape
    bp = band_powers(wins, fs, bands, nperseg)
    tot = bp.sum(axis=-1, keepdims=True) + EPS
    logp = np.log10(bp + EPS)
    rel = bp / tot
    ll = np.log10(np.abs(np.diff(wins, axis=-1)).mean(axis=-1) + EPS)
    lv = np.log10(wins.var(axis=-1) + EPS)
    names = []
    for c in ch_names:
        names += [f"{c}_logp_{b}" for b in bands] + [f"{c}_rel_{b}" for b in bands] + [f"{c}_linelen", f"{c}_logvar"]
    per_ch = np.concatenate([logp, rel, ll[..., None], lv[..., None]], axis=-1).reshape(n, -1)
    gmean = rel.mean(axis=1)
    names += [f"mean_rel_{b}" for b in bands]
    feats = np.concatenate([per_ch, gmean], axis=1)
    return np.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32), names


def ecg_features(wins: np.ndarray, fs: float, ch_names: list[str]):
    """Per lead: mean, std, skew, kurtosis, log band powers, line length. Wins: (n, leads, samples)."""
    wins = np.asarray(wins, dtype=np.float64)
    n, ch, _ = wins.shape
    bp = np.log10(band_powers(wins, fs, ECG_BANDS, nperseg=min(256, wins.shape[-1])) + EPS)
    st = np.stack([wins.mean(-1), wins.std(-1), skew(wins, axis=-1), kurtosis(wins, axis=-1),
                   np.log10(np.abs(np.diff(wins, axis=-1)).mean(-1) + EPS)], axis=-1)
    names = []
    for c in ch_names:
        names += [f"{c}_{k}" for k in ECG_BANDS] + [f"{c}_mean", f"{c}_std", f"{c}_skew", f"{c}_kurt", f"{c}_linelen"]
    feats = np.concatenate([bp, st], axis=-1).reshape(n, -1)
    return np.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32), names
