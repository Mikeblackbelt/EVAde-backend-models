"""Hourly aggregation, forecast features and labels. One implementation for training and serving (train/serve parity).

Input to everything here is the normalized schema (schema.FIELDS) at any cadence. Training builds the same records
table from the NCEI GOES archive (train.py), the API builds it from a provider's Observations.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .schema import Observations

LOOKBACK_H = 48          # features use at most the last 48 hours (the NOAA 3-day feeds cover it)
HORIZON_H = 24           # forecast horizon
S1_PFU = 10.0            # NOAA S1 threshold on >=10 MeV integral protons
M_CLASS, X_CLASS = 1e-5, 1e-4
FLOOR = {"p": 1e-3, "x": 1e-9}

FEATURES = [
    "lp10", "lp50", "lp100", "lp10_max24", "lp10_min48", "dlp10_1", "dlp10_3", "dlp10_6",
    "hard50", "hard100", "lx", "lx_max3", "lx_max6", "lx_max24", "lx_max48", "lx_fluence24",
    "n_m48", "n_x48", "hours_since_m",
]
HOURLY_COLS = ["p10", "p50", "p100", "xmax", "xmean"]


def records_frame(obs: Observations) -> pd.DataFrame:
    """Observations -> long table indexed by UTC time with normalized field columns (NaN where absent)."""
    rows = [{"t": r.t, **r.values} for r in obs.records]
    if not rows:
        return pd.DataFrame(columns=["proton_10mev_pfu"]).rename_axis("t")
    df = pd.DataFrame(rows)
    df["t"] = pd.to_datetime(df["t"], utc=True)
    return df.groupby("t").last().sort_index()


def to_hourly(df: pd.DataFrame, end: pd.Timestamp | None = None) -> pd.DataFrame:
    """Hourly table on a complete hour grid. Hour h holds samples in [h, h+1h).

    p10/p50/p100: hourly mean of the integral proton flux (the mean damps single-sample spikes, a proxy for
    NOAA's three-consecutive-5-min-points event rule). xmax / xmean: hourly max / mean of the 0.1-0.8 nm X-ray flux.
    """
    def col(name):
        return df[name] if name in df.columns else pd.Series(dtype=float, index=df.index)
    if df.empty:
        return pd.DataFrame(columns=HOURLY_COLS)
    g = lambda s, how: getattr(s.dropna().resample("1h"), how)()
    out = pd.DataFrame({"p10": g(col("proton_10mev_pfu"), "mean"), "p50": g(col("proton_50mev_pfu"), "mean"),
                        "p100": g(col("proton_100mev_pfu"), "mean"), "xmax": g(col("xray_long_wm2"), "max"),
                        "xmean": g(col("xray_long_wm2"), "mean")})
    start = df.index.min().floor("1h")
    stop = (end if end is not None else df.index.max()).floor("1h")
    return out.reindex(pd.date_range(start, stop, freq="1h", tz="UTC"))


def feature_frame(h: pd.DataFrame) -> pd.DataFrame:
    """Forecast features for every hour of a complete hourly grid (NaN where inputs are missing)."""
    lg = lambda s, k: np.log10(s.clip(lower=FLOOR[k]))
    lp10, lp50, lp100 = lg(h["p10"], "p"), lg(h["p50"], "p"), lg(h["p100"], "p")
    lx = lg(h["xmax"], "x")
    roll = lambda s, n, how: getattr(s.rolling(n, min_periods=1), how)()
    f = pd.DataFrame(index=h.index)
    f["lp10"], f["lp50"], f["lp100"] = lp10, lp50, lp100
    f["lp10_max24"], f["lp10_min48"] = roll(lp10, 24, "max"), roll(lp10, LOOKBACK_H, "min")
    for k in (1, 3, 6):
        f[f"dlp10_{k}"] = lp10 - lp10.shift(k)
    f["hard50"], f["hard100"] = lp50 - lp10, lp100 - lp10
    f["lx"] = lx
    for n in (3, 6, 24, LOOKBACK_H):
        f[f"lx_max{n}"] = roll(lx, n, "max")
    fl = h["xmean"].rolling(24, min_periods=1).sum() * 3600.0           # J m-2 over the last 24 h (approx.)
    f["lx_fluence24"] = np.log10(fl.where(fl > 0).clip(lower=1e-6))
    m = (h["xmax"] >= M_CLASS).astype(float).where(h["xmax"].notna())
    x = (h["xmax"] >= X_CLASS).astype(float).where(h["xmax"].notna())
    f["n_m48"] = m.rolling(LOOKBACK_H, min_periods=1).sum()
    f["n_x48"] = x.rolling(LOOKBACK_H, min_periods=1).sum()
    idx = pd.Series(np.arange(len(h), dtype=float), index=h.index)
    last_m = idx.where(m == 1).ffill()
    f["hours_since_m"] = (idx - last_m).clip(upper=LOOKBACK_H).fillna(LOOKBACK_H)
    # beyond the lookback the series is unknown; cap so a long archive and a 3-day feed give the same value
    return f[FEATURES]


def labels(h: pd.DataFrame) -> pd.DataFrame:
    """y = 1 if the hourly >=10 MeV flux reaches S1 in any of the next 24 hours; eligible = quiet now (onset task)."""
    fut = pd.concat([h["p10"].shift(-k) for k in range(1, HORIZON_H + 1)], axis=1)
    n_fut = fut.notna().sum(axis=1)
    y = (fut.max(axis=1) >= S1_PFU).astype(int)
    eligible = h["p10"].notna() & (h["p10"] < S1_PFU) & (n_fut >= HORIZON_H // 2)
    return pd.DataFrame({"y": y, "eligible": eligible}, index=h.index)


def onset_times(h: pd.DataFrame, quiet_gap_h: int = 24) -> list[pd.Timestamp]:
    """Hours where the hourly >=10 MeV flux crosses S1 after at least `quiet_gap_h` hours below it (independent events)."""
    above = (h["p10"] >= S1_PFU).fillna(False).to_numpy()
    out, last_above = [], -10**9
    for i, a in enumerate(above):
        if a:
            if i - last_above > quiet_gap_h:
                out.append(h.index[i])
            last_above = i
    return out


def latest_features(obs: Observations, now: pd.Timestamp | None = None) -> tuple[dict | None, dict]:
    """Features at the most recent complete-enough hour of an Observations object, plus coverage diagnostics."""
    df = records_frame(obs)
    if df.empty or "proton_10mev_pfu" not in df.columns:
        return None, {"reason": "no proton data"}
    end = now if now is not None else df.index.max()
    h = to_hourly(df, end=end)
    h = h.iloc[-(LOOKBACK_H + 1):]
    f = feature_frame(h)
    row = f.iloc[-1]
    cov = {"hours": int(len(h)), "proton_hours": int(h["p10"].notna().sum()), "xray_hours": int(h["xmax"].notna().sum()),
           "feature_hour_utc": h.index[-1].isoformat()}
    if not np.isfinite(row["lp10"]):
        return None, {**cov, "reason": "no >=10 MeV proton value in the latest hour"}
    return {k: (float(v) if np.isfinite(v) else None) for k, v in row.items()}, cov
