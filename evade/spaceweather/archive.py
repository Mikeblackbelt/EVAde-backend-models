"""Readers for the historical archives used in training (not used when serving).

- GOES-R SEISS SGPS L2 5-min averages (NCEI daily netCDF4): only differential channels (P1-P10) and the >500 MeV
  integral channel are published, so the >=10/50/100 MeV integral flux is integrated here: sum of j x dE over the
  channels above the threshold (the channel that straddles it counts pro rata), plus the >500 MeV channel, averaged
  over the two sensor units that pass QC. This approximates, but is not identical to, the SWPC operational integral.
- GOES-R EXIS XRS L2 1-min science averages (NCEI yearly netCDF4), already in true irradiance.
- SOHO/LASCO CDAW CME catalogue (univ_all.txt), the historical stand-in for DONKI CME analyses.
All files are untrusted data: netCDF is read with h5py as arrays only; text is parsed, never executed.
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

J2000 = pd.Timestamp("2000-01-01T12:00:00Z")
THRESH_KEV = {"proton_10mev_pfu": 10_000.0, "proton_50mev_pfu": 50_000.0, "proton_100mev_pfu": 100_000.0}


def _arr(h, name, fill_attr="_FillValue"):
    d = h[name]
    a = d[()].astype(float)
    if fill_attr in d.attrs:
        fv = np.asarray(d.attrs[fill_attr]).astype(float).ravel()
        for f in fv:
            a[a == f] = np.nan
    return a


def integral_from_differential(j: np.ndarray, lo: np.ndarray, hi: np.ndarray, p11: np.ndarray, e_kev: float) -> np.ndarray:
    """j: (time, channels) in protons/(cm2 sr keV s); lo/hi: (channels,) keV; p11: (time,) >500 MeV integral.
    Returns the integral flux above e_kev in pfu. NaN if any needed channel is missing."""
    frac = np.clip((hi - np.maximum(lo, e_kev)) / (hi - lo), 0.0, 1.0)            # share of each channel above E
    use = frac > 0
    contrib = j[:, use] * ((hi - lo) * frac)[use]
    out = contrib.sum(axis=1) + np.nan_to_num(p11, nan=0.0)
    out[np.isnan(contrib).any(axis=1)] = np.nan
    return out


def read_sgps_day(path: Path) -> pd.DataFrame:
    import h5py
    with h5py.File(path, "r") as h:
        t = J2000 + pd.to_timedelta(_arr(h, "L2_SciData_TimeStamp"), unit="s")
        j = _arr(h, "AvgDiffProtonFlux")                         # (time, unit, channel)
        valid = h["DiffValidL1bSamplesInAvg"][()]
        j[(valid == 0) | (valid == 4294967295)] = np.nan
        j[j < 0] = np.nan
        p11 = _arr(h, "AvgIntProtonFlux")                        # (time, unit)
        p11[p11 < 0] = np.nan
        lo, hi = _arr(h, "DiffProtonLowerEnergy"), _arr(h, "DiffProtonUpperEnergy")   # (unit, channel)
    import warnings
    out = {}
    for name, e in THRESH_KEV.items():
        per_unit = [integral_from_differential(j[:, u, :], lo[u], hi[u], p11[:, u], e) for u in range(j.shape[1])]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)          # both units missing -> NaN, expected
            out[name] = np.nanmean(np.vstack(per_unit), axis=0) if per_unit else np.nan
    df = pd.DataFrame(out, index=pd.DatetimeIndex(t))
    return df[df.index.notna()]


def read_xrs_year(path: Path) -> pd.DataFrame:
    import h5py
    with h5py.File(path, "r") as h:
        t = J2000 + pd.to_timedelta(_arr(h, "time"), unit="s")
        b = _arr(h, "xrsb_flux")
        a = _arr(h, "xrsa_flux")
        fb = h["xrsb_flag"][()]
        fa = h["xrsa_flag"][()]
    b[(fb != 0) | ~(b > 0)] = np.nan
    a[(fa != 0) | ~(a > 0)] = np.nan
    df = pd.DataFrame({"xray_long_wm2": b, "xray_short_wm2": a}, index=pd.DatetimeIndex(t))
    return df[df.index.notna()]


# ---------------- SOHO/LASCO CDAW CME catalogue ----------------
_ROW = re.compile(r"^(\d{4}/\d{2}/\d{2})\s+(\d{2}:\d{2}:\d{2})\s+(\S+)\s+(\d+)\s+(\d+|-+)")


def read_cdaw(path: Path) -> pd.DataFrame:
    """CDAW univ_all.txt -> DataFrame(t, speed_kms, width_deg, halo). Width 360 = halo; speed is the linear fit."""
    rows = []
    with open(path, "r", encoding="ascii", errors="replace") as f:
        for line in f:
            m = _ROW.match(line)
            if not m:
                continue
            date, tm, cpa, width, speed = m.groups()
            if not speed.isdigit():
                continue
            rows.append((pd.Timestamp(f"{date.replace('/', '-')}T{tm}Z"), float(speed), float(width), cpa == "Halo"))
    return pd.DataFrame(rows, columns=["t", "speed_kms", "width_deg", "halo"])
