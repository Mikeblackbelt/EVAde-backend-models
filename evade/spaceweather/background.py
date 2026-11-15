"""Per-satellite quiet-time proton background correction, shared by training and serving.

Each GOES proton instrument has its own quiet-time floor (GCR plus instrument background), and the floors differ by
up to ~7x at >=100 MeV between GOES-15 and GOES-18. The correction maps every satellite's floor onto a reference
floor (GOES-15, the last training-era instrument):  corrected = max(flux - b_sat + b_ref, 0.1 x b_ref).
During events the flux is far above both floors, so the shift is negligible there; in quiet times it removes the
instrument offset that the forecast features would otherwise see.

b_sat per channel = 25th percentile of the hourly flux over that satellite's quiet hours (hourly >=10 MeV below
10 pfu and no hour at or above 10 pfu within 72 h). Values are re-estimated by training (saved with the model) and
the defaults below are the documented fallback. An unknown satellite gets no correction (b_sat = b_ref), with a warning.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

CHANNELS = {"p10": "proton_10mev_pfu", "p50": "proton_50mev_pfu", "p100": "proton_100mev_pfu"}
REFERENCE_SAT = "15"
DEFAULT_PATH = Path(__file__).with_name("background_default.json")
QUIET_PCTL = 25


def load_table(path: Path | None = None) -> dict:
    p = path or DEFAULT_PATH
    return json.loads(Path(p).read_text())


def correct_hourly(h: pd.DataFrame, table: dict, prefix: str = "") -> tuple[pd.DataFrame, list[str]]:
    """Apply the correction to an hourly table with p10/p50/p100 and a `sat` column (GOES number, NaN if unknown).

    `prefix` selects the data product: "" for the NCEI archive values used in training, "swpc:" for the SWPC
    operational integral feed (its GOES-R integral is computed differently from our archive integration)."""
    h = h.copy()
    warn = []
    ref = table["satellites"].get(table.get("reference", REFERENCE_SAT))
    if ref is None:
        return h, ["background table has no reference satellite: not corrected"]
    sat = h["sat"] if "sat" in h.columns else pd.Series(np.nan, index=h.index)
    keys = sat.map(lambda s: f"{prefix}{int(s)}" if pd.notna(s) else None)
    unknown = sorted({k for k in keys.dropna().unique() if k not in table["satellites"]})
    if unknown:
        warn.append(f"no background values for {', '.join(unknown)}: proton background not corrected")
    if keys.isna().any() and h["p10"].notna().any():
        warn.append("proton satellite unknown for some hours: background not corrected there")
    for col in CHANNELS:
        if col not in h.columns:
            continue
        def bval(k, col=col):
            v = table["satellites"].get(k, ref).get(col) if k else None
            return ref[col] if v is None else v
        b = keys.map(bval).astype(float)
        h[col] = np.maximum(h[col] - b + ref[col], 0.1 * ref[col]).where(h[col].notna())
    return h, warn


def quiet_mask(p10: pd.Series, guard_h: int = 72) -> pd.Series:
    """Hours with p10 below 10 pfu and no hour at or above 10 pfu within +-guard_h."""
    ev = (p10 >= 10.0).astype(float)
    near = ev.rolling(2 * guard_h + 1, center=True, min_periods=1).max() > 0
    return p10.notna() & (p10 < 10.0) & ~near


def estimate(hourly_by_sat: dict[str, pd.DataFrame], exclude: list[tuple[str, str]] | None = None) -> dict:
    """b_sat per channel from each satellite's own uncorrected hourly series. `exclude` drops (start, end) windows
    (the overlap periods kept back for validation)."""
    sats = {}
    for s, h in hourly_by_sat.items():
        m = quiet_mask(h["p10"])
        for a, b in exclude or []:
            m &= ~((h.index >= pd.Timestamp(a, tz="UTC")) & (h.index < pd.Timestamp(b, tz="UTC")))
        sats[s] = {c: float(np.nanpercentile(h.loc[m, c], QUIET_PCTL)) if h.loc[m, c].notna().any() else None
                   for c in CHANNELS}
        sats[s]["quiet_hours"] = int(m.sum())
    ok = [s for s in sats if all(sats[s][c] is not None for c in CHANNELS)]
    ref = REFERENCE_SAT if REFERENCE_SAT in ok else (sorted(ok)[-1] if ok else REFERENCE_SAT)
    return {"reference": ref, "percentile": QUIET_PCTL, "satellites": sats}


def overlap_agreement(a: pd.DataFrame, b: pd.DataFrame, table: dict | None, sat_a: str, sat_b: str) -> dict:
    """Median |log10 ratio| between two satellites over common quiet hours and common event hours, per channel."""
    a, b = a.copy(), b.copy()
    if table is not None:
        a["sat"], b["sat"] = float(sat_a), float(sat_b)
        a, _ = correct_hourly(a, table)
        b, _ = correct_hourly(b, table)
    j = a[list(CHANNELS)].add_suffix("_a").join(b[list(CHANNELS)].add_suffix("_b"), how="inner").dropna()
    quiet = quiet_mask(j["p10_a"]) & quiet_mask(j["p10_b"])
    event = (j["p10_a"] >= 10) & (j["p10_b"] >= 10)
    out = {"hours": int(len(j)), "quiet_hours": int(quiet.sum()), "event_hours": int(event.sum())}
    for c in CHANNELS:
        r = np.log10(j[f"{c}_a"].clip(lower=1e-4)) - np.log10(j[f"{c}_b"].clip(lower=1e-4))
        out[c] = {"quiet_median_abs_log10_ratio": round(float(r[quiet].abs().median()), 3) if quiet.any() else None,
                  "event_median_abs_log10_ratio": round(float(r[event].abs().median()), 3) if event.any() else None}
    return out
