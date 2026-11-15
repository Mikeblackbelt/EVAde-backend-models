"""Per-satellite quiet-time proton background correction, shared by training and serving.

Each GOES proton instrument has its own quiet-time floor (GCR plus instrument background), and the floors differ by
up to ~7x at >=100 MeV between GOES-15 and GOES-18. The correction maps every satellite's floor onto a reference
floor (GOES-15, the last training-era instrument):  corrected = max(flux - b_sat + b_ref, 0.1 x b_ref).
During events the flux is far above both floors, so the shift is negligible there; in quiet times it removes the
instrument offset that the forecast features would otherwise see.

b_sat per channel = 25th percentile of the hourly flux over that satellite's quiet hours (hourly >=10 MeV below
10 pfu and no hour at or above 10 pfu within 72 h), per satellite and calendar year (floors drift with the solar
cycle), with the nearest year or the all-years value as fallback. A trailing-window floor would track the drift
better; it is not built (proof of concept).
The reference floor is GOES-15's all-years value. Values are estimated by training (saved with the model); the
committed background_default.json is the documented fallback. An unknown satellite gets no correction, with a warning.
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
MIN_QUIET_H_YEAR = 500


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
    sats = table["satellites"]
    sat = h["sat"] if "sat" in h.columns else pd.Series(np.nan, index=h.index)
    base = sat.map(lambda s: f"{prefix}{int(s)}" if pd.notna(s) else None)
    years = pd.Series(h.index.year, index=h.index)
    per_year: dict[str, list[int]] = {}
    for k in sats:
        head, _, tail = k.rpartition(":")
        if head and len(tail) == 4 and tail.isdigit():          # "16:2021" -> year entry of "16"
            per_year.setdefault(head, []).append(int(tail))

    def pick(bk, y):
        """Background entry: same satellite and year, else its nearest year, else its all-years value."""
        ys = per_year.get(bk)
        if ys:
            return sats[f"{bk}:{min(ys, key=lambda v: abs(v - y))}"]
        return sats.get(bk)
    unknown, b = set(), {c: pd.Series(np.nan, index=h.index) for c in CHANNELS}
    for (bk, y), idx in h.groupby([base, years]).groups.items():
        e = pick(bk, y)
        if e is None:
            unknown.add(bk)
            e = ref
        for c in CHANNELS:
            b[c].loc[idx] = ref[c] if e.get(c) is None else e[c]
    if unknown:
        warn.append(f"no background values for {', '.join(sorted(unknown))}: proton background not corrected")
    if base.isna().any() and h["p10"].notna().any():
        warn.append("proton satellite unknown for some hours: background not corrected there")
    for col in CHANNELS:
        if col not in h.columns:
            continue
        bb = b[col].fillna(ref[col])
        h[col] = np.maximum(h[col] - bb + ref[col], 0.1 * ref[col]).where(h[col].notna())
    return h, warn


def quiet_mask(p10: pd.Series, guard_h: int = 72) -> pd.Series:
    """Hours with p10 below 10 pfu and no hour at or above 10 pfu within +-guard_h."""
    ev = (p10 >= 10.0).astype(float)
    near = ev.rolling(2 * guard_h + 1, center=True, min_periods=1).max() > 0
    return p10.notna() & (p10 < 10.0) & ~near


def estimate(hourly_by_sat: dict[str, pd.DataFrame], exclude: list[tuple[str, str]] | None = None,
             by_year: bool = True) -> dict:
    """b_sat per channel from each satellite's own uncorrected hourly series: an all-years value ("16") and one per
    calendar year with enough quiet hours ("16:2021"). `exclude` drops (start, end) windows (overlaps kept back
    for validation)."""
    sats = {}
    est = lambda h, m: {**{c: float(np.nanpercentile(h.loc[m, c], QUIET_PCTL)) if h.loc[m, c].notna().any() else None
                           for c in CHANNELS}, "quiet_hours": int(m.sum())}
    for s, h in hourly_by_sat.items():
        m = quiet_mask(h["p10"])
        for a, b in exclude or []:
            m &= ~((h.index >= pd.Timestamp(a, tz="UTC")) & (h.index < pd.Timestamp(b, tz="UTC")))
        sats[s] = est(h, m)
        if by_year:                          # floors drift with the solar cycle and with processing versions
            for y in sorted(set(h.index.year)):
                my = m & (h.index.year == y)
                if my.sum() >= MIN_QUIET_H_YEAR:
                    sats[f"{s}:{y}"] = est(h, my)
    ok = [s for s in sats if ":" not in s and all(sats[s][c] is not None for c in CHANNELS)]
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
