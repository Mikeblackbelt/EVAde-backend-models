"""Risk score R (aggregation v2, noisy-OR in log space) and maximum days to return (MDTR).

Implements lit review/planning/risk-score.md section 3-4 and max-days-to-return.md. Every constant is a
design assumption from those documents, not a fitted or clinical value. Decision support only.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

DOMAINS = ("cardiac", "behavioral", "neuro", "radiation", "bone")
INFLUENCE = {"cardiac": 1.0, "behavioral": 1.0, "radiation": 0.8, "neuro": 0.6, "bone": 0.6}
DEAD_ZONE, CAP = 0.10, 0.75
FLAG_S, FLAG_FLOOR, OVERRIDE_FLOOR = 0.30, 0.60, 0.80
LOW_Q = 0.5
BANDS = ((0.2, "low"), (0.4, "guarded"), (0.6, "elevated"), (0.8, "high"), (1.01, "critical"))
ACUTE_OVERRIDES = {"O2", "O3", "O4", "O5"}
CAREER_LIMIT_MSV, DOSE_MARGIN_MSV, SPE_LIMIT_MSV = 600.0, 25.0, 250.0


def interp(x: float, anchors: list[tuple[float, float]]) -> float:
    """Piecewise-linear anchor table; below the first anchor = its value, above the last = the last value."""
    if x <= anchors[0][0]:
        return anchors[0][1]
    for (x0, y0), (x1, y1) in zip(anchors, anchors[1:]):
        if x <= x1:
            return y0 + (y1 - y0) * (x - x0) / (x1 - x0)
    return anchors[-1][1]


def band(r: float) -> str:
    return next(name for edge, name in BANDS if r < edge)


@dataclass
class DomainInput:
    s: float                      # domain score in [0, 1]
    q: float = 1.0                # data quality in [0, 1]
    s_validated: float | None = None   # score without shadow-mode inputs, used for the flag and T_extreme

    def eligible_s(self) -> float | None:
        if self.q < LOW_Q:
            return None           # low-data floor never trips the flag
        return self.s if self.s_validated is None else self.s_validated


@dataclass
class RiskResult:
    r_agg: float
    r_disp: float
    r_mdtr: float
    band: str
    flag: bool
    flag_domains: list[str]
    override_ids: list[str]
    domains: dict = field(default_factory=dict)


def risk(domains: dict[str, DomainInput], overrides: list[str] | None = None) -> RiskResult:
    overrides = sorted(set(overrides or []))
    L, per = 0.0, {}
    for d in DOMAINS:
        di = domains.get(d, DomainInput(0.0))
        s = di.s if di.q >= LOW_Q else max(di.s, 0.30)     # missing data is not good news
        p = INFLUENCE[d] * CAP * max(0.0, (s - DEAD_ZONE) / (1 - DEAD_ZONE))
        L += -math.log(1 - p)
        per[d] = {"s": round(s, 4), "p": round(p, 4), "q": di.q, "low_confidence": di.q < LOW_Q}
    r_agg = 0.005 + 0.99 * (1 - math.exp(-L))
    flag_domains = [d for d in DOMAINS if d in domains and (domains[d].eligible_s() or 0.0) >= FLAG_S]
    r_disp = max(r_agg, FLAG_FLOOR) if flag_domains else r_agg
    r_mdtr = r_agg
    if overrides:
        r_disp, r_mdtr = max(r_disp, OVERRIDE_FLOOR), max(r_mdtr, OVERRIDE_FLOOR)
    return RiskResult(r_agg, r_disp, r_mdtr, band(r_disp), bool(flag_domains), flag_domains, overrides, per)


def display_days(v: float) -> str:
    if v < 5:
        return "<5 days"
    if v > 100:
        return ">100 days"
    return f"{round(v * 2) / 2:g} days" if v < 10 else f"{round(v):d} days"


def mdtr(res: RiskResult, domains: dict[str, DomainInput], *, logistics_floor_days: float | None,
         r_7d_ago: float | None = None, cumulative_dose_msv: float | None = None,
         dose_rate_msv_day: float | None = None, bone_loss_pct: float | None = None,
         bone_slope_pct_day: float | None = None) -> dict:
    """Maximum days to return: min of the risk curve (with trend) and the hard limits, floored by logistics."""
    if logistics_floor_days is None:
        return {"status": "withheld", "value_days": None, "display": None, "driver": None, "drivers_detail": {}}
    terms = {"risk": 150.0 * (1 - res.r_mdtr) ** 1.5}
    m = math.exp(-5 * max(0.0, res.r_mdtr - r_7d_ago)) if r_7d_ago is not None else 1.0
    terms["risk"] *= m
    if cumulative_dose_msv is not None and dose_rate_msv_day and dose_rate_msv_day > 0:
        terms["radiation_limit"] = max(0.0, (CAREER_LIMIT_MSV - cumulative_dose_msv - DOSE_MARGIN_MSV) / dose_rate_msv_day)
    if bone_loss_pct is not None and bone_slope_pct_day is not None and bone_slope_pct_day > 0.01:
        terms["bone_limit"] = max(0.0, (10.0 - bone_loss_pct) / bone_slope_pct_day)
    if ACUTE_OVERRIDES & set(res.override_ids):
        terms["acute"] = 0.5
    ext = [9.0 - 35.0 * (s - 0.80) for s in (di.eligible_s() for di in domains.values()) if s is not None and s >= 0.80]
    if ext:
        terms["extreme_domain"] = min(ext)
    driver = min(terms, key=terms.get)
    if driver == "risk" and m < 1.0:
        driver = "trend"
    value = terms[min(terms, key=terms.get)]
    status = "ok"
    if value < logistics_floor_days:
        status, value = "window_cannot_be_met", logistics_floor_days
    return {"status": status, "value_days": round(value * 2) / 2, "display": display_days(value), "driver": driver,
            "trend_multiplier": round(m, 3), "drivers_detail": {k: round(v, 2) for k, v in terms.items()}}


def radiation_domain(cumulative_dose_msv: float, spe_dose_msv: float = 0.0, rate_ratio_to_7d_median: float = 1.0,
                     seu_per_hour: int = 0) -> tuple[DomainInput, dict]:
    """Radiation domain from risk-score.md: s_rad = max(R1 cumulative, R2 solar particle event, R3 rate anomaly)."""
    r1 = interp(cumulative_dose_msv / CAREER_LIMIT_MSV, [(0.05, 0.0), (0.25, 0.25), (0.5, 0.5), (0.75, 0.8), (1.0, 1.0)])
    r2 = interp(spe_dose_msv / SPE_LIMIT_MSV, [(0.0, 0.0), (0.10, 0.2), (0.5, 0.6), (1.0, 1.0)])
    r3 = interp(rate_ratio_to_7d_median, [(1.5, 0.0), (3.0, 0.6), (10.0, 1.0)]) if rate_ratio_to_7d_median >= 1.5 else 0.0
    if 1.5 <= rate_ratio_to_7d_median < 3.0:
        r3 = interp(rate_ratio_to_7d_median, [(1.5, 0.2), (3.0, 0.6)])
    q = 0.4 if seu_per_hour > 1 else 1.0          # data-trust rule: >1 SEU per hour -> low confidence
    return DomainInput(max(r1, r2, r3), q), {"r1_cumulative": round(r1, 3), "r2_spe": round(r2, 3), "r3_rate": round(r3, 3)}
