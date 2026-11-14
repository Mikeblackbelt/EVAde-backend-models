"""The risk engine must reproduce lit review/planning/worked-examples.md."""
import pytest

from evade.risk.engine import DomainInput, display_days, mdtr, radiation_domain, risk


def D(**s):
    return {k: DomainInput(v) for k, v in s.items()}


def test_scenario_a_nominal():
    d = D(cardiac=0.05, behavioral=0.12, neuro=0.03, radiation=0.15, bone=0.10)
    r = risk(d)
    assert r.r_agg == pytest.approx(0.054, abs=0.001) and r.band == "low" and not r.flag
    m = mdtr(r, d, logistics_floor_days=0.5, cumulative_dose_msv=300, dose_rate_msv_day=0.7)
    assert m["display"] == ">100 days" and m["driver"] == "risk"
    assert m["drivers_detail"]["risk"] == pytest.approx(138.0, abs=0.2)
    assert m["drivers_detail"]["radiation_limit"] == pytest.approx(392.9, abs=0.2)


def test_scenario_b_behavioral_decline():
    d = D(cardiac=0.20, behavioral=0.65, neuro=0.25, radiation=0.40, bone=0.25)
    r = risk(d)
    assert r.r_agg == pytest.approx(0.659, abs=0.001) and r.band == "high"
    assert r.flag and set(r.flag_domains) == {"behavioral", "radiation"}
    m = mdtr(r, d, logistics_floor_days=0.5, r_7d_ago=r.r_mdtr - 0.12, cumulative_dose_msv=400, dose_rate_msv_day=1.8)
    assert m["drivers_detail"]["risk"] == pytest.approx(16.4, abs=0.1)
    assert m["driver"] == "trend" and m["display"] == "16 days"


def test_scenario_c_acute():
    d = D(cardiac=0.70, behavioral=0.30, neuro=1.00, radiation=0.85, bone=0.15)
    r = risk(d, overrides=["O4"])
    assert r.r_agg == pytest.approx(0.884, abs=0.001) and r.band == "critical"
    m = mdtr(r, d, logistics_floor_days=0.5, r_7d_ago=r.r_mdtr - 0.40)
    assert m["drivers_detail"]["extreme_domain"] == pytest.approx(2.0)
    assert m["value_days"] == 0.5 and m["driver"] == "acute" and m["display"] == "<5 days" and m["status"] == "ok"


@pytest.mark.parametrize("dom,r_agg,mdtr_days", [("cardiac", 0.747, 2.0), ("radiation", 0.599, 2.0), ("bone", 0.451, 2.0)])
def test_single_domain_at_one(dom, r_agg, mdtr_days):
    d = {k: DomainInput(0.10) for k in ("cardiac", "behavioral", "neuro", "radiation", "bone")}
    d[dom] = DomainInput(1.0)
    r = risk(d)
    assert r.r_agg == pytest.approx(r_agg, abs=0.001) and r.r_disp >= 0.60 and r.band == "high"
    assert mdtr(r, d, logistics_floor_days=0.5)["value_days"] == mdtr_days


def test_low_quality_floor_does_not_flag():
    d = {"cardiac": DomainInput(0.0, q=0.2)}
    r = risk(d)
    assert r.domains["cardiac"]["s"] == 0.30 and not r.flag


def test_shadow_inputs_excluded_from_flag():
    d = {"behavioral": DomainInput(0.5, s_validated=0.2)}
    assert not risk(d).flag


def test_window_cannot_be_met_and_withheld():
    d = D(cardiac=1.0)
    r = risk(d)
    assert mdtr(r, d, logistics_floor_days=6.0)["status"] == "window_cannot_be_met"
    assert mdtr(r, d, logistics_floor_days=None)["status"] == "withheld"


def test_display_rules():
    assert display_days(4.9) == "<5 days" and display_days(7.3) == "7.5 days" and display_days(42.4) == "42 days"
    assert display_days(101) == ">100 days"


def test_radiation_domain_anchors():
    s, parts = radiation_domain(cumulative_dose_msv=300)          # f = 0.5 -> 0.5
    assert s.s == pytest.approx(0.5)
    s, _ = radiation_domain(cumulative_dose_msv=0, spe_dose_msv=125)   # 0.5 of 250 -> 0.6
    assert s.s == pytest.approx(0.6)
    s, _ = radiation_domain(cumulative_dose_msv=0, rate_ratio_to_7d_median=1.5)
    assert s.s == pytest.approx(0.2)
    assert radiation_domain(0, seu_per_hour=3)[0].q < 0.5
