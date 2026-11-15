"""Space-weather adapter, nowcast, dose mapping, features, training parser and API endpoint. No network anywhere."""
import json
import os
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import pytest

os.environ["EVADE_SPACEWX_DISABLE_NETWORK"] = "1"
os.environ.pop("EVADE_MODEL_DIR", None)

from evade.risk.engine import radiation_domain  # noqa: E402
from evade.spaceweather import model as SW  # noqa: E402
from evade.spaceweather import providers as P  # noqa: E402
from evade.spaceweather.features import (FEATURES, LOOKBACK_H, feature_frame, labels, latest_features,  # noqa: E402
                                         onset_times, records_frame, to_hourly)
from evade.spaceweather.schema import Observations, Record  # noqa: E402

NOW = datetime.now(timezone.utc).replace(second=0, microsecond=0)


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def boom(*a, **k):
        raise AssertionError("network access in tests")
    monkeypatch.setattr("urllib.request.urlopen", boom)
    P.reset_registry()
    yield
    P.reset_registry()


def series_obs(p10, p50=None, p100=None, xl=None, step_min=5, end=NOW, source="test"):
    """Observations from per-sample lists, the last sample at `end`."""
    n = len(p10)
    recs = []
    for i in range(n):
        t = end - timedelta(minutes=step_min * (n - 1 - i))
        v = {"proton_10mev_pfu": p10[i]}
        if p50 is not None:
            v["proton_50mev_pfu"] = p50[i]
        if p100 is not None:
            v["proton_100mev_pfu"] = p100[i]
        if xl is not None:
            v["xray_long_wm2"] = xl[i]
        recs.append(Record.make(t, **v))
    return Observations(source, recs)


# ---------------- adapter normalization ----------------
def iso(dt):
    return dt.strftime("%Y-%m-%dT%H:%M:%SZ")


def fake_swpc(fail=()):
    t1, t2 = NOW - timedelta(minutes=10), NOW - timedelta(minutes=5)
    feeds = {
        "integral-protons": [{"time_tag": iso(t), "satellite": 18, "flux": f, "energy": e}
                             for t in (t1, t2) for e, f in ((">=10 MeV", 0.5), (">=50 MeV", 0.2), (">=100 MeV", 0.1),
                                                            (">=1 MeV", 3.0))],
        "xrays": [{"time_tag": iso(t2), "satellite": 18, "flux": 2.3e-6, "energy": "0.1-0.8nm"},
                  {"time_tag": iso(t2), "satellite": 18, "flux": 1e-7, "energy": "0.05-0.4nm"}],
        "k-index": [["time_tag", "Kp", "a_running", "station_count"], [t2.strftime("%Y-%m-%d %H:%M:%S.000"), "5.33", "56", "8"]],
        "rtsw_wind": [{"time_tag": t2.strftime("%Y-%m-%dT%H:%M:%S"), "active": True, "source": "SOLAR1",
                       "proton_speed": 512.3, "proton_density": 4.1},
                      {"time_tag": t2.strftime("%Y-%m-%dT%H:%M:%S"), "active": False, "source": "ACE",
                       "proton_speed": 999.0, "proton_density": 99.0}],
        "rtsw_mag": [{"time_tag": t2.strftime("%Y-%m-%dT%H:%M:%S"), "active": True, "bx_gsm": 1, "by_gsm": 2, "bz_gsm": -7.5}],
    }
    calls = []

    def fetch(url):
        calls.append(url)
        for k, v in feeds.items():
            if k in url:
                if k in fail:
                    raise P.ProviderUnavailable("simulated outage")
                return v
        raise P.ProviderUnavailable("unknown feed")
    return fetch, calls


def test_noaa_adapter_normalizes_all_feeds():
    fetch, calls = fake_swpc()
    obs = P.NoaaSwpcProvider(fetch_json=fetch).fetch(72)
    assert all(u.startswith("https://services.swpc.noaa.gov/") for u in calls)
    assert any("3-day" in u for u in calls)
    assert obs.latest("proton_10mev_pfu")[1] == 0.5 and obs.latest("proton_100mev_pfu")[1] == 0.1
    assert obs.latest("xray_long_wm2")[1] == pytest.approx(2.3e-6)
    assert obs.latest("kp")[1] == pytest.approx(5.33)
    assert obs.latest("sw_speed_kms")[1] == pytest.approx(512.3) and obs.latest("bz_gsm_nt")[1] == -7.5
    assert obs.warnings == []
    assert not any("1 MeV" in k for r in obs.records for k in r.values)      # unmapped channels dropped
    assert obs.latest("proton_satellite")[1] == 18 and obs.meta["background_prefix"] == "swpc:"


def test_noaa_adapter_optional_feeds_warn_and_protons_required():
    fetch, _ = fake_swpc(fail=("xrays", "k-index"))
    obs = P.NoaaSwpcProvider(fetch_json=fetch).fetch(72)
    assert obs.has("proton_10mev_pfu") and not obs.has("xray_long_wm2")
    assert any("x-ray" in w for w in obs.warnings) and any("Kp" in w for w in obs.warnings)
    fetch, _ = fake_swpc(fail=("integral-protons",))
    with pytest.raises(P.ProviderUnavailable):
        P.NoaaSwpcProvider(fetch_json=fetch).fetch(72)


def test_fill_values_and_bad_values_drop_out():
    r = Record.make(NOW, proton_10mev_pfu=-99999, xray_long_wm2="nan", kp=12, bz_gsm_nt=-3.0, junk=5)
    assert r.values == {"bz_gsm_nt": -3.0}


def test_http_rejects_other_hosts_without_network():
    with pytest.raises(P.ProviderUnavailable):
        P.http_get_json("https://example.com/x.json")
    with pytest.raises(P.ProviderUnavailable):
        P.http_get_json("http://services.swpc.noaa.gov/json/x.json")      # https only


def test_http_tolerates_nul_padding(monkeypatch):
    class Resp:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def read(self, n):
            return b'[{"time_tag": "2026-11-15T01:30:00Z", "flux": 0.2, "energy": ">=10 MeV"}]' + b"\x00" * 50
    monkeypatch.setattr("urllib.request.urlopen", lambda req, timeout: Resp())
    assert P.http_get_json("https://services.swpc.noaa.gov/json/goes/primary/integral-protons-3-day.json")[0]["flux"] == 0.2


def test_cache_ttl_and_failure_caching():
    class Inner(P.SpaceWeatherProvider):
        name = "inner"
        n, fail = 0, False

        def fetch(self, hours=72):
            Inner.n += 1
            if Inner.fail:
                raise P.ProviderUnavailable("down")
            return series_obs([1.0])
    clock = [0.0]
    c = P.CachedProvider(Inner(), ttl_s=300, fail_ttl_s=60, clock=lambda: clock[0])
    c.fetch(); c.fetch()
    assert Inner.n == 1
    clock[0] = 301; Inner.fail = True
    with pytest.raises(P.ProviderUnavailable):
        c.fetch()
    with pytest.raises(P.ProviderUnavailable):
        c.fetch()                       # failure cached: no second call inside 60 s
    assert Inner.n == 2
    clock[0] = 400; Inner.fail = False
    c.fetch()
    assert Inner.n == 3


def test_file_provider_replay(tmp_path):
    old = datetime(2017, 9, 10, 18, 0, tzinfo=timezone.utc)
    recs = [{"t_utc": (old - timedelta(hours=k)).isoformat(), "proton_10mev_pfu": 100.0 - k} for k in range(100)]
    f = tmp_path / "storm.json"
    f.write_text(json.dumps({"source": "goes15-2017-09", "records": recs}))
    obs = P.FileProvider(f).fetch(72)
    assert len(obs.records) == 73
    assert abs((obs.latest("proton_10mev_pfu")[0] - datetime.now(timezone.utc)).total_seconds()) < 60
    bad = tmp_path / "bad.json"
    bad.write_text("{not json")
    with pytest.raises(P.ProviderUnavailable):
        P.FileProvider(bad).fetch()


# ---------------- nowcast thresholds ----------------
@pytest.mark.parametrize("pfu,level", [(0.3, 0), (9.99, 0), (10.0, 1), (99.0, 1), (100.0, 2), (1e3, 3), (1e4, 4), (2e5, 5)])
def test_s_scale(pfu, level):
    assert SW.scale_level(pfu, SW.S_SCALE) == level


@pytest.mark.parametrize("x,level,cls", [(9.9e-6, 0, "C9.9"), (1e-5, 1, "M1.0"), (1.99e-5, 1, "M1.9"), (5e-5, 2, "M5.0"),
                                         (1e-4, 3, "X1.0"), (1e-3, 4, "X10.0"), (2.5e-3, 5, "X25.0"), (3e-8, 0, "A3.0")])
def test_r_scale_and_flare_class(x, level, cls):
    assert SW.scale_level(x, SW.R_SCALE) == level
    assert SW.flare_class(x) == cls


def test_nowcast_spe_flag_needs_three_points():
    quiet = SW.nowcast(series_obs([0.4] * 20 + [12.0]))          # one 5-min point above S1: not yet an event
    assert quiet["spe_in_progress"] is False and quiet["s_scale"] == 1
    ev = SW.nowcast(series_obs([0.4] * 20 + [12.0, 15.0, 20.0], xl=[1e-6] * 22 + [2e-4]))
    assert ev["spe_in_progress"] is True and ev["r_scale"] == 3 and ev["flare_class_max1h"] == "X2.0"
    obs = series_obs([0.5]); obs.records.append(Record.make(NOW, kp=7.0))
    assert SW.nowcast(obs)["g_scale"] == 3


def test_nowcast_stale_warning():
    nc = SW.nowcast(series_obs([0.5] * 3, end=NOW - timedelta(hours=2)))
    assert any("old" in w for w in nc["warnings"])


# ---------------- dose mapping ----------------
def test_quiet_dose_equals_gcr_baseline():
    d = SW.dose_estimate(series_obs([0.4] * 30, [0.15] * 30, [0.1] * 30), "deep_space", "spacecraft")
    assert d["estimated"]["spe_msv_per_day"] == 0.0
    assert d["dose_equivalent_rate_msv_per_day"] == pytest.approx(1.8)
    assert d["rate_ratio_to_quiet_baseline"] == pytest.approx(1.0) and d["spe_active"] is False
    rin = SW.radiation_inputs(d, 100.0)
    di, parts = radiation_domain(100.0, rin["spe_dose_msv"], rin["rate_ratio_to_7d_median"])
    assert parts["r2_spe"] == 0.0 and parts["r3_rate"] == 0.0


def test_spe_dose_rises_with_flux_and_shielding_order():
    n = 36                                   # 3 h of an event at 5-min cadence
    obs = series_obs([1000.0] * n, [100.0] * n, [30.0] * n)
    rates = {s: SW.dose_estimate(obs, "deep_space", s)["estimated"]["spe_msv_per_day"] for s in SW.SHIELDING}
    assert rates["eva_suit"] > rates["spacecraft"] > rates["storm_shelter"] > 0
    d = SW.dose_estimate(obs, "deep_space", "spacecraft")
    assert d["spe_active"] and d["spe_dose_msv_event_so_far"] > 0
    assert d["spe_projected_dose_msv_24h"] > d["spe_dose_msv_event_so_far"]
    assert d["rate_ratio_to_quiet_baseline"] > 1.5
    leo = SW.dose_estimate(obs, "leo_iss", "spacecraft")["estimated"]["spe_msv_per_day"]
    assert leo < d["estimated"]["spe_msv_per_day"]
    rin = SW.radiation_inputs(d, 100.0)
    _, parts = radiation_domain(100.0, rin["spe_dose_msv"], rin["rate_ratio_to_7d_median"])
    assert parts["r3_rate"] > 0


def test_formula_value_by_hand():
    # spacecraft, deep space: 4*pi x (10.2 - 0.2) x 6.5 x 1.602e-10 Gy/s x 2 x 3600 x 1e6 microSv/h
    v = SW.spe_rate_usv_h({"proton_100mev_pfu": 10.2}, "deep_space", "spacecraft")
    assert v == pytest.approx(4 * np.pi * 10.0 * 6.5 * 1.602e-10 * 2 * 3600 * 1e6)


def test_measured_rate_overrides_estimate_and_o5_candidate():
    obs = series_obs([5e4] * 36, [5e3] * 36, [2e3] * 36)
    d = SW.dose_estimate(obs, "deep_space", "spacecraft", measured_dose_rate_usv_h=50.0)
    assert d["source"] == "measured" and d["dose_equivalent_rate_msv_per_day"] == pytest.approx(1.2)
    assert SW.o5_candidate(d)["candidate"] is True           # extreme flux -> projection beyond 250 mSv
    q = SW.dose_estimate(series_obs([0.4] * 5), "deep_space", "spacecraft")
    assert SW.o5_candidate(q)["candidate"] is False


# ---------------- features: train/serve parity and labels ----------------
def synthetic_hourly_obs(hours=200, seed=0, event_at=150):
    rng = np.random.default_rng(seed)
    end = pd.Timestamp("2015-06-01T00:55Z")         # first sample falls on a full hour, so hour bin k = i // 12
    recs = []
    for i in range(hours * 12):
        t = end - pd.Timedelta(minutes=5 * (hours * 12 - 1 - i))
        hr = i // 12
        p = 0.3 * np.exp(rng.normal(0, 0.1)) + (200.0 * np.exp(-(hr - event_at) / 10) if hr >= event_at else 0)
        x = 1e-6 * np.exp(rng.normal(0, 0.3)) + (3e-4 if hr == event_at - 2 else 0)
        recs.append(Record.make(t.to_pydatetime(), proton_10mev_pfu=p, proton_50mev_pfu=p / 5, proton_100mev_pfu=p / 20,
                                xray_long_wm2=x))
    return Observations("synthetic", recs)


def test_feature_parity_long_history_vs_trailing_window():
    obs = synthetic_hourly_obs()
    h = to_hourly(records_frame(obs))
    full = feature_frame(h)
    for k in (60, 140, 160, 199):
        cut = Observations("cut", [r for r in obs.records if pd.Timestamp(r.t) < h.index[k] + pd.Timedelta(hours=1)])
        row, cov = latest_features(cut)
        assert cov["feature_hour_utc"] == h.index[k].isoformat()
        for f in FEATURES:
            a, b = full.iloc[k][f], row[f]
            assert (b is None and not np.isfinite(a)) or a == pytest.approx(b, rel=1e-9, abs=1e-12), f


def test_latest_features_bounded_by_time_span():
    import time as _t
    far = [Record.make(NOW - timedelta(days=5 * 365), proton_10mev_pfu=0.5), Record.make(NOW, proton_10mev_pfu=0.6)]
    t0 = _t.perf_counter()
    row, cov = latest_features(Observations("x", far))
    assert _t.perf_counter() - t0 < 1.0
    assert cov["hours"] <= LOOKBACK_H + 1 and row["lp10"] == pytest.approx(np.log10(0.6))


def test_latest_features_anchor_to_protons_across_hour_boundary():
    hh = pd.Timestamp("2026-11-14T10:00Z")
    recs = [Record.make((hh - pd.Timedelta(minutes=5 * k)).to_pydatetime() + timedelta(minutes=55), proton_10mev_pfu=0.3)
            for k in range(700)]                                       # protons end at 10:55
    recs += [Record.make((hh + pd.Timedelta(hours=1, minutes=2) - pd.Timedelta(minutes=k)).to_pydatetime(),
                         xray_long_wm2=1e-6) for k in range(3000)]    # X-rays end at 11:02
    row, cov = latest_features(Observations("lag", recs))
    assert row is not None and cov["feature_hour_utc"] == hh.isoformat()
    assert cov["proton_hours"] == LOOKBACK_H + 1


def test_labels_onset_task():
    obs = synthetic_hourly_obs()
    h = to_hourly(records_frame(obs))
    lab = labels(h)
    ons = onset_times(h)
    assert len(ons) == 1 and ons[0] == h.index[150]
    assert lab["y"].iloc[149] == 1 and lab["y"].iloc[126] == 1 and lab["y"].iloc[125] == 0
    assert not lab["eligible"].iloc[150]                # in-event hours are not onset samples
    assert not lab["eligible"].iloc[-1]                 # no future to label


def test_features_hours_since_m_and_counts():
    obs = synthetic_hourly_obs()
    f = feature_frame(to_hourly(records_frame(obs)))
    assert f["n_x48"].iloc[148] == 1 and f["hours_since_m"].iloc[148] == 0
    assert f["hours_since_m"].iloc[150] == 2 and f["hours_since_m"].iloc[60] == LOOKBACK_H


# ---------------- training parser (NCEI CSV layout, synthetic files) ----------------
def ncei_file(path, header, rows):
    path.write_text("File name: x\nMetadata Description:\nfoo:units = \"pfu\";\n\ndata: \n" + header + "\n" +
                    "\n".join(rows) + "\n")


def test_parse_goes_csv_variants(tmp_path):
    from evade.spaceweather.train import parse_goes_csv
    eps = tmp_path / "g11_eps_5m_20031001_20031031.csv"
    cols = ["time_tag"] + [f"p{i}_flux_ic" for i in range(1, 8)]
    ncei_file(eps, ",".join(cols), ["2003-10-28 12:00:00.000,1,2,3000.0,4,500.0,6,80.0",
                                    "2003-10-28 12:05:00.000,1,2,-99999.0,4,5,6,7"])
    d = parse_goes_csv(eps)
    assert d["proton_10mev_pfu"].iloc[0] == 3000.0 and np.isnan(d["proton_10mev_pfu"].iloc[1])
    assert d["proton_100mev_pfu"].iloc[0] == 80.0
    cp = tmp_path / "g15_epead_cpflux_5m_20120301_20120331.csv"
    head ="time_tag," + ",".join(f"ZPGT{b}{e}_QUAL_FLAG,ZPGT{b}{e}" for b in ("10", "50", "100") for e in ("E", "W"))
    ncei_file(cp, head, ["2012-03-07 06:00:00.000,0,20.0,0,40.0,0,2,0,4,0,1,1,9"])
    d = parse_goes_csv(cp)
    assert d["proton_10mev_pfu"].iloc[0] == 30.0           # mean of E and W
    assert d["proton_100mev_pfu"].iloc[0] == 1.0           # W flagged -> E only
    xr = tmp_path / "g15_xrs_1m_20120301_20120331.csv"
    ncei_file(xr, "time_tag,A_QUAL_FLAG,A_NUM_PTS,A_AVG,B_QUAL_FLAG,B_NUM_PTS,B_AVG",
              ["2012-03-07 00:24:00.000,0, 30,1.0E-05,0, 30,3.92E-04", "2012-03-07 00:25:00.000,0, 30,1.0E-05,4, 30,1.0E-04"])
    d = parse_goes_csv(xr)
    assert d["xray_long_wm2"].iloc[0] == pytest.approx(3.92e-4 / 0.7) and np.isnan(d["xray_long_wm2"].iloc[1])


# ---------------- artifacts ----------------
def write_tiny_model(root):
    from sklearn.linear_model import LogisticRegression
    import joblib
    rng = np.random.default_rng(0)
    X = rng.normal(size=(400, len(FEATURES)))
    y = (X[:, 0] + rng.normal(size=400) > 1).astype(int)
    d = root / "spaceweather"
    d.mkdir(parents=True)
    joblib.dump({"estimator": LogisticRegression().fit(X, y), "platt": [1.0, 0.0]}, d / "model.joblib")
    (d / "feature_names.json").write_text(json.dumps(FEATURES))
    (d / "metrics.json").write_text(json.dumps({"climatology_rate": 0.02}))
    (d / "manifest.json").write_text(json.dumps({"model_id": "EVA-8", "status": "draft",
                                                 "sha256": {"model.joblib": SW.sha256(d / "model.joblib")}}))
    return d


def test_forecaster_loads_with_sha_and_predicts(tmp_path):
    write_tiny_model(tmp_path)
    art = SW.load_forecaster(tmp_path)
    obs = synthetic_hourly_obs(hours=80, event_at=1000)
    fc = SW.forecast(art, obs, {"spe_in_progress": False})
    assert fc["status"] == "ok" and 0 <= fc["p_onset_24h"] <= 1 and fc["model_id"] == "EVA-8"
    assert SW.forecast(art, obs, {"spe_in_progress": True})["status"] == "event_in_progress"
    assert SW.forecast(None, obs, {"spe_in_progress": False})["status"] == "model_not_loaded"


def test_forecaster_refuses_hash_mismatch(tmp_path):
    d = write_tiny_model(tmp_path)
    with open(d / "model.joblib", "ab") as f:
        f.write(b"tampered")
    with pytest.raises(ValueError):
        SW.load_forecaster(tmp_path)


# ---------------- API endpoint ----------------
fastapi = pytest.importorskip("fastapi")
pytest.importorskip("httpx")
from fastapi.testclient import TestClient  # noqa: E402

from evade.api import app as appmod  # noqa: E402
from tests.conftest import KEY_HEADER  # noqa: E402

H = KEY_HEADER                                     # X-EVAde-Key, checked by the global middleware
C = TestClient(appmod.app)


def new_user():
    return C.post("/v1/users", json={"consent": True}, headers=H).json()["data"]["user_id"]


def obs_payload(p10, p100=None, step_min=5):
    n = len(p10)
    out = []
    for i, v in enumerate(p10):
        t = NOW - timedelta(minutes=step_min * (n - 1 - i))
        row = {"t_utc": t.isoformat(), "proton_10mev_pfu": v, "xray_long_wm2": 2e-6}
        if p100 is not None:
            row["proton_100mev_pfu"] = p100[i]
        out.append(row)
    return out


def test_api_raw_observations_quiet():
    uid = new_user()
    r = C.post("/v1/predict/space-weather", headers=H,
               json={"user_id": uid, "observations": obs_payload([0.4] * 30), "cumulative_dose_msv": 120})
    assert r.status_code == 200, r.text
    d = r.json()["data"]
    assert d["nowcast"]["s_scale"] == 0 and d["nowcast"]["spe_in_progress"] is False
    assert d["forecast"]["status"] == "model_not_loaded" and d["forecast"]["p_onset_24h"] is None
    assert d["radiation_inputs"]["dose_equivalent_rate_msv_per_day"] == pytest.approx(1.8)
    assert d["radiation_domain_preview"]["t_rad_days"] == pytest.approx((600 - 120 - 25) / 1.8, abs=0.1)
    assert d["o5"]["candidate"] is False and r.json()["model_ids"] == []
    assert "assumptions" in d and any("estimate" in w for w in r.json()["warnings"])


def test_api_fake_provider_storm_and_inputs_roundtrip():
    uid = new_user()
    storm = series_obs([0.5] * 12 + [500.0] * 36, None, [0.1] * 12 + [40.0] * 36, xl=[1e-6] * 47 + [3e-4],
                       source="fake")
    P.register("noaa_swpc", P.StaticProvider(storm, "noaa_swpc"))
    r = C.post("/v1/predict/space-weather", headers=H, json={"user_id": uid, "provider": "noaa_swpc",
                                                              "cumulative_dose_msv": 200, "location": "deep_space"})
    assert r.status_code == 200, r.text
    d = r.json()["data"]
    assert d["nowcast"]["s_scale"] == 2 and d["nowcast"]["spe_in_progress"] and d["nowcast"]["r_scale"] == 3
    assert d["forecast"]["status"] == "event_in_progress"
    rin = d["radiation_inputs"]
    assert rin["spe_active"] and rin["spe_dose_msv"] > 0 and rin["rate_ratio_to_7d_median"] > 1
    body = {k: v for k, v in rin.items() if v is not None}
    r2 = C.post("/v1/predict/radiation-risk", headers=H, json={"user_id": uid, "t_utc": NOW.isoformat(), **body})
    assert r2.status_code == 200, r2.text
    assert r2.json()["data"]["s"] == d["radiation_domain_preview"]["s"]


def test_api_provider_unavailable_and_validation():
    uid = new_user()

    class Down(P.SpaceWeatherProvider):
        name = "noaa_swpc"

        def fetch(self, hours=72):
            raise P.ProviderUnavailable("timeout")
    P.register("noaa_swpc", Down())
    r = C.post("/v1/predict/space-weather", headers=H, json={"user_id": uid, "provider": "noaa_swpc"})
    assert r.status_code == 503 and r.json()["error"]["code"] == "provider_unavailable"
    r = C.post("/v1/predict/space-weather", headers=H, json={"user_id": uid, "provider": "replay"})
    assert r.status_code == 503                               # replay not configured on this server
    r = C.post("/v1/predict/space-weather", headers=H, json={"user_id": uid, "provider": "file:///etc/passwd"})
    assert r.status_code == 422                               # not on the allow-list
    r = C.post("/v1/predict/space-weather", headers=H,
               json={"user_id": uid, "provider": "noaa_swpc", "observations": obs_payload([1.0])})
    assert r.status_code == 422
    r = C.post("/v1/predict/space-weather", headers=H,
               json={"user_id": uid, "observations": [{"t_utc": NOW.isoformat(), "kp": 3}]})
    assert r.status_code == 422                               # no proton data
    assert C.post("/v1/predict/space-weather", json={"user_id": uid, "provider": "noaa_swpc"}).status_code == 401
    bad = {"X-EVAde-Key": "wrong-key-0123456789abcdef-0123456789"}
    assert C.post("/v1/predict/space-weather", headers=bad, json={"user_id": uid, "provider": "noaa_swpc"}).status_code == 401


def test_api_rejects_long_time_spans():
    uid = new_user()
    old = [{"t_utc": (NOW - timedelta(days=5 * 365)).isoformat(), "proton_10mev_pfu": 0.5},
           {"t_utc": NOW.isoformat(), "proton_10mev_pfu": 0.5}]
    r = C.post("/v1/predict/space-weather", headers=H, json={"user_id": uid, "observations": old})
    assert r.status_code == 422                               # older than 7 days
    wide = [{"t_utc": (NOW - timedelta(days=6)).isoformat(), "proton_10mev_pfu": 0.5},
            {"t_utc": NOW.isoformat(), "proton_10mev_pfu": 0.5}]
    r = C.post("/v1/predict/space-weather", headers=H, json={"user_id": uid, "observations": wide})
    assert r.status_code == 422 and "span" in r.json()["error"]["message"]
    fut = [{"t_utc": (NOW + timedelta(days=1)).isoformat(), "proton_10mev_pfu": 0.5}]
    r = C.post("/v1/predict/space-weather", headers=H, json={"user_id": uid, "observations": fut})
    assert r.status_code == 422 and r.json()["error"]["code"] == "future_timestamp"


def test_api_with_loaded_model(tmp_path, monkeypatch):
    write_tiny_model(tmp_path)
    monkeypatch.setitem(appmod.STATE.models, "spaceweather", SW.load_forecaster(tmp_path))
    uid = new_user()
    obs = synthetic_hourly_obs(hours=60, event_at=1000)
    shift = NOW - obs.records[-1].t
    payload = [{"t_utc": (r.t + shift).isoformat(), **r.values} for r in obs.records[::3]]
    r = C.post("/v1/predict/space-weather", headers=H, json={"user_id": uid, "observations": payload})
    assert r.status_code == 200, r.text
    fc = r.json()["data"]["forecast"]
    assert fc["status"] == "ok" and r.json()["model_ids"] == ["EVA-8"] and "draft model output" in r.json()["warnings"]
    assert C.get("/v1/models", headers=H).json()["data"]["models"][0]["task"] in ("spaceweather", "seizure", "ecg")


def test_service_start_loads_spaceweather_only_with_draft_and_matching_hash(tmp_path, monkeypatch):
    from evade.models.registry import ModelRegistry
    d = write_tiny_model(tmp_path)
    monkeypatch.setenv("EVADE_MODEL_DIR", str(tmp_path))
    monkeypatch.delenv("EVADE_ALLOW_DRAFT", raising=False)
    assert appmod.load_spaceweather() is None                 # draft model: needs EVADE_ALLOW_DRAFT=1
    monkeypatch.setenv("EVADE_ALLOW_DRAFT", "1")
    sw = appmod.load_spaceweather()
    assert sw["model_id"] == "EVA-8" and sw["sha256_verified"] is False   # sidecars not hashed in this manifest
    reg = ModelRegistry(tmp_path, allow_draft=True)
    reg.scan()
    monkeypatch.setattr(appmod.STATE, "registry", reg)
    monkeypatch.setitem(appmod.STATE.models, "spaceweather", sw)
    d_models = C.get("/v1/models", headers=H).json()["data"]
    assert [m for m in d_models["models"] if m["task"] == "spaceweather"][0]["model_id"] == "EVA-8"
    assert all(u["task"] != "spaceweather" for u in d_models["unavailable"])
    with open(d / "model.joblib", "ab") as f:
        f.write(b"tampered")
    assert appmod.load_spaceweather() is None                 # hash mismatch: nowcast and dose run without it


# ---------------- v2: background correction, CME features, DONKI, GOES-R archive ----------------
from evade.spaceweather import background as BG  # noqa: E402
from evade.spaceweather.features import CME_FEATURES, cme_frame  # noqa: E402

TABLE = {"reference": "15", "satellites": {"15": {"p10": 0.2, "p50": 0.05, "p100": 0.02},
                                           "18": {"p10": 0.3, "p50": 0.2, "p100": 0.15},
                                           "swpc:18": {"p10": 0.25, "p50": 0.18, "p100": 0.17}}}


def hourly(vals, sat):
    idx = pd.date_range("2024-01-01", periods=len(vals), freq="1h", tz="UTC")
    return pd.DataFrame({"p10": vals, "p50": [v / 2 for v in vals], "p100": [v / 4 for v in vals], "sat": sat}, index=idx)


def test_background_maps_floor_to_reference_and_keeps_events():
    h = hourly([0.3, 1000.0], 18.0)
    h["p50"], h["p100"] = [0.2, 500.0], [0.15, 250.0]
    c, warn = BG.correct_hourly(h, TABLE)
    assert c["p10"].iloc[0] == pytest.approx(0.2) and c["p100"].iloc[0] == pytest.approx(0.02)
    assert c["p10"].iloc[1] == pytest.approx(1000.0 - 0.1) and warn == []
    c2, _ = BG.correct_hourly(h, TABLE, prefix="swpc:")
    assert c2["p100"].iloc[0] == pytest.approx(0.002)                  # floored at 0.1 x reference
    u, warn = BG.correct_hourly(hourly([0.3], 99.0), TABLE)
    assert u["p10"].iloc[0] == pytest.approx(0.3) and "99" in warn[0]
    n, warn = BG.correct_hourly(hourly([0.3], np.nan), TABLE)
    assert n["p10"].iloc[0] == pytest.approx(0.3) and warn


def test_background_per_year_nearest():
    tab = {"reference": "15", "satellites": {"15": {"p10": 0.2, "p50": 0.05, "p100": 0.02},
                                             "16": {"p10": 9.0, "p50": 9.0, "p100": 9.0},
                                             "16:2021": {"p10": 0.7, "p50": 0.5, "p100": 0.45},
                                             "16:2023": {"p10": 0.4, "p50": 0.25, "p100": 0.2}}}
    idx = pd.DatetimeIndex(["2021-06-01", "2024-06-01"], tz="UTC")
    h = pd.DataFrame({"p10": [0.7, 0.4], "p50": [0.5, 0.25], "p100": [0.45, 0.2], "sat": 16.0}, index=idx)
    c, warn = BG.correct_hourly(h, tab)
    assert list(c["p10"].round(6)) == [0.2, 0.2] and list(c["p100"].round(6)) == [0.02, 0.02] and not warn


def test_posted_operational_flux_uses_swpc_floor():
    recs = [Record.make(NOW - timedelta(minutes=5 * k), proton_10mev_pfu=0.25, proton_50mev_pfu=0.18,
                        proton_100mev_pfu=0.17, proton_satellite=18) for k in range(600)]
    op = Observations("request", recs, meta={"background_prefix": "swpc:"})
    row, cov = latest_features(op, table=TABLE)
    assert row["lp10"] == pytest.approx(np.log10(0.2)) and not cov["warnings"]   # SWPC floor -> reference floor
    arch, _ = latest_features(Observations("request", recs), table=TABLE)
    assert arch["lp10"] < row["lp10"]           # the archive floor (0.3) would push operational values too low


def test_background_estimate_and_overlap_agreement():
    rng = np.random.default_rng(0)
    n = 24 * 40
    base = np.exp(rng.normal(0, 0.05, n))
    a = hourly(list(0.2 * base), 15.0)
    b = hourly(list(0.6 * base), 18.0)
    for df in (a, b):
        df["p50"], df["p100"] = df["p10"] / 4, df["p10"] / 10
    tab = BG.estimate({"15": a.drop(columns="sat"), "18": b.drop(columns="sat")})
    assert tab["satellites"]["18"]["p10"] == pytest.approx(0.6, rel=0.1)
    before = BG.overlap_agreement(a, b, None, "15", "18")["p10"]["quiet_median_abs_log10_ratio"]
    after = BG.overlap_agreement(a, b, tab, "15", "18")["p10"]["quiet_median_abs_log10_ratio"]
    assert before == pytest.approx(np.log10(3), abs=0.02) and after < 0.1


def test_cme_features():
    idx = pd.date_range("2024-05-01", periods=100, freq="1h", tz="UTC")
    c = pd.DataFrame({"t": [idx[10], idx[20]], "speed_kms": [900.0, 1500.0], "width_deg": [60.0, 360.0]})
    f = cme_frame(idx, c)
    assert f["cme_max_speed72"].iloc[9] == 0 and f["cme_max_speed72"].iloc[10] == 900
    assert f["cme_max_speed72"].iloc[25] == 1500 and f["cme_wide72"].iloc[25] == 1 and f["cme_wide_speed72"].iloc[25] == 1500
    assert f["cme_wide72"].iloc[15] == 0
    assert f["cme_max_speed72"].iloc[20 + 72] == 0                      # out of the 72 h window
    assert cme_frame(idx, None)[CME_FEATURES].isna().all().all()
    assert (cme_frame(idx, c.iloc[0:0])[CME_FEATURES] == 0).all().all()


def donki_rows(t0):
    return {
        "CME": [{"activityID": "c1", "startTime": iso(t0), "cmeAnalyses": [
            {"speed": 700, "halfAngle": 20, "isMostAccurate": False},
            {"speed": 1400, "halfAngle": 70, "latitude": 5, "longitude": -10, "isMostAccurate": True,
             "enlilList": [{"estimatedShockArrivalTime": iso(t0 + timedelta(days=2)), "isEarthGB": False}]}]},
            {"activityID": "c2", "startTime": iso(t0), "cmeAnalyses": None}, {"bad": 1}],
        "FLR": [{"flrID": "f1", "beginTime": iso(t0), "peakTime": iso(t0 + timedelta(minutes=10)), "classType": "X1.2"}],
        "SEP": [{"sepID": "s1", "eventTime": iso(t0 + timedelta(hours=2))}],
        "IPS": [{"activityID": "i1", "eventTime": iso(t0), "location": "Earth"}],
        "GST": [{"gstID": "g1", "startTime": iso(t0), "allKpIndex": [{"kpIndex": 6.33}, {"kpIndex": 7.0}]}],
    }


def test_parse_donki():
    rows = donki_rows(NOW - timedelta(hours=5))
    c = P.parse_donki("CME", rows["CME"])
    assert c[0]["speed_kms"] == 1400 and c[0]["width_deg"] == 140 and c[0]["earth_directed"] is True
    assert c[1]["speed_kms"] is None and len(c) == 2                    # malformed row skipped
    assert P.parse_donki("FLR", rows["FLR"])[0]["class"] == "X1.2"
    assert P.parse_donki("GST", rows["GST"])[0]["kp_max"] == 7.0


def test_donki_falls_back_to_api_nasa_with_env_key(monkeypatch):
    monkeypatch.setenv("NASA_API_KEY", "test-key-123")
    rows = donki_rows(NOW - timedelta(hours=5))
    calls = []

    def fetch(url):
        calls.append(url)
        if "kauai.ccmc" in url:
            raise P.ProviderUnavailable("timeout")
        return rows[url.split("/DONKI/")[1].split("?")[0]]
    obs = P.DonkiProvider(fetch_json=fetch).fetch(96)
    assert sum("kauai" in u for u in calls) == 1                        # CCMC tried once, then skipped
    assert all("api_key=test-key-123" in u for u in calls if "api.nasa.gov" in u)
    assert "test-key-123" not in json.dumps(obs.meta) and obs.meta["key"] == "NASA_API_KEY"
    s = SW.donki_summary(obs.meta["events"])
    assert s["cme_fastest_72h_kms"] == 1400 and s["cme_earth_directed_72h"] and s["ips_at_earth_48h"]
    assert s["flares_m_or_x_72h"][0]["class"] == "X1.2" and s["sep_reported_72h"] and s["gst_kp_max_72h"] == 7.0

    def down(url):
        raise P.ProviderUnavailable("down")
    with pytest.raises(P.ProviderUnavailable):
        P.DonkiProvider(fetch_json=down).fetch(96)
    monkeypatch.delenv("NASA_API_KEY")
    calls.clear()
    P.DonkiProvider(fetch_json=fetch).fetch(96)
    assert any("api_key=DEMO_KEY" in u for u in calls)


def test_integral_from_differential_pro_rata():
    from evade.spaceweather.archive import integral_from_differential
    lo, hi = np.array([5e3, 8e3, 15e3, 40e3]), np.array([8e3, 15e3, 40e3, 80e3])
    j = np.array([[1.0, 2.0, 0.1, 0.01]])
    p11 = np.array([0.05])
    v = integral_from_differential(j, lo, hi, p11, 10e3)
    assert v[0] == pytest.approx(2.0 * 5e3 + 0.1 * 25e3 + 0.01 * 40e3 + 0.05)
    j2 = j.copy()
    j2[0, 3] = np.nan
    assert np.isnan(integral_from_differential(j2, lo, hi, p11, 10e3)[0])


def test_read_cdaw(tmp_path):
    from evade.spaceweather.archive import read_cdaw
    f = tmp_path / "univ_all.txt"
    f.write_text(" ===== SOHO/LASCO CME CATALOG =====\n Date Time ...\n\n"
                 "2017/09/10  16:00:07   Halo    360   3163    3289   3037   3620    -51.4*   -------  -------  -999   \n"
                 "2017/09/09  23:12:05    263     40    401     380    422    500      4.4    -------  -------   265   \n"
                 "2017/09/09  23:30:05    263     40   ----    ----   ----   ----     ----    -------  -------   265   \n")
    d = read_cdaw(f)
    assert len(d) == 2 and d["halo"].iloc[0] and d["width_deg"].iloc[0] == 360 and d["speed_kms"].iloc[1] == 401


def test_forecast_uses_background_and_cmes(tmp_path):
    from sklearn.linear_model import LogisticRegression
    import joblib
    feats = FEATURES + CME_FEATURES
    rng = np.random.default_rng(1)
    X = rng.normal(size=(300, len(feats)))
    y = (X[:, -1] > 0.5).astype(int)
    d = tmp_path / "spaceweather"
    d.mkdir()
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import make_pipeline
    est = make_pipeline(SimpleImputer(strategy="median"), LogisticRegression()).fit(X, y)
    joblib.dump({"estimator": est, "platt": None, "background": TABLE}, d / "model.joblib")
    (d / "feature_names.json").write_text(json.dumps(feats))
    (d / "manifest.json").write_text(json.dumps({"model_id": "EVA-X", "sha256": {"model.joblib": SW.sha256(d / "model.joblib")}}))
    art = SW.load_forecaster(tmp_path)
    obs = synthetic_hourly_obs(hours=80, event_at=1000)
    fc = SW.forecast(art, obs, {"spe_in_progress": False})
    assert fc["status"] == "ok" and any("CME" in w for w in fc["warnings"]) and fc["coverage"]["background_corrected"]
    cm = pd.DataFrame({"t": [obs.records[-1].t - timedelta(hours=3)], "speed_kms": [2000.0], "width_deg": [360.0]})
    fc2 = SW.forecast(art, obs, {"spe_in_progress": False}, cmes=cm)
    assert fc2["status"] == "ok" and not any("CME" in w for w in fc2["warnings"])
    assert fc2["p_onset_24h"] != fc["p_onset_24h"]


def test_api_donki_context_and_failure():
    uid = new_user()
    quiet = series_obs([0.4] * 40, None, [0.2] * 40, xl=[1e-6] * 40, source="fake")
    P.register("noaa_swpc", P.StaticProvider(quiet, "noaa_swpc"))
    ev = Observations("donki", [], meta={"events": {"cme": P.parse_donki("CME", donki_rows(NOW - timedelta(hours=5))["CME"]),
                                                    "ips": []}, "donki_sources": {"CME": "ccmc"}})
    P.register("donki", P.StaticProvider(ev, "donki"))
    r = C.post("/v1/predict/space-weather", headers=H, json={"user_id": uid, "provider": "noaa_swpc"})
    assert r.status_code == 200, r.text
    assert r.json()["data"]["donki"]["summary"]["cme_fastest_72h_kms"] == 1400

    class Down(P.SpaceWeatherProvider):
        name = "donki"

        def fetch(self, hours=96):
            raise P.ProviderUnavailable("timeout")
    P.register("donki", Down())
    r = C.post("/v1/predict/space-weather", headers=H, json={"user_id": uid, "provider": "noaa_swpc"})
    assert r.status_code == 200 and any("DONKI unavailable" in w for w in r.json()["warnings"])
    assert r.json()["data"]["donki"]["summary"] is None
    r = C.post("/v1/predict/space-weather", headers=H, json={
        "user_id": uid, "observations": obs_payload([0.4] * 10),
        "cmes": [{"t_utc": (NOW - timedelta(hours=2)).isoformat(), "speed_kms": 1800, "half_angle_deg": 90}]})
    assert r.status_code == 200 and r.json()["data"]["donki"]["source"] == "request"
    assert r.json()["data"]["donki"]["summary"]["cme_fastest_72h_kms"] == 1800
