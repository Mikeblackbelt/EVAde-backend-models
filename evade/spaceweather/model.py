"""Space-weather nowcast, SPE onset forecast and crew dose-rate estimate, mapped to the radiation risk domain.

Everything here is a flag or an estimate for human review, not dosimetry. Three parts:
  (a) nowcast: NOAA S-, R- and G-scale levels and an SPE-in-progress flag from the latest observations;
  (b) forecast: P(>=10 MeV flux reaches S1 within 24 h | quiet now), a trained model (EVA-8) loaded from disk;
  (c) dose: GCR baseline by location plus an SPE term from proton flux and a shielding preset. All parameters are
      assumptions with their evidence label (see ASSUMPTIONS); a measured onboard dose rate replaces the estimate.
The outputs fill the inputs of evade.risk.engine.radiation_domain and the MDTR radiation limit (T_rad).
"""
from __future__ import annotations

import bisect
import hashlib
import json
import math
from datetime import datetime, timedelta, timezone
from pathlib import Path

from .schema import Observations

# ---------------- (a) nowcast ----------------
# NOAA space weather scales (https://www.swpc.noaa.gov/noaa-scales-explanation): thresholds are standard NOAA values.
S_SCALE = [(1e5, 5), (1e4, 4), (1e3, 3), (1e2, 2), (1e1, 1)]          # >=10 MeV integral protons, pfu
R_SCALE = [(2e-3, 5), (1e-3, 4), (1e-4, 3), (5e-5, 2), (1e-5, 1)]     # 0.1-0.8 nm X-ray peak, W m-2
G_SCALE = [(9.0, 5), (8.0, 4), (7.0, 3), (6.0, 2), (5.0, 1)]          # Kp
SPE_PFU = 10.0
STALE_MIN = 30


def scale_level(x: float | None, table: list[tuple[float, int]]) -> int | None:
    if x is None:
        return None
    return next((lvl for thr, lvl in table if x >= thr), 0)


def flare_class(x: float | None) -> str | None:
    """GOES flare class from the 0.1-0.8 nm flux (truncated, not rounded, as NOAA defines it)."""
    if x is None or x <= 0:
        return None
    for letter, base in (("X", 1e-4), ("M", 1e-5), ("C", 1e-6), ("B", 1e-7), ("A", 1e-8)):
        if x >= base:
            return f"{letter}{math.floor(x / base * 10 + 1e-6) / 10:.1f}"     # epsilon: 3e-8/1e-8*10 = 29.999...
    return "<A1.0"


def spe_flag(p: list[tuple[datetime, float]]) -> bool:
    """NOAA SEP event rule: three consecutive 5-min points at or above 10 pfu. With fewer than three points in the
    last 15 minutes (coarser feeds), the latest value decides."""
    if not p:
        return False
    t_last, v_last = p[-1]
    recent = [v for t, v in p if t >= t_last - timedelta(minutes=15)]
    return all(v >= SPE_PFU for v in recent[-3:]) if len(recent) >= 3 else v_last >= SPE_PFU


def nowcast(obs: Observations, now: datetime | None = None) -> dict:
    now = now or datetime.now(timezone.utc)
    p = obs.series("proton_10mev_pfu")
    out: dict = {"s_scale": None, "r_scale": None, "g_scale": None, "spe_in_progress": None, "warnings": []}
    if p:
        t_last, v_last = p[-1]
        spe = spe_flag(p)
        age = (now - t_last).total_seconds() / 60
        out.update(s_scale=scale_level(v_last, S_SCALE), spe_in_progress=bool(spe), proton_10mev_pfu=v_last,
                   proton_time_utc=t_last.isoformat(), proton_age_min=round(age, 1))
        for k in ("proton_50mev_pfu", "proton_100mev_pfu"):
            lt = obs.latest(k)
            out[k] = lt[1] if lt else None
        day = [v for t, v in p if t >= t_last - timedelta(hours=24)]
        out["proton_10mev_max24h_pfu"] = max(day)
        if age > STALE_MIN:
            out["warnings"].append(f"proton data are {age:.0f} min old")
    else:
        out["warnings"].append("no >=10 MeV proton data: S-scale and SPE flag unavailable")
    x = obs.series("xray_long_wm2")
    if x:
        t_last = x[-1][0]
        hour = [v for t, v in x if t >= t_last - timedelta(hours=1)]
        day = [v for t, v in x if t >= t_last - timedelta(hours=24)]
        out.update(r_scale=scale_level(max(hour), R_SCALE), xray_long_wm2=x[-1][1], xray_max1h_wm2=max(hour),
                   flare_class_now=flare_class(x[-1][1]), flare_class_max1h=flare_class(max(hour)),
                   flare_class_max24h=flare_class(max(day)), xray_time_utc=t_last.isoformat())
    else:
        out["warnings"].append("no X-ray data: R-scale unavailable")
    kp = obs.latest("kp")
    if kp:
        out.update(g_scale=scale_level(kp[1], G_SCALE), kp=kp[1], kp_time_utc=kp[0].isoformat())
    for k in ("sw_speed_kms", "sw_density_cm3", "bz_gsm_nt"):
        lt = obs.latest(k)
        out[k] = lt[1] if lt else None
    return out


# ---------------- (c) dose-rate estimate ----------------
# Every number below is an ASSUMPTION for a demo estimate, with its evidence label. Not dosimetry.
GCR_BASELINE_MSV_DAY = {
    # location: (dose-equivalent rate mSv/day, source, label)
    "leo_iss": (0.647, "ISS Columbus, up to 647 microSv/day dose equivalent, end of May 2016 (Berger 2017)", "PARTIAL"),
    "deep_space": (1.8, "MSL cruise: 0.66 Sv for a ~360-day round trip (Zeitlin 2013) -> ~1.8 mSv/day; "
                        "same value as max-days-to-return.md", "PARTIAL (derived)"),
    "lunar_surface": (1.37, "13.2 microGy/h absorbed in silicon (Zhang 2020, VERIFIED) x 24 h x assumed mean "
                            "quality factor 4.3", "UNVERIFIED (quality factor assumed)"),
    "mars_surface": (0.67, "Mars surface rate quoted in max-days-to-return.md (Hassler 2014); the abstract has no "
                           "numeric rate", "UNVERIFIED"),
}
# SPE term: dose rate = T_loc x T_shield x 4*pi x max(0, F_channel - F_background) x S x 1.602e-10 Gy/(MeV/g) x w_R
SHIELDING = {
    # preset: (integral channel that penetrates, stopping power in tissue MeV cm2/g, extra transmission, note)
    "eva_suit": ("proton_10mev_pfu", 15.0, 1.0, "suit ~1 g/cm2: >=10 MeV protons reach tissue"),
    "spacecraft": ("proton_100mev_pfu", 6.5, 1.0, "~10 g/cm2 Al stops protons below ~100 MeV"),
    "storm_shelter": ("proton_100mev_pfu", 5.0, 0.3, "~30 g/cm2: only ~200 MeV and above; 0.3 of the >=100 MeV flux"),
}
LOCATION_SPE_TRANSMISSION = {
    "deep_space": (1.0, "GOES flux taken as free-space flux at 1 AU"),
    "lunar_surface": (0.5, "the Moon blocks half the sky"),
    "mars_surface": (0.25, "planet blocks half the sky; ~20 g/cm2 CO2 column halves the rest (1.5 AU ignored)"),
    "leo_iss": (0.01, "geomagnetic shielding in low Earth orbit; exposure mostly on high-latitude passes"),
}
BACKGROUND_PFU = {"proton_10mev_pfu": 1.0, "proton_50mev_pfu": 0.3, "proton_100mev_pfu": 0.2}
W_R_PROTON = 2.0
MEV_PER_G_TO_GY = 1.602e-10
ASSUMPTIONS = {
    "gcr_baseline": {k: {"msv_per_day": v[0], "source": v[1], "evidence": v[2]} for k, v in GCR_BASELINE_MSV_DAY.items()},
    "spe_formula": "T_location x T_shielding x 4*pi x max(0, F - F_background) x S_tissue x 1.602e-10 Gy per MeV/g x w_R",
    "shielding_presets": {k: {"channel": v[0], "stopping_power_mev_cm2_g": v[1], "transmission": v[2], "note": v[3],
                              "evidence": "UNVERIFIED (order-of-magnitude physics, not transport-code output)"}
                          for k, v in SHIELDING.items()},
    "location_spe_transmission": {k: {"value": v[0], "note": v[1], "evidence": "UNVERIFIED"}
                                  for k, v in LOCATION_SPE_TRANSMISSION.items()},
    "background_pfu": {"value": BACKGROUND_PFU, "evidence": "UNVERIFIED (typical quiet GOES levels; subtracted so "
                                                            "instrument background is not counted as dose)"},
    "w_r_proton": {"value": W_R_PROTON, "evidence": "UNVERIFIED here (ICRP 103 proton weighting factor, not in lit review)"},
    "isotropic_free_space_flux": {"value": "4*pi sr", "evidence": "UNVERIFIED simplification"},
    "limits": {"career_msv": 600, "per_spe_msv": 250, "source": "NASA-STD-3001 Vol 1 Rev B", "evidence": "VERIFIED"},
    "caveat": "Demo estimate. Dose behind real shielding depends on spectrum, geometry and secondaries "
              "(Artemis I saw ~4x differences between locations, George 2024). Use a measured dosimeter when available.",
}


def spe_rate_usv_h(flux: dict[str, float | None], location: str, shielding: str) -> float:
    ch, s_tissue, t_shield, _ = SHIELDING[shielding]
    f = flux.get(ch)
    if f is None:
        return 0.0
    excess = max(0.0, f - BACKGROUND_PFU[ch])
    t_loc = LOCATION_SPE_TRANSMISSION[location][0]
    gy_per_s = t_loc * t_shield * 4 * math.pi * excess * s_tissue * MEV_PER_G_TO_GY
    return gy_per_s * W_R_PROTON * 3600 * 1e6


def _flux_lookup(obs: Observations):
    """Return f(t) -> {channel: last value at or before t} using one sorted pass per channel."""
    ser = {k: obs.series(k) for k in BACKGROUND_PFU}
    times = {k: [t for t, _ in s] for k, s in ser.items()}

    def at(t: datetime) -> dict[str, float | None]:
        out = {}
        for k, s in ser.items():
            i = bisect.bisect_right(times[k], t) - 1
            out[k] = s[i][1] if i >= 0 else None
        return out
    return at


def dose_estimate(obs: Observations, location: str = "deep_space", shielding: str = "spacecraft",
                  measured_dose_rate_usv_h: float | None = None) -> dict:
    """Current dose-equivalent rate, the SPE dose so far in the ongoing event, and a 24 h persistence projection."""
    gcr = GCR_BASELINE_MSV_DAY[location][0]
    p = obs.series("proton_10mev_pfu")
    flux_now = {k: (obs.latest(k)[1] if obs.latest(k) else None) for k in BACKGROUND_PFU}
    spe_now = spe_rate_usv_h(flux_now, location, shielding) * 24 / 1000          # mSv/day
    # SPE dose so far: integrate the SPE term over the current run of samples at or above 10 pfu (within the window)
    spe_dose = 0.0
    spe_active = spe_flag(p)
    if spe_active:
        start = len(p) - 1
        while start > 0 and p[start - 1][1] >= SPE_PFU:
            start -= 1
        run = [t for t, _ in p[start:]]
        flux_at = _flux_lookup(obs)
        for t0, t1 in zip(run, run[1:]):
            dt_h = min((t1 - t0).total_seconds() / 3600, 1.0)              # do not bridge gaps longer than 1 h
            spe_dose += spe_rate_usv_h(flux_at(t0), location, shielding) * dt_h / 1000
    projected = spe_dose + spe_now if spe_active else None                     # persistence: current rate for 24 h
    measured = measured_dose_rate_usv_h
    if measured is None and obs.latest("dose_rate_usv_h"):
        measured = obs.latest("dose_rate_usv_h")[1]
    model_rate = gcr + spe_now
    rate = measured * 24 / 1000 if measured is not None else model_rate
    return {"dose_equivalent_rate_msv_per_day": round(rate, 4), "source": "measured" if measured is not None else "estimated",
            "estimated": {"gcr_msv_per_day": gcr, "spe_msv_per_day": round(spe_now, 4), "total_msv_per_day": round(model_rate, 4)},
            "measured_usv_per_h": measured, "spe_active": spe_active, "spe_dose_msv_event_so_far": round(spe_dose, 3),
            "spe_projected_dose_msv_24h": round(projected, 2) if projected is not None else None,
            "rate_ratio_to_quiet_baseline": round(rate / gcr, 3), "location": location, "shielding": shielding,
            "projection_method": "event dose so far + current SPE rate held for 24 h (persistence; conservative near peak)"}


def radiation_inputs(dose: dict, cumulative_dose_msv: float | None, seu_count_last_hour: int = 0) -> dict:
    """Map the estimate to the fields of /v1/predict/radiation-risk and radiation_domain().

    rate_ratio_to_7d_median uses the location's quiet GCR baseline as the 7-day median (no 7-day history here).
    """
    return {"cumulative_dose_msv": cumulative_dose_msv,
            "dose_equivalent_rate_msv_per_day": dose["dose_equivalent_rate_msv_per_day"],
            "rate_ratio_to_7d_median": dose["rate_ratio_to_quiet_baseline"],
            "spe_active": dose["spe_active"], "spe_dose_msv": dose["spe_dose_msv_event_so_far"],
            "spe_projected_dose_msv_24h": dose["spe_projected_dose_msv_24h"], "seu_count_last_hour": seu_count_last_hour}


def o5_candidate(dose: dict) -> dict:
    """Override O5 (risk-score.md): SPE dose >= 250 mSv, or projected to reach it within 24 h. Candidate only: a human
    confirms it; a flux-derived estimate never writes the override by itself."""
    d, proj = dose["spe_dose_msv_event_so_far"], dose["spe_projected_dose_msv_24h"] or 0.0
    hit = d >= 250 or proj >= 250
    return {"override": "O5", "candidate": hit,
            "reason": (f"estimated SPE dose {d:.1f} mSv, 24 h projection {proj:.1f} mSv (limit 250 mSv)"),
            "note": "candidate for human review; confirm with a dosimeter before raising O5"}


# ---------------- (b) forecast model artifacts ----------------
MODEL_ID, ARCH_ID, TASK = "EVA-8", "EVA-7", "spe_onset_24h"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def load_forecaster(root: str | Path) -> dict | None:
    """Load <root>/spaceweather/ after checking the manifest sha256. Each file is read once and the bytes that are
    hashed are the bytes that are deserialized (as evade/models/registry.py), so nothing can swap the file between the
    check and the load. joblib unpickles code: only load artifacts built by evade.spaceweather.train on a machine we
    control (the hash catches corruption and swaps, not a forged manifest)."""
    import hmac
    import io
    d = Path(root).expanduser() / "spaceweather"
    if not (d / "manifest.json").exists() or not (d / "model.joblib").exists():
        return None
    man = json.loads((d / "manifest.json").read_text())
    # two manifest layouts: the original one (sha256 = {file: digest}) and the model registry's (sha256 = weights digest,
    # files = {sidecar: digest}), which scripts/fetch_models.py needs to push the folder to and pull it from R2
    if isinstance(man.get("sha256"), dict):
        digests = man["sha256"]
    elif isinstance(man.get("sha256"), str):
        digests = {**(man.get("files") or {}), man.get("weights", "model.joblib"): man["sha256"]}
    else:
        digests = {}

    unhashed: list[str] = []

    def read(name: str, required: bool) -> bytes | None:
        """The weights need a matching digest; a sidecar is checked when the manifest lists it, else it marks the
        folder unverified (as the manifest registry does)."""
        p = d / name
        if not p.is_file():
            if required:
                raise ValueError(f"spaceweather {name} missing")
            return None
        data = p.read_bytes()
        want = digests.get(name)
        if want is None and name != "model.joblib":
            unhashed.append(name)
        elif not (isinstance(want, str) and hmac.compare_digest(hashlib.sha256(data).hexdigest(), want)):
            raise ValueError(f"spaceweather {name} does not match manifest sha256; refusing to load")
        return data

    weights = read("model.joblib", True)
    feats = json.loads(read("feature_names.json", True))
    raw_metrics = read("metrics.json", False)
    import joblib
    art = joblib.load(io.BytesIO(weights))
    metrics = json.loads(raw_metrics) if raw_metrics is not None else {}
    return {"model": art, "features": feats, "manifest": man, "metrics": metrics,
            "model_id": man.get("model_id", MODEL_ID), "status": man.get("status", "draft"),
            "sha256_verified": not unhashed}


def predict_proba(art: dict, row: dict) -> float:
    """Apply the saved estimator (+ optional Platt recalibration) to one feature row."""
    import numpy as np
    X = np.array([[np.nan if row.get(f) is None else row[f] for f in art["features"]]], dtype=float)
    est = art["model"]["estimator"]
    p = float(est.predict_proba(X)[0, 1])
    platt = art["model"].get("platt")
    if platt:
        a, b = platt
        z = a * math.log(max(p, 1e-6) / max(1 - p, 1e-6)) + b
        p = 1 / (1 + math.exp(-z))
    return p


def donki_summary(events: dict | None, now: datetime | None = None) -> dict | None:
    """Context from DONKI events for the response (display only, except CMEs, which also feed the forecast)."""
    if events is None:
        return None
    now = now or datetime.now(timezone.utc)
    since = lambda h: (now - timedelta(hours=h)).isoformat()
    fl = [e for e in events.get("flr", []) if e["t_utc"] >= since(72) and str(e.get("class") or "")[:1] in ("M", "X")]
    cm = [e for e in events.get("cme", []) if e["t_utc"] >= since(72)]
    return {"flares_m_or_x_72h": [{"t_utc": e["t_utc"], "class": e.get("class")} for e in fl],
            "cme_count_72h": len(cm),
            "cme_fastest_72h_kms": max([e["speed_kms"] for e in cm if e.get("speed_kms")], default=None),
            "cme_earth_directed_72h": any(e.get("earth_directed") for e in cm) if cm else False,
            "ips_at_earth_48h": any(e["t_utc"] >= since(48) and (e.get("location") or "").lower() == "earth"
                                    for e in events.get("ips", [])),
            "sep_reported_72h": any(e["t_utc"] >= since(72) for e in events.get("sep", [])),
            "gst_kp_max_72h": max([e["kp_max"] for e in events.get("gst", []) if e["t_utc"] >= since(72)
                                   and e.get("kp_max") is not None], default=None),
            "feeds": sorted(events)}


def forecast(art: dict | None, obs: Observations, now_cast: dict, cmes=None) -> dict:
    from .features import CME_FEATURES, latest_features
    out = {"task": "P(>=10 MeV proton flux reaches 10 pfu (S1) within 24 h), given it is below 10 pfu now",
           "p_onset_24h": None, "model_id": None}
    if now_cast.get("spe_in_progress"):
        return {**out, "status": "event_in_progress", "reason": "flux already at or above S1; onset forecast not applicable"}
    if art is None:
        return {**out, "status": "model_not_loaded", "reason": "no forecast model in EVADE_MODEL_DIR/spaceweather "
                                                               "(or EVADE_ALLOW_DRAFT is not 1)"}
    row, cov = latest_features(obs, table=art["model"].get("background"), cmes=cmes)
    if row is None:
        return {**out, "status": "insufficient_data", "reason": cov.get("reason"), "coverage": cov}
    p = predict_proba(art, row)
    m = art.get("metrics", {})
    warn = list(cov.pop("warnings", []))
    if set(CME_FEATURES) & set(art["features"]) and cmes is None:
        warn.append("no CME source (DONKI unavailable): CME features imputed")
    if cov["xray_hours"] == 0:
        warn.append("no X-ray data: forecast uses imputed X-ray features")
    if cov["proton_hours"] < 24:
        warn.append(f"only {cov['proton_hours']} h of proton history (model uses 48 h)")
    return {**out, "status": "ok", "p_onset_24h": round(p, 4), "model_id": art["model_id"], "model_status": art["status"],
            "climatology": m.get("climatology_rate"), "coverage": cov, "warnings": warn,
            "note": m.get("serving_note", "draft model; a probability for review, not a warning service")}
