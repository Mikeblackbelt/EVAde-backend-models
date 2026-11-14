"""EVAde model service (beta). Implements the hackathon-minimum subset of lit review/planning/api-schema.md.

Run:  EVADE_API_TOKEN=... EVADE_MODEL_DIR=~/evade_train_beta EVADE_ALLOW_DRAFT=1 uvicorn evade.api.app:app
Env:
  EVADE_API_TOKEN      shared bearer token (required; the hackathon uses one shared token plus user_id)
  EVADE_CORS_ORIGINS   comma-separated allow-list (default: none)
  EVADE_MODEL_DIR      folder with seizure/ and ecg/ beta model outputs (optional)
  EVADE_ALLOW_DRAFT    "1" to serve draft (beta) models; they are labelled draft in every response
State is in memory (lost on restart). Outputs are flags for human review, never diagnoses.
"""
from __future__ import annotations

import json
import os
import secrets
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from ..baseline.store import BaselineStore, domain_scores
from ..preprocessing import PIPELINE_VERSION, config
from ..preprocessing.ecg import detect_qrs, hrv_features, preprocess_ecg
from ..preprocessing.eeg import bipolar_from_monopolar, preprocess_eeg, select_bipolar
from ..preprocessing.signal import make_windows, signal_quality
from ..risk.engine import DomainInput, mdtr, radiation_domain, risk
from ..training.features import EEG_BANDS, ecg_features, spectral_features
from . import schemas as S

API_VERSION = "0.1.0-beta"
MAX_BODY = 1_000_000
ACTIONS = {
    ("cardiac", "yellow"): ("Heart metrics above your baseline", ["Pause strenuous work if safe", "Re-check in 10 minutes at rest"]),
    ("cardiac", "red"): ("Heart metrics well above your baseline", ["Stop strenuous work", "Re-measure at rest", "Contact the flight surgeon if it persists"]),
    ("neuro", "yellow"): ("Signs of fatigue in EEG", ["Avoid starting complex procedures", "Take a short break and log your last sleep"]),
    ("neuro", "red"): ("Strong EEG fatigue or seizure-like pattern flagged", ["Do not perform critical tasks", "Notify a crewmate and the flight surgeon for review"]),
    ("behavioral", "yellow"): ("Mood, stress or sleep below your baseline", ["Log a short journal entry", "Plan a rest or recreation block"]),
    ("behavioral", "red"): ("Mood, stress or sleep well below your baseline", ["Schedule a check-in with the behavioral health team"]),
    ("radiation", "yellow"): ("Radiation exposure elevated", ["Review shielding and planned EVA timing"]),
    ("radiation", "red"): ("Radiation exposure high", ["Move to the shielded area if advised", "Defer EVA", "Alert ground"]),
}


class State:
    def __init__(self):
        self.users: dict[str, dict] = {}
        self.baselines = BaselineStore()
        self.latest: dict[str, dict] = {}        # user_id -> {modality: [comparisons]}
        self.radiation: dict[str, dict] = {}
        self.overrides: dict[str, set] = {}
        self.risk_history: dict[str, list] = {}
        self.inputs: dict[str, list] = {}
        self.lock = threading.Lock()
        self.models = load_models()


def load_models() -> dict:
    out, root = {}, os.environ.get("EVADE_MODEL_DIR")
    allow = os.environ.get("EVADE_ALLOW_DRAFT") == "1"
    if not root or not allow:
        return out
    import joblib
    for task, eva in (("seizure", "EVA-2"), ("ecg", "EVA-6")):
        d = Path(root).expanduser() / task
        if (d / "model.joblib").exists():
            m = json.loads((d / "metrics.json").read_text())
            # joblib unpickles code: EVADE_MODEL_DIR must hold only artifacts built by evade.training on a machine we control
            out[task] = {"model": joblib.load(d / "model.joblib"),
                         "features": json.loads((d / "feature_names.json").read_text()),
                         "metrics": m, "model_id": eva, "status": "draft"}
    return out


app = FastAPI(title="EVAde model service", version=API_VERSION,
              description="Beta. Flags for human review, not diagnoses.")
_origins = [o.strip() for o in os.environ.get("EVADE_CORS_ORIGINS", "").split(",") if o.strip()]
if _origins:
    app.add_middleware(CORSMiddleware, allow_origins=_origins, allow_methods=["GET", "POST", "DELETE"],
                       allow_headers=["Authorization", "Content-Type", "Idempotency-Key"])
STATE = State()


def err(status: int, code: str, message: str, field: str | None = None) -> HTTPException:
    return HTTPException(status, {"code": code, "message": message, "field": field})


@app.exception_handler(HTTPException)
async def _http_err(request: Request, exc: HTTPException):
    d = exc.detail if isinstance(exc.detail, dict) else {"code": "error", "message": str(exc.detail)}
    return JSONResponse({"error": {**d, "request_id": request.state.rid if hasattr(request.state, "rid") else None}},
                        status_code=exc.status_code)


@app.exception_handler(RequestValidationError)
async def _val_err(request: Request, exc: RequestValidationError):
    e = exc.errors()[0] if exc.errors() else {}
    msg = str(e.get("msg", "validation error"))
    code = next((c for c in ("future_timestamp", "nan_in_signal", "bad_shape", "payload_too_large") if c in msg), "validation_error")
    return JSONResponse({"error": {"code": code, "message": msg, "field": ".".join(str(x) for x in e.get("loc", []))}},
                        status_code=413 if code == "payload_too_large" else 422)


@app.middleware("http")
async def _mw(request: Request, call_next):
    request.state.rid = uuid.uuid4().hex[:12]
    if int(request.headers.get("content-length") or 0) > MAX_BODY:
        return JSONResponse({"error": {"code": "payload_too_large", "message": "body over 1 MB"}}, status_code=413)
    return await call_next(request)


def auth(authorization: str | None = Header(None)) -> None:
    token = os.environ.get("EVADE_API_TOKEN")
    if not token:
        raise err(503, "service_not_configured", "EVADE_API_TOKEN is not set")
    if not authorization or not secrets.compare_digest(authorization, f"Bearer {token}"):
        raise err(401, "unauthenticated", "missing or invalid token")


def user(user_id: str) -> dict:
    u = STATE.users.get(user_id)
    if u is None:
        raise err(404, "user_not_found", "unknown user_id", "user_id")
    return u


def envelope(user_id: str | None, data: dict, model_ids: list[str] | None = None, warnings: list[str] | None = None) -> dict:
    return {"request_id": uuid.uuid4().hex[:12], "user_id": user_id, "server_time": datetime.now(timezone.utc).isoformat(),
            "api_version": API_VERSION, "model_ids": model_ids or [], "pipeline_version": PIPELINE_VERSION,
            "data": data, "warnings": warnings or []}


def remember(user_id: str, comps: list[dict]) -> None:
    with STATE.lock:
        cur = STATE.latest.setdefault(user_id, {})
        for c in comps:
            cur[(c["modality"], c["metric"])] = c


def alerts_for(comps: list[dict]) -> list[dict]:
    seen, out = set(), []
    for c in comps:
        k = (c["domain"], c["level"])
        if c["level"] != "green" and not c.get("shadow") and k in ACTIONS and k not in seen and c.get("confidence") != "low_quality":
            seen.add(k)
            title, steps = ACTIONS[k]
            out.append({"action_id": f"{k[0]}_{k[1]}", "domain": k[0], "level": k[1], "title": title, "steps": steps,
                        "source_status": "draft wording, to be reviewed"})
    return out


# ---------------- feature extraction for windows ----------------
def ecg_window_metrics(w: S.Window) -> tuple[dict, float]:
    x = np.asarray(w.samples, float)
    lead = w.channels.index("II") if "II" in w.channels else 0
    xp, fs = preprocess_ecg(x[lead], w.fs_hz)
    q = signal_quality(xp[0])
    peaks = detect_qrs(xp[0], fs)
    rr = np.diff(peaks) / fs
    h = hrv_features(rr)
    irr = (h["rmssd_ms"] / 1000.0) / (60.0 / h["hr_bpm"]) if np.isfinite(h["hr_bpm"]) else float("nan")
    return {"hr_bpm": h["hr_bpm"], "rmssd_ms": h["rmssd_ms"], "sdnn_ms": h["sdnn_ms"], "rr_irregularity": irr}, q


def eeg_window_metrics(w: S.Window):
    x = np.asarray(w.samples, float)
    if any("-" in c for c in w.channels):
        xb, present = select_bipolar(x, w.channels)
    else:
        xb, present = bipolar_from_monopolar(x, w.channels)
    n_present = sum(1 for p in present if p)
    xp, fs = preprocess_eeg(xb, w.fs_hz)
    wins, _ = make_windows(xp, fs, config.EEG_WIN_S, config.EEG_STEP_S)
    feats, names = spectral_features(wins, fs, EEG_BANDS, config.BIPOLAR_NAMES)
    idx = {n: i for i, n in enumerate(names)}
    q = float(np.mean([signal_quality(wi[[i for i, p in enumerate(present) if p]]) for wi in wins])) if n_present else 0.0
    if n_present < 12:
        q = min(q, 0.5)
    rel = lambda b: float(np.mean(feats[:, idx[f"mean_rel_{b}"]]))
    return feats, names, {"theta_beta": rel("theta") / max(rel("beta"), 1e-9), "rel_beta": rel("beta"),
                          "rel_theta": rel("theta"), "rel_alpha": rel("alpha")}, q, n_present


def window_comparisons(user_id: str, w: S.Window) -> list[dict]:
    if w.modality == "ecg":
        m, q = ecg_window_metrics(w)
        units = {"hr_bpm": "bpm", "rmssd_ms": "ms", "sdnn_ms": "ms", "rr_irregularity": "ratio"}
    elif w.modality == "eeg":
        _, _, m, q, _ = eeg_window_metrics(w)
        units = {k: "ratio" for k in m}
    else:
        raise err(422, "validation_error", "ppg windows are not supported in the beta", "modality")
    simulated = w.source in ("simulated", "replay")
    out = []
    for k, v in m.items():
        c = STATE.baselines.compare(user_id, w.modality, k, None if v is None or not np.isfinite(v) else float(v),
                                    quality=q, unit=units[k])
        if simulated and c["confidence"] in ("ok", "provisional", "demo"):
            c["confidence"] = "synthetic"
        out.append(c)
    return out


# ---------------- endpoints ----------------
@app.get("/v1/health")
def health():
    return {"status": "ok", "api_version": API_VERSION, "models_loaded": sorted(STATE.models),
            "server_time": datetime.now(timezone.utc).isoformat()}


@app.get("/v1/models", dependencies=[Depends(auth)])
def models():
    rows = []
    for task, m in STATE.models.items():
        rows.append({"task": task, "model_id": m["model_id"], "status": m["status"], "headline": headline(task, m["metrics"])})
    return envelope(None, {"models": rows, "draft_served": os.environ.get("EVADE_ALLOW_DRAFT") == "1",
                           "rule_based": ["radiation domain (risk-score.md)", "risk score v2 and MDTR", "baseline comparison"]})


def headline(task: str, m: dict) -> dict:
    if task == "seizure":
        h = m["leave_one_subject_out_chbmit"]["hgb"]
        return {"loso_pooled_auroc": h["pooled_auroc"], "subject_auroc_ci": h["subject_auroc_ci"],
                "external_siena_auroc": m.get("external_siena", {}).get("hgb", {}).get("auroc")}
    if task == "ecg":
        return {"macro_auroc_test_fold": m["hgb"]["macro_auroc"], "ci95": m["hgb"]["macro_auroc_ci95"]}
    return {}


@app.post("/v1/users", dependencies=[Depends(auth)])
def create_user(body: S.UserCreate):
    uid = str(uuid.uuid4())
    with STATE.lock:
        STATE.users[uid] = {"created_at": time.time(), "cohort": body.cohort, "age_band": body.age_band, "sex": body.sex}
    return envelope(uid, {"user_id": uid, "created_at": datetime.now(timezone.utc).isoformat()})


@app.delete("/v1/users/{user_id}", dependencies=[Depends(auth)])
def delete_user(user_id: str):
    user(user_id)
    with STATE.lock:
        for d in (STATE.users, STATE.latest, STATE.radiation, STATE.overrides, STATE.risk_history, STATE.inputs):
            d.pop(user_id, None)
    STATE.baselines.delete_user(user_id)
    return envelope(user_id, {"deleted": True})


@app.post("/v1/baseline/{modality}", dependencies=[Depends(auth)])
def baseline_submit(modality: S.Modality, body: S.BaselineSubmit):
    user(body.user_id)
    excluded = {"low_quality": 0, "wrong_modality": 0}
    for w in body.windows:
        if w.modality != modality:
            excluded["wrong_modality"] += 1
            continue
        if w.modality == "ecg":
            m, q = ecg_window_metrics(w)
        elif w.modality == "eeg":
            _, _, m, q, _ = eeg_window_metrics(w)
        else:
            raise err(422, "validation_error", "ppg windows are not supported in the beta", "modality")
        if q < config.QUALITY_MIN:
            excluded["low_quality"] += 1
            continue
        for k, v in m.items():
            if v is not None and np.isfinite(v):
                STATE.baselines.add(body.user_id, modality, k, float(v), quality=q, mode=body.mode, phase=body.phase)
    for r in body.readings:
        STATE.baselines.add(body.user_id, modality, r.metric, r.value, quality=r.quality, mode=body.mode, phase=body.phase)
    rows = [b for b in STATE.baselines.for_user(body.user_id) if b["modality"] == modality]
    return envelope(body.user_id, {"modality": modality, "baselines": rows, "excluded": excluded},
                    warnings=["demo baseline: short windows, label every result as demo"] if body.mode == "demo" else [])


@app.post("/v1/baseline/{modality}/confirm", dependencies=[Depends(auth)])
def baseline_confirm(modality: S.Modality, body: S.UserRef):
    user(body.user_id)
    return envelope(body.user_id, {"confirmed": STATE.baselines.confirm(body.user_id, modality)})


@app.post("/v1/baseline/{modality}/reset", dependencies=[Depends(auth)])
def baseline_reset(modality: S.Modality, body: S.BaselineReset):
    user(body.user_id)
    return envelope(body.user_id, {"archived": STATE.baselines.reset(body.user_id, modality, body.phase), "reason": body.reason})


@app.get("/v1/users/{user_id}/baseline", dependencies=[Depends(auth)])
def baseline_get(user_id: str):
    user(user_id)
    return envelope(user_id, {"baselines": STATE.baselines.for_user(user_id)})


@app.post("/v1/compare", dependencies=[Depends(auth)])
def compare(body: S.CompareRequest):
    user(body.user_id)
    comps = []
    for w in body.windows:
        comps += window_comparisons(body.user_id, w)
    for r in body.readings:
        c = STATE.baselines.compare(body.user_id, r.modality, r.metric, r.value, quality=r.quality, unit=r.unit)
        comps.append(c)
    remember(body.user_id, comps)
    return envelope(body.user_id, {"comparisons": comps, "domain_scores": domain_scores(comps),
                                   "flags": [{"domain": c["domain"], "metric": c["metric"], "level": c["level"]}
                                             for c in comps if c["level"] != "green"],
                                   "alerts": alerts_for(comps)})


@app.post("/v1/predict/ecg", dependencies=[Depends(auth)])
def predict_ecg(body: S.EcgRequest):
    user(body.user_id)
    w = body.window
    if w.modality != "ecg":
        raise err(422, "validation_error", "window.modality must be ecg", "window.modality")
    m, q = ecg_window_metrics(w)
    comps = window_comparisons(body.user_id, w)
    remember(body.user_id, comps)
    hr, irr = m["hr_bpm"], m["rr_irregularity"]
    if q < 0.5 or not np.isfinite(hr):
        rhythm = {"flag": None, "low_quality": True}
    else:
        flag = "suspect_tachy" if hr > 130 else "suspect_brady" if hr < 40 else "suspect_irregular" if irr > 0.20 else "none"
        rhythm = {"flag": flag, "rule": "absolute backstop: HR above 130 or below 40 bpm, RR irregularity above 0.20"}
    data = {"quality_index": round(q, 3), "rhythm": rhythm,
            "features": {k: (round(v, 3) if v is not None and np.isfinite(v) else None) for k, v in m.items()},
            "comparison": comps}
    ids = []
    mdl = STATE.models.get("ecg")
    if mdl and len(w.channels) == 12 and abs(w.fs_hz - 100) < 1 and len(w.samples[0]) == 1000:
        from scipy.signal import butter, sosfiltfilt
        x = sosfiltfilt(butter(4, [0.5, 40.0], btype="band", fs=100.0, output="sos"), np.asarray(w.samples, float), axis=-1)
        f, _ = ecg_features(x[None], 100.0, [str(c) for c in range(12)])
        data["superclass_probability"] = {k: round(float(c.predict_proba(f)[0, 1]), 3) for k, c in mdl["model"].items()}
        data["superclass_note"] = "beta draft model (PTB-XL, 12-lead hospital ECG); a probability for review, not a diagnosis"
        ids.append(mdl["model_id"])
    return envelope(body.user_id, data, ids, ["draft model output"] if ids else [])


@app.post("/v1/predict/eeg", dependencies=[Depends(auth)])
def predict_eeg(body: S.EegRequest):
    user(body.user_id)
    w = body.window
    if w.modality != "eeg":
        raise err(422, "validation_error", "window.modality must be eeg", "window.modality")
    if len(w.samples[0]) / w.fs_hz < config.EEG_WIN_S:
        raise err(422, "bad_shape", "EEG window must be at least 4 s", "window.samples")
    feats, names, m, q, n_present = eeg_window_metrics(w)
    comps = window_comparisons(body.user_id, w)
    remember(body.user_id, comps)
    data = {"quality_index": round(q, 3), "channels_mapped": n_present,
            "features": {k: round(v, 4) for k, v in m.items()}, "comparison": comps}
    ids, warn = [], []
    mdl = STATE.models.get("seizure")
    if mdl and q >= 0.5:
        p = mdl["model"].predict_proba(feats[:, [names.index(n) for n in mdl["features"]]])[:, 1]
        thr = mdl["metrics"]["leave_one_subject_out_chbmit"]["hgb"]["threshold_at_1pct_fpr"]
        pmax = float(p.max())
        data["seizure"] = {"flag": "suspect_seizure" if pmax >= thr else "none", "probability": round(pmax, 3),
                           "threshold": round(thr, 3), "override_candidate": pmax >= 0.9,
                           "note": "beta draft model; never raises override O4 on its own, a human reviews"}
        ids.append(mdl["model_id"]); warn.append("draft model output")
    elif mdl:
        data["seizure"] = {"flag": None, "low_quality": True}
    return envelope(body.user_id, data, ids, warn)


@app.post("/v1/predict/radiation-risk", dependencies=[Depends(auth)])
def predict_radiation(body: S.RadiationRequest):
    user(body.user_id)
    di, parts = radiation_domain(body.cumulative_dose_msv, body.spe_dose_msv, body.rate_ratio_to_7d_median, body.seu_count_last_hour)
    proj = (body.spe_projected_dose_msv_24h or 0.0)
    o5 = body.spe_dose_msv >= 250 or proj >= 250
    t_rad = (600 - body.cumulative_dose_msv - 25) / body.dose_equivalent_rate_msv_per_day if body.dose_equivalent_rate_msv_per_day > 0 else None
    with STATE.lock:
        STATE.radiation[body.user_id] = {"domain": di, "cumulative": body.cumulative_dose_msv, "rate": body.dose_equivalent_rate_msv_per_day}
        if o5:
            STATE.overrides.setdefault(body.user_id, set()).add("O5")
    return envelope(body.user_id, {"s": round(di.s, 3), "components": parts, "confidence": "low_confidence" if di.q < 0.5 else "ok",
                                   "remaining_allowance_msv": round(600 - body.cumulative_dose_msv, 1),
                                   "t_rad_days": round(max(t_rad, 0.0), 1) if t_rad is not None else None,
                                   "spe_override": o5, "rule_version": "risk-score.md v2"})


@app.post("/v1/predict/risk", dependencies=[Depends(auth)])
def predict_risk(body: S.RiskRequest):
    user(body.user_id)
    warn = []
    if body.domain_inputs:
        doms = {k: DomainInput(v.s, v.q, v.s_validated) for k, v in body.domain_inputs.items()}
    else:
        comps = list(STATE.latest.get(body.user_id, {}).values())
        ds = domain_scores(comps)
        doms = {k: DomainInput(v["s"], 1.0, v["s_validated"]) for k, v in ds.items()}
        if body.user_id in STATE.radiation:
            doms["radiation"] = STATE.radiation[body.user_id]["domain"]
        if any(c.get("confidence") in ("demo", "synthetic", "provisional") for c in comps):
            warn.append("inputs include demo, synthetic or provisional comparisons: result is provisional")
    overrides = sorted(set(body.overrides) | STATE.overrides.get(body.user_id, set()))
    r = risk(doms, overrides)
    rad = STATE.radiation.get(body.user_id, {})
    hist = STATE.risk_history.setdefault(body.user_id, [])
    r7 = body.r_7d_ago
    if r7 is None:
        old = [x for x in hist if time.time() - x[0] >= 7 * 86400]
        r7 = old[-1][1] if old else None
    m = mdtr(r, doms, logistics_floor_days=body.logistics_floor_days, r_7d_ago=r7,
             cumulative_dose_msv=body.cumulative_dose_msv if body.cumulative_dose_msv is not None else rad.get("cumulative"),
             dose_rate_msv_day=body.dose_rate_msv_day if body.dose_rate_msv_day is not None else rad.get("rate"),
             bone_loss_pct=body.bone_loss_pct, bone_slope_pct_day=body.bone_slope_pct_day)
    hist.append((time.time(), r.r_mdtr))
    acts = [{"action_id": f"{d}_{'red' if r.domains[d]['s'] >= 0.6 else 'yellow'}"} for d in r.flag_domains
            if (d, "red" if r.domains[d]["s"] >= 0.6 else "yellow") in ACTIONS]
    return envelope(body.user_id, {
        "risk": {"r_agg": round(r.r_agg, 3), "r_disp": round(r.r_disp, 3), "band": r.band, "flag": r.flag,
                 "flag_domains": r.flag_domains, "override_ids": r.override_ids, "provisional": bool(warn), "interval": None},
        "domains": r.domains, "mdtr": m, "actions": [a["action_id"] for a in acts],
        "note": "decision support for human review; overrides escalate to a clinician and take no action"}, warnings=warn)


@app.post("/v1/input/self-report", dependencies=[Depends(auth)])
def self_report(body: S.SelfReport):
    user(body.user_id)
    comps = []
    for k in ("sleep_quality", "stress", "fatigue", "mood"):
        v = float(getattr(body, k))
        comps.append(STATE.baselines.compare(body.user_id, "behavioral", k, v, unit="1-5"))
        STATE.baselines.add(body.user_id, "behavioral", k, v)
    remember(body.user_id, comps)
    with STATE.lock:
        STATE.inputs.setdefault(body.user_id, []).append({"type": "self_report", "t": body.t_utc.isoformat()})
    return envelope(body.user_id, {"record_id": uuid.uuid4().hex[:12], "comparison": comps})


@app.post("/v1/input/instrument", dependencies=[Depends(auth)])
def instrument(body: S.Instrument):
    user(body.user_id)
    score = sum(body.answers)
    edges = (5, 10, 15, 20) if body.instrument == "phq9" else (5, 10, 15)
    names = ("minimal", "mild", "moderate", "moderately_severe", "severe") if body.instrument == "phq9" else ("minimal", "mild", "moderate", "severe")
    sev = names[sum(score >= e for e in edges)]
    esc = None
    if body.instrument == "phq9" and body.answers[8] > 0:
        with STATE.lock:
            STATE.overrides.setdefault(body.user_id, set()).add("O1")
        esc = {"override": "O1", "message": "clinician contact required within 1 hour"}
    return envelope(body.user_id, {"score": score, "severity_band": sev, "escalation": esc,
                                   "note": "screening score, not a diagnosis"})


@app.get("/v1/actions", dependencies=[Depends(auth)])
def actions(user_id: str):
    user(user_id)
    return envelope(user_id, {"actions": alerts_for(list(STATE.latest.get(user_id, {}).values()))})
