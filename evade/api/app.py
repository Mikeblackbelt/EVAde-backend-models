"""EVAde model service (beta). Implements the hackathon-minimum subset of lit review/planning/api-schema.md.

Run:  EVADE_API_KEY=<32+ characters> EVADE_MODEL_DIR=~/evade_train_beta EVADE_ALLOW_DRAFT=1 \
      uvicorn evade.api.app:app --no-access-log
Env:
  EVADE_API_KEY        shared API key, at least 32 characters (required: the service refuses to start without it).
                       Every request, /v1/health and unknown paths included, must send it in the X-EVAde-Key header.
  EVADE_ENABLE_DOCS    "1" to serve /docs, /redoc and /openapi.json (still behind the key); off by default
  EVADE_REQUIRE_HTTPS  "1" to reject requests whose X-Forwarded-Proto is not https (TLS ends at Cloud Run/Cloudflare)
  EVADE_MODEL_DIR      folder of model folders <task>/manifest.json (evade/models/README.md), plus spaceweather/
                       (its own sha256-checked loader, evade/spaceweather/model.py); optional
  EVADE_ALLOW_DRAFT    "1" to serve draft (beta) models; they are labelled draft in every response
  EVADE_STORAGE        memory (default, state lost on restart) or cloudflare (D1 + R2; see evade/storage)
  EVADE_STORAGE_DEADLINE_S  per-request time budget for storage calls (default 8); /v1/health never touches storage
  EVADE_SPACEWX_REPLAY_FILE      normalized space-weather JSON for provider "replay" (optional)
  EVADE_SPACEWX_DISABLE_NETWORK  "1" disables the NOAA SWPC provider (offline demos, tests)
Backend-to-backend only: no CORS, so browsers cannot call it. Logs never contain request bodies, signal values,
self-report answers, user IDs or header values; uvicorn's access log (which has URL paths) must be off.
Outputs are flags for human review, never diagnoses.
"""
from __future__ import annotations

import hmac
import logging
import os
import re
import threading
import time
import uuid
from collections import OrderedDict
from dataclasses import asdict
from datetime import datetime, timezone

import numpy as np
from typing import Annotated

from fastapi import FastAPI, HTTPException, Path, Query, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from .. import storage as storage_mod
from ..baseline.store import BaselineStore, domain_scores
from ..models.export import headline_from_metrics
from ..models.registry import ModelRegistry
from ..storage import StorageError, deadline_s, storage_deadline
from ..storage.base import bound_history
from ..preprocessing import PIPELINE_VERSION, config
from ..preprocessing.ecg import detect_qrs, hrv_features, preprocess_ecg
from ..preprocessing.eeg import bipolar_from_monopolar, preprocess_eeg, select_bipolar
from ..preprocessing.signal import make_windows, signal_quality
from ..risk.engine import DomainInput, mdtr, radiation_domain, risk
from ..spaceweather import model as SW
from ..spaceweather.model import load_forecaster
from ..spaceweather.providers import ProviderUnavailable, cmes_frame, get_provider
from ..spaceweather.schema import Observations, Record
from ..training.features import EEG_BANDS, ecg_features, spectral_features
from . import schemas as S
from .deep_models import DEEP_TASKS, build_router as build_deep_router, deep_listing

API_VERSION = "0.1.0-beta"
MAX_BODY = 1_000_000
MAX_SEQ_JUMP = 100_000
RISK_KEEP = 100            # newest risk points kept per user (plus the 7-day trend points, see bound_history)
RISK_MAX = RISK_KEEP + 7 * 24 + 2   # upper bound of a bounded history: + one point per hour bucket of 7 days + 1
MISS_TTL_S, MISS_MAX = 60.0, 10_000
SOFT_WARNING = "stored in this service only: the storage write failed (it is not retried for this request)"
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


log = logging.getLogger("evade.api")
# library loggers stay at WARNING: at DEBUG, botocore logs object keys (which contain user IDs) and request bodies
for _name in ("botocore", "boto3", "s3transfer", "urllib3", "httpx", "httpcore"):
    logging.getLogger(_name).setLevel(logging.WARNING)

API_KEY_ENV, API_KEY_HEADER, API_KEY_MIN_LEN = "EVADE_API_KEY", "x-evade-key", 32


def api_key_problem(key: str | None) -> str | None:
    """Why a configured key is unusable, or None. Never includes the key itself."""
    if not key:
        return f"{API_KEY_ENV} is not set"
    if len(key) < API_KEY_MIN_LEN:
        return f"{API_KEY_ENV} must be at least {API_KEY_MIN_LEN} characters"
    if not re.fullmatch(r"[!-~]+", key):
        return f"{API_KEY_ENV} must be printable ASCII with no spaces (a header cannot carry other characters reliably)"
    return None


def require_api_key() -> None:
    """Called at import: the service refuses to start without a usable key (put it in Secret Manager, not in code)."""
    problem = api_key_problem(os.environ.get(API_KEY_ENV))
    if problem:
        raise RuntimeError(f"refusing to start: {problem}")


def key_ok(sent: str | None) -> bool:
    """Constant-time comparison of the X-EVAde-Key header with EVADE_API_KEY, on the bytes."""
    key = os.environ.get(API_KEY_ENV)
    if api_key_problem(key) or sent is None:
        return False
    return hmac.compare_digest(sent.encode("utf-8"), key.encode("utf-8"))


require_api_key()


class MissCache:
    """user_ids recently not found in storage (bounded LRU with a TTL), so unknown IDs do not hit D1 every time."""

    def __init__(self, ttl_s: float = MISS_TTL_S, max_n: int = MISS_MAX):
        self.ttl_s, self.max_n = ttl_s, max_n
        self._d: OrderedDict[str, float] = OrderedDict()
        self._lock = threading.Lock()

    def hit(self, key: str) -> bool:
        with self._lock:
            t = self._d.get(key)
            if t is None:
                return False
            if time.monotonic() - t > self.ttl_s:
                del self._d[key]
                return False
            return True

    def add(self, key: str) -> None:
        with self._lock:
            self._d[key] = time.monotonic()
            self._d.move_to_end(key)
            while len(self._d) > self.max_n:
                self._d.popitem(last=False)

    def discard(self, key: str) -> None:
        with self._lock:
            self._d.pop(key, None)


class State:
    """In-memory working state. Changes are also written to self.storage (nothing is kept twice with the default
    memory backend); with a persistent backend, a user unknown to this process is loaded from storage on first use.

    Write rules: state that later requests depend on (users, device bindings, overrides, radiation, baseline
    archives) is written to storage before the response, and either storage goes first or the change is idempotent,
    so a 503 and a client retry never apply it twice. Audit rows (inputs, flags, risk results) and the latest
    comparisons are best effort: a failure gives a warning, not a 503; latest and the EEG baseline are retried by the
    flush of the next request that does storage work."""

    def __init__(self):
        self.storage = storage_mod.from_env()
        self.users: dict[str, dict] = {}
        self.pending_deletes: set[str] = set()   # storage deletion failed; DELETE may be retried
        self.missing = MissCache()                # recently unknown user_ids
        self.state_dirty: set[tuple] = set()      # (user_id, state key) whose storage write failed; flushed later
        self.baselines = BaselineStore(self.storage)
        self.latest: dict[str, dict] = {}        # user_id -> {modality: [comparisons]}
        self.radiation: dict[str, dict] = {}
        self.overrides: dict[str, set] = {}
        self.risk_history: dict[str, list] = {}
        self.inputs: dict[str, list] = {}
        self.devices: dict[str, dict] = {}      # device_id -> {user_id, kind, ..., last_seq, last_seen}
        self.eeg_feats: dict[str, dict] = {}    # user_id -> {rows: [...], med, mad}: per-user EEG feature baseline
        self.lock = threading.Lock()
        # manifests are validated and sha256-checked before loading; bad folders are listed as unavailable
        self.registry = ModelRegistry.from_env()
        self.models = self.registry.served()       # task -> {model, features, metrics, model_id, status, manifest}
        sw = load_spaceweather()
        if sw:
            self.models["spaceweather"] = sw


def load_spaceweather() -> dict | None:
    """The space-weather forecaster has its own folder layout (<EVADE_MODEL_DIR>/spaceweather/, manifest with a sha256
    per file), so the manifest registry does not load it. Draft only, so it needs EVADE_ALLOW_DRAFT=1. A hash mismatch
    or unreadable folder leaves the nowcast and dose estimate running without the forecast."""
    root = os.environ.get("EVADE_MODEL_DIR")
    if not root or os.environ.get("EVADE_ALLOW_DRAFT") != "1":
        return None
    try:
        return load_forecaster(root)            # verifies the manifest sha256 on the bytes it then unpickles
    except Exception as e:                      # noqa: BLE001 - one bad folder must not stop the service
        log.warning("spaceweather model folder unavailable: %s", type(e).__name__)
        return None


_DOCS = os.environ.get("EVADE_ENABLE_DOCS") == "1"      # off by default; behind the key like every path when on
app = FastAPI(title="EVAde model service", version=API_VERSION, description="Beta. Flags for human review, not diagnoses.",
              docs_url="/docs" if _DOCS else None, redoc_url="/redoc" if _DOCS else None,
              openapi_url="/openapi.json" if _DOCS else None, redirect_slashes=False)
STATE = State()


class WebSocketKeyGuard:
    """The http middleware below never sees WebSocket connections: close any without the key (policy violation)."""

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] == "websocket":
            want = API_KEY_HEADER.encode("latin-1")
            sent = next((v for k, v in scope.get("headers") or [] if k.lower() == want), None)
            if not key_ok(sent.decode("latin-1") if sent is not None else None):
                await send({"type": "websocket.close", "code": 1008})
                return
        await self.app(scope, receive, send)


app.add_middleware(WebSocketKeyGuard)

# path and query parameters get the same patterns as the request bodies
UserIdPath = Annotated[str, Path(min_length=36, max_length=36, pattern=S.UUID_PATTERN)]
UserIdQuery = Annotated[str, Query(min_length=36, max_length=36, pattern=S.UUID_PATTERN)]
DeviceIdPath = Annotated[str, Path(min_length=3, max_length=64, pattern=S.DEVICE_PATTERN)]


def err(status: int, code: str, message: str, field: str | None = None) -> HTTPException:
    return HTTPException(status, {"code": code, "message": message, "field": field})


def error_response(status: int, code: str, message: str, field: str | None = None) -> JSONResponse:
    """The one error shape: {"error": {"code", "message", "field"}}. Never contains submitted values."""
    return JSONResponse({"error": {"code": code, "message": message, "field": field}}, status_code=status)


_HTTP_CODES = {401: "unauthenticated", 403: "forbidden", 404: "not_found", 405: "method_not_allowed",
               413: "payload_too_large"}


@app.exception_handler(StarletteHTTPException)     # FastAPI's HTTPException and the router's own 404/405
async def _http_err(request: Request, exc: StarletteHTTPException):
    d = exc.detail if isinstance(exc.detail, dict) else {}
    return error_response(exc.status_code, d.get("code") or _HTTP_CODES.get(exc.status_code, "error"),
                          d.get("message") or "request failed", d.get("field"))


_SAFE_LOC = re.compile(r"^[A-Za-z_][A-Za-z0-9_]{0,40}$")
_VALUE_CODES = ("future_timestamp", "nan_in_signal", "bad_shape", "payload_too_large")


def _loc(loc) -> str:
    """Field path of a validation error. Dict keys a client chose are not echoed (pydantic puts them in loc)."""
    loc = list(loc)
    out = [str(x) if (isinstance(x, int) or _SAFE_LOC.match(str(x))) and not (i + 1 < len(loc) and loc[i + 1] == "[key]")
           else "<key>" for i, x in enumerate(loc)]
    return ".".join(out[1:] if out and out[0] in ("body", "query", "path", "header") else out) or None


@app.exception_handler(RequestValidationError)
async def _val_err(request: Request, exc: RequestValidationError):
    # pydantic's "input" and "ctx" hold the submitted value: only the message template and the location are used
    errs = exc.errors()
    e = errs[0] if errs else {}
    msg = str(e.get("msg", "validation error")).removeprefix("Value error, ")
    code = next((c for c in _VALUE_CODES if c in msg), "validation_error")
    if e.get("type") == "json_invalid":
        msg = "body is not valid JSON"
    return error_response(413 if code == "payload_too_large" else 422, code, msg[:200], _loc(e.get("loc", ())))


@app.exception_handler(StorageError)
async def _storage_err(request: Request, exc: StorageError):
    # backend messages never contain credentials, user IDs or submitted values
    log.error("storage error (request %s): %s", getattr(request.state, "rid", "-"), exc)
    return error_response(503, "storage_unavailable", "storage backend unavailable; retry")


def _https_ok(request: Request) -> bool:
    if os.environ.get("EVADE_REQUIRE_HTTPS") != "1":
        return True
    proto = request.headers.get("x-forwarded-proto", "").split(",")[0].strip().lower()
    return proto != "http"


@app.middleware("http")
async def _mw(request: Request, call_next):
    """Runs for every path, matched or not, before any route: HTTPS check, API key, size limits, then one storage
    deadline for the endpoint and the flush after it."""
    rid = request.state.rid = uuid.uuid4().hex[:12]
    if not _https_ok(request):
        return error_response(403, "https_required", "use HTTPS")
    if not key_ok(request.headers.get(API_KEY_HEADER)):
        return error_response(401, "unauthenticated", "missing or invalid X-EVAde-Key header")
    if "transfer-encoding" in request.headers:
        # with Transfer-Encoding the body is framed by chunks and Content-Length is ignored: the cap would not hold
        return error_response(411, "length_required", "Content-Length header required; chunked bodies are not accepted")
    if request.method in ("POST", "PUT", "PATCH"):
        cl = request.headers.get("content-length")
        if cl is None or not cl.isdigit():
            # no chunked uploads: the size cap must be checkable before the body is read
            return error_response(411, "length_required", "Content-Length header required")
        if int(cl) > MAX_BODY:
            return error_response(413, "payload_too_large", "body over 1 MB")
    if request.url.path == "/v1/health":           # never waits on storage
        return await call_next(request)
    with storage_deadline(deadline_s()) as touched:   # one time budget for every storage call of this request
        try:
            resp = await call_next(request)
        except Exception as e:                     # noqa: BLE001 - logged without its message, which may hold values
            log.error("unhandled %s in request %s", type(e).__name__, rid)
            return error_response(500, "internal_error", "internal error")
        # batched writes after the endpoint, outside every lock, inside the same deadline; skipped for requests that
        # did no storage work (/v1/models, 404s from memory), so they never wait on another request's dirty state
        if touched and (STATE.baselines.dirty or STATE.state_dirty):
            try:
                await run_in_threadpool(flush_all)
            except StorageError as e:              # kept in memory and marked dirty; a later request retries
                log.warning("flush failed: %s", e)
            except Exception as e:                 # noqa: BLE001 - the response is done; log the type only
                log.error("flush failed with %s in request %s", type(e).__name__, rid)
    return resp


_SKIP = object()


def state_value(user_id: str, key: str):
    """The current in-memory value of a best-effort state document, or _SKIP if the user is gone."""
    with STATE.lock:
        if user_id not in STATE.users:
            return _SKIP
        if key == "latest":
            return list(STATE.latest.get(user_id, {}).values())
        if key == "eeg_baseline":
            e = STATE.eeg_feats.get(user_id) or {}
            return {"med": e["med"].tolist(), "mad": e["mad"].tolist()} if e.get("med") is not None else None
    raise ValueError(key)


def _write_states(keys) -> None:
    for uid, key in keys:
        v = state_value(uid, key)
        if v is not _SKIP:
            STATE.storage.put_state(uid, key, v)


def flush_all() -> None:
    """Rewrite failed best-effort state documents (current values), then the changed baselines. Runs inside the
    request's own storage deadline (the middleware's), never a second one."""
    with STATE.lock:
        pending, STATE.state_dirty = STATE.state_dirty, set()
    if pending:
        try:
            with STATE.storage.buffered():
                _write_states(sorted(pending))
        except StorageError:
            with STATE.lock:
                STATE.state_dirty |= pending
            raise
    STATE.baselines.flush()


def persist_soft(user_id: str, warn: list, *, state_keys=(), flags=(), windows=(), readings=(), endpoint: str = "",
                 inputs=(), risk=None, device=None) -> None:
    """Best-effort writes for one request, sent as one batch. A failure logs, adds a warning and marks the state
    keys for the next flush; it never fails the request, so a client retry cannot apply the change twice."""
    keys = {(user_id, k) for k in state_keys}
    with STATE.lock:
        STATE.state_dirty -= keys                  # the write below carries the current value
    st = STATE.storage
    try:
        with st.buffered():
            if device is not None:
                st.update_device(*device)          # monotonic last_seq; never re-inserts an unbound device
            _write_states(sorted(keys))
            if flags:
                st.add_flags(user_id, flags)
            rows = input_rows(user_id, windows, endpoint, readings) + list(inputs)
            if rows:
                st.add_inputs(user_id, rows)
            if risk is not None:
                st.add_risk(user_id, *risk)
    except StorageError as e:
        with STATE.lock:
            STATE.state_dirty |= keys
        log.warning("best-effort storage write failed (%s): %s", endpoint or "request", e)
        if SOFT_WARNING not in warn:
            warn.append(SOFT_WARNING)


def user(user_id: str) -> dict:
    u = STATE.users.get(user_id)
    if u is None and STATE.storage.persistent and user_id not in STATE.pending_deletes:
        u = hydrate(user_id)
    if u is None:
        raise err(404, "user_not_found", "unknown user_id", "user_id")
    return u


def hydrate(user_id: str) -> dict | None:
    """Load a user this process has not seen from persistent storage in one request (outside the lock).
    Tombstoned users are refused; misses are cached for MISS_TTL_S."""
    if STATE.missing.hit(user_id):
        return None
    st = STATE.storage
    data = st.load_user(user_id, RISK_KEEP)
    return hydrate_from(user_id, data)


def hydrate_from(user_id: str, data: dict | None) -> dict | None:
    """Install what load_user returned. A miss is cached; a deletion that has not finished is not (a retried DELETE
    must still find it)."""
    st = STATE.storage
    if data is None:
        STATE.missing.add(user_id)
        return None
    if data.get("deletion_incomplete"):
        return None
    rec, doc, devs = data["user"], data["state"], data["devices"]
    with STATE.lock:
        if user_id in STATE.users:
            return STATE.users[user_id]
        if st.is_deleted(user_id, check_remote=False) or user_id in STATE.pending_deletes:
            return None
        STATE.users[user_id] = rec
        if doc.get("latest"):
            STATE.latest[user_id] = {(c["modality"], c["metric"]): c for c in doc["latest"]}
        if doc.get("radiation"):
            r = doc["radiation"]
            STATE.radiation[user_id] = {**r, "domain": DomainInput(**r["domain"])}
        if doc.get("overrides"):
            STATE.overrides[user_id] = set(doc["overrides"])
        if doc.get("eeg_baseline"):
            e = doc["eeg_baseline"]
            STATE.eeg_feats[user_id] = {"rows": [], "med": np.asarray(e["med"], float), "mad": np.asarray(e["mad"], float)}
        for k, v in devs.items():
            STATE.devices.setdefault(k, v)
        STATE.risk_history[user_id] = [tuple(x) for x in data["risk"]]
    STATE.baselines.load(data["baselines"])
    return rec


def envelope(user_id: str | None, data: dict, model_ids: list[str] | None = None, warnings: list[str] | None = None) -> dict:
    return {"request_id": uuid.uuid4().hex[:12], "user_id": user_id, "server_time": datetime.now(timezone.utc).isoformat(),
            "api_version": API_VERSION, "model_ids": model_ids or [], "pipeline_version": PIPELINE_VERSION,
            "data": data, "warnings": warnings or []}


def remember(user_id: str, comps: list[dict]) -> list[dict]:
    """Update the latest comparisons in memory; returns the flag rows for persist_soft."""
    with STATE.lock:
        if user_id not in STATE.users:          # deleted while this request was running
            return []
        cur = STATE.latest.setdefault(user_id, {})
        for c in comps:
            cur[(c["modality"], c["metric"])] = c
    now = time.time()
    return [{"t": now, **{k: c.get(k) for k in ("domain", "metric", "modality", "level", "z", "component_score",
                                               "confidence")}} for c in comps if c.get("level") != "green"]


def input_rows(user_id: str, windows: list, endpoint: str, readings: list = ()) -> list[dict]:
    """Reading values and window metadata (never the samples, unless EVADE_STORE_RAW_WINDOWS=1 puts them in R2)."""
    rows = [{"kind": f"reading:{r.modality}", "t_utc": r.t_utc.isoformat(),
             "meta": {"endpoint": endpoint, "metric": r.metric, "value": r.value, "unit": r.unit, "quality": r.quality}}
            for r in readings]
    keys = []
    try:
        for w in windows:
            key = STATE.storage.put_raw_window(user_id, uuid.uuid4().hex, w.model_dump_json().encode())                 if STATE.storage.store_raw_windows else None
            if key:
                keys.append(key)
            rows.append({"kind": f"window:{w.modality}", "t_utc": w.t0_utc.isoformat(), "blob_key": key,
                         "meta": {"endpoint": endpoint, "fs_hz": w.fs_hz, "channels": len(w.channels),
                                  "n_samples": len(w.samples[0]) if w.samples else 0, "source": w.source}})
    finally:
        # R2 has no tombstone guard: a DELETE that listed raw/<user_id>/ while these PUTs were in flight missed them.
        # Its tombstone is written before that listing, so checking it after the PUTs catches the race, also when a
        # later PUT failed (best effort: a crash between the PUT and this check can still leave an orphan blob).
        if keys and STATE.storage.is_deleted(user_id):
            STATE.storage.delete_blobs(keys)
            keys.clear()
            rows = []
    return rows


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


def eeg_rows_metrics(feats: np.ndarray) -> list[dict]:
    names = spectral_names()
    i = {b: names.index(f"mean_rel_{b}") for b in ("theta", "alpha", "beta")}
    return [{"theta_beta": r[i["theta"]] / max(r[i["beta"]], 1e-9), "rel_beta": r[i["beta"]], "rel_theta": r[i["theta"]],
             "rel_alpha": r[i["alpha"]]} for r in feats]


_NAMES: list[str] = []


def spectral_names() -> list[str]:
    if not _NAMES:
        _NAMES.extend(spectral_features(np.zeros((1, 18, 1000)), 250.0, EEG_BANDS, config.BIPOLAR_NAMES)[1])
    return _NAMES


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


EEG_FEAT_MIN = 30     # 4 s windows (about 1 minute of EEG) before the seizure model can run for a user


def add_eeg_feature_baseline(user_id: str, feats: np.ndarray) -> bool:
    """Returns True when this call locked the baseline (the caller stores the median and MAD, never the rows)."""
    with STATE.lock:
        if user_id not in STATE.users:
            return False
        st = STATE.eeg_feats.setdefault(user_id, {"rows": [], "med": None, "mad": None})
        if st["med"] is not None:
            return False                        # locked; reset by /v1/baseline/eeg/reset
        st["rows"].extend(np.asarray(feats, np.float64).tolist())
        if len(st["rows"]) >= EEG_FEAT_MIN:
            X = np.asarray(st["rows"])
            med = np.median(X, axis=0)
            st["med"], st["mad"] = med, np.maximum(1.4826 * np.median(np.abs(X - med), axis=0), 1e-6)
            return True
    return False


# ---------------- endpoints ----------------
@app.get("/v1/health")
async def health():               # async: never queued behind a threadpool busy with storage calls; key required
    return {"status": "ok", "api_version": API_VERSION, "models_loaded": sorted(STATE.models),
            "server_time": datetime.now(timezone.utc).isoformat()}


@app.get("/v1/models")
def models():
    rows = []
    for task, m in STATE.models.items():
        if task in DEEP_TASKS:                   # listed once, under deep_models
            continue
        man = m.get("manifest") or {}
        verified = bool(m.get("sha256_verified"))
        rows.append({"task": task, "model_id": m["model_id"], "status": m["status"], "available": True,
                     "legacy": bool(m.get("legacy")), "sha256_verified": verified,
                     "integrity": "sha256_verified" if verified else "unverified",
                     "headline": headline(task, m), "format": man.get("format"), "input": man.get("input"),
                     "output": man.get("output"), "code_commit": man.get("code_commit"),
                     "trained_at": man.get("trained_at"), "data": man.get("data")})
    # spaceweather/ is loaded by load_spaceweather (its manifest predates the registry contract): once loaded it is
    # in STATE.models and drops out of this list; if it did not load, the registry's entry says it is unavailable
    unavailable = [e for e in STATE.registry.listing() if not e["available"] and e["task"] not in STATE.models
                   and e["task"] not in DEEP_TASKS]
    return envelope(None, {"models": rows, "unavailable": unavailable, "deep_models": deep_listing(),
                           "draft_served": os.environ.get("EVADE_ALLOW_DRAFT") == "1",
                           "rule_based": ["radiation domain (risk-score.md)", "risk score v2 and MDTR", "baseline comparison",
                                          "space-weather nowcast (NOAA scales) and dose-rate estimate (assumptions)"]})


def headline(task: str, m: dict) -> dict:
    return (m.get("manifest") or {}).get("headline") or headline_from_metrics(task, m.get("metrics") or {})


@app.post("/v1/users")
def create_user(body: S.UserCreate):
    uid = str(uuid.uuid4())
    rec = {"created_at": time.time(), "cohort": body.cohort, "age_band": body.age_band, "sex": body.sex}
    # persisted first: a 503 here leaves no half-created user in memory. If the row commits but the response is lost,
    # the client retries and gets a new user; the first row is an orphan nobody holds the ID of, with only created_at
    # and the coarse attributes above (no health data). Accepted for the beta (see the README).
    STATE.storage.put_user(uid, rec)
    with STATE.lock:
        STATE.users[uid] = rec
    return envelope(uid, {"user_id": uid, "created_at": datetime.now(timezone.utc).isoformat()})


@app.delete("/v1/users/{user_id}")
def delete_user(user_id: UserIdPath):
    st = STATE.storage
    if user_id not in STATE.pending_deletes and user_id not in STATE.users:
        # unknown here: the miss cache first (misses are cached only when no deletion is pending), then one storage
        # request. A deletion whose rows were removed but whose blobs were not (R2 failed, then maybe a restart) is
        # finished by rerunning the idempotent storage delete; a completed deletion gives 404 like any unknown user.
        if not st.persistent or STATE.missing.hit(user_id):
            raise err(404, "user_not_found", "unknown user_id", "user_id")
        data = st.load_user(user_id, RISK_KEEP)
        if not (data and data.get("deletion_incomplete")) and hydrate_from(user_id, data) is None \
                and user_id not in STATE.users:
            raise err(404, "user_not_found", "unknown user_id", "user_id")
    with STATE.lock:
        STATE.baselines.delete_user(user_id)     # tombstone first, so concurrent requests cannot recreate data
        for d in (STATE.users, STATE.latest, STATE.radiation, STATE.overrides, STATE.risk_history, STATE.inputs, STATE.eeg_feats):
            d.pop(user_id, None)
        for dev in [k for k, v in STATE.devices.items() if v["user_id"] == user_id]:
            del STATE.devices[dev]
        STATE.state_dirty = {k for k in STATE.state_dirty if k[0] != user_id}
        STATE.pending_deletes.add(user_id)
    try:
        removed = st.delete_user(user_id)   # storage tombstone, every row, every blob under the user's prefixes
    except StorageError as e:
        log.error("delete_user storage step failed: %s", e)
        raise err(503, "deletion_incomplete", "user data removed from this service but not yet from storage; "
                  "retry DELETE", "user_id") from None
    with STATE.lock:
        STATE.pending_deletes.discard(user_id)
    return envelope(user_id, {"deleted": True, "storage": STATE.storage.name, "blobs_removed": removed.get("blobs", 0)})


@app.post("/v1/baseline/{modality}")
def baseline_submit(modality: S.Modality, body: S.BaselineSubmit):
    user(body.user_id)
    excluded = {"low_quality": 0, "wrong_modality": 0}
    eeg_locked = False
    for w in body.windows:
        if w.modality != modality:
            excluded["wrong_modality"] += 1
            continue
        if w.modality == "ecg":
            m, q = ecg_window_metrics(w)
        elif w.modality == "eeg":
            feats, _, m, q, _ = eeg_window_metrics(w)
        else:
            raise err(422, "validation_error", "ppg windows are not supported in the beta", "modality")
        if q < config.QUALITY_MIN:
            excluded["low_quality"] += 1
            continue
        if w.modality == "eeg":
            eeg_locked |= add_eeg_feature_baseline(body.user_id, feats)
            for row in eeg_rows_metrics(feats):          # one observation per 4 s sub-window
                for k, v in row.items():
                    STATE.baselines.add(body.user_id, "eeg", k, float(v), quality=q, mode=body.mode, phase=body.phase)
            continue
        for k, v in m.items():
            if v is not None and np.isfinite(v):
                STATE.baselines.add(body.user_id, modality, k, float(v), quality=q, mode=body.mode, phase=body.phase)
    for r in body.readings:
        STATE.baselines.add(body.user_id, modality, r.metric, r.value, quality=r.quality, mode=body.mode, phase=body.phase)
    warn = ["demo baseline: short windows, label every result as demo"] if body.mode == "demo" else []
    persist_soft(body.user_id, warn, state_keys=("eeg_baseline",) if eeg_locked else (), windows=body.windows,
                 readings=body.readings, endpoint=f"baseline/{modality}")
    rows = [b for b in STATE.baselines.for_user(body.user_id) if b["modality"] == modality]
    return envelope(body.user_id, {"modality": modality, "baselines": rows, "excluded": excluded}, warnings=warn)


@app.post("/v1/baseline/{modality}/confirm")
def baseline_confirm(modality: S.Modality, body: S.UserRef):
    user(body.user_id)
    return envelope(body.user_id, {"confirmed": STATE.baselines.confirm(body.user_id, modality)})


@app.post("/v1/baseline/{modality}/reset")
def baseline_reset(modality: S.Modality, body: S.BaselineReset):
    user(body.user_id)
    if modality == "eeg":
        with STATE.lock:
            STATE.state_dirty.discard((body.user_id, "eeg_baseline"))
        STATE.storage.put_state(body.user_id, "eeg_baseline", None)     # storage first; idempotent
        with STATE.lock:
            STATE.eeg_feats.pop(body.user_id, None)
    archived = STATE.baselines.reset(body.user_id, modality, body.phase)   # archives in storage, then in memory
    return envelope(body.user_id, {"archived": archived, "reason": body.reason})


@app.get("/v1/users/{user_id}/baseline")
def baseline_get(user_id: UserIdPath):
    user(user_id)
    return envelope(user_id, {"baselines": STATE.baselines.for_user(user_id)})


@app.post("/v1/compare")
def compare(body: S.CompareRequest):
    user(body.user_id)
    comps = []
    for w in body.windows:
        comps += window_comparisons(body.user_id, w)
    for r in body.readings:
        c = STATE.baselines.compare(body.user_id, r.modality, r.metric, r.value, quality=r.quality, unit=r.unit)
        comps.append(c)
    warn: list[str] = []
    persist_soft(body.user_id, warn, state_keys=("latest",), flags=remember(body.user_id, comps), windows=body.windows,
                 readings=body.readings, endpoint="compare")
    return envelope(body.user_id, {"comparisons": comps, "domain_scores": domain_scores(comps),
                                   "flags": [{"domain": c["domain"], "metric": c["metric"], "level": c["level"]}
                                             for c in comps if c["level"] != "green"],
                                   "alerts": alerts_for(comps)}, warnings=warn)


@app.post("/v1/predict/ecg")
def predict_ecg(body: S.EcgRequest):
    user(body.user_id)
    w = body.window
    if w.modality != "ecg":
        raise err(422, "validation_error", "window.modality must be ecg", "window.modality")
    m, q = ecg_window_metrics(w)
    comps = window_comparisons(body.user_id, w)
    warn: list[str] = []
    persist_soft(body.user_id, warn, state_keys=("latest",), flags=remember(body.user_id, comps), windows=[w],
                 endpoint="predict/ecg")
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
    return envelope(body.user_id, data, ids, (["draft model output"] if ids else []) + warn)


@app.post("/v1/predict/eeg")
def predict_eeg(body: S.EegRequest):
    user(body.user_id)
    w = body.window
    if w.modality != "eeg":
        raise err(422, "validation_error", "window.modality must be eeg", "window.modality")
    if len(w.samples[0]) / w.fs_hz < config.EEG_WIN_S:
        raise err(422, "bad_shape", "EEG window must be at least 4 s", "window.samples")
    feats, names, m, q, n_present = eeg_window_metrics(w)
    comps = window_comparisons(body.user_id, w)
    flags = remember(body.user_id, comps)
    data = {"quality_index": round(q, 3), "channels_mapped": n_present,
            "features": {k: round(v, 4) for k, v in m.items()}, "comparison": comps}
    ids, warn = [], []
    mdl = STATE.models.get("seizure")
    stats = STATE.eeg_feats.get(body.user_id, {})
    if mdl and q >= 0.5 and stats.get("med") is not None:
        X = feats[:, [names.index(n) for n in mdl["features"]]]
        X = np.clip((X - stats["med"]) / stats["mad"], -20, 20)       # same per-person normalization as training
        p = mdl["model"].predict_proba(X.astype(np.float32))[:, 1]
        p = np.convolve(p, np.ones(3) / 3, mode="same") if len(p) >= 3 else p   # 3-window smoothing as in training
        hgb = mdl["metrics"]["leave_one_subject_out_chbmit"]["hgb"]
        ops = {round(o["target_fpr"], 2): o["threshold"] for o in hgb.get("operating_points", [])}
        t_red = ops.get(0.01, hgb["threshold_at_1pct_fpr"])
        t_yellow = ops.get(0.05, t_red)
        pmax = float(p.max())
        lvl = "red" if pmax > t_red else "yellow" if pmax > t_yellow else "green"
        data["seizure"] = {"flag": {"red": "suspect_seizure", "yellow": "review", "green": "none"}[lvl], "level": lvl,
                           "probability": round(pmax, 3),
                           "thresholds": {"review_5pct_fpr": round(t_yellow, 3), "suspect_1pct_fpr": round(t_red, 3)},
                           "override_candidate": pmax > t_red,
                           "note": "beta draft model; never raises override O4 on its own, a human reviews"}
        # feed the neuro domain as a shadow input: it raises R but cannot trip the flag or T_extreme alone
        sc = {"red": 0.70, "yellow": 0.40, "green": 0.0}[lvl]
        flags += remember(body.user_id, [{"metric": "seizure_probability", "modality": "eeg", "domain": "neuro", "value": round(pmax, 3),
                                 "level": lvl, "component_score": sc, "shadow": True, "confidence": "demo",
                                 "unit": "probability", "quality_index": round(q, 3)}])
        ids.append(mdl["model_id"]); warn.append("draft model output")
    elif mdl and q >= 0.5:
        data["seizure"] = {"flag": None, "needs_baseline": True,
                           "note": f"submit at least {EEG_FEAT_MIN} 4 s windows of EEG to /v1/baseline/eeg first"}
    elif mdl:
        data["seizure"] = {"flag": None, "low_quality": True}
    persist_soft(body.user_id, warn, state_keys=("latest",), flags=flags, windows=[w], endpoint="predict/eeg")
    return envelope(body.user_id, data, ids, warn)


@app.post("/v1/predict/radiation-risk")
def predict_radiation(body: S.RadiationRequest):
    user(body.user_id)
    di, parts = radiation_domain(body.cumulative_dose_msv, body.spe_dose_msv, body.rate_ratio_to_7d_median, body.seu_count_last_hour)
    proj = (body.spe_projected_dose_msv_24h or 0.0)
    o5 = body.spe_dose_msv >= 250 or proj >= 250
    t_rad = (600 - body.cumulative_dose_msv - 25) / body.dose_equivalent_rate_msv_per_day if body.dose_equivalent_rate_msv_per_day > 0 else None
    with STATE.lock:
        if body.user_id not in STATE.users:
            raise err(404, "user_not_found", "unknown user_id", "user_id")
        STATE.radiation[body.user_id] = {"domain": di, "cumulative": body.cumulative_dose_msv, "rate": body.dose_equivalent_rate_msv_per_day}
        if o5:
            STATE.overrides.setdefault(body.user_id, set()).add("O5")
        ovr = sorted(STATE.overrides.get(body.user_id, set()))
    STATE.storage.put_state(body.user_id, "radiation", {"domain": asdict(di), "cumulative": body.cumulative_dose_msv,
                                                        "rate": body.dose_equivalent_rate_msv_per_day})
    if o5:
        STATE.storage.put_state(body.user_id, "overrides", ovr)
    return envelope(body.user_id, {"s": round(di.s, 3), "components": parts, "confidence": "low_confidence" if di.q < 0.5 else "ok",
                                   "remaining_allowance_msv": round(600 - body.cumulative_dose_msv, 1),
                                   "t_rad_days": round(max(t_rad, 0.0), 1) if t_rad is not None else None,
                                   "spe_override": o5, "rule_version": "risk-score.md v2"})


SPACEWX_HOURS = 72      # the forecast features need 48 h; the NOAA 3-day feeds cover it


@app.post("/v1/predict/space-weather")      # the key is checked by the global middleware
def predict_space_weather(body: S.SpaceWeatherRequest):
    """Nowcast (NOAA S/R/G scales, SPE flag), SPE onset probability (draft model), dose-rate estimate, and the inputs
    for /v1/predict/radiation-risk. Nothing is stored and no override is raised: O5 is returned as a candidate."""
    user(body.user_id)
    if body.observations is not None:
        obs = Observations("request", [Record.make(o.t_utc, **o.model_dump(exclude={"t_utc"}, exclude_none=True))
                                       for o in body.observations],
                           meta={"background_prefix": "swpc:" if body.proton_source == "swpc_operational" else ""})
    else:
        try:
            obs = get_provider(body.provider).fetch(SPACEWX_HOURS)
        except ProviderUnavailable as e:
            raise err(503, "provider_unavailable", f"space-weather provider '{body.provider}' unavailable: {e}", "provider")
    if not obs.has("proton_10mev_pfu"):
        raise err(422, "validation_error", "observations need at least one proton_10mev_pfu value", "observations")
    events, donki_warn, donki_src = None, [], None
    if body.observations is not None and body.cmes is not None:
        events = {"cme": [{"t_utc": c.t_utc.isoformat(), "speed_kms": c.speed_kms, "half_angle_deg": c.half_angle_deg,
                           "width_deg": 2 * c.half_angle_deg if c.half_angle_deg is not None else None,
                           "earth_directed": c.earth_directed} for c in body.cmes]}
        donki_src = "request"
    elif body.provider is not None and body.include_donki:
        try:
            d = get_provider("donki").fetch(96)
            events, donki_warn, donki_src = d.meta.get("events"), d.warnings, d.meta.get("donki_sources")
        except ProviderUnavailable as e:
            donki_warn = [f"DONKI unavailable: {e}"]
    nc = SW.nowcast(obs)
    fc = SW.forecast(STATE.models.get("spaceweather"), obs, nc, cmes=cmes_frame(events))
    dose = SW.dose_estimate(obs, body.location, body.shielding, body.measured_dose_rate_usv_h)
    rin = SW.radiation_inputs(dose, body.cumulative_dose_msv, body.seu_count_last_hour)
    data = {"nowcast": nc, "forecast": fc, "dose": dose, "radiation_inputs": rin, "o5": SW.o5_candidate(dose),
            "donki": {"summary": SW.donki_summary(events), "source": donki_src},
            "assumptions": SW.ASSUMPTIONS,
            "source": {"name": obs.source, "fetched_at": obs.fetched_at.isoformat(), "records": len(obs.records),
                       "warnings": obs.warnings, "meta": obs.meta},
            "note": "flags and estimates for human review, not dosimetry; forward radiation_inputs to "
                    "/v1/predict/radiation-risk to update the radiation domain"}
    if body.cumulative_dose_msv is not None:
        di, parts = radiation_domain(body.cumulative_dose_msv, rin["spe_dose_msv"], rin["rate_ratio_to_7d_median"],
                                     body.seu_count_last_hour)
        rate = rin["dose_equivalent_rate_msv_per_day"]
        t_rad = (600 - body.cumulative_dose_msv - 25) / rate if rate > 0 else None
        data["radiation_domain_preview"] = {"s": round(di.s, 3), "components": parts,
                                            "confidence": "low_confidence" if di.q < 0.5 else "ok",
                                            "t_rad_days": round(max(t_rad, 0.0), 1) if t_rad is not None else None,
                                            "rule_version": "risk-score.md v2"}
    ids = [fc["model_id"]] if fc.get("model_id") else []
    warn = nc["warnings"] + fc.get("warnings", []) + obs.warnings + donki_warn + (["draft model output"] if ids else [])
    if dose["source"] == "estimated":
        warn.append("dose rate is a model estimate from assumptions, not a measurement")
    return envelope(body.user_id, data, ids, warn)


@app.post("/v1/predict/risk")
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
    hist = list(STATE.risk_history.get(body.user_id, []))
    r7 = body.r_7d_ago
    if r7 is None:
        old = [x for x in hist if time.time() - x[0] >= 7 * 86400]
        r7 = old[-1][1] if old else None
    m = mdtr(r, doms, logistics_floor_days=body.logistics_floor_days, r_7d_ago=r7,
             cumulative_dose_msv=body.cumulative_dose_msv if body.cumulative_dose_msv is not None else rad.get("cumulative"),
             dose_rate_msv_day=body.dose_rate_msv_day if body.dose_rate_msv_day is not None else rad.get("rate"),
             bone_loss_pct=body.bone_loss_pct, bone_slope_pct_day=body.bone_slope_pct_day)
    now = time.time()
    provisional = bool(warn)
    with STATE.lock:
        if body.user_id in STATE.users:
            # bounded like the hydrated copy: the newest RISK_KEEP points plus the oldest point of every hour of the
            # last 7 days and the newest older point, which is all the 7-day trend lookup above ever needs
            hist = STATE.risk_history.setdefault(body.user_id, [])
            hist.append((now, r.r_mdtr))
            if len(hist) > RISK_MAX:
                STATE.risk_history[body.user_id] = [tuple(x) for x in bound_history(hist, RISK_KEEP, now)]
    persist_soft(body.user_id, warn, endpoint="predict/risk",
                 risk=(now, r.r_mdtr, r.band, {"r_agg": r.r_agg, "r_disp": r.r_disp, "flag": r.flag,
                                               "flag_domains": r.flag_domains, "override_ids": r.override_ids,
                                               "provisional": provisional, "mdtr": m}))
    acts =[{"action_id": f"{d}_{'red' if r.domains[d]['s'] >= 0.6 else 'yellow'}"} for d in r.flag_domains
            if (d, "red" if r.domains[d]["s"] >= 0.6 else "yellow") in ACTIONS]
    return envelope(body.user_id, {
        "risk": {"r_agg": round(r.r_agg, 3), "r_disp": round(r.r_disp, 3), "band": r.band, "flag": r.flag,
                 "flag_domains": r.flag_domains, "override_ids": r.override_ids, "provisional": provisional, "interval": None},
        "domains": r.domains, "mdtr": m, "actions": [a["action_id"] for a in acts],
        "note": "decision support for human review; overrides escalate to a clinician and take no action"}, warnings=warn)


@app.post("/v1/input/self-report")
def self_report(body: S.SelfReport):
    user(body.user_id)
    comps = []
    for k in ("sleep_quality", "stress", "fatigue", "mood"):
        v = float(getattr(body, k))
        comps.append(STATE.baselines.compare(body.user_id, "behavioral", k, v, unit="1-5"))
        STATE.baselines.add(body.user_id, "behavioral", k, v)
    flags = remember(body.user_id, comps)
    with STATE.lock:
        if body.user_id in STATE.users:          # deleted while this request was running: nothing is put back
            STATE.inputs.setdefault(body.user_id, []).append({"type": "self_report", "t": body.t_utc.isoformat()})
    warn: list[str] = []
    persist_soft(body.user_id, warn, state_keys=("latest",), flags=flags, endpoint="input/self-report",
                 inputs=[{"kind": "self_report", "t_utc": body.t_utc.isoformat(),
                          "meta": {k: getattr(body, k) for k in ("sleep_quality", "stress", "fatigue", "mood")}}])
    return envelope(body.user_id, {"record_id": uuid.uuid4().hex[:12], "comparison": comps}, warnings=warn)


@app.post("/v1/input/instrument")
def instrument(body: S.Instrument):
    user(body.user_id)
    score = sum(body.answers)
    edges = (5, 10, 15, 20) if body.instrument == "phq9" else (5, 10, 15)
    names = ("minimal", "mild", "moderate", "moderately_severe", "severe") if body.instrument == "phq9" else ("minimal", "mild", "moderate", "severe")
    sev = names[sum(score >= e for e in edges)]
    esc = None
    if body.instrument == "phq9" and body.answers[8] > 0:
        with STATE.lock:
            if body.user_id not in STATE.users:  # deleted while this request was running: nothing is put back
                raise err(404, "user_not_found", "unknown user_id", "user_id")
            STATE.overrides.setdefault(body.user_id, set()).add("O1")
            ovr = sorted(STATE.overrides[body.user_id])
        STATE.storage.put_state(body.user_id, "overrides", ovr)     # the answers themselves are not stored
        esc ={"override": "O1", "message": "clinician contact required within 1 hour"}
    return envelope(body.user_id, {"score": score, "severity_band": sev, "escalation": esc,
                                   "note": "screening score, not a diagnosis"})


@app.get("/v1/actions")
def actions(user_id: UserIdQuery):
    user(user_id)
    return envelope(user_id, {"actions": alerts_for(list(STATE.latest.get(user_id, {}).values()))})


@app.post("/v1/devices/bind")
def bind_device(body: S.DeviceBind):
    user(body.user_id)
    # a binding stored for a user this process has not loaded still counts (deleting a user deletes its bindings)
    stored = STATE.storage.get_device(body.device_id) if STATE.storage.persistent else None
    with STATE.lock:
        if body.user_id not in STATE.users:      # deleted while this request was running: nothing is put back
            raise err(404, "user_not_found", "unknown user_id", "user_id")
        prev = STATE.devices.get(body.device_id)
        cur = prev or stored
        if cur and cur["user_id"] != body.user_id and (cur["user_id"] in STATE.users or cur is stored):
            raise err(409, "device_already_bound", "device is bound to another user; unbind it first", "device_id")
        rec = STATE.devices[body.device_id] = {"user_id": body.user_id, "kind": body.kind, "vendor": body.vendor,
                                               "model": body.model, "firmware": body.firmware, "last_seq": -1,
                                               "last_seen": None, "battery_pct": None, "gaps": 0}
        snapshot = dict(rec)
    try:
        STATE.storage.put_device(body.device_id, snapshot)
    except StorageError:                         # roll back, so the 503 leaves the service as it was
        with STATE.lock:
            if STATE.devices.get(body.device_id) is rec:
                if prev is None:
                    del STATE.devices[body.device_id]
                else:
                    STATE.devices[body.device_id] = prev
        raise
    return envelope(body.user_id, {"device_id": body.device_id, "bound": True})


@app.delete("/v1/devices/{device_id}")
def unbind_device(device_id: DeviceIdPath, user_id: UserIdQuery):
    user(user_id)
    st = STATE.storage
    with STATE.lock:
        cur = STATE.devices.get(device_id)
    if cur is None and st.persistent:
        cur = st.get_device(device_id)           # stored binding this process does not hold
    if not cur or cur["user_id"] != user_id:
        raise err(404, "device_not_found", "no such device for this user", "device_id")
    st.delete_device(device_id, user_id)         # storage first: a 503 leaves the binding in place for the retry
    with STATE.lock:
        c = STATE.devices.get(device_id)
        if c is not None and c["user_id"] == user_id:
            del STATE.devices[device_id]
    return envelope(user_id, {"device_id": device_id, "bound": False})


@app.get("/v1/devices/{device_id}")
def device_status(device_id: DeviceIdPath, user_id: UserIdQuery):
    user(user_id)
    cur = STATE.devices.get(device_id)
    if not cur or cur["user_id"] != user_id:
        raise err(404, "device_not_found", "no such device for this user", "device_id")
    return envelope(user_id, {"device_id": device_id, **{k: v for k, v in cur.items() if k != "user_id"}})


@app.post("/v1/ingest/batch")
def ingest_batch(body: S.IngestBatch):
    user(body.user_id)
    def check():
        dev = STATE.devices.get(body.device_id)
        if not dev or dev["user_id"] != body.user_id:
            raise err(404, "device_not_found", "bind the device to this user first", "device_id")
        if dev["last_seq"] >= 0 and body.seq - dev["last_seq"] > MAX_SEQ_JUMP:
            raise err(422, "seq_jump_too_large", f"seq may advance by at most {MAX_SEQ_JUMP}", "seq")
        return dev
    with STATE.lock:
        dev = check()
        if body.seq <= dev["last_seq"]:
            return envelope(body.user_id, {"accepted": 0, "duplicates": 1, "next_seq_expected": dev["last_seq"] + 1})
    # process first; the sequence number is committed only after the batch succeeded
    comps = []
    for w in body.windows:
        comps += window_comparisons(body.user_id, w)
    for r in body.readings:
        comps.append(STATE.baselines.compare(body.user_id, r.modality, r.metric, r.value, quality=r.quality, unit=r.unit))
    with STATE.lock:
        dev = check()
        if body.seq <= dev["last_seq"]:            # a concurrent request with the same seq won the race
            return envelope(body.user_id, {"accepted": 0, "duplicates": 1, "next_seq_expected": dev["last_seq"] + 1})
        gap = body.seq - dev["last_seq"] - 1 if dev["last_seq"] >= 0 else 0
        dev["gaps"] += max(gap, 0)
        dev["last_seq"], dev["last_seen"] = body.seq, datetime.now(timezone.utc).isoformat()
        if body.battery_pct is not None:
            dev["battery_pct"] = body.battery_pct
        dev_rec = dict(dev)
    # the seq is committed: a storage failure from here on must not fail the request, or the device's retry
    # would be counted as a duplicate and the batch lost; the in-memory state is already updated
    warn: list[str] = []
    persist_soft(body.user_id, warn, state_keys=("latest",), flags=remember(body.user_id, comps), windows=body.windows,
                 readings=body.readings, endpoint="ingest/batch", device=(body.device_id, dev_rec))
    flagged = [c for c in comps if c["level"] != "green"]
    return envelope(body.user_id, {"accepted": len(body.windows) + len(body.readings), "duplicates": 0,
                                   "gap_before": max(gap, 0), "next_seq_expected": body.seq + 1,
                                   "comparison_summary": {"n": len(comps), "n_flagged": len(flagged),
                                                          "levels": sorted({c["level"] for c in comps})},
                                   "alerts": alerts_for(comps)}, warnings=warn)


app.include_router(build_deep_router())         # deep model endpoints (evade/api/deep_models.py); the middleware checks the key
