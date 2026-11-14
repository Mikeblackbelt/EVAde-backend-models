# EVAde model service (beta)

FastAPI service for the hackathon-minimum subset of [api-schema.md](../../lit%20review/planning/api-schema.md). It returns flags and estimates for human review, never diagnoses. State is kept in memory and is lost on restart.

## Run

```
pip install -r requirements.txt
export EVADE_API_TOKEN=<shared token>             # required; the service refuses requests without it
export EVADE_CORS_ORIGINS=https://<frontend origin> # comma-separated allow-list; no wildcard
export EVADE_MODEL_DIR=<folder with seizure/ and ecg/ from evade.training>   # optional
export EVADE_ALLOW_DRAFT=1                         # serve the beta (draft) models, labelled draft
uvicorn evade.api.app:app --host 127.0.0.1 --port 8000
```

Every request except `GET /v1/health` needs `Authorization: Bearer <token>`. Interactive docs are at `/docs` (OpenAPI).

## Endpoints

| Endpoint | What it does |
|---|---|
| `GET /v1/health` | liveness and which models loaded |
| `GET /v1/models` | served model IDs (`EVA-[n]`), status, headline metrics |
| `POST /v1/users`, `DELETE /v1/users/{id}` | create a pseudonymous user (UUID), delete all of their data |
| `POST /v1/baseline/{modality}` | add ECG or EEG windows, or readings, to the user's baseline; locks after the minimum count (demo: 30 windows) |
| `POST /v1/baseline/{modality}/confirm`, `/reset` | confirm a provisional baseline; archive and start over |
| `GET /v1/users/{id}/baseline` | baselines with center, spread, count, status |
| `POST /v1/compare` | compare windows or readings with the user's own baseline: robust z, percent change, level, component score, domain scores, alerts |
| `POST /v1/predict/ecg` | HR and HRV, quality, rule-based rhythm backstop, baseline comparison; for a 12-lead 10 s window at 100 Hz also the beta PTB-XL superclass probabilities (EVA-6) |
| `POST /v1/predict/eeg` | quality, band-power ratios, baseline comparison; with the model loaded, a seizure-suspect flag from the beta model (EVA-2), which never raises override O4 on its own |
| `POST /v1/predict/radiation-risk` | radiation domain score from the risk-score anchors, time to the career limit, solar particle event override O5 |
| `POST /v1/predict/risk` | risk score R (v2), band, flag, overrides, and maximum days to return (MDTR) with driver; MDTR is withheld without `logistics_floor_days` |
| `POST /v1/input/self-report` | 1-5 ratings compared with the user's own baseline |
| `POST /v1/input/instrument` | PHQ-9 or GAD-7 score and band; PHQ-9 item 9 above 0 raises override O1 (clinician contact) |
| `POST /v1/devices/bind`, `GET`/`DELETE /v1/devices/{id}?user_id=` | bind a Bluetooth device to one user (409 if bound to another), status, unbind |
| `POST /v1/ingest/batch` | readings or windows from a bound device with a per-device `seq`; duplicates ignored, gaps counted, results compared with the baseline |
| `GET /v1/actions?user_id=` | action text for current flags (draft wording, to be reviewed) |

## Example

```
curl -s -X POST localhost:8000/v1/users -H "Authorization: Bearer $EVADE_API_TOKEN" \
     -H "Content-Type: application/json" -d '{"consent": true}'
curl -s -X POST localhost:8000/v1/predict/risk -H "Authorization: Bearer $EVADE_API_TOKEN" \
     -H "Content-Type: application/json" \
     -d '{"user_id": "<id>", "domain_inputs": {"behavioral": {"s": 0.65}, "radiation": {"s": 0.4}}, "logistics_floor_days": 0.5}'
```

## Behavior rules (from the schema)

- Low-quality windows return null values, `confidence: low_quality`, and no flag.
- Demo, provisional, simulated, or replayed inputs are labelled and make the risk result provisional.
- A request can only touch its own `user_id`; unknown IDs return 404.
- Overrides escalate to a human; the API takes no action.
- Draft models are served only with `EVADE_ALLOW_DRAFT=1` and are labelled in every response.

## Not done yet

Per-user tokens, persistent storage (Postgres or the team's Cloudflare database), the WebSocket stream (`/v1/ingest/stream`, `/v1/ws`), context tags, manual measurements, and the clinician summary. The EEG and ECG windows are filtered per request, so short windows have edge effects that the training data (filtered per recording) did not.
