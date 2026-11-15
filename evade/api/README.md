# EVAde model service (beta)

FastAPI service for the hackathon-minimum subset of [api-schema.md](../../lit%20review/planning/api-schema.md). It returns flags and estimates for human review, never diagnoses. By default, state is kept in memory and is lost on restart. With `EVADE_STORAGE=cloudflare` it is written through to Cloudflare D1, with blobs in R2 (see [Storage](#storage)). Models are loaded from manifest folders; [evade/models/README.md](../models/README.md) describes the contract.

## Run

```
pip install -r requirements.txt
export EVADE_API_TOKEN=<shared token>             # required; the service refuses requests without it
export EVADE_CORS_ORIGINS=https://<frontend origin> # comma-separated allow-list; no wildcard
export EVADE_MODEL_DIR=<folder of <task>/manifest.json model folders>   # optional; see evade/models/README.md
export EVADE_ALLOW_DRAFT=1                         # serve the beta (draft) models, labelled draft
uvicorn evade.api.app:app --host 127.0.0.1 --port 8000
```

With Docker (models mounted read-only, never baked into the image):

```
docker build -t evade-api .
docker run -p 8000:8000 -e EVADE_API_TOKEN=<token> -e EVADE_CORS_ORIGINS=https://<frontend>            -e EVADE_ALLOW_DRAFT=1 -v /path/to/evade_train_beta:/models:ro evade-api
```

Put it behind HTTPS (for example a Cloudflare Tunnel or proxy); the token travels in a header.

Every request except `GET /v1/health` needs `Authorization: Bearer <token>`. Interactive docs are at `/docs` (OpenAPI).

## Endpoints

| Endpoint | What it does |
|---|---|
| `GET /v1/health` | liveness and which models loaded |
| `GET /v1/models` | served models from their manifests (`EVA-[n]`, status, format, input and output spec, headline metrics, commit, data), plus `unavailable` folders with the reason (draft without the flag, hash mismatch, missing runtime) |
| `POST /v1/users`, `DELETE /v1/users/{id}` | create a pseudonymous user (UUID); delete all of their data, including D1 rows and R2 objects with Cloudflare storage. Returns 503 `deletion_incomplete` if storage fails, and the DELETE can be retried |
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

## Sizes and the seizure model

Requests are capped at 1 MB, which is about 30 s of 18-channel EEG at 250 Hz as JSON, so send EEG baselines in several calls. The beta seizure model (EVA-2) uses per-person feature normalization, so it runs for a user only after at least 30 four-second EEG windows (about 1 minute) have been sent to `/v1/baseline/eeg`; before that `/v1/predict/eeg` returns `needs_baseline`. Its output has two levels: `review` (yellow, threshold at 5% window false-positive rate in cross-validation) and `suspect_seizure` (red, 1%). The probability also enters the neuro risk domain as a shadow input, so it raises R but cannot trip the abnormal-component flag or the extreme-domain MDTR cap by itself. Try it with `scripts/demo_replay.py`.

## Behavior rules (from the schema)

- Low-quality windows return null values, `confidence: low_quality`, and no flag.
- Demo, provisional, simulated, or replayed inputs are labelled and make the risk result provisional.
- A request can only touch its own `user_id`; unknown IDs return 404.
- Overrides escalate to a human; the API takes no action.
- Draft models are served only with `EVADE_ALLOW_DRAFT=1` and are labelled in every response.

## Storage

`EVADE_STORAGE=memory` (the default) needs no setup. `EVADE_STORAGE=cloudflare` writes every change through to **D1**, using the REST API with parameterized SQL, timeouts, and retries with backoff. Blobs go to **R2**, through the S3 API with boto3. The in-memory state stays the working copy. A user that this process has not seen, for example after a restart, is loaded from D1 on first use.

| Stored in D1 | Not stored |
|---|---|
| users (UUID, created time, cohort, age band, sex as given), baselines and their archive (derived metric values, center, spread), per-user state (radiation domain, overrides, latest comparisons, locked EEG feature median and MAD), reading values and window metadata, non-green flags, risk results, device bindings, tombstones (SHA-256 of deleted IDs) | names, emails, PHQ-9 or GAD-7 answers, and raw waveforms (unless `EVADE_STORE_RAW_WINDOWS=1`, which keeps them in R2 under `raw/<user_id>/`) |

`DELETE /v1/users/{id}` blocks writes for the user, writes the tombstone, deletes the rows in every table, then deletes every R2 object under `raw/<user_id>/` and `exports/<user_id>/`.

Setup (once):

1. Create a D1 database (dashboard, Storage & Databases > D1, or `npx wrangler d1 create evade`) and note its database ID.
2. Create an API token (My Profile or Manage Account > API Tokens > Create Custom Token) with the permission **Account > D1 > Edit**, limited to this account.
3. Create an R2 bucket. Then go to R2 > Manage API tokens, create a token with **Object Read & Write** scoped to that bucket, and copy the Access Key ID and Secret Access Key (the secret is shown once).
4. Apply the schema: `python -m evade.storage.migrate`, or `npx wrangler d1 execute <db-name> --remote --file evade/storage/schema.sql`.
5. `pip install boto3` (only needed for R2), set the variables below as secrets in the deployment (never in the repo), and set `EVADE_STORAGE=cloudflare`.

| Variable | Value |
|---|---|
| `EVADE_STORAGE` | `cloudflare` |
| `CF_ACCOUNT_ID` | account ID (dashboard sidebar or `wrangler whoami`) |
| `CF_API_TOKEN` | API token with Account > D1 > Edit |
| `CF_D1_DATABASE_ID` | D1 database UUID |
| `CF_R2_BUCKET` | bucket name (optional: without the R2 variables, blob storage is off) |
| `CF_R2_ACCESS_KEY_ID`, `CF_R2_SECRET_ACCESS_KEY` | from the R2 API token |
| `CF_R2_ENDPOINT` | only for jurisdiction buckets, for example `https://<account>.eu.r2.cloudflarestorage.com` |
| `EVADE_STORE_RAW_WINDOWS` | `1` to keep raw windows in R2 (off by default) |

**Run one API process (one worker, one replica) per database.** The in-memory working copy, the deletion gate and the device sequence checks are per process, so several workers writing to the same D1 database could overwrite each other's baselines or let a late write slip past a deletion in another worker. Apply `schema.sql` before the first start with `EVADE_STORAGE=cloudflare`.

Model folders can live in the same bucket under `models/<task>/` and be synced with `scripts/fetch_models.py`. Credentials are never logged or included in error messages. A storage outage returns 503 `storage_unavailable`. Baseline updates that fail to write stay in memory and are retried on the next request.

## Not done yet

Per-user tokens, the WebSocket stream (`/v1/ingest/stream`, `/v1/ws`), context tags, manual measurements, and the clinician summary. The EEG and ECG windows are filtered per request, so short windows have edge effects that the training data (filtered per recording) did not.
