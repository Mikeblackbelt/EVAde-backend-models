# EVAde model service (beta)

FastAPI service for the hackathon-minimum subset of [api-schema.md](../../lit%20review/planning/api-schema.md). It returns flags and estimates for human review, never diagnoses. By default, state is kept in memory only (one copy, no history rows) and is lost on restart. With `EVADE_STORAGE=cloudflare` it is written through to Cloudflare D1, with blobs in R2 (see [Storage](#storage)). Models are loaded from manifest folders; [evade/models/README.md](../models/README.md) describes the contract.

## Run

```
pip install -r requirements.txt
export EVADE_API_KEY=<shared key, at least 32 characters>   # required; the service refuses to start without it
export EVADE_MODEL_DIR=<folder of <task>/manifest.json model folders, plus spaceweather/>   # optional; see evade/models/README.md
export EVADE_ALLOW_DRAFT=1                         # serve the beta (draft) models, labelled draft
export EVADE_SPACEWX_REPLAY_FILE=<storm.json>       # optional: enables provider "replay" (normalized JSON)
export EVADE_SPACEWX_DISABLE_NETWORK=1             # optional: no NOAA or DONKI provider (offline demos, tests)
export NASA_API_KEY=<key>                          # optional: DONKI via api.nasa.gov when CCMC is down (else DEMO_KEY)
uvicorn evade.api.app:app --host 127.0.0.1 --port 8000 --no-access-log
```

Generate a key with `python -c "import secrets; print(secrets.token_urlsafe(48))"`. Locally, keep it in your shell; on a server it lives in Google Secret Manager (below), never in code, the image or the repo.

With Docker (models mounted read-only, never baked into the image; the image's command already has `--no-access-log`):

```
docker build -t evade-api .
docker run -p 8000:8000 --env-file <file outside the repo with EVADE_API_KEY=...> \
           -e EVADE_ALLOW_DRAFT=1 -v /path/to/evade_train_beta:/models:ro evade-api
```

## Security

- **A key on every request.** Every path needs the header `X-EVAde-Key: <EVADE_API_KEY>`: `GET /v1/health`, unknown paths (401, not 404) and the docs included. The check runs in a middleware before routing, so a route added later cannot skip it, and it compares the bytes in constant time (`hmac.compare_digest`). The service refuses to start without a key or with one shorter than 32 characters. If Cloud Run allows unauthenticated invocations, this key is the only lock.
- **Backend to backend only.** Only our own backend calls this service. There is no CORS, so browsers cannot call it, and the key never goes to a browser or an app.
- **WebSockets too.** No WebSocket route exists yet; a guard in front of the app already closes any WebSocket connection without the key (code 1008).
- **Docs off.** `/docs`, `/redoc` and `/openapi.json` are off unless `EVADE_ENABLE_DOCS=1`, and then they are still behind the key.
- **HTTPS only.** TLS is terminated by Cloud Run or Cloudflare (Tunnel or proxy) in front of the container; the container itself speaks plain HTTP on its private port. With `EVADE_REQUIRE_HTTPS=1` the service also rejects any request whose `X-Forwarded-Proto` is `http` (403 `https_required`). Forwarded headers are trusted only from `FORWARDED_ALLOW_IPS` (default `127.0.0.1`, right for a local Cloudflare Tunnel). On Cloud Run set `FORWARDED_ALLOW_IPS='*'`, because only Google's front end can reach the container and it overwrites `X-Forwarded-Proto`. Never expose port 8000 directly.
- **No health data in logs.** No log line contains a request body, signal values, readings, self-report or instrument answers, journal text, user IDs or header values. Storage and model errors are logged with generic messages, an unexpected exception only by its type and the request ID, and the `botocore`, `httpx` and `httpcore` loggers stay at WARNING. Run uvicorn with `--no-access-log` (the Dockerfile does), because access lines contain URL paths with user IDs. Cloud Run's own request log still records URL paths, which contain pseudonymous user IDs (not names), and never bodies.
- **Validated inputs.** Every request body is a Pydantic model with limits: string lengths and patterns, list lengths and numeric ranges. `user_id` must be a lowercase UUID (as `POST /v1/users` returns it) in bodies, paths and queries alike. `device_id` must match `[A-Za-z0-9][A-Za-z0-9._:-]{2,63}`. Metric, unit and channel names are short tokens, and enums are allow-lists. Bodies are capped at 1 MB and need a `Content-Length`.
- **One error shape.** Every error is `{"error": {"code": ..., "message": ..., "field": ...}}`, with `field` set when one field is at fault. Validation errors never echo the submitted value (pydantic's `input` and `ctx` are dropped).

### Cloud Run deployment (documented here, not automated)

```
printf '%s' "<key>" | gcloud secrets create evade-api-key --data-file=-
gcloud secrets add-iam-policy-binding evade-api-key \
    --member=serviceAccount:<run service account> --role=roles/secretmanager.secretAccessor
gcloud run deploy evade-api --image <image> --region <region> \
    --set-secrets EVADE_API_KEY=evade-api-key:latest \
    --set-env-vars EVADE_REQUIRE_HTTPS=1,FORWARDED_ALLOW_IPS='*' \
    --max-instances 1 --concurrency 20
```

`--max-instances 1` caps the cost and matches the one-process rule for storage below. For a billing budget alert, go to Billing > Budgets & alerts in the console, create a budget for the project with alerts at 50%, 90% and 100%, or run `gcloud billing budgets create --billing-account <account> --display-name evade --budget-amount <amount> --threshold-rule percent=0.5 --threshold-rule percent=0.9 --threshold-rule percent=1.0`.

## Endpoints

| Endpoint | What it does |
|---|---|
| `GET /v1/health` | liveness and which models loaded (key required); never touches storage, so it answers during a D1 outage |
| `GET /v1/models` | served models from their manifests (`EVA-[n]`, status, format, input and output spec, headline metrics, commit, data, `integrity`: `sha256_verified` or `unverified` for a legacy folder without a pinned hash), plus `unavailable` folders with a generic `reason_code` and `reason` (`draft_not_allowed`, `sha256_mismatch`, `runtime_missing`, `manifest_invalid`, ...; details only in the server log), and `deep_models`: each deep task's state (`mock`, `available` or `unavailable` with a reason) |
| `POST /v1/users`, `DELETE /v1/users/{id}` | create a pseudonymous user (a server-generated UUID, or the caller's own `user_id`: idempotent, 409 `user_deleted` for a deleted ID; see [Worker integration](#worker-integration)); delete all of their data, including D1 rows and R2 objects with Cloudflare storage. Returns 503 `deletion_incomplete` if storage fails; the DELETE can be retried, also after a restart, until it returns 200. A completed deletion then gives 404 |
| `POST /v1/baseline/{modality}` | add ECG or EEG windows, or readings, to the user's baseline; locks after the minimum count (demo: 30 windows) |
| `POST /v1/baseline/{modality}/confirm`, `/reset` | confirm a provisional baseline; archive and start over |
| `GET /v1/users/{id}/baseline` | baselines with center, spread, count, status |
| `POST /v1/compare` | compare windows or readings with the user's own baseline: robust z, percent change, level, component score, domain scores, alerts |
| `POST /v1/predict/ecg` | HR and HRV, quality, rule-based rhythm backstop, baseline comparison; for a 12-lead 10 s window at 100 Hz also the beta PTB-XL superclass probabilities (EVA-6) |
| `POST /v1/predict/eeg` | quality, band-power ratios, baseline comparison; with the model loaded, a seizure-suspect flag from the beta model (EVA-2), which never raises override O4 on its own |
| `POST /v1/predict/radiation-risk` | radiation domain score from the risk-score anchors, time to the career limit, solar particle event override O5 |
| `POST /v1/predict/space-weather` | from a server-side provider (`noaa_swpc`, or `replay` if configured) or posted observations: NOAA S/R/G nowcast, SPE-in-progress flag, P(SPE onset within 24 h) from the draft model (EVA-12, if loaded), NASA DONKI context (CMEs, flares, shocks, SEPs; a DONKI outage is only a warning), crew dose-rate estimate, the matching `/v1/predict/radiation-risk` inputs, and O5 as a candidate only; 503 `provider_unavailable` if the feed is down. See [spaceweather README](../spaceweather/README.md) |
| `POST /v1/predict/risk` | risk score R (v2), band, flag, overrides, and maximum days to return (MDTR) with driver; MDTR is withheld without `logistics_floor_days` |
| `POST /v1/input/self-report` | 1-5 ratings compared with the user's own baseline |
| `POST /v1/input/instrument` | PHQ-9 or GAD-7 score and band; PHQ-9 item 9 above 0 raises override O1 (clinician contact) |
| `POST /v1/devices/bind`, `GET`/`DELETE /v1/devices/{id}?user_id=` | bind a Bluetooth device to one user (409 if bound to another), status, unbind (also removes a stored binding this process has not loaded, only for its owner) |
| `POST /v1/ingest/batch` | readings or windows from a bound device with a per-device `seq`; duplicates ignored, gaps counted, results compared with the baseline |
| `GET /v1/actions?user_id=` | action text for current flags (draft wording, to be reviewed) |
| `POST /v1/predict/ecg-deep`, `/sleep-deep`, `/seizure-deep`, `POST /v1/models/reload` | deep models. Until their weights exist they return placeholder outputs (`source: "mock"`, `model_id: null`) in the final schema; they switch to the model when its folder appears. See [README_deep_models.md](README_deep_models.md) |

## Example

```
curl -s -X POST localhost:8000/v1/users -H "X-EVAde-Key: $EVADE_API_KEY" \
     -H "Content-Type: application/json" -d '{"consent": true}'
curl -s -X POST localhost:8000/v1/predict/risk -H "X-EVAde-Key: $EVADE_API_KEY" \
     -H "Content-Type: application/json" \
     -d '{"user_id": "<id>", "domain_inputs": {"behavioral": {"s": 0.65}, "radiation": {"s": 0.4}}, "logistics_floor_days": 0.5}'
```

## Worker integration

The team's backend Worker is the only caller. It sends `X-EVAde-Key` from a Worker secret on every call, and uses the team's own `users.user_id` (a lowercase UUID) as the EVAde `user_id`.

**Register a user** when the team creates one, or lazily before its first call:

```
curl -s -X POST $EVADE_API_URL/v1/users -H "X-EVAde-Key: $EVADE_API_KEY" -H "Content-Type: application/json" \
     -d '{"consent": true, "user_id": "3f2b8c1e-7a4d-4e8b-9c3f-2d1e0a9b8c7d"}'
```

| Case | Response |
|---|---|
| unknown ID | 200, `data: {"user_id", "created": true, "created_at"}`; the user is created with that ID |
| ID already registered | 200, the existing user with `"created": false` (idempotent: retry freely; the attributes sent are ignored) |
| ID was deleted | 409 `user_deleted`: a deleted user is never brought back. Don't retry |
| no `user_id` | 200, a server-generated UUID (unchanged behaviour) |
| malformed ID (not a lowercase UUID) | 422 `validation_error`, `field: "user_id"` |

**Delete a user** when the team deletes theirs:

```
curl -s -X DELETE $EVADE_API_URL/v1/users/3f2b8c1e-7a4d-4e8b-9c3f-2d1e0a9b8c7d -H "X-EVAde-Key: $EVADE_API_KEY"
```

| Case | Response |
|---|---|
| first delete | 200, `data: {"deleted": true, "storage": "memory" or "cloudflare", "blobs_removed": <n>}` |
| unknown or already deleted ID | 404 `user_not_found` |
| storage failed part-way | 503 `deletion_incomplete`: retry the DELETE (also after a restart) until it returns 200 or 404 |

Treat **200 and 404 as success**, and retry a 503 with backoff. After a delete, `POST /v1/users` with that ID returns 409.

**Reading responses.** Every 2xx response uses the same envelope: `request_id`, `user_id`, `server_time`, `api_version`, `model_ids`, `pipeline_version`, `data`, `warnings`. The Worker reads four fields:

- `data`: the result;
- `model_ids`: which `EVA-[n]` models produced it, empty for rule-based results and placeholders;
- `server_time`: when it was computed;
- `pipeline_version`: the preprocessing version, so results can be compared over time.

Errors are `{"error": {"code", "message", "field"}}`.

**Skip storing `source: mock`.** Until their weights are installed, the deep endpoints (`/v1/predict/ecg-deep`, `/sleep-deep`, `/seizure-deep`) answer with `data.source: "mock"` and `model_id: null`. Those outputs are placeholders, not predictions. Show them labelled as placeholders, or not at all, and don't store them as health results. Store a deep result only when `data.source` is `"model"`.

## Sizes and the seizure model

Requests are capped at 1 MB, which is about 30 s of 18-channel EEG at 250 Hz as JSON, so send EEG baselines in several calls. The beta seizure model (EVA-2) uses per-person feature normalization, so it runs for a user only after at least 30 four-second EEG windows (about 1 minute) have been sent to `/v1/baseline/eeg`; before that `/v1/predict/eeg` returns `needs_baseline`. Its output has two levels: `review` (yellow, threshold at 5% window false-positive rate in cross-validation) and `suspect_seizure` (red, 1%). The probability also enters the neuro risk domain as a shadow input, so it raises R but cannot trip the abnormal-component flag or the extreme-domain MDTR cap by itself. Try it with `scripts/demo_replay.py`.

## Behavior rules (from the schema)

- Low-quality windows return null values, `confidence: low_quality`, and no flag.
- Demo, provisional, simulated, or replayed inputs are labelled and make the risk result provisional.
- A request can only touch its own `user_id`; unknown IDs return 404, malformed ones 422.
- Overrides escalate to a human; the API takes no action.
- Draft models are served only with `EVADE_ALLOW_DRAFT=1` and are labelled in every response.

## Storage

`EVADE_STORAGE=memory` (the default) needs no setup. `EVADE_STORAGE=cloudflare` writes every change through to **D1**, using the REST API with parameterized SQL, timeouts, and retries with backoff. Blobs go to **R2**, through the S3 API with boto3. The in-memory state stays the working copy. A user that this process has not seen, for example after a restart, is loaded from D1 on first use.

| Stored in D1 | Not stored |
|---|---|
| users (UUID, created time, cohort, age band, sex as given), baselines and their archive (derived metric values, center, spread), per-user state (radiation domain, overrides, latest comparisons, locked EEG feature median and MAD), reading values and window metadata, non-green flags, risk results, device bindings, tombstones (SHA-256 of deleted IDs) | names, emails, PHQ-9 or GAD-7 answers, and raw waveforms (unless `EVADE_STORE_RAW_WINDOWS=1`, which keeps them in R2 under `raw/<user_id>/`) |

`DELETE /v1/users/{id}` blocks writes for the user, writes the tombstone and a pending-deletion marker, deletes the rows in every table, deletes every R2 object under `raw/<user_id>/` and `exports/<user_id>/`, and clears the marker last. If R2 fails, the marker stays, so a retried DELETE finishes the job even after a restart. The tombstone is enforced inside every D1 write (`INSERT ... SELECT ... WHERE NOT EXISTS (SELECT 1 FROM tombstones ...)`), so a write already in flight when the DELETE commits cannot recreate rows. A user whose tombstone is in D1 is never loaded again. A DELETE for an ID this process does not hold checks the miss cache first, then makes one D1 request; an unfinished deletion is never cached as a miss, so its retry still works. Endpoints re-check under the lock that the user still exists before changing in-memory state, so a request running during the DELETE puts nothing back.

R2 has no tombstone guard. A raw-window upload already in flight when the DELETE lists `raw/<user_id>/` can land after the listing, so the API re-checks the tombstone after its uploads and deletes those keys. This is best effort: a crash between the upload and the re-check can still leave an orphan blob (only with `EVADE_STORE_RAW_WINDOWS=1`).

`POST /v1/users` writes the user row before answering. Without `user_id`, if the row commits but the response is lost, the client's retry creates a second user, and the first row is an orphan whose ID nobody holds. It holds only the creation time and the coarse attributes given (cohort, age band, sex), no health data. This is accepted for the beta. With `user_id` the call is idempotent, so a retry returns the same user. A tombstoned ID gets 409 `user_deleted` (checked in memory, then in D1, and enforced again by the tombstone guard in the D1 insert), so registering can never bring a deleted user back.

**Write rules.** State that later requests depend on is written to storage before the response, and a 503 there never leaves a change that a client retry would apply twice: users and device bindings are written first (a failed bind is rolled back in memory), unbind deletes the stored row before the in-memory one, and a baseline reset archives in D1 before memory. Overrides and the radiation state are idempotent on retry. Audit rows (readings and window metadata, flags, risk results) and the latest comparisons are best effort: a failed write returns 200 with the warning `stored in this service only: ...` instead of a 503. The latest comparisons and the locked EEG feature baseline are rewritten by the flush of the next request that does storage work, and so are changed baselines. A request's best-effort writes go to D1 as one batch. A stale baseline flush cannot bring back a reset baseline, because a reset archives the baseline IDs and archived IDs are never written back. A device's `last_seq` in D1 only moves forward.

**Outages.** D1 calls use a 3 s connect and 5 s read timeout, at most 2 retries that honour `Retry-After`, and a circuit breaker: after 5 consecutive failed attempts every call fails at once for 30 s, so requests return 503 `storage_unavailable` immediately instead of tying up the threadpool. After the 30 s, exactly one trial request goes through; its result closes the circuit or opens it again. R2 calls use a 3 s connect and 5 s read timeout and at most 2 attempts in total, and no retry starts after the request's deadline. Each request has one storage time budget of `EVADE_STORAGE_DEADLINE_S` seconds (default 8) across all its calls, the flush of changed baselines after the endpoint included. A request that did no storage work (`/v1/models`, a 401, a 404 for a user cached as unknown) does not flush, so it never waits on another request's pending writes. `GET /v1/health` never waits on D1. Unknown `user_id`s are remembered for 60 s (bounded), so repeated lookups do not hit D1, and loading a user is one D1 request. The risk history is bounded, both as loaded and in memory: the last 100 points, the oldest point of each hour of the last 7 days, and the newest point at least 7 days old (at most 270 points per user). The MDTR 7-day trend input is then always present when the full history would have one, and at most an hour older.

Setup (once):

1. Create a D1 database (dashboard, Storage & Databases > D1, or `npx wrangler d1 create evade`) and note its database ID.
2. Create an API token (My Profile or Manage Account > API Tokens > Create Custom Token) with the permission **Account > D1 > Edit**, limited to this account.
3. Create an R2 data bucket for raw windows and exports. Then go to R2 > Manage API tokens, create a token with **Object Read & Write** scoped to that bucket only, and copy the Access Key ID and Secret Access Key (the secret is shown once). Model artifacts go in a separate models bucket, and the API's token gets no access to it (see [evade/models/README.md](../models/README.md#syncing-with-cloudflare-r2)).
4. Apply the schema: `python -m evade.storage.migrate`, or `npx wrangler d1 execute <db-name> --remote --file evade/storage/schema.sql`. It is idempotent; apply it again after pulling a change to it (migration 2 adds `pending_deletions`). With `EVADE_STORAGE=cloudflare` the service checks `schema_migrations` at startup and refuses to start if a migration is missing, naming it.
5. `pip install boto3` (only needed for R2), set the variables below as secrets in the deployment (never in the repo), and set `EVADE_STORAGE=cloudflare`.

| Variable | Value |
|---|---|
| `EVADE_STORAGE` | `cloudflare` |
| `CF_ACCOUNT_ID` | account ID (dashboard sidebar or `wrangler whoami`) |
| `CF_API_TOKEN` | API token with Account > D1 > Edit |
| `CF_D1_DATABASE_ID` | D1 database UUID |
| `CF_R2_BUCKET` | data bucket name (optional: without the R2 variables, blob storage is off) |
| `CF_R2_ACCESS_KEY_ID`, `CF_R2_SECRET_ACCESS_KEY` | from the data bucket's R2 API token |
| `EVADE_STORAGE_DEADLINE_S` | per-request storage time budget in seconds (default 8) |
| `CF_R2_ENDPOINT` | only for jurisdiction buckets, for example `https://<account>.eu.r2.cloudflarestorage.com` |
| `EVADE_STORE_RAW_WINDOWS` | `1` to keep raw windows in R2 (off by default) |

**Run one API process (one worker, one replica, `--max-instances 1`) per database.** The in-memory working copy, the deletion gate and the device sequence checks are per process, so several workers writing to the same D1 database could overwrite each other's baselines or let a late write slip past a deletion in another worker. Apply `schema.sql` before the first start with `EVADE_STORAGE=cloudflare`.

Model folders live in their own bucket under `models/<task>/` and are synced with `scripts/fetch_models.py` using a read-only key on servers. Credentials are never logged or included in error messages. A storage outage returns 503 `storage_unavailable` for writes that must be stored, and a warning for best-effort writes. Baseline and state updates that fail to write stay in memory and are retried by the next request that does storage work.

## Not done yet

Per-user tokens, the WebSocket stream (`/v1/ingest/stream`, `/v1/ws`), context tags, manual measurements, and the clinician summary. The EEG and ECG windows are filtered per request, so short windows have edge effects that the training data (filtered per recording) did not.
