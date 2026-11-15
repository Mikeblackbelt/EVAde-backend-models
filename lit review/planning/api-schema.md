# API Schema (EVAde model service)

**Status: plan only, no code.** This is the single contract for the FastAPI service. It consolidates the endpoints scattered across [implementation-pipeline.md](implementation-pipeline.md), [baseline-comparison.md](baseline-comparison.md), and [bluetooth-device-integration.md](bluetooth-device-integration.md), and uses the score and return-time definitions in [risk-score.md](risk-score.md) and [max-days-to-return.md](max-days-to-return.md). Each table maps one-to-one to a Pydantic model when code starts. Values in examples are invented.

## 1. Conventions

| Topic | Rule |
|---|---|
| Base path and version | `/v1/...`; a breaking change needs `/v2` |
| Format | JSON, UTF-8; WebSocket messages are JSON text frames |
| Field names | `snake_case`; enum values lowercase |
| Time | UTC ISO 8601 with `Z` (`2026-11-14T13:05:00Z`); durations in seconds; reject timestamps more than 5 minutes in the future |
| Units | always explicit in a `unit` field or in the field name suffix (`_bpm`, `_ms`, `_mmhg`, `_c`, `_msv`, `_days`); no implicit units |
| Missing value | `null`, never 0 or NaN; NaN and Infinity are rejected on input |
| Numbers | floats rounded to 3 significant decimals in responses |
| Identity | every request carries `user_id` (UUID4, pseudonymous), except `POST /v1/users` and `GET /v1/health`; `user_id` goes in the body for POST and the path or query for GET |
| Auth | `X-EVAde-Key: <key>` on every request (health included), compared in constant time; hackathon: one shared service key (32+ characters, in Google Secret Manager) plus `user_id`, backend to backend only; target: per-user token issued by `POST /v1/users` and checked against the path or body `user_id` |
| Idempotency | write calls accept an `Idempotency-Key` header; ingest deduplicates by (`device_id`, `seq`) |
| CORS | none: only our backend calls the service, never a browser |
| Size limits | request body at most 1 MB; ingest batch at most 500 readings or 60 s of waveform; larger streams use `/ingest/stream` |
| Rate limits | enforced at the edge and in the service; return `429` with `retry_after_s` |
| Docs | OpenAPI generated from the models; every field has a description, unit, and range |
| Claim language | responses use "flag", "estimate", "deviation"; no field is named `diagnosis` |

## 2. Common objects

### 2.1 Enums

| Enum | Values |
|---|---|
| `modality` | `ecg`, `ppg`, `eeg`, `vitals`, `sleep`, `behavioral`, `radiation`, `bone` |
| `level` | `green`, `yellow`, `red` |
| `confidence` | `ok`, `provisional`, `low_quality`, `demo`, `synthetic` |
| `band` | `low`, `guarded`, `elevated`, `high`, `critical` |
| `domain` | `cardiac`, `behavioral`, `neuro`, `radiation`, `bone` |
| `baseline_status` | `building`, `provisional`, `confirmed`, `archived` |
| `data_source` | `device`, `manual`, `simulated`, `replay` |
| `mdtr_driver` | `risk`, `trend`, `radiation_limit`, `bone_limit`, `acute`, `extreme_domain` |

### 2.2 Envelope

Every successful response is wrapped:

| Field | Type | Meaning |
|---|---|---|
| `request_id` | string | server-generated, echoed in logs |
| `user_id` | uuid | echoed |
| `server_time` | datetime | UTC |
| `api_version` | string | for example `1.0.0` |
| `model_ids` | array of `EVA-[n]` | models used, from [MODEL_REGISTRY.md](../../MODEL_REGISTRY.md); empty for rule-based paths |
| `pipeline_version` | string | preprocessing and feature version |
| `data` | object | endpoint-specific payload |
| `warnings` | array of string | non-fatal issues (for example `baseline_provisional`) |

### 2.3 Signal window (input)

| Field | Type | Rules |
|---|---|---|
| `modality` | enum | must match the endpoint |
| `t0_utc` | datetime | window start |
| `fs_hz` | number | allowed range per modality (ecg 100-1000, eeg 100-1000, ppg 25-500) |
| `unit` | string | `mv` for ECG and EEG (`uv` for EEG allowed), `au` for PPG |
| `channels` | array of string | lead or electrode names; EEG from the fixed montage list |
| `samples` | array of array of number | shape channels by samples; 2 s to 60 s of data; length must equal `fs_hz` times duration within 1 sample |
| `quality` | object, optional | `lead_off` bool, `contact` 0-1, `motion` 0-1 |
| `device_id` | string, optional | for bound devices |
| `source` | `data_source` | `device` by default |

### 2.4 Comparison object (output, one per metric)

| Field | Type | Meaning |
|---|---|---|
| `metric` | string | for example `hr_bpm`, `rmssd_ms`, `theta_beta`, `sleep_hours` |
| `modality`, `domain` | enums | routing |
| `value`, `unit` | number or null, string | null when `low_quality` |
| `quality_index` | number 0-1 | |
| `baseline_id`, `baseline_status` | string, enum | which baseline was used |
| `baseline_center`, `baseline_spread`, `n_baseline` | number, number, integer | |
| `z` | number or null | signed robust z, clipped to plus or minus 5 |
| `delta_abs`, `delta_pct` | number or null | |
| `level` | enum | bad-direction level |
| `component_score` | number 0-1 | feeds the risk model |
| `confidence` | enum | |
| `context_used` | array of string | user-input tags applied |
| `shadow` | bool | true for shadow-mode inputs (excluded from the flag test) |

### 2.5 Error object

All errors share one shape.

| Field | Meaning |
|---|---|
| `error.code` | stable machine code (see 8) |
| `error.message` | short human text, no stack traces |
| `error.field` | offending field path, when relevant |
| `error.request_id` | for support |
| `error.retry_after_s` | for 429 and 503 |

## 3. Endpoint catalog

| Group | Endpoint | Method | Purpose |
|---|---|---|---|
| Meta | `/v1/health` | GET | liveness and model load state |
| Meta | `/v1/models` | GET | registry IDs, status, data version of every served model |
| Users | `/v1/users` | POST | create a pseudonymous user |
| Users | `/v1/users/{user_id}` | GET | status and consent flag |
| Users | `/v1/users/{user_id}` | DELETE | delete the user and all records |
| Devices | `/v1/devices/bind` | POST | bind a device to the user |
| Devices | `/v1/devices/{device_id}` | GET | status, last seen, battery, quality |
| Devices | `/v1/devices/{device_id}` | DELETE | unbind |
| Ingest | `/v1/ingest/batch` | POST | upload windows or readings |
| Ingest | `/v1/ingest/stream` | WebSocket | live samples with acknowledgements |
| Baseline | `/v1/baseline/{modality}` | POST | submit windows for baseline building |
| Baseline | `/v1/users/{user_id}/baseline` | GET | all baselines and status |
| Baseline | `/v1/baseline/{modality}/confirm` | POST | confirm a typical period |
| Baseline | `/v1/baseline/{modality}/reset` | POST | archive and start a new phase |
| Compare | `/v1/compare` | POST | compare current data with the baseline |
| Compare | `/v1/compare/summary` | GET | per-modality level and trend |
| Predict | `/v1/predict/ecg` | POST | ECG window to flags and features |
| Predict | `/v1/predict/eeg` | POST | EEG window to flags |
| Predict | `/v1/predict/radiation-risk` | POST | radiation domain score |
| Predict | `/v1/predict/risk` | POST | risk score and maximum days to return |
| Input | `/v1/input/self-report` | POST | mood, sleep quality, stress, fatigue |
| Input | `/v1/input/context` | POST | context tags with a time span |
| Input | `/v1/input/instrument` | POST | PHQ-9, GAD-7, PVT |
| Input | `/v1/input/measurement` | POST | manual vitals, scan values |
| Actions | `/v1/actions` | GET | action protocols for current flags |
| Actions | `/v1/actions/log` | POST, GET | log and read actions taken |
| Live | `/v1/ws` | WebSocket | prototype-style live stream keyed by user |
| Review | `/v1/review/summary` | GET | clinician summary of a user (role-restricted; later phase) |

Hackathon minimum: `health`, `models`, `users`, `baseline/{modality}`, `compare`, `predict/ecg`, `predict/eeg`, `predict/radiation-risk`, `predict/risk`, `input/self-report`, `actions`. The rest follows after the demo.

## 4. Request and response schemas

### 4.1 Users

`POST /v1/users` request:

| Field | Type | Rules |
|---|---|---|
| `consent` | bool | must be true |
| `cohort` | string, optional | for example `demo`, `pilot` |
| `profile` | object, optional | `age_band` (`18-29` ... `60+`), `sex` (`f`, `m`, `x`, `unspecified`), `skin_tone_scale` (Fitzpatrick I-VI or null); no name, email, or free text |

Response `data`: `user_id`, `token` (shown once), `created_at`.

### 4.2 Devices and ingest

`POST /v1/devices/bind` request: `user_id`, `device_id` (client-generated stable ID), `kind` (`hr_strap`, `ecg_patch`, `eeg_headband`, `ppg_wrist`, `phone`, `simulated`), `vendor`, `model`, `firmware`, `modalities` (array of enum). A device is bound to one user at a time; a second bind returns `409 device_already_bound`.

`POST /v1/ingest/batch` request:

| Field | Type | Rules |
|---|---|---|
| `user_id`, `device_id` | uuid, string | device must be bound to the user |
| `seq` | integer | monotonically increasing per device; gaps reported, duplicates ignored |
| `t0_utc` | datetime | first sample time |
| `clock_offset_ms` | number, optional | client clock offset estimated at connection |
| `items` | array of signal window or reading | at most 500 |
| `battery_pct` | number, optional | 0-100 |
| `client_info` | object | `type`, `version` |

A `reading` is `{metric, value, unit, t_utc, quality}` for spot values such as heart rate, SpO2, temperature, and blood pressure.

Response `data`: `accepted` (integer), `duplicates` (integer), `rejected` (array of `{index, code}`), `next_seq_expected`, and a short `comparison_summary` when enough baseline exists.

`/v1/ingest/stream` messages (client to server): `hello` (`user_id`, token, `device_id`), `data` (same fields as a batch item plus `seq`), `ping`. Server to client: `ack` (`seq`), `gap` (`from_seq`, `to_seq`), `assessment` (comparison objects), `error`.

### 4.3 Baseline

`POST /v1/baseline/{modality}` request: `user_id`, `windows` (array of signal window or reading), `phase` (`pre_mission`, `early_flight`, `steady_state`, `post_event`, `demo`), `mode` (`standard` or `demo`; demo uses the short windows and is labelled in every response).

Response `data`: `baseline_id`, `baseline_status`, `n_windows`, `n_needed`, `progress` (0-1), `excluded` (counts by reason: `low_quality`, `context`, `flagged`), `locks_after` (human-readable remainder).

`POST /v1/baseline/{modality}/confirm`: `user_id`, `baseline_id`, `note` (optional). `POST .../reset`: `user_id`, `reason` (enum: `new_phase`, `device_change`, `illness`, `user_request`), `phase`. The old baseline is archived, never deleted.

`GET /v1/users/{user_id}/baseline` returns one entry per modality: `baseline_id`, `status`, `phase`, `n`, `created`, `window_range`, `pipeline_version`, and the feature centers and spreads.

### 4.4 Compare

`POST /v1/compare` request: `user_id`, `items` (array of signal windows or readings, at most 20 windows), `use_context` (bool, default true).

Response `data`: `comparisons` (array of comparison objects from 2.4), `domain_scores` (object mapping each `domain` to `{s, q, shadow_excluded_s, low_confidence}`), `flags` (array of `{domain, metric, level}` for yellow and red), `alerts` (escalations only, with `action_id`).

`GET /v1/compare/summary?user_id=...&window=24h|7d`: per modality `current_level`, `trend` (`rising`, `flat`, `falling`), `last_updated`, `n_flagged_windows`.

### 4.5 Predict

`POST /v1/predict/ecg` request: `user_id`, `window` (signal window with modality `ecg`), `want` (array: `rhythm`, `hrv`, `quality`).

Response `data`:

| Field | Meaning |
|---|---|
| `quality_index` | 0-1; below 0.5 returns null predictions and `low_quality` |
| `rhythm` | `{flag, probability, classes}` where `flag` is `none`, `suspect_irregular`, `suspect_tachy`, `suspect_brady` |
| `features` | `hr_bpm`, `rmssd_ms`, `sdnn_ms`, `rr_irregularity` (null if too few beats) |
| `uncertainty` | interval or confidence for the rhythm probability |
| `comparison` | the comparison objects for the features |

`POST /v1/predict/eeg` request: `user_id`, `window` (modality `eeg`, montage named), `want` (`seizure`, `drowsiness`, `workload`). Response `data`: `quality_index`, `seizure` (`flag`, `probability`, `event_confidence`), `drowsiness` (`flag`, `probability`), `features` (`rel_theta`, `rel_alpha`, `rel_beta`, `theta_beta`), `uncertainty`, `comparison`. Affective markers, if requested, return with `shadow: true` and never affect the flag.

`POST /v1/predict/radiation-risk` request:

| Field | Type | Rules |
|---|---|---|
| `user_id` | uuid | |
| `t_utc` | datetime | |
| `dose_rate_usv_per_h` | number | at least 0 |
| `dose_equivalent_rate_msv_per_day` | number, optional | preferred if the detector gives it |
| `cumulative_dose_msv` | number | person's career or mission dose to date |
| `spe_active` | bool | solar particle event flag |
| `spe_projected_dose_msv_24h` | number, optional | |
| `seu_count_last_hour` | integer | above 1 per hour marks the domain low confidence |
| `location` | string, optional | for example `iss_columbus`, `cabin`, `eva` |

Response `data`: `s` (radiation domain score 0-1), `remaining_allowance_msv`, `t_rad_days` (the radiation-limited time), `confidence`, `rule_version`, `spe_override` (bool, true if override O5 conditions are met).

`POST /v1/predict/risk` request: `user_id`, `domain_inputs` (optional object of precomputed `domain` to `s` and `q`; if omitted the service uses the latest stored comparisons), `logistics_floor_days` (the lead time L; required, else MDTR is withheld), `r_history` (optional array of `{t_utc, r}` for the 7-day trend if not stored).

Response `data`:

| Field | Meaning |
|---|---|
| `risk` | `{r_agg, r_disp, band, flag, flag_domains, override_ids, provisional, interval: {low, high}}` |
| `domains` | per domain `{s, p, q, low_confidence, shadow_excluded}` |
| `mdtr` | `{status, value_days, display, interval_days: {low, high}, driver, provisional, drivers_detail}` |
| `actions` | array of `action_id` for the current band and flags |
| `explain` | top contributing domains and components with their z-scores |

`mdtr.status` is `ok`, `window_cannot_be_met`, or `withheld` (no logistics input). `mdtr.display` is the string shown to people: `"<5 days"`, a number from 5 to 100, or `">100 days"`; `value_days` is the exact value to the nearest 0.5 day. `mdtr.driver` is one of the `mdtr_driver` values.

### 4.6 Inputs

| Endpoint | Request fields | Response `data` |
|---|---|---|
| `/v1/input/self-report` | `user_id`, `t_utc`, `sleep_quality` 1-5, `stress` 1-5, `fatigue` 1-5, `mood` 1-5, `note` (max 500 characters, optional; stored, not analyzed unless the journaling service is enabled) | `record_id`, `comparison` against the person's own self-report baseline |
| `/v1/input/context` | `user_id`, `tag` (`caffeine`, `exercise`, `illness`, `medication_change`, `eva`, `high_workload`, `poor_device_wear`), `start_utc`, `end_utc`, `note` | `record_id`, `affected_metrics` |
| `/v1/input/instrument` | `user_id`, `instrument` (`phq9`, `gad7`, `pvt`), `t_utc`, `answers` (array of integers 0-3 for PHQ-9 and GAD-7) or `pvt` object (`n_trials`, `lapses`, `median_rt_ms`, `duration_s`) | `score`, `severity_band`, `component_score`, `escalation` (for PHQ-9 item 9 above 0: `{override: "O1", message: "clinician contact required"}`) |
| `/v1/input/measurement` | `user_id`, `metric` (`bp_systolic_mmhg`, `bp_diastolic_mmhg`, `temp_c`, `spo2_pct`, `glucose_mgdl`, `weight_kg`, `bmd_g_cm2`, `bmd_site`), `value`, `unit`, `device`, `t_utc` | `record_id`, `comparison` |

Answers, sleep, mood, and blood glucose are stored with the audit fields in 6.

### 4.7 Actions and live stream

`GET /v1/actions?user_id=...`: array of `{action_id, trigger: {metric, level}, title, why, steps, source_status}`. `POST /v1/actions/log`: `user_id`, `action_id`, `t_utc`, `outcome` (`done`, `skipped`, `escalated`), `note`. The action text comes from the prototype's `models/actions.py`, reviewed and sourced by the story owner.

`/v1/ws?user_id=...&token=...`: server pushes a message per 0.5 s chunk or per assessment: `t`, `scenario` (when simulated), `assessment` (comparison objects), `alerts` (escalation only, re-armed on green); client may send `{type: "scenario", name}` (simulated sources only), `{type: "reset_baseline", modality}`, and `{type: "log_action", entry}`.

## 5. Example payloads

`POST /v1/compare` request:

    {
      "user_id": "5c1f6c2e-3a0b-4a7e-9e7a-2b7f3d1e8a10",
      "items": [{
        "modality": "ecg", "t0_utc": "2026-11-14T13:05:00Z", "fs_hz": 250,
        "unit": "mv", "channels": ["II"], "samples": [[0.01, 0.02, 0.05]],
        "source": "device"
      }],
      "use_context": true
    }

Response `data` (abridged):

    {
      "comparisons": [{
        "metric": "rmssd_ms", "modality": "ecg", "domain": "cardiac",
        "value": 24.1, "unit": "ms", "quality_index": 0.92,
        "baseline_id": "b_ecg_0003", "baseline_status": "provisional",
        "baseline_center": 41.0, "baseline_spread": 6.2, "n_baseline": 31,
        "z": -2.7, "delta_pct": -41.2, "level": "red",
        "component_score": 0.52, "confidence": "provisional",
        "context_used": [], "shadow": false
      }],
      "domain_scores": {"cardiac": {"s": 0.52, "q": 0.9, "low_confidence": false}},
      "flags": [{"domain": "cardiac", "metric": "rmssd_ms", "level": "red"}]
    }

`POST /v1/predict/risk` response `data` (abridged):

    {
      "risk": {"r_agg": 0.659, "r_disp": 0.659, "band": "high", "flag": true,
               "flag_domains": ["behavioral", "radiation"], "override_ids": [],
               "provisional": false, "interval": {"low": 0.55, "high": 0.75}},
      "mdtr": {"status": "ok", "value_days": 16.5, "display": "16.5 days",
               "interval_days": {"low": 10.0, "high": 25.0},
               "driver": "trend", "provisional": false}
    }

## 6. Persistence and audit (schema level)

| Record | Key fields |
|---|---|
| user | `user_id`, `created_at`, `consent`, `profile`, `token_hash` |
| device | `device_id`, `user_id`, `kind`, `vendor`, `model`, `firmware`, `bound_at`, `unbound_at` |
| observation | `user_id`, `modality`, `metric`, `t_utc`, `value`, `unit`, `quality_index`, `device_id`, `source`, `context_tags` |
| baseline | `baseline_id`, `user_id`, `modality`, `phase`, `center`, `spread`, `n`, `status`, `window_range`, `pipeline_version`, `created_at`, `archived_at` |
| input | `record_id`, `user_id`, `type`, `t_utc`, `payload`, `created_at`, `created_by`, `previous_value` (for edits) |
| comparison | `user_id`, `t_utc`, the comparison object |
| risk_snapshot | `user_id`, `t_utc`, risk and MDTR blocks, `model_ids`, `pipeline_version` |
| action_log | `user_id`, `action_id`, `t_utc`, `outcome`, `note` |

Every write stores who and when; edits keep the previous value. Raw waveforms are not stored by default (summary statistics and the latest windows only), which the hackathon demo states.

## 7. Behavior rules

1. **Quality gate.** A window below the quality threshold returns null values, `low_quality`, and no flag.
2. **Provisional baselines.** A comparison against a baseline that is not `confirmed` carries `confidence: provisional` or `demo`, and no escalation to red can trigger an override on its own (overrides need absolute triggers from [risk-score.md](risk-score.md)).
3. **Synthetic and simulated data.** Any `source` of `simulated` or `replay` makes the response `confidence: synthetic` or `demo` and the risk response `provisional`.
4. **Authorization.** A request may only touch the `user_id` its token belongs to; the service never returns another user's record, including in error messages.
5. **Overrides.** O1 to O5 produce `override_ids` and an `alerts` entry; the API never takes an action.
6. **MDTR withheld.** Without `logistics_floor_days`, `mdtr.status` is `withheld` and only risk is returned.
7. **Alert on escalation only.** An alert is created when a level rises, and re-armed when it returns to green (the prototype's behavior).
8. **Model gating.** A model whose registry status is `draft` is not served; `models` lists status for every served model.

## 8. Error codes

| HTTP | `error.code` | When |
|---|---|---|
| 400 | `bad_request` | malformed JSON |
| 401 | `unauthenticated` | missing or invalid token |
| 403 | `forbidden` | token does not match the `user_id` |
| 404 | `user_not_found`, `device_not_found`, `baseline_not_found` | unknown identifier |
| 409 | `device_already_bound`, `duplicate_seq`, `baseline_not_confirmable` | state conflict |
| 413 | `payload_too_large` | over the limit |
| 422 | `validation_error`, `bad_sampling_rate`, `bad_shape`, `nan_in_signal`, `future_timestamp`, `stale_timestamp` | schema or content problem, with `error.field` |
| 429 | `rate_limited` | over the rate limit; `retry_after_s` |
| 500 | `internal_error` | unexpected; no details in the body |
| 503 | `model_unavailable`, `baseline_service_unavailable` | dependency down; `retry_after_s` |

Typed errors only; the service never silently drops data and never returns a stale value without its timestamp.

## 9. Mapping to the preprocessing output

The `evade/preprocessing` package emits a table with a fixed schema (windows at 250 Hz, a quality score, ictal labels for training data). The API's signal window and comparison objects use the same field meanings: `t0_utc`, `fs_hz`, `channels`, `quality_index`. A mapping table (preprocessing column to API field) is written when the first endpoint is implemented, so training and serving cannot drift.

## 10. Open items

- Per-user tokens or a shared service token with `user_id` for the demo.
- Whether `/v1/review/summary` and the journaling or voice-agent endpoints belong in this service or a separate one (the journaling and LLM pipeline is a teammate's component; its contract is not defined here).
- Whether self-report free text is stored at all (privacy tradeoff) and whether the journaling service reads it.
- Final enum lists for devices and context tags, after the device list is fixed.
- Rate and size limits, after the host and the latency budget are known.
