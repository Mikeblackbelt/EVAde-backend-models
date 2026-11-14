# Implementation Pipeline: Ingestion, Models, and Serving

**Status: theoretical plan, no code.** This file turns the review into the three-stage engineering task now assigned to the model backend: (1) acquire and preprocess data, (2) develop and train models, (3) serve them through a FastAPI service that the frontend ("Frankend") and the team's cloud services (Cloudflare Workers, databases) call. Values marked *design* are assumptions to be tuned; values marked *lit* trace to a source in this review.

## Stage 1. Data acquisition, ingestion, and preprocessing

### 1.1 Sources and order of use

| Priority | Source | Use | Access note |
|---|---|---|---|
| 1 | PhysioNet (MIT-BIH, PTB-XL, BIDMC, Sleep-EDFx), TUSZ, CHB-MIT, SeizeIT2, WESAD | Reproduce published baselines; build and test the pipeline on open data | see [data-collection-sources](../data-collection-sources/README.md) |
| 2 | NASA Open Science Data Repository (OSDR) incl. RadLab | Radiation and spaceflight biology; public processed data | public; controlled human data needs a request |
| 3 | NASA Life Sciences Data Archive (LSDA) via the NASA Human Research Program portal | Astronaut physiology, bone, behavioral data | request, review, and IRB-style approval; weeks to months, so start the request early |
| 4 | Simulated-mission and analog data (collected by the team) | Time-in-mission behavior | needs ethics approval; none exists yet |

Important limit: the NASA human datasets are small, de-identified, and slow to obtain. Real-time EEG and ECG streams from astronauts are not publicly available, so Stage 2 models are first trained and validated on terrestrial open data, and nothing is claimed about spaceflight performance until analog or NASA data are tested (see [calibration-and-validation.md](calibration-and-validation.md)).

### 1.2 Ingestion layer (design)

- **Raw store:** immutable, content-hashed files per dataset version, kept out of git; a manifest records source URL, licence, date, checksum.
- **Canonical format:** one table or array file per recording with the shared schema in [integration-and-evaluation.md](../integration-and-evaluation.md) (participant_id, UTC timestamp, modality, device, value, units, quality_index, radiation_flag).
- **Timestamps:** convert everything to UTC with the source clock ID; record device drift where known; resample onto a common grid only inside the windowing step, never in the raw store.
- **Splits:** assign subject-wise folds once at ingestion (and a time-forward holdout where timestamps exist), store the split file with a version, and make loaders refuse to run without it.

### 1.3 Preprocessing specification (design, to be tuned per dataset)

| Signal | Steps | Starting values |
|---|---|---|
| ECG | baseline-wander removal, band-pass, mains notch, resample, R-peak detection (Pan-Tompkins baseline), beat-quality index | band-pass about 0.5-40 Hz; notch 50 or 60 Hz; resample to 250 Hz; 30 s windows with 50% overlap for rhythm, 5 min windows for HRV |
| PPG / rPPG | band-pass, motion gating, peak detection, signal quality index | pulse band 0.7-3.5 Hz; reject windows below the SQI threshold and report retention rate |
| EEG | band-pass, mains notch, bad-channel handling, artifact handling (ICA or ASR), re-reference, channel mapping to a fixed montage | band-pass 0.5-45 Hz; resample to 250 Hz; 4 s model windows; fixed 10-20 subset for wearables |
| Radiation telemetry | unit normalization, outlier and dropout handling, event flag from single-event-upset counter | 1-min dose-rate bins; SEU rate above 1 per hour marks the radiation domain low-confidence (matches [risk-score.md](risk-score.md)) |
| Behavioral | instrument scoring (PHQ-9, GAD-7, PVT), per-person robust z-score | baseline from the first stable days of the mission; rolling window 14 days |

Normalization rule: fit scalers on training subjects only; for per-person features, fit on that person's own baseline period. Never fit on the full dataset before splitting (a standard leakage source; see integration document, item 10).

Libraries named in the task (pandas, NumPy, SciPy) cover this stage; domain libraries (for example MNE for EEG, a PPG or ECG toolbox) are candidates, to be chosen when code starts.

## Stage 2. Model development and training

### 2.1 Targets (first versions)

| Module | Target | First model (from [model-architectures](../model-architectures/README.md)) |
|---|---|---|
| ECG | rhythm or quality class and HRV features per window | stage-1 rules, then a 1D ResNet |
| EEG | seizure or drowsiness flag per window | features plus boosted trees, then EEGNet |
| Radiation | exposure-based radiation domain score | rule-based formulas from [risk-score.md](risk-score.md); no learned model because no labeled spaceflight outcome data exist |
| Behavioral | deviation from personal baseline | per-person robust z-score |
| Aggregate | risk score R and days to return | formulas in [risk-score.md](risk-score.md) and [days-to-return.md](days-to-return.md) |

"Cardiovascular strain", "cognitive fatigue", and "high-risk radiation threshold" in the task are treated as *flag outputs for human review*, not diagnoses.

### 2.2 Training protocol (design)

1. Frameworks: PyTorch or TensorFlow; pick one and keep it for all `EVA-[n]` architectures.
2. Splits: subject-wise train, validation, and test, plus an external dataset when available; test set opened once per pre-registered claim.
3. Tuning: hyperparameters tuned on validation only, with search budget recorded.
4. Always report a trivial baseline (logistic regression and per-person mean) beside every model.
5. Metrics: event-based seizure scoring, Bland-Altman for HR and HRV, sensitivity at fixed false alarms, calibration, subgroup results (see integration document, section 3).
6. Latency and size: record parameter count, inference time on the target CPU, and memory; a model that cannot meet the serving budget below is not promoted.
7. Registration: every retained architecture and trained model gets an `EVA-[n]` ID in [MODEL_REGISTRY.md](../../MODEL_REGISTRY.md) before it is served. A model with `Status = draft` is not exposed by any production endpoint.

## Stage 3. FastAPI backend and delivery

### 3.1 Endpoints (design)

| Endpoint | Method | Input | Output |
|---|---|---|---|
| `/predict/ecg` | POST | one window of samples plus sampling rate, lead name, participant ID, UTC start | quality index, rhythm or quality flag with probability, HR and HRV features, model `EVA-[n]`, uncertainty |
| `/predict/eeg` | POST | one window (channels by samples), montage, sampling rate, participant ID, UTC start | quality index, seizure or drowsiness flag with probability, model ID, uncertainty |
| `/predict/radiation-risk` | POST | dose-rate or dose samples, SEU count, solar-particle-event flag, cumulative dose to date | radiation domain score in [0, 1], remaining-dose margin, confidence, rule version |
| `/predict/risk` | POST | latest domain scores or raw features for one participant | R in (0, 1), band, DTR display ("<5", 5-100, ">100"), top contributing domains |
| `/models` | GET | none | registry IDs, status, and data version of every served model |
| `/health` | GET | none | liveness, model load state, version |

Rules: every response carries `model_id`, `model_version`, `data_version`, `quality_index`, and an `uncertainty`; a window below the quality threshold returns the value as null with `low_quality` set, not a guess.

### 3.2 Service requirements (design)

| Topic | Choice |
|---|---|
| Validation | Pydantic request and response models with explicit units, ranges, and maximum lengths; reject NaN, wrong sampling rate, or oversized payloads with a clear 422 |
| Concurrency | async endpoints for I/O; run model inference in a worker pool so the event loop is not blocked |
| Model loading | load once at startup from a registry-keyed path; refuse to start if a served model is not in the registry |
| Latency budget | ECG and EEG window at or below 500 ms median, 2 s 99th percentile on the target host; radiation under 100 ms (design values, measure first) |
| Payload cap | design limit about 1 MB per request; larger streams use windowed batches |
| CORS | explicit allow-list of the frontend origins; no wildcard in production; restrict methods and headers |
| Authentication | a token shared with the cloud layer (for example a signed header from the Cloudflare Worker); no unauthenticated prediction endpoint |
| Rate limiting | enforced at the edge (Cloudflare) and a lower bound in the service |
| Privacy | participant IDs pseudonymous; raw waveforms never logged; logs keep IDs, timings, model ID, and result class only |
| Documentation | auto-generated OpenAPI, plus a short integration page with example payloads for the frontend and cloud teams |
| Testing | schema tests, golden-file tests on a fixed small recording per modality, load test against the latency budget |
| Versioning | URL or header API version; breaking changes need a new version |
| Failure behavior | on model error return a typed error and no prediction; the frontend shows "no estimate", never a stale value without its timestamp |

### 3.3 Integration with the rest of the system

- **Frontend ("Frankend"):** consumes JSON; needs the display rules for R and DTR from [days-to-return.md](days-to-return.md) and the claim language in the integration document ("flags a window as suspect for review", not "detects arrhythmia").
- **Cloud services:** Workers handle auth, rate limiting, and storage writes; the model service is stateless so it can scale and restart safely. Storage of results follows the retention rules in the integration document.
- **Human review:** hard-rule overrides in [risk-score.md](risk-score.md) escalate through the frontend to a named reviewer; the API never triggers an action.

## Build order

| Step | Deliverable | Exit criterion |
|---|---|---|
| 1 | Ingestion and manifest for MIT-BIH, TUSZ or CHB-MIT, and radiation telemetry | reproducible loaders; split files versioned |
| 2 | Preprocessing as tested functions | unit tests on synthetic and real fixtures; retention rates reported |
| 3 | Baseline models, registered | baseline numbers within tolerance of published values |
| 4 | FastAPI skeleton with `/health`, `/models`, schemas, CORS | frontend can call a mock |
| 5 | `/predict/ecg`, `/predict/eeg`, `/predict/radiation-risk` on baseline models | latency budget met; contract tests pass |
| 6 | `/predict/risk` using the planning formulas | worked examples in [worked-examples.md](worked-examples.md) reproduced exactly |
| 7 | Stronger models, external validation, analog or NASA data | reported with intervals, including failures |

## Open items

- Which NASA Human Research Program datasets are obtainable and on what timeline (request not yet made).
- Where the service runs (host, CPU or GPU) fixes the latency budget; the numbers above are placeholders.
- Whether the frontend sends windows or the service pulls from storage.
- Authentication scheme shared with the cloud layer.
