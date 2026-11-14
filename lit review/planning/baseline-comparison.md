# Baseline-vs-Current Comparison Plan, Per User, All Modalities

**Status: plan only, no code.** This defines how every modality is compared with the same person's own baseline, how a unique user ID travels through the API, how user input enters, and how this repository integrates the team's existing prototype, `Blindsight-Data-Nasa` (added as a git submodule at `external/Blindsight-Data-Nasa`, pinned to commit `5012314`; it has no license file, so it is referenced, not copied). Numbers are design assumptions unless a source is named.

## 1. What the existing prototype already does

| Piece | What it does | Limits that matter here |
|---|---|---|
| `models/analyzer.py` | Learns a baseline from the first 30 windows (about 15 s), then scores each window as sigma from baseline (green below 1, yellow 1-2, red above 2); clips z at 5; floors the spread at 15% of the baseline mean; smooths with an EMA (0.3) | Baseline lives in memory per WebSocket connection, so it is lost on disconnect and there is no user identity; 15 s is far too short for HRV; mean and standard deviation are not robust to outliers; baseline can absorb an already stressed state |
| `models/features.py` | EEG relative band power and ratios (theta, alpha, beta); ECG heart rate, RMSSD, RR irregularity from a simple peak finder | peak detector uses a fixed height threshold; no quality index; no log transform |
| `stream/simulator.py` | simulated 4-channel EEG plus ECG at 256 Hz with scenarios | simulated only |
| `models/mission.py`, `docs/mission_trends.json` | 180-day mission trend generator (radiation, bone, sleep, HRV, BP, EEG alpha) with sourced values | labels each series backed, derived, or illustrative; reuse as the synthetic input for bone, sleep, and radiation in the hackathon |
| `models/actions.py` | action protocols per alert | to be reviewed against sources by the story owner |
| `api/main.py` | WebSocket stream plus `/api/mission`, `/api/log` | no CORS, no auth, global action log shared by everyone, no user ID |
| `models/state_model.joblib`, `models/train.py` | random forest trained on simulated data | its 100 percent accuracy is not a result; loading a joblib file executes pickle code, so only load a file you built yourself; do not serve it as a registered model until it is retrained on real data and given an `EVA-[n]` ID |

## 2. Design: baseline model per user

### 2.1 Identity

- Every call carries `user_id`: a client-supplied or server-issued **UUID4**, pseudonymous, never an email or name. It is the key for baselines, history, and the action log.
- `POST /users` creates one and returns it with a secret token; later calls send `user_id` in the body and the token in a header. For the hackathon a shared API token plus `user_id` in the body is acceptable; state that a real system needs per-user tokens.
- A call with an unknown `user_id` returns 404, not an implicit new baseline.
- Per-user data is isolated: a request can only read or write its own `user_id`.

### 2.2 Baseline definition

| Item | Rule |
|---|---|
| Robust statistics | center = median; spread = 1.4826 x MAD; floor the spread at 15% of |center| (the prototype's floor) so tiny baseline noise does not create false alarms |
| Skewed features | log-transform HRV (RMSSD), band power, and ratios before computing center and spread |
| z-score | z = (x - center) / spread, clipped to plus or minus 5 |
| Direction | each feature has a bad direction (for example RMSSD down, heart rate up, theta/beta up); only deviation in the bad direction counts toward flags, but the signed z is always reported |
| Minimum data to lock | see 2.3; below the minimum the baseline is `provisional` and comparisons are returned with `low_confidence` |
| Frozen during flags | a window is excluded from baseline updating when the person is yellow or red, quality is poor, or the user marks an exception (illness, caffeine, alcohol, medication change) |
| Slow update | after locking, update with an exponentially weighted median over about 14 days (half-life about 7 days) so the baseline follows adaptation without chasing a decline; the update is paused while any domain flag is active |
| Phase baselines | keep a separate baseline for each mission phase (pre-mission, early flight, steady state, post-event). EEG alpha is known to drop after launch, which the prototype handles by re-learning; here the user or system starts a new phase explicitly and the old one is kept for comparison |
| Time of day | when at least 3 days of data exist, keep four time-of-day bins for HR, HRV, temperature, and sleepiness-related EEG; compare a window to its own bin |
| Versioning | every baseline has a `baseline_id`, creation time, number of windows, the window range, and the feature-pipeline version |

### 2.3 Minimum data and window sizes (design)

| Modality | Feature window | Baseline locks after | Hackathon demo setting |
|---|---|---|---|
| ECG / HRV | 5 min windows for HRV (the standard short-term length); 30 s for heart rate | 24 valid resting windows over at least 3 days | 15-30 s windows, 30 windows (as in the prototype), marked "demo baseline" |
| PPG / rPPG | 30 s | 24 valid windows | same as ECG |
| EEG band power and ratios | 4-10 s windows | 60 valid windows over at least 2 sessions | 30 windows |
| Respiration, SpO2, temperature, blood pressure | per reading or 5 min median | 20 readings across at least 3 days | 10 readings |
| Sleep and activity | nightly or daily summaries | 7 nights | 3 synthetic nights |
| Behavioral questionnaires (PHQ-9, GAD-7) and PVT | each administration | 2 administrations at least a week apart (PVT: 3 sessions) | the person's first entry; flagged "single point" |
| Radiation dose | per-day dose | no personal baseline; compare with the mission or location reference rate and the person's own cumulative total | reference rate from the prototype's mission file |
| Bone | each scan | the first scan (pre-mission) | user-entered scan value and date |

The prototype's 15 s baseline stays available as an explicit demo mode, labelled in every response.

## 3. Comparison output per metric

For each metric and each call, return:

| Field | Meaning |
|---|---|
| `user_id`, `baseline_id`, `metric` | identity and which baseline was used |
| `value`, `unit`, `quality_index` | the current value; if quality is below threshold, value is null and `low_quality` is true |
| `baseline_center`, `baseline_spread`, `n_baseline` | what it was compared with |
| `z` | signed robust z-score |
| `delta_abs`, `delta_pct` | difference and percent change from the baseline center |
| `level` | green below 1, yellow 1-2, red above 2 (the prototype's bands), only in the bad direction |
| `component_score` | 0 to 1 score for the risk model: 0 below z = 1; interpolated 0.15 at z = 1, 0.40 at z = 2, 0.70 at z = 3, 1.0 at z = 4 and above (design assumption, to be calibrated) |
| `confidence` | `ok`, `provisional` (baseline not locked), `low_quality`, or `demo` |
| `context_used` | the user-input tags applied (see 4) |

Component scores feed the domain scores in [risk-score.md](risk-score.md). Where [risk-score.md](risk-score.md) already defines anchors on absolute values (PHQ-9, GAD-7, SpO2 floor, dose limits), keep those absolute anchors and add the personal-baseline deviation as a separate component; do not count one feature twice (design principle 2).

## 4. User input

| Input | Where entered | Effect |
|---|---|---|
| Self-reported state (sleep quality 1-5, stress 1-5, fatigue 1-5, mood 1-5) | `POST /input/self-report` | stored as a time series with its own baseline; used as context and a behavioral component; not a diagnosis |
| Context tags (caffeine, exercise, illness, medication change, EVA, high-workload task, poor-quality device wear) | `POST /input/context` with start and end times | windows inside the span are excluded from baseline updates and returned with the tag; they can lower the level by one step only if the tag is physiologically relevant to that metric (design rule: exercise for heart rate and HRV, caffeine for heart rate and EEG beta) and never lower a red |
| Baseline confirmation | `POST /baseline/{modality}/confirm` | the user confirms "this was a typical period"; required before a baseline can be called `confirmed` rather than `provisional` |
| Baseline reset or new phase | `POST /baseline/{modality}/reset` with a reason | starts a new baseline; the old one is archived, never deleted |
| Questionnaires (PHQ-9, GAD-7) and PVT | `POST /input/instrument` | scored server-side with the anchors in [risk-score.md](risk-score.md); PHQ-9 item 9 above 0 triggers override O1 and escalation to a human |
| Manual measurements (blood pressure, temperature, SpO2 from a cuff or finger device, bone scan) | `POST /input/measurement` | value, unit, device, time; compared with the person's own baseline |

Input validation: Pydantic ranges per field; timestamps in UTC; reject future times; keep an audit record of who changed what, with the original value.

## 5. API surface (additions to [implementation-pipeline.md](implementation-pipeline.md))

| Endpoint | Method | Purpose |
|---|---|---|
| `/users` | POST | create a pseudonymous user; returns `user_id` |
| `/users/{user_id}/baseline` | GET | current baselines per modality, status, and counts |
| `/baseline/{modality}` | POST | submit windows for baseline building (`user_id` in body); modality is one of `ecg`, `ppg`, `eeg`, `vitals`, `sleep`, `behavioral`, `radiation`, `bone` |
| `/compare` | POST | `user_id` plus one or more current windows or measurements; returns the comparison objects of section 3 and the component scores |
| `/compare/summary` | GET | per-user, per-modality current level and trend over the last 24 h and 7 days |
| `/predict/ecg`, `/predict/eeg`, `/predict/radiation-risk`, `/predict/risk` | POST | as before, now accept `user_id` and return the comparison fields next to the model output |
| `/ws?user_id=...` | WebSocket | the prototype's live stream, keyed by user so the baseline persists across reconnects |
| `/input/*` | POST | section 4 |
| `/devices/bind`, `/ingest/batch`, `/ingest/stream` | POST, POST, WebSocket | Bluetooth device binding and data ingest; see [bluetooth-device-integration.md](bluetooth-device-integration.md) |

Rules: a request without a valid `user_id` is rejected; the action log and every stored record are keyed by `user_id`; responses carry `baseline_id` and the pipeline version; nothing returns another user's data.

## 6. Storage (design)

| Table | Key | Content |
|---|---|---|
| users | user_id | creation time, token hash, consent flag |
| baselines | baseline_id, user_id, modality | center, spread, n, status, phase, window range, pipeline version, created, archived |
| observations | user_id, modality, metric, time | value, unit, quality, device, context tags |
| inputs | user_id, type, time | self-report, context, instrument, measurement, with audit fields |
| comparisons | user_id, time | the section 3 objects, for trend views |
| action_log | user_id, time | the prototype's action log entries |

Hackathon: SQLite or the team's cloud database; store summary statistics and recent windows, not raw waveforms. Retain nothing beyond what the demo needs and say so; a real deployment needs the retention rules in [integration-and-evaluation.md](../integration-and-evaluation.md).

## 7. Integration plan with `Blindsight-Data-Nasa`

Work is on branch `integrate-blindsight-data-nasa`. The prototype is a submodule, so it stays unmodified and attributable to its authors; the integration code (when written) lives in this repository and imports from it, or the team forks it and the submodule points to the fork.

| Step | Action | Result |
|---|---|---|
| 1 | Pin the submodule at `5012314`; record the commit in this file and the data register | reproducible reference |
| 2 | Wrap `analyzer.py`'s baseline idea in a per-user baseline service using section 2 (robust statistics, persistence, phases) | baselines survive disconnects and are per `user_id` |
| 3 | Reuse `features.py` for the EEG band-power features; replace the ECG peak detector with a validated QRS detector and add a quality index and 5-min HRV windows | features match [implementation-pipeline.md](implementation-pipeline.md) |
| 4 | Add `user_id` to the WebSocket and the REST routes; replace the global `ACTION_LOG` with a per-user store; add CORS allow-list, token auth, and payload limits | meets the service requirements |
| 5 | Use `stream/simulator.py` scenarios as the test source for `/compare`; add a `ReplaySource` for open datasets (WESAD, Siena, CHB-MIT) | tests with known deviations |
| 6 | Use `models/mission.py` to generate synthetic sleep, radiation, bone, HRV, and blood-pressure series for the hackathon, labelled synthetic | covers modalities with no instant data |
| 7 | Map `component_score` to the domain scores and call the risk and DTR formulas | `/predict/risk` works end to end |
| 8 | Keep `models/actions.py` as the action text, with the story owner adding sources; show actions only for flags, with the claim language from [integration-and-evaluation.md](../integration-and-evaluation.md) | consistent wording |
| 9 | Do not serve `state_model.joblib`; keep `train.py` as a starting point; register any retrained model in [MODEL_REGISTRY.md](../../MODEL_REGISTRY.md) | no unvalidated model in the API |

Known conflicts to settle at integration:
- The prototype's cardiac rule uses absolute heart rate limits (130 and 40 bpm) and irregularity; keep them as absolute backstops next to the personal baseline, not replaced by it.
- The prototype's alert-on-escalation logic (alert only when the level rises, re-arm on green) should stay, and move into the per-user state.
- The prototype works at 256 Hz with four frontal and temporal EEG channels; the planned pipeline resamples to 250 Hz and maps to a fixed 10-20 subset. The comparison uses features, so only the feature extraction needs the mapping.
- Naming: the prototype says "Blingsight"; this repository used "Blindsight" earlier and now uses "EVAde". Use EVAde in the demo and the API.

## 8. Testing and evaluation

| Test | Pass condition |
|---|---|
| Identity | a request with another user's ID, an unknown ID, or no ID is rejected; two users sending the same values get independent baselines |
| Persistence | disconnect and reconnect keeps the baseline |
| Known shift | simulator stress, fatigue, and arrhythmia scenarios produce the expected metric deviating in the bad direction after the baseline locks, and no flag for a calm scenario over a long run |
| Robustness | one artifact window does not move the baseline center by more than 5% (median and MAD) |
| False alarms | on a calm replay of 24 h of synthetic or open data, at most the per-domain alert budget in [risk-score.md](risk-score.md) |
| Contamination | starting a baseline during an induced stress scenario is flagged or refused when quality or context says so |
| Phase handling | after a new phase starts, comparisons use the new baseline and the old one stays retrievable |
| Context | a tagged window is excluded from baseline updates; a red is never lowered |
| Provisional | before the baseline locks, every response says `provisional` or `demo` |

Limitations to state in the demo: the baseline comparison is only as good as the baseline; people with an already abnormal baseline will not be flagged for that; single-session baselines are weak; sensors and simulations are not astronauts; flags are for human review.

## 9. Open items

- Who owns the fork of `Blindsight-Data-Nasa` and where the integration code lives.
- Whether per-user storage is SQLite or the team's cloud database.
- Whether self-report and questionnaire entry happen in the existing frontend or a new form.
- Confirm the prototype's `docs/mission_trends.json` sources before using any of its numbers as more than illustrative.
