# EVAde-backend-models

Backend models for **EVAde**, a space health monitoring platform that helps astronauts track and manage physical and mental well-being on spacecraft, stations, rovers, and planetary bases (formerly `blindsight-backend-models`). This repository is the model and backend side: literature review, theory-only planning, a model registry, and later the FastAPI service that delivers model results to the frontend.

**Status: beta.** Preprocessing, three CPU beta models (registered as `draft`), the risk engine, and a FastAPI service exist and are tested; everything is a hackathon beta, not a validated system. All outputs are designed as flags for human review, not diagnoses, and nothing here is clinical guidance.

## What the backend will do

| Area | Plan |
|---|---|
| Signals | ECG and EEG processing; heart rate, blood pressure, temperature, SpO2, glucose; sleep and mood inputs; Bluetooth wearables |
| Hazards | radiation exposure, bone density loss |
| Per-person baseline | every metric is compared with the person's own baseline; every API call carries a unique pseudonymous `user_id` |
| Risk score | an automatic score R in (0, 1) from five domains (noisy-OR aggregation) |
| maximum days to return | a recommended maximum time to return, shown as "<5", 5 to 100, or ">100" days |
| API | FastAPI endpoints such as `/predict/ecg`, `/predict/eeg`, `/predict/radiation-risk`, `/predict/risk`, `/compare`, and `/ingest/*` |

## Where to look

| Path | Contents |
|---|---|
| [lit review/](lit%20review/README.md) | Verified literature review by topic (radiation, vital signs, ECG, EEG, bone, behavioral health), model architectures, integration and evaluation protocol, and data sources. Each source is labeled VERIFIED, PARTIAL, or UNVERIFIED. |
| [lit review/lit-review.pdf](lit%20review/lit-review.pdf) | The whole review and planning set as one research-style PDF (rebuild after edits). |
| [lit review/planning/](lit%20review/planning/README.md) | Theory-only design: [risk score](lit%20review/planning/risk-score.md), [maximum days to return](lit%20review/planning/max-days-to-return.md), [worked examples](lit%20review/planning/worked-examples.md), [implementation pipeline](lit%20review/planning/implementation-pipeline.md), [data acquisition plan](lit%20review/planning/data-acquisition-plan.md), [baseline comparison and integration](lit%20review/planning/baseline-comparison.md), [Bluetooth integration](lit%20review/planning/bluetooth-device-integration.md), and [calibration and validation](lit%20review/planning/calibration-and-validation.md). |
| [evade/preprocessing/](evade/preprocessing/README.md) | Loaders and preprocessing for the open datasets (filtering, 250 Hz, windows, quality, labels, subject-wise folds) |
| [evade/training/](evade/training/README.md) | Beta training (seizure-suspect windows, sleep staging, PTB-XL superclass) with subject-wise results and baselines |
| `evade/risk/`, `evade/baseline/` | Risk score v2 and maximum days to return (reproduces the worked examples); per-user baseline store and comparison |
| [evade/api/](evade/api/README.md) | FastAPI service: users, baselines, compare, ECG/EEG/radiation/risk prediction, self-report, PHQ-9/GAD-7, devices and ingest |
| `tests/` | Unit and API contract tests (`python -m pytest -q tests`) |
| [MODEL_REGISTRY.md](MODEL_REGISTRY.md) | Every retained model architecture and trained model gets a unique ID `EVA-[n]`, numbered in order of creation. |
| `external/Blindsight-Data-Nasa` | The team's earlier prototype, added as a git submodule (pinned commit, no license, so referenced and not copied). Run `git submodule update --init` to fetch it. |
| `data/` | Local data folder. Git-ignored except `data/README.md` and `data/MANIFEST.csv`. Raw datasets are never committed. |
| `.claude/skills/evade-project/` | A project skill that gives Claude Code the project background and working rules. |

## Key design choices

- **Hackathon scope:** two days, so only datasets that download without credentials or approval are used (MIT-BIH, PTB-XL, BIDMC, Siena, CHB-MIT, Sleep-EDFx, EEG During Mental Arithmetic, NHANES DXA, and similar). TUSZ is not used. Behavioral and bone inputs may be synthetic and must be labeled so.
- **Data handling:** raw data stays out of git. The manifest in `data/` records source, license, and checksum for each dataset. Datasets are treated as untrusted files.
- **Evaluation:** subject-wise splits, a trivial baseline beside every model, uncertainty reported, and a registry entry (`EVA-[n]`) before any model is served.
- **Compute:** training happens on a shared GPU server under strict limits (one SSH session, at most half the compute, tests before heavy runs); those rules live in a Claude Code skill, not in this repository, and no credentials are stored here.

## Limitations

- Sources were checked at the record and abstract level; full texts were mostly not read, so numbers must be re-checked before formal use.
- No spaceflight ECG or EEG at scale is publicly available. Models would be trained on terrestrial data, and spaceflight performance is unknown until analog or NASA data are tested.
- Risk score and max-days-to-return parameters are design assumptions pending expert elicitation and calibration.
- Known open gaps are listed in each section of the review.

## Working with the repository

- Branches: `main` holds the merged plan (PR #1); `integrate-blindsight-data-nasa` holds the newer code (preprocessing, training, risk engine, API) until it is merged.
- After editing the review or planning documents, rebuild `lit review/lit-review.pdf` and commit both.
- Do not commit data, model weights from untrusted sources, or credentials.
