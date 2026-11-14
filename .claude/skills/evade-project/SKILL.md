---
name: evade-project
description: Background on EVAde, a space health monitoring platform for astronauts (hackathon project). Use when working in this repo on models, the FastAPI backend, planning docs, data sources, baseline comparison, Bluetooth device ingestion, or when asked about the project, its team roles, tech stack, or claim and safety rules.
---

# EVAde project context

**EVAde** is a space health monitoring platform that helps astronauts track and manage physical and mental well-being in deep-space settings: spacecraft, space stations, rovers, and planetary bases. Astronauts cannot be expected to self-assess complex medical conditions, so the system combines automated physiological telemetry, environmental hazard tracking, AI diagnostics, and a guided, gamified interface.

This repository (`EVAde-backend-models`, formerly `blindsight-backend-models`) is the **model and backend side**: literature review, planning, model registry, and (later) the FastAPI service. Some files still say "Blindsight" or "Blingsight" (the older name and the team prototype); the project name is EVAde.

## What the platform does

- **Monitoring:** heart rate, blood pressure, blood sugar, body temperature, sleep hours, mood, EEG, ECG; plus hazards: radiation exposure, bone density loss, immune changes, inadequate nutrition, cardiovascular events.
- **AI:** models on telemetry and NASA Human Research Program data judge whether an anomaly is critical, estimate days until intervention or return, and predict low-radiation windows for spacewalks.
- **Voice and mental health:** a Twilio voice agent for check-ins; journal entries analyzed by an LLM (Grok) to estimate mental-health state and write summaries for flight surgeons and therapists.
- **Frontend:** tutorials, a gamified daily health to-do list, multi-view charts, color-coded status, and an admin panel for therapists and doctors.
- **Devices:** Bluetooth (BLE) wearables feed the system; see `lit review/planning/bluetooth-device-integration.md`.

## Tech stack

Cloudflare (hosting, Workers, database/edge functions), Auth0 moving to Cloudflare authentication, GoDaddy domain, Python and FastAPI on Workers and Vultr compute, Postgres for longitudinal data, Grok for text analysis, custom ML for EEG/ECG and health metrics, Twilio voice, possible Gemini API and Solana integrations.

## Team roles (by role, not name, because the repo is public)

| Role | Work |
|---|---|
| Data and Workers lead | Cloudflare Workers, team workspace, training datasets, Python data scripts, README |
| Frontend and auth | frontend tasks, Cloudflare DB edge functions, Auth0/Cloudflare auth, voice AI help |
| Journaling and LLM | journaling pipeline with Grok, mental-health summaries for therapists |
| Infra and pipelines | Workers, database, Python health-data pipelines, domain, voice AI help, event coordination |
| **Models and backend (this repo)** | EEG and ECG signal processing, models for bone, radiation and other metrics, FastAPI service delivering results to the frontend |

## Repo map (read these before changing things)

- `lit review/` : verified literature review by topic (radiation, vitals, ECG, EEG, bone, behavioral, architectures, integration, data sources) and `lit-review.pdf` (rebuild after edits).
- `lit review/planning/` : theory-only design. Key files: `risk-score.md` (noisy-OR, version 2), `max-days-to-return.md`, `worked-examples.md`, `implementation-pipeline.md`, `data-acquisition-plan.md`, `baseline-comparison.md`, `bluetooth-device-integration.md`, `calibration-and-validation.md`.
- `MODEL_REGISTRY.md` : every retained architecture or trained model gets `EVA-[n]` in creation order; never reused.
- `external/Blindsight-Data-Nasa` : team prototype as a git submodule (no license; reference, do not copy).

## Design facts to keep consistent

- Risk score R in (0, 1): per-domain p = c x 0.75 x max(0, (s - 0.10)/0.90), L = sum of -ln(1 - p), R = 0.005 + 0.99 x (1 - e^-L). A domain with s at or above 0.30 floors displayed R at 0.60. Hard-rule overrides floor R at 0.80.
- maximum days to return: MDTR = 150 x (1 - R)^1.5 with trend, radiation, bone, acute, and extreme-domain caps (any eligible domain at s 0.80 or more gives under 10 days). Display "<5", 5 to 100, or ">100".
- Everything is compared with the person's **own baseline** (robust z, median and MAD); every API call carries a unique pseudonymous `user_id`.
- Endpoints planned: `/predict/ecg`, `/predict/eeg`, `/predict/radiation-risk`, `/predict/risk`, `/compare`, `/ingest/*`, `/users`, `/models`, `/health`.

## Rules when working here

1. **Claim language:** outputs are flags for human review, not diagnoses; no "medical-grade" or "detects arrhythmia".
2. **Evidence labels:** VERIFIED, PARTIAL, UNVERIFIED on every source; do not add numbers from press releases; mark unverified items.
3. **Hackathon scope (2 days):** use only instant-download data (no credentialing, no approval waits); TUSZ is not used; behavioral and bone inputs may be synthetic and must be labelled so.
4. **Privacy:** no names, emails, or raw physiological data in the repo; pseudonymous IDs; no raw waveforms in logs.
5. **Models:** subject-wise splits, a trivial baseline beside every model, and an `EVA-[n]` registry row before a model is served. Do not load untrusted joblib or pickle files.
6. **Code status:** the repo is plan-first; do not add code unless asked.
7. After editing planning or lit-review docs, rebuild the PDF with the builder script and commit.
