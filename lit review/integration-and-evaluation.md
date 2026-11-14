# Integration and Evaluation

This document ties the six topic areas together: how the streams combine, what a shared data schema needs, how to evaluate claims honestly, and what the limits are. It contains no code.

## 1. Data flow

1. **Acquisition.** Radiation detector, PPG/rPPG, ECG, EEG, actigraphy, and (optionally) behavioral inputs such as PVT results and questionnaires.
2. **Quality layer.** Each stream produces its own quality index (SQI, lead-off, artifact rate, radiation-event flag). Data below quality threshold is marked, not deleted.
3. **Feature layer.** Fixed-length windows: HR/HRV, respiration, SpO2, EEG band powers and seizure scores, sleep and activity, stress features, bone-trend estimates (slow).
4. **Per-person model layer.** Baselines, trends, and uncertainty per participant.
5. **Flag layer.** Rule- and threshold-based flags with named owners (who reviews, how fast), never autonomous clinical decisions.

## 2. Shared schema requirements

| Field | Purpose |
|---|---|
| participant_id (pseudonymous) | per-person baselines and subject-wise splits |
| timestamp (UTC) and source clock ID | cross-stream alignment and drift tracking |
| modality and device ID | sensor-specific bias tracking |
| window_start, window_length | common windowing across ECG/EEG/PPG |
| value, units, uncertainty | every output carries an uncertainty |
| quality_index | the value is only valid above a threshold |
| radiation_flag | whether an SEU or particle event occurred nearby |
| context tags | activity, sleep state, medication, mission day |
| model_version, data_version | reproducibility |

Open design choices: how to handle clock drift between devices, whether the radiation flag suppresses or only annotates, and how to store context tags without over-collecting.

## 3. Evaluation protocol (applies to every module)

1. **Subject-wise splits only.** No participant appears in both train and test. Where possible, add a time-forward split (train earlier, test later).
2. **External validation.** At least one dataset from a different site or device than the training data. GLOBEM-style cross-year tests are an example.
3. **Report uncertainty.** Subject-level bootstrap confidence intervals, not only point estimates.
4. **Report operating points.** For flags: sensitivity at a fixed false-alarm rate (for example per 24 h for seizures, per person-month for behavioral flags), not accuracy alone.
5. **Calibration.** Check that predicted risk matches observed frequency.
6. **Subgroup analysis.** Age, sex, skin tone (for PPG/rPPG), medication use, and motion level.
7. **Negative controls.** Shuffled labels, artifact-only segments.
8. **Pre-registration of claims.** Write the claim, metric, and threshold before looking at the test set.
9. **Structure the evidence plan with V3.** Verification (sensor and pipeline), analytical validation (algorithm against a reference), and clinical validation (target users), per Goldsack et al. (2020), *npj Digit Med*, 3, 55, DOI 10.1038/s41746-020-0260-4.
10. **Use a leakage checklist.** Subject-wise and time-forward splits and a model info sheet (Saeb et al. 2017, DOI 10.1093/gigascience/gix019; Kapoor & Narayanan 2023, DOI 10.1016/j.patter.2023.100804). For EEG, segment-wise splits strongly overestimate performance on new subjects (Brookshire et al. 2024, DOI 10.3389/fnins.2024.1373515).
11. **Use domain-standard metrics.** Event-based scoring for seizures (SzCORE; Dan et al. 2025, DOI 10.1111/epi.18113); Bland-Altman bias and limits of agreement for HR, RR, and HRV (Chan et al. 2022, DOI 10.1111/anae.15834); least significant change for bone (ISCD 2019 positions).
12. **Fairness gate.** Pre-register a maximum allowed between-group error difference, with skin tone measured objectively for PPG and rPPG.
13. **Include a trivial baseline.** Always report a logistic regression and a per-person mean baseline next to any complex model.

## 4. Cross-modality risks

| Risk | Where it appears | Mitigation |
|---|---|---|
| Subject leakage | all ML evaluations, especially EEG and behavioral | enforce subject-wise splits in the data loader contract |
| Label noise | TUSZ annotations, self-report | report inter-rater or test-retest reliability where available |
| Motion artifact | PPG, rPPG, ECG, EEG | SQI gating; report data retention rate |
| Population mismatch | MIT-BIH, CHB-MIT (pediatric), MrOS (older men), WESAD (15 subjects) | state the training population; do not claim generalization |
| Confounding by medication | HRV, sleep, EEG | record medication as covariate |
| Environmental artifact | radiation events, lighting for rPPG | radiation flag, light logging |
| Over-claiming | all | scope statements; flag-for-review language |
| Physiology drifts with time in isolation or orbit | HRV (HF power falls in orbit and over an Antarctic winter), sleep, bone | in-mission personal baseline and a time-in-mission term; exclude early post-mission weeks |
| Device and firmware change | PPG, HRV, sleep trackers | log device model and firmware; never pool across brands without a bridging study |
| Cross-site shift | ECG, EEG, rPPG | external validation; expect a performance drop (about 10% in the PhysioNet 2020 challenge) |
| Skin-tone bias | PPG SpO2, rPPG | tone-stratified validation with objective tone measurement; report tail error as well as mean bias |
| Label ceiling | behavioral health | no ICE-validated questionnaire cutoff exists; use several labels and report what each means |

## 5. Claim language

| Allowed | Not allowed |
|---|---|
| "Flags a window as suspect for review." | "Detects arrhythmia." |
| "Estimates heart rate with MAE X bpm against a chest strap." | "Medical-grade heart rate." |
| "Reports a sleep-duration trend." | "Diagnoses insomnia." |
| "Estimates a bone-loss trend relative to personal baseline." | "Diagnoses osteoporosis." |

Regulatory context is summarized in [behavioral-health/instruments-and-governance.md](behavioral-health/instruments-and-governance.md): the FDA general wellness guidance (reissued 6 Jan 2026) treats stress, sleep, and mental-acuity claims as lower risk and disease claims as device territory; the FDA pulse oximeter draft guidance (7 Jan 2025) was not found to be finalized; and EU AI Act high-risk dates were amended to 2 Dec 2027 (Annex III) and 2 Aug 2028 (Annex I). This is context, not legal advice. Get qualified advice before any use beyond research.

## 5a. Serving layer

Models reach the frontend and cloud services through a stateless FastAPI service with endpoints `/predict/ecg`, `/predict/eeg`, `/predict/radiation-risk`, and `/predict/risk`. Every response carries model ID (`EVA-[n]`), data version, quality index, and uncertainty; low-quality windows return no value. Only models with a registry entry and a subject-wise evaluation are served. Details, CORS and authentication choices, and the latency budget are in [planning/implementation-pipeline.md](planning/implementation-pipeline.md).

## 6. Privacy and governance

- Treat all physiological and behavioral data as sensitive and identifiable, especially in small cohorts.
- Keep raw data out of version control; keep a data manifest with checksums and license notes (see [data-collection-sources](data-collection-sources/README.md)).
- Define retention, access, and deletion rules before collection starts.
- For human-subjects data collection, institutional review (IRB or equivalent) is required; this review does not address it in detail.

## 7. Suggested phasing

| Phase | Goal | Exit criterion |
|---|---|---|
| 0 | Reproduce published baselines on public data (MIT-BIH QRS, Siena or CHB-MIT seizure with SeizeIT2 as external test, WESAD stress) | numbers within a stated tolerance of published results |
| 1 | Single-stream pilots with reference sensors | MAE and agreement targets met on a pilot cohort |
| 2 | Quality layer and shared schema | SQI and radiation flag in every record; schema documented |
| 3 | Per-person baselines and trend flags | false-alarm rate and sensitivity reported with intervals |
| 4 | Fusion and external validation | cross-site result reported, including failures |

## 8. Open questions across the review

- Does a body-worn radiation detector add value beyond a cabin dosimeter for data quality flagging?
- How well do wearable-derived HRV and sleep trends track behavioral change in confined crews?
- Which portable proxy, if any, tracks bone loss closely enough to be useful?
- How should medication, light, and exercise context enter each model?
- What is the acceptable false-alarm rate for each flag type, and who decides?
