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

## 5. Claim language

| Allowed | Not allowed |
|---|---|
| "Flags a window as suspect for review." | "Detects arrhythmia." |
| "Estimates heart rate with MAE X bpm against a chest strap." | "Medical-grade heart rate." |
| "Reports a sleep-duration trend." | "Diagnoses insomnia." |
| "Estimates a bone-loss trend relative to personal baseline." | "Diagnoses osteoporosis." |

Regulatory classification of health claims (for example, wellness versus medical device) was not researched in this review. Get qualified advice before any use beyond research.

## 6. Privacy and governance

- Treat all physiological and behavioral data as sensitive and identifiable, especially in small cohorts.
- Keep raw data out of version control; keep a data manifest with checksums and license notes (see [data-collection-sources](data-collection-sources/README.md)).
- Define retention, access, and deletion rules before collection starts.
- For human-subjects data collection, institutional review (IRB or equivalent) is required; this review does not address it in detail.

## 7. Suggested phasing

| Phase | Goal | Exit criterion |
|---|---|---|
| 0 | Reproduce published baselines on public data (MIT-BIH QRS, TUSZ or CHB-MIT seizure, WESAD stress) | numbers within a stated tolerance of published results |
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
