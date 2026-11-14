# Fitness Assessment With Established Protocol Metrics

**Status: plan only.** "Fitness" here means **fitness for purpose of each model and of the risk score**: does it meet the metric that the relevant standard, guideline, or challenge uses for that task. It replaces a single general loss value (cross-entropy, MSE, log-loss) as the acceptance test. A loss is still allowed as the *training objective*; it is never the *acceptance criterion*. This is separate from crew physical fitness, which is not planned here (see section 8).

**Evidence labels.** Standards and metrics marked *cited* have a source already checked in this review (see the DOI list in [integration-and-evaluation.md](../integration-and-evaluation.md)). Items marked *recalled* come from my knowledge of the standard and were **not re-read**; confirm the current edition and numbers before any formal claim. Thresholds marked *design* are project choices, not standard values.

## 1. Why not one loss value

- A loss hides the operating point: a seizure detector can have a low loss and still miss events or raise many false alarms per day.
- Losses are not comparable across datasets, windowing, or class balance, so they cannot be checked against published results.
- The people who use these outputs (flight surgeons, regulators, reviewers) judge them with domain metrics: event sensitivity and false alarms per 24 h, limits of agreement, least significant change, calibration.
- Standards define pass criteria on reference data. Meeting them is the evidence; a lower loss is not.

## 2. Fitness gates per module

Each gate lists the protocol metric, the reference, and the pass criterion. A model is "fit" only if **all** its gates pass on a held-out, subject-wise (and, where possible, external) set, with subject-level bootstrap confidence intervals.

### ECG and heart rate variability

| Task | Protocol and metrics | Pass criterion |
|---|---|---|
| QRS or beat detection | ANSI/AAMI EC57 style scoring against annotated data (MIT-BIH): beat sensitivity and positive predictive value, 150 ms matching window (*recalled*) | sensitivity and PPV at or above the baseline Pan-Tompkins result reproduced in phase 0, with CI; target at least 0.99 on clean records (*design*) |
| Rhythm classification | per-class sensitivity, specificity, and F1 (macro), plus the PhysioNet/CinC challenge scoring for the multi-label task (*cited*: challenges 2020 and 2021 used as external validation; exact metric definitions *recalled*) | macro F1 not below the reproduced baseline; cross-site drop reported (about 10% was seen in 2020, *cited*) |
| HR and HRV agreement | Bland-Altman bias and 95% limits of agreement against a reference ECG (Chan et al. 2022, *cited*); HRV measured per Task Force 1996 conventions (*recalled*) | bias within 1 bpm and limits within 5 bpm for HR (*design*); HRV limits reported, no pass number until a reference cohort exists |

### EEG

| Task | Protocol and metrics | Pass criterion |
|---|---|---|
| Seizure detection | SzCORE event-based scoring: event sensitivity, false alarms per 24 h, event F1 (Dan et al. 2025, *cited*) | sensitivity and false alarms per 24 h at a pre-registered operating point, for example at least 0.75 and at most 1 per 24 h (*design*); report on a wearable external set (SeizeIT2) separately |
| Sleep staging | AASM scoring rules for reference labels (*recalled*); Cohen's kappa and per-stage F1 | kappa not below the reproduced baseline on Sleep-EDFx; reported per stage, not accuracy alone |
| Drowsiness or workload | agreement with a behavioral reference (Psychomotor Vigilance Task lapses, a sleepiness scale); correlation with CI and event-level sensitivity at a fixed false-alarm rate | shadow mode only until the reference agreement is shown |

### PPG, rPPG, and vitals

| Signal | Protocol and metrics | Pass criterion |
|---|---|---|
| Heart rate from PPG or video | mean absolute error and the share of windows within 5 bpm; Bland-Altman; per skin-tone group | MAE and group differences under the pre-registered fairness gate (*design*) |
| Respiration rate | MAE and limits of agreement (BIDMC reference) | limits reported; threshold *design* |
| SpO2 | accuracy as Arms against arterial blood gas over a saturation range, per ISO 80601-2-61 and the FDA pulse oximeter guidance (*recalled*: Arms at or below 3%; the draft guidance of January 2025 was not found finalized, *cited* in the review) | Arms at or below 3% (*recalled*) and a skin-tone-stratified result; consumer or wrist devices stay in shadow mode for the flag |
| Blood pressure | ISO 81060-2 style criteria: mean error at most 5 mmHg with standard deviation at most 8 mmHg (*recalled*) | pass or the reading is labeled "uncalibrated spot value" |
| Glucose | ISO 15197:2013 style criteria for meters: 95% of readings within 15 mg/dL (or 15%) of reference (*recalled*) | pass or not used for flags |
| Temperature | clinical thermometer accuracy standard for the device class (*recalled*, value not checked) | device-specific |

### Radiation

| Task | Protocol and metrics | Pass criterion |
|---|---|---|
| Dose-rate estimate | relative error and bias against a reference detector series (for example ISS or MSL RAD data) with the calibration version recorded | relative error within a stated band, for example 20% (*design*) |
| Solar particle event detection | lead time and miss rate against event start times; false alarm rate; the NOAA solar radiation storm scale threshold (S1 at 10 pfu in the greater-than-10 MeV channel, *recalled*) | miss rate and lead time reported; no event missed in the replay set (*design*) |
| Electronics fault flag | rate of detected upsets against injected faults | not assessable until a fault-injection set exists |

### Bone

| Task | Protocol and metrics | Pass criterion |
|---|---|---|
| Trend estimate | least significant change from the device's precision error (ISCD 2019 positions, *cited*); Bland-Altman against DXA | a change is reported only if it exceeds the LSC (*cited rule*) |
| Fracture-risk use of any tool | discrimination (AUC) and calibration as in FRAX validations (*recalled*) | not used for flags unless validated |

### Behavioral health

| Task | Protocol and metrics | Pass criterion |
|---|---|---|
| Questionnaire scoring | PHQ-9 and GAD-7 scored with published cut-offs; sensitivity and specificity against a structured interview where available (Kroenke 2001, Spitzer 2006 *cited*) | scoring identical to the published method (unit tests) |
| Sensor-derived trend flags | sensitivity at a fixed false-alarm rate per person-month; AUROC with subject-level CI; time-forward and cross-year validation (GLOBEM style, *cited*) | operating point pre-registered; no claim of diagnosis |
| Reliability | test-retest intraclass correlation for repeated measures; agreement of PVT lapses with the standard definition (*cited*: Basner and Dinges 2011) | ICC reported; no pass number |

## 3. Fitness gates for the risk score and maximum days to return

| What | Protocol and metrics | Pass criterion (design unless noted) |
|---|---|---|
| Discrimination | AUROC and AUPRC for the 14-day intervention label; time-dependent AUC | at least 0.80 AUROC with the lower CI bound above 0.70; always beside the per-person-mean baseline |
| Calibration | calibration slope and intercept, calibration plot, expected calibration error, Brier score | slope between 0.8 and 1.2; intercept near 0; reliability plot by band |
| Operating points | alerts per person-month by band; sensitivity for override-class events | the targets in [calibration-and-validation.md](calibration-and-validation.md): at most 1 per person-month at "high" or above in a healthy cohort; sensitivity at least 0.90 for override-class events |
| Clinical usefulness | decision curve analysis (net benefit across threshold probabilities, Vickers and Elkin) (*recalled*) | net benefit at the "high" threshold above treat-all and treat-none |
| Band agreement | weighted Cohen's kappa between model bands and an expert panel on rated cases | at least 0.60 (*design*) |
| maximum days to return | MAE in days against the expert panel's recommended return time; share within plus or minus 20%; asymmetric error check, because a too-long MDTR is worse than a too-short one | MAE and within-20% share reported; the share of too-long errors above 20% capped (*design*) |
| Domain-score consistency | each domain's component scores checked against that domain's protocol gate in section 2 | a domain whose gate fails is shown in shadow mode and excluded from the flag test |
| Reporting | TRIPOD+AI for the prediction model report and PROBAST+AI for risk of bias (*recalled*) | checklists completed with each model card |

Replace "validated" with "meets its gate": a module is promoted from shadow mode to the flag test only when its section 2 gates pass.

## 4. Fitness gates for the baseline comparison

| Check | Metric | Pass criterion (design) |
|---|---|---|
| Baseline stability | within-person coefficient of variation and intraclass correlation of the baseline center across days | reported; a baseline with high variability is marked low confidence |
| False flags in stable people | flags per person-month in a stable synthetic or open cohort | within the per-domain budget in [risk-score.md](risk-score.md) |
| Sensitivity to induced change | detection of a known shift (simulator scenarios, tagged exercise or sleep restriction) at the green/yellow/red bands | detects shifts of 2 sigma or more in the bad direction at least 90% of the time (*design*) |
| Time to detection | windows from the start of a shift to the first yellow | reported |

## 5. Replacing the loss-based rules in the plan

| Current text | Replacement |
|---|---|
| Calibration refit accepted when held-out log-loss improves by at least 5% | refit accepted only if (a) every gate in sections 2-4 still passes, (b) calibration slope stays in range, and (c) sensitivity at the per-domain false-alarm budget does not fall; log-loss may be the fitting objective only |
| Model comparison "headline metric" in the registry | the headline metric is the module's protocol metric from section 2 with its CI (see [MODEL_REGISTRY.md](../../MODEL_REGISTRY.md)) |
| Training logs with validation loss | allowed for debugging and early stopping; reports and model cards give protocol metrics |

## 6. Reporting rules

1. State the protocol, edition, and reference data for every metric.
2. Give subject-level bootstrap confidence intervals, not only point values.
3. Show the baseline result (per-person mean, logistic regression, the reproduced published baseline) next to every model.
4. Report subgroup results (age, sex, skin tone, motion level, device).
5. Pre-register the claim, metric, and threshold before opening the test set.
6. Report failures and gate misses; a model that misses a gate is labeled "does not meet" in the registry.
7. Do not use accuracy alone for imbalanced tasks.

## 7. Hackathon scope

Do not attempt full protocol compliance in two days. Compute only the cheap protocol metrics and label the rest as not assessed:

| Module | Report in the demo |
|---|---|
| QRS | sensitivity and PPV on a subset of MIT-BIH |
| Rhythm | per-class sensitivity and F1 on a held-out PTB-XL fold (note the dataset's recommended folds) |
| Seizure | event sensitivity and false alarms per 24 h on held-out subjects from Siena or CHB-MIT |
| Sleep | kappa on held-out Sleep-EDFx subjects |
| Risk score | band agreement on the worked examples; calibration cannot be assessed on simulated data, so say so |

State that the SpO2, blood pressure, glucose, radiation-event, and bone gates are **not assessed**, and that behavioral and bone inputs are synthetic.

## 8. Not covered and open items

- **Crew physical fitness** (aerobic capacity, strength, and similar countermeasure-exercise testing) is a different meaning of "fitness" and is not planned here. If wanted, it needs its own research against established spaceflight and exercise testing protocols.
- The *recalled* standards (EC57, ISO 80601-2-61, ISO 81060-2, ISO 15197, AASM, TRIPOD+AI, PROBAST+AI, decision curve analysis) must be checked for current editions and exact pass numbers; this review did not open them.
- Thresholds marked *design* need expert elicitation (see [calibration-and-validation.md](calibration-and-validation.md)).
- There is no ICE-validated questionnaire cutoff and no spaceflight reference set, so several gates can only be assessed on terrestrial data.
