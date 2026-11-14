# Behavioral Health

This section covers behavioral health monitoring in isolated, confined, and extreme (ICE) environments such as spaceflight and analogs. It is split into a hub (this file) and four detail documents:

| Document | Contents |
|---|---|
| [ecg-hrv.md](ecg-hrv.md) | ECG and heart rate variability (HRV) for stress, mood, anxiety, and sleep strain |
| [eeg.md](eeg.md) | EEG for mood, fatigue, vigilance, and workload |
| [multimodal-ecg-eeg.md](multimodal-ecg-eeg.md) | Combining ECG and EEG: what the evidence supports and a proposed study design |
| [instruments-and-governance.md](instruments-and-governance.md) | Questionnaires, cognitive and vigilance tests, sleep trackers, regulation, ethics, and safety |

## 1. Summary

Behavioral health here means mood, anxiety, stress, sleep, cognitive performance, and team functioning. Four conclusions come out of the verified literature:

1. **Physiological markers are real but weak and non-specific.** Resting HRV is lower in depression, anxiety disorders, and PTSD, with small to moderate group effects (Hedges' g about -0.2 to -0.6). It is also lowered by sleep loss. EEG markers of depression are weaker still. Neither is a diagnostic test for an individual.
2. **Published machine-learning accuracies are inflated.** Records from the same person end up in both training and test sets in many studies. Subject-wise evaluation is the minimum standard, and results that hold up under it are far more modest.
3. **Spaceflight and isolation change the physiology themselves.** HF-HRV falls in orbit and over a long Antarctic winter, with no mood outcome needed to explain it. Any model needs an in-mission personal baseline and a time-in-mission term. An Earth-derived population cutoff will misfire.
4. **Sleep, activity, and cognition are the best-supported channels.** Sleep deficiency is the most replicated ISS finding, and a validated, sleep-loss-sensitive cognition battery exists.

The recommended EVAde output is a **personal-baseline trend with an uncertainty interval and a flag for human review**, not a diagnosis and not a crisis predictor.

## 2. Evidence from spaceflight and analogs

Sources below were verified against Crossref, PubMed, Europe PMC, or the agency page; the status of each is stated.

- **NASA Human Research Program, Behavioral Medicine (BMed) evidence report.** Dev, S. I., Whiting, S. E., Nelson, G. A., Dotson, V., Mulavara, A., Schorn, J. M., & Williams, T. J. *Risk of Adverse Cognitive or Behavioral Changes and Psychiatric Disorders Leading to In-Mission Health and Performance and Long-Term Health Effects (Evidence Report).* NASA Technical Reports Server 20250001454, dated 10 Apr 2025 (record verified; document body not read in full). For LEO missions up to 6 months, adverse behavioral health events are relatively infrequent: subclinical mood and anxiety disturbances occurred and no behavioral emergencies have been reported. Likelihood is expected to rise with duration, distance, and hazard exposure.
- **Dev, S. I., et al. (2024).** Cognitive performance in ISS astronauts on 6-month low earth orbit missions. *Frontiers in Physiology*, 15, 1451269. DOI 10.3389/fphys.2024.1451269. Verified. N = 25. No systematic decline. Early-flight slowing on processing speed, visual working memory, and sustained attention; 11.8% of flight and post-flight scores were at or below 1.5 SD under baseline. The authors list small n, imperfect practice-effect correction, and no testing in acute transition phases as limits.
- **Barger, L. K., et al. (2014).** Prevalence of sleep deficiency and use of hypnotic drugs in astronauts before, during, and after spaceflight. *The Lancet Neurology*, 13(9), 904-912. DOI 10.1016/S1474-4422(14)70122-X. Verified: 64 astronauts on 80 shuttle missions and 21 on 13 ISS missions. ISS in-flight sleep averaged 6.09 h (SD 0.67) versus 6.95 h post-flight; 12 of 16 ISS crew reported sleep-promoting drug use.
- **Flynn-Evans, E. E., et al. (2016).** Circadian misalignment affects sleep and medication use before and during spaceflight. *npj Microgravity*, 2, 15019. DOI 10.1038/npjmgrav.2015.19. Verified: 21 astronauts, 3,248 ISS days; temperature minimum fell outside the sleep period in 19% of in-flight sleep episodes; ISS sleep was 6.4 h when aligned versus 5.4 h when misaligned.
- **Otsuka, K., et al. (2021).** Astronauts well-being and possibly anti-aging improved during long-duration spaceflight. *Scientific Reports*, 11, 14907. DOI 10.1038/s41598-021-94478-w. Verified from the abstract. N = 10; sleep quality metrics improved in a subset while duration rose in 3 and fell in 7. The "anti-aging" interpretation is speculative.
- **Basner, M., et al. (2013).** Mars 520-d mission simulation reveals protracted crew hypokinesis and alterations of sleep duration and timing. *PNAS*, 110(7), 2635-2640. DOI 10.1073/pnas.1212646110. Verified: six crew; activity fell and sleep and rest time rose; light exposure fell; most problems began in the first 180 days.
- **Basner, M., et al. (2014).** Psychological and behavioral changes during confinement in a 520-day simulated interplanetary mission to Mars. *PLoS ONE*, 9(3), e93298. DOI 10.1371/journal.pone.0093298. Verified at abstract level: six healthy men, 550 m3 chamber. One member developed persistent sleep-onset insomnia with chronic partial sleep deprivation and lapses of alertness; two showed no disturbance.
- **Roalf, D. R., et al. (2025).** Transient gray matter decline during Antarctic isolation: roles of sleep, exercise, and cognition. *npj Microgravity*, 11, 39. DOI 10.1038/s41526-025-00497-6. Verified from the abstract. 25 crew over 12 months at Concordia; gray and white matter volume fell and mostly recovered by 5 months; better sleep correlated with less loss.
- **Cantisani, A., et al. (2026).** Social interactions in isolated, confined, and extreme environments: a study of Antarctic winter teams using wearable sensors. *PNAS*, 123(22), e2533420123. DOI 10.1073/pnas.2533420123. Verified (Crossref and Europe PMC abstract). N = 12 over 10 months: loneliness and conflict rose, cohesion and individual performance fell. Close-range interaction was linked to more conflict and paranoid thoughts and lower performance, so "more contact" is not automatically supportive.
- **Bell, S. T., Brown, S. G., & Mitchell, T. (2019).** What we know about team dynamics for long-distance space missions: a systematic review of analog research. *Frontiers in Psychology*, 10, 811. DOI 10.3389/fpsyg.2019.00811. Verified. 72 sources, 253 effect sizes; the authors found the data unsuitable for conventional meta-analysis. By 40% of the mission or 90 days all teams reported at least one conflict; the "third-quarter phenomenon" was not consistently found.
- **Rahill, K., Mulavara, A., George, K., Gore, B., & Whitmire, A. (2025).** Increasing fidelity in lunar and martian analogs for behavioral health and performance research. *Frontiers in Space Technologies*, 6, 1505823. DOI 10.3389/frspt.2025.1505823. Verified. Narrative review: no analog reproduces all spaceflight stressors; the authors call for standard measures and shared data.
- **Barbour, B. N., et al. (2025).** Biopsychosocial health considerations for astronauts in long-duration spaceflight: a narrative review. *Wilderness & Environmental Medicine*, 36(1_suppl), 123S-137S. DOI 10.1177/10806032241289106. Verified from the abstract; narrative, no quantitative synthesis.
- **Supolkina, N., et al. (2021).** External communication of autonomous crews under simulation of interplanetary missions. *Frontiers in Physiology*, 12, 751170. DOI 10.3389/fphys.2021.751170. Verified. SIRIUS crews showed "detachment" in crew-ground communication, replicating Mars-500.
- **Patel, Z. S., et al. (2020).** Red risks for a journey to the red planet: the highest priority human health risks for a mission to Mars. *npj Microgravity*, 6, 33. DOI 10.1038/s41526-020-00124-6. Verified. Names radiation, SANS, behavioral health and performance, and food and nutrition as the four "red" risks.
- **Palinkas, L. A., & Suedfeld, P. (2008).** Psychological effects of polar expeditions. *The Lancet*, 371, 153-163. Historical review of polar expeditions. The commonly quoted "40-60%" figure comes from press coverage and was **not verified in the paper**, so no numeric claim is made here.

**Gaps (searched, not found):** peer-reviewed Mars-500, NEK, and LunAres results after 2019; peer-reviewed HI-SEAS sociometric-badge results; any spaceflight use of PHQ-9 or GAD-7. Per NASA slides (not read in full), NASA's standard measures suite uses instruments such as BDI-II and POMS rather than PHQ-9.

## 3. What each modality can and cannot support

| Channel | Supported use | Main weakness | Detail |
|---|---|---|---|
| Sleep and activity (wrist) | sleep duration and timing trends | staging is coarse; device-specific bias | [instruments-and-governance.md](instruments-and-governance.md) |
| ECG-derived HRV (rest or sleep) | autonomic strain, transdiagnostic | small effects; many confounders; drifts with time in isolation | [ecg-hrv.md](ecg-hrv.md) |
| PPG-derived pulse rate variability | same, with limits | agrees with ECG only at rest or controlled conditions | [ecg-hrv.md](ecg-hrv.md) |
| EEG (scalp, ear, dry) | sleep architecture; drowsiness; exploratory mood | weak mood biomarkers; leakage; hardware burden | [eeg.md](eeg.md) |
| ECG plus EEG | untested for crews | no verified depression-specific fusion study | [multimodal-ecg-eeg.md](multimodal-ecg-eeg.md) |
| Cognition and vigilance tests | alertness and sleep-loss effects | practice effects; user participation | [instruments-and-governance.md](instruments-and-governance.md) |
| Passive phone or badge data | routine, contact patterns | privacy; poor cross-cohort transfer | [instruments-and-governance.md](instruments-and-governance.md) |

## 4. Reference standard and labels

There is **no ICE-validated cutoff** for any questionnaire. The review found no spaceflight study using PHQ-9 or GAD-7. A defensible plan therefore uses several labels and states what each one means:

- Self-report symptom scales for trend, not diagnosis (PHQ-9 and GAD-7, with the caveats in the instruments document).
- An objective alertness test (3-minute PVT-B or the adaptive PVT-BA) and the NASA cognition battery with practice-effect correction.
- Sleep measured by wrist device, with documented device error against polysomnography.
- A clinician interview in a pilot subset, if resources allow.
- For crews: team-level conflict and communication measures.

## 5. Proposed processing stages

1. **Acquisition and quality.** ECG or PPG, wrist actigraphy, optional EEG; each with a quality index; radiation-event flag from the radiation module.
2. **Covariate layer.** Sleep, medication class and start date, caffeine, posture, breathing rate, activity, and (where available) habitat CO2 and temperature.
3. **Feature layer.** Standardized-window HRV (RMSSD headline), sleep duration, timing, and regularity, PVT lapses and 1/RT, optional EEG sleep and drowsiness features.
4. **Per-person baseline.** Built from the first days of the mission or pilot; includes a time-in-mission term.
5. **Trend and flag layer.** Deviation from own trajectory with uncertainty; flags go to a named human reviewer.
6. **Reporting layer.** Plain-language trend summaries with confidence, never a diagnostic label.

## 6. Implementation steps

1. Write the intended-use statement first: wellness trend monitor or screening tool. This decides the regulatory path (see [instruments-and-governance.md](instruments-and-governance.md)).
2. Fix a daily standardized measurement protocol: same posture, same time, 5-minute resting or sleep-period window.
3. Start with sleep and activity, and validate the wrist device against polysomnography in a subset.
4. Add ECG-derived HRV with RMSSD as the headline index, and log covariates (see [ecg-hrv.md](ecg-hrv.md)).
5. Add a 3-minute PVT-B on a fixed schedule with practice-effect handling; baseline each person before the mission.
6. Add self-report sparingly (PHQ-2 as a gate, PHQ-9 and GAD-7 periodically) as labels and as a user-facing check-in.
7. Treat EEG as optional. Use ear or dry EEG first for sleep, and run a pre-registered pilot before any mood claim (see [eeg.md](eeg.md)).
8. Pre-register the fusion study (see [multimodal-ecg-eeg.md](multimodal-ecg-eeg.md)) before collecting data.
9. Evaluate subject-wise and time-forward; always include a simple baseline model and a leakage checklist.
10. Include a time-in-mission term and an in-mission baseline; exclude the first weeks after return from mood analysis.
11. Add team-level measures in analog trials (proximity, conflict reports, communication length), with self-report as ground truth.
12. Write the safety-escalation protocol and the data-governance plan before collecting any human data.
13. Validate in more than one analog type and report which analog each result comes from.
14. Preregister thresholds and a fairness gate; archive datasets, splits, and metric code.

## 7. Evaluation and robustness

- **Primary metrics:** per-person detection of a clinically meaningful change (sensitivity, false flags per person-month), agreement with PHQ-9/GAD-7 change, and agreement of sleep estimates with polysomnography.
- **Splits:** subject-wise and time-forward; cross-cohort tests (for example GLOBEM years).
- **Stress tests:** missing-data (30% of days missing), wear compliance, device change, and artifacts from travel or illness.
- **Calibration and baselines:** report calibration of any score; compare against logistic regression and against a trivial per-person mean.
- **Subgroups:** age, sex, medication class, and (for PPG) skin tone.

## 8. Limitations and open questions

- Mars-500 had six men and no radiation or microgravity; Concordia and ISS cohorts are small. Generalization to Mars-class missions is unproven.
- Whether within-person HRV or EEG change tracks mood change in healthy confined crews is not established. The only longitudinal HRV-depression design found was clinical and two weeks long.
- Which flag threshold balances missed events against alert fatigue is an empirical question for a pilot.
- How radiation exposure (CNS effects) should enter a behavioral model is open; no validated quantitative model was found (see [space-radiation-detection](../space-radiation-detection/README.md)).
