# Behavioral Health

## Summary

Behavioral health here means mood, anxiety, stress, sleep, cognitive performance, and interpersonal functioning. In isolated and confined environments (spaceflight, polar stations, Mars simulations) these are among the most consequential risks, and they interact with the physiological streams in the other sections: sleep loss changes HRV, stress changes electrodermal activity (EDA) and heart rate, and chronic partial sleep deprivation degrades vigilance.

Three points shape the design of any Blindsight behavioral module:

1. **Individual variation is large.** In the Mars-500 520-day simulation (six healthy men), one crew member developed persistent sleep-onset insomnia with chronic partial sleep deprivation and lapses in alertness, while two others showed no behavioral disturbance or distress (Basner et al., 2014). A single population threshold is the wrong model; per-person baselines and trends are the right one.
2. **Behavior shows up first in sleep and activity.** The same simulation found crew activity fell and sleep and rest time rose over the mission, with light exposure decreasing, and most problems emerged in the first 180 days (Basner et al., 2013). Actigraphy and sleep timing are cheap, low-burden signals.
3. **Physiological markers are real but modest.** Depression is associated with lower heart rate variability: a 2010 meta-analysis of 18 articles (673 depressed participants, 407 controls) found lower HRV in depression, with effect sizes around Hedges' g of -0.3 and a negative correlation with severity (r = -0.354) (Kemp et al., 2010). That is a population-level association, not a diagnostic test for an individual.

Behavioral output from Blindsight should therefore be a **monitored trend with a flag for human review**, never a diagnosis.

## Evidence from spaceflight and analogs

- **Isolation and confinement are an active NASA risk.** NASA's Human Research Program tracks "Behavioral Medicine" (BMed) as an active risk covering cognitive and behavioral decrements from isolation and confinement. Its program page says selection, training, and onboard countermeasures on the ISS have mainly mitigated the risk so far; the evidence report (most recent version found dated February 2025, NTRS document 20250001454) is the primary source for details and was not opened in full here.
- **Astronaut sleep is chronically short.** An observational study of 85 astronauts (64 on 80 shuttle missions, 21 on the ISS; over 4,000 nights on Earth and over 4,200 nights in space) found about 6 hours of sleep per night in flight against 8.5 hours scheduled, under 6.5 hours during pre-flight training, and sleep-aid use by roughly three quarters of crew (Barger et al., 2014). Figures are from press coverage of the paper.
- **Polar expeditions** report high rates of depression, sleep disruption, irritability, and interpersonal conflict, alongside positive ("salutogenic") effects; interpersonal conflict was named the greatest source of tension (Palinkas & Suedfeld, 2008). A commonly quoted 40-60% figure comes from press coverage and should be checked in the paper.
- **Space radiation and the CNS.** Cucinotta and colleagues describe central nervous system risks from space radiation, including possible premature aging and late neurological disease. Quantitative CNS risk models were described as not yet developed, so this is a motivation for monitoring, not a model to implement. The volume and pages of the 2014 *Life Sciences in Space Research* review were not confirmed.

## Sensing modalities and what they can support

| Signal | What it can indicate | Evidence base | Main weakness |
|---|---|---|---|
| HRV (from ECG or PPG) | autonomic state, stress, depression association | Kemp 2010 meta-analysis | modest effect sizes, large between-person variance |
| EDA and heart rate | acute stress | Healey & Picard 2005 (24 drives; over 97% accuracy separating three stress levels in the abstract) | driving task, small sample, needs subject-specific calibration |
| Actigraphy | activity level, sleep duration and timing | Basner 2013 | wrist wear compliance, sleep staging is coarse |
| Psychomotor Vigilance Test (PVT) | alertness and sleep loss | Basner & Dinges 2011 (lapses and 1/RT metrics most sensitive) | requires user participation, learning effects |
| Self-report (PHQ-9, GAD-7) | symptom severity | Kroenke 2001; Spitzer 2006 | burden, reporting bias, item copyright terms |
| Phone/passive behavior (digital phenotyping) | routine, mobility, sociability | Onnela & Rauch 2016; GLOBEM | privacy, generalization across years and sites |
| EEG | affective and cognitive state, sleep | limited here; see the EEG section | hardware burden, artifacts |

## Sources

- **Basner, M., et al. (2014).** Psychological and behavioral changes during confinement in a 520-day simulated interplanetary mission to Mars. *PLoS ONE*, 9(3), e93298. DOI 10.1371/journal.pone.0093298. Verified: citation and abstract-level findings (crew of six men, 550 m3 chamber, 520 days, run by the Institute for Biomedical Problems, Moscow). Full text not reviewed.
- **Basner, M., et al. (2013).** Mars 520-d mission simulation reveals protracted crew hypokinesis and alterations of sleep duration and timing. *PNAS*, 110(7), 2635-2640. DOI 10.1073/pnas.1212646110. Verified. Actigraphy-based; full text is in PMC (PMC3574912).
- **Barger, L. K., et al. (2014).** Prevalence of sleep deficiency and use of hypnotic drugs in astronauts before, during, and after spaceflight: an observational study. *The Lancet Neurology*, 13(9), 904-912. DOI 10.1016/S1474-4422(14)70122-X. Verified: citation. Numbers above are from press coverage.
- **Palinkas, L. A., & Suedfeld, P. (2008).** Psychological effects of polar expeditions. *The Lancet*, 371, 153-163. Verified: existence and volume; end page 163 is from one reference listing. DOI not confirmed.
- **Kemp, A. H., et al. (2010).** Impact of depression and antidepressant treatment on heart rate variability: a review and meta-analysis. *Biological Psychiatry*, 67(11), 1067-1074. DOI 10.1016/j.biopsych.2009.12.012. Verified from the abstract. The antidepressant findings were not checked.
- **Healey, J. A., & Picard, R. W. (2005).** Detecting stress during real-world driving tasks using physiological sensors. *IEEE Transactions on Intelligent Transportation Systems*, 6(2), 156-166. DOI 10.1109/TITS.2005.848368. Verified; accuracy figure is from the abstract.
- **Basner, M., & Dinges, D. F. (2011).** Maximizing sensitivity of the psychomotor vigilance test (PVT) to sleep loss. *Sleep*, 34(5), 581-591. DOI 10.1093/sleep/34.5.581. Verified: 74 healthy adults; response-speed and lapse metrics gave the largest effect sizes.
- **Kroenke, K., Spitzer, R. L., & Williams, J. B. W. (2001).** The PHQ-9: validity of a brief depression severity measure. *Journal of General Internal Medicine*, 16(9), 606-613. Verified. Cutoff of 10 or higher gave about 88% sensitivity and 88% specificity for major depression against a structured interview in 580 patients; severity bands at 5, 10, 15, 20.
- **Spitzer, R. L., Kroenke, K., Williams, J. B. W., & Lowe, B. (2006).** A brief measure for assessing generalized anxiety disorder: the GAD-7. *Archives of Internal Medicine*, 166(10), 1092-1097. DOI 10.1001/archinte.166.10.1092. Verified. Cut point of 10 gave about 89% sensitivity and 82% specificity. Copyright and reuse terms are unclear in the sources; confirm before embedding the item text.
- **Onnela, J.-P., & Rauch, S. L. (2016).** Harnessing smartphone-based digital phenotyping to enhance behavioral and mental health. *Neuropsychopharmacology*, 41(7), 1691-1696. DOI 10.1038/npp.2016.7. Verified. Introduces the active (surveys) versus passive (sensor) data distinction.
- **Xu, X., et al. (2022).** GLOBEM dataset: multi-year datasets for longitudinal human behavior modeling generalization. *NeurIPS 2022 Datasets and Benchmarks Track*. Verified: arXiv 2211.02733, over 700 user-years from 497 users over four years, benchmark of 18 algorithms on depression detection. Author list beyond the lead author was not checked.
- **NASA Human Research Program, Behavioral Medicine (BMed) evidence report** (NTRS 20250001454) and the program's risk page. Not read in full.

## Implementation steps

1. **Fix scope in writing.** Targets are: sleep sufficiency, stress load, mood and anxiety symptom trend, and alertness. Non-targets: diagnosing any disorder, predicting suicide, or any covert monitoring.
2. **Build the baseline-first design.** Collect at least 2 weeks of baseline per person before any flag is allowed. Report deviation from the person's own baseline (z-score or percentile of own history), plus uncertainty.
3. **Start with sleep and activity.** Derive sleep duration, onset and midpoint, and daily activity from wrist actigraphy or PPG-plus-accelerometer. This is the highest-value, lowest-burden signal given the Mars-500 findings. Validate against a sleep diary first, polysomnography only in a subset.
4. **Add stress features.** Compute HRV (reusing the ECG/PPG pipelines), EDA tonic and phasic components, and heart rate. Validate on WESAD with subject-wise splits, then test on your own pilot data. Treat the Healey & Picard accuracy as an upper bound for a constrained task.
5. **Add a short alertness test.** Implement a brief PVT variant using lapses and mean 1/RT as primary outputs (Basner & Dinges, 2011). Run it on a fixed schedule and compare against sleep history.
6. **Add self-report sparingly.** Use PHQ-9 and GAD-7 weekly at most, as ground-truth labels and as a user-facing check-in, not as the only signal. Confirm licensing for the exact instrument text before deployment.
7. **Passive behavior only with consent.** If phone or app usage signals are added, follow GLOBEM-style feature extraction and test cross-year generalization explicitly, since models degrade across cohorts.
8. **Fuse with the physiological streams.** Use the shared window and label schema from the other sections. Fuse at the feature level first; a simple per-person trend model before any deep model.
9. **Define escalation and safety.** Write the response for high scores or concerning item responses before collecting any data: who is notified, within what time, and how the participant is told. Route to a human clinician, not an automated message.
10. **Evaluate against labels you trust.** Use PHQ-9/GAD-7 and clinician interviews as labels in a pilot, with the subject-wise protocol in [integration-and-evaluation.md](../integration-and-evaluation.md).

## Evaluation and robustness

- **Primary metrics:** per-person detection of a clinically meaningful change (sensitivity, false flags per person-month), agreement with PHQ-9/GAD-7 change, and agreement of sleep estimates with diary or polysomnography.
- **Robustness checks:** subject-wise and time-forward splits (train on earlier weeks, test on later); cross-cohort tests (GLOBEM years); missing-data stress tests (what happens with 30% of days missing); sensitivity to wear compliance; performance across sex, age, and skin tone where PPG is used.
- **Calibration:** report calibration of any risk score; a flag rate that does not match actual change rates will erode trust.
- **Negative controls:** run the pipeline on shuffled labels and on days with known artifacts (travel, illness) to see what it picks up.

## Ethics, privacy, and safety

- **Consent and control.** Participants must know what is collected, who sees it, and be able to opt out of each stream. In small crews, data can identify individuals even when "anonymized"; treat it as identifiable.
- **No diagnosis.** State in the UI and the schema that outputs are monitoring trends for review. Mental-health claims carry regulatory implications that were not researched in this review; get qualified advice before any claim beyond research use.
- **Data minimization.** Store features, not raw audio, video, or text, unless there is a specific justified need.
- **Fairness.** Evaluate by subgroup. HRV and PPG signals vary with age, sex, medication (including antidepressants and sleep aids), and skin tone.
- **Medication confound.** The Kemp meta-analysis concerns antidepressant effects on HRV, and astronauts commonly use sleep aids (Barger 2014); medication use must be recorded as a covariate.

## Limitations and open questions

- Mars-500 had six male participants and no radiation or microgravity. Generalizing the findings is limited.
- Several headline numbers here come from abstracts or press coverage; full texts were not reviewed.
- Whether wearable-derived stress or sleep trends predict clinically meaningful behavioral change in crews is not established by these sources.
- Which flag threshold balances missed events against alert fatigue is an empirical question for a pilot.
- How radiation exposure (CNS effects) should enter a behavioral model is open; no validated quantitative model was found.
