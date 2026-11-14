# ECG and HRV for Behavioral Health

Status labels: **VERIFIED** = bibliographic record and abstract checked; **PARTIAL** = record checked, abstract or some fields not read. Full texts were not reviewed. Numbers are from abstracts.

## 1. Summary

HRV, especially the parasympathetic indices RMSSD and HF power, is lower on average in depression, anxiety disorders, PTSD, and under sleep loss. The effects are small to moderate at group level, are shared across diagnoses, and are strongly affected by sleep, breathing, medication, posture, and measurement method. In spaceflight and long isolation, HF-HRV also drifts downward without any mood explanation. The practical consequence is a design built on **within-person trends against an in-mission baseline**, with sleep and breathing as covariates, rather than population cutoffs.

## 2. Effect sizes by condition

| Condition | Source | Size | Main finding |
|---|---|---|---|
| Major depression (unmedicated adults) | Koch 2019 | 21 studies, 2,250 patients, 1,982 controls | RMSSD g = -0.462; HF -0.318; SDNN -0.266; LF/HF +0.195 |
| Depression (older, medication analysis) | Kemp 2010 | 18 articles, 673 patients, 407 controls | HF g = -0.293; severity vs HRV r = -0.354; tricyclics lowered HRV, SSRIs mostly not |
| Psychiatric disorders overall | Alvares 2016 | 140 case-control studies | HRV reduced in all groups (g = -0.583); quality moderated effect sizes |
| Anxiety disorders | Cheng 2022 | 99 studies, 4,897 patients, 5,559 controls | resting HRV lower (g = -0.390); reactivity not different |
| PTSD | Schneider 2020 | meta-analysis | lower RMSSD and HF, higher resting heart rate; effect sizes small to medium |
| Late-life depression | Brown 2018 | 11 studies, 550 patients | g = -0.334; only LF separated groups |
| Panic disorder | Zhang 2020 | drug-naive patients | LF, HF not significant; SDNN lower (SMD -0.313) |
| Sleep deprivation | Zhang 2025 | 11 RCTs, 549 participants | RMSSD fell; LF and LF/HF rose |

Takeaways: reduced HRV is **transdiagnostic and non-specific**; RMSSD is the most sensitive resting index in major depression; findings for specific anxiety disorders are inconsistent; and sleep loss moves HRV in the same direction as psychological strain, so it must be tracked as a covariate.

## 3. Measurement standards (use these to design the protocol)

- **Quigley et al. (2024)** Publication guidelines for human heart rate and heart rate variability studies in psychophysiology, Part 1. *Psychophysiology*, 61(9), e14604. DOI 10.1111/psyp.14604. VERIFIED (title, journal, volume, issue; last authors partial). Committee report with best practices for ECG and PPG HR/HRV in laboratory and ambulatory settings and reporting checklists. Use as the reporting template.
- **Shaffer & Ginsberg (2017)** An overview of HRV metrics and norms. *Frontiers in Public Health*, 5, 258. DOI 10.3389/fpubh.2017.00258. PARTIAL. Metric glossary.
- **Laborde, Mosley & Thayer (2017)** HRV and cardiac vagal tone in psychophysiological research. *Frontiers in Psychology*, 8, 213. DOI 10.3389/fpsyg.2017.00213. PARTIAL. Three-phase design (baseline, reactivity, recovery).
- **Task Force of the ESC and NASPE (1996)** HRV: standards of measurement, physiological interpretation and clinical use. *Circulation*, 93(5), 1043-1065. DOI 10.1161/01.cir.93.5.1043. VERIFIED (older, still the foundational definition source).
- **Shaffer, Meehan & Zerr (2020)** A critical review of ultra-short-term HRV norms research. *Frontiers in Neuroscience*, 14, 594880. DOI 10.3389/fnins.2020.594880. VERIFIED. 28 studies of at least 20 participants; most did not test criterion validity adequately.
- **Kim, J. W., et al. (2021)** Is ultra-short-term HRV valid in non-static conditions? *Frontiers in Physiology*, 12, 596060. DOI 10.3389/fphys.2021.596060. VERIFIED. N = 30; during exercise and recovery at least 120 s was needed; under lenient criteria 10-60 s worked at rest.
- **Nunan, Sandercock & Brodie (2010)** Normal values for short-term HRV in healthy adults. *PACE*, 33(11), 1407-1417. DOI 10.1111/j.1540-8159.2010.02841.x. VERIFIED. 44 studies, 21,438 participants; very large inter-individual variation, which favors personal baselines.

## 4. Device validity

- **Dobbs et al. (2019)** Accuracy of acquiring HRV from portable devices. *Sports Medicine*, 49(3), 417-435. DOI 10.1007/s40279-019-01061-5. VERIFIED (core fields). 23 studies, 301 effects; ES 0.23 vs ECG, I2 = 78.6%; error higher in tilt and recovery positions. Search ended 2017.
- **Stone et al. (2021)** Accuracy of popular commercial technologies for resting HR and HRV. *Frontiers in Sports and Active Living*, 3, 585870. DOI 10.3389/fspor.2021.585870. VERIFIED. 5 young adults, 148 trials; RMSSD error ranged from 4.1% (a chest-strap app, CCC 0.98) to 112% (a camera app). Device and algorithm choice matter.
- **Schaffarczyk et al. (2022)** Validity of the Polar H10 for HRV at rest and during incremental exercise. *Sensors*, 22(17), 6536. DOI 10.3390/s22176536. VERIFIED. n = 25; high resting agreement with ECG (r = 0.95); significant bias during exercise.
- **Blalock et al. (2026)** Validity of the Polar H10 for HRV and cardiac autonomic reflex tests. *Autonomic Neuroscience*, 266, 103447. DOI 10.1016/j.autneu.2026.103447. VERIFIED. 43 healthy adults; CCC at least 0.99, MAPE under 1%.
- **Xu et al. (2026)** Accuracy of PPG-derived pulse rate variability compared with ECG-derived HRV: a systematic review and meta-analysis. *Sensors*, 26(16), 5192. DOI 10.3390/s26165192. VERIFIED (core fields). 43 studies qualitative, 10 quantitative; pooled absolute standardized error 0.188 for RMSSD and 0.134 for SDNN, from resting or controlled conditions only; the authors warn against generalizing to sleep, exercise, stress, or free-living use.
- **Mejia-Mejia et al. (2020)** HRV and pulse rate variability (PRV) for assessing autonomic responses. *Frontiers in Physiology*, 11, 779. DOI 10.3389/fphys.2020.00779. VERIFIED. 20 volunteers; HRV and PRV differ under physiological stress (cold exposure).
- **Georgiou et al. (2018)** Can wearable devices accurately measure HRV? *Folia Medica*, 60(1), 7-20. DOI 10.2478/folmed-2018-0012. VERIFIED (core fields); 18 studies, 16 ECG-based and only 2 PPG-based.

**Use:** chest-strap or patch ECG as the reference stream; PPG only as labeled pulse rate variability at rest or during sleep, quality-gated; never pool across devices.

## 5. Machine learning from ECG and HRV: what holds up

- **Schmidt et al. (2018)** Introducing WESAD. *Proc. ICMI*, 400-408. DOI 10.1145/3242969.3242985. VERIFIED. 15 subjects, lab stress; reported up to 93% binary (stress vs non-stress) and 80% three-class. The cross-validation scheme was not verified. Use only as a smoke test.
- **Vos et al. (2023)** Generalizable machine learning for stress monitoring from wearable devices: a systematic literature review. *Int J Med Inform*, 173, 105026. DOI 10.1016/j.ijmedinf.2023.105026. VERIFIED. 33 articles; most public datasets are under 24 hours of data; generalization and labeling are weak.
- **Coutts et al. (2020)** Deep learning with wearable-based HRV for prediction of mental and general health. *J Biomed Inform*, 112, 103610. DOI 10.1016/j.jbi.2020.103610. VERIFIED (core fields). 652 participants; up to 83% accuracy from 5-minute HRV. The abstract does not state whether splits were subject-wise, so the headline needs a subject-wise check.
- **Saeb et al. (2017)** The need to approximate the use-case in clinical machine learning. *GigaScience*, 6(5), gix019. DOI 10.1093/gigascience/gix019. VERIFIED. Record-wise cross-validation massively overestimates accuracy versus subject-wise; almost half of retrieved wearable studies used record-wise.
- **Kapoor & Narayanan (2023)** Leakage and the reproducibility crisis in machine-learning-based science. *Patterns*, 4(9), 100804. DOI 10.1016/j.patter.2023.100804. VERIFIED. Leakage in 17 fields affecting 294 papers; proposes model info sheets.
- **Gedam & Paul (2021)**, *IEEE Access*, 9, 84045-84066, DOI 10.1109/ACCESS.2021.3085502, and **Geng et al. (2023)**, *Comput Biol Med*, 162, 107060, DOI 10.1016/j.compbiomed.2023.107060: PARTIAL (records only; abstracts not read). Background reading.

## 6. Confounders to log

- **Breathing.** Ritz (2024), *Biological Psychology*, 185, 108728, DOI 10.1016/j.biopsycho.2023.108728 (VERIFIED): respiration pattern can change HF-HRV amplitude without any change in vagal tone; within-person interpretation is conclusive only relative to breathing.
- **Sleep loss.** Zhang 2025 (above).
- **Medication.** Tricyclics lowered HRV in Kemp 2010. De Oliveira et al. (2024), *J Affect Disord*, 355, 200-209, DOI 10.1016/j.jad.2024.03.071 (VERIFIED): paroxetine, low-quality evidence of small LF and HF increases. Log class and start date.
- **Caffeine.** Almeida et al. (2024), *Curr Cardiol Rev*, 20(3), 87-100, DOI 10.2174/011573403X289842240307114736 (VERIFIED): no effect on post-exercise vagal recovery (RMSSD SMD -0.03); says little about resting HRV in sleep-restricted people.
- **Posture, sex, fitness.** Dobbs 2019 reports error varying with posture and sex. **No verified 2019+ meta-analysis of posture or fitness effects, and none for benzodiazepines, was found**; these remain evidence gaps.
- **Environment.** Möller et al. (2026), UHAB-1, *Life Sci Space Res*, 52, 146-153, DOI 10.1016/j.lssr.2026.02.006 (VERIFIED): a single-occupant case study in which reduced HRV coincided with elevated ambient CO2. n = 1; treat CO2 as a plausible confounder to log, not an established effect.

## 7. HRV in spaceflight and analogs

- **Hughson et al. (2012)** *J Appl Physiol*, 112(5), 719-727, DOI 10.1152/japplphysiol.01196.2011 (VERIFIED, core fields). 6 male astronauts; HF R-R interval power fell in flight.
- **Xu et al. (2013)** Reduced HRV during sleep in long-duration spaceflight. *Am J Physiol Regul Integr Comp Physiol*, 305(2), R164-R170, DOI 10.1152/ajpregu.00423.2012 (VERIFIED, core fields). 7 astronauts; HRV diminished in flight and partly recovered after landing.
- **Vandeput et al. (2013)** *Med Sci Monit*, 19, 9-17, DOI 10.12659/MSM.883724 (VERIFIED, core fields). 8 astronauts; vagal modulation reduced 5 days after return, nearly recovered by 30 days.
- **Otsuka et al. (2016)** *Heliyon*, 2(12), e00211, DOI 10.1016/j.heliyon.2016.e00211 (VERIFIED, core fields). 10 astronauts; spectral power behaved very differently in long (180-minute) windows.
- **Maggioni et al. (2020)** Reduced vagal modulations of heart rate during overwintering in Antarctica. *Scientific Reports*, 10, 21810. DOI 10.1038/s41598-020-78722-3. VERIFIED (core fields). 25 crew over 14 months; baseline-adjusted HF power fell across the expedition and LF/HF shifted sympathetic in the last trimester.
- **Solbiati & Caiani (2025)** HRV adaptation due to head-down tilt bed rest. *npj Microgravity*, 11. DOI 10.1038/s41526-025-00509-5. VERIFIED. 66 men, six ESA campaigns; HF and LF fell during bed rest.
- **Lima et al. (2026)** Exercise and physiological monitoring in ICE environments and space analogues: a scoping review. *Experimental Physiology*. DOI 10.1113/EP093836. VERIFIED (abstract). 17 studies; heterogeneous methods limit comparability.

**Implication:** vagal HRV declines in orbit, in bed rest, and over a winter-over, so deconditioning and psychological strain can look alike. Judge HRV against the person's own in-mission trajectory, and exclude the first weeks after return.

## 8. ECG for arrhythmia and wearable screening (context for the same hardware)

See [ecg-applications](../ecg-applications/README.md) for arrhythmia detection, public datasets, and wearable-trial evidence.

## 9. Implementation steps

1. Fix the protocol: a daily 5-minute resting or sleep-period window, same posture and time (Quigley 2024).
2. Use chest-strap or patch ECG as the primary R-R source; headline index RMSSD plus mean heart rate.
3. Label PPG-derived values as pulse rate variability, gate by quality and motion, and use only at rest or sleep.
4. Validate any ultra-short window against 5-minute values on Blindsight data (bias and limits of agreement); use at least 120 s outside strict rest.
5. Build personal baselines from the first days and add a time-in-mission term.
6. Log sleep, medication class and start date, caffeine, activity, age, sex, posture, breathing or ECG-derived respiration, and habitat CO2 where available.
7. Evaluate subject-wise and time-forward, with a logistic-regression baseline and a leakage checklist.
8. Treat WESAD as a smoke test only; plan a prospective calibration study.
9. Frame outputs as a transdiagnostic autonomic strain indicator, not a diagnosis.
10. Exclude the post-mission recovery period from mood analysis.
11. Pre-register the validation plan with a sample-size justification.

## 10. Limitations and open questions

- Does within-person RMSSD change track within-person change in mood or stress in healthy confined crews? The only longitudinal design found (Hartmann et al. 2019, *Frontiers in Psychiatry*, 9, 735, DOI 10.3389/fpsyt.2018.00735; 62 unmedicated patients, 65 controls, 2-week follow-up) was clinical.
- No verified Mars-500 HRV study was found, and no society consensus statement specific to wearable HRV (Quigley 2024 is the closest).
- WESAD and Coutts cross-validation schemes remain unchecked.
- PARTIAL items need abstract or full-text review before any number is quoted: Shaffer & Ginsberg 2017, Laborde 2017, Gedam & Paul 2021, Geng 2023, Choe 2025 (Apple Watch accuracy; HR only, not HRV), Reyna 2021.
