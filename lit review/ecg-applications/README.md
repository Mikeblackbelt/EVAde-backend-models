# ECG Applications

Status labels: **VERIFIED** = bibliographic record and abstract checked; **PARTIAL** = record checked, abstract or numbers not read; **UNVERIFIED** = not confirmed. Behavioral and HRV uses of ECG are covered in [behavioral-health/ecg-hrv.md](../behavioral-health/ecg-hrv.md).

## 1. Summary

ECG gives beat-level timing. From it come QRS detection, heart rate, HRV, rhythm classification, and signal quality. The classical pipeline (band-pass filter, QRS detection, beat classification) is still the reference for detection, and deep networks now match or exceed cardiologists on several rhythm tasks given large labeled datasets. The main practical problems are **generalization across sites and devices** (PhysioNet challenges show cross-database differences exceeding lead-combination differences) and **transfer from clinical ECG to wearable single-lead data**, which is not established. Large wearable trials show PPG-based atrial fibrillation (AF) notification works at population scale in self-selected cohorts, with a confirmation step.

## 2. Classical and deep-learning detection

- **Pan & Tompkins (1985).** A real-time QRS detection algorithm. *IEEE Trans Biomed Eng*, BME-32(3), 230-236. DOI 10.1109/TBME.1985.325532. VERIFIED. 99.3% QRS detection on the MIT-BIH database (abstract). Implementation details (Z80, 200 Hz, 5-15 Hz energy band) come from secondary sources.
- **Hannun et al. (2019).** Cardiologist-level arrhythmia detection and classification in ambulatory electrocardiograms using a deep neural network. *Nature Medicine*, 25(1), 65-69. DOI 10.1038/s41591-018-0268-3. VERIFIED. 91,232 single-lead ECGs from 53,549 patients, 12 rhythm classes, average AUC 0.97, F1 0.837 (network) vs 0.780 (cardiologists); 328-patient test set. Correction: *Nat Med* 25(3), 530, DOI 10.1038/s41591-019-0359-9 (axis labels only). The relationship with the monitor manufacturer is **unverified**; check the competing-interests statement.
- **Ribeiro et al. (2020).** Automatic diagnosis of the 12-lead ECG using a deep neural network. *Nature Communications*, 11, 1760. DOI 10.1038/s41467-020-15432-4. VERIFIED. More than 2 million labeled exams (CODE study), 6 abnormalities, F1 above 80%, specificity above 99%, outperforming cardiology residents. Single-network source.
- **Hong et al. (2020).** Opportunities and challenges of deep learning methods for electrocardiogram data: a systematic review. *Comput Biol Med*, 122, 103801. DOI 10.1016/j.compbiomed.2020.103801. VERIFIED (record). 191 papers. A dedicated ECG denoising review was not found.

## 3. Datasets and benchmarks

- **PTB-XL.** Wagner et al. (2020). *Scientific Data*, 7, 154. DOI 10.1038/s41597-020-0495-6. VERIFIED. 21,837 12-lead records from 18,885 patients with cardiologist annotations and suggested folds (the PhysioNet v1.0.3 page lists 21,799 and 18,869; counts differ by version). Strodthoff et al. (2021), *IEEE J Biomed Health Inform*, 25(5), 1519-1528, DOI 10.1109/JBHI.2020.3022989 (VERIFIED): ResNet- and Inception-style CNNs strongest across tasks.
- **PhysioNet/CinC Challenge 2020.** Perez Alday et al. (2020). *Physiol Meas*, 41(12), 124003. DOI 10.1088/1361-6579/abc960. VERIFIED. 66,361 12-lead recordings from six hospital systems in four countries; 217 teams, 1,395 algorithms; top algorithms dropped about 10% on hidden test data.
- **Challenge 2021 follow-up.** Reyna et al. (2022). *Physiol Meas*, 43(8), 084001. DOI 10.1088/1361-6579/ac79fd. VERIFIED. 131,149 recordings from ten sources; 68 teams, 1,056 algorithms; **differences across test databases exceeded differences across lead combinations**; a voting ensemble improved results by 3.5%. Reyna et al. (2021) "Will two do?", *Computing in Cardiology*, DOI 10.23919/CinC53138.2021.9662687: PARTIAL.
- **Foundation and self-supervised models.** Mehari & Strodthoff (2022), *Comput Biol Med*, 141, 105114, DOI 10.1016/j.compbiomed.2021.105114 (VERIFIED; first comprehensive assessment of self-supervised learning on clinical 12-lead ECG). McKeen et al. (2025), ECG-FM, *JAMIA Open*, 8(5), ooaf122, DOI 10.1093/jamiaopen/ooaf122 (VERIFIED; open model, code, and benchmark; training data type not confirmed). ECGFounder (NEJM AI, 2025, DOI 10.1056/AIx2500913; PARTIAL; over 10 million ECGs per a search summary). Kiyasseh et al. (2021), *Nat Commun*, 12, 4221, DOI 10.1038/s41467-021-24483-0 (PARTIAL). None is shown to transfer to motion-contaminated wearable single-lead data.

## 4. Wearable atrial fibrillation trials

| Study | Cohort | Finding |
|---|---|---|
| Perez et al. (2019), *N Engl J Med*, 381(20), 1909-1917, DOI 10.1056/NEJMoa1901183 (VERIFIED) | 419,297 participants, 8 months | 0.52% got irregular-pulse notifications; of 450 with analyzable ECG patches, AF in 34% |
| Guo et al. (2019), *J Am Coll Cardiol*, 74(19), 2365-2375, DOI 10.1016/j.jacc.2019.08.019 (VERIFIED, core fields) | 187,912 monitored | 0.23% suspected AF; 227 of 262 followed up confirmed; PPV 91.6%; mean age 35, 86.7% male |
| Lubitz et al. (2022), *Circulation*, 146(19), 1415-1424, DOI 10.1161/CIRCULATIONAHA.122.060291 (VERIFIED) | 455,699 participants | irregular rhythm in 1%; of 1,057 with notification and patch, AF in 32.2% |

**Caveat for the Fitbit result:** the 98.2% positive predictive value is conditional on a second detection during patch wear and is not the same quantity as the 32.2% first-notification yield. Report both. All three cohorts are self-selected, young, or mostly White; AF notification performance is not equivalent to HR accuracy.

## 5. Standards and general references

- **Task Force of the ESC and NASPE (1996).** *Circulation*, 93(5), 1043-1065. DOI 10.1161/01.cir.93.5.1043. VERIFIED (older). **Quigley et al. (2024).** *Psychophysiology*, 61(9), e14604. DOI 10.1111/psyp.14604. VERIFIED (core fields). **Goldberger et al. (2000).** PhysioBank, PhysioToolkit, and PhysioNet. *Circulation*, 101(23), e215-e220. DOI 10.1161/01.CIR.101.23.e215. VERIFIED.

## 6. Implementation steps

1. Benchmark QRS detection on the MIT-BIH Arrhythmia Database with Pan-Tompkins; report sensitivity and positive predictive value.
2. Compute HRV from RR intervals following Quigley 2024 and the Task Force; flag windows with ectopic beats; add lead-off and noise quality gates.
3. For wearable use, define the hardware (chest strap, patch, or single-lead watch) before choosing models, since transfer from clinical ECG is unproven.
4. Benchmark any learned model on PTB-XL folds and on CinC data with held-out institutions; expect roughly a 10% cross-site drop.
5. Hold out an independent wearable dataset and run a prospective sample before claiming wearable performance.
6. Consider fine-tuning a pretrained model (ECG-FM or a similar open model), but measure domain shift explicitly.
7. Design alerts as confirm-before-alert: report yield and conditional PPV, never PPV alone.
8. Keep the classical detector as baseline and fallback.
9. State the clinical boundary in the schema and UI: the output is a flag for review, not a diagnosis.

## 7. Evaluation and robustness

- **Metrics:** QRS sensitivity and PPV; HRV bias and limits of agreement per measure; per-class F1 with confidence intervals; yield and conditional PPV for alerts.
- **Window choice:** HRV depends on window length; confirm the Task Force convention for short-term recordings, and test ultra-short windows against 5-minute values (see the behavioral ECG document).
- **Robustness checks:** noise-stress records; simulated baseline wander, muscle noise, and electrode motion; ectopic handling; per-record as well as pooled results (MIT-BIH has 47 subjects).

## 8. Limitations and open questions

- Deep models were trained on clinical or ambulatory-patch data; wearable single-lead generalization is not established, and no rigorous single-lead wearable external-validation paper was verified.
- HRV is an indirect marker of stress and mood and is heavily confounded; see the behavioral document.
- PARTIAL items (Reyna 2021, ECGFounder, Kiyasseh 2021) need abstract review before any number is quoted.
