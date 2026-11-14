# EEG Applications

Status labels: **VERIFIED** = bibliographic record and abstract checked; **PARTIAL** = record checked, abstract not read; **UNVERIFIED** = not confirmed. Behavioral uses of EEG (mood, fatigue, vigilance) are covered in [behavioral-health/eeg.md](../behavioral-health/eeg.md).

## 1. Summary

The mature EEG applications here are seizure detection and sleep staging. Seizure detection has a clinical guideline (ILAE and IFCN), an event-based evaluation framework (SzCORE), large public corpora (TUSZ for clinical scalp EEG, SeizeIT2 for wearable behind-the-ear EEG), and wearable validation trials. The guideline supports wearable detection of **convulsive** seizures, and not other seizure types. Preprocessing and artifact handling dominate performance, and **subject leakage** is the most common reason published numbers do not hold on new patients.

## 2. Seizure detection

- **TUSZ.** Shah et al. (2018). The Temple University Hospital Seizure Detection Corpus. *Front Neuroinform*, 12, 83. DOI 10.3389/fninf.2018.00083. VERIFIED (record). **Obeid & Picone (2016).** The Temple University Hospital EEG Data Corpus. *Front Neurosci*, 10, 196. DOI 10.3389/fnins.2016.00196. VERIFIED. Access requires registration (nedcdata.org). No 2020+ TUSZ benchmark paper beyond SzCORE was verified.
- **SzCORE.** Dan et al. (2025). *Epilepsia*, 66(Suppl 3), 14-24. DOI 10.1111/epi.18113. VERIFIED. Standardizes datasets, file formats, cross-validation, and event-based metrics, with an open "EEG 10-20 seizure detection benchmark". Benchmark companion: Samanos et al. (2025), *IEEE EMBC*, DOI 10.1109/EMBC58623.2025.11254038 (PARTIAL).
- **ILAE and IFCN guideline.** Beniczky et al. (2021). Automated seizure detection using wearable devices: a clinical practice guideline. *Epilepsia*, 62(3), 632-646. DOI 10.1111/epi.16818. VERIFIED. High-level evidence for generalized and focal-to-bilateral tonic-clonic seizure detection; moderate for other types; not recommended for clinical use for other seizure types; unclear whether alarms improve outcomes. Validation standards: Beniczky & Ryvlin (2018), *Epilepsia*, 59(Suppl 1), 9-13, DOI 10.1111/epi.14049 (VERIFIED; older).
- **SeizeIT2.** Bhagubai et al. (2025). Wearable dataset of patients with focal epilepsy. *Scientific Data*, 12, 1228. DOI 10.1038/s41597-025-05580-x. VERIFIED. Over 11,000 hours of multimodal data (behind-the-ear EEG, ECG, EMG, motion); 883 focal seizures from 125 patients at five European centers; BIDS on OpenNeuro.
- **Wearable device studies.** Zhang et al. (2024), *Physiol Meas*, 45(6), DOI 10.1088/1361-6579/ad4e94 (VERIFIED; 27 participants, 44 tonic-clonic seizures; multimodal improved accuracy). Lehnen et al. (2025), *J Clin Neurophysiol*, 42(2), 118-125, DOI 10.1097/WNP.0000000000001076 (VERIFIED; 32 patients, 4 scalp electrodes versus 21-channel video-EEG). Jeppesen et al. (2025), *EBioMedicine*, 120, 105952, DOI 10.1016/j.ebiom.2025.105952 (VERIFIED; phase 3 wearable ECG detector). Onorati et al. (2021), *Front Neurol*, 12, 724904, DOI 10.3389/fneur.2021.724904 (PARTIAL). "Empaviot" was not found in any index (UNVERIFIED; possibly a misnomer).
- **Classic reference.** Shoeb & Guttag (2010). *Proc 27th ICML*, 975-982. VERIFIED. Patient-specific classifiers on the CHB-MIT database.
- **Updated review (not verified):** Mohamed et al. (2026), *Front Bioeng Biotechnol*, 14, 1833080, DOI 10.3389/fbioe.2026.1833080 (record only).

## 3. Models, preprocessing, and reproducibility

- **EEGNet.** Lawhern et al. (2018). *J Neural Eng*, 15(5), 056013. DOI 10.1088/1741-2552/aace8c. VERIFIED. Compact CNN for BCIs.
- **Leakage.** Brookshire et al. (2024). *Front Neurosci*, 18, 1373515. DOI 10.3389/fnins.2024.1373515. VERIFIED. Segment-based holdout strongly overestimated performance on unseen subjects in seizure and Alzheimer's data.
- **Foundation models.** Kuruppu et al. (2026). *J Neural Eng*, 23(2), 021001. DOI 10.1088/1741-2552/ae4455. VERIFIED. Evaluations are heterogeneous and limited; off-the-shelf utility is hard to assess.
- **Standards and tools.** Pernet et al. (2020) COBIDAS MEEG, *Nat Neurosci*, 23(12), 1473-1483, DOI 10.1038/s41593-020-00709-0 (PARTIAL); Pernet et al. (2019) EEG-BIDS, *Sci Data*, 6, 103, DOI 10.1038/s41597-019-0104-8 (PARTIAL); Appelhoff et al. (2019) MNE-BIDS, *JOSS*, 4(44), 1896, DOI 10.21105/joss.01896 (VERIFIED); Bigdely-Shamlo et al. (2015) PREP, *Front Neuroinform*, 9, 16, DOI 10.3389/fninf.2015.00016 (VERIFIED); Gramfort et al. (2013) MNE-Python, *Front Neurosci*, 7, 267, DOI 10.3389/fnins.2013.00267 (VERIFIED); Delorme & Makeig (2004) EEGLAB, *J Neurosci Methods*, 134(1), 9-21, DOI 10.1016/j.jneumeth.2003.10.009 (VERIFIED).
- **Artifacts.** Gorjan et al. (2022), *J Neural Eng*, 19(1), 011004, DOI 10.1088/1741-2552/ac542c (VERIFIED); Ledwidge et al. (2025), *Sensors*, 25(15), 4810, DOI 10.3390/s25154810 (VERIFIED); Chang et al. (2020), *IEEE TBME*, 67(4), 1114-1121, DOI 10.1109/TBME.2019.2930186 (PARTIAL); ICLabel, Pion-Tonachini et al. (2019), *NeuroImage*, 198, 181-197, DOI 10.1016/j.neuroimage.2019.05.026 (PARTIAL).
- **Hardware.** Niso et al. (2023), *NeuroImage*, 269, 119774, DOI 10.1016/j.neuroimage.2022.119774 (VERIFIED; 48 wireless devices); Hinrichs et al. (2020), *Sci Rep*, 10, 5218, DOI 10.1038/s41598-020-62154-0 (VERIFIED; dry versus wet electrodes, theta and delta biased upward); Mikkelsen et al. (2019), *Sci Rep*, 9, 16824, DOI 10.1038/s41598-019-53115-3 (VERIFIED; ear-EEG sleep staging kappa 0.73).

## 4. Implementation steps

1. Use TUSZ for scalp seizure detection and SeizeIT2 as the external wearable test set; reproduce a published baseline before changing anything.
2. Scope claims to convulsive seizures for wearable devices unless focal-seizure evidence is generated.
3. Preprocessing spec in a config file: re-reference, 0.5-50 Hz band-pass, line-frequency notch (60 Hz in the US), fixed resampling rate, fixed windows.
4. Artifact handling by activity level: ICA for eye and muscle components, ASR or similar as a second pass; log rejection rates; exclude recordings with high rejection.
5. Baseline model: band-power and line-length features with a gradient-boosted classifier; then EEGNet; report both.
6. Evaluate with SzCORE event-based metrics (sensitivity, false alarms per 24 hours, event F1) and patient-disjoint splits; do not compare with epoch-level literature numbers.
7. Store data in EEG-BIDS via MNE-BIDS and report against COBIDAS-MEEG.
8. Treat TUSZ-trained models as unvalidated for behind-the-ear or in-ear montages until tested on SeizeIT2.
9. Use the same window and label schema as the ECG and vital-sign pipelines to allow later fusion.

## 5. Evaluation and robustness

- **Metrics:** event-based sensitivity, false alarms per 24 hours, detection latency; number of subjects and seizures in each split.
- **Robustness checks:** channel dropout and montage change; noise injection (blink, muscle); performance by seizure type and recording duration; sensitivity to the artifact-rejection threshold.
- **Uncertainty:** subject-level bootstrap confidence intervals; seizure counts per patient are small, so intervals will be wide.

## 6. Limitations and open questions

- Full-montage clinical EEG differs from few-channel wearable EEG; a channel-mapping and retraining step is required.
- Annotation noise sets a ceiling on measurable performance.
- Which 2020+ TUSZ benchmark papers are citable, and what is the minimum channel count and placement for useful wearable seizure detection?
- PARTIAL items (Samanos 2025, Onorati 2021, COBIDAS, EEG-BIDS, Chang 2020, ICLabel) need abstract review before any number is quoted.
