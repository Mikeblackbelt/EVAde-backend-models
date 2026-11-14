# Combining ECG and EEG for Behavioral Health

Status labels: **VERIFIED** = bibliographic record and abstract checked; **PARTIAL** = record checked, abstract not read.

## 1. Summary

Combining EEG and ECG is plausible: they capture different systems (cortical activity and autonomic control), their information is partly complementary, and low-cost wireless hardware can record both. **However, the evidence for using the combination to detect depression, anxiety, or chronic stress in crews is thin.** The EEG-ECG fusion literature is dominated by healthy-volunteer emotion datasets using film clips. No high-quality, depression-specific EEG-ECG fusion study with subject-independent validation was found. That is a research gap, so this document proposes a study design instead of claiming a result.

## 2. What the evidence shows

- **Emotion datasets with EEG and ECG.** DREAMER (Katsigiannis & Ramzan, 2018, *IEEE J Biomed Health Inform*, 22(1), 98-107, DOI 10.1109/JBHI.2017.2688239; VERIFIED): 23 participants, low-cost wireless EEG and ECG, valence-arousal-dominance ratings, baseline classifiers with EEG, ECG, and fusion. DEAP (Koelstra et al., 2012, *IEEE Trans Affect Comput*, 3(1), 18-31, DOI 10.1109/T-AFFC.2011.15; PARTIAL), MAHNOB-HCI (Soleymani et al., 2012, 3(1), 42-55, DOI 10.1109/T-AFFC.2011.25; PARTIAL), SEED (Zheng & Lu, 2015, *IEEE Trans Auton Ment Dev*, 7(3), 162-175, DOI 10.1109/TAMD.2015.2431497; PARTIAL), and AMIGOS (Miranda-Correa et al., 2021, *IEEE Trans Affect Comput*, 12(2), 479-493, DOI 10.1109/TAFFC.2018.2884461; PARTIAL).
- **Reviews.** Lee et al. (2025), A review of hybrid EEG-based multimodal human-computer interfaces using deep learning, *Biomed Eng Lett*, 15(4), 587-618, DOI 10.1007/s13534-025-00469-5 (VERIFIED): 124 studies (2016-2024) of EEG combined with other biosignals including ECG. Shikha, Sethia & Indu (2025), A systematic review on physiology-based anxiety detection using machine learning, *Biomed Phys Eng Express*, DOI 10.1088/2057-1976/add5fc (PARTIAL): covers EEG, ECG, EMG, EDA, and respiration for anxiety.
- **A clinical fusion example (seizures, not mood).** Zhang et al. (2024), Multimodal wearable EEG, EMG and accelerometry measurements improve the accuracy of tonic-clonic seizure detection, *Physiol Meas*, 45(6), DOI 10.1088/1361-6579/ad4e94 (VERIFIED): 27 participants, 44 tonic-clonic seizures; fusing behind-the-ear EEG with other signals improved accuracy over unimodal detection (specific numbers not read).
- **ECG alone can carry a clinical task.** Jeppesen et al. (2025), seizure detection with a wearable ECG, *EBioMedicine*, 120, 105952, DOI 10.1016/j.ebiom.2025.105952 (VERIFIED): a phase 3 prospective, blinded, multicentre design, useful as an evaluation template.
- **What each stream is individually good for:** ECG-derived HRV for transdiagnostic autonomic strain (see [ecg-hrv.md](ecg-hrv.md)); EEG for sleep architecture and drowsiness (see [eeg.md](eeg.md)).

## 3. Why fusion might help, and why it might not

| Possible benefit | Risk |
|---|---|
| Drowsiness has an EEG signature (slow activity) and an autonomic signature (HRV, heart rate); agreement can raise confidence | Both are confounded by sleep, medication, and posture, so errors can be correlated rather than independent |
| EEG sleep staging plus ECG HRV gives sleep-stage-specific HRV, a candidate depression feature | Sleep-stage-specific HRV for depression classification was only partially reviewed (Geng 2023, PARTIAL) |
| Fusion can cover for a failed channel | More hardware, wear burden, and artifact handling; dry and ear EEG have known biases |
| Emotion datasets show gains from fusion | Datasets are healthy volunteers watching film clips; gains may not transfer to chronic states |

## 4. Proposed study design (not a result)

**Question.** Does adding EEG to ECG-derived features improve detection of within-person change in sleep-loss-related strain (and, secondarily, stress and mood) compared with ECG alone, under subject-independent evaluation?

**Population.** Healthy adult volunteers in a controlled sleep-restriction protocol, ideally followed into an isolation analog if one is accessible. Sample size should be set from a pilot; none is stated here because no verified effect size for the fusion gain exists.

**Reference labels.**
- Objective: 3-minute PVT-B (lapses and mean 1/RT were the most sensitive metrics; Basner & Dinges 2011) and polysomnography in a subset.
- Subjective: short mood and stress check-in; PHQ-9 and GAD-7 for change only (see [instruments-and-governance.md](instruments-and-governance.md)).
- Optional: cortisol, given Gemignani et al. (2014).

**Signals.**
- ECG: chest strap or patch, 5-minute windows during rest and sleep (RMSSD, mean heart rate, ECG-derived respiration).
- EEG: an ear or dry-electrode sleep and resting montage, plus a 19-channel scalp reference arm in the lab so the channel-reduction question can be answered.
- Covariates: sleep, medication, caffeine, posture, activity.

**Model arms.**
1. ECG-only features.
2. EEG-only features.
3. Feature-level fusion of quality-gated features.
4. Decision-level fusion (average calibrated probabilities).
5. A trivial per-person baseline and a logistic regression on a minimal feature set.

**Evaluation.**
- Subject-wise and time-forward splits; no subject appears in both train and test.
- Primary metric: detection of a pre-defined PVT-defined alertness drop at a fixed false-flag rate per person-day. Secondary: correlation with change in self-reported strain.
- Subject-level bootstrap confidence intervals and calibration.
- Ablations: remove each channel group, simulate channel dropout, degrade signal quality.
- Negative controls: shuffled labels and artifact-only segments.

**Pre-registration.** Feature set, windows, montage, artifact pipeline, model arms, metric, and thresholds are written before examining test data (reporting checklists: Brookshire 2024, Demirezen 2024, Kapoor & Narayanan 2023).

## 5. Implementation steps

1. Collect single-stream baselines first and confirm each beats a trivial baseline under subject-wise evaluation.
2. Align streams on a common clock; document drift and the maximum tolerated offset.
3. Compute per-stream quality indices and use them to gate fusion; do not fuse low-quality windows.
4. Start with decision-level fusion because it degrades gracefully when one stream drops out.
5. Add feature-level fusion only if a stream adds predictive value in the ablation.
6. Treat sleep-stage-specific HRV (EEG staging plus ECG) as an exploratory candidate feature.
7. Report the cost of the extra stream (wear burden, data loss rate) alongside any gain.
8. Decide in advance what result would lead to dropping EEG from the product.

## 6. Failure modes to watch

- Correlated errors (both streams hit by the same sleep loss or motion).
- Subject leakage through shared sessions or overlapping windows.
- Label noise, and reliance on a single label type.
- Population mismatch: a model trained on healthy film-clip data applied to chronic crew states.
- Hardware drift across devices and firmware versions.

## 7. Limitations and open questions

- No verified depression-specific EEG-ECG fusion study with subject-independent validation exists in this review.
- Whether fusion adds value beyond ECG alone for within-person sleep-loss strain is untested.
- PARTIAL items (DEAP, MAHNOB-HCI, SEED, AMIGOS, Shikha 2025, Geng 2023) need abstract review before any number is quoted.
