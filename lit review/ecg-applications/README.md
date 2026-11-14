# ECG Applications

## Summary

ECG gives beat-level timing. From it you derive:

- **Heart rate variability (HRV)**: time and frequency measures of RR-interval variation. Used as a proxy for autonomic state, stress, and fatigue. The 1996 Task Force defines the standard measures.
- **Arrhythmia detection**: rhythm classification, from classical QRS-based rules to deep networks.
- **Signal quality**: lead-off, motion, and muscle artifact detection, which matter more for single-lead wearables than for clinical 12-lead systems.

The classical pipeline is band-pass filter, QRS detection, and beat classification. Pan-Tompkins (1985) remains the reference detector. Deep networks now perform well on single-lead arrhythmia classification with large labeled datasets. Hannun et al. (2019) is the key example, but its test set is small (328 patients), and the study's links to the monitor manufacturer are unverified here, so treat its headline numbers with care.

## Sources

- **Pan, J., & Tompkins, W. J. (1985).** A real-time QRS detection algorithm. *IEEE Transactions on Biomedical Engineering*, BME-32(3), 230-236. DOI 10.1109/TBME.1985.325532. Verified. Reports 99.3% QRS detection on the MIT-BIH arrhythmia database. The original implementation ran on a Z80 or NSC800 microprocessor with integer arithmetic. The filter targets QRS energy at about 5-15 Hz; sample rate in the original was 200 Hz.
- **Hannun, A. Y., et al. (2019).** Cardiologist-level arrhythmia detection and classification in ambulatory electrocardiograms using a deep neural network. *Nature Medicine*, 25, 65-69. DOI 10.1038/s41591-018-0268-3. Verified: citation. Study facts from search-result abstract text: 91,232 single-lead ECGs from 53,549 patients (one secondary source gives 53,877 patients; treat as unresolved), 12 rhythm classes, average AUC 0.97, average F1 0.837 (DNN) vs. 0.780 (cardiologists). The test set was 328 patients. A correction (Nat Med 25(3):530, 2019, DOI 10.1038/s41591-019-0359-9) fixes axis labels in Figures 1a and 2b only. The funding and co-authorship relationship with the monitor manufacturer (iRhythm) is **UNVERIFIED** here; check the competing-interests statement in the paper before stating it.
- **Task Force of the ESC and NASPE (1996).** Heart rate variability: standards of measurement, physiological interpretation and clinical use. *Circulation*, 93(5), 1043-1065. DOI 10.1161/01.cir.93.5.1043. Verified. Defines HRV nomenclature, measurement standards, and frequency bands. Read the frequency-domain section before computing LF/HF.
- **Goldberger, A. L., et al. (2000).** PhysioBank, PhysioToolkit, and PhysioNet. *Circulation*, 101(23), e215-e220. DOI 10.1161/01.CIR.101.23.e215. Verified. Introduces the data archive and tools used throughout this review.

## Implementation steps

1. **Benchmark on public data.** Run Pan-Tompkins on the MIT-BIH Arrhythmia Database and report sensitivity and positive predictive value against the annotated beats.
2. **HRV pipeline.** Compute SDNN, RMSSD, and frequency-domain power from RR intervals using the Task Force definitions. Flag windows containing ectopic beats or more than a set fraction of corrected beats.
3. **Quality gate.** Add lead-off detection and a noise index before any HRV feature is emitted. Test it on MIT-BIH noise-stress records.
4. **Stress/fatigue proxy.** Validate HRV features against a labeled stress dataset (WESAD; see data sources) using subject-wise splits. Report the effect size, not only accuracy.
5. **Learned arrhythmia model (later).** Train on PTB-XL or a similar large 12-lead dataset, then test transfer to single-lead data. Keep Pan-Tompkins and rule-based flags as the fallback and as a comparison.
6. **Clinical boundary.** The output is a flag for review, not a diagnosis. Put this in the schema and in the UI text.

## Evaluation and robustness

- **Metrics:** QRS detection sensitivity and positive predictive value; HRV agreement with a reference (bias and limits of agreement per HRV measure); rhythm classification F1 per class, with confidence intervals.
- **Window choice:** HRV measures depend on window length. The Task Force describes short-term (about 5-minute) and long-term (24-hour) recordings as standard conventions; confirm the exact recommendation before fixing a window. Report how ultra-short windows differ from the reference.
- **Robustness checks:** noise-stress records; baseline wander, muscle noise, and electrode motion simulated at known levels; ectopic beat handling.
- **Population:** MIT-BIH has 47 subjects; per-subject variance is large, so report per-record results as well as pooled results.

## Limitations and open questions

- Deep arrhythmia models were trained on clinical or ambulatory-patch data; transfer to wearable single-lead data is not guaranteed.
- HRV is an indirect marker of stress or behavioral state; confounded by medication, caffeine, posture, and breathing (see [behavioral-health](../behavioral-health/README.md)).
- Open: what quality threshold makes HRV usable for a behavioral flag? Is PPG-derived HRV adequate where ECG is not available?
