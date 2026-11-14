# EEG Applications

## Summary

EEG records cortical electrical activity at the scalp. Applications relevant to Blindsight:

- **Seizure detection**: the most mature EEG task. Patient-specific classifiers on scalp EEG have been studied since the 2000s.
- **Cognitive and attention state**: spectral features and event-related potentials. Often confounded by artifacts.
- **Sleep staging**: established but needs its own labeled data.

The signal is small (microvolts), and artifacts from eye movement, muscle, and electrode drift are often larger than the signal of interest. Preprocessing and artifact handling usually decide results more than model choice. Compact CNNs (EEGNet) are a common starting point. The hard problem is generalization to new patients; many published results leak subjects across train and test splits.

## Sources

- **Shoeb, A. H., & Guttag, J. V. (2010).** Application of machine learning to epileptic seizure detection. *Proceedings of the 27th International Conference on Machine Learning (ICML 2010)*, pp. 975-982, Omnipress. Verified. Patient-specific classifiers on scalp EEG, using the CHB-MIT database (on PhysioNet). Shoeb's 2009 MIT thesis covers the same work in more depth.
- **Lawhern, V. J., Solon, A. J., Waytowich, N. R., Gordon, S. M., Hung, C. P., & Lance, B. J. (2018).** EEGNet: a compact convolutional neural network for EEG-based brain-computer interfaces. *Journal of Neural Engineering*, 15(5), 056013. DOI 10.1088/1741-2552/aace8c. Verified. The authors' GitHub (ARL EEGModels) has reference code, which is outside this review's no-code scope but useful to know about.
- **Delorme, A., & Makeig, S. (2004).** EEGLAB: an open source toolbox for analysis of single-trial EEG dynamics including independent component analysis. *Journal of Neuroscience Methods*, 134(1), 9-21. DOI 10.1016/j.jneumeth.2003.10.009. Verified. Source of the standard ICA-based artifact-removal approach. Check the current EEGLAB license before bundling it; it has changed over time.
- **Obeid, I., & Picone, J. (2016).** The Temple University Hospital EEG Data Corpus. *Frontiers in Neuroscience*, 10, 196. DOI 10.3389/fnins.2016.00196. Verified. Describes the TUH EEG corpus, which the TUSZ seizure work in this lab builds on. Access requires registration at nedcdata.org.

## Implementation steps

1. **Reference pipeline on TUSZ.** Use the TUH EEG Seizure Corpus with a documented preprocessing spec (see data sources). Reproduce a published baseline before changing anything.
2. **Preprocessing.** Re-reference (average or linked-ears), band-pass 0.5-50 Hz, 60 Hz notch (50 Hz outside the US), resample to a fixed rate, and epoch into fixed windows. Write these parameters into a config file so runs are reproducible.
3. **Artifact handling.** Run ICA for eye and muscle components, then amplitude-based rejection as a second pass. Log the rejection rate per recording; a high rate means the recording is excluded, not cleaned.
4. **Baseline model.** Feature-based (band powers, line length, etc.) with a gradient-boosted classifier. Only then try EEGNet, and report it against the baseline.
5. **Evaluation.** Use patient-independent splits only. Report event-based sensitivity, false alarms per 24 h, and the number of subjects in each split. Include a subject-level bootstrap confidence interval.
6. **Schema alignment.** Use the same window length and label schema as the ECG and vital-sign pipelines so Blindsight can fuse them later.
