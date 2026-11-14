# EEG Applications

## Summary
EEG measures cortical electrical activity at the scalp. Applications relevant here are seizure detection, cognitive and attention state, and sleep staging. Signals are low amplitude and noisy (eye movement, muscle, electrode drift), so preprocessing and artifact handling dominate performance. Compact convolutional networks (EEGNet) are a common starting point; patient-independent generalization is the hard part.

## Sources
- Shoeb, A. H., & Guttag, J. V. (2010). Application of machine learning to epileptic seizure detection. *Proceedings of the 27th International Conference on Machine Learning (ICML)*.
- Lawhern, V. J., et al. (2018). EEGNet: a compact convolutional neural network for EEG-based brain-computer interfaces. *Journal of Neural Engineering*, 15(5), 056013.
- Delorme, A., & Makeig, S. (2004). EEGLAB: an open source toolbox for analysis of single-trial EEG dynamics. *Journal of Neuroscience Methods*, 134(1), 9-21.
- Obeid, I., & Picone, J. (2016). The Temple University Hospital EEG Data Corpus. *Frontiers in Neuroscience*, 10, 196. (TUH EEG corpus, also the basis of the TUSZ work in this lab.)

## Implementation steps
1. Use the TUH EEG Seizure Corpus (TUSZ) for seizure detection, with EEGLAB-style preprocessing as the reference pipeline.
2. Preprocessing spec: re-reference (average or linked-ears), 0.5-50 Hz band-pass, 60 Hz notch, resample to a fixed rate, epoch into fixed windows.
3. Artifact handling: ICA for eye and muscle components, amplitude-based rejection as a second pass; log the rejection rate per recording.
4. Baseline model: feature-based (band powers, line length) with a gradient-boosted classifier; then EEGNet.
5. Evaluation: patient-independent splits only (no subject leakage); report event-based sensitivity and false alarms per 24 h.
6. Keep the same window and label schema as the ECG and vital-sign pipelines so Blindsight can fuse them later.
