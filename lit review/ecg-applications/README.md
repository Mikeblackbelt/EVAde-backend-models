# ECG Applications

## Summary
ECG gives beat-level timing, from which heart rate variability (HRV), arrhythmia, and autonomic/stress markers are derived. The classical pipeline is band-pass filtering, QRS detection, and beat classification. Deep learning now matches cardiologist-level performance on single-lead arrhythmia detection given large datasets. For Blindsight, ECG is a strong input for stress and fatigue state and arrhythmia flagging, but single-lead wearables have artifact problems that must be handled explicitly.

## Sources
- Pan, J., & Tompkins, W. J. (1985). A real-time QRS detection algorithm. *IEEE Transactions on Biomedical Engineering*, BME-32(3), 230-236.
- Hannun, A. Y., et al. (2019). Cardiologist-level arrhythmia detection and classification in ambulatory electrocardiograms using a deep neural network. *Nature Medicine*, 25, 65-69.
- Task Force of the European Society of Cardiology and NASPE (1996). Heart rate variability: standards of measurement, physiological interpretation and clinical use. *Circulation*, 93(5), 1043-1065.
- Goldberger, A. L., et al. (2000). PhysioBank, PhysioToolkit, and PhysioNet. *Circulation*, 101(23), e215-e220.

## Implementation steps
1. Start with public data: the MIT-BIH Arrhythmia Database on PhysioNet for QRS detection benchmarks.
2. Implement Pan-Tompkins as the reference detector; report sensitivity and positive predictive value on MIT-BIH.
3. Compute HRV from RR intervals (SDNN, RMSSD, LF/HF) per the 1996 Task Force definitions; flag windows containing ectopic beats.
4. Add a quality gate (noise, lead-off detection) before any HRV feature is passed on.
5. Only then consider a learned arrhythmia model; keep the classical detector as baseline and fallback.
6. State the clinical boundary in the README: the system flags for review, it does not diagnose.
