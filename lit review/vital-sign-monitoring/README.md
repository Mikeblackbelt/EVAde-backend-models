# Vital Sign Monitoring

## Summary

Heart rate (HR), respiration rate (RR), and blood oxygen saturation (SpO2) can be measured in three ways:

- **Contact PPG** (wrist or finger): mature, low cost, sensitive to motion. Widely deployed.
- **Remote PPG (rPPG)** from video: a camera measures skin color changes caused by the pulse. Verkruysse et al. (2008) showed a consumer camera and ambient light could give usable HR, and the field grew from there. Accuracy is good in stable light with a still subject. Motion and skin-tone variation are the main error sources.
- **Radar**: contactless, works in the dark, but needs its own hardware.

For Blindsight, PPG is the non-EEG/ECG physiological stream. rPPG is a later-phase option. It is useful when electrodes are unavailable, but it needs skin-tone-stratified validation before any claim.

## Sources

- **Allen, J. (2007).** Photoplethysmography and its application in clinical physiological measurement. *Physiological Measurement*, 28(3), R1-R39. DOI 10.1088/0967-3334/28/3/R01. Verified. The standard review of PPG principles, instrumentation, measurement protocol, and pulse wave analysis. Read sections on instrumentation and pulse wave analysis first.
- **Verkruysse, W., Svaasand, L. O., & Nelson, J. S. (2008).** Remote plethysmographic imaging using ambient light. *Optics Express*, 16(26), 21434-21445. DOI 10.1364/OE.16.021434. Verified. Measured HR and RR remotely (over 1 m) with a consumer camera in movie mode. The green channel gave the strongest signal.
- **Poh, M.-Z., McDuff, D. J., & Picard, R. W. (2010).** Non-contact, automated cardiac pulse measurements using video imaging and blind source separation. *Optics Express*, 18(10), 10762-10774. DOI 10.1364/OE.18.010762. Verified. Introduced automated face tracking with ICA on color channels. Full text is open access via MIT DSpace (handle 1721.1/66243).
- **Wang, W., den Brinker, A. C., Stuijk, S., & de Haan, G. (2017).** Algorithmic principles of remote PPG. *IEEE Transactions on Biomedical Engineering*, 64(7), 1479-1491. DOI 10.1109/TBME.2016.2609282. Verified. Proposes the Plane-Orthogonal-to-Skin (POS) method, which is a standard baseline for rPPG.

## Implementation steps

1. **Start with contact PPG.** Wrist PPG is the simplest hardware and has the most public datasets (see [data-collection-sources](../data-collection-sources/physiological-signals.md)).
2. **Collect a reference-labeled pilot.** Use a chest-strap ECG for HR and a finger pulse oximeter for SpO2. Record rest, light motion, and talking. Aim for enough subjects to span skin tones; record Fitzpatrick type where ethically approved.
3. **Baseline pipeline.** Band-pass 0.7-4 Hz (42-240 bpm), peak detection, HR from inter-beat intervals. Report MAE and Bland-Altman limits of agreement against the reference.
4. **Signal-quality index (SQI).** Compute per window: spectral peak sharpness, peak regularity, and motion energy from an accelerometer. Suppress output below threshold rather than reporting a low-confidence number.
5. **Respiration.** Derive RR from the PPG baseline wander and amplitude modulation; validate against a respiration belt before reporting.
6. **rPPG (later phase).** Track a face ROI, then apply POS or chrominance-based extraction. Compare against your contact-PPG baseline on the same recordings.
7. **Output schema.** Each record: HR, RR, SpO2 (if available), SQI, timestamp, modality tag, and a radiation-health flag from the space-radiation module.

## Evaluation and robustness

- **Metrics:** MAE and RMSE for HR and RR; Bland-Altman bias and limits of agreement; percent of windows retained after SQI gating; SpO2 error against a reference oximeter.
- **Stratification:** report results by motion level (rest, walking, talking), ambient light (for rPPG), sensor placement, and skin tone.
- **Robustness checks:** deliberately degrade signals (added motion, reduced light, shifted sensor position) and report how SQI responds; check that the SQI actually predicts error.
- **Reference quality:** the reference device has its own error; state it.

## Limitations and open questions

- rPPG accuracy depends on stable lighting and little motion; skin-tone effects need explicit testing, and the literature on measurement bias across skin tones was not reviewed in this pass.
- Wrist SpO2 requires its own calibration and validation; HR performance does not transfer to SpO2.
- Open: which modality is the fallback when the primary sensor fails? What minimum SQI justifies reporting a number?
