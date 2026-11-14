# Vital Sign Monitoring

## Summary
Heart rate, respiration rate, and SpO2 can be obtained from photoplethysmography (PPG) on a wearable, or contactlessly from video (remote PPG, rPPG) and radar. Wearable PPG is mature. rPPG is accurate under stable lighting and a still subject, but degrades with motion and skin-tone variation. For Blindsight this is the non-EEG/ECG physiological stream and a fallback when electrodes are unavailable.

## Sources
- Allen, J. (2007). Photoplethysmography and its application in clinical physiological measurement. *Physiological Measurement*, 28(3), R1-R39.
- Verkruysse, W., Svaasand, L. O., & Nelson, J. S. (2008). Remote plethysmographic imaging using ambient light. *Optics Express*, 16(26), 21434-21445.
- Poh, M.-Z., McDuff, D. J., & Picard, R. W. (2010). Non-contact, automated cardiac pulse measurements using video imaging and blind source separation. *Optics Express*, 18(10), 10762-10774.
- Wang, W., et al. (2017). Algorithmic principles of remote PPG. *IEEE Transactions on Biomedical Engineering*, 64(7), 1479-1491.

## Implementation steps
1. Choose the first modality: wrist PPG (simplest hardware, most public datasets) before rPPG.
2. Collect a labeled pilot set against a reference (chest-strap ECG for HR, finger pulse oximeter for SpO2) under rest and light-motion conditions.
3. Implement a baseline pipeline: band-pass filter (0.7-4 Hz), peak detection, HR from inter-beat intervals; evaluate MAE against the reference.
4. Add a signal-quality index (SQI) per window; reject windows below threshold rather than reporting a number.
5. For rPPG (later phase): face ROI tracking, chrominance-based or POS projection, and skin-tone-stratified evaluation before any claim.
6. Define the output schema: HR, RR, SpO2 (if available), SQI, timestamp, modality tag.
