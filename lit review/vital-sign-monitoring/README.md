# Vital Sign Monitoring

Status labels: **VERIFIED** = bibliographic record and abstract checked; **PARTIAL** = record checked, abstract or numbers not read, or only secondary sources; **UNVERIFIED** = not confirmed. Author lists marked "from memory" in the underlying research were left out here.

## 1. Summary

Heart rate (HR), respiration rate (RR), and blood oxygen saturation (SpO2) come from contact PPG, camera-based remote PPG (rPPG), or radar. HR from wrist PPG is mature, with error dominated by activity. RR and HRV-like measures are less mature. SpO2 is the highest-stakes signal because of **documented skin-tone bias in pulse oximetry**, and camera SpO2 is research-grade. rPPG accuracy is good under stable light and low motion, and its generalization across datasets, skin tones, and lighting remains the main open problem.

## 2. Skin tone, demographics, and pulse oximetry bias

- **Sjoding et al. (2020)** Racial bias in pulse oximetry measurement. *N Engl J Med*, 383(25), 2477-2478. DOI 10.1056/NEJMc2029240. VERIFIED (record; numbers from secondary summaries). Among SpO2 of 92-96%, SaO2 below 88% (occult hypoxemia) in 11.7% of Black versus 3.6% of White patients in one cohort. Retrospective research letter.
- **Wong et al. (2021)** *JAMA Netw Open*, 4(11), e2131674. DOI 10.1001/jamanetworkopen.2021.31674. VERIFIED. 87,971 patients; hidden hypoxemia 6.8% in Black, 6.0% Hispanic, 4.8% Asian, 4.9% White patients; associated with worse outcomes.
- **Fawzy et al. (2022)** *JAMA Intern Med*, 182(7), 730-738. DOI 10.1001/jamainternmed.2022.1906. VERIFIED. 7,126 COVID-19 patients; occult hypoxemia about 28-30% in Asian, Black, and Hispanic patients versus 17.2% in White patients; delayed treatment eligibility.
- **Gottlieb et al. (2022)** *JAMA Intern Med*, 182(8), 849-858. DOI 10.1001/jamainternmed.2022.2587. VERIFIED (record and abstract; author list beyond the first three unconfirmed). 3,069 ICU patients; less supplemental oxygen in Asian, Black, and Hispanic patients.
- **Shi et al. (2022)** The accuracy of pulse oximetry by levels of skin pigmentation: a systematic review and meta-analysis. *BMC Med*, 20, 267. DOI 10.1186/s12916-022-02452-8. VERIFIED. 32 studies, 6,505 participants; pooled mean bias about +1.1% in high pigmentation, imprecision "unacceptably large" (pooled SD above 1%). **Disagreement:** this is more moderate than the EHR studies above; differences stem from design (healthy-volunteer desaturation versus sick ICU patients), ethnicity labels, and threshold-based outcomes. Report both mean bias and threshold error.
- **FDA draft guidance (7 Jan 2025)** Pulse oximeters for medical purposes: non-clinical and clinical performance testing, labeling, and premarket submission recommendations (docket FDA-2023-N-4976). PARTIAL (confirmed via Federal Register listing and secondary reports; full text not read). Reported contents: at least 150 participants (from 10 in 2013), three skin-tone groups assessed with the Monk Skin Tone scale plus an objective measurement, about 25% in each group; general-wellness devices excluded. **No sign of finalization was found**; re-check the FDA page before citing as draft or final.
- **ISO 80601-2-61:2017** (ARMS criterion of 4% or less consistent across sources). A 2026 edition appears to exist; contents are UNVERIFIED (secondary sources only). The standard covers contact oximeters, so camera and wrist consumer sensors would not be certified under it, but they should borrow its ARMS metric and desaturation protocol.
- **rPPG and wearable PPG across tones:**
  - Nowara, McDuff & Veeraraghavan (2020), CVPR Workshops, 1148-1155, DOI 10.1109/CVPRW50498.2020.00150 (PARTIAL): three datasets, 73 people; accuracy dropped notably for the darkest skin type under some conditions.
  - Dasari et al. (2021), *npj Digit Med*, 4, 91, DOI 10.1038/s41746-021-00462-z (VERIFIED): rPPG methods showed similar biases across demographic groups and environments in new data from India and Sierra Leone. **The two papers disagree; test under EVAde's own lighting.**
  - Bondarenko, Menon & Elgendi (2025), *npj Digit Med*, 8, 593, DOI 10.1038/s41746-025-01973-9 (VERIFIED): audit of 100 rPPG studies found darker skin tones and gender balance underrepresented in public datasets.
  - Bent et al. (2020), *npj Digit Med*, 3, 18, DOI 10.1038/s41746-020-0226-6 (VERIFIED; sample size unconfirmed): no significant HR accuracy difference across skin tones for wearables, but error was about 30% higher during activity than at rest.
- **Gap:** no fairness-methods paper for rPPG was verified, and no peer-reviewed, large-n camera-SpO2 study with tone stratification (only preprints).

## 3. rPPG methods, datasets, and benchmarks

- **Foundational:** Allen (2007), *Physiol Meas*, 28(3), R1-R39, DOI 10.1088/0967-3334/28/3/R01 (VERIFIED; PPG review); Verkruysse et al. (2008), *Optics Express*, 16(26), 21434-21445, DOI 10.1364/OE.16.021434 (VERIFIED; consumer camera, ambient light, green channel strongest); Poh, McDuff & Picard (2010), *Optics Express*, 18(10), 10762-10774, DOI 10.1364/OE.18.010762 (VERIFIED; face tracking plus ICA); Wang et al. (2017), *IEEE TBME*, 64(7), 1479-1491, DOI 10.1109/TBME.2016.2609282 (VERIFIED; POS method); Chen & McDuff (2018) DeepPhys, ECCV, DOI 10.1007/978-3-030-01216-8_22 (VERIFIED record).
- **Toolbox:** Liu et al. (2023) rPPG-Toolbox, *NeurIPS 2023 Datasets and Benchmarks*, 68485-68510, DOI 10.52202/075280-2995 (VERIFIED venue; cross-dataset numbers not read). Standardized evaluation across six public datasets.
- **Models:** MTTS-CAN / TS-CAN (Liu, Fromm, Patel & McDuff, NeurIPS 2020; VERIFIED via proceedings; pulse and respiration jointly, over 150 fps on an ARM CPU per the authors); PhysFormer (Yu et al., CVPR 2022, 4176-4186, DOI 10.1109/CVPR52688.2022.00415; VERIFIED record) and PhysFormer++ (*Int J Comput Vis*, 131(6), 1307-1330, DOI 10.1007/s11263-023-01758-1; VERIFIED record); RhythmFormer (Zou et al., *Pattern Recognition*, 164, 111511, 2025, DOI 10.1016/j.patcog.2025.111511; VERIFIED venue; MAE figures are author claims on easy, light-skinned controlled datasets). PhysNet (BMVC 2019) is UNVERIFIED as a record.
- **Data:** SCAMPS synthetic data (McDuff et al., NeurIPS 2022, 3744-3757, DOI 10.52202/068431-0271; VERIFIED record); MMPD (Tang et al., EMBC 2023, DOI 10.1109/EMBC40787.2023.10340857; VERIFIED record; includes skin-tone, motion, and lighting conditions on a phone); UBFC-rPPG (Bobbia et al., *Pattern Recognit Lett*, 124, 82-90, 2019, DOI 10.1016/j.patrec.2017.10.017; VERIFIED); PURE (Stricker et al., RO-MAN 2014, DOI 10.1109/ROMAN.2014.6926392; VERIFIED); VIPL-HR (Niu et al., ACCV 2018, DOI 10.1007/978-3-030-20873-8_36; VERIFIED record).
- **Reviews:** Cheng et al. (2021), *Sensors*, 21(18), 6296, DOI 10.3390/s21186296 (VERIFIED record). Saikevicius et al. (2024), *Sensors*, 24(12), 3963, DOI 10.3390/s24123963 (VERIFIED record). Chen et al. (2024), *Front Bioeng Biotechnol*, 12, 1420100, DOI 10.3389/fbioe.2024.1420100 (VERIFIED). These are narrative reviews; no rigorous PRISMA review of cross-dataset generalization was found.
- **Respiration and SpO2 from video:** Boccignone et al. (2023), *Sensors*, 23(7), 3387, DOI 10.3390/s23073387 (VERIFIED): systematic evaluation of rPPG respiratory rate methods on public datasets; rPPG respiration is far less mature than HR. Chan et al. (2023), EMBC, DOI 10.1109/EMBC40787.2023.10340025 (PARTIAL): camera SpO2 feasibility with induced desaturation to about 81%. Treat camera SpO2 as research-grade.
- **Caution:** MAE on PURE and UBFC (both small, controlled, lighter-skinned) overstates real-world generalization.

## 4. Wearable PPG validation, signal quality, and frameworks

- **V3 framework.** Goldsack et al. (2020). *npj Digit Med*, 3, 55. DOI 10.1038/s41746-020-0260-4. VERIFIED. Verification, analytical validation, and clinical validation; structure the evidence plan this way.
- **Systematic reviews.** Fuller et al. (2020), *JMIR mHealth uHealth*, 8(9), e18694, DOI 10.2196/18694 (VERIFIED; 158 publications; conclusions not read). Chan et al. (2022), *Anaesthesia*, 77(11), 1268-1280, DOI 10.1111/anae.15834 (VERIFIED): 84 studies; wearable HR limits of agreement -9.36 to 10.08 bpm; wearable RR bias 0.68 breaths/min; only two wearable SpO2 studies; studies were low quality, small, often unblinded. This is the main justification for blinded, adequately powered, Bland-Altman-with-limits validation.
- **Consumer wearable studies.** Nelson & Allen (2019), *JMIR mHealth uHealth*, 7(3), e10828, DOI 10.2196/10828 (VERIFIED; n = 1, protocol example only). Schröder et al. (2023), *npj Digit Med*, 6, 63, DOI 10.1038/s41746-023-00814-x (VERIFIED): 38 healthy users; 14% of watch SpO2 readings below 95%, none below 92%; low-side outliers need handling.
- **PPG reviews and signal quality.** Charlton et al. (2022), *Proc IEEE*, 110(3), 355-381, DOI 10.1109/JPROC.2022.3149785 (VERIFIED). Charlton et al. (2025), *PLOS Digit Health*, 4(6), e0000585, DOI 10.1371/journal.pdig.0000585 (VERIFIED): 1,142 subjects; PPG quality highest supine (SNR 18.6 dB) and lower in other postures and arm heights. Desquins et al. (2022), *Appl Sci*, 12(19), 9582, DOI 10.3390/app12199582 (VERIFIED record; spans contact PPG and imaging PPG quality assessment). Park et al. (2021), *Front Physiol*, 12, 808451, DOI 10.3389/fphys.2021.808451 (PARTIAL). No standard SQI thresholds exist.
- **Respiration and variability.** Charlton et al. (2018), *IEEE Rev Biomed Eng*, 11, 2-20, DOI 10.1109/RBME.2017.2763681 (VERIFIED; assessment framework for breathing-rate algorithms). Mejia-Mejia et al. (2020), *Front Physiol*, 11, 779, DOI 10.3389/fphys.2020.00779 (VERIFIED): PRV and HRV differ under physiological stress (cold exposure, 20 volunteers); do not label PPG-derived variability as HRV.
- **Standards not verified:** CTA wearable HR standards and IEEE standards (GAP).

## 5. Implementation steps

1. Use the V3 framework as the master evidence plan for each modality.
2. Define claim level per signal: HR and RR for wellness; SpO2 and rhythm alerts as research-only until validated. Re-check the FDA draft status and obtain the ISO 80601-2-61:2026 text.
3. Build a tone-stratified validation cohort: record objective skin tone (colorimeter) plus the Monk scale; for any SpO2 claim, follow a design like the FDA draft (at least 150 participants, three groups, about 25% each); report error per stratum, never pooled.
4. Report agreement with Bland-Altman bias and 95% limits of agreement, MAE, and tail metrics (for example false-reassurance at SpO2 92-96%).
5. Run candidate rPPG models (TS-CAN, PhysFormer, RhythmFormer) in rPPG-Toolbox with identical preprocessing; train on one dataset and test on others (PURE, UBFC, MMPD, VIPL-HR, SCAMPS); treat MMPD and EVAde's own phone data as the hard tests.
6. Add synthetic pretraining (SCAMPS) and tone-balanced augmentation, then re-measure per-tone error; collect your own darker-skin, low-light, and motion data.
7. Implement and report signal-quality indices for contact PPG and camera (template matching, SNR, perfusion); publish the percentage of valid time by activity, tone, and posture; suppress output below threshold.
8. For wearable HR and HRV-like measures, test at rest, graded exercise, and free-living against a chest strap or ECG; call it pulse rate variability unless validated.
9. For respiration, use the Charlton framework for contact PPG and ECG and public-dataset evaluation for rPPG; keep respiration labeled experimental.
10. Pre-register thresholds and a fairness gate (maximum allowed between-group error difference); archive dataset versions, splits, and metric code.

## 6. Evaluation and robustness

- **Metrics:** MAE, RMSE, Bland-Altman bias and limits of agreement for HR and RR; percent of windows retained after SQI gating; SpO2 error against a reference oximeter.
- **Stratification:** motion level, ambient light (rPPG), sensor placement, and skin tone.
- **Robustness checks:** deliberately degrade signals and check that SQI predicts error; state the reference device's own error.

## 7. Limitations and open questions

- Is EVAde SpO2 intended for medical use? That decides whether FDA and ISO 80601-2-61 apply or only the general-wellness path.
- What is the final FDA status of the pulse oximeter guidance, and what does ISO 80601-2-61:2026 require?
- Fairness methods for rPPG and rigorous camera-SpO2 evidence with tone stratification are not verified.
- Unverified items to chase: the PhysNet BMVC 2019 record; "Empaviot" (not found); 2020+ TUSZ benchmark papers; ECG denoising reviews.
