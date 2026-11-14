# Physiological Signal Datasets (ECG, PPG, rPPG)

## MIT-BIH Arrhythmia Database (PhysioNet)
- **What:** 48 half-hour, two-channel ambulatory ECG recordings from 47 subjects, with beat-level annotations.
- **Access:** open on PhysioNet. License: Open Data Commons Attribution License v1.0 (per the PhysioNet registry). Attribution required.
- **Use:** QRS detector benchmarking (see [ecg-applications](../ecg-applications/README.md)).
- **Limit:** old recordings (1970s-era acquisition) and a small number of subjects.

## PTB-XL (PhysioNet)
- **What:** about 21,800 clinical 12-lead ECGs from about 18,870 patients, 10 s each, with recommended train/test splits. Counts differ by version: the data paper (Wagner et al., 2020) reports 21,837 records and 18,885 patients; the PhysioNet v1.0.3 page lists 21,799 and 18,869.
- **Access:** open on PhysioNet. License: CC BY 4.0 (per the PhysioNet entry).
- **Use:** pretraining arrhythmia or rhythm models. Transfer to single-lead wearable data must be tested separately.
- **Limit:** clinical 12-lead, not wearable single-lead. Lead layout differs.

## BIDMC PPG and Respiration Dataset (PhysioNet)
- **What:** PPG, ECG, and respiration from ICU patients, with reference HR and RR.
- **Access:** open on PhysioNet. License: Open Data Commons Attribution License v1.0 (per the PhysioNet LICENSE file).
- **Contents:** 53 recordings of PPG and impedance respiration from ICU patients at BIDMC, 125 Hz, with reference HR and RR. Cite Pimentel et al., IEEE TBME 64(8), 2016 (not re-checked).
- **Use:** a contact-PPG baseline with reference respiration.

## WESAD (UCI Machine Learning Repository, ID 465)
- **What:** multimodal wrist (Empatica E4) and chest (RespiBAN) recordings from 15 subjects during a lab stress protocol, with self-reports. Includes ECG, EDA, EMG, respiration, temperature, and accelerometer.
- **Access:** non-commercial scientific use with attribution, per the Siegen dataset disclaimer. The UCI page (ID 465) lists no license of its own and refers users to the original site's terms. Download from the Siegen page; cite Schmidt et al. (2018, ICMI). The UCI DOI (10.24432/C57K5T) was not re-checked.
- **Use:** stress and affect benchmarks for the HRV pipeline. Only 15 subjects, so use subject-wise cross-validation.
- **Limit:** lab setting, so it does not reflect real-world motion.

## UBFC-rPPG
- **What:** video of faces with reference pulse-oximeter ground truth, for rPPG. Two sub-datasets are described across sources.
- **Access:** download from the dataset's Google Sites page at the University of Burgundy. The page links to MATLAB and Python code for reading the ground truth. No explicit license was found; contact the dataset maintainers (listed on the page) before any shared use.
- **Use:** rPPG algorithm benchmarking, alongside the POS baseline.
- **Limit:** small, controlled lighting, limited skin-tone diversity. Check which subjects are included.

## PURE (rPPG)
- **Status:** a 10-subject rPPG dataset. No access terms were found; verify before use.

## PhysioNet/CinC Challenge data (2020 and 2021)
- **What:** multi-source 12-lead ECG collections used in the challenges (66,361 recordings from six hospital systems in 2020; 131,149 recordings from ten sources in the 2021 follow-up paper). Hidden test sets were not released.
- **Use:** multi-institution training and external-validation design, and reduced-lead experiments. **Limit:** hospital 12-lead data, not wearable single-lead.
- **Access:** public training data on PhysioNet; check each source's terms.

## rPPG datasets (verified records, access terms not checked)
- **MMPD** (Tang et al., EMBC 2023): phone video with skin-tone, motion, and lighting conditions; the recommended hard test set.
- **SCAMPS** (McDuff et al., NeurIPS 2022): synthetic videos for pretraining; validate on real data.
- **VIPL-HR** (Niu et al., ACCV 2018): large less-constrained set; check the license.
- A 2025 audit (Bondarenko et al., *npj Digit Med*, 8, 593) found darker skin tones and gender balance underrepresented across public rPPG datasets, which is the case for a self-collected, tone-balanced validation set.

## Other to consider (not yet verified)
- **rPPG-Toolbox** (GitHub, UW ubicomplab): a framework with dataset loaders. Useful to learn the expected data layout. Check its license.
- **MAHNOB-HCI**: multimodal affect dataset with ECG and physiological signals. Verify access terms.
