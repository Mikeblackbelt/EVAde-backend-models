# Physiological Signal Datasets (ECG, PPG, rPPG)

## MIT-BIH Arrhythmia Database (PhysioNet)
- **What:** 48 half-hour, two-channel ambulatory ECG recordings from 47 subjects, with beat-level annotations.
- **Access:** open on PhysioNet. License: Open Data Commons Attribution License v1.0 (per the PhysioNet registry). Attribution required.
- **Use:** QRS detector benchmarking (see [ecg-applications](../ecg-applications/README.md)).
- **Limit:** old recordings (1970s-era acquisition) and a small number of subjects.

## PTB-XL (PhysioNet)
- **What:** 21,799 clinical 12-lead ECGs from 18,869 patients, 10 s each, with recommended train/test splits.
- **Access:** open on PhysioNet. License: CC BY 4.0 (per the PhysioNet entry).
- **Use:** pretraining arrhythmia or rhythm models. Transfer to single-lead wearable data must be tested separately.
- **Limit:** clinical 12-lead, not wearable single-lead. Lead layout differs.

## BIDMC PPG and Respiration Dataset (PhysioNet)
- **What:** PPG, ECG, and respiration from ICU patients, with reference HR and RR.
- **Access:** open on PhysioNet. Check the license on the dataset page before use.
- **Use:** a contact-PPG baseline with reference respiration. Not yet verified in this review; confirm contents on the page.

## WESAD (UCI Machine Learning Repository, ID 465)
- **What:** multimodal wrist (Empatica E4) and chest (RespiBAN) recordings from 15 subjects during a lab stress protocol, with self-reports. Includes ECG, EDA, EMG, respiration, temperature, and accelerometer.
- **Access:** free for non-commercial scientific use with attribution (per the original Siegen download page and UCI). Download from the Siegen page; cite Schmidt et al. (2018, ICMI) and the UCI DOI (10.24432/C57K5T).
- **Use:** stress and affect benchmarks for the HRV pipeline. Only 15 subjects, so use subject-wise cross-validation.
- **Limit:** lab setting, so it does not reflect real-world motion.

## UBFC-rPPG
- **What:** video of faces with reference pulse-oximeter ground truth, for rPPG. Two sub-datasets are described across sources.
- **Access:** download from the dataset's Google Sites page at the University of Burgundy. The page links to MATLAB and Python code for reading the ground truth. No explicit license was found; contact the dataset maintainers (listed on the page) before any shared use.
- **Use:** rPPG algorithm benchmarking, alongside the POS baseline.
- **Limit:** small, controlled lighting, limited skin-tone diversity. Check which subjects are included.

## PURE (rPPG)
- **Status:** not checked in this pass. Commonly cited rPPG dataset; verify its access terms before use.

## Other to consider (not yet verified)
- **rPPG-Toolbox** (GitHub, UW ubicomplab): a framework with dataset loaders. Useful to learn the expected data layout. Check its license.
- **MAHNOB-HCI**: multimodal affect dataset with ECG and physiological signals. Verify access terms.
