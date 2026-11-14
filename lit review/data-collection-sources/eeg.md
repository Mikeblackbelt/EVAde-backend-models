# EEG Datasets

## TUH EEG Seizure Corpus (TUSZ) and TUH EEG Corpus
- **What:** clinical EEG from Temple University Hospital, with seizure annotations for TUSZ. Obeid & Picone (2016) describe the corpus. The total EEG count was **UNVERIFIABLE** in this review (searches found only per-version file counts, such as 34,698 files for v0.6.0). Check the NEDC release notes for the version in use.
- **Access:** registration at nedcdata.org. Terms restrict use to research and require the data-use agreement. Confirm the current agreement before use.
- **Use:** the main seizure-detection benchmark for this lab's work.
- **Limit:** clinical setting with many electrodes and montage variation. A wearable system will have fewer channels, so a channel-mapping step is required.

## CHB-MIT Scalp EEG Database (PhysioNet)
- **What:** pediatric scalp EEG with seizure annotations, from the Boston Children's Hospital group. Used in Shoeb & Guttag (2010).
- **Access:** open on PhysioNet. Check the license on the dataset page (not verified in this pass).
- **Use:** an easier, open alternative for seizure detection experiments.
- **Limit:** pediatric population. Results may not transfer to adults.

## Sleep-EDF (PhysioNet)
- **What:** polysomnography with sleep stage labels.
- **Status:** the original Sleep-EDF is marked obsolete. Use **Sleep-EDFx** (DOI 10.13026/C2X676) for new work.
- **Access:** open on PhysioNet. License: ODC Attribution v1.0 (per the PhysioNet entry).
- **Use:** sleep staging, if that feature is added later.

## WESAD (EEG not included)
- WESAD has no EEG, but it is the stress benchmark used in the ECG section. See [physiological-signals.md](physiological-signals.md).
