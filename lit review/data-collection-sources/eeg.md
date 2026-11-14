# EEG Datasets

## TUH EEG Seizure Corpus (TUSZ) and TUH EEG Corpus
- **What:** clinical EEG from Temple University Hospital, with seizure annotations for TUSZ. Obeid & Picone (2016) describe the corpus. The total EEG count was **UNVERIFIABLE** in this review (searches found only per-version file counts, such as 34,698 files for v0.6.0). Check the NEDC release notes for the version in use.
- **Access:** registration at nedcdata.org. Terms restrict use to research and require the data-use agreement. Confirm the current agreement before use.
- **Use:** the main seizure-detection benchmark in the literature. **Project decision: not used here;** Siena, CHB-MIT, and SeizeIT2 are used instead (see [planning/data-acquisition-plan.md](../planning/data-acquisition-plan.md)).
- **Limit:** clinical setting with many electrodes and montage variation. A wearable system will have fewer channels, so a channel-mapping step is required.

## Siena Scalp EEG Database (PhysioNet)
- **What:** 14 adult patients (9 male, ages 25-71; 5 female, ages 20-58), 47 seizures over about 128 hours, 512 Hz, international 10-20 layout, with ECG channels (Detti 2020; *Processes* 2020 paper).
- **Access:** open access; Creative Commons Attribution 4.0 (VERIFIED via the PhysioNet listing).
- **Use:** adult seizure evaluation set and leave-one-subject-out tests. **Limit:** only 14 subjects.

## EEG During Mental Arithmetic Tasks (PhysioNet)
- **What:** 36 subjects, 23-channel 10/20 EEG, one resting and one serial-subtraction recording each, with good and bad counter groups (Zyma et al. 2019, *Data*, 4(1), 14, DOI 10.3390/data4010014).
- **Access:** open; Open Data Commons Attribution License v1.0 on PhysioNet (one mirror lists PDDL, so check the PhysioNet page).
- **Use:** workload and stress feature checks. **Limit:** lab task, small sample.

## SeizeIT2 (OpenNeuro, BIDS)
- **What:** wearable dataset of patients with focal epilepsy: over 11,000 hours of behind-the-ear EEG with ECG, EMG, and motion; 883 focal seizures from 125 patients at five European centers (Bhagubai et al., 2025, *Scientific Data*, 12, 1228).
- **Access:** published on OpenNeuro in BIDS format with a training and validation split and two benchmark approaches; license not checked.
- **Use:** the external wearable test set for any seizure model trained on TUSZ, and a multimodal (EEG plus ECG) example.

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
