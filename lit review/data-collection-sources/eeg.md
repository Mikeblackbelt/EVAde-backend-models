# EEG Datasets

## TUH EEG Seizure Corpus (TUSZ) and TUH EEG Corpus
- **What:** clinical EEG from Temple University Hospital, with seizure annotations for TUSZ. Obeid & Picone (2016) describe the corpus. Recent descriptions cite over 30,000 EEGs from 2002 to present, but counts differ by version, so check the release notes.
- **Access:** registration at nedcdata.org. Terms restrict use to research and require the data-use agreement. Confirm the current agreement before use.
- **Use:** the main seizure-detection benchmark for this lab's work.
- **Limit:** clinical setting with many electrodes and montage variation. A wearable system will have fewer channels, so a channel-mapping step is required.

## CHB-MIT Scalp EEG Database (PhysioNet)
- **What:** pediatric scalp EEG with seizure annotations, from the Boston Children's Hospital group. Used in Shoeb & Guttag (2010).
- **Access:** open on PhysioNet. Check the license on the dataset page (it is open access; attribution is expected).
- **Use:** an easier, open alternative for seizure detection experiments.
- **Limit:** pediatric population. Results may not transfer to adults.

## Sleep-EDF (PhysioNet)
- **What:** polysomnography with sleep stage labels.
- **Access:** open on PhysioNet. Check the license on the dataset page.
- **Use:** sleep staging, if that feature is added later.
- **Limit:** not verified in this pass; confirm the current version on PhysioNet.

## WESAD (EEG not included)
- WESAD has no EEG, but it is the stress benchmark used in the ECG section. See [physiological-signals.md](physiological-signals.md).
