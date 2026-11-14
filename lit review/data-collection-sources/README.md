# Data Collection Sources

Where training and validation data for EVAde can come from, with access terms and known limits. Licenses change, so check each source's page before use. Nothing here has been downloaded or tested.

| File | Covers |
|---|---|
| [physiological-signals.md](physiological-signals.md) | ECG, PPG, and rPPG datasets |
| [eeg.md](eeg.md) | EEG seizure, sleep, and stress datasets |
| [space-radiation.md](space-radiation.md) | Radiation measurements from spaceflight and planetary missions |
| [bone-and-spaceflight.md](bone-and-spaceflight.md) | Bone density data from spaceflight and older-adult cohorts |
| [additional-candidates.md](additional-candidates.md) | Further open datasets researched for gaps (stress wearables, sleep, EEG fatigue, BP and SpO2 PPG, large 12-lead ECG, solar particle flux), with a hackathon priority list |
| [behavioral-health.md](behavioral-health.md) | Passive sensing, interview, EEG-affect, and questionnaire sources for behavioral health |

## Rules for all sources

1. **Record the license and access terms** in the file when you first use a dataset. Revisit the terms before any commercial or shared use.
2. **Use subject-level splits.** Any dataset with repeated recordings per person needs train/test separation by person, or results will be inflated.
3. **Keep raw files read-only** and store a checksum with each. Do not commit raw data to this repository. Put only download scripts and a manifest here, once code exists.
4. **Note demographics.** Record age, sex, and skin tone where available, because they change how well the sensors work.

## Order of acquisition

Start with open datasets to build and test the pipeline (PhysioNet including Siena and CHB-MIT, SeizeIT2, WESAD; TUSZ is not used). Submit NASA OSDR and LSDA (Human Research Program) requests early because approval takes time; real astronaut EEG or ECG streams are not public. See [planning/implementation-pipeline.md](../planning/implementation-pipeline.md).
