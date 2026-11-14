# Additional Dataset Candidates (researched 2026-11-14)

Open or self-serve datasets that could fill gaps in the current set (MIT-BIH, PTB-XL, BIDMC, Siena, CHB-MIT, Sleep-EDFx, EEG During Mental Arithmetic, NHANES). Nothing here has been downloaded. Status labels: **VERIFIED** (named on an official page or registry in a search result this session), **PARTIAL** (search summary only), **UNVERIFIED** (not confirmed; check before use). License strings come from search summaries of the official listings and must be re-read on the dataset page before use. Dataset sizes were not checked.

## Priority for the 2-day hackathon

| Priority | Dataset | Fills the gap | Why |
|---|---|---|---|
| 1 | Wearable Exam Stress (PhysioNet) | behavioral and stress baseline-vs-current on a wearable | real-world stress with HR, EDA, BVP, temperature, accelerometer; open |
| 2 | Wearable Device Dataset, induced stress and structured exercise (PhysioNet) | stress versus exercise confound | tests the "exercise context tag" design with labeled sessions |
| 3 | MMASH (PhysioNet) | sleep and 24 h heart rhythm | beat-to-beat data, accelerometer, and sleep quality for sleep and HRV baseline trends |
| 4 | Cao et al. 2019 driving EEG | EEG drowsiness and fatigue | 27 subjects, 32 channels, a sustained-attention task; the closest open set to "cognitive fatigue from EEG" |
| 5 | UCI Cuff-Less Blood Pressure Estimation | blood pressure from PPG and ECG | CC BY 4.0, simple files |
| 6 | BUT PPG | smartphone PPG with BP and SpO2 annotations | CC BY 4.0, mobile-camera PPG, fits the contactless goal |
| 7 | Chapman-Shaoxing "ECG Arrhythmia" (PhysioNet) | large 12-lead ECG set for external validation | 45,152 patients; check license and size |
| 8 | NOAA GOES proton flux (SWPC and NCEI) | radiation event timing | real solar particle event series for the radiation replay test |
| Later | OpenOximetry, ds003478, MODMA, PulseDB | SpO2 with skin tone; depression EEG; BP | license or access unconfirmed |

## Behavioral, stress, sleep, wearables

| Dataset | What | License and access | Status |
|---|---|---|---|
| **Wearable Exam Stress** (PhysioNet `wearable-exam-stress`) | students during exams: electrodermal activity, heart rate, blood volume pulse, skin temperature, accelerometer | Open Data Commons Attribution License v1.0; open access | VERIFIED (listing); contents beyond the summary not read |
| **Wearable Device Dataset from Induced Stress and Structured Exercise Sessions** (PhysioNet `wearable-device-dataset`, v1.0.1) | Empatica E4 data from stress induction (36 healthy volunteers) and aerobic and anaerobic exercise | Open Data Commons Attribution License v1.0 | VERIFIED (listing) |
| **MMASH** (PhysioNet `mmash`) | 24 h beat-to-beat heart data, accelerometer, sleep quality, activity in healthy people | Open Database License (ODbL) v1.0 per the data descriptor (share-alike terms, check implications) | VERIFIED (summary) |
| **Non-EEG dataset for neurological status** (PhysioNet `noneeg`) | physiological non-EEG signals | license not confirmed | UNVERIFIED |
| **WESAD** | already in the plan; wrist and chest sensors, 15 subjects | Siegen page: scientific, non-commercial use with credit; UCI lists no formal license; about 2.5 GB download; DOI 10.24432/C57K5T | VERIFIED (terms) |
| SWELL-KW, Emotion in Motion, LifeSnaps | other stress and lifestyle sets | not found in this research | UNVERIFIED |

Fit with the project: the exam-stress and induced-stress sets give per-person recordings across conditions, which is what the baseline-comparison design needs (a person's baseline, then a stressed period). They do not contain depression or confinement labels, so they support stress and context handling, not mood claims.

## EEG

| Dataset | What | License and access | Status |
|---|---|---|---|
| **Multi-channel EEG during a sustained-attention driving task** (Cao et al. 2019, *Scientific Data*, DOI 10.1038/s41597-019-0027-4; University of Tasmania repository) | 62 sessions, 32-channel EEG, 27 subjects, simulated highway driving with lane-deviation events used for drowsiness and reaction time | CC BY 4.0 per the repository listing | VERIFIED (listing and paper record) |
| EEG driver drowsiness dataset (figshare 14273687) | 11 subjects, alert versus drowsy, extracted by an uploader from the Cao data | license not shown; use the original instead | PARTIAL |
| **OpenNeuro ds003478** | resting EEG, 64 channels at 500 Hz, 122 participants (76 controls, 23 subclinical depression, 23 major depressive disorder), eyes open and closed, with Beck Depression Inventory and STAI scores | OpenNeuro license not confirmed (CC0 was not verified) | PARTIAL |
| **MODMA** (UK Data Service, DOI 10.5255/UKDA-SN-854301) | 128-channel EEG in 53 participants and 3-electrode wearable EEG in 55, depression versus controls, plus audio | registration and conditions not confirmed | PARTIAL |

Fit: the driving data provide fatigue labels for the EEG drowsiness flag. The depression sets would only feed the shadow-mode affect markers (the large meta-analyses show weak EEG mood markers). A 3-electrode wearable EEG in MODMA is closer to consumer headbands than clinical sets.

## ECG

| Dataset | What | License and access | Status |
|---|---|---|---|
| **A large scale 12-lead ECG database for arrhythmia study** (PhysioNet `ecg-arrhythmia`, v1.0.0, DOI 10.13026/wgex-er52) | 12-lead ECGs of 45,152 patients at 500 Hz (Chapman and Shaoxing hospitals) | PhysioNet open access; exact license not confirmed | PARTIAL |
| **Georgia 12-lead ECG database** (inside PhysioNet Challenge 2020) | 10,344 ECGs of 10 s at 500 Hz, Southeastern US population | follows the Challenge 2020 page terms | PARTIAL |

Fit: more patients and a second population for external validation of the PTB-XL rhythm model. Both are hospital 12-lead data, not wearable single-lead.

## PPG, SpO2, blood pressure

| Dataset | What | License and access | Status |
|---|---|---|---|
| **BUT PPG** (Brno University of Technology, hosted via PhysioNet and the AWS registry) | smartphone PPG with one BP and one SpO2 annotation per signal, accelerometer; extended to about 50 subjects | CC BY 4.0 per the AWS registry | VERIFIED (registry) |
| **UCI Cuff-Less Blood Pressure Estimation** (UCI dataset 340) | fingertip PPG, arterial blood pressure, ECG at 125 Hz, derived from MIMIC-II | CC BY 4.0 | VERIFIED (UCI page) |
| **PulseDB** | cleaned ECG, PPG, and arterial BP from MIMIC-III and VitalDB | license not checked; source data may carry their own terms | PARTIAL |
| **OpenOximetry repository** (*Scientific Data*, 2025) | controlled desaturation studies with arterial blood gas SpO2, raw and processed PPG, and skin color measurements: the best match for the pulse-oximetry skin-tone bias question | license not confirmed | PARTIAL |
| PhysioNet Pulse Transit Time PPG dataset | PPG, inertial, pressure, ECG, 22 healthy subjects | open access; license and contents not checked | PARTIAL |
| ENCoDE (PhysioNet) | skin tone with clinical data from acute care patients | credentialed access | excluded (needs credentials) |

Fit: BUT PPG and the UCI set give BP and SpO2 targets for the vitals endpoint; OpenOximetry is the right set for a skin-tone fairness test once its terms are confirmed. MIMIC-derived data are ICU patients, not healthy wearers.

## Radiation and spaceflight

| Dataset | What | License and access | Status |
|---|---|---|---|
| **NOAA GOES proton and particle flux** (SWPC real-time JSON at services.swpc.noaa.gov/json/goes/; NCEI archive from 1974; NCEI list of solar proton events from 1976) | real solar energetic particle series and event dates | no explicit terms found; US government data, but check the "use constraints" on each NCEI dataset page or contact NCEI | PARTIAL (locations confirmed, terms not found) |
| **NASA OSDR Inspiration4 and SOMA** | processed commercial astronaut data (omics, biospecimens, cabin microbiome), public; raw data needs an application and board approval | processed data public | PARTIAL: no wearable heart rate, sleep, or bone data was identified in these results |

Fit: GOES gives real solar particle event shapes for the radiation replay and lead-time tests. Inspiration4 and SOMA are biology and context data; they do not feed the ECG, EEG, or bone models.

## What this adds to the plan

| Project gap | Candidate |
|---|---|
| Per-person stress and context baselines on a wearable | Wearable Exam Stress; Induced Stress and Exercise; WESAD |
| Sleep and 24 h HRV trends | MMASH; Sleep-EDFx |
| EEG fatigue labels | Cao 2019 driving EEG |
| Blood pressure and SpO2 targets | UCI cuff-less BP; BUT PPG; OpenOximetry (later) |
| Larger ECG external validation | Chapman-Shaoxing; Georgia |
| Solar particle event series | NOAA GOES proton flux |
| Still missing | spaceflight ECG and EEG at scale; confinement behavior (Mars-500 and analogs); human spaceflight bone data; wearable SpO2 with skin tone, open and ready |

## Next steps

1. Confirm each license on the dataset page (the strings above come from search summaries) and record it in the license register.
2. Check file sizes against the server disk budget before download; do not download anything over about 5 GB without a subset plan.
3. For the top four, add loaders to `evade/preprocessing` and a smoke test, following the existing pattern.
4. Add the final choices to the [data acquisition plan](../planning/data-acquisition-plan.md) hackathon list.
