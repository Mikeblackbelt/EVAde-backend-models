# Data Acquisition Plan by Modality

**Status: plan only. Nothing has been downloaded, requested, or tested. Section 0 is the 2-day hackathon scope; the rest assumes time for credentialing and requests.** Any public research data is acceptable; NASA data is not required. TUSZ and the other Temple University corpora are **not** used in this plan; seizure work uses Siena, CHB-MIT, and SeizeIT2 instead. Per-source detail and caveats are in [data-collection-sources](../data-collection-sources/README.md). Status labels: VERIFIED (access page or record checked this session or earlier), PARTIAL (search summary only), UNVERIFIED (not checked; confirm before relying on it).

## 0. Hackathon scope (2 days, no credentials, no approval waits)

The rest of this file is the full plan. For a 2-day hackathon, use **only** sources you can download today without an account approval, and treat everything else as post-hackathon. Anything that needs a request, a credential, a data use agreement, or an application is out of scope.

### Use these (instant or self-serve)

| Modality | Dataset | Why | Check before relying on it |
|---|---|---|---|
| ECG | MIT-BIH Arrhythmia (PhysioNet, open) | QRS and beat baseline; small, fast | none; open, ODC Attribution |
| ECG | PTB-XL (PhysioNet, open, CC BY 4.0) | rhythm classes; large enough for a 1D ResNet | file size before download |
| PPG | BIDMC (PhysioNet, open) | HR and respiration with references | none |
| rPPG | UBFC-rPPG (direct download, no formal license) | optional demo only | license unclear; fine for a hackathon demo, not for redistribution |
| EEG seizure | Siena Scalp EEG (PhysioNet, open, CC BY 4.0) | adult seizures with ECG channels | only 14 subjects, so test with leave-one-subject-out |
| EEG seizure | CHB-MIT (PhysioNet) | more seizures; pediatric | confirm it is open access on its page |
| EEG seizure (optional) | SeizeIT2 (OpenNeuro, open BIDS) | wearable-style test set | size; license page |
| EEG sleep or workload | Sleep-EDFx, EEG During Mental Arithmetic (PhysioNet, open) | drowsiness and workload features | none |
| Stress and behavior | WESAD (direct download from the Siegen or UCI page) | stress label with ECG, EDA, accelerometer | 15 subjects only; non-commercial terms |
| Radiation | MSL RAD and LRO CRaTER (NASA PDS, public); ISS series via OSDR RadLab (public GUI and API) | replay real dose-rate shapes through the radiation formula | file formats (PDS3 and PDS4); pick one small series |
| Bone | NHANES DXA (CDC, public, no application) | reference BMD by age and sex | cycle file names; femur and spine in 2007-2010, 2013-2014, 2017-2020 |

### Do not plan on these during the hackathon

GLOBEM and any other PhysioNet **credentialed** set, NSRR (SHHS, MrOS sleep), MrOS and SOF, UK Biobank, NASA LSDA and LSAH, DAIC-WOZ, DEAP (EULA with institutional sign-off), MODMA (account and conditions unconfirmed), and any analog-mission data. Treat their absence as a stated limitation, not a blocker.

### How each endpoint is covered without restricted data

| Endpoint or domain | Hackathon approach |
|---|---|
| `/predict/ecg` | train or fine-tune on PTB-XL; QRS and HRV with MIT-BIH; test on a held-out set of subjects |
| `/predict/eeg` | seizure flag from Siena plus CHB-MIT with leave-one-subject-out; if time is short, ship a features-plus-gradient-boosting model and skip deep nets |
| `/predict/radiation-risk` | rule-based formula; replay one MSL RAD or ISS series plus synthetic solar particle event profiles as tests |
| Behavioral domain | no depression or confinement data is available instantly. Use WESAD for a stress feature demo, and generate **synthetic** per-person trajectories (sleep, HRV, mood score) from the planning anchors to exercise the pipeline. Label it clearly as synthetic |
| Bone domain | the prior from the literature (about 1 percent per month areal loss) with a user-entered scan; NHANES only for the reference table |
| `/predict/risk` | the formulas in [risk-score.md](risk-score.md) and [max-days-to-return.md](max-days-to-return.md), driven by the worked examples and synthetic scenarios, since no outcome data exists |

### 48-hour schedule (suggested)

| Time | Task |
|---|---|
| Hours 0-2 | Download MIT-BIH, PTB-XL (or a subset), Siena, CHB-MIT subset, WESAD; write the manifest and subject-wise splits |
| Hours 2-10 | Preprocessing for ECG and EEG; baseline models (rules for ECG, features plus trees for EEG) |
| Hours 10-18 | FastAPI skeleton: `/health`, `/models`, schemas, CORS; radiation formula and replay test |
| Hours 18-28 | `/predict/ecg` and `/predict/eeg` on baselines; `/predict/risk` from the formulas |
| Hours 28-40 | Synthetic behavioral and bone inputs; integrate with the frontend and cloud layer; tests for the worked examples |
| Hours 40-48 | One stronger model only if baselines are done (for example a 1D ResNet on PTB-XL); a demo script; write the limitations slide |

### What to say in the demo

State plainly: trained on open terrestrial datasets, no spaceflight data, behavioral and bone inputs are simulated, and outputs are flags for human review, not diagnoses. This is accurate and judges tend to prefer it over an overclaimed number.

### Reduced task list

| # | Action | Time |
|---|---|---|
| 1 | Download the open sets above; save checksums | 1-2 h |
| 2 | Record each license in a one-page register | 30 min |
| 3 | Fix subject-wise splits; write the loaders | 2 h |
| 4 | Pick one MSL RAD or ISS dose series for the radiation test | 1 h |
| 5 | Write the synthetic behavioral and bone generators | 2 h |
| 6 | Record the credentialed sources as "future work" in the README | 15 min |

Anything below this point in the file is the full plan for after the hackathon.

## Full plan (post-hackathon)

## 1. Summary table

| Modality (endpoint) | Primary open sources | Role | Access effort | Biggest gap |
|---|---|---|---|---|
| ECG (`/predict/ecg`) | MIT-BIH, PTB-XL, PhysioNet/CinC 2020-2021 collections, SHHS overnight ECG (NSRR) | QRS and rhythm baselines, pretraining, HRV in sleep | open (PhysioNet) or 1-2 week request (NSRR) | single-lead wearable and in-motion ECG; spaceflight ECG |
| PPG and rPPG (vitals) | BIDMC, UBFC-rPPG, MMPD, SCAMPS (synthetic), VIPL-HR, PURE | HR and respiration baselines; skin-tone stress tests | open or contact authors | tone-balanced real data; SpO2 reference data |
| EEG (`/predict/eeg`) | Siena Scalp EEG, CHB-MIT, SeizeIT2 (seizure); Sleep-EDFx, SHHS/MrOS sleep (NSRR) (sleep, drowsiness); EEG During Mental Arithmetic (workload) | seizure, sleep, and workload flags | open (PhysioNet, OpenNeuro) | wearable EEG with fatigue labels; adult long-term seizure data (Siena has 14 subjects) |
| Radiation (`/predict/radiation-risk`) | MSL RAD, LRO CRaTER (PDS); ISS data via OSDR RadLab; published DOSIS tables; NOAA space-weather flux (UNVERIFIED) | exposure and dose-rate inputs, event shapes | public (PDS, OSDR) | labeled SEU-in-electronics data; no body-worn spaceflight dose data |
| Bone (bone domain) | NHANES DXA, MrOS and SOF (request), UK Biobank (application), OSDR rodent studies | reference distributions, bone-loss trend models | open (NHANES) or request | human spaceflight bone data (needs LSDA); no portable proxy |
| Behavioral | GLOBEM, WESAD, MODMA, DEAP, EEG During Mental Arithmetic, SHHS and MrOS sleep | per-person baselines, trend and mood features, affect checks | credentialed or EULA | confinement-specific data (Mars-500 and analogs not found public) |
| Aggregate (`/predict/risk`) | none directly; built from the domain outputs and simulation | formulas and thresholds | none | no outcome labels for spaceflight; needs expert elicitation and a pilot |

## 2. Per-modality plan

### ECG
- **Use first:** MIT-BIH (48 recordings, open, ODC Attribution; VERIFIED in this review) for QRS detection; PTB-XL (CC BY 4.0, 12-lead; VERIFIED) for rhythm models; PhysioNet/CinC 2020 and 2021 collections for multi-site external validation (VERIFIED, terms per source).
- **For HRV and sleep:** SHHS overnight ECG via NSRR (data access request, committee reply in about 1-2 weeks, signed agreement; PARTIAL, from NSRR staff posts). Overnight bipolar ECG at 125 Hz (SHHS-1) or 250 Hz (SHHS-2), so resample to the pipeline's 250 Hz or keep native rate for HRV.
- **Stress and affect labels:** WESAD (15 subjects, non-commercial terms).
- **Do:** download open sets, record checksums and licenses, build a lead-and-rate harmonization step (12-lead to single lead; 125, 250, 360, 500 Hz to common), define subject-wise splits. Submit the NSRR request early.
- **Does not exist publicly (to my knowledge):** spaceflight ECG at scale. Plan to test transfer on analog or pilot data only.

### PPG and rPPG
- **Use first:** BIDMC (53 ICU recordings with reference HR and respiration; open) for contact PPG; UBFC-rPPG and PURE for algorithm baselines; MMPD for the hard test set; SCAMPS for synthetic pretraining; VIPL-HR (check license).
- **Fairness:** public sets under-represent dark skin tones (Bondarenko et al. 2025 audit). Plan a small self-collected, objectively tone-measured validation set (needs ethics approval).
- **Do:** confirm licenses for UBFC, PURE, MMPD, VIPL-HR (contact maintainers where no license is stated); no public wearable SpO2 reference set has been identified in this review (gap).

### EEG
- **Seizure (replaces TUSZ):**
  - **Siena Scalp EEG** (PhysioNet; 14 subjects, 47 seizures, about 128 hours, 512 Hz, 10-20 layout with ECG channels; CC BY 4.0, open access; VERIFIED via the PhysioNet listing). Small, so use it for evaluation and leave-one-subject-out, not pretraining.
  - **CHB-MIT** (PhysioNet; pediatric; check license).
  - **SeizeIT2** (OpenNeuro, BIDS; behind-the-ear EEG with ECG, EMG, motion; 125 patients, 883 focal seizures; license not checked). Best match to a wearable system, so use it as the external test set.
  - Plan to train on CHB-MIT plus Siena and test on SeizeIT2, with subject-wise splits and event-based scoring (SzCORE).
- **Sleep and drowsiness:** Sleep-EDFx (open, ODC Attribution; VERIFIED) and SHHS or MrOS sleep through NSRR.
- **Workload and stress:** EEG During Mental Arithmetic Tasks (PhysioNet; 36 subjects, 23 channels, resting and task recordings; ODC Attribution, one mirror lists PDDL, so check the PhysioNet page; VERIFIED via search listing).
- **Affect and depression (shadow mode only):** MODMA (UK Data Service, DOI 10.5255/UKDA-SN-854301; 128-channel EEG for 53 subjects and 3-electrode wearable EEG for 55; license and download conditions not confirmed; PARTIAL) and DEAP (EULA, academic). Use only for shadow-mode checks, because the large meta-analyses show weak EEG mood markers (see [behavioral-health/eeg.md](../behavioral-health/eeg.md)).
- **Do:** build a channel-mapping step to a fixed 10-20 subset and a wearable subset; decide on the sampling rate (250 Hz) and resample.

### Radiation
- **Use first:** MSL RAD (PDS, public; Mars surface), LRO CRaTER (PDS; check restrictions), OSDR RadLab for ISS detectors (public GUI and API; bulk terms unconfirmed), and published DOSIS tables for benchmarking.
- **Space weather:** NOAA GOES solar energetic particle flux archives are the standard public source for solar particle event timing (UNVERIFIED here; confirm the archive and terms).
- **Reality check:** the radiation endpoint is rule-based, so the data are for **testing the formulas** (dose rate, cumulative dose, event shape), not for training. No labeled data on single-event upsets in wearable electronics is public. If SEU rates matter, the options are fault-injection simulation in software, or a beam-time proposal (HIMAC, NSRL, GSI) that needs an institution.
- **Do:** pull a few ISS and Mars series, store the calibration version, replay them through the formula as a regression test, and build synthetic solar particle event profiles for scenario tests.

### Bone
- **Use first:** NHANES DXA (public, no application; femur and spine in 2007-2010, 2013-2014, and 2017-March 2020; VERIFIED by cycle). Gives age and sex reference distributions.
- **Longitudinal trends:** MrOS (registered, data use agreement; has CT and HR-pQCT subsets) and SOF (request through NIA); UK Biobank imaging subset (application, fees; confirm).
- **Mechanism context:** OSDR rodent bone studies (public; not human).
- **Reality check:** human spaceflight bone data is mainly in LSDA (NASA request) and published papers. Without it, the bone domain stays a prior (about 1 percent per month areal loss from the literature) plus a user-entered scan, and is capped as the design already says.
- **Do:** register for MrOS and request SOF materials; confirm the UK Biobank cost tier; write the NHANES cycle merge as the reference table.

### Behavioral health
- **Use first:** GLOBEM (PhysioNet credentialed access; feature-level passive sensing, 497 users, four years; PARTIAL on timing) for cross-year generalization and trend-model testing; WESAD for stress; sleep data from NSRR for sleep features.
- **Questionnaire labels:** GLOBEM includes surveys; terms for embedding PHQ-9 and GAD-7 text must be confirmed.
- **Confinement:** Mars-500, NEK, LunAres, and HI-SEAS data were not found public. Ask the authors (Basner and colleagues) and the sponsoring institutes; this is the most important unmet need.
- **Do:** complete the PhysioNet credentialing, apply for GLOBEM, define the shared behavioral feature set so GLOBEM features map to the planned features (sleep duration, activity, HRV), and write the analog-data request.

### Aggregate risk score and MDTR
- No public data has the outcome labels the risk score needs. Plan: (1) simulation of 2,000 crew-missions for behavior and alert rates, (2) expert elicitation of parameters, (3) a terrestrial analog pilot, per [calibration-and-validation.md](calibration-and-validation.md). Public datasets validate individual modules only.

## 3. What needs to be done (checklist)

| # | Action | Owner (suggested) | Lead time |
|---|---|---|---|
| 1 | Create PhysioNet account; start credentialing (needed for GLOBEM and other restricted sets) | data lead | days to weeks |
| 2 | Download open sets: MIT-BIH, PTB-XL, BIDMC, Siena, CHB-MIT, Sleep-EDFx, EEG During Mental Arithmetic, WESAD, UBFC-rPPG | data lead | days |
| 3 | Download SeizeIT2 from OpenNeuro; confirm its license | data lead | days |
| 4 | Submit NSRR requests (SHHS, MrOS sleep); state the purpose in the Specific Purpose field | data lead | 1-2 weeks reply |
| 5 | Register for MrOS Online; request SOF materials; check UK Biobank fees | data lead | weeks |
| 6 | Apply for GLOBEM, MODMA (UK Data Service account), DEAP EULA if needed | data lead | weeks |
| 7 | Submit NASA LSDA and OSDR controlled-access requests (human bone, behavioral, ECG or HRV, sleep) | project lead (NASA contact) | weeks to months |
| 8 | Contact Mars-500, NEK, LunAres, and HI-SEAS investigators about sharing | project lead | open-ended |
| 9 | Write a license register (source, license, date checked, restrictions, commercial use) | data lead | ongoing |
| 10 | Build the manifest with checksums, and the subject-wise split files, before any model work | model lead | with downloads |
| 11 | Write harmonization specs: sampling rates, leads and channels, units, timestamps | model lead | 1-2 weeks |
| 12 | Decide storage and compute location; most large sets need disk planning (sizes not checked) | infra lead | before downloads |
| 13 | Draft the IRB protocol for a small tone-balanced rPPG validation set and a wearable pilot | project lead | weeks to months |
| 14 | Build radiation replay tests from MSL RAD, CRaTER, and OSDR series, plus synthetic solar particle events | model lead | 1-2 weeks |
| 15 | Confirm UNVERIFIED items: NOAA GOES archive terms, CHB-MIT and VIPL-HR licenses, SeizeIT2 license, MODMA conditions | data lead | days |

## 4. Order of work

1. **Week 1:** items 1, 2, 3, 9, 10. Start requests 4 to 8 in the same week because they have the longest lead times.
2. **Weeks 2-3:** harmonization (11), radiation replay (14), baseline models on open data.
3. **Later:** controlled data (NASA, MrOS, GLOBEM) as approvals arrive; the pilot (13) in parallel.

## 5. Risks

| Risk | Mitigation |
|---|---|
| Restricted data never arrives | the pipeline is built and tested on open data first; no module depends on a restricted set |
| Small open sets overfit (Siena has 14 subjects, WESAD 15) | leave-one-subject-out and report intervals; use them for testing, not pretraining |
| License conflicts (non-commercial, academic-only) | the license register; check before any shared or commercial use |
| Population mismatch (pediatric, older adults, students, ICU) | record demographics; state the training population; no generalization claims |
| Spaceflight performance unknown | no claim until analog or NASA data are tested |
| Privacy of restricted human data | follow each data use agreement; no raw data in the repository |
