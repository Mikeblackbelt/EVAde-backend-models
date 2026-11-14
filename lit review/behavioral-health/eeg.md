# EEG for Behavioral Health

Status labels: **VERIFIED** = bibliographic record and abstract checked; **PARTIAL** = record checked, abstract not read. Full texts were not reviewed. Numbers are from abstracts.

## 1. Summary

EEG is a legitimate tool for **sleep architecture** and for **tracking drowsiness and workload**, and an unreliable one for **diagnosing mood or anxiety disorders**. The largest meta-analyses find the most-studied depression marker (frontal alpha asymmetry) to be negligible or small. Connectivity findings are inconsistent in direction. Deep-learning papers report 76-100% accuracy for depression detection, but those numbers cluster in small samples with subject-dependent validation. For Blindsight, EEG should be an **optional, pre-registered** channel, used first for sleep, with mood claims withheld until a controlled pilot supports them.

## 2. Resting-state EEG markers of depression and anxiety

- **van der Vinne et al. (2017)** Frontal alpha asymmetry as a diagnostic marker in depression: fact or fiction? *NeuroImage: Clinical*, 16, 79-87. DOI 10.1016/j.nicl.2017.07.006. VERIFIED. 16 studies, 1,883 patients and 2,161 controls; after excluding heterogeneous studies the effect was d = -0.007 (CI -0.090 to 0.075), negligible. A sex-by-age-by-severity interaction was found and replicated in an independent sample. (Older than 2019, kept as the key negative result.)
- **Luo, Tang & Fan (2025)** Meta-analysis of resting frontal alpha asymmetry as a biomarker of depression. *npj Mental Health Research*, 4(1), 2. DOI 10.1038/s44184-025-00117-x. VERIFIED. 23 studies, 1,928 patients and 2,604 controls; small but significant effect, high heterogeneity, no moderator explained it; authors conclude limited stand-alone diagnostic value.
- **de Aguiar Neto & Rosa (2019)** Depression biomarkers using non-invasive EEG: a review. *Neurosci Biobehav Rev*, 105, 83-93. DOI 10.1016/j.neubiorev.2019.07.021. VERIFIED. Mapping review of 2014-2018 studies (more than 250 screened, 42 discussed); the depressed brain appeared to have a more random network structure. No pooled effect sizes.
- **Miljevic et al. (2023)** Alterations in EEG functional connectivity in individuals with depression: a systematic review. *J Affect Disord*, 328, 287-302. DOI 10.1016/j.jad.2023.01.126. VERIFIED. Consistent differences for delta and gamma; for alpha, theta, and beta the direction was inconsistent.
- **Zhai et al. (2025)** A systematic review of EEG-based biomarkers for depression, anxiety, and bipolar disorder: trends in explainable AI. *BMC Psychiatry*, 26(1), 90. DOI 10.1186/s12888-025-07589-3. VERIFIED. 145 studies (2015-2025), accuracies 75-99%, about 65% at high risk of bias. The review includes only studies using explainable-AI methods, and some highlighted features reproduce the weak asymmetry finding, so agreement is not independent evidence.
- **Newson & Thiagarajan (2019)**, *Frontiers in Human Neuroscience*, 12, 521, DOI 10.3389/fnhum.2018.00521: PARTIAL (record only). **Arns et al. (2015)** iSPOT-D abstract, *Brain Stimulation*, 8(2), 337, DOI 10.1016/j.brs.2015.01.090: PARTIAL (conference-type abstract; no claim made from it).

**Anxiety:** no high-quality anxiety-only EEG meta-analysis was found; treat anxiety-specific EEG claims as low-evidence.

## 3. Deep learning and the leakage problem

- **Brookshire et al. (2024)** Data leakage in deep learning studies of translational EEG. *Frontiers in Neuroscience*, 18, 1373515. DOI 10.3389/fnins.2024.1373515. VERIFIED. Segment-based holdout "strongly overestimated" performance on unseen subjects versus subject-based holdout (tested on Alzheimer's and seizure data); a literature survey found most translational deep-learning EEG studies use segment-based holdout.
- **Luna-Guevara et al. (2026)** Systematic review of ML and DL models for EEG-based detection of depression. *J Psychiatr Res*, 199, 113-121. DOI 10.1016/j.jpsychires.2026.04.030. VERIFIED. 42 studies (2020-2024); accuracy about 76-100%; mean 93.92% for DL vs 90.78% for ML (not significant); near-perfect results clustered in small samples with subject-dependent or internal-only validation.
- **Vos et al. (2025)** Stress monitoring using low-cost EEG devices: a systematic literature review. *Int J Med Inform*, 198, 105859. DOI 10.1016/j.ijmedinf.2025.105859. VERIFIED. 69 articles; 54% had pre-experiment health screening; 58% were low-powered; few used independent validation or cortisol as a reference.
- **Shim, Hwang & Lee (2023)** Toward practical ML-based diagnosis for drug-naive women with MDD using EEG channel reduction. *J Affect Disord*, 338, 199-206. DOI 10.1016/j.jad.2023.06.007. VERIFIED. 49 patients and 49 controls; best features gave 83.67% accuracy, AUC 0.92 (leave-one-out, single site); accuracy stayed above 80% down to 19 channels. Women only, no external validation.
- **Roy et al. (2019)** *J Neural Eng*, 16(5), 051001, DOI 10.1088/1741-2552/ab260c, and **Craik et al. (2019)**, 16(3), 031001, DOI 10.1088/1741-2552/ab0ab5: PARTIAL (records only). General deep-learning-for-EEG reviews.

## 4. Fatigue, drowsiness, vigilance, and workload

- **Stancin, Cifrek & Jovic (2021)** EEG features in driver drowsiness detection. *Sensors*, 21(11), 3786. DOI 10.3390/s21113786. VERIFIED. Review; driving context.
- **Hassan et al. (2025)** EEG-based driver fatigue detection: a survey. *Cognitive Neurodynamics*, 19(1), 142. DOI 10.1007/s11571-025-10320-3. VERIFIED. PRISMA review; 87 of 267 publications analysed in depth.
- **Demirezen, Temizel & Brouwer (2024)** Reproducible machine learning research in mental workload classification using EEG. *Frontiers in Neuroergonomics*, 5, 1346794. DOI 10.3389/fnrgo.2024.1346794. VERIFIED. Reporting guidelines; main gaps were unseen-test-data reporting, open data and code. Check the second author's surname spelling against the journal page.
- **Pusica et al. (2025)** ML performance in EEG-based mental workload classification across task types. *Frontiers in Neuroergonomics*, 6, 1621309. DOI 10.3389/fnrgo.2025.1621309. VERIFIED. No standard benchmark; accuracy dropped in multitasking designs.
- **Morrone et al. (2025)** Neural indicators of sleep loss and sleep propensity in male military trainees: dry-electrode EEG, an exploratory study. *Physiological Reports*, 13(7), e70301. DOI 10.14814/phy2.70301. VERIFIED. n = 7; sleep 5.4 vs 7.4 h; theta and alpha rose during PVT (d about 0.4, uncorrected p-values). Direction supported; predictive validity not shown.
- **Maccora, Manousakis & Anderson (2019)** Pupillary instability as an accurate, objective marker of alertness failure and performance impairment. *J Sleep Res*, 28(2), e12739. DOI 10.1111/jsr.12739. VERIFIED. n = 18, 40 hours awake; pupillary instability classified PVT lapses and microsleeps with AUC above 0.78. A non-EEG comparator, not an EEG marker.
- **Ladouce et al. (2025)** EEG-based assessment of long-term vigilance using frequency tagging. *J Neural Eng*, 22(3). DOI 10.1088/1741-2552/add771. PARTIAL. Frequency tagging needs an active stimulus, which conflicts with passive monitoring.

## 5. Wearable, dry, and in-ear EEG

- **Niso et al. (2023)** Wireless EEG: a survey of systems and studies. *NeuroImage*, 269, 119774. DOI 10.1016/j.neuroimage.2022.119774. VERIFIED. 48 wireless devices compared; 110 publications reviewed. Use for device selection.
- **Hinrichs et al. (2020)** Wireless dry vs conventional wet electrode EEG for clinical applications. *Scientific Reports*, 10, 5218. DOI 10.1038/s41598-020-62154-0. VERIFIED. 16 healthy and 16 patients; alpha and beta did not differ, theta and delta were slightly higher with dry electrodes. Drowsiness markers live in theta and delta, so this bias matters.
- **Moumane et al. (2024)** Signal quality evaluation of an in-ear EEG device compared with a conventional cap. *Frontiers in Neuroscience*, 18, 1441897. DOI 10.3389/fnins.2024.1441897. VERIFIED. n = 30; alpha and nap theta captured, lower SNR and intermittent movement artifacts.
- **Mikkelsen et al. (2019)** Accurate whole-night sleep monitoring with dry-contact ear-EEG. *Scientific Reports*, 9, 16824. DOI 10.1038/s41598-019-53115-3. VERIFIED. 80 recordings; automatic scoring kappa 0.73, close to manual scoring of scalp EEG. Healthy participants, single lab.
- **Haria et al. (2026)** In-ear EEG: a systematic review. *Frontiers in Neurology*, 17, 1914159. DOI 10.3389/fneur.2026.1914159. VERIFIED. 24 studies; validation still needed for signal quality, coverage, and motion.
- **Artifact handling:** Gorjan et al. (2022), *J Neural Eng*, 19(1), 011004, DOI 10.1088/1741-2552/ac542c (VERIFIED; best method depends on movement type and intensity); Ledwidge et al. (2025), *Sensors*, 25(15), 4810, DOI 10.3390/s25154810 (VERIFIED; iCanClean somewhat better than ASR during running); Chang et al. (2020), *IEEE TBME*, 67(4), 1114-1121, DOI 10.1109/TBME.2019.2930186 (PARTIAL); Pion-Tonachini et al. (2019) ICLabel, *NeuroImage*, 198, 181-197, DOI 10.1016/j.neuroimage.2019.05.026 (PARTIAL); Klug & Gramann (2021), *Eur J Neurosci*, 54(12), 8406-8420, DOI 10.1111/ejn.14992 (PARTIAL).
- **Gaps:** no verified review of deep-learning EEG denoising, and no head-to-head validation of consumer headbands against a clinical system.

## 6. Preprocessing and reproducibility standards

- **Pernet et al. (2020)** Issues and recommendations from the OHBM COBIDAS MEEG committee. *Nature Neuroscience*, 23(12), 1473-1483. DOI 10.1038/s41593-020-00709-0. PARTIAL. Core reporting guideline.
- **Pernet et al. (2019)** EEG-BIDS. *Scientific Data*, 6, 103. DOI 10.1038/s41597-019-0104-8. PARTIAL. **Appelhoff et al. (2019)** MNE-BIDS. *J Open Source Softw*, 4(44), 1896. DOI 10.21105/joss.01896. VERIFIED.
- **Bigdely-Shamlo et al. (2015)** The PREP pipeline. *Frontiers in Neuroinformatics*, 9, 16. DOI 10.3389/fninf.2015.00016. VERIFIED (older, standard citation). **Gramfort et al. (2013)** MEG and EEG data analysis with MNE-Python. *Frontiers in Neuroscience*, 7, 267. DOI 10.3389/fnins.2013.00267. VERIFIED (older).
- **Delorme & Makeig (2004)** EEGLAB. *J Neurosci Methods*, 134(1), 9-21. DOI 10.1016/j.jneumeth.2003.10.009. VERIFIED. Older, widely used.
- Makoto's preprocessing pipeline is a wiki page and is **not a peer-reviewed source**; cite only as a practitioner guide.

## 7. Foundation models

- **Kuruppu et al. (2026)** EEG foundation models: a critical review of current progress and future directions. *J Neural Eng*, 23(2), 021001. DOI 10.1088/1741-2552/ae4455. VERIFIED. Ten early models; evaluations are "heterogeneous and largely limited", making off-the-shelf utility hard to assess.
- **LaBraM** (Jiang, Zhao & Lu, ICLR 2024, arXiv 2405.18765; verified by web search, about 2,500 hours from about 20 datasets); **BIOT** (Yang, Westover & Sun, NeurIPS 2023, pp. 78240-78260; PARTIAL); **EEGPT** (NeurIPS 2024, pp. 39249-39280; PARTIAL; the author names in the research notes came from memory and are unverified, so none are given here). None of these is a mental-health model.
- **Chen, Lu & Wu (2025)**, *IEEE Signal Processing Magazine*, 42(5), 36-57, DOI 10.1109/MSP.2025.3595320: PARTIAL.

**Use:** research baseline only, evaluated on subject-held-out data against a small supervised model.

## 8. EEG in spaceflight and isolation

- **Fiedler et al. (2023)** Noise characteristics in spaceflight multichannel EEG. *PLoS ONE*, 18(2), e0280822. DOI 10.1371/journal.pone.0280822. VERIFIED. ISS recordings had an order of magnitude lower noise than pre- and post-flight ground recordings with the same system. Noise only, no behavioral outcome.
- **Abeln et al. (2015)** Exercise in isolation: a countermeasure for electrocortical, mental and cognitive impairments. *PLoS ONE*, 10(5), e0126356. DOI 10.1371/journal.pone.0126356. VERIFIED. n = 8 men at Concordia; active people showed decreasing alpha and beta and steady mood, inactive people the opposite. Post hoc grouping; motivates rather than validates.
- **Gemignani et al. (2014)** How stressful are 105 days of isolation? Sleep EEG patterns and tonic cortisol. *Int J Psychophysiol*, 93(2), 211-219. DOI 10.1016/j.ijpsycho.2014.04.008. VERIFIED. n = 6; no cognitive or emotional alteration; higher cortisol linked to shorter sleep. Published 2014.
- **Kovrov et al. (2018)** *Bull Exp Biol Med*, 165(2), 177-180, DOI 10.1007/s10517-018-4124-8 (VERIFIED). Three Mars-500 participants; alpha peak frequency before sleep onset related to sleep-onset duration.

**Gap:** no verified EEG study of astronaut mood in orbit and no systematic review of EEG in spaceflight or isolation. The evidence is a handful of small studies (n = 3 to 8).

## 9. Seizure detection (for hardware overlap)

See [eeg-applications](../eeg-applications/README.md).

## 10. Implementation steps

1. Define the claim narrowly: within-person state tracking (sleep, drowsiness, workload), not diagnosing depression or anxiety.
2. Treat frontal alpha asymmetry as an exploratory feature, stratified by sex and age.
3. Require subject-wise and session-wise held-out evaluation; report segment-split results only as a labeled leakage demonstration.
4. Pre-register the feature set, montage, reference, metric, and artifact pipeline before collecting crew or analog data.
5. Choose the artifact method by activity level (rest vs exercise) and log all parameters.
6. Store data in EEG-BIDS using MNE-BIDS and report against COBIDAS-MEEG.
7. Run a channel-count ablation (for example 62, 30, 19, and 10 channels) on your own data, with 19 as the starting reference.
8. Validate each wearable device against a wet clinical montage on your own protocol; expect inflated theta and delta with dry electrodes and lower alpha SNR in-ear.
9. Use ear-EEG first for sleep, where the evidence is strongest.
10. Pair EEG fatigue markers with the PVT (and optionally pupillometry) and test in a controlled sleep-restriction study before any in-habitat use.
11. Treat foundation models as a research baseline only.

## 11. Limitations and open questions

- Is there a large, pre-registered resting-EEG study for depression or anxiety with out-of-sample replication? Meta-analyses exist (n about 4,500) but no such study was located; the iSPOT-D EEG paper was not read.
- What does within-person EEG variation look like for mood in isolation? Evidence is n = 3 to 8.
- Does ear or dry-electrode theta and delta power track PVT lapses? No validation study found.
- Mood datasets such as DEAP and DREAMER use film clips in healthy volunteers; transfer to chronic states in crews is unproven.
- PARTIAL items (Roy 2019, Craik 2019, Chang 2020, ICLabel, Klug 2021, COBIDAS, EEG-BIDS, BIOT, EEGPT, Newson 2019) need abstract review before any number is quoted.
