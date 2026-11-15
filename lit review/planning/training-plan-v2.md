# Training Plan v2: Deep Models on the CUNY HPCC Cluster

**Status: plan only. Nothing here has been run on the cluster.** It covers the next training sessions now that GPU work moves from the shared lab server to the CUNY HPCC cluster (arrow, SLURM). Job scripts are in [evade/training/hpc/](../../evade/training/hpc/README.md). Every model below outputs **flags for human review, not diagnoses**. Numbers marked *estimate* are guesses, to be replaced by smoke-run timings. Numbers marked *design* are choices, not findings.

Evidence labels follow the repo rules:
- **VERIFIED:** checked at the source on 2026-11-14.
- **UNVERIFIED:** recalled or assumed; confirm before relying on it.

## 1. Why a v2

These beta results come from [evade/training/README.md](../../evade/training/README.md): three CPU models, registered `draft` as EVA-1 to EVA-6.

| Task | Beta result | Weakness this plan addresses |
|---|---|---|
| ECG, PTB-XL superclass (EVA-6) | macro AUROC 0.867 (0.858-0.878), hand-built features | Deep models on the same split report 0.921-0.930 (section 5.1). Fold 9 was unused. |
| Sleep staging (EVA-4) | subject kappa 0.669 (0.545-0.785); the logistic baseline (0.683) was better | Only 5 subjects and 10 nights. The full sleep-cassette set has 153 nights. |
| Seizure-suspect windows (EVA-2) | subject AUROC 0.986, but sensitivity at 1% FPR only 0.35 (0.20-0.51); Siena AUROC 0.920, AP 0.173 | 5 of 24 CHB-MIT cases and 6 of 14 Siena subjects. Window-level metrics only. Normalization used all of the person's windows, including ictal ones, while the API uses baseline windows only. Seizure-free files were subsampled, so false alarms per hour are not realistic. |

The GPU was never used for the beta. Each SLURM job on arrow reserves its own GPU, so the deep models move there.

## 2. Cluster ground rules (from the cuny-hpc skill)

- **One GPU per job, used fully.**
  - `--gres=gpu:1` with 8 CPUs. No memory fraction.
  - `--mem` is sized from a smoke run plus about 50%.
  - `--time` is realistic, not the partition maximum.
- **No distribution.**
  - No multi-node runs, no job arrays, no fan-out.
  - Folds, seeds and model variants run one after another inside the session's single job.
  - At most one of our jobs runs and one waits in the queue.
- **Partition and GPU.**
  - `partnsf` with `--qos=qosnsf`. Treat 120 h as the cap until `scontrol` says otherwise. Every job below asks for 14 h or less.
  - Pin the GPU type to V100 or A100 using the name `discover.sh` reports. The torch CUDA 12.x build cannot use the K20m (sm_35). Every job also checks the GPU itself and exits with code 3 on an unsupported one.
- **Logs and outputs.**
  - Every job defines `--output` and `--error`.
  - Program output goes to a `.log` file in the run folder.
  - Jobs start from `/scratch/<hpc-user>`.
- **Scratch is purged and not backed up.**
  - Data and run outputs on scratch can disappear after about 2 weeks, or sooner if the filesystem passes 70%.
  - Code lives in git and raw data on the lab server, so scratch can always be rebuilt.
  - Results are copied off within days of a run.
  - No "touch jobs" to refresh timestamps.
- **Checkpoint everything over an hour.**
  - The code saves a checkpoint after every epoch and every finished fold.
  - The job sends `SIGUSR1` 10 minutes before its time limit, and the code saves and exits with code 99.
  - The run folder is named after the session and the code commit, not the job ID. Resubmitting the same script therefore resumes and skips finished folds.
- **Cluster facts not yet known.**
  - `ARROW_PW` is not set yet, so nobody has logged in.
  - `discover.sh` must run first. It fills in the GPU gres names, features, account and limits in the skill's `references/cluster-facts.md`.
  - Until then, the job scripts hold `CHANGE_ME` placeholders. `sbatch` is expected to reject the gres placeholder, but this is untested. The real gate before any submit is `grep -n CHANGE_ME *.sbatch` (only commented lines may remain) and `sbatch --test-only`.

## 3. Data: what each session needs and where it is

All of this data is open PhysioNet data available by instant download (ODC-By 1.0 or CC BY 4.0), which fits the hackathon data rule in [data-acquisition-plan.md](data-acquisition-plan.md) section 0. The lab server holds the original copies under `~/evade_data`. Sizes "on the lab server" come from [data/MANIFEST.csv](../../data/MANIFEST.csv). Full-set sizes were read from each PhysioNet page on 2026-11-14 (VERIFIED).

| Dataset | Session | On the lab server now (MANIFEST) | Full set (PhysioNet) | Needed for v2 | Download first? | Sent to arrow as |
|---|---|---|---|---|---|---|
| PTB-XL 1.0.3 | S1 | `~/evade_data/ecg_ppg/ptb-xl`, 87,205 files, 5.0 GB (official zip plus extracted copy) | 21,799 records from 18,869 patients; 3.0 GB uncompressed, 1.7 GB zip | 100 Hz records and the two CSVs | no | one packed `.npz` (12 x 1,000 per record, about 0.5-1.1 GB) plus the CSVs, built on the lab server. Never the 87k-file tree. |
| Sleep-EDFx | S2 | `~/evade_data/eeg/sleep-edfx`, 25 files, 0.50 GB (SC4001-SC4042, 5 subjects, 10 nights) | 8.1 GB for the whole set; 153 sleep-cassette (SC) files; EEG at 100 Hz | all SC PSG and hypnogram files | **yes**: about 7-8 GB more (*estimate*, scaled from 0.50 GB per 10 nights) | the SC folder (about 306 files in one directory) by rsync |
| CHB-MIT | S3 | `~/evade_data/eeg/chbmit`, 230 files, 13.3 GB (chb01-chb05, 197 EDF) | 42.6 GB; 664 EDF; 24 case folders chb01-chb24 (the page: 23 cases from 22 subjects, chb21 is chb01 1.5 years later; chb24 added later and not in SUBJECT-INFO) | all 24 cases | **yes**: about 29 GB more | the tree (24 case folders, each under 1,000 files) by rsync |
| Siena Scalp EEG | S3 | `~/evade_data/eeg/siena-scalp-eeg`, 26 files, 9.2 GB (6 subjects) | 20.3 GB; 14 subjects; 47 seizures; 512 Hz | all 14 subjects (external test) | **yes**: about 11 GB more | the tree (14 subject folders) by rsync |
| MIT-BIH, BIDMC | S4 (optional) | 109 MB and 218 MB | - | full | no | one tar each |

**Total moved to arrow:** about 75 GB (*estimate*), mostly CHB-MIT and Siena.

The new downloads go to the lab server, not arrow, following evade-server-rules. Each one gets a MANIFEST row with a new listing hash. The beta subsets stay identifiable inside the full sets, so the beta comparisons can be repeated exactly (section 6).

## 4. Transfer plan (lab server to arrow)

Every transfer is **run by the user** in tmux on the lab server, because each hop asks for a password. Data goes directly from the lab server to arrow through the chizen jump host:
- never staged through this PC;
- never copied to chizen first, because files left on chizen are deleted.

1. **Notify HPC staff** before the big transfer (about 75 GB in total, 60 GB of it for S3). The notice goes to hpchelp@csi.cuny.edu from a CUNY address. Suggested text:
   > "User <hpc-user> will transfer about 75 GB of public PhysioNet EEG/ECG data (CHB-MIT, Siena, Sleep-EDFx, PTB-XL) from a CSI lab server to /scratch/<hpc-user> on Arrow via chizen, rate-limited to about 20 MB/s, over the next two weeks. Please advise if you prefer another time or rate."
2. **Prepare the layout on arrow** (login node, one broker call): `mkdir -p /scratch/<hpc-user>/evade/{data,runs,slurm,code}`.
3. **Pack PTB-XL on the lab server.** The script is `evade.training.deep.prep --task ecg`, still to be written (section 7). It reads the 100 Hz records and writes `ptbxl100.npz` (signals, ecg_id, patient_id, strat_fold, superclass labels) and `ptbxl100.sha256`. It is a CPU job of a few minutes, run under the lab server guard.
4. **Run rsync, one dataset at a time, just before its session.** Example for CHB-MIT:
   ```
   rsync -rlp --partial --size-only --bwlimit=20000 --info=progress2 \
     -e "ssh -J <hpc-user>@chizen.csi.cuny.edu" \
     ~/evade_data/eeg/chbmit/ \
     <hpc-user>@arrow.csi.cuny.edu:/scratch/<hpc-user>/evade/data/chbmit/
   ```
   - `-rlp` without `-t` gives files a fresh arrival time. This has the same effect as unpacking with `tar -xm`; it is not a touch job.
   - `--size-only` lets an interrupted transfer resume.
   - `--bwlimit=20000` (about 20 MB/s) is a *design* value; hpchelp may ask for a different one. At that rate 60 GB takes about 50 minutes.
   - The same pattern applies to `sleep-edfx/sleep-cassette/`, `siena-scalp-eeg/` and the PTB-XL `.npz`.
5. **Verify on arrow** (login node, cheap).
   - Compare the file count and total bytes with the lab-server listing, using `find ... -type f | wc -l` and `du -sb`.
   - For the PTB-XL pack, check its sha256.
   - Then write `data/<dataset>/VERIFIED`. Every job refuses to start if that file is missing.
6. **Timing against the purge.**
   - Send each dataset no more than a few days before its session, and run its CPU prep job (S2a, S3a) right before its GPU job.
   - If the data has been purged, repeat steps 4 and 5. rsync then copies only what is missing.
7. **Code.** Code goes up as one `git archive` tar with the broker's `put`, is unpacked with `tar -xmf` into `evade/code`, and gets a `COMMIT` file. Uncommitted code is never uploaded. Details are in the hpc README.

## 5. Models

Each model has:
- a **primary configuration, chosen in this document before any result is seen**;
- secondary configurations reported beside it.

The model is chosen on validation subjects only; test subjects are never used for choosing. Nothing is tuned on a test fold.

### 5.1 ECG: PTB-XL diagnostic superclass (session S1)

- **Target:** five multi-label superclasses (NORM, MI, STTC, CD, HYP) from 12-lead 10 s ECG at 100 Hz. This is the same label mapping as EVA-6 (`scp_statements.csv`, diagnostic == 1).
- **Published reference:** the PTB-XL benchmark repository (Strodthoff et al., "Deep Learning for ECG Analysis: Benchmarks and Insights from PTB-XL") lists these superclass AUCs on test fold 10 (VERIFIED from the repository README, github.com/helme/ecg_ptbxl_benchmarking, on 2026-11-14):

  | Model | AUC |
  |---|---|
  | resnet1d_wang | 0.930 |
  | xresnet1d101 | 0.928 |
  | fcn_wang | 0.925 |
  | inception1d | 0.921 |
  | Wavelet+NN | 0.874 |

  Our feature model (0.867) sits close to their feature baseline. Their exact training recipe has not been re-checked here (UNVERIFIED).
- **Primary model:** `xresnet1d101`, a 1D xresnet with concat pooling and 5 sigmoid outputs.
  - Input: 12 x 1,000 samples, standardized per lead with training-fold mean and SD; these statistics are stored in the manifest.
  - Training: random 2.5 s crops, BCE, AdamW with a one-cycle schedule, 50 epochs (*design*).
  - Evaluation: one forward pass on the full 10 s input, so the evaluated model is the one served.
  - Seeds: 5. The headline reports the seed mean. The **served** seed is the one with the best fold-9 macro AUROC.
- **Secondary configurations (same job, run in sequence):**
  - `resnet1d_wang`, 1 seed;
  - `xresnet1d101` on **lead I only**, 1 seed. This is the wearable-relevant variant, because chest straps and patches are single-lead.
- **Split:** identical to the beta. Folds 1-8 train (the same 17,418 records), fold 9 validates (early stopping, the choice of seed, thresholds), and fold 10 tests (2,198 records), opened once. Fold 9 was unused in the beta, so the training data is identical.
- **Baseline:** EVA-6 (HistGradientBoosting) and its logistic baseline, re-scored on fold 10 on the lab server with per-record predictions saved. This needs `--save-preds` in `evade.training.train`. The paired bootstrap needs those predictions.
- **Metrics:**
  - macro and per-class AUROC;
  - a paired patient-level bootstrap (2,000 resamples) of the difference from EVA-6;
  - per-class sensitivity and specificity at the fold-9 operating points;
  - calibration (Brier score, reliability bins);
  - parameter count and onnxruntime CPU latency.
- **Operating points (manifest):** for each class, the fold-9 threshold at 90% specificity, labelled as a `review` flag (*design*).
- **GPU time:** *estimate* about 2.5 GPU-hours on a V100 (7 trainings of about 15-20 minutes each, plus evaluation and export); A100s are faster. The job asks for `--time=05:00:00`.

### 5.2 Sleep staging: Sleep-EDFx sleep-cassette (sessions S2a and S2)

- **Target:** 5 classes (W, N1, N2, N3, REM) per 30 s epoch, with the same trimming as the beta (the sleep period plus 30 minutes of wake on each side).
- **Data:** all 153 SC nights. The subject count comes from `SC-subjects.xls`; it is commonly cited as 78, which is UNVERIFIED.
- **Primary model:** a TinySleepNet-style network.
  - A per-epoch 1D CNN feeds a one-layer LSTM over sequences of 15 epochs.
  - Input: Fpz-Cz and Pz-Oz at 100 Hz (2 x 3,000 samples per epoch), z-scored per channel per night. The statistics come only from that night's own signal, so the API can compute them the same way.
  - Training: weighted cross-entropy, about 30 epochs with early stopping on validation kappa (*design*).
- **Secondary configuration:** the same network on Fpz-Cz only (single-channel wearable headbands).
- **Splits:**
  - **Beta-comparable track:** leave-one-subject-out on the same 5 beta subjects (SC400-SC404, the same 10,589 epochs). The deep model is trained on 4 subjects per fold, which is small. It is reported for continuity and is not the decision metric.
  - **Expanded track (decision):** subject-wise 10-fold cross-validation over all SC subjects, using a frozen split file `data/splits/sleep_sc_10fold_v1.csv` whose sha256 goes in the manifest. Inside each training fold, 10% of the training subjects are held out for early stopping. Both nights of one subject always stay in one fold.
  - The 5 beta subjects are reported separately within the expanded track.
- **Baseline:** the beta HGB (EVA-3 architecture) and the logistic model, **re-run on exactly the expanded folds** in the CPU job S2a, with predictions saved. The beta showed the logistic model can be the stronger baseline, so the deep model is compared with **both**.
- **Metrics:**
  - pooled and per-subject Cohen's kappa, macro F1, accuracy and per-class F1;
  - a subject-level bootstrap CI (2,000 resamples) and a paired bootstrap of the kappa difference from each baseline;
  - the confusion matrix.
- **Published reference:** deep models on the SC 78-subject set are reported around kappa 0.75-0.80 (UNVERIFIED, recalled; check the source before quoting).
- **Serving note:** the API schema has no sleep endpoint. For now the model output would feed total sleep time into component B1 of [risk-score.md](risk-score.md), computed offline per night. That is an open question (section 12).
- **GPU time:** *estimate* about 4 GPU-hours: 10 folds of about 20 minutes, a final all-subject model, and the 5-fold beta track of about 3 minutes per fold. The job asks for `--time=08:00:00`. S2a is a CPU job of about 1 hour on 8 cores (`--time=02:00:00`).

### 5.3 EEG seizure-suspect windows: CHB-MIT with Siena external test (sessions S3a and S3)

- **Target:** a seizure-suspect flag per 4 s window with a 2 s step, on the 18-channel bipolar montage at 250 Hz. These are the beta windows, labels and buffers (`evade/preprocessing/config.py`), with scores smoothed over 3 windows. Siena is converted from monopolar to bipolar with the existing code, with a 50 Hz notch.
- **Per-person normalization is kept and moved to the signal.**
  - Each person's raw signal is robust-scaled per channel: median and 1.4826 x MAD.
  - The statistics come from that person's **baseline windows**: the first 10 minutes of the person's first seizure-free recording.
  - The API can reproduce this from `/v1/baseline/eeg`, which closes the beta limit where normalization included ictal windows.
  - The manifest's input spec records the rule, the number of baseline windows and the clipping (±20).
  - The baseline windows are excluded from scoring for **all** models in this track, so the comparison stays fair.
- **Primary model:** a small 1D ResNet over (18, 1,000).
  - About 8 residual blocks with 64-128 filters, global average pooling and a sigmoid (*design*).
  - Batches are balanced between ictal and non-ictal windows. Augmentation: time shift and ±20% amplitude scaling, never mixing two people.
  - BCE, AdamW, at most 20 epochs, early stopping on the AUROC of inner validation subjects.
- **Secondary model:** EEGNet. It is named in [implementation-pipeline.md](implementation-pipeline.md) 2.1 and reported beside the primary, never swapped in after seeing test results.
- **Tracks:**

  | Track | Data | Normalization | Split | Purpose |
  |---|---|---|---|---|
  | A. beta-comparable | chb01-chb05, the beta's file selection (all seizure files plus 3 seizure-free files per subject; 97,624 windows) | as in the beta: all of the person's windows | leave-one-subject-out over 5 subjects; Siena-6 external, with the threshold from CHB-MIT | the exact comparison with EVA-2 |
  | B. API-matched | same as A | baseline windows only (above) | same as A | what the API can actually deliver on the beta data; decides serve or not against EVA-2 (section 8) |
  | C. expanded | all 24 cases, grouped as 23 subjects (chb21 belongs to chb01; chb24, whose subject is not listed, is treated as its own person: UNVERIFIED) | baseline windows only | grouped 5-fold by subject, frozen in `data/splits/chbmit_23subj_5fold_v1.csv`; 2 training subjects per fold held out for validation. Training uses the beta's file-selection rule. **Testing uses every file of the test subjects** (continuous). | realistic false alarms per hour, more subjects; its pooled out-of-fold scores set the served model's thresholds (1% and 5% FPR) |
  | Final | all 23 subjects | baseline windows only | train on everything, then test once on Siena-14, also reporting Siena-6 | the external result and the served model |

- **Baseline:** the beta HGB (EVA-1 architecture) and the logistic model.
  - Track A reuses the beta setup. Track A predictions come from re-running the beta pipeline on the lab server with `--save-preds`.
  - Tracks B, C and Final re-run the beta model with the same windows and folds in the CPU job S3a. That job computes the beta features for every file and saves predictions.
- **Metrics:** everything in the beta table, plus:
  - window AUROC and AP, pooled and per subject;
  - subject-mean sensitivity at 1% and 5% pooled out-of-fold FPR (the API's `suspect_seizure` and `review` levels) and false-positive windows per hour;
  - **event-level** metrics: seizure-level sensitivity (a seizure counts as detected if any window inside it is flagged), false alarms per hour after merging consecutive flagged windows into one alarm with a 30 s refractory period (*design*), and detection latency;
  - paired subject-level bootstrap differences (2,000 resamples) against the HGB in the same track.

  The scoring follows the SzCORE event idea but is not a certified SzCORE run; the README will say so.
- **GPU time:** *estimate* about 7 GPU-hours:

  | Part | Estimate |
  |---|---|
  | Tracks A and B (2 x 5 folds of about 15 minutes) | about 2.5 h |
  | Track C (5 folds of about 45 minutes) | about 3.75 h |
  | Final model plus inference on Siena | about 0.75 h |

  The job asks for `--time=14:00:00`. S3a is a CPU job of about 4-6 hours on 8 cores with 64 GB, needed because single Siena files reach 1.7 GB and took 30-40 GB of memory in the beta. It asks for `--time=08:00:00`.

### 5.4 Optional: HRV and PPG (session S4, only if S1 to S3 are finished)

- **MIT-BIH beat classes:** the AAMI inter-patient split (DS1 trains, DS2 tests), with a small 1D CNN on beat-centred windows and a feature HGB baseline.
- **BIDMC:** PPG heart-rate estimation compared with the reference HR. Bland-Altman analysis, leave-one-recording-out.
- **Value:** these are small datasets (48 and 53 recordings), so a deep model may not beat rules. The value is a registered `/predict/ecg` quality and beat model for single-lead input.
- **GPU time:** *estimate* about 1 GPU-hour (`--time=03:00:00`). Skip it if time is short; the CPU rules already cover HR and HRV.

## 6. Evaluation protocol (all sessions)

1. **Subject-wise always.** No person appears in both training and test data. For CHB-MIT, chb01 and chb21 count as one person. PTB-XL patients stay inside their official fold.
2. **Same splits as the beta for every comparison with the beta.** The beta-comparable tracks are 5.1 (official folds), 5.2 (beta track) and 5.3 (track A).
   - When the data grows, new split files are frozen **before** training, committed under `data/splits/` with their sha256, and the baseline is re-run on those exact folds.
   - A model and its baseline are never compared on different test people.
3. **The test data is opened once per pre-registered claim.** Model choice, early stopping, seed choice and thresholds use validation data only (PTB-XL fold 9; held-out training subjects elsewhere).
4. **Confidence intervals.**
   - The resampling unit is the subject or patient, with 2,000 resamples. The beta ECG used 200; v2 uses 2,000 everywhere.
   - Each model gets its own CI, and the comparison uses a **paired** CI of the difference, computed from both models scored on the same resamples.
5. **The beta model is the baseline.**
   - The new model replaces the served beta model only if the lower bound of the paired difference CI is above 0 on the decision metric, and no other headline metric drops by more than its own CI half-width.
   - Otherwise the beta model stays served and the new result is reported anyway, failures included.
6. **The baseline sits beside the model.** The logistic baseline is reported in every table, as the beta did.
7. **Leakage checks.**
   - Scalers are fitted on training data only (ECG), or on the person's own baseline (EEG) or own night (sleep).
   - Each run checks that split files do not overlap, and the job fails if they do.
8. **Everything a claim rests on is written to `metrics.json`:**
   - per-subject results and the split file hashes;
   - the code commit and the GPU type;
   - the epochs actually run;
   - the bootstrap seed.

## 7. Code to write before the GPU sessions (part of session 0)

These modules do not exist yet. The job scripts call them and stop with exit code 4 if a module is missing. Write them on a separate branch and unit-test them on the lab server through the broker (`~/evade_venv/bin/python -m pytest -q`), as evade-server-rules require. Upload only committed code.

| Module (planned) | Purpose |
|---|---|
| `evade/training/deep/prep.py` | `--task ecg`: packs PTB-XL into `.npz`. `--task sleep`: writes per-night float16 arrays and an epoch index. `--task seizure`: per-record float16 memmaps on the bipolar montage at 250 Hz, plus a window index with labels and baseline-window marks. `--synthetic` for smoke tests. |
| `evade/training/deep/models.py` | xresnet1d101, resnet1d_wang, TinySleepNet-style CNN-LSTM, 1D ResNet-small, EEGNet |
| `evade/training/deep/train.py` | `--task --protocol --arch --seeds --splits --run --resume`. Checkpoints every epoch, writes a `DONE` file per fold and skips finished folds. On `SIGUSR1` it saves and exits 99. `--smoke` runs a tiny subset for one epoch. |
| `evade/training/deep/evaluate.py` | metrics and subject or paired bootstrap from saved predictions of both models. Seizure event scoring. Writes `metrics.json`. |
| `evade/training/deep/export.py` | writes the servable folder (section 9): ONNX export (opset 17, dynamic batch axis), torch versus onnxruntime parity check, CPU latency, `manifest.json`, sha256 |
| `evade/training/train.py` (change) | `--splits <csv>` and `--save-preds` so the beta pipeline can be re-run on fixed folds. For the seizure task also `--windows <index.parquet>` (score exactly the prep windows) and `--norm all-windows,baseline-windows` (both per-person normalizations) |
| `data/splits/*.csv` | frozen split files (subject to fold), committed with sha256 |

Tests should cover:
- shapes;
- split disjointness;
- per-person normalization using baseline windows only;
- resume after a fake `SIGUSR1`;
- ONNX parity on a tiny random model;
- that the manifest's fields are present.

## 8. Session order, GPU-hours and go/no-go

The GPU-hours are *estimates* on a V100 basis, to be replaced by (smoke epoch time x planned epochs) after S0. The sessions run one at a time: one job running, at most one waiting.

| # | Session (job script) | Resources | Estimated use | `--time` | Go / no-go before the next step |
|---|---|---|---|---|---|
| 0a | User sets `ARROW_PW`; `discover.sh` once; fill in `cluster-facts.md`; replace the `CHANGE_ME` gres type and account in all scripts; `sbatch --test-only` each script | login node | - | - | Go if `--test-only` accepts the pinned V100 or A100 request on partnsf/qosnsf. |
| 0b | `setup_env.sh` (venv on scratch, torch cu126); write and test the section 7 code on the lab server; send the hpchelp notice; pack PTB-XL; download the full CHB-MIT, Siena and Sleep-EDFx SC sets to the lab server | login node, lab server | - | - | Go if lab-server pytest passes and MANIFEST rows exist for the new downloads. |
| 0c | `s0_smoke.sbatch`: pytest on a compute node, then synthetic train, evaluate and export for all 3 tasks | 1 GPU, 8 CPUs, 16 GB | about 0.3 GPU-h | 00:30:00 | Go if: (1) the GPU is a V100 or A100 and torch sees it; (2) pytest passes; (3) each task writes all four contract files and `manifest.json` validates; (4) ONNX max absolute difference ≤ 1e-4 on probabilities; (5) the resume test passes. Record seconds per epoch and MaxRSS (`sacct`). |
| 1 | `s1_ecg_ptbxl.sbatch` (first a 15-minute run with `SMOKE=1` on real data) | 1 GPU, 8 CPUs, 32 GB | about 2.5 GPU-h | 05:00:00 | **Serve or not:** paired macro AUROC difference from EVA-6 has a lower bound above 0. **Sanity check:** macro AUROC ≥ 0.90. Below that, check lead order, units, label mapping and standardization before anything else; never tune on fold 10. The export and registration path proven here is reused by S2 and S3. |
| 2a | `s2a_sleep_prep_cpu.sbatch` (after the SC rsync) | 0 GPU, 8 CPUs, 32 GB | about 8 CPU-h | 02:00:00 | Go if the epoch counts for the 5 beta subjects match the beta (10,589) and the split file hash is recorded. |
| 2 | `s2_sleep_edfx.sbatch` | 1 GPU, 8 CPUs, 48 GB | about 4 GPU-h | 08:00:00 | **Serve or not:** paired subject kappa difference has a lower bound above 0 against **both** the re-run HGB and logistic baselines on the expanded folds, and kappa ≥ 0.70. Otherwise the better baseline stays. |
| 3a | `s3a_seizure_prep_cpu.sbatch` (after the CHB-MIT and Siena rsync) | 0 GPU, 8 CPUs, 64 GB | about 40 CPU-h | 08:00:00 | Go if track A window and ictal counts equal the beta's (97,624 and 992; Siena-6: 120,032 and 386) and no file is skipped without a logged reason. |
| 3 | `s3_seizure_chbmit.sbatch` | 1 GPU, 8 CPUs, 64 GB | about 7 GPU-h | 14:00:00 | **Serve or not:** in track B, paired subject sensitivity at 1% FPR has a lower bound above 0 against the HGB, and the Siena-6 AUROC difference lower bound is above -0.02 (*design* margin). Report the track C event metrics either way. Investigate any subject with AUROC below 0.80. |
| 4 | `s4_hrv_ppg_optional.sbatch` | 1 GPU, 8 CPUs, 16 GB | about 1 GPU-h | 03:00:00 | Only if S1-S3 are done and time remains. |

**Totals (*estimate*):**
- about 15 GPU-hours used, against about 32 h of requested wall time;
- about 50 CPU-hours.

The wiki gives new MAP projects a one-time 1,440 GPU-hours. Whether this account is on MAP or on the NSF condo is unknown.

**Why this order:**
- ECG comes first because its data is complete, already on the lab server and small. It also has a published target, so it is the cheapest way to prove the whole path end to end: transfer, training, export, fetch and registration.
- Sleep comes next, because it needs only about 8 GB of new data.
- Seizure comes last. It needs the largest download and transfer (and the hpchelp notice sent in 0b), and its protocol is the most involved.
- A failed session does not block the next one, except for export or path bugs, which must be fixed first.

After every job, read the end of `run.log` once and run `sacct -j <id> --format=JobID,State,Elapsed,MaxRSS,ExitCode`. Set the next job's `--mem` to MaxRSS x 1.5, and copy the export folder off scratch the same week.

## 9. Export: the servable folder

Every session writes `<run>/export/<task>/` with:

| File | Content |
|---|---|
| `manifest.json` | the fields below |
| `model.onnx` (preferred; CPU-servable with onnxruntime) or `model.pt` (only if ONNX export fails, with the reason in the manifest) | weights |
| `feature_names.json` | the input channel or lead names, in order (for deep models, the "features" are channels) |
| `metrics.json` | section 6, item 8 |

`<run>/export/SHA256SUMS` lists every file. Task folder names follow the beta (`ecg`, `sleep`, `seizure`), so the API's model folder layout keeps working.

**Contract.** The fields follow the model artifact contract being built on branch `api-models-storage` (`evade/models/README.md` there). That file **did not exist yet** on 2026-11-14, so these fields come from the task description and must be reconciled when that branch lands. No other fields are added here.

| Field | Content |
|---|---|
| `model_id` | `EVA-[n]`, the trained model's registry ID |
| `task` | `ecg_superclass`, `sleep_staging` or `seizure_window` |
| `status` | `draft` until the go criterion is met and a person reviews it |
| `format` | `onnx` or `torch` |
| `weights` | file name |
| `sha256` | of the weights file |
| `input` | shape, dtype, sampling rate, channel order (also in `feature_names.json`), units, filtering, and the normalization rule with its statistics or how the API computes them (for example the seizure baseline-window rule) |
| `output` | classes, activation, and **operating points** (threshold, target FPR or specificity, the validation sensitivity at that point, and the API level it maps to, such as `review` or `suspect_seizure`) |
| `code_commit` | git hash from `code/COMMIT` |
| `trained_at` | UTC time |
| `data` | dataset names and versions, listing hashes from MANIFEST, split file sha256 |

Example (seizure; values are placeholders):
```json
{
  "model_id": "EVA-<n>", "task": "seizure_window", "status": "draft",
  "format": "onnx", "weights": "model.onnx", "sha256": "<64 hex>",
  "input": {"shape": ["batch", 18, 1000], "dtype": "float32", "fs_hz": 250, "window_s": 4.0, "step_s": 2.0,
            "channels": "feature_names.json", "filter": "0.5-45 Hz band-pass, mains notch",
            "normalization": "per-person per-channel median and 1.4826*MAD from >= 300 baseline windows, clip +-20",
            "post": "centered moving average of 3 window scores within a recording"},
  "output": {"classes": ["seizure_suspect"], "activation": "sigmoid",
             "operating_points": [{"name": "review", "target_fpr": 0.05, "threshold": "<t>", "sensitivity_val": "<s>"},
                                  {"name": "suspect_seizure", "target_fpr": 0.01, "threshold": "<t>", "sensitivity_val": "<s>"}]},
  "code_commit": "<hash>", "trained_at": "<UTC>",
  "data": {"train": "CHB-MIT 1.0.0 (listing <hash>)", "external_test": "Siena 1.0.0 (listing <hash>)", "splits": {"chbmit_23subj_5fold_v1.csv": "<sha256>"}}
}
```

**Checks inside the export step:**
- ONNX and torch outputs match to 1e-4 on 64 validation inputs;
- onnxruntime CPU latency, single thread, median and 99th percentile per window, against the 500 ms budget in [implementation-pipeline.md](implementation-pipeline.md) 3.2;
- parameter count.

**Fetching it back.**
1. The models are expected to be small (well under 50 MB; *estimate*). The user pulls the export folder to the lab server the same week:
   `rsync -rltv -e "ssh -J <hpc-user>@chizen.csi.cuny.edu" <hpc-user>@arrow.csi.cuny.edu:/scratch/<hpc-user>/evade/runs/<run>/export/ ~/evade_models/<run>/`
   For files under a few MB, `srv.py --host arrow get` also works.
2. Run `sha256sum -c SHA256SUMS` on arrival.
3. From the lab server, the folder goes to the API host's model mount, or the API team pushes it to Cloudflare R2 from a machine that already holds the R2 keys in its environment.
4. **R2 keys never go on arrow, in scratch, in job scripts or in the repo.** Check sha256 again after every hop.

**Serving dependency.** Today the API (`evade/api/app.py`) loads only `model.joblib`. ONNX models need the loader and onnxruntime from `api-models-storage` before they can be served. Until then they can be evaluated but not served.

## 10. Registration (MODEL_REGISTRY.md)

- **Current state.** MODEL_REGISTRY.md on `integrate-blindsight-data-nasa` says "Next free ID: EVA-7". The space-weather model being built on branch `spaceweather-model` is likely to take EVA-7, and possibly EVA-8 if it registers an arch and a model.
- **Assignment.** IDs are therefore **not** assigned in this plan. Each one is taken at registration time from "Next free ID" on the integrated branch, after any open registry changes are merged. The registry row goes in the same commit as the artifact's manifest, metrics and card. Weights stay out of git, referenced by sha256.
- **IDs needed.** Expected at least 8 (in the order created):

  | Created in | Type | Name (proposed) | Parent | Illustrative ID if space-weather takes EVA-7 and EVA-8 |
  |---|---|---|---|---|
  | S1 | arch | ecg-xresnet1d101 | - | EVA-9 |
  | S1 | model | ecg-xresnet1d101 v1 (PTB-XL folds 1-8) | ecg arch | EVA-10 |
  | S2a | model | sleep-stage-hgb expanded (beta arch re-run, SC 10-fold) | EVA-3 | EVA-11 |
  | S2 | arch | sleep-cnn-lstm | - | EVA-12 |
  | S2 | model | sleep-cnn-lstm v1 | sleep arch | EVA-13 |
  | S3a | model | seizure-window-hgb expanded (23 subjects, baseline-window normalization) | EVA-1 | EVA-14 |
  | S3 | arch | seizure-resnet1d-small | - | EVA-15 |
  | S3 | model | seizure-resnet1d-small v1 | seizure arch | EVA-16 |

- **Secondary configurations** need their own IDs only if they are cited in a result table (registry rule 7). Examples: ECG lead I only (a model under the ECG arch); resnet1d_wang and EEGNet (new archs). Otherwise they stay in `metrics.json` as unretained runs.
- **Status.** New rows start as `draft`. A row moves to `active` only after its go criterion holds and someone reviews it. When a new model replaces a beta model in serving, the beta row changes to `superseded`; it is never deleted or renumbered.

## 11. Privacy and safety

- Training uses only public, de-identified research data. No EVAde user data (pseudonymous UUID `user_id`s, baselines, waveforms sent to the API) is ever copied to arrow or used for training in v2.
  - A user's delete-on-request therefore never reaches cluster storage.
  - If user data is ever used for training, that needs a separate plan: consent, lineage from model to `user_id`s, and retraining or removal on deletion.
- Logs hold IDs, counts, timings and metrics only. No raw waveforms go into logs or `metrics.json`.
- No passwords or API keys go in job scripts, files, logs or the repo. `ARROW_PW` stays in the user's environment.
- Model outputs stay labelled as flags for human review. No model card or README may say "detects", "diagnoses" or "medical-grade". All data is terrestrial and clinical; nothing here says how these models behave in spaceflight or on wearables.
- Only weights we trained are loaded (ONNX or our own `.pt` with `weights_only=True`). Downloaded files are never unpickled.

## 12. Risks and open questions

1. **Cluster facts.** The GPU gres type and feature names, the account, the real partnsf time limit, the preemption mode and the Python module are unknown until `ARROW_PW` is set and `discover.sh` runs.
2. **Downloads.** Downloading the rest of CHB-MIT (about 29 GB), Siena (about 11 GB) and Sleep-EDFx SC (about 7-8 GB) onto the lab server needs a go-ahead, because it uses lab-server disk.
3. **hpchelp notice.** Who sends it from a CUNY address, and when?
4. **Sleep in the API.** Should the sleep model get an endpoint or a `/predict/eeg` mode, or only feed B1 sleep hours offline?
5. **Artifact contract.** When will `api-models-storage` publish `evade/models/README.md`? The manifest fields here must match it. Who owns the R2 bucket, and from which machine is it pushed?
6. **Account type.** Is the account on the NSF condo (`qosnsf`) or a MAP allocation? This decides the GPU-hour budget.
7. **Purge timing.** If a session slips by more than about 10 days after its transfer, scratch data may be purged and the transfer has to be repeated.
8. **Unverified references.** The sleep reference range (kappa 0.75-0.80) and the PTB-XL training recipe are UNVERIFIED. The PTB-XL AUCs are VERIFIED from the benchmark repository only, not the paper's tables.
9. **PDF.** `lit review/lit-review.pdf` has not been rebuilt with this file, because the builder script is not in the repo.
10. **chb24.** CHB-MIT's chb24 is not in SUBJECT-INFO. It is treated as its own person, but if it is the same child as another case, the subject-wise split leaks. Check the case notes, or leave chb24 out of track C.
11. **Untested sbatch behaviour.** It has not been checked on arrow whether `sbatch` rejects the `CHANGE_ME_GPU_TYPE` placeholder outright or queues the job forever. The gate is `grep CHANGE_ME` plus `sbatch --test-only`.
