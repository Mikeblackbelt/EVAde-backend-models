# Deep seizure-suspect windows: CHB-MIT, with Siena as an external test

This is the deep version of the beta EEG seizure model (EVA-2) for the CUNY HPCC cluster. It is a hackathon proof of concept. It follows the seizure section of [training-plan-v2.md](../../../lit%20review/planning/training-plan-v2.md) (section 5.3) and the patterns of the deep ECG code in this folder.

**Claim:** the model marks 4 s EEG windows as *seizure-suspect*, as a flag for a person to review. It is not a diagnosis and not a seizure detector. Every number below is a validation or external-test result on research data.

**Status:** written and tested on synthetic data on a CPU. Nothing has run on the cluster yet, and no result exists.

**PoC scope (default):** CHB-MIT chb01-chb05, the beta's subjects with the beta file rule, for training and LOSO. The beta's six Siena subjects (PN00, PN01, PN03, PN05, PN06, PN07) are the external test. This is about 11 GB to download. More data stays an option (see the download script).

## Files

| File | What it does |
|---|---|
| `evade/training/hpc/seizure_download.sh` | Polite, resumable `wget` of the PoC scope, about 11 GB, so no notice to HPC staff is needed. Every file is checked against the dataset's `SHA256SUMS.txt`, and `VERIFIED` is written when all pass. `CHB_CASES`, `SIENA_SUBJECTS` and `MAX_FREE` fetch more. Everything together is about 63 GB, and **then hpchelp must be told first**. |
| `evade/training/deep/seizure_select.py` | Picks the EDFs the download fetches, with the pack's own summary parser and file rule |
| `evade/training/deep/seizure_pack.py` | CPU packing into one float16 `.npz` shard per CHB-MIT case or Siena subject. A shard holds labels, record IDs, a seizure table, and windows already normalized per window (`per_window_robust`). |
| `evade/training/deep/seizure_models.py` | `seizure_resnet1d`: a channel-wise conv, then cross-channel mixing, then a temporal residual stack (`seizure_tiny` is for tests) |
| `evade/training/deep/seizure_train.py` | Leave-one-subject-out folds and the final model. Resumable: SIGUSR1 saves a checkpoint and exits 99. Writes `metrics.json`. |
| `evade/training/deep/seizure_eval.py` | Window AUROC and AP, operating points, false-positive windows per hour, event-level sensitivity, false alarms per hour, and subject bootstrap CIs |
| `evade/training/deep/seizure_export.py` | ONNX export (`dynamo=False`, with a TorchScript fallback), an onnxruntime parity check (a failure exits 1), `manifest.json` in the API's `seizure_deep` form, and `SHA256SUMS` |
| `evade/training/hpc/seizure_prep.sbatch` | The CPU job that runs the packing |
| `evade/training/hpc/seizure_deep.sbatch` | The GPU job: LOSO, then the final model with Siena, then the export |
| `tests/test_deep_seizure.py` | End-to-end test on synthetic EDFs. It covers packing, LOSO with a fake SIGUSR1 mid-epoch and on the last step of a 1-epoch run, resume, the final model with an external set, and export. Also checked: normalization against the API formula, log replay and the stale-shard rebuild on a resumed pack, the download file selection against the pack's rule, and the metric helpers against sklearn. |

## Data and packing

Packing reuses the beta preprocessing:
- `read_edf`; Siena is read with `label_prefix="EEG"`, so the EKG and SpO2 channels are left out.
- The fixed 18-channel bipolar montage:
  - CHB-MIT uses `select_bipolar`. Missing channels are zero-filled, and each record stores which channels are present (`rec_present`).
  - Records with fewer than 12 of the 18 channels (`--min-channels`) are left out. They are logged as `EXCLUDED` and counted, with any seizures they hold, in `pack_summary.json`.
  - Siena uses `bipolar_from_monopolar`.
- `preprocess_eeg`: a 0.5-45 Hz zero-phase Butterworth band-pass and a Q 30 notch, then resampling. The notch is 60 Hz for CHB-MIT and 50 Hz for Siena. The output rate is set explicitly to **256 Hz**: CHB-MIT is native, and Siena is resampled from 512 Hz. The code runs one channel at a time to bound memory. The API runs the same function.
- The beta labels:
  - A window is ictal (1) when it overlaps a seizure by at least 2 s.
  - Windows from 60 s before a seizure to 120 s after it are -1.
  - Siena annotations that end after the recording does are never labelled ictal (the PN00-3 case).
  - **Siena seizure lists differ from the beta.** A list repeats `File name:` for every seizure, so a file can have several blocks, and the beta parser read only the first block per file. This pack reads every block, and also accepts `Start time:` and `End time:` lines without the word "Seizure". Siena labels, and therefore Siena metrics, can differ from the beta's on the same subjects.
- The beta file rule from `evade/training/data.py` (`chb_jobs`, `siena_jobs`): every file with a seizure, plus 3 seizure-free files per subject. The download fetches exactly these files, so the pack sees the same set. `--max-free -1` (and `MAX_FREE=-1` for the download) keeps every file.

Windows are 4 s (1,024 samples) with a 2 s step. They are normalized (below) and stored as float16 in `data/seizure256/{chbmit,siena}/<case>.npz` with these arrays:
- `X`, `y`, `rec`, `start_s`;
- `records`, `rec_dur_s`, `rec_channels`, `rec_present`;
- the seizure table `sz_rec`, `sz_start_s` and `sz_end_s`;
- `norm`, which is `"per_window_robust"`;
- `pack_info`: the shard's log lines, excluded records and counts. They are replayed when a resumed prep skips the shard.

Label -1 windows are kept, because smoothing and event scoring need them. Window metrics and training leave them out.

**Per-window normalization (`per_window_robust`).** Each window is scaled per channel with the median and 1.4826 x MAD of **that window's own samples**: `x_norm = clip((x - median) / max(1.4826 x MAD, 1e-6), -20, 20)`.
- This is the API's `per_window_robust` rule, line for line (`evade/models/deep_tasks.py`, `SeizureAdapter.predict`). Packing, training and serving therefore use the same numbers. A test checks the pack's function against the API's formula.
- No per-person statistics exist, so the API needs no stored baseline. No statistic comes from labels or from other windows. This removes the test-label leak of the earlier per-person rule.
- A channel missing from a record is zero-filled and stays exactly 0.
- There is no raw ±2,000 µV clip, because the API has none.
- Training refuses shards without `norm`, and the prep job rebuilds such stale shards.

The beta normalized per person over **all** windows, ictal ones included. A comparison with EVA-2 therefore mixes a model change with an input change.

chb21 is chb01 recorded 1.5 years later. When it is packed, it keeps its own shard and the training code groups it with chb01, so the person is never on both sides of a split. Neither chb21 nor chb24 (whose person identity is UNVERIFIED) is in the default scope.

## Model and training

`seizure_resnet1d` takes input of shape (batch, 18, 1024):
1. A depthwise temporal conv (groups = 18, 4 filters per channel, k = 15, stride 2) filters each channel separately.
2. A 1x1 conv mixes the channels.
3. Eight residual basic blocks (k = 7, 64-128 filters, 3 downsamplings).
4. Global average and max pooling, then one logit.

It has about 1.3 M parameters (a *design* choice, not tuned).

Training settings:
- **Batches** are balanced: half ictal windows (drawn with replacement) and half non-ictal. `--sampler random --loss focal` is the alternative.
- **Loss and optimizer:** BCE, AdamW (lr 1e-3, weight decay 1e-2) and a one-cycle schedule.
- **Augmentation:** a circular shift of up to ±0.5 s. A shift keeps each window's median and MAD, so the input stays exactly normalized. There is no amplitude scaling, because the per-window normalization cancels amplitude.
- **Length:** at most 20 epochs of up to 200,000 windows each, with early stopping (patience 5).
- **Precision:** fp16 autocast for training on CUDA (V100). Scoring runs in fp32, as the API does, so the thresholds match the served precision.
- **Resume:** SIGUSR1 mid-epoch saves `last.pt` and exits 99. On the last step of an epoch, the epoch is validated and checkpointed first, then the job exits 99. A 1-epoch smoke run therefore resumes cleanly.

Model selection uses validation subjects held out from the **training** side: 1 for chb01-05, and 2 with `CASES=all`. The test subject is scored once, with `best.pt`.

## Protocols (one GPU job, in sequence)

| Step | Data | Split | Purpose |
|---|---|---|---|
| `loso` | `CASES` (default chb01-05, beta file rule) | LOSO over 5 subjects (3 train + 1 val + 1 test) | the same subjects as the beta (EVA-2); its out-of-fold scores set the **served operating points** |
| `final` | train on `CASES` | the packed Siena subjects (default Siena-6) once, at the `loso` thresholds | the external result, the operating-point check, and the exported model |

With more downloaded CHB-MIT cases, `CASES=all` runs the same two steps over every packed case.

## Evaluation (`metrics.json`)

All scores are smoothed with a centered 3-window moving average inside each recording, as the beta did. The unsmoothed AUROC and AP are reported too.

| Metric | Definition |
|---|---|
| Window AUROC and AP | pooled out-of-fold, plus per subject and the subject mean |
| Operating points | `review` at 5% and `suspect_seizure` at 1% pooled out-of-fold FPR, from `evade.training.metrics.threshold_at_fpr` (tie-safe "higher" quantile; flag = score > threshold). External sets use the CHB-MIT thresholds unchanged. A Siena-derived threshold is reported only as a labelled oracle, never as a headline. |
| Per-fold thresholds | each test subject's own threshold at each FPR, and their spread (`per_fold_thresholds`) |
| Sensitivity | subject mean (the beta's number) and pooled, at each operating point |
| False-positive windows per hour | flagged non-ictal windows / (non-ictal windows x 2 s / 3600). With overlapping windows, one minute of false activity is about 30 windows. |
| Event-level sensitivity | a seizure counts as detected if any window overlapping it is flagged |
| False alarms per hour | runs of flagged non-ictal windows, merged within 30 s |
| CIs | 95% subject bootstrap with 2,000 resamples: subjects are resampled with replacement, the pooled metrics are recomputed, and thresholds stay fixed |

The operating-point numbers are **OOF fold-ensemble estimates**: they come from pooled out-of-fold scores of the LOSO fold models, not from the served final model, and the FPR holds by construction on those scores. The Siena result at those thresholds is the real operating-point check.

**Beta baseline (EVA-2, HGB on band-power features)**, stored in every `metrics.json` as `baseline_beta_eva2`:
- CHB-MIT chb01-05 LOSO: pooled AUROC 0.967, subject AUROC 0.986 (0.969-0.997), sensitivity at 1% FPR 0.35 (0.20-0.51), 8.8 FP windows per hour;
- Siena-6: AUROC 0.920 and AP 0.173.

This is window and simple event scoring in the spirit of SzCORE, not a certified SzCORE evaluation.

## Export (`runs/<run>/export/seizure_deep/`)

The export folder is the API's `<EVADE_MODEL_DIR>/seizure_deep/` folder. It holds:
- `model.onnx` (or `model.ts`);
- `feature_names.json` (the 18 channels in order);
- `metrics.json`;
- `manifest.json`;
- `SHA256SUMS`.

The manifest follows the API contract (`evade/api/README_deep_models.md` and `evade/models/deep_tasks.py` on `api-model-stubs`):
- `task` `seizure_deep`, `format` `onnx`, `sha256` and `files_sha256`;
- **input:** `fs_hz` 256, `window_s` 4, `step_s` 2, `channels` (the 18 bipolar pairs), `tensor_shape` `["batch", 18, 1024]`, `normalization` `{"type": "per_window_robust", ...}`, `smooth_windows` 3;
- **output:** `activation` `sigmoid` and `operating_points`: `review` (`target_fpr` 0.05) and `suspect_seizure` (0.01), each with its threshold, labelled as OOF estimates with the per-fold spread. The Siena numbers at those thresholds sit under `operating_point_check`.

The API branch's loader was run once on a test export: the model loaded as `available`, with `per_window_robust` and both thresholds read.

`model_id` stays `UNASSIGNED` and `status` stays `draft` until someone adds an EVA row to `MODEL_REGISTRY.md` and reviews the model. Then re-stamp it with `python -m evade.models.export --dir <folder> --model-id EVA-<n>`, or edit the field. Serve the draft with `EVADE_ALLOW_DRAFT=1`.

## Cluster commands

The main session runs these commands on the login node. `$PROJECT` is `/scratch/$USER/evade`, and the code is the committed revision unpacked in `$PROJECT/code` with `COMMIT`.

```bash
# 0. once: onnxruntime and pyedflib in the uv venv (setup_env_uv.sh installs both; for an existing venv:)
$PROJECT/tools/bin/uv pip install --python $PROJECT/venv/bin/python pyedflib onnxruntime
mkdir -p $PROJECT/slurm $PROJECT/data
# 1. download, on the login node: PoC scope, about 11 GB, 15-40 min; re-run the same line after interruptions
nohup bash $PROJECT/code/evade/training/hpc/seizure_download.sh > $PROJECT/data/seizure_download.log 2>&1 &
# 2. prep (CPU): smoke, then full (about 20-45 min, 8 cores, 64 GB; output about 8-10 GB)
sbatch --export=ALL,SMOKE=1 --time=00:30:00 $PROJECT/code/evade/training/hpc/seizure_prep.sbatch
sbatch $PROJECT/code/evade/training/hpc/seizure_prep.sbatch
# 3. train (1 GPU): smoke, then full (about 0.5-1 h on a V100, 4 h requested, 64 GB RAM)
sbatch --export=ALL,SMOKE=1 --time=00:30:00 --signal=B:USR1@60 $PROJECT/code/evade/training/hpc/seizure_deep.sbatch
sbatch $PROJECT/code/evade/training/hpc/seizure_deep.sbatch      # exit 99 = checkpointed; resubmit the same line
```

Before training, read the lines of `data/seizure256/pack.log` that the prep job printed after the `===` marker of its run. Check these counts against the beta:
- chb01-05 should give about 97,624 windows and 992 ictal ones (the beta ran at 250 Hz, so a few windows per file may differ).
- Siena-6: compare with the beta's Siena-6 counts. The "47 seizures" check only runs when all 14 Siena subjects are packed.

The pack prints a warning when a Siena list's seizure count does not match the parsed seizures, because the list parser was checked on PN00 only.

## Tests

```bash
python -m pytest -q tests/test_deep_seizure.py
```

The tests run on CPU on synthetic data in about a minute and a half. If pyedflib is missing, as on Windows with Python 3.13 where it has no wheel, the synthetic EDFs are fed through an in-memory stand-in for `read_edf` with the same return contract. The cluster venv reads real EDF files.

## Limits and open questions

- **The beta file rule** keeps every seizure file and 3 seizure-free files per subject. That subsamples seizure-free time, so false alarms per hour are optimistic or noisy.
- **Five training subjects** (the PoC scope) is little data for a deep model; `CASES=all` after a larger download is the next step.
- **Short API inputs:** the API filters each 4-30 s request on its own, while training filtered hours-long records. Filter edge effects at the start and end of a short request are not modelled.
- **Smoothing at the edges:** the API's 3-window moving average pads with zeros at a request's edges, while training averages over the windows that exist. Only the first and last window of a request differ.
- The operating points come from out-of-fold fold models but are applied to the final model. That is standard but approximate; the Siena check at those thresholds is the guard.
- No paired bootstrap against the beta HGB on identical windows yet.
