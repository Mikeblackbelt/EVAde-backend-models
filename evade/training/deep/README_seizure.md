# Deep seizure-suspect windows: CHB-MIT, with Siena as an external test

This is the deep version of the beta EEG seizure model (EVA-2) for the CUNY HPCC cluster. It follows the seizure section of [training-plan-v2.md](../../../lit%20review/planning/training-plan-v2.md) (section 5.3) and the patterns of the deep ECG code in this folder.

**Claim:** the model marks 4 s EEG windows as *seizure-suspect*, as a flag for a person to review. It is not a diagnosis and not a seizure detector. Every number below is a validation or external-test result on research data.

**Status:** written and tested on synthetic data on a CPU. Nothing has run on the cluster yet, and no result exists.

## Files

| File | What it does |
|---|---|
| `evade/training/hpc/seizure_download.sh` | Polite, resumable `wget` of all of CHB-MIT (24 cases, summaries, `.seizures` files) and Siena (14 subjects). It checks every file against the dataset's `SHA256SUMS.txt` and writes `VERIFIED` when all pass. **Email hpchelp before you run it:** it moves about 63 GB. |
| `evade/training/deep/seizure_pack.py` | CPU packing into one float16 `.npz` shard per CHB-MIT case or Siena subject, with labels, record IDs, a seizure table and per-person normalization statistics |
| `evade/training/deep/seizure_models.py` | `seizure_resnet1d`: a channel-wise conv, then cross-channel mixing, then a temporal residual stack (`seizure_tiny` is for tests) |
| `evade/training/deep/seizure_train.py` | Leave-one-subject-out folds and the final model. Resumable: SIGUSR1 saves a checkpoint and exits 99. Writes `metrics.json`. |
| `evade/training/deep/seizure_eval.py` | Window AUROC and AP, operating points, false-positive windows per hour, event-level sensitivity, false alarms per hour, and subject bootstrap CIs |
| `evade/training/deep/seizure_export.py` | ONNX export (`dynamo=False`, with a TorchScript fallback), an onnxruntime parity check, `manifest.json` and `SHA256SUMS` |
| `evade/training/hpc/seizure_prep.sbatch` | The CPU job that runs the packing |
| `evade/training/hpc/seizure_deep.sbatch` | The GPU job: every protocol in sequence, then the export |
| `tests/test_deep_seizure.py` | End-to-end test on synthetic EDFs: pack, LOSO smoke training with a fake SIGUSR1 and resume, the final model with an external set, and export. It also checks the metric helpers against sklearn. |

## Data and packing

Packing reuses the beta preprocessing:
- `read_edf`; Siena is read with `label_prefix="EEG"`, so the EKG and SpO2 channels are left out.
- The fixed 18-channel bipolar montage:
  - CHB-MIT uses `select_bipolar`. Missing channels are zero-filled, and the number of channels present is stored for each record.
  - Siena uses `bipolar_from_monopolar`.
- `preprocess_eeg`: a 0.5-45 Hz zero-phase Butterworth band-pass and a Q 30 notch, then resampling. The notch is 60 Hz for CHB-MIT and 50 Hz for Siena. The output rate is set explicitly to **256 Hz**: CHB-MIT is native, and Siena is resampled from 512 Hz. The code runs one channel at a time to bound memory.
- The beta labels:
  - A window is ictal (1) when it overlaps a seizure by at least 2 s.
  - Windows from 60 s before a seizure to 120 s after it are -1.
  - Siena annotations that end after the recording does are never labelled ictal (the PN00-3 case).
- The beta file rule from `evade/training/data.py` (`chb_jobs`, `siena_jobs`): every file with a seizure, plus 3 seizure-free files per subject. `--max-free -1` keeps every file (see the open questions).

Windows are 4 s (1,024 samples) with a 2 s step. They are clipped to ±2,000 µV and stored as float16 in `data/seizure256/{chbmit,siena}/<case>.npz` with these arrays:
- `X`, `y`, `rec`, `start_s`;
- `records`, `rec_dur_s`, `rec_channels`;
- the seizure table `sz_rec`, `sz_start_s` and `sz_end_s`;
- `norm_median` and `norm_mad`.

Label -1 windows are kept, because smoothing and event scoring need them. Window metrics and training leave them out.

**Per-person normalization.** Each person's raw signal is scaled per channel with that person's median and 1.4826 x MAD.
- The statistics come from the person's **non-ictal (label 0) windows**: up to 4,000 windows, evenly spaced.
- The MAD has a floor of 0.5 µV, so zero-filled channels stay at 0. The scaled signal is clipped to ±20.
- The API does the same with the user's seizure-free baseline recording.

The beta used **all** of a person's windows, ictal ones included. That leaks a little label information into the features and cannot be reproduced at serving time. The shards also store the all-windows statistics (`norm_median_all`, `norm_mad_all`), so `--norm all` can reproduce the beta rule as an ablation. It is not used by default.

chb21 is chb01 recorded 1.5 years later. It keeps its own shard and its own baseline statistics (a different session), and the training code groups it with chb01, so the person is never on both sides of a split. chb24 is treated as its own person. That is UNVERIFIED, because it is not in SUBJECT-INFO.

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
- **Augmentation:** a circular shift of up to ±0.5 s and amplitude scaling by 0.8-1.2.
- **Length:** at most 20 epochs of up to 200,000 windows each, with early stopping (patience 5).
- **Precision:** fp16 autocast on CUDA (V100).

Model selection uses validation subjects held out from the **training** side. Each LOSO fold uses 1 for beta5 and 2 for the all-subjects runs. The test subject is scored once, with `best.pt`.

## Protocols (one GPU job, in sequence)

| Step | Data | Split | Purpose |
|---|---|---|---|
| `beta5_loso` | chb01-05, beta file rule | LOSO over 5 subjects (3 train + 1 val + 1 test) | the same subjects as the beta (EVA-2) |
| `beta5_final` | train on chb01-05 | Siena once, at the beta5 out-of-fold thresholds | the same setup as the beta's Siena number |
| `all_loso` | all 24 cases = 23 subjects | LOSO over 23 subjects (20 train + 2 val + 1 test) | more subjects; its out-of-fold scores set the **served operating points** |
| `all_final` | train on all 23 subjects | Siena-14 once, plus the beta's Siena-6 (PN00, 01, 03, 05, 06, 07), at the `all_loso` thresholds | the external result and the exported model |

## Evaluation (`metrics.json`)

All scores are smoothed with a centered 3-window moving average inside each recording, as the beta did. The unsmoothed AUROC and AP are reported too.

| Metric | Definition |
|---|---|
| Window AUROC and AP | pooled out-of-fold, plus per subject and the subject mean |
| Operating points | thresholds at 1% (`suspect_seizure`) and 5% (`review`) pooled out-of-fold FPR, from `evade.training.metrics.threshold_at_fpr` (tie-safe "higher" quantile; flag = score > threshold). External sets use the CHB-MIT thresholds unchanged. A Siena-derived threshold is reported only as a labelled oracle, never as a headline. |
| Sensitivity | subject mean (the beta's number) and pooled, at each operating point |
| False-positive windows per hour | flagged non-ictal windows / (non-ictal windows x 2 s / 3600). With overlapping windows, one minute of false activity is about 30 windows. |
| Event-level sensitivity | a seizure counts as detected if any window overlapping it is flagged |
| False alarms per hour | runs of flagged non-ictal windows, merged within 30 s |
| CIs | 95% subject bootstrap with 2,000 resamples: subjects are resampled with replacement, the pooled metrics are recomputed, and thresholds stay fixed |

**Beta baseline (EVA-2, HGB on band-power features)**, stored in every `metrics.json` as `baseline_beta_eva2`:
- CHB-MIT chb01-05 LOSO: pooled AUROC 0.967, subject AUROC 0.986 (0.969-0.997), sensitivity at 1% FPR 0.35 (0.20-0.51), 8.8 FP windows per hour;
- Siena-6: AUROC 0.920 and AP 0.173.

`beta5_loso` and `beta5_final` are the direct comparison. One difference remains: the beta normalized with all windows, and this model normalizes with non-ictal windows (see above).

This is window and simple event scoring in the spirit of SzCORE, not a certified SzCORE evaluation.

## Export (`runs/<run>/export/seizure/`)

The export folder holds:
- `model.onnx` (or `model.ts`);
- `feature_names.json` (the 18 channels in order);
- `metrics.json`;
- `manifest.json`;
- `SHA256SUMS`.

The manifest records:
- the sha256 of every file;
- the **input spec**: shape (batch, 18, 1024), 256 Hz, 4 s windows with a 2 s step, µV, montage, filter, raw clip, and the normalization formula and the rule for its statistics;
- the **output spec**: sigmoid, 3-window smoothing, and the operating points (threshold, target FPR, out-of-fold subject-mean sensitivity, event sensitivity, FP windows per hour, false alarms per hour);
- the onnxruntime parity result;
- the Siena headline next to the beta.

`model_id` stays `UNASSIGNED` and `status` stays `draft` until someone adds an EVA row to `MODEL_REGISTRY.md` and reviews the model. Serving the model also needs the ONNX loader from `api-models-storage`.

## Cluster commands

The main session runs these commands on the login node. `$PROJECT` is `/scratch/$USER/evade`, and the code is the committed revision unpacked in `$PROJECT/code` with `COMMIT`.

```bash
# 0. once: pyedflib in the uv venv (setup_env_uv.sh now installs it; for an existing venv:)
$PROJECT/tools/bin/uv pip install --python $PROJECT/venv/bin/python pyedflib
mkdir -p $PROJECT/slurm
# 1. download (after hpchelp agreed), on the login node, 1.5-4 h, about 63 GB; re-run after interruptions
nohup bash $PROJECT/code/evade/training/hpc/seizure_download.sh > $PROJECT/data/seizure_download.log 2>&1 &
# 2. prep (CPU): smoke, then full (1.5-3 h, 8 cores, 64 GB; output about 23 GB)
sbatch --export=ALL,SMOKE=1 --time=00:30:00 $PROJECT/code/evade/training/hpc/seizure_prep.sbatch
sbatch $PROJECT/code/evade/training/hpc/seizure_prep.sbatch
# 3. train (1 GPU): smoke, then full (about 4-5 h on a V100, 12 h requested, 64 GB RAM)
sbatch --export=ALL,SMOKE=1 --time=00:30:00 --signal=B:USR1@60 $PROJECT/code/evade/training/hpc/seizure_deep.sbatch
sbatch $PROJECT/code/evade/training/hpc/seizure_deep.sbatch      # exit 99 = checkpointed; resubmit the same line
```

Read `data/seizure256/pack.log` before training. Check these counts against the beta:
- chb01-05 should give about 97,624 windows and 992 ictal ones (the beta ran at 250 Hz, so a few windows per file may differ).
- Siena should give 47 seizures, minus the inconsistent PN00-3 annotation.

The pack prints a warning for any listed Siena file with no parsed seizure, because the list parser was checked on PN00 only.

## Tests

```bash
python -m pytest -q tests/test_deep_seizure.py
```

The tests run on CPU on synthetic data in under a minute. If pyedflib is missing, as on Windows with Python 3.13 where it has no wheel, the synthetic EDFs are fed through an in-memory stand-in for `read_edf` with the same return contract. The cluster venv reads real EDF files.

## Limits and open questions

- **The beta file rule** keeps every seizure file and 3 seizure-free files per subject. That subsamples seizure-free time, so false alarms per hour are optimistic or noisy.
  - Packing every CHB-MIT file (`--max-free -1`) is about 1.8 M windows, or about 65 GB in float16. That does not fit the 64 GB node with the current in-RAM loader.
  - A continuous-record format (windows cut on the fly) would halve the size. Streaming the test subject from disk would make all-file testing possible.
- Normalization from non-ictal windows uses the test person's labels to pick their seizure-free time. The serving equivalent is a known seizure-free baseline recording. A stricter variant, the first 10 minutes of the first seizure-free file as in plan v2, is not implemented.
- chb24's person identity is UNVERIFIED.
- The operating points come from out-of-fold fold models but are applied to the final model, which is trained on all subjects. That is standard but approximate.
- No paired bootstrap against the beta HGB on identical windows yet. The plan's S3a step, which re-runs the beta pipeline on these exact windows, is not part of this pipeline.
