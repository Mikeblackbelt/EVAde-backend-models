# Deep sleep staging (Sleep-EDFx sleep-cassette)

This is the deep model for sleep staging, from section 5.2 of [training-plan-v2.md](../../../lit%20review/planning/training-plan-v2.md). It scores each 30 s EEG epoch as W, N1, N2, N3 or REM. **Its output is a sleep-stage estimate flagged for human review, not a diagnosis.** Every export is registered as `draft`.

**Status:** the code is written and tested on synthetic EDF files only. It has not run on real data yet.

## What it does

| Step | File | Detail |
|---|---|---|
| Download | `hpc/sleep_download.sh` | Runs on the login node. By default fetches the first 40 SC files (`LIMIT=40`): 20 nights of subjects SC400-SC409, about 1 GB, including the 10 beta nights. `LIMIT=0` fetches all 306 files (153 nights, about 8.1 GB). Source: `physionet.org/files/sleep-edfx/1.0.0/sleep-cassette/`. Details below. |
| Pack | `deep/sleep_pack.py` | A pure numpy EDF/EDF+ reader (the cluster venv has no pyedflib, and pyedflib needs a C build where no wheel exists). Details below. |
| Model | `deep/sleep_models.py` (`cnn_bigru`) | Input normalization inside the graph (per-epoch robust z, below), then a per-epoch 1D CNN in the TinySleepNet style (a 0.49 s first kernel with stride 6, then 3 more convolutions, 1,920 features per epoch), followed by a bidirectional GRU (128 per direction) over 20 consecutive epochs. About 2 M parameters. |
| Train | `deep/sleep_train.py` | Follows the resumable ECG pattern. Details below. |
| Export | `deep/sleep_export.py` | Exports `model.onnx` (`dynamo=False`; TorchScript `model.ts` as a fallback), `feature_names.json`, `metrics.json` and `manifest.json` (sha256 of every file, plus the input and output specs) into `export/sleep_deep`, the API's folder name. Details below. |
| SLURM | `hpc/sleep_prep.sbatch` (CPU), `hpc/sleep_deep.sbatch` (GPU) | Use `partnsf`/`qosnsf`, one GPU, `VOLTA\|AMPERE`, the uv venv and `PYTHONPATH=$PROJECT/code`. Configs run in sequence in one job. |

**Download details:**
- The file list and checksums come from PhysioNet's `SHA256SUMS.txt`.
- Polite: one file at a time, a 2 s pause between files, and a rate cap of 5 MB/s by default.
- Resumable: `wget -c` continues partial files, and verified files are skipped. A file that fails its checksum is deleted, so the next run fetches it again.
- Files get their arrival time as the timestamp, because scratch is purged by file age.
- The script writes a `VERIFIED` marker (with the file count) only when every listed file passed its checksum. `sleep_prep.sbatch` requires it.
- The sorted list is checked against PhysioNet's `SHA256SUMS.txt`: `LIMIT=20` is exactly the 10 beta nights (SC400-SC404) and `LIMIT=40` exactly the 20 nights of SC400-SC409.

**Packing details:**
- **Signals:** Fpz-Cz and Pz-Oz at 100 Hz, raw microvolts, no filtering, cut into 30 s epochs of 3,000 samples.
- **Labels:** the same as the beta (`hypnogram_to_epochs`): W, N1, N2, N3 (R&K stages 3 and 4 merged), REM.
- **Trimming and dropping:** trim on stages N1-REM to the sleep period plus 60 epochs (30 min) on each side, as `data.py sleep_features` does with `WAKE_PAD_EPOCHS=60`. Then drop movement and `?` epochs.
- **No normalization at this step:** `X` stays in raw microvolts. Trimming only chooses which epochs are trained and scored; it never feeds an input statistic.
- **Output:** `X` (float16, N x 2 x 3000), `y`, `subject` (for example SC400), `night` (for example SC4001E) and `epoch_index` (the original position, so gaps are visible).
- **Beta check:** the script prints the epoch count for SC400-SC404 next to the beta's 10,589.

**Training details:**
- Checkpoints: `last.pt` after every epoch and `best.pt` on the inner-validation kappa, loaded with `torch.load(weights_only=True)`.
- Precision: fp16 autocast on the GPU.
- On SIGUSR1 the job exits with code 99 at once and keeps the `last.pt` from the last finished epoch, so the one-cycle schedule resumes at the right step. The dropout RNG is not restored.
- **Input normalization, inside the model:** each 30 s epoch and channel is robust z-scored on its own: median and 1.4826 x MAD over its 3,000 samples (the median of an even count is the mean of the two middle values), scale floor 0.1, clipped to ±20 (`sleep_models.epoch_robust_z`, the first step of `encode()`).
  - It needs no night, no trimming and no labels. Training, evaluation, the ONNX graph and the API therefore compute exactly the same thing, whatever chunk of a night the API receives.
  - The API sends raw microvolts and the manifest says `normalization: {"type": "none", "in_model": {...}}`.
  - The cost: absolute amplitude between epochs is lost (relevant for N3). The waveform, the spectral shape within each epoch and the GRU context remain.
  - This replaces the earlier per-night rule, whose statistics came from the label-trimmed night and which the API could not reproduce.
- Loss: class-balanced cross-entropy on random 20-epoch windows taken inside one night.
- Evaluation: every epoch is encoded once, then softmax is averaged over 20-epoch GRU windows with a stride of 5.

**Export details:**
- **Input:** `x` with shape `[batch, seq, 2, 3000]` in raw microvolts, where batch and sequence length are dynamic.
- **Output:** `logits` with shape `[batch, seq, 5]`.
- **API contract** (`api-model-stubs`: `evade/api/README_deep_models.md` and `evade/models/deep_tasks.py` `SleepAdapter`):
  - `task` is `sleep_deep`, and the manifest sets `input.fs_hz` 100 and `input.channels` `["Fpz-Cz", "Pz-Oz"]`.
  - `tensor_shape` is `["batch", "seq", 2, 3000]`. The name `seq` makes the API pass the whole submitted sequence (1-20 epochs) in one call.
  - `normalization.type` is `"none"`, `output.type` is `multiclass`, and `output.activation` is `softmax`.
  - Checked once against that branch's loader on a synthetic smoke export (`EVADE_ALLOW_DRAFT=1`, model id `EVA-99`): the state was `available`. The API's probabilities matched PyTorch on the same raw epochs to within 1.3e-4, which is the API's 4-decimal rounding.
  - To serve it, copy `export/sleep_deep` to `<EVADE_MODEL_DIR>/sleep_deep/` and re-stamp `model_id` with a registry row.
- **Operating points:** argmax, plus "review if max softmax < t" at 5%, 10% and 20% of epochs flagged, set on the k-fold out-of-fold scores. They are only approximate for the final model.
- **Parity:** ONNX output is checked against PyTorch with onnxruntime at 3 shapes when onnxruntime is installed. The cluster venv does not have it, so the parity result in the manifest will say "not checked" there. Check parity again on the lab server or the API host.

## Evaluation protocols (one `sleep_train` call each)

| Protocol | Units | Selection | Purpose |
|---|---|---|---|
| `kfold` | 5 folds by subject (seeded permutation of the sorted subjects, written to `splits.csv` with its sha256; or a frozen `--splits` csv). Both nights of a subject stay in one fold. | 10% of the training subjects in each fold are held out (at least one); `best.pt` = best inner-validation kappa. The test fold is scored once. | The estimate. With the default 10 subjects, each fold tests 2 subjects. The 5 beta subjects are also reported separately (`beta_subjects_within_kfold`). |
| `beta5` | Leave-one-subject-out on SC400-SC404 only, the beta's 5 subjects and 10 nights | None (`--val-frac 0`, last epoch; the beta had no selection step either) | Like-for-like comparison with the beta: HGB subject-mean kappa 0.669 (0.545-0.785), logistic 0.683 (0.616-0.745) |
| `final` | One model on all subjects, 10% held out for checkpoint selection | Inner-validation kappa | The exported model. Its expected performance is the `kfold` estimate. |

`metrics.json` reports, for the out-of-fold predictions:
- pooled Cohen's kappa, macro F1, accuracy, per-class F1 and the confusion matrix;
- **subject bootstrap 95% CIs**: subjects (with their confusion matrices) are resampled with replacement, 2,000 times;
- the beta's own statistic: the mean of per-subject kappa with a subject bootstrap, the number the 0.669 and 0.683 figures are;
- per-subject scores;
- per-night total-sleep-time error in minutes, the quantity that feeds risk component B1 (measured only inside the label-trimmed sleep period, so it is optimistic);
- operating points;
- the beta baseline numbers (`baseline_beta`).

## Cluster commands

Run these on arrow as the cluster user, with `PROJECT=/scratch/$USER/evade`. The code is uploaded as in [hpc/README.md](../hpc/README.md) (`git archive` of a commit, unpacked to `$PROJECT/code`, hash in `code/COMMIT`). The venv comes from `setup_env_uv.sh`.

```
PROJECT=/scratch/$USER/evade
mkdir -p $PROJECT/slurm $PROJECT/data/raw
# 1. download (login node, network only; resumable, rerun the same line): default 40 files = 20 nights SC400-SC409
nohup bash $PROJECT/code/evade/training/hpc/sleep_download.sh > $PROJECT/data/raw/sleep-download.log 2>&1 &
tail -n 3 $PROJECT/data/raw/sleep-download.log    # ends with "verified 40/40, failed 0"; VERIFIED marker written

# 2. pack (CPU jobs): smoke = the 10 beta nights, then all downloaded nights
cd $PROJECT/code/evade/training/hpc
sbatch --export=ALL,SMOKE=1 sleep_prep.sbatch
tail -n 3 $PROJECT/data/sleep-sc/sleep_sc_smoke.pack.log   # "beta check: SC400-SC404 kept 10589 epochs ... MATCH"
sbatch sleep_prep.sbatch
tail -n 3 $PROJECT/data/sleep-sc/sleep_sc.pack.log         # 20 nights, 10 subjects

# 3. train (GPU job): smoke, then full (kfold 5 folds, beta5, final, export)
sbatch --export=ALL,SMOKE=1 --time=00:20:00 --signal=B:USR1@60 sleep_deep.sbatch
sbatch sleep_deep.sbatch
sacct -j <id> --format=JobID,State,Elapsed,MaxRSS,ExitCode
tail -n 20 $PROJECT/runs/sleep-deep-<commit10>/job_info.txt $PROJECT/runs/sleep-deep-<commit10>/run.log
# exit 99 = checkpointed at the time limit: resubmit the same sbatch line, and it resumes
```

The whole SC set (153 nights) is optional: `LIMIT=0` on the download, then the same prep and train lines. For that set, raise `--time` to about 04:00:00 and run with `EPOCHS=30` (`sbatch --export=ALL,EPOCHS=30 sleep_deep.sbatch`).

## Estimates

These are *estimates*, not measured. They are scaled from the earlier full-set estimates; replace them with the smoke-run timings.

| Step | Resources | Expected (default 20 nights) | Limit asked |
|---|---|---|---|
| Download | Login node, network only | about 1 GB at up to 5 MB/s plus 2 s per file: about 5 min, up to about 15 min if PhysioNet is slow | - |
| Prep | 2 CPUs, 16 GB (peak under 1 GB) | 2-4 min each (smoke and full) | 01:00:00 |
| Deep, smoke | 1 GPU, 8 CPUs, 32 GB | 3-5 min (mostly data loading and full-night predictions) | 00:20:00 |
| Deep, full | 1 GPU (V100 or A100), 8 CPUs, 32 GB RAM; under 1 GB of GPU memory for the data | about 20k epochs, so about 20-25 steps per training epoch. Training runs up to 100 epochs, with patience 15. `kfold` 5 x 3-5 min, `beta5` 5 x about 2 min, `final` about 5 min, export about 1 min: about 30-45 min in total | 02:00:00 |

The default set is about 20k epochs after trimming (*estimated* from the beta's 10,589 for 10 nights). The pack log gives the real number.

## Run folder

```
runs/sleep-deep-<commit10>/
  job_info.txt run.log
  kfold/{splits.csv, metrics.json, fold0..4/{best.pt, result.json, preds.npz}}
  beta5/{metrics.json, loso_SC400..SC404/...}
  final/{metrics.json, final/best.pt}
  export/sleep_deep/{model.onnx, feature_names.json, metrics.json, manifest.json}, export/SHA256SUMS
```

## Local test

`python -m pytest -q tests/test_deep_sleep.py` runs on CPU in about 1-1.5 minutes on a busy laptop (the end-to-end test takes about 40 s), on synthetic EDF files written by the test. It covers:
- reading, checked against mne's EDF reader;
- labels and trimming, checked against the beta functions;
- metrics, checked against sklearn;
- the in-model per-epoch normalization, checked against numpy, plus the same encoding for an epoch alone, inside a chunk or under a gain;
- pack, a 1-epoch smoke run of `kfold`, `beta5` and `final`, then export with ONNX parity and the API's sleep_deep manifest fields;
- an interrupt-and-resume check.

The reader was also checked once, outside the tests, on one real night (SC4001E0-PSG and SC4001EC-Hypnogram, with sha256 matching PhysioNet). The results:
- The signals match mne to within 1e-5 µV.
- The 154 annotations are identical.
- The night packs to 841 epochs (W 188, N1 58, N2 250, N3 220, REM 125).

## Limits and safety

- The data are healthy terrestrial sleep-lab recordings (sleep-cassette, ages 25-101) on two EEG channels. Nothing here says how the model behaves on wearable headbands or in spaceflight.
- The comparison with the beta uses the beta's published summary numbers. The HGB and logistic baselines have not been re-run on the same 5 folds; the plan's serve criterion needs that paired comparison.
- Outputs are stage estimates and sleep-time summaries for human review. The API never lets this model raise an override on its own. There is no sleep endpoint yet (plan section 12).
