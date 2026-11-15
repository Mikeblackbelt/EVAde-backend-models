# Deep sleep staging (Sleep-EDFx sleep-cassette)

This is the deep model for sleep staging, from section 5.2 of [training-plan-v2.md](../../../lit%20review/planning/training-plan-v2.md). It scores each 30 s EEG epoch as W, N1, N2, N3 or REM. **Its output is a sleep-stage estimate flagged for human review, not a diagnosis.** Every export is registered as `draft`.

**Status:** the code is written and tested on synthetic EDF files only. It has not run on real data yet.

## What it does

| Step | File | Detail |
|---|---|---|
| Download | `hpc/sleep_download.sh` | Runs on the login node. Fetches the 306 SC `*-PSG.edf` and `*-Hypnogram.edf` files (153 nights, about 8.1 GB) from `physionet.org/files/sleep-edfx/1.0.0/sleep-cassette/`. Details below. |
| Pack | `deep/sleep_pack.py` | A pure numpy EDF/EDF+ reader (the cluster venv has no pyedflib, and pyedflib needs a C build where no wheel exists). Details below. |
| Model | `deep/sleep_models.py` (`cnn_bigru`) | A per-epoch 1D CNN in the TinySleepNet style (a 0.49 s first kernel with stride 6, then 3 more convolutions, 1,920 features per epoch), followed by a bidirectional GRU (128 per direction) over 20 consecutive epochs. About 2 M parameters. |
| Train | `deep/sleep_train.py` | Follows the resumable ECG pattern. Details below. |
| Export | `deep/sleep_export.py` | Exports `model.onnx` (`dynamo=False`; TorchScript `model.ts` as a fallback), `feature_names.json`, `metrics.json` and `manifest.json` (sha256 of every file, plus the input and output specs). Details below. |
| SLURM | `hpc/sleep_prep.sbatch` (CPU), `hpc/sleep_deep.sbatch` (GPU) | Use `partnsf`/`qosnsf`, one GPU, `VOLTA\|AMPERE`, the uv venv and `PYTHONPATH=$PROJECT/code`. Configs run in sequence in one job. |

**Download details:**
- The file list and checksums come from PhysioNet's `SHA256SUMS.txt`.
- Polite: one file at a time, a 2 s pause between files, and a rate cap of 5 MB/s by default.
- Resumable: `wget -c` continues partial files, and verified files are skipped. A file that fails its checksum is deleted, so the next run fetches it again.
- Files get their arrival time as the timestamp, because scratch is purged by file age.
- The script writes a `VERIFIED` marker only when the full set of 306 files is verified, never after a `LIMIT=` run. `sleep_prep.sbatch` requires the marker for a full pack.

**Packing details:**
- **Signals:** Fpz-Cz and Pz-Oz at 100 Hz, raw microvolts, no filtering, cut into 30 s epochs of 3,000 samples.
- **Labels:** the same as the beta (`hypnogram_to_epochs`): W, N1, N2, N3 (R&K stages 3 and 4 merged), REM.
- **Trimming and dropping:** trim on stages N1-REM to the sleep period plus 60 epochs (30 min) on each side, as `data.py sleep_features` does with `WAKE_PAD_EPOCHS=60`. Then drop movement and `?` epochs.
- **Output:** `X` (float16, N x 2 x 3000), `y`, `subject` (for example SC400), `night` (for example SC4001E) and `epoch_index` (the original position, so gaps are visible).
- **Beta check:** the script prints the epoch count for SC400-SC404 next to the beta's 10,589.

**Training details:**
- Checkpoints: `last.pt` after every epoch and `best.pt` on the inner-validation kappa, loaded with `torch.load(weights_only=True)`.
- Precision: fp16 autocast on the GPU.
- On SIGUSR1 the job exits with code 99 at once and keeps the `last.pt` from the last finished epoch, so the one-cycle schedule resumes exactly.
- Input normalization: a robust z-score per night and channel (median and 1.4826 x MAD over that night's epochs, clipped to ±20). The API can compute this from the night itself.
- Loss: class-balanced cross-entropy on random 20-epoch windows taken inside one night.
- Evaluation: every epoch is encoded once, then softmax is averaged over 20-epoch GRU windows with a stride of 5.

**Export details:**
- **Input:** `x` with shape `[batch, seq, channels, 3000]`, where batch and sequence length are dynamic.
- **Output:** `logits` with shape `[batch, seq, 5]`.
- **Operating points:** argmax, plus "review if max softmax < t" at 5%, 10% and 20% of epochs flagged on the out-of-fold scores.
- **Parity:** ONNX output is checked against PyTorch with onnxruntime at 3 shapes when onnxruntime is installed. The cluster venv does not have it, so the parity result in the manifest will say "not checked" there. Check parity again on the lab server or the API host.

## Evaluation protocols (one `sleep_train` call each)

| Protocol | Units | Selection | Purpose |
|---|---|---|---|
| `kfold` | 10 folds by subject (seeded permutation of the sorted subjects, written to `splits.csv` with its sha256; or a frozen `--splits` csv). Both nights of a subject stay in one fold. | 10% of the training subjects in each fold are held out; `best.pt` = best inner-validation kappa. The test fold is scored once. | The decision metric. The 5 beta subjects are also reported separately (`beta_subjects_within_kfold`). |
| `beta5` | Leave-one-subject-out on SC400-SC404 only, the beta's 5 subjects and 10 nights | None (`--val-frac 0`, last epoch; the beta had no selection step either) | Like-for-like comparison with the beta: HGB subject-mean kappa 0.669 (0.545-0.785), logistic 0.683 (0.616-0.745) |
| `final` | One model on all subjects, 10% held out for checkpoint selection | Inner-validation kappa | The exported model. Its expected performance is the `kfold` estimate. |

`metrics.json` reports, for the out-of-fold predictions:
- pooled Cohen's kappa, macro F1, accuracy, per-class F1 and the confusion matrix;
- **subject bootstrap 95% CIs**: subjects (with their confusion matrices) are resampled with replacement, 2,000 times;
- the beta's own statistic: the mean of per-subject kappa with a subject bootstrap, the number the 0.669 and 0.683 figures are;
- per-subject scores;
- per-night total-sleep-time error in minutes, the quantity that feeds risk component B1;
- operating points;
- the beta baseline numbers (`baseline_beta`).

## Cluster commands

Run these on arrow as the cluster user, with `PROJECT=/scratch/$USER/evade`. The code is uploaded as in [hpc/README.md](../hpc/README.md) (`git archive` of a commit, unpacked to `$PROJECT/code`, hash in `code/COMMIT`). The venv comes from `setup_env_uv.sh`.

```
# 1a. smoke download (login node): the first 20 files are exactly the 10 beta nights (SC400-SC404), about 0.5 GB
mkdir -p $PROJECT/slurm $PROJECT/data/raw
LIMIT=20 bash $PROJECT/code/evade/training/hpc/sleep_download.sh       # no VERIFIED marker for a partial set
# 1b. smoke pack of those 10 nights: the log must say "beta check: SC400-SC404 kept 10589 epochs ... MATCH"
cd $PROJECT/code/evade/training/hpc
sbatch --export=ALL,SMOKE=1 sleep_prep.sbatch
tail -n 3 $PROJECT/data/sleep-sc/sleep_sc_smoke.pack.log

# 2. full download (login node; network only), in tmux or with nohup; it skips the files it already verified
nohup bash $PROJECT/code/evade/training/hpc/sleep_download.sh > $PROJECT/data/raw/sleep-download.log 2>&1 &
tail -n 3 $PROJECT/data/raw/sleep-download.log         # ends with "verified 306/306, failed 0"; rerun to resume

# 3. full pack (CPU job; needs the VERIFIED marker)
sbatch sleep_prep.sbatch
tail -n 3 $PROJECT/data/sleep-sc/sleep_sc.pack.log      # "beta check: SC400-SC404 kept N epochs (beta: 10589)"

# 4. train (GPU job): smoke, then full
sbatch --export=ALL,SMOKE=1 --time=00:20:00 --signal=B:USR1@60 sleep_deep.sbatch
sbatch sleep_deep.sbatch
sacct -j <id> --format=JobID,State,Elapsed,MaxRSS,ExitCode
tail -n 20 $PROJECT/runs/sleep-deep-<commit10>/job_info.txt $PROJECT/runs/sleep-deep-<commit10>/run.log
# exit 99 = checkpointed at the time limit: resubmit the same sbatch line, and it resumes
```

## Estimates

These are *estimates*, not measured. Replace them with the smoke-run timings.

| Step | Resources | Expected | Limit asked |
|---|---|---|---|
| Download | Login node, network only | 8.1 GB at up to 5 MB/s plus 2 s per file: about 40 min, up to about 1.5 h if PhysioNet is slow | - |
| Prep | 2 CPUs, 16 GB (peak about 6 GB: about 2.3 GB of float16 epochs, held twice while stacking) | 10-20 min | 01:00:00 |
| Deep, smoke | 1 GPU, 8 CPUs, 32 GB | 3-5 min (mostly data loading and full-night predictions) | 00:20:00 |
| Deep, full | 1 GPU (V100 16-32 GB or A100), 8 CPUs, 32 GB RAM; about 3 GB of GPU memory for the data plus a small batch | V100: `kfold` about 1 h (10 x about 5 min), `beta5` about 5 min, `final` about 6 min, export about 1 min, `kfold_fpzcz` about 1 h, so about 2-2.5 h in total; 3x slower would still fit | 08:00:00 |

The epoch count is about 195k for the 153 trimmed nights (*recalled* for this trimming, UNVERIFIED). The pack log gives the real number.

## Run folder

```
runs/sleep-deep-<commit10>/
  job_info.txt run.log
  kfold/{splits.csv, metrics.json, fold0..9/{best.pt, result.json, preds.npz}}
  beta5/{metrics.json, loso_SC400..SC404/...}
  final/{metrics.json, final/best.pt}
  export/sleep/{model.onnx, feature_names.json, metrics.json, manifest.json}, export/SHA256SUMS
  kfold_fpzcz/...                   (secondary: Fpz-Cz only)
```

## Local test

`python -m pytest -q tests/test_deep_sleep.py` runs on CPU in about 1-1.5 minutes on a busy laptop (the end-to-end test takes about 40 s), on synthetic EDF files written by the test. It covers:
- reading, checked against mne's EDF reader;
- labels and trimming, checked against the beta functions;
- metrics, checked against sklearn;
- pack, a 1-epoch smoke run of `kfold`, `beta5` and `final`, then export with ONNX parity;
- an interrupt-and-resume check.

The reader was also checked once, outside the tests, on one real night (SC4001E0-PSG and SC4001EC-Hypnogram, with sha256 matching PhysioNet). The results:
- The signals match mne to within 1e-5 µV.
- The 154 annotations are identical.
- The night packs to 841 epochs (W 188, N1 58, N2 250, N3 220, REM 125).

## Limits and safety

- The data are healthy terrestrial sleep-lab recordings (sleep-cassette, ages 25-101) on two EEG channels. Nothing here says how the model behaves on wearable headbands or in spaceflight.
- The comparison with the beta uses the beta's published summary numbers. The HGB and logistic baselines have not been re-run on the same 10 folds; the plan's serve criterion needs that paired comparison.
- Outputs are stage estimates and sleep-time summaries for human review. The API never lets this model raise an override on its own. There is no sleep endpoint yet (plan section 12).
