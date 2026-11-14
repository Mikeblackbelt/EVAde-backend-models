# EVAde beta training

Three CPU models trained on open data with subject-wise evaluation, each with a logistic-regression baseline in the same protocol. **Beta: not tuned, not clinically validated, registered as `draft`.** The GPU was not used: when training started, another user held about 129 of 143 GB of GPU memory, below the 72 GB this project may use.

## Run

```
python -m evade.training.train --task seizure --chb <chbmit> --siena <siena> --out <dir> --threads 1
python -m evade.training.train --task sleep   --sleep <sleep-edfx> --out <dir>
python -m evade.training.train --task ecg     --ptbxl <ptb-xl folder> --out <dir> --threads 8
```

Add `--smoke` for a tiny end-to-end check. On the shared server, run through `guarded_job.sh` (the seizure task needs about 30-40 GB with one worker because some Siena recordings are single 1.7 GB files; more workers were killed by the guard at 34, 55 and 56 GB). Each task writes `metrics.json`, `model.joblib` (load only artifacts you built), `feature_names.json`, and a features parquet cache.

## Models

| ID | Task | Features | Model | Evaluation |
|---|---|---|---|---|
| EVA-1 / EVA-2 | seizure-suspect window flag (4 s windows, 2 s step, 18-channel bipolar montage) | per channel: log and relative band power (delta to gamma), line length, log variance; channel-mean relative powers (131) | HistGradientBoosting, balanced sample weights | leave-one-subject-out on CHB-MIT; external test on Siena |
| EVA-3 / EVA-4 | sleep staging (30 s epochs, W/N1/N2/N3/REM) | band powers (delta, theta, alpha, sigma, beta) of Fpz-Cz and Pz-Oz, plus the previous and next epoch | HistGradientBoosting | leave-one-subject-out on Sleep-EDFx sleep-cassette |
| EVA-5 / EVA-6 | PTB-XL diagnostic superclass (NORM, MI, STTC, CD, HYP), 12-lead 10 s at 100 Hz | per lead: 4 log band powers, mean, SD, skew, kurtosis, line length | one-vs-rest HistGradientBoosting | official folds: train 1-8, test 10 |

## Results (2026-11-14)

### Sleep staging (Sleep-EDFx, 5 subjects, 10 nights, 10,589 epochs after trimming wake)

| Model | Accuracy | Macro F1 | Cohen's kappa (pooled) | Kappa across subjects, mean (95% CI) |
|---|---|---|---|---|
| HistGradientBoosting | 0.758 | 0.712 | 0.668 | 0.669 (0.545-0.785) |
| Logistic regression baseline | 0.767 | 0.730 | 0.685 | 0.683 (0.616-0.745) |

The first run kept all wake epochs (18,612 of 27,597 epochs were W) and gave kappa 0.80; that number was inflated by easy out-of-bed wake, so the run was repeated with the standard trim (sleep period plus 30 min of wake on each side). With only 5 subjects the boosted model does not beat the linear baseline; the baseline is the honest choice for now.

### PTB-XL superclass (train 17,4xx, test fold 10)

| Model | Macro AUROC (95% CI, patient bootstrap) | NORM | MI | STTC | CD | HYP |
|---|---|---|---|---|---|---|
| HistGradientBoosting | 0.867 (0.858-0.878) | 0.897 | 0.831 | 0.867 | 0.860 | 0.879 |
| Logistic regression baseline | 0.851 (0.842-0.862) | 0.890 | 0.813 | 0.851 | 0.847 | 0.855 |

Published deep models on the same split report higher macro AUROC (about 0.93, recalled from the PTB-XL benchmark paper and not re-checked here), so these hand-built features are a baseline, not a competitive model.

### Seizure-suspect windows

SEIZURE_RESULTS

## Limits

- Window-level seizure metrics, not event-based SzCORE scoring; false-positive windows per hour count overlapping windows, not alarms.
- Seizure-free files are subsampled (3 per subject), so false-alarm rates are not representative of continuous monitoring.
- Few subjects everywhere; intervals are wide and the subject is the unit of generalization.
- Terrestrial clinical data only (pediatric CHB-MIT, adult Siena, healthy Sleep-EDFx, hospital PTB-XL). Nothing here says how the models behave in spaceflight or on wearables.
- Models output flags for human review; the API never lets a beta model raise an override on its own.
