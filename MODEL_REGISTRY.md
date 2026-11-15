# Model Registry

Every retained model architecture and trained model in this repository gets a unique identifier of the form **`EVA-[n]`**, where `n` is a positive integer assigned in order of creation. This file is the single source of truth.

**Next free ID: EVA-16**

## Rules

1. **One sequence for everything.** Architectures and trained models share a single counter. An ID's number says only "this was created Nth"; the `Type` column says what it is.
2. **Types.**
   - `arch`: a model architecture definition (the layer structure, inputs and outputs, and training recipe, written as a spec).
   - `model`: a trained instance of an architecture (a specific set of weights or fitted parameters). It must list its parent `arch` ID.
   - A new architecture is a new ID. A retrain of the same architecture on new data or with a new seed is a new `model` ID.
3. **Assigned at creation, in order.** Take the number in "Next free ID", add a registry row in the same commit that introduces the artifact, then increment "Next free ID". No gaps are planned, but a gap is allowed if an ID is withdrawn.
4. **Never reused, never renumbered.** A retired or withdrawn entry keeps its row and ID. Its `Status` changes; the number does not.
5. **Written as `EVA-n`** with no zero padding (`EVA-7`, `EVA-112`). Use the ID in filenames, model cards, logs, and result tables (for example `EVA-7_model_card.md`).
6. **Candidates are not numbered.** Architectures listed in [lit review/model-architectures](lit%20review/model-architectures/README.md) are candidates until someone creates one. They receive an ID when the spec or the trained artifact is first committed.
7. **No ID for throwaway experiments.** A scratch run that is not retained does not get an ID. Anything cited in a result table, report, or comparison must have one.
8. **Required for every row:** the git commit, the data version, the split version, the evaluation protocol, and the headline metric with a confidence interval. The headline metric is the module's established protocol metric (see [fitness-assessment.md](lit%20review/planning/fitness-assessment.md)), not a training loss. A model with no subject-wise evaluation is registered with `Status = draft` and may not be reported as a result.

## Status values

`draft` (created, not yet evaluated), `active` (evaluated and in use or comparison), `superseded` (replaced by a later ID, kept for the record), `retired` (no longer used), `withdrawn` (found faulty, for example data leakage; never to be cited as a result).

## Registry

| ID | Type | Name | Parent | Module | Task | Created | Commit | Data version | Split version | Evaluation | Metric (95% CI) | Status | Card |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| EVA-1 | arch | seizure-window-hgb | - | EEG | seizure-suspect flag per 4 s window | 2026-11-14 | 8404bb1 | - | - | - | - | draft | [training README](evade/training/README.md) |
| EVA-2 | model | seizure-window-hgb beta | EVA-1 | EEG | seizure-suspect flag per 4 s window | 2026-11-14 | 8404bb1 | CHB-MIT chb01-05 (listing 344e9a12), Siena 6 subjects (e593ab2a) | leave-one-subject-out, 5 CHB-MIT subjects; Siena external | window AUROC, AP, sensitivity at 1% FPR | subject AUROC 0.986 (0.969-0.997); Siena AUROC 0.920 | draft | [training README](evade/training/README.md) |
| EVA-3 | arch | sleep-stage-hgb | - | EEG | 5-class sleep staging per 30 s epoch | 2026-11-14 | c98e542 | - | - | - | - | draft | [training README](evade/training/README.md) |
| EVA-4 | model | sleep-stage-hgb beta | EVA-3 | EEG | 5-class sleep staging per 30 s epoch | 2026-11-14 | c98e542 | Sleep-EDFx SC400-SC404 (b99fb35a) | leave-one-subject-out, 5 subjects | accuracy, macro F1, Cohen's kappa | subject kappa 0.669 (0.545-0.785); logistic baseline 0.683 | draft | [training README](evade/training/README.md) |
| EVA-5 | arch | ptbxl-superclass-hgb | - | ECG | 5 diagnostic superclasses, 12-lead 10 s | 2026-11-14 | c98e542 | - | - | - | - | draft | [training README](evade/training/README.md) |
| EVA-6 | model | ptbxl-superclass-hgb beta | EVA-5 | ECG | 5 diagnostic superclasses, 12-lead 10 s | 2026-11-14 | c98e542 | PTB-XL 1.0.3 (f52b3af6) | official strat_fold: train 1-8, test 10 | macro AUROC, patient bootstrap | 0.867 (0.858-0.878) | draft | [training README](evade/training/README.md) |
| EVA-7 | arch | spe-onset-24h | - | radiation | P(>=10 MeV protons reach 10 pfu within 24 h), quiet hours only; 19 GOES proton and X-ray history features (48 h lookback); logistic or gradient boosting chosen by blocked CV, Platt step | 2026-11-14 | a99b862 | - | - | - | - | draft | [spaceweather README](evade/spaceweather/README.md) |
| EVA-8 | model | spe-onset-24h logistic beta | EVA-7 | radiation | SPE onset within 24 h (hourly) | 2026-11-14 | a99b862 | NOAA NCEI GOES-8..15 SEM 5-min averages 1998-2019, XRS unscaled /0.7 (hourly table hash 7998390607791038415) | time-forward: train 1998-2010, Platt 2011-2012, test 2013-2019, 72 h embargo; also 9-block leave-one-block-out | Brier, BSS, AUROC, reliability; 27-day block bootstrap; persistence and climatology baselines | test Brier 0.00594 (0.00332-0.00895), AUROC 0.918 (0.870-0.952) vs persistence 0.714; 15 test onsets; over-forecasts about 2x | draft | [spaceweather README](evade/spaceweather/README.md) |
| EVA-9 | model | spe-onset-24h logistic v2 (GOES-R, background-corrected) | EVA-7 | radiation | SPE onset within 24 h (hourly); not served (replaced by EVA-12, same split, without the correction) | 2026-11-15 | 1fa7ec7 | NOAA NCEI GOES-8..15 CSV 1998-2020-03 + GOES-16/18 SGPS/XRS L2 2020-11..2026-11 (integral integrated from differential); per-satellite per-year quiet floors | time-forward: train 1998-2016, test 2017-01..2026-11, 72 h embargo; no recalibration | Brier and AUROC vs persistence, overall and GOES-R era; reliability | test Brier 0.0118, AUROC 0.890 (persistence 0.763); GOES-R era (37 onsets) Brier 0.0172, AUROC 0.832 (persistence 0.0169 / 0.741); over-forecasts ~1.5x | draft | [spaceweather README](evade/spaceweather/README.md) |
| EVA-10 | arch | spe-onset-24h-cme | - | radiation | EVA-7 features plus 3 CME features (fastest CME speed in 72 h, wide-CME flag, fastest wide CME); CDAW catalogue in training, NASA DONKI when serving | 2026-11-15 | 1fa7ec7 | - | - | - | - | draft | [spaceweather README](evade/spaceweather/README.md) |
| EVA-11 | model | spe-onset-24h-cme logistic | EVA-10 | radiation | SPE onset within 24 h (hourly); not served, weights not retained (rerun train.py to rebuild) | 2026-11-15 | 1fa7ec7 | as EVA-9 plus SOHO/LASCO CDAW CME catalogue (3 h availability lag) | as EVA-9 | as EVA-9; 27-day block bootstrap of the difference vs EVA-9 | test Brier 0.0117, AUROC 0.892; GOES-R era AUROC +0.005 vs EVA-9 (95% CI +0.0001 to +0.012), Brier difference CI includes 0 | draft | [spaceweather README](evade/spaceweather/README.md) |
| EVA-12 | model | spe-onset-24h logistic v2 (GOES-R, no background correction) | EVA-7 | radiation | SPE onset within 24 h (hourly); served | 2026-11-15 | 302a584 | as EVA-9 (GOES-8..15 1998-2020-03 + GOES-16/18 2020-11..2026-11), no background correction | as EVA-9 | as EVA-9 | test Brier 0.0113, AUROC 0.892 (persistence 0.0117 / 0.763); GOES-R era (37 onsets) Brier 0.0165, AUROC 0.836 (persistence 0.0169 / 0.741) | draft | [spaceweather README](evade/spaceweather/README.md) |
| EVA-13 | model | ptbxl-superclass resnet1d_wang (deep ECG, served as `ecg_deep`) | - (resnet1d_wang from evade/training/deep/models.py; arch not numbered separately) | ECG | 5 diagnostic superclasses, 12-lead 10 s at 100 Hz; ONNX, 2 MB; served by `/v1/predict/ecg-deep` | 2026-11-15 | 853ad95 (training code; the run's commit is in its manifest `code_commit`) | PTB-XL 1.0.3 superclass labels | official strat_fold: train 1-8, val 9 (selection), test 10 (scored once) | macro AUROC, patient bootstrap (1000 resamples); beta EVA-6 as the comparison baseline | test 0.927 (0.920-0.934) vs beta EVA-6 0.867 (0.858-0.878); trained on the CUNY HPCC cluster | draft | [deep training README](evade/training/README.md) |
| EVA-14 | model | sleep-edfx 5-stage deep sleep staging (served as `sleep_deep`) | - (epoch CNN + bi-GRU from evade/training/deep/sleep_models.py) | EEG (Fpz-Cz, Pz-Oz) | W/N1/N2/N3/REM per 30 s epoch, 1-20 epochs per request at 100 Hz; per-epoch robust z-score inside the ONNX graph | 2026-11-15 | 3560393 (training code; run commit in manifest) | Sleep-EDFx sleep-cassette SC400-SC409 (20 nights, 10 subjects) | 5-fold by subject (val 10% of training subjects); also the beta's 5-subject leave-one-out | Cohen's kappa, subject bootstrap | 5-fold kappa 0.756 (0.716-0.796), macro F1 0.787; beta protocol 0.626 (0.536-0.717) vs beta HGB 0.669 / logistic 0.683; trained on the CUNY HPCC cluster (V100) | draft | [sleep README](evade/training/deep/README_sleep.md) |
| EVA-15 | model | chb-mit seizure-suspect deep 1D ResNet (`seizure_deep`, **not served**) | - (seizure_resnet1d) | EEG 18-ch bipolar | 4 s windows at 256 Hz, per-window robust normalization | 2026-11-15 | 3560393 (training code; run commit in manifest) | CHB-MIT chb01-05 (beta file rule), Siena PN00/01/03/05/06/07 external | leave-one-subject-out on chb01-05; Siena external | window AUROC, events detected | LOSO pooled AUROC 0.711 (0.636-0.851), events 11/26; Siena 0.869 (0.845-0.894), events 13/17; worse than beta EVA-2 (0.967 / 0.920), so EVA-2 stays the served seizure model and `seizure-deep` stays on placeholders; trained on the lab H200 | draft (not served) | [seizure README](evade/training/deep/README_seizure.md) |

<!-- Deep ECG note: the larger xresnet1d101 on the same split scored test macro AUROC 0.933 (0.926-0.940). It is not the served default (EVA-13 is: 2 MB ONNX, CI overlaps) and has no ID; give it the next free ID before it is served or cited in a result table (rule 7). -->

## Entry template

```
ID:            EVA-[n]
Type:          arch | model
Name:          short descriptive name
Parent:        EVA-[m] (models only)
Module:        radiation | vitals | ecg | eeg | behavioral | fusion | bone
Task:          what it predicts or flags
Created:       YYYY-MM-DD
Commit:        git hash
Data version:  dataset names, versions, license notes
Split version: subject-wise / time-forward split definition and hash
Evaluation:    protocol (subject-wise, external site, event-based, etc.)
Metric:        headline metric with 95% CI (subject-level bootstrap)
Status:        draft | active | superseded | retired | withdrawn
Card:          link to the model card
Notes:         known failure modes, leakage checks, fairness gate result
```
