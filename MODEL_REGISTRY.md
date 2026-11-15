# Model Registry

Every retained model architecture and trained model in this repository gets a unique identifier of the form **`EVA-[n]`**, where `n` is a positive integer assigned in order of creation. This file is the single source of truth.

**Next free ID: EVA-9**

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
