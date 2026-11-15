# EVAde preprocessing

Turns downloaded open datasets into fixed windows plus a schema table (see `config.SCHEMA_COLUMNS`). Starting values come from `lit review/planning/implementation-pipeline.md` and are design assumptions.

## Run

```
pip install -r requirements.txt
python -m pytest -q tests                       # unit tests, synthetic signals, no data needed
EVADE_DATA=/path/to/evade_data python -m pytest -q tests   # adds smoke tests on real data
python -m evade.preprocessing.run --dataset mitdb --root <dataset folder> --out <out dir> [--limit N]
```

Datasets: `mitdb`, `bidmc`, `ptbxl`, `chbmit`, `siena`, `eegmat`, `sleepedfx`, `nhanes`. Output: `<out>/<dataset>_windows.parquet` (schema table) and `<out>/<dataset>/<record>.npz` (float32 windows). Raw data is never written here and never committed (`data/` is git-ignored). Dataset folders are read as data only, nothing in them is executed.

On the shared lab server, run through the guard (`guarded_job.sh`) with 4 threads or fewer, after the unit tests pass. See the `evade-server-rules` skill.

## What each step does

| Signal | Steps |
|---|---|
| ECG | band-pass 0.5-40 Hz, mains notch, resample to 250 Hz, 30 s windows (15 s step); simplified Pan-Tompkins QRS detection scored against MIT-BIH annotations (beat sensitivity and PPV, 150 ms tolerance) |
| PPG, respiration | band-pass 0.5-8 Hz (PPG) or 0.1-0.8 Hz (respiration), resample (250 Hz, 25 Hz), windows |
| EEG | map to a fixed 18-channel bipolar montage (selected from bipolar files, derived from monopolar files with T3/T4/T5/T6 renamed to T7/T8/P7/P8), band-pass 0.5-45 Hz, mains notch, resample to 250 Hz, 4 s windows (2 s step) |
| Seizure labels | window ictal (1) if at least 2 s overlap a seizure; non-ictal windows within 60 s before onset or 120 s after offset are excluded (-1); the rest are 0 |
| Sleep labels | 30 s epochs from the hypnogram: 0 W, 1 N1, 2 N2, 3 N3 (stages 3 and 4 merged), 4 REM, -1 unknown |
| Quality | index in [0, 1] from NaN, flat-line, saturation, and extreme-amplitude fractions; below 0.6 is flagged `low_quality` (kept, not deleted) |
| Folds | subject-wise fold from a hash of the participant ID (same person, same fold); PTB-XL also carries its official `strat_fold` |
| NHANES | femur and spine BMD merged with age and sex by SEQN into a bone reference table |

## Results on the downloaded data (smoke runs, 2026-11-14)

| Dataset | Check | Result |
|---|---|---|
| MIT-BIH, records 100-107 | QRS detector vs annotations | sensitivity 0.96-0.997 and PPV 0.99-1.00 on 7 of 8 records; **record 104 (paced, noisy) sensitivity 0.35** with PPV 0.996 |
| BIDMC | windows for PPG, ECG, respiration | runs; CSVs are in `bidmc_csv/` |
| EEG During Mental Arithmetic | rest vs task windows | rest 90 and task 30 windows per subject (180 s and 60 s) |
| CHB-MIT chb01 | seizure labels | `chb01_03` has 21 ictal windows for a 40 s seizure, as expected |
| Siena PN00 | seizure labels | 70 s seizure gives 35 ictal windows; **PN00-3 annotation is inconsistent** (seizure end time is an hour after the registration end) and is flagged `annotation_suspect` and excluded from onset |
| PTB-XL (first 50 records) | 100 Hz records | runs; one 10 s window per record, official fold and diagnostic class kept |
| Sleep-EDFx (sleep-cassette, subject SC400, 2 nights, `--limit 2`) | hypnogram labels | 5,480 epochs of 30 s; W 3,882, N1 117, N2 623, N3 517, REM 340, unknown 1. EEG channels Fpz-Cz and Pz-Oz read at their own 100 Hz rate |
| NHANES 2017-March 2020 DXA | bone reference table | 4,593 participants aged 50-80 (DXA femur was measured in this age range); femoral neck BMD for 3,545 (mean 0.770 g/cm2), total femur 3,545 (mean 0.929), spine 2,121 (mean 1.010) |

Bugs fixed during the Sleep-EDFx and NHANES runs:
- **Sleep-EDFx:** the reader took the most common sampling rate in the file, which was 1 Hz (temperature, event marker), so the EEG band-pass failed. `read_edf(..., label_prefix="EEG")` now reads only the EEG channels at their own rate.
- **NHANES:** the femur file has no `DXXOSBMD` (that is total spine, in the spine file). Total femur is `DXXOFBMD`, femoral neck `DXXNKBMD`. Checked against the 2017-2020 files.

Download status (server, 2026-11-14): one full pass plus one gap-filling re-run finished. Present: MIT-BIH, BIDMC, PTB-XL (zip and extracted), CHB-MIT chb01-chb05 (197 EDF files; empty folders exist for the other subjects), Siena PN00, PN01, PN03, PN05, PN06, PN07 (17 EDF files, 6 subjects), EEG During Mental Arithmetic, Sleep-EDFx (10 sleep-cassette nights from 5 subjects), NHANES DXA femur, spine and demographics. The log has two 404s and several transient 502s (mostly BIDMC CSVs in the re-run); unreadable files are skipped by the loaders and training builders, which print `SKIPPED`. About 27 GB in total.

Known limits: the QRS detector is a baseline, not validated; the quality index is a heuristic and rates most clean research data as 1.0; Siena seizure-list parsing was checked on PN00 only; NHANES gives a cross-sectional reference, not longitudinal loss. Beta models trained on these outputs are described in `evade/training/README.md`.
