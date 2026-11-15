#!/bin/bash
# Session 3a (CPU only, no GPU): CHB-MIT (24 cases chb01-chb24, 23 subjects; chb24 treated as its own subject) and Siena (14 subjects) preparation and the
# beta HGB baseline on the same windows and folds (training-plan-v2.md section 5.3).
# 1. per-record float16 memmaps, 18-channel bipolar montage at 250 Hz, + window index (4 s / 2 s, beta labels and
#    buffers, baseline-window marks: first 10 min of each person's first seizure-free recording); chb21 -> subject chb01
# 2. beta features for every file; beta HGB + logistic re-run for tracks B, C and Final, predictions saved
# 3. track A count check: chb01-05 with the beta file selection must give 97,624 windows / 992 ictal;
#    Siena-6 must give 120,032 / 386 (stops with exit 5 if not)
# Memory: single Siena files reach 1.7 GB and took 30-40 GB with one worker in the beta, so large files run with
# one worker (the prep code decides per file size). Estimate: 4-6 h on 8 cores.
# Needs: data/chbmit, data/siena-scalp-eeg (each with VERIFIED) and code/data/splits/chbmit_23subj_5fold_v1.csv
#SBATCH --job-name=evade-s3a-seizure-prep
#SBATCH --partition=partnsf
#SBATCH --qos=qosnsf
##SBATCH --account=CHANGE_ME_ACCOUNT
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=/scratch/%u/evade/slurm/%x-%j.out
#SBATCH --error=/scratch/%u/evade/slurm/%x-%j.err

set -euo pipefail
PROJECT=/scratch/$USER/evade
source "$PROJECT/code/evade/training/hpc/lib.sh"
RUN=$(run_dir s3a-seizure-prep)
[ "${SMOKE:-0}" = "1" ] && rm -rf "$RUN"
mkdir -p "$RUN"; cd "$RUN"
export OMP_NUM_THREADS=1

job_header
require_code evade.training.deep.prep evade.training.train
require_data chbmit siena-scalp-eeg
SPLIT=$CODE/data/splits/chbmit_23subj_5fold_v1.csv
[ -s "$SPLIT" ] || { echo "missing split file $SPLIT" >> job_info.txt; exit 4; }
SM=(); OUT=$DATA/seizure-prep
if [ "${SMOKE:-0}" = "1" ]; then SM=(--smoke); OUT=$OUT-smoke; fi

# resumable per record: records with a finished memmap and index are skipped
step "$RUN/prep.log" -m evade.training.deep.prep --task seizure ${SM[@]+"${SM[@]}"} \
     --chb "$DATA/chbmit" --siena "$DATA/siena-scalp-eeg" --out "$OUT" --splits "$SPLIT" \
     --subject-alias chb21=chb01 --baseline-minutes 10 --max-free-train 3 --resume \
     --workers "${SLURM_CPUS_PER_TASK:-8}" --large-file-gb 1.0
# track A check (skipped in smoke mode): the prep module exits 5 and writes the counts to prep.log on a mismatch,
# which stops this job before the baseline step; investigate before running S3
if [ "${SMOKE:-0}" != "1" ]; then
  step "$RUN/prep.log" -m evade.training.deep.prep --task seizure --check-beta-counts --out "$OUT" \
       --expect-chb 97624:992 --expect-siena6 120032:386
fi

# beta pipeline (EVA-1 arch) on the same windows and folds, both normalizations, predictions saved
step "$RUN/baseline.log" -m evade.training.train --task seizure ${SM[@]+"${SM[@]}"} \
     --chb "$DATA/chbmit" --siena "$DATA/siena-scalp-eeg" --splits "$SPLIT" --windows "$OUT/index.parquet" \
     --norm all-windows,baseline-windows --save-preds --out "$OUT/baseline" --threads 1

sha256sum "$SPLIT" > "$OUT/SPLIT_SHA256"
echo "done $(date '+%F %T'); prep output in $OUT" >> job_info.txt
