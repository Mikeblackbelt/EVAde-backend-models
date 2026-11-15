#!/bin/bash
# Session 2a (CPU only, no GPU): Sleep-EDFx sleep-cassette preparation and baseline re-run (training-plan-v2.md 5.2).
# 1. per-night float16 arrays (Fpz-Cz, Pz-Oz at 100 Hz) + epoch index, beta trimming (sleep period +/- 30 min wake)
# 2. beta HGB (EVA-3 arch) and logistic baselines re-run on the frozen expanded split, predictions saved
# 3. the beta-comparable 5-subject LOSO features, to check epoch counts against the beta (10,589)
# Needs: data/sleep-edfx/{*-PSG.edf,*-Hypnogram.edf,VERIFIED} and code/data/splits/sleep_sc_10fold_v1.csv
# Estimate: about 1 h on 8 cores. Output on scratch is purged after ~2 weeks: run S2 within a few days.
#SBATCH --job-name=evade-s2a-sleep-prep
#SBATCH --partition=partnsf
#SBATCH --qos=qosnsf
##SBATCH --account=CHANGE_ME_ACCOUNT
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/%u/evade/slurm/%x-%j.out
#SBATCH --error=/scratch/%u/evade/slurm/%x-%j.err

set -euo pipefail
PROJECT=/scratch/$USER/evade
source "$PROJECT/code/evade/training/hpc/lib.sh"
RUN=$(run_dir s2a-sleep-prep)
[ "${SMOKE:-0}" = "1" ] && rm -rf "$RUN"
mkdir -p "$RUN"; cd "$RUN"
export OMP_NUM_THREADS=1

job_header
require_code evade.training.deep.prep evade.training.train
require_data sleep-edfx
SPLIT=$CODE/data/splits/sleep_sc_10fold_v1.csv
[ -s "$SPLIT" ] || { echo "missing split file $SPLIT" >> job_info.txt; exit 4; }
SM=(); OUT=$DATA/sleep-edfx-prep
if [ "${SMOKE:-0}" = "1" ]; then SM=(--smoke); OUT=$OUT-smoke; fi

step "$RUN/prep.log" -m evade.training.deep.prep --task sleep ${SM[@]+"${SM[@]}"} \
     --raw "$DATA/sleep-edfx" --out "$OUT" --splits "$SPLIT" --workers "${SLURM_CPUS_PER_TASK:-8}"

# beta pipeline on the same folds (baseline for the paired comparison); predictions saved next to the prep output
step "$RUN/baseline.log" -m evade.training.train --task sleep ${SM[@]+"${SM[@]}"} --sleep "$DATA/sleep-edfx" \
     --splits "$SPLIT" --save-preds --out "$OUT/baseline" --threads "${SLURM_CPUS_PER_TASK:-8}"

sha256sum "$SPLIT" > "$OUT/SPLIT_SHA256"
echo "done $(date '+%F %T'); prep output in $OUT" >> job_info.txt
