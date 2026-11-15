#!/bin/bash
# Session 3: EEG seizure-suspect window CNN on CHB-MIT, Siena external test (training-plan-v2.md section 5.3).
# Per-person normalization kept: per-channel median / 1.4826*MAD of the raw signal, from each person's baseline
# windows (tracks B, C, Final) or from all of the person's windows exactly as in the beta (track A).
# Runs in sequence in this single job (no arrays):
#   A  beta-comparable: chb01-05 LOSO, beta file selection, beta normalization; Siena-6 at the CHB threshold
#   B  API-matched:     same data and folds as A, baseline-window normalization (decides the operating points)
#   C  expanded:        23 subjects, grouped 5-fold, test on every file of the test subjects (realistic FA/h)
#   Final               all 23 subjects -> Siena-14 (and Siena-6) once; this is the served model
# Secondary EEGNet runs on track B only. Finished folds are skipped on resubmit; SIGUSR1 -> checkpoint, exit 99.
# Needs: S3a output in data/seizure-prep (incl. baseline predictions and COUNTS_OK).
# Smoke first:  sbatch --export=ALL,SMOKE=1 --time=00:15:00 --signal=B:USR1@60 s3_seizure_chbmit.sbatch   (after an S3a smoke)
# Estimate: about 7 GPU-hours on a V100.
#SBATCH --job-name=evade-s3-seizure
#SBATCH --partition=partnsf
#SBATCH --qos=qosnsf
##SBATCH --account=CHANGE_ME_ACCOUNT
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:CHANGE_ME_GPU_TYPE:1
##SBATCH --constraint=CHANGE_ME_FEATURE
#SBATCH --mem=64G
#SBATCH --time=14:00:00
#SBATCH --signal=B:USR1@600
#SBATCH --output=/scratch/%u/evade/slurm/%x-%j.out
#SBATCH --error=/scratch/%u/evade/slurm/%x-%j.err

set -euo pipefail
PROJECT=/scratch/$USER/evade
source "$PROJECT/code/evade/training/hpc/lib.sh"
RUN=$(run_dir s3-seizure)
[ "${SMOKE:-0}" = "1" ] && rm -rf "$RUN"
mkdir -p "$RUN"; cd "$RUN"
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}

job_header
gpu_guard
require_code evade.training.deep.train evade.training.deep.evaluate evade.training.deep.export
SM=(); PREP=$DATA/seizure-prep
if [ "${SMOKE:-0}" = "1" ]; then SM=(--smoke); PREP=$PREP-smoke; fi
[ -s "$PREP/SPLIT_SHA256" ] || { echo "no S3a output in $PREP" >> job_info.txt; exit 4; }
if [ "${SMOKE:-0}" != "1" ] && [ ! -e "$PREP/COUNTS_OK" ]; then
  echo "S3a count check did not pass; see the S3a prep.log" >> job_info.txt; exit 4
fi
sha256sum -c --status "$PREP/SPLIT_SHA256" || { echo "split file changed since S3a; re-run S3a" >> job_info.txt; exit 4; }
SPLIT=$CODE/data/splits/chbmit_23subj_5fold_v1.csv
BETA5=chb01,chb02,chb03,chb04,chb05

COMMON=(--task seizure --data "$PREP" --resume --workers "${SLURM_CPUS_PER_TASK:-8}" --smooth 3 --max-epochs 20)
P=resnet1d_small

# A: beta-comparable
step "$RUN/run.log" -m evade.training.deep.train "${COMMON[@]}" ${SM[@]+"${SM[@]}"} --arch $P --track A \
     --protocol loso --subjects $BETA5 --norm all-windows --file-selection beta --inner-val 1 --seeds 0 --run "$RUN/A_$P"
# B: API-matched normalization (baseline windows only; baseline windows excluded from scoring for all models)
step "$RUN/run.log" -m evade.training.deep.train "${COMMON[@]}" ${SM[@]+"${SM[@]}"} --arch $P --track B \
     --protocol loso --subjects $BETA5 --norm baseline-windows --file-selection beta --inner-val 1 --seeds 0 --run "$RUN/B_$P"
# C: expanded, grouped 5-fold; train on the beta file-selection rule, test on every file of the test subjects
step "$RUN/run.log" -m evade.training.deep.train "${COMMON[@]}" ${SM[@]+"${SM[@]}"} --arch $P --track C \
     --protocol cv --splits "$SPLIT" --norm baseline-windows --file-selection beta --test-all-files --inner-val 2 \
     --seeds 0 --run "$RUN/C_$P"
# Final: all 23 subjects, then the Siena external test once (14 subjects, Siena-6 reported separately)
step "$RUN/run.log" -m evade.training.deep.train "${COMMON[@]}" ${SM[@]+"${SM[@]}"} --arch $P --track final \
     --protocol all --norm baseline-windows --file-selection beta --inner-val 2 --seeds 0 \
     --external siena --run "$RUN/final_$P"
# secondary: EEGNet on track B only (reported beside the primary, never swapped in after seeing test results)
step "$RUN/run.log" -m evade.training.deep.train "${COMMON[@]}" ${SM[@]+"${SM[@]}"} --arch eegnet --track B \
     --protocol loso --subjects $BETA5 --norm baseline-windows --file-selection beta --inner-val 1 --seeds 0 --run "$RUN/B_eegnet"

# window metrics, operating points at 1% / 5% pooled out-of-fold FPR within each track (the served model's thresholds
# come from track C: same normalization, 23 subjects), event metrics, paired subject bootstrap (2000) against the
# beta HGB in the same track
step "$RUN/run.log" -m evade.training.deep.evaluate --task seizure --run "$RUN" --primary $P \
     --baseline-preds "$PREP/baseline" --fprs 0.01,0.02,0.05,0.10 --served-thresholds-from C \
     --event-refractory-s 30 --n-boot 2000

step "$RUN/run.log" -m evade.training.deep.export --task seizure --run "$RUN/final_$P" --metrics "$RUN/metrics.json" \
     --out "$RUN/export/seizure" --parity-tol 1e-4 --latency

finish_export "$RUN"
