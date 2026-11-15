#!/bin/bash
# Session 2: sleep staging deep model on Sleep-EDFx sleep-cassette (training-plan-v2.md section 5.2).
# Primary (pre-registered): CNN-LSTM over 15-epoch sequences, Fpz-Cz + Pz-Oz at 100 Hz, per-night z-score.
# Runs in sequence in this single job: expanded subject-wise 10-fold CV (decision), final all-subject model,
# beta-comparable 5-subject LOSO track, then the single-channel (Fpz-Cz) secondary on the expanded folds.
# Finished folds are skipped on resubmit (DONE markers); SIGUSR1 near wall time -> checkpoint, exit 99.
# Needs: S2a output in data/sleep-edfx-prep (incl. baseline predictions) and the same split file.
# Smoke first:  sbatch --export=ALL,SMOKE=1 --time=00:15:00 --signal=B:USR1@60 s2_sleep_edfx.sbatch   (after an S2a smoke)
# Estimate: about 4 GPU-hours on a V100.
#SBATCH --job-name=evade-s2-sleep
#SBATCH --partition=partnsf
#SBATCH --qos=qosnsf
##SBATCH --account=CHANGE_ME_ACCOUNT
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:CHANGE_ME_GPU_TYPE:1
##SBATCH --constraint=CHANGE_ME_FEATURE
#SBATCH --mem=48G
#SBATCH --time=08:00:00
#SBATCH --signal=B:USR1@600
#SBATCH --output=/scratch/%u/evade/slurm/%x-%j.out
#SBATCH --error=/scratch/%u/evade/slurm/%x-%j.err

set -euo pipefail
PROJECT=/scratch/$USER/evade
source "$PROJECT/code/evade/training/hpc/lib.sh"
RUN=$(run_dir s2-sleep)
[ "${SMOKE:-0}" = "1" ] && rm -rf "$RUN"
mkdir -p "$RUN"; cd "$RUN"
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}

job_header
gpu_guard
require_code evade.training.deep.train evade.training.deep.evaluate evade.training.deep.export
SM=(); PREP=$DATA/sleep-edfx-prep
if [ "${SMOKE:-0}" = "1" ]; then SM=(--smoke); PREP=$PREP-smoke; fi
SPLIT=$CODE/data/splits/sleep_sc_10fold_v1.csv
[ -s "$PREP/SPLIT_SHA256" ] || { echo "no S2a output in $PREP" >> job_info.txt; exit 4; }
# S2a stored the split hash with the absolute path; a changed split file means S2a must be re-run
sha256sum -c --status "$PREP/SPLIT_SHA256" || { echo "split file changed since S2a; re-run S2a" >> job_info.txt; exit 4; }

COMMON=(--task sleep --data "$PREP" --seq-len 15 --resume --workers "${SLURM_CPUS_PER_TASK:-8}")

# decision track: expanded 10-fold, 10% of training subjects held out for early stopping, then the final model
step "$RUN/run.log" -m evade.training.deep.train "${COMMON[@]}" ${SM[@]+"${SM[@]}"} --arch cnn_lstm --channels Fpz-Cz,Pz-Oz \
     --protocol cv --splits "$SPLIT" --val-frac 0.1 --seeds 0 --final-model --run "$RUN/cnn_lstm"
# beta-comparable track: LOSO on SC400-SC404 only (continuity, not the decision metric)
step "$RUN/run.log" -m evade.training.deep.train "${COMMON[@]}" ${SM[@]+"${SM[@]}"} --arch cnn_lstm --channels Fpz-Cz,Pz-Oz \
     --protocol loso --subjects SC400,SC401,SC402,SC403,SC404 --seeds 0 --run "$RUN/cnn_lstm_beta5"
# secondary: single channel (wearable headband case)
step "$RUN/run.log" -m evade.training.deep.train "${COMMON[@]}" ${SM[@]+"${SM[@]}"} --arch cnn_lstm --channels Fpz-Cz \
     --protocol cv --splits "$SPLIT" --val-frac 0.1 --seeds 0 --run "$RUN/cnn_lstm_fpzcz"

# paired subject bootstrap of kappa against BOTH beta baselines (HGB and logistic) on the same folds
step "$RUN/run.log" -m evade.training.deep.evaluate --task sleep --run "$RUN" --primary cnn_lstm \
     --baseline-preds "$PREP/baseline" --n-boot 2000

step "$RUN/run.log" -m evade.training.deep.export --task sleep --run "$RUN/cnn_lstm" --use-final-model \
     --metrics "$RUN/metrics.json" --out "$RUN/export/sleep" --parity-tol 1e-4 --latency

finish_export "$RUN"
