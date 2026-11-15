#!/bin/bash
# Session 1: ECG PTB-XL diagnostic superclass, deep model (training-plan-v2.md section 5.1).
# Split identical to the beta EVA-6: train folds 1-8, validate fold 9, test fold 10 once.
# Configs run one after another in this single job (no arrays): primary xresnet1d101 x 5 seeds,
# then resnet1d_wang x 1 seed, then xresnet1d101 on lead I only x 1 seed. Finished seeds are skipped on resubmit.
# Needs: data/ptbxl100/{ptbxl100.npz,VERIFIED} (packed on the lab server, plan section 4 step 3) and
#        data/baseline_preds/ecg_eva6_fold10.parquet (beta predictions for the paired bootstrap).
# Smoke first (15 min, real data, tiny subset):
#   sbatch --export=ALL,SMOKE=1 --time=00:15:00 --signal=B:USR1@60 s1_ecg_ptbxl.sbatch
# Estimate: about 2.5 GPU-hours on a V100 (replace with smoke epoch time x epochs). Resubmit the same file to resume.
#SBATCH --job-name=evade-s1-ecg
#SBATCH --partition=partnsf
#SBATCH --qos=qosnsf
##SBATCH --account=CHANGE_ME_ACCOUNT
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:CHANGE_ME_GPU_TYPE:1
##SBATCH --constraint=CHANGE_ME_FEATURE
#SBATCH --mem=32G
#SBATCH --time=05:00:00
#SBATCH --signal=B:USR1@600
#SBATCH --output=/scratch/%u/evade/slurm/%x-%j.out
#SBATCH --error=/scratch/%u/evade/slurm/%x-%j.err

set -euo pipefail
PROJECT=/scratch/$USER/evade
source "$PROJECT/code/evade/training/hpc/lib.sh"
RUN=$(run_dir s1-ecg)
[ "${SMOKE:-0}" = "1" ] && rm -rf "$RUN"
mkdir -p "$RUN"; cd "$RUN"
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}

job_header
gpu_guard
require_code evade.training.deep.train evade.training.deep.evaluate evade.training.deep.export
require_data ptbxl100
# beta predictions are needed by evaluate; check now, not after hours of GPU work
[ "${SMOKE:-0}" = "1" ] || [ -s "$DATA/baseline_preds/ecg_eva6_fold10.parquet" ] \
  || { echo "missing $DATA/baseline_preds/ecg_eva6_fold10.parquet (re-score EVA-6 on the lab server, put it here)" >> job_info.txt; exit 4; }

SM=(); [ "${SMOKE:-0}" = "1" ] && SM=(--smoke)
COMMON=(--task ecg --data "$DATA/ptbxl100/ptbxl100.npz" --protocol official-folds
        --train-folds 1-8 --val-fold 9 --test-fold 10 --resume --workers "${SLURM_CPUS_PER_TASK:-8}")

# primary (pre-registered): xresnet1d101, 12 leads, 5 seeds
step "$RUN/run.log" -m evade.training.deep.train "${COMMON[@]}" ${SM[@]+"${SM[@]}"} \
     --arch xresnet1d101 --leads all --seeds 0,1,2,3,4 --run "$RUN/xresnet1d101"
# secondary configs, same job, sequential
step "$RUN/run.log" -m evade.training.deep.train "${COMMON[@]}" ${SM[@]+"${SM[@]}"} \
     --arch resnet1d_wang --leads all --seeds 0 --run "$RUN/resnet1d_wang"
step "$RUN/run.log" -m evade.training.deep.train "${COMMON[@]}" ${SM[@]+"${SM[@]}"} \
     --arch xresnet1d101 --leads I --seeds 0 --run "$RUN/xresnet1d101_leadI"

# evaluation: fold 10 opened here once; paired patient bootstrap (2000) against the beta EVA-6 predictions
step "$RUN/run.log" -m evade.training.deep.evaluate --task ecg --run "$RUN" --primary xresnet1d101 \
     --select-seed-on val --baseline-preds "$DATA/baseline_preds/ecg_eva6_fold10.parquet" --n-boot 2000

# servable folder: manifest.json, model.onnx, feature_names.json, metrics.json (+ SHA256SUMS below)
step "$RUN/run.log" -m evade.training.deep.export --task ecg --run "$RUN/xresnet1d101" --metrics "$RUN/metrics.json" \
     --out "$RUN/export/ecg" --parity-tol 1e-4 --latency

finish_export "$RUN"
