#!/bin/bash
# Session 4 (OPTIONAL, only after S1-S3): single-lead beat model on MIT-BIH (AAMI inter-patient split DS1 -> DS2)
# and PPG heart-rate check on BIDMC (leave-one-recording-out, Bland-Altman) (training-plan-v2.md section 5.4).
# Both small; a deep model may not beat rules, which is a valid result. Feature HGB baseline in the same split.
# Needs: data/mitdb, data/bidmc (each with VERIFIED; sent as one tar each, unpacked with tar -xmf).
# Estimate: about 1 GPU-hour.
#SBATCH --job-name=evade-s4-hrv-ppg
#SBATCH --partition=partnsf
#SBATCH --qos=qosnsf
##SBATCH --account=CHANGE_ME_ACCOUNT
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:CHANGE_ME_GPU_TYPE:1
##SBATCH --constraint=CHANGE_ME_FEATURE
#SBATCH --mem=16G
#SBATCH --time=03:00:00
#SBATCH --signal=B:USR1@300
#SBATCH --output=/scratch/%u/evade/slurm/%x-%j.out
#SBATCH --error=/scratch/%u/evade/slurm/%x-%j.err

set -euo pipefail
PROJECT=/scratch/$USER/evade
source "$PROJECT/code/evade/training/hpc/lib.sh"
RUN=$(run_dir s4-hrv-ppg)
[ "${SMOKE:-0}" = "1" ] && rm -rf "$RUN"
mkdir -p "$RUN"; cd "$RUN"
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}

job_header
gpu_guard
require_code evade.training.deep.train evade.training.deep.evaluate evade.training.deep.export
require_data mitdb bidmc
SM=(); [ "${SMOKE:-0}" = "1" ] && SM=(--smoke)

step "$RUN/run.log" -m evade.training.deep.train --task beats ${SM[@]+"${SM[@]}"} --data "$DATA/mitdb" \
     --protocol aami-ds1-ds2 --arch cnn1d_beat --leads MLII --seeds 0,1,2 --resume --run "$RUN/beats"
step "$RUN/run.log" -m evade.training.deep.train --task ppg_hr ${SM[@]+"${SM[@]}"} --data "$DATA/bidmc" \
     --protocol loro --arch cnn1d_ppg --seeds 0 --resume --run "$RUN/ppg_hr"
step "$RUN/run.log" -m evade.training.deep.evaluate --task beats --run "$RUN/beats" --with-hgb-baseline --n-boot 2000
step "$RUN/run.log" -m evade.training.deep.evaluate --task ppg_hr --run "$RUN/ppg_hr" --bland-altman
step "$RUN/run.log" -m evade.training.deep.export --task beats --run "$RUN/beats" --metrics "$RUN/beats/metrics.json" \
     --out "$RUN/export/beats" --parity-tol 1e-4 --latency

finish_export "$RUN"
