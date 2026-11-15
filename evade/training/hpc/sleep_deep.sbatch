#!/bin/bash
# GPU job: deep sleep staging (per-epoch CNN + bi-GRU) on Sleep-EDFx SC (one GPU, one node). Configs run one after
# another in this single job:
#   1. kfold   10-fold by subject, Fpz-Cz + Pz-Oz (the estimate; also reports the 5 beta subjects separately)
#   2. beta5   leave-one-subject-out on SC400-SC404 only (the beta's protocol, for the 0.669 / 0.683 comparison)
#   3. final   all subjects (10% held out for checkpoint selection): the exported model
#   4. export  ONNX + manifest from final, metrics from kfold and beta5
#   5. kfold_fpzcz  secondary config: Fpz-Cz only (single-channel headbands), after the export
# Resubmitting the same file resumes (finished folds are skipped; SIGUSR1 near the time limit exits 99).
#   smoke: sbatch --export=ALL,SMOKE=1 --time=00:20:00 --signal=B:USR1@60 sleep_deep.sbatch
#   full:  sbatch sleep_deep.sbatch
#SBATCH --job-name=evade-sleep-deep
#SBATCH --partition=partnsf
#SBATCH --qos=qosnsf
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --constraint=VOLTA|AMPERE
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --signal=B:USR1@300
#SBATCH --output=/scratch/%u/evade/slurm/%x-%j.out
#SBATCH --error=/scratch/%u/evade/slurm/%x-%j.err

set -uo pipefail
PROJECT=/scratch/$USER/evade
PY=${EVADE_PY:-$PROJECT/venv/bin/python}
# the venv uses a self-contained Python installed with uv under $PROJECT/python (compute nodes lack the login
# node's /usr/bin/python3.12, and the cluster Python modules miss _ctypes/_bz2); no module load needed
export PYTHONPATH=$PROJECT/code:$PROJECT/pylib
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export EVADE_CODE_COMMIT=$(cat "$PROJECT/code/COMMIT" 2>/dev/null || echo unknown)
TAG=$(echo "$EVADE_CODE_COMMIT" | cut -c1-10)
RUN=$PROJECT/runs/sleep-deep-$TAG
DATA=$PROJECT/data/sleep-sc/sleep_sc.npz
SM=(); EPOCHS=30; BETA_EPOCHS=100
if [ "${SMOKE:-0}" = "1" ]; then
  RUN=$RUN-smoke; rm -rf "$RUN"; SM=(--smoke)
  [ -s "$DATA" ] || DATA=$PROJECT/data/sleep-sc/sleep_sc_smoke.npz
fi
mkdir -p "$RUN"; cd "$RUN"

{
  echo "job ${SLURM_JOB_ID:-?} on $(hostname) at $(date '+%F %T'), partition ${SLURM_JOB_PARTITION:-?}, smoke ${SMOKE:-0}"
  echo "code $EVADE_CODE_COMMIT data $DATA"
  nvidia-smi --query-gpu=name,memory.total,driver_version,compute_cap --format=csv,noheader
} >> job_info.txt 2>&1
GPU=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
case "$GPU" in *K20*|*K40*|*K80*|"") echo "unsupported or missing GPU '$GPU'" >> job_info.txt; exit 3 ;; esac
[ -s "$DATA" ] || { echo "missing packed data $DATA; run sleep_prep.sbatch" >> job_info.txt; exit 4; }

# forward SIGUSR1 from the batch shell to the running python step, which exits 99 (last.pt = last finished epoch)
child=0
trap '[ $child -ne 0 ] && kill -USR1 $child' USR1
step() {   # step <log> <args...>
  local log=$1; shift
  "$PY" "$@" >> "$log" 2>&1 &
  child=$!
  wait $child; local rc=$?
  while [ $rc -gt 128 ] && kill -0 $child 2>/dev/null; do wait $child; rc=$?; done   # wait again after the trap
  child=0
  echo "step rc=$rc: $*" >> job_info.txt
  return $rc
}

T=(-m evade.training.deep.sleep_train --data "$DATA" --resume ${SM[@]+"${SM[@]}"})
step run.log "${T[@]}" --protocol kfold --folds 10 --epochs $EPOCHS --run "$RUN/kfold" || exit $?
step run.log "${T[@]}" --protocol beta5 --val-frac 0 --epochs $BETA_EPOCHS --run "$RUN/beta5" || exit $?
step run.log "${T[@]}" --protocol final --epochs $EPOCHS --run "$RUN/final" || exit $?
if step run.log -m evade.training.deep.sleep_export --run "$RUN" --out "$RUN/export/sleep"; then
  (cd "$RUN/export" && find . -type f ! -name SHA256SUMS -exec sha256sum {} + > SHA256SUMS)
else
  echo "export failed (training results are kept)" >> job_info.txt
fi
step run.log "${T[@]}" --protocol kfold --folds 10 --epochs $EPOCHS --channels Fpz-Cz --run "$RUN/kfold_fpzcz" \
     || exit $?
echo "done $(date '+%F %T')" >> job_info.txt
