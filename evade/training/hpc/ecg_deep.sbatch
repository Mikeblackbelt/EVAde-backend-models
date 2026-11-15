#!/bin/bash
# GPU job: deep ECG on PTB-XL (one GPU, one node). Configs run one after another in this single job:
# xresnet1d101 x seeds, then resnet1d_wang x 1 seed, then export of the selected xresnet1d101 seed.
# Resubmitting the same file resumes (finished seeds are skipped; SIGUSR1 near the time limit exits 99 and a
# resubmit resumes from the last finished epoch).
#   smoke: sbatch --export=ALL,SMOKE=1 --time=00:20:00 ecg_deep.sbatch
#   full:  sbatch ecg_deep.sbatch
#SBATCH --job-name=evade-ecg-deep
#SBATCH --partition=partnsf
#SBATCH --qos=qosnsf
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --constraint=VOLTA|AMPERE
#SBATCH --mem=48G
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
RUN=$PROJECT/runs/ecg-deep-$TAG
SM=(); EPOCHS=50; SEEDS=0,1,2,3,4
if [ "${SMOKE:-0}" = "1" ]; then RUN=$RUN-smoke; rm -rf "$RUN"; SM=(--smoke); fi
mkdir -p "$RUN"; cd "$RUN"

{
  echo "job ${SLURM_JOB_ID:-?} on $(hostname) at $(date '+%F %T'), partition ${SLURM_JOB_PARTITION:-?}, smoke ${SMOKE:-0}"
  echo "code $EVADE_CODE_COMMIT"
  nvidia-smi --query-gpu=name,memory.total,driver_version,compute_cap --format=csv,noheader
} >> job_info.txt 2>&1
GPU=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
case "$GPU" in *K20*|*K40*|*K80*|"") echo "unsupported or missing GPU '$GPU'" >> job_info.txt; exit 3 ;; esac
[ -s "$PROJECT/data/ptbxl100/ptbxl100.npz" ] || { echo "missing packed data; run ecg_prep.sbatch" >> job_info.txt; exit 4; }

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

DATA=$PROJECT/data/ptbxl100/ptbxl100.npz
step run.log -m evade.training.deep.train --data "$DATA" --arch xresnet1d101 --seeds $SEEDS --epochs $EPOCHS \
     --run "$RUN/xresnet1d101" --resume ${SM[@]+"${SM[@]}"} || exit $?
step run.log -m evade.training.deep.train --data "$DATA" --arch resnet1d_wang --seeds 0 --epochs $EPOCHS \
     --run "$RUN/resnet1d_wang" --resume ${SM[@]+"${SM[@]}"} || exit $?
step run.log -m evade.training.deep.export --run "$RUN/xresnet1d101" --arch xresnet1d101 --out "$RUN/export/ecg_deep" \
     || echo "export failed (training results are kept)" >> job_info.txt
echo "done $(date '+%F %T')" >> job_info.txt
