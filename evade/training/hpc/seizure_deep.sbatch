#!/bin/bash
# GPU job: deep seizure-suspect window model on CHB-MIT with the Siena external test (one GPU, one node).
# Everything runs one after another in this single job (no arrays):
#   1. beta5_loso    leave-one-subject-out over chb01-05 (the beta's subjects and file rule), 1 inner-val subject
#   2. beta5_final   train on chb01-05, Siena once at the beta5 out-of-fold thresholds   (compare with EVA-2)
#   3. all_loso      leave-one-subject-out over all 24 cases = 23 subjects (chb21 grouped with chb01),
#                    2 inner-val subjects; its out-of-fold scores set the served operating points (1% and 5% FPR)
#   4. all_final     train on all 23 subjects, Siena-14 (and the beta's Siena-6) once at the all_loso thresholds
#   5. export        ONNX (TorchScript fallback) + manifest of all_final
# Resubmitting the same file resumes: finished folds are skipped, SIGUSR1 near the time limit checkpoints mid-epoch.
# Needs: data/seizure256 from seizure_prep.sbatch.
#   smoke: sbatch --export=ALL,SMOKE=1 --time=00:30:00 --signal=B:USR1@60 seizure_deep.sbatch  (after a prep smoke)
#   full:  sbatch seizure_deep.sbatch
# Estimate on a V100 (design arithmetic, replace with smoke seconds per epoch): beta5 ~0.5 h, all_loso 23 folds x
# 6-10 min ~3-4 h, all_final + Siena ~0.3 h; about 4-5 h in total, 12 h requested. RAM: all_loso holds ~13 GB of
# float16 windows; Siena is scored after the CHB-MIT windows are freed.
#SBATCH --job-name=evade-seizure-deep
#SBATCH --partition=partnsf
#SBATCH --qos=qosnsf
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --constraint=VOLTA|AMPERE
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --signal=B:USR1@300
#SBATCH --output=/scratch/%u/evade/slurm/%x-%j.out
#SBATCH --error=/scratch/%u/evade/slurm/%x-%j.err

set -uo pipefail
PROJECT=/scratch/$USER/evade
PY=${EVADE_PY:-$PROJECT/venv/bin/python}
# the uv venv (self-contained Python under $PROJECT/python); no module load needed
export PYTHONPATH=$PROJECT/code
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export EVADE_CODE_COMMIT=$(cat "$PROJECT/code/COMMIT" 2>/dev/null || echo unknown)
TAG=$(echo "$EVADE_CODE_COMMIT" | cut -c1-10)
RUN=$PROJECT/runs/seizure-deep-$TAG
DATA=$PROJECT/data/seizure256
SM=()
if [ "${SMOKE:-0}" = "1" ]; then RUN=$RUN-smoke; DATA=$DATA-smoke; rm -rf "$RUN"; SM=(--smoke); fi
mkdir -p "$RUN"; cd "$RUN"

{
  echo "job ${SLURM_JOB_ID:-?} on $(hostname) at $(date '+%F %T'), partition ${SLURM_JOB_PARTITION:-?}, smoke ${SMOKE:-0}"
  echo "code $EVADE_CODE_COMMIT"
  nvidia-smi --query-gpu=name,memory.total,driver_version,compute_cap --format=csv,noheader
} >> job_info.txt 2>&1
GPU=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
case "$GPU" in *K20*|*K40*|*K80*|"") echo "unsupported or missing GPU '$GPU'" >> job_info.txt; exit 3 ;; esac
[ -s "$DATA/pack_summary.json" ] || { echo "missing packed data in $DATA; run seizure_prep.sbatch" >> job_info.txt; exit 4; }

# forward SIGUSR1 from the batch shell to the running python step, which checkpoints and exits 99
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

T=(-m evade.training.deep.seizure_train --data "$DATA" --arch seizure_resnet1d --epochs 20 --patience 5 \
   --sampler balanced --loss bce --norm nonictal --n-boot 2000 --resume ${SM[@]+"${SM[@]}"})
step run.log "${T[@]}" --protocol loso --cases beta5 --inner-val 1 --run "$RUN/beta5_loso" || exit $?
step run.log "${T[@]}" --protocol final --cases beta5 --inner-val 1 --external siena \
     --thresholds-from "$RUN/beta5_loso/metrics.json" --run "$RUN/beta5_final" || exit $?
step run.log "${T[@]}" --protocol loso --cases all --inner-val 2 --run "$RUN/all_loso" || exit $?
step run.log "${T[@]}" --protocol final --cases all --inner-val 2 --external siena \
     --thresholds-from "$RUN/all_loso/metrics.json" --run "$RUN/all_final" || exit $?
step run.log -m evade.training.deep.seizure_export --run "$RUN/all_final" --out "$RUN/export/seizure" \
     || echo "export failed (training results are kept)" >> job_info.txt
grep -h "^DONE" run.log >> job_info.txt
echo "done $(date '+%F %T')" >> job_info.txt
