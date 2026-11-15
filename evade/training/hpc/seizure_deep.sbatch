#!/bin/bash
# GPU job: deep seizure-suspect window model on CHB-MIT with the Siena external test (one GPU, one node).
# Everything runs one after another in this single job (no arrays):
#   1. loso     leave-one-subject-out over CASES (default beta5 = chb01-05, the beta's subjects and file rule;
#               1 inner-val subject; CASES=all uses every packed case, chb21 grouped with chb01, 2 inner-val);
#               its out-of-fold scores set the served operating points (review 5% FPR, suspect_seizure 1% FPR)
#   2. final    train on CASES, then score the packed Siena subjects (default the beta's Siena-6) once at the loso
#               thresholds: the external operating-point check, compared with EVA-2
#   3. export   ONNX + manifest in the API's seizure_deep form (export/seizure_deep), onnxruntime parity required
# Resubmitting the same command resumes: finished folds are skipped, SIGUSR1 near the time limit checkpoints.
# Needs: data/seizure256 from seizure_prep.sbatch; onnxruntime in the venv (setup_env_uv.sh) for the export.
#   smoke: sbatch --export=ALL,SMOKE=1 --time=00:30:00 --signal=B:USR1@60 seizure_deep.sbatch  (after a prep smoke)
#   full:  sbatch seizure_deep.sbatch
# RUN_DIR=<dir> keeps using one run folder after a code re-upload (default: runs/seizure-deep-<commit>).
# Estimate on a V100 for beta5 (design arithmetic; replace with the smoke's seconds per epoch): ~98k windows, 5 folds
# x up to 20 epochs x ~230 steps of 256 (~50 ms per step) = 5 x ~4-6 min; final + Siena-6 scoring + bootstrap +
# export ~10-15 min; about 0.5-1 h in total, 4 h requested. CASES=all (needs more downloaded cases): about 4-6 h,
# submit with --time=12:00:00.
#SBATCH --job-name=evade-seizure-deep
#SBATCH --partition=partnsf
#SBATCH --qos=qosnsf
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --constraint=VOLTA|AMPERE
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --signal=B:USR1@300
#SBATCH --output=/scratch/%u/evade/slurm/%x-%j.out
#SBATCH --error=/scratch/%u/evade/slurm/%x-%j.err

set -uo pipefail
PROJECT=/scratch/$USER/evade
PY=${EVADE_PY:-$PROJECT/venv/bin/python}
# the uv venv (self-contained Python under $PROJECT/python); no module load needed
export PYTHONPATH=$PROJECT/code
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
[ -s "$PROJECT/code/COMMIT" ] || { echo "missing $PROJECT/code/COMMIT: upload the code with its commit id"; exit 4; }
export EVADE_CODE_COMMIT=$(cat "$PROJECT/code/COMMIT")
TAG=$(echo "$EVADE_CODE_COMMIT" | cut -c1-10)
CASES=${CASES:-beta5}
IV=1; [ "$CASES" = all ] && IV=2
RUN=${RUN_DIR:-$PROJECT/runs/seizure-deep-$TAG}
DATA=$PROJECT/data/seizure256
SM=()
if [ "${SMOKE:-0}" = "1" ]; then RUN=$RUN-smoke; DATA=$DATA-smoke; rm -rf "$RUN"; SM=(--smoke); fi
mkdir -p "$RUN"; cd "$RUN"

{
  echo "job ${SLURM_JOB_ID:-?} on $(hostname) at $(date '+%F %T'), partition ${SLURM_JOB_PARTITION:-?}, smoke ${SMOKE:-0}, cases $CASES"
  echo "code $EVADE_CODE_COMMIT"
  nvidia-smi --query-gpu=name,memory.total,driver_version,compute_cap --format=csv,noheader
} >> job_info.txt 2>&1
GPU=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
case "$GPU" in *K20*|*K40*|*K80*|"") echo "unsupported or missing GPU '$GPU'" >> job_info.txt; exit 3 ;; esac
[ -s "$DATA/pack_summary.json" ] || { echo "missing packed data in $DATA; run seizure_prep.sbatch" >> job_info.txt; exit 4; }

# SIGUSR1 from SLURM: remember it, forward it to a running python step (which checkpoints and exits 99), and do not
# start another step afterwards
child=0; stop=0
trap 'stop=1; [ $child -ne 0 ] && kill -USR1 $child 2>/dev/null' USR1
step() {   # step <log> <args...>
  local log=$1; shift
  [ $stop -eq 1 ] && { echo "SIGUSR1 before: $*; exiting 99" >> job_info.txt; return 99; }
  "$PY" "$@" >> "$log" 2>&1 &
  child=$!
  local rc=0
  wait $child; rc=$?
  while [ $rc -gt 128 ]; do wait $child; rc=$?; [ $rc -eq 127 ] && break; done   # wait again after the trap
  child=0
  echo "step rc=$rc: $*" >> job_info.txt
  [ $rc -eq 0 ] && [ $stop -eq 1 ] && return 99      # signal landed in a phase that does not check it
  return $rc
}

T=(-m evade.training.deep.seizure_train --data "$DATA" --arch seizure_resnet1d --epochs 20 --patience 5 \
   --sampler balanced --loss bce --n-boot 2000 --resume ${SM[@]+"${SM[@]}"})
step run.log "${T[@]}" --protocol loso --cases "$CASES" --inner-val $IV --run "$RUN/loso" || exit $?
step run.log "${T[@]}" --protocol final --cases "$CASES" --inner-val $IV --external siena \
     --thresholds-from "$RUN/loso/metrics.json" --run "$RUN/final" || exit $?
step run.log -m evade.training.deep.seizure_export --run "$RUN/final" --out "$RUN/export/seizure_deep" \
     --require-parity || echo "export or parity failed (training results are kept); see run.log" >> job_info.txt
grep -h "^DONE" run.log >> job_info.txt
echo "done $(date '+%F %T')" >> job_info.txt
