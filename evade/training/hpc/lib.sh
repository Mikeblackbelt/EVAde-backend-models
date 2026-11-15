#!/bin/bash
# Shared helpers for the EVAde SLURM job scripts. Sourced by every *.sbatch in this folder:
#   source "$PROJECT/code/evade/training/hpc/lib.sh"
# Layout on arrow (see README.md):  /scratch/$USER/evade/{venv,code,data,runs,slurm}
# Exit codes used by the jobs:  3 unsupported or missing GPU, 4 code or data not ready, 99 checkpointed near wall time.

PROJECT=${PROJECT:-/scratch/$USER/evade}
CODE=$PROJECT/code
DATA=$PROJECT/data
PY=$PROJECT/venv/bin/python

# If setup_env.sh built the venv from a python module, load that module here too (batch shells are not login shells).
_pymod=$(cat "$PROJECT/venv/PYTHON_MODULE" 2>/dev/null || true)
if [ -n "$_pymod" ]; then
  set +u
  type module >/dev/null 2>&1 || { [ -f /etc/profile.d/lmod.sh ] && . /etc/profile.d/lmod.sh; } || true
  module load "$_pymod" || echo "warning: could not load module $_pymod" >&2
  set -u
fi

# Stable run folder: session name + short code commit, so a resubmitted job resumes in the same place
# (the template's per-job-id folder would start from scratch every time).
run_dir() {   # run_dir <session-name>  -> prints the folder path
  local commit
  commit=$(head -c 12 "$CODE/COMMIT" 2>/dev/null || echo nocommit)
  if [ "${SMOKE:-0}" = "1" ]; then
    echo "$PROJECT/runs/$1-$commit-smoke"
  else
    echo "$PROJECT/runs/$1-$commit"
  fi
}

job_header() {   # writes job_info.txt in the current folder
  {
    echo "job ${SLURM_JOB_ID:-?} ($SLURM_JOB_NAME) on $(hostname) at $(date '+%F %T'), partition ${SLURM_JOB_PARTITION:-?}"
    echo "cpus ${SLURM_CPUS_PER_TASK:-?}, gpus ${CUDA_VISIBLE_DEVICES:-none}, smoke ${SMOKE:-0}"
    nvidia-smi --query-gpu=name,memory.total,driver_version,compute_cap --format=csv,noheader 2>/dev/null || true
    echo "code commit: $(cat "$CODE/COMMIT" 2>/dev/null || echo MISSING)"
  } >> job_info.txt 2>&1
}

gpu_guard() {   # fail fast on GPUs the CUDA 12.x torch build cannot use (K20m is sm_35)
  local name cc
  name=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
  case "$name" in
    *K20*|*K40*|*K80*|"") echo "unsupported or missing GPU: '$name'; pin the GPU type with --gres/--constraint" >> job_info.txt; exit 3 ;;
  esac
  cc=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | tr -d ' ')
  if [ -n "$cc" ] && [ "${cc%%.*}" -lt 7 ] 2>/dev/null; then
    echo "GPU '$name' has compute capability $cc (< 7.0); torch cu12x needs sm_70+" >> job_info.txt; exit 3
  fi
  "$PY" -c "import torch,sys; ok=torch.cuda.is_available(); print('torch', torch.__version__, 'cuda', torch.version.cuda, 'available', ok); sys.exit(0 if ok else 1)" \
    >> job_info.txt 2>&1 || { echo "torch cannot use the GPU" >> job_info.txt; exit 3; }
}

require_code() {   # require_code <python.module> ...   (exit 4 if the planned module is not written yet)
  [ -x "$PY" ] || { echo "no venv at $PY: run setup_env.sh first" >> job_info.txt; exit 4; }
  [ -s "$CODE/COMMIT" ] || { echo "no $CODE/COMMIT: upload committed code (README step 3)" >> job_info.txt; exit 4; }
  local m
  for m in "$@"; do
    (cd "$CODE" && "$PY" -c "import importlib.util,sys; sys.exit(0 if importlib.util.find_spec('$m') else 1)") \
      || { echo "module $m not found in $CODE (planned in training-plan-v2.md section 7)" >> job_info.txt; exit 4; }
  done
}

require_data() {   # require_data <dataset-folder-under-data> ...   (each needs a VERIFIED marker, README step 5)
  local d
  for d in "$@"; do
    [ -e "$DATA/$d/VERIFIED" ] || { echo "data $DATA/$d not verified (missing VERIFIED marker)" >> job_info.txt; exit 4; }
  done
}

# Run python in the background so SIGUSR1 (sent by SLURM via --signal=B:USR1@600) can be forwarded;
# the training code saves a checkpoint and exits 99, and resubmitting the same script resumes.
# Output always goes to a log file (undefined stdout/stderr is a forbidden practice on HPCC).
run_py() {   # run_py <logfile> <python args...>
  local log=$1; shift
  ( cd "$CODE" && exec "$PY" "$@" ) >> "$log" 2>&1 &
  local pid=$! rc=0
  trap 'echo "$(date "+%F %T") USR1: wall time near, forwarding to $pid" >> job_info.txt; kill -USR1 '"$pid"' 2>/dev/null' USR1
  while :; do
    wait "$pid" && rc=0 || rc=$?
    kill -0 "$pid" 2>/dev/null || break      # wait was interrupted by the trap: keep waiting for the child
  done
  trap - USR1
  echo "$(date '+%F %T') exit $rc: $*" >> job_info.txt
  return $rc
}

step() {   # step <logfile> <python args...>   run_py, and stop the job on failure (99 = checkpointed, resubmit to resume)
  local rc=0
  run_py "$@" || rc=$?
  if [ "$rc" -ne 0 ]; then
    [ "$rc" -eq 99 ] && echo "checkpointed near wall time; resubmit the same script to resume" >> job_info.txt
    exit "$rc"
  fi
}

finish_export() {   # finish_export <run-folder>   checksums of the servable folder
  local run=$1
  if [ -d "$run/export" ]; then
    (cd "$run/export" && find . -type f ! -name SHA256SUMS -print0 | sort -z | xargs -0 sha256sum > SHA256SUMS)
    echo "export checksums written: $run/export/SHA256SUMS" >> job_info.txt
  fi
  echo "done $(date '+%F %T'); copy $run/export off scratch this week (scratch is purged, not backed up)" >> job_info.txt
}
