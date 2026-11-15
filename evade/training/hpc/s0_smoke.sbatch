#!/bin/bash
# Session 0c: smoke test on a compute node. Unit tests, then a tiny synthetic train -> evaluate -> export for each task,
# one after another in this single job. No real data needed. Read run.log end to end before submitting S1.
# Before the first submit: replace CHANGE_ME_GPU_TYPE (and the account/constraint lines if needed) from discover.sh,
# then check with:  sbatch --test-only s0_smoke.sbatch      (sbatch is expected to reject the placeholder; the real gate is: grep -n CHANGE_ME *.sbatch)
# Log folder must exist first (SLURM does not create it):  mkdir -p /scratch/$USER/evade/slurm
#SBATCH --job-name=evade-s0-smoke
#SBATCH --partition=partnsf
#SBATCH --qos=qosnsf
##SBATCH --account=CHANGE_ME_ACCOUNT
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:CHANGE_ME_GPU_TYPE:1
##SBATCH --constraint=CHANGE_ME_FEATURE
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --signal=B:USR1@120
#SBATCH --output=/scratch/%u/evade/slurm/%x-%j.out
#SBATCH --error=/scratch/%u/evade/slurm/%x-%j.err
# --account: only if discover.sh shows the partition needs one.  --gres type: the V100 or A100 name from discover.sh.
# --constraint: use instead of a typed gres if the cluster pins GPU types by feature.

set -euo pipefail
PROJECT=/scratch/$USER/evade
source "$PROJECT/code/evade/training/hpc/lib.sh"
export SMOKE=1
RUN=$(run_dir s0)
rm -rf "$RUN"; mkdir -p "$RUN"; cd "$RUN"
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}

job_header
gpu_guard
require_code evade.training.deep.train evade.training.deep.evaluate evade.training.deep.export

# 1. unit tests on the compute node, not the login node
step "$RUN/pytest.log" -m pytest -q tests

# 2. synthetic end-to-end per task: tiny data, 1 epoch, 2 folds; then export with ONNX parity and latency checks
for TASK in ecg sleep seizure; do
  step "$RUN/run.log" -m evade.training.deep.train --task "$TASK" --synthetic --smoke \
       --run "$RUN/$TASK" --resume --seeds 0
  # resume check: a second call must find every fold DONE and return at once
  step "$RUN/run.log" -m evade.training.deep.train --task "$TASK" --synthetic --smoke \
       --run "$RUN/$TASK" --resume --seeds 0 --expect-done
  step "$RUN/run.log" -m evade.training.deep.evaluate --task "$TASK" --run "$RUN/$TASK"
  step "$RUN/run.log" -m evade.training.deep.export --task "$TASK" --run "$RUN/$TASK" \
       --out "$RUN/export/$TASK" --parity-tol 1e-4 --latency
done

finish_export "$RUN"
