#!/bin/bash
# CPU job: pack the Sleep-EDFx sleep-cassette EDF files (downloaded by sleep_download.sh) into
# data/sleep-sc/sleep_sc.npz for the deep sleep model. No GPU.
#   sbatch sleep_prep.sbatch                 all nights (needs the VERIFIED marker written by sleep_download.sh)
#   sbatch --export=ALL,SMOKE=1 sleep_prep.sbatch   first 10 nights only, into sleep_sc_smoke.npz
#SBATCH --job-name=evade-sleep-prep
#SBATCH --partition=partnsf
#SBATCH --qos=qosnsf
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=/scratch/%u/evade/slurm/%x-%j.out
#SBATCH --error=/scratch/%u/evade/slurm/%x-%j.err

set -euo pipefail
PROJECT=/scratch/$USER/evade
PY=${EVADE_PY:-$PROJECT/venv/bin/python}
# the venv uses a self-contained Python installed with uv under $PROJECT/python (compute nodes lack the login
# node's /usr/bin/python3.12, and the cluster Python modules miss _ctypes/_bz2); no module load needed
export PYTHONPATH=$PROJECT/code:$PROJECT/pylib
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
RAW=$PROJECT/data/raw/sleep-edfx
OUT=$PROJECT/data/sleep-sc
mkdir -p "$OUT"
echo "prep job ${SLURM_JOB_ID:-?} on $(hostname) at $(date '+%F %T')"
LIMIT=0; NAME=sleep_sc
if [ "${SMOKE:-0}" = "1" ]; then LIMIT=10; NAME=sleep_sc_smoke
else [ -f "$RAW/VERIFIED" ] || { echo "no $RAW/VERIFIED: run or finish sleep_download.sh first"; exit 4; }
fi
"$PY" -m evade.training.deep.sleep_pack --src "$RAW" --out "$OUT/$NAME.npz" --limit "$LIMIT" > "$OUT/$NAME.pack.log" 2>&1
tail -n 3 "$OUT/$NAME.pack.log"
grep -c SKIPPED "$OUT/$NAME.pack.log" || true
(cd "$OUT" && sha256sum "$NAME.npz" > "$NAME.SHA256SUMS")
echo "done $(date '+%F %T')"
