#!/bin/bash
# CPU job: unpack PTB-XL (100 Hz records only) and pack it into data/ptbxl100/ptbxl100.npz. No GPU.
#   sbatch ecg_prep.sbatch            (SMOKE=1 packs the first 600 records only)
#SBATCH --job-name=evade-ecg-prep
#SBATCH --partition=partnsf
#SBATCH --qos=qosnsf
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:45:00
#SBATCH --output=/scratch/%u/evade/slurm/%x-%j.out
#SBATCH --error=/scratch/%u/evade/slurm/%x-%j.err

set -euo pipefail
PROJECT=/scratch/$USER/evade
PY=${EVADE_PY:-$PROJECT/venv/bin/python}
# the venv uses a self-contained Python installed with uv under $PROJECT/python (compute nodes lack the login
# node's /usr/bin/python3.12, and the cluster Python modules miss _ctypes/_bz2); no module load needed
export PYTHONPATH=$PROJECT/code:$PROJECT/pylib
RAW=$PROJECT/data/raw
OUT=$PROJECT/data/ptbxl100
mkdir -p "$OUT"
cd "$RAW"
echo "prep job ${SLURM_JOB_ID:-?} on $(hostname) at $(date '+%F %T')"
# -DD: give extracted files the current time (scratch purges by age); only what training needs
unzip -o -q -DD ptbxl.zip '*/records100/*' '*/ptbxl_database.csv' '*/scp_statements.csv' -d ptbxl_x > unzip.log 2>&1
SRC=$(dirname "$(find ptbxl_x -name ptbxl_database.csv | head -1)")
LIMIT=0; [ "${SMOKE:-0}" = "1" ] && LIMIT=600
"$PY" -m evade.training.deep.ptbxl_pack --src "$SRC" --out "$OUT/ptbxl100.npz" --limit "$LIMIT" > "$OUT/pack.log" 2>&1
cat "$OUT/pack.log"
rm -rf ptbxl_x
sha256sum "$OUT/ptbxl100.npz" > "$OUT/SHA256SUMS"
echo "done $(date '+%F %T')"
