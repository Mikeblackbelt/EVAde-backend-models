#!/bin/bash
# CPU job (no GPU): pack CHB-MIT (24 cases) and Siena (14 subjects) into per-subject float16 window shards for the
# deep seizure-suspect model: 18-channel bipolar montage, 0.5-45 Hz band-pass, mains notch, 256 Hz, 4 s windows,
# 2 s step, beta labels and buffers, beta file selection (every seizure file + 3 seizure-free files per subject),
# per-person median / MAD from that person's non-ictal windows. Output: data/seizure256/{chbmit,siena}/*.npz.
# Resumable: finished shards are skipped on resubmit.
# Needs: data/chbmit/VERIFIED and data/siena-scalp-eeg/VERIFIED (seizure_download.sh), pyedflib in the venv.
#   smoke: sbatch --export=ALL,SMOKE=1 --time=00:30:00 seizure_prep.sbatch     (3 files per case, writes -smoke)
#   full:  sbatch seizure_prep.sbatch
# Estimate: 1.5-3 h on 8 cores; Siena files (up to 1.7 GB, ~13 h at 512 Hz) are packed one subject at a time.
# Output size: about 13 GB CHB-MIT + 10 GB Siena (float16, uncompressed npz).
#SBATCH --job-name=evade-seizure-prep
#SBATCH --partition=partnsf
#SBATCH --qos=qosnsf
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=/scratch/%u/evade/slurm/%x-%j.out
#SBATCH --error=/scratch/%u/evade/slurm/%x-%j.err

set -euo pipefail
PROJECT=/scratch/$USER/evade
PY=${EVADE_PY:-$PROJECT/venv/bin/python}
# the uv venv (self-contained Python under $PROJECT/python); no module load needed
export PYTHONPATH=$PROJECT/code
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
RAW=$PROJECT/data
OUT=$PROJECT/data/seizure256
SM=(); WORKERS=${SLURM_CPUS_PER_TASK:-8}
if [ "${SMOKE:-0}" = "1" ]; then OUT=$OUT-smoke; rm -rf "$OUT"; SM=(--smoke); fi
mkdir -p "$OUT"
echo "prep job ${SLURM_JOB_ID:-?} on $(hostname) at $(date '+%F %T'), smoke ${SMOKE:-0}"
for d in chbmit siena-scalp-eeg; do
  [ -e "$RAW/$d/VERIFIED" ] || { echo "missing $RAW/$d/VERIFIED: run seizure_download.sh first"; exit 4; }
done
"$PY" -c 'import pyedflib' 2>/dev/null || {
  echo "pyedflib missing in the venv; install it once on the login node:"
  echo "  $PROJECT/tools/bin/uv pip install --python $PROJECT/venv/bin/python pyedflib"; exit 4; }

# CHB-MIT cases are small (1-4 h files at 256 Hz): $WORKERS in parallel (about 2-6 GB each).
# Siena: one subject at a time (a 1.7 GB EDF needs about 7 GB as float64 plus the window copies).
"$PY" -m evade.training.deep.seizure_pack --chb "$RAW/chbmit" --siena "$RAW/siena-scalp-eeg" --out "$OUT" \
     --fs 256 --max-free 3 --workers "$WORKERS" --siena-workers 1 ${SM[@]+"${SM[@]}"} > "$OUT/pack.log" 2>&1
tail -n 3 "$OUT/pack.log"
grep -E "SKIPPED|WARNING" "$OUT/pack.log" || true
(cd "$OUT" && sha256sum chbmit/*.npz siena/*.npz > SHA256SUMS)
echo "done $(date '+%F %T')"
