#!/bin/bash
# CPU job (no GPU): pack the downloaded CHB-MIT cases and Siena subjects (default scope of seizure_download.sh:
# chb01-05 and the beta's Siena-6) into per-subject float16 window shards for the deep seizure-suspect model:
# 18-channel bipolar montage, 0.5-45 Hz band-pass, mains notch, 256 Hz, 4 s windows, 2 s step, beta labels and
# buffers, beta file rule (every seizure file + 3 seizure-free files per subject), and per-window robust
# normalization (the API's per_window_robust). Output: data/seizure256/{chbmit,siena}/*.npz.
# Resumable: finished shards are skipped on resubmit and their log lines replayed; shards from an older
# normalization are rebuilt automatically (delete data/seizure256/<dataset> after a parser or --min-channels change).
# Needs: data/chbmit/VERIFIED and data/siena-scalp-eeg/VERIFIED (seizure_download.sh), pyedflib in the venv.
#   smoke: sbatch --export=ALL,SMOKE=1 --time=00:30:00 seizure_prep.sbatch     (3 files per case, 2 per Siena subject; -smoke)
#   full:  sbatch seizure_prep.sbatch
# Estimate for the default scope: about 20-45 min on 8 cores (40 CHB-MIT files, ~2-3 GB; Siena-6 ~8.7 GB, up to
# 1.6 GB per file, one subject at a time). Output about 3.6 GB CHB-MIT (~98k windows) + 4-6 GB Siena-6 (~80 h of EEG) as float16, uncompressed npz.
#SBATCH --job-name=evade-seizure-prep
#SBATCH --partition=partnsf
#SBATCH --qos=qosnsf
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=03:00:00
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
  cat "$RAW/$d/VERIFIED"
done
"$PY" -c 'import pyedflib' 2>/dev/null || {
  echo "pyedflib missing in the venv; install it once on the login node:"
  echo "  $PROJECT/tools/bin/uv pip install --python $PROJECT/venv/bin/python pyedflib"; exit 4; }

# CHB-MIT cases are small (1-4 h files at 256 Hz): $WORKERS in parallel (about 2-6 GB each).
# Siena: one subject at a time (a 1.6 GB EDF needs about 7 GB as float64 plus the window copies).
echo "=== prep job ${SLURM_JOB_ID:-?} $(date '+%F %T')" >> "$OUT/pack.log"
"$PY" -m evade.training.deep.seizure_pack --chb "$RAW/chbmit" --siena "$RAW/siena-scalp-eeg" --out "$OUT" \
     --fs 256 --max-free 3 --workers "$WORKERS" --siena-workers 1 ${SM[@]+"${SM[@]}"} >> "$OUT/pack.log" 2>&1
tail -n 3 "$OUT/pack.log"
# gate: any SKIPPED / WARNING / EXCLUDED line from this run (finished shards replay theirs) needs a look
sed -n "/^=== prep job ${SLURM_JOB_ID:-?} /,\$p" "$OUT/pack.log" | grep -E "SKIPPED|WARNING|EXCLUDED" || true
(cd "$OUT" && sha256sum chbmit/*.npz siena/*.npz > SHA256SUMS)
echo "done $(date '+%F %T')"
