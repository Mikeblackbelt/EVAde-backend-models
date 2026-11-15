#!/bin/bash
# Download the PoC seizure data from PhysioNet into $PROJECT/data (open datasets, no credentials).
#
# Default scope (about 11 GB, so no advance notice to HPC staff is needed):
#   CHB-MIT 1.0.0     chb01-chb05 (the beta's subjects), the beta file rule: every EDF with a seizure plus the
#                     first 3 seizure-free EDFs per case (sorted), as evade.training.data.chb_jobs and the pack use.
#                     About 2-3 GB (estimated from the PhysioNet listing: chb01 files ~40 MB, chb04 ~160 MB).
#   Siena Scalp EEG   the beta's six external-test subjects PN00, PN01, PN03, PN05, PN06, PN07, same file rule
#   1.0.0             (in practice every file: they all hold seizures). About 8.7 GB by the PhysioNet listing.
# More data is an option (environment variables below). Everything (CHB_CASES=all SIENA_SUBJECTS=all MAX_FREE=-1)
# is about 63 GB: a transfer of tens of GB must be announced to HPC staff (hpchelp) first, and wait for their answer.
#
# Polite and resumable: one file at a time (no parallel connections), a pause between requests, a bandwidth cap,
# wget -c resumes partial files, and re-running skips files that already match their checksum. Files get the
# download time, not the server's old timestamp (--no-use-server-timestamps), because /scratch purges by age.
# Per dataset: SHA256SUMS.txt, then the summaries / seizure lists, then the EDFs the file rule picks
# (evade.training.deep.seizure_select, the pack's own summary parser). Every wanted file is checked against the
# dataset's SHA256SUMS.txt; a VERIFIED marker is written only if all are present and correct, and corrupt files are
# deleted so the next run fetches them again. Data only: nothing downloaded is executed.
#
#   bash seizure_download.sh                  both datasets, default scope
#   bash seizure_download.sh chbmit           one dataset (chbmit or siena)
#   CHB_CASES="chb01 chb06" SIENA_SUBJECTS=all MAX_FREE=-1 bash seizure_download.sh     more data (see above)
#   RATE=20m WAIT=1 bash seizure_download.sh  bandwidth cap per second and seconds between files
# Run it on the login node under nohup (it is network-bound, not CPU-bound), for example:
#   nohup bash $PROJECT/code/evade/training/hpc/seizure_download.sh > $PROJECT/data/seizure_download.log 2>&1 &
# Expect 15-40 min for the default scope (PhysioNet often serves 5-15 MB/s). Re-run the same command after any
# interruption.
set -uo pipefail
PROJECT=${PROJECT:-/scratch/$USER/evade}
DATA=$PROJECT/data
PY=${EVADE_PY:-$PROJECT/venv/bin/python}
RATE=${RATE:-30m}
WAIT=${WAIT:-0.5}
BASE=https://physionet.org/files
WHICH=${1:-all}
CHB_CASES=${CHB_CASES:-"chb01 chb02 chb03 chb04 chb05"}
SIENA_SUBJECTS=${SIENA_SUBJECTS:-"PN00 PN01 PN03 PN05 PN06 PN07"}
MAX_FREE=${MAX_FREE:-3}
mkdir -p "$DATA"

# free space: default scope ~11 GB, everything ~63 GB; what is already downloaded counts as done
need_gb=15
if [ "$CHB_CASES" = all ] || [ "$SIENA_SUBJECTS" = all ] || [ "$MAX_FREE" -lt 0 ]; then need_gb=70; fi
have_gb=$( (du -s --block-size=1G "$DATA/chbmit" "$DATA/siena-scalp-eeg" 2>/dev/null || true) | awk '{s += $1} END {print s + 0}')
need_gb=$((need_gb - have_gb)); [ "$need_gb" -lt 1 ] && need_gb=1
avail_gb=$(df -BG --output=avail "$DATA" 2>/dev/null | tail -1 | tr -dc 0-9)
if [ -n "$avail_gb" ] && [ "$avail_gb" -lt "$need_gb" ]; then
  echo "only ${avail_gb} GB free under $DATA; need about ${need_gb} GB more"; exit 75
fi

alt() {   # "a b c" -> "(a|b|c)"; "all" -> the given pattern
  if [ "$1" = all ]; then echo "$2"; else echo "($(echo "$1" | tr -s ' ' '|'))"; fi
}

get_list() {   # get_list <url> <list file>: download the "hash  path" files of the list (skip ones that match)
  local url=$1 list=$2 n=0 fail=0 h f
  while read -r h f; do
    n=$((n + 1))
    if [ -s "$f" ] && echo "$h  $f" | sha256sum -c --status - 2>/dev/null; then continue; fi
    mkdir -p "$(dirname "$f")"
    if ! wget -q -c --no-use-server-timestamps --limit-rate="$RATE" --tries=5 --waitretry=30 -O "$f" "$url/$f"; then
      echo "FAILED $f"; fail=$((fail + 1))
    fi
    sleep "$WAIT"
    [ $((n % 25)) -eq 0 ] && echo "$(date '+%F %T') $n files done"
  done < "$list"
  FAILS=$fail
  [ "$fail" -eq 0 ]
}

fetch() {   # fetch <dataset slug> <version> <local dir> <awk regex of the relative paths to keep> <chbmit|siena>
  local slug=$1 ver=$2 dest=$3 keep=$4 ds=$5
  local url="$BASE/$slug/$ver"
  mkdir -p "$dest"; cd "$dest" || return 1
  echo "$(date '+%F %T') $slug: SHA256SUMS.txt"
  wget -q --no-use-server-timestamps -O SHA256SUMS.txt.part "$url/SHA256SUMS.txt" && mv SHA256SUMS.txt.part SHA256SUMS.txt \
    || { echo "cannot fetch $url/SHA256SUMS.txt"; return 1; }
  # normalized "hash  path" lines of the chosen subjects, sorted by path (C locale, as Python sorts names)
  awk '{print $1"  "$2}' SHA256SUMS.txt | awk -v re="$keep" '$2 ~ re' | LC_ALL=C sort -k2 > SHA256SUMS.listed
  # 1. summaries and seizure lists (the file rule needs them)
  grep -v '[.]edf$' SHA256SUMS.listed > SHA256SUMS.meta
  get_list "$url" SHA256SUMS.meta || { echo "$slug: summaries incomplete; re-run"; return 1; }
  # 2. the EDFs the beta file rule keeps (run from the code dir, never from the data dir)
  if [ "$MAX_FREE" -lt 0 ]; then
    cp SHA256SUMS.listed SHA256SUMS.wanted
  else
    (cd "$PROJECT/code" && PYTHONPATH="$PROJECT/code" "$PY" -m evade.training.deep.seizure_select --dataset "$ds" \
       --root "$dest" --listing "$dest/SHA256SUMS.listed" --max-free "$MAX_FREE") > SHA256SUMS.wanted.part \
      && mv SHA256SUMS.wanted.part SHA256SUMS.wanted || { echo "$slug: file selection failed"; return 1; }
  fi
  echo "$(date '+%F %T') $slug: $(grep -c '[.]edf$' SHA256SUMS.wanted) EDF files wanted"
  local fail=0
  get_list "$url" SHA256SUMS.wanted || fail=$FAILS
  echo "$(date '+%F %T') $slug: verifying"
  local bad
  bad=$(sha256sum -c SHA256SUMS.wanted 2>/dev/null | grep -v ': OK$' | sed 's/: .*$//')
  if [ "$fail" -eq 0 ] && [ -z "$bad" ]; then
    echo "$(date '+%F %T') $(wc -l < SHA256SUMS.wanted) files (CHB_CASES=$CHB_CASES SIENA_SUBJECTS=$SIENA_SUBJECTS MAX_FREE=$MAX_FREE)" > VERIFIED
    echo "$slug: VERIFIED"
  else
    rm -f VERIFIED
    # corrupt files are removed so the next run downloads them again (wget -c cannot repair a full-size file)
    for f in $bad; do [ -f "$f" ] && rm -f "$f"; done
    echo "$slug: NOT verified ($fail failed downloads, $(echo "$bad" | grep -c .) bad or missing files); re-run"
    return 1
  fi
}

rc=0
if [ "$WHICH" = all ] || [ "$WHICH" = chbmit ]; then
  c=$(alt "$CHB_CASES" 'chb[0-9][0-9]')
  # EDFs, per-case summaries, .seizures annotation files, and the top-level lists
  fetch chbmit 1.0.0 "$DATA/chbmit" "^(${c}/.*[.](edf|edf[.]seizures|txt)|RECORDS.*|SUBJECT-INFO|ANNOTATORS)\$" chbmit || rc=1
fi
if [ "$WHICH" = all ] || [ "$WHICH" = siena ]; then
  s=$(alt "$SIENA_SUBJECTS" 'PN[0-9][0-9]')
  fetch siena-scalp-eeg 1.0.0 "$DATA/siena-scalp-eeg" "^(${s}/.*[.](edf|txt)|RECORDS|subject_info[.]csv)\$" siena || rc=1
fi
exit $rc
