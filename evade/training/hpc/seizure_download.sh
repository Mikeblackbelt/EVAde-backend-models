#!/bin/bash
# ##################################################################################################################
# #                                                                                                                #
# #   STOP: EMAIL HPC HELP (hpchelp, the CUNY HPCC support address) BEFORE RUNNING THIS SCRIPT.                    #
# #                                                                                                                #
# #   It downloads about 63 GB from PhysioNet into /scratch (CHB-MIT 42.6 GB, 686 EDF files in 24 case folders;    #
# #   Siena Scalp EEG 20.3 GB, 41 EDF files in 14 subject folders). A transfer of tens of GB through the login    #
# #   node must be announced: say what (two open PhysioNet EEG datasets), how much (about 63 GB), where            #
# #   (/scratch/<user>/evade/data) and when, and wait for their answer. Do not run it before they agree.          #
# #                                                                                                                #
# ##################################################################################################################
#
# Polite and resumable: one file at a time (no parallel connections), a pause between requests, a bandwidth cap,
# wget -c resumes partial files, and re-running skips files that already match their checksum. Files get the
# download time, not the server's old timestamp (--no-use-server-timestamps), because /scratch purges by age.
# At the end every file is checked against the dataset's own SHA256SUMS.txt; a VERIFIED marker is written only if
# every wanted file is present and correct, and corrupt files are deleted so the next run fetches them again.
# Data only: nothing downloaded is executed.
#
#   bash seizure_download.sh                  both datasets
#   bash seizure_download.sh chbmit           one dataset (chbmit or siena)
#   RATE=20m WAIT=1 bash seizure_download.sh  bandwidth cap per second and seconds between files
# Run it on the login node under nohup (it is network-bound, not CPU-bound), for example:
#   nohup bash $PROJECT/code/evade/training/hpc/seizure_download.sh > $PROJECT/data/seizure_download.log 2>&1 &
# Expect 1.5-4 h (PhysioNet often serves 5-15 MB/s; the default cap is 30 MB/s). Re-run the same command after
# any interruption.
set -uo pipefail
PROJECT=${PROJECT:-/scratch/$USER/evade}
DATA=$PROJECT/data
RATE=${RATE:-30m}
WAIT=${WAIT:-0.5}
BASE=https://physionet.org/files
WHICH=${1:-all}

need_gb=70
avail_gb=$(df -BG --output=avail "$PROJECT" 2>/dev/null | tail -1 | tr -dc 0-9)
if [ -n "$avail_gb" ] && [ "$avail_gb" -lt "$need_gb" ]; then
  echo "only ${avail_gb} GB free under $PROJECT; need about ${need_gb} GB"; exit 75
fi

fetch() {   # fetch <dataset slug> <version> <local dir> <awk regex of the relative paths to keep>
  local slug=$1 ver=$2 dest=$3 keep=$4
  local url="$BASE/$slug/$ver"
  mkdir -p "$dest"; cd "$dest" || return 1
  echo "$(date '+%F %T') $slug: SHA256SUMS.txt"
  wget -q --no-use-server-timestamps -O SHA256SUMS.txt.part "$url/SHA256SUMS.txt" && mv SHA256SUMS.txt.part SHA256SUMS.txt \
    || { echo "cannot fetch $url/SHA256SUMS.txt"; return 1; }
  # normalized "hash  path" lines for the wanted files only
  awk '{print $1"  "$2}' SHA256SUMS.txt | awk -v re="$keep" '$2 ~ re' > SHA256SUMS.wanted
  echo "$(date '+%F %T') $slug: $(wc -l < SHA256SUMS.wanted) files listed"
  local n=0 fail=0 h f
  while read -r h f; do
    n=$((n + 1))
    # a finished file that already matches its checksum is skipped without contacting the server
    if [ -s "$f" ] && echo "$h  $f" | sha256sum -c --status - 2>/dev/null; then continue; fi
    mkdir -p "$(dirname "$f")"
    if ! wget -q -c --no-use-server-timestamps --limit-rate="$RATE" --tries=5 --waitretry=30 -O "$f" "$url/$f"; then
      echo "FAILED $f"; fail=$((fail + 1))
    fi
    sleep "$WAIT"
    [ $((n % 25)) -eq 0 ] && echo "$(date '+%F %T') $slug: $n files done"
  done < SHA256SUMS.wanted
  echo "$(date '+%F %T') $slug: verifying"
  local bad
  bad=$(sha256sum -c SHA256SUMS.wanted 2>/dev/null | grep -v ': OK$' | sed 's/: .*$//')
  if [ "$fail" -eq 0 ] && [ -z "$bad" ]; then
    echo "$(date '+%F %T') $(wc -l < SHA256SUMS.wanted) files" > VERIFIED
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
  # all 24 cases: every EDF, the per-case summary files, the .seizures annotation files and the top-level lists
  fetch chbmit 1.0.0 "$DATA/chbmit" '^(chb[0-9][0-9]/.*[.](edf|edf[.]seizures|txt)|RECORDS.*|SUBJECT-INFO|ANNOTATORS)$' || rc=1
fi
if [ "$WHICH" = all ] || [ "$WHICH" = siena ]; then
  # all 14 subjects: EDFs, seizure lists, subject info and records
  fetch siena-scalp-eeg 1.0.0 "$DATA/siena-scalp-eeg" '^(PN[0-9][0-9]/.*[.](edf|txt)|RECORDS|subject_info[.]csv)$' || rc=1
fi
exit $rc
