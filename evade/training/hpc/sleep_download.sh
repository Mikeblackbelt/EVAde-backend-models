#!/bin/bash
# Download the Sleep-EDFx sleep-cassette (SC) PSG and hypnogram EDF files from PhysioNet into
# $PROJECT/data/raw/sleep-edfx/sleep-cassette, then check them against PhysioNet's SHA256SUMS.txt.
# Runs on the LOGIN node (network only, no compute). Polite and resumable:
#   - one file at a time, a pause between files, a bandwidth cap (RATE, default 5m = 5 MB/s);
#   - wget -c continues partial files, and files whose sha256 already matches are skipped;
#   - --no-use-server-timestamps gives files the arrival time (scratch is purged by file age).
# Usage (in tmux or with nohup; about 8.1 GB, roughly 30-40 min at 5 MB/s):
#   bash sleep_download.sh                 all 153 nights (306 files)
#   LIMIT=20 bash sleep_download.sh        the first 20 files = the 10 beta nights SC400-SC404 (smoke; no VERIFIED)
#   RATE=10m WAIT=1 bash sleep_download.sh
# Rerun the same command after an interruption; it resumes. Exit 0 = every listed file verified.
set -uo pipefail
PROJECT=${PROJECT:-/scratch/$USER/evade}
BASE=https://physionet.org/files/sleep-edfx/1.0.0
OUT=$PROJECT/data/raw/sleep-edfx
RATE=${RATE:-5m}
WAIT=${WAIT:-2}
LIMIT=${LIMIT:-0}
EXPECTED=306                  # 153 SC nights x (PSG + hypnogram) in Sleep-EDFx 1.0.0
mkdir -p "$OUT/sleep-cassette"
cd "$OUT"
UA="evade-research-download (wget)"
WG=(wget -q --no-use-server-timestamps --limit-rate="$RATE" --tries=5 --waitretry=30 --timeout=60 -U "$UA")

rm -f SHA256SUMS.txt.part
"${WG[@]}" -O SHA256SUMS.txt.part "$BASE/SHA256SUMS.txt" && mv SHA256SUMS.txt.part SHA256SUMS.txt
[ -s SHA256SUMS.txt ] || { echo "could not fetch SHA256SUMS.txt" >&2; exit 2; }
# only SC PSG and hypnogram EDF files; the checksum list is the file list
grep -E ' sleep-cassette/SC4[0-9]{3}[A-Z][0-9A-Z]-(PSG|Hypnogram)\.edf$' SHA256SUMS.txt | LC_ALL=C sort -k2 > sc.sha256
ALL=$(wc -l < sc.sha256)
[ "$ALL" -gt 0 ] || { echo "no sleep-cassette EDF lines in SHA256SUMS.txt" >&2; exit 2; }
# sorted order puts each PSG before its hypnogram; LIMIT=20 is exactly the 10 beta nights (SC400-SC404)
[ "$LIMIT" -gt 0 ] && head -n "$LIMIT" sc.sha256 > sc.sha256.tmp && mv sc.sha256.tmp sc.sha256
N=$(wc -l < sc.sha256)
echo "$(date '+%F %T') $N files listed, rate $RATE, wait ${WAIT}s, into $OUT"

ok=0; fail=0; i=0
while read -r sum rel; do
  i=$((i + 1))
  if [ -s "$rel" ] && echo "$sum  $rel" | sha256sum -c --status - 2>/dev/null; then ok=$((ok + 1)); continue; fi
  "${WG[@]}" -c -P sleep-cassette "$BASE/$rel"     # -c resumes a partial file of the same name
  if echo "$sum  $rel" | sha256sum -c --status - 2>/dev/null; then
    ok=$((ok + 1))
  else   # a corrupt partial file would make wget -c append forever: delete it, the next run starts it fresh
    echo "checksum mismatch: $rel (deleted; rerun to fetch again)" >&2; rm -f "$rel"; fail=$((fail + 1))
  fi
  [ $((i % 20)) -eq 0 ] && echo "$(date '+%F %T') $i/$N"
  sleep "$WAIT"
done < sc.sha256

echo "$(date '+%F %T') verified $ok/$N, failed $fail; $(du -sh sleep-cassette | cut -f1) on disk"
[ "$fail" -eq 0 ] && [ "$ok" -eq "$N" ] || exit 1
# the marker that sleep_prep.sbatch requires means "the whole SC set": never after a LIMIT run or a short list
if [ "$LIMIT" -eq 0 ] && [ "$N" -eq "$EXPECTED" ]; then
  date -u > VERIFIED
else
  rm -f VERIFIED
  echo "partial set ($N files; full set is $EXPECTED): no VERIFIED marker"
fi
exit 0
