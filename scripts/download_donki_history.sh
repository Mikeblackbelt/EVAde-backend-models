#!/usr/bin/env bash
# Download the NASA DONKI event catalogue (FLR, CME, SEP, IPS, GST) for 2010 to now, for training the EVAde forecast.
# Sequential and slow on purpose: one request, then a long pause (api.nasa.gov DEMO_KEY allows about 30 requests/hour).
# Source order: the keyless CCMC endpoint (kauai.ccmc.gsfc.nasa.gov) first; if it does not answer, api.nasa.gov with
# $NASA_API_KEY, or the public DEMO_KEY if that variable is not set. The key is read from the environment only.
# Usage: download_donki_history.sh [first_year] [last_year]   (default 2010 to the current year)
# Output: ~/evade_data/other/spaceweather/donki/<TYPE>_<start>_<end>.json and donki_download.log next to it.
set -u
Y0=${1:-2010}; Y1=${2:-$(date -u +%Y)}
OUT="$HOME/evade_data/other/spaceweather/donki"; LOG="$HOME/evade_data/other/spaceweather/donki_download.log"
CURL="curl -sS --fail --max-time 300 -A evade-research-download"
PAUSE=${DONKI_PAUSE_S:-130}
mkdir -p "$OUT"
log() { echo "$(date -u +%FT%TZ) $*" | tee -a "$LOG"; }

CCMC_OK=0
if curl -s -o /dev/null --connect-timeout 15 --max-time 40 "https://kauai.ccmc.gsfc.nasa.gov/DONKI/WS/get/IPS?startDate=2017-09-01&endDate=2017-09-02"; then CCMC_OK=1; fi
log "START $Y0-$Y1 ccmc_reachable=$CCMC_OK"
KEY=${NASA_API_KEY:-DEMO_KEY}

fetch() { # fetch <type> <start> <end>
  local f="$OUT/$1_$2_$3.json"
  [ -s "$f" ] && return 0
  local url
  if [ "$CCMC_OK" = 1 ]; then url="https://kauai.ccmc.gsfc.nasa.gov/DONKI/WS/get/$1?startDate=$2&endDate=$3"
  else url="https://api.nasa.gov/DONKI/$1?startDate=$2&endDate=$3&api_key=$KEY"; fi
  if $CURL -o "$f.part" "$url"; then mv "$f.part" "$f"; log "OK $1 $2..$3 $(stat -c %s "$f") bytes"
  else rm -f "$f.part"; log "FAIL $1 $2..$3"; fi
  sleep "$PAUSE"
}

for t in FLR SEP IPS GST; do fetch "$t" "$Y0-01-01" "$Y1-12-31"; done
y=$Y0
while [ "$y" -le "$Y1" ]; do                      # CMEs carry analyses and model runs: 2-year chunks
  e=$((y + 1)); [ "$e" -gt "$Y1" ] && e=$Y1
  fetch CME "$y-01-01" "$e-12-31"
  y=$((e + 1))
done
log "DONE"
