#!/usr/bin/env bash
# Download historical GOES proton and X-ray averages from NOAA NCEI for the EVA space-weather forecast.
# Sequential and polite: one request at a time, a pause between requests, resumable (skips finished files).
# Usage: download_goes_history.sh [first_year] [last_year]   (default 1998 2019)
# Output: ~/evade_data/other/spaceweather/raw/<YYYY>/<MM>/<file>.csv and download.log next to raw/.
# Protons: one satellite per month (first in the preference list with a 5-min integral file).
# X-rays: every preferred satellite's file (small); preprocessing picks the one with the most valid samples.
# Source: https://www.ncei.noaa.gov/data/goes-space-environment-monitor/access/avg/ (NOAA, public domain).
# Treat the files as untrusted data: they are parsed as CSV text only, never executed.
set -u
Y0=${1:-1998}; Y1=${2:-2019}
ROOT="$HOME/evade_data/other/spaceweather"
RAW="$ROOT/raw"; LOG="$ROOT/download.log"
BASE="https://www.ncei.noaa.gov/data/goes-space-environment-monitor/access/avg"
CURL="curl -sS --fail --max-time 300 --retry 2 --retry-delay 10 -A evade-research-download"
mkdir -p "$RAW"
log() { echo "$(date -u +%FT%TZ) $*" >> "$LOG"; }
avail_gb() { df -BG --output=avail "$ROOT" | tail -1 | tr -dc 0-9; }
if [ "$(avail_gb)" -lt 60 ]; then echo "less than 60 GB free; stopping"; exit 75; fi

get() { # get <dir> <url> <file>: resumable single-file download
  [ -f "$1/$3" ] && return 0
  if $CURL -o "$1/$3.part" "$2/$3"; then mv "$1/$3.part" "$1/$3"; sleep 2; return 0; fi
  log "FAIL $2/$3"; sleep 2; return 1
}

log "START $Y0-$Y1"
for y in $(seq "$Y0" "$Y1"); do
  if [ "$y" -lt 2010 ]; then PREF="goes11 goes10 goes08 goes12"; else PREF="goes15 goes13 goes11"; fi
  for m in 01 02 03 04 05 06 07 08 09 10 11 12; do
    out="$RAW/$y/$m"
    [ -f "$out/.done" ] && continue
    mkdir -p "$out"
    if ! top=$($CURL "$BASE/$y/$m/"); then log "LISTFAIL $y-$m"; sleep 5; continue; fi
    sleep 1
    sats=$(echo "$top" | grep -o 'goes[0-9][0-9]/' | tr -d / | sort -u)
    ok=1; psat=""; nx=0
    for s in $PREF; do
      echo "$sats" | grep -qx "$s" || continue
      if ! listing=$($CURL "$BASE/$y/$m/$s/csv/"); then log "LISTFAIL $y-$m $s"; ok=0; sleep 5; continue; fi
      sleep 1
      n=${s#goes}
      if [ -z "$psat" ]; then
        p=$(echo "$listing" | grep -o "g${n}_\(eps_5m\|epead_cpflux_5m\)_[0-9]*_[0-9]*\.csv" | head -1)
        if [ -n "$p" ]; then get "$out" "$BASE/$y/$m/$s/csv" "$p" && psat="$s" || ok=0; fi
      fi
      x=$(echo "$listing" | grep -o "g${n}_xrs_5m_[0-9]*_[0-9]*\.csv" | head -1)
      [ -z "$x" ] && x=$(echo "$listing" | grep -o "g${n}_xrs_1m_[0-9]*_[0-9]*\.csv" | head -1)
      if [ -n "$x" ]; then get "$out" "$BASE/$y/$m/$s/csv" "$x" && nx=$((nx+1)) || ok=0; fi
    done
    if [ "$ok" = 1 ]; then echo "protons=$psat xrs_files=$nx" > "$out/.done"; log "OK $y-$m protons=$psat xrs_files=$nx"; fi
  done
done
log "DONE $Y0-$Y1"
