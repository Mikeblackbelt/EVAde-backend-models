#!/usr/bin/env bash
# Download GOES-R (GOES-16/18/19) proton and X-ray L2 data from NOAA NCEI for the EVAde space-weather forecast.
# Sequential and polite: one request at a time, a pause after each, resumable (finished files are skipped).
# Usage: download_goesr_history.sh [--dry-run]
#   --dry-run lists what would be fetched (file counts and listed sizes) without downloading data files.
# Output: ~/evade_data/other/spaceweather/goesr/{sgps,xrs}/g<NN>/... and goesr_download.log next to it.
# Products (NCEI, public): SEISS SGPS L2 5-min averages (daily netCDF, ~0.6 MB) and EXIS XRS L2 1-min science
# averages (yearly netCDF, ~30 MB). Files are untrusted data: read as netCDF/HDF5 only, never executed.
set -u
DRY=0; [ "${1:-}" = "--dry-run" ] && DRY=1
ROOT="$HOME/evade_data/other/spaceweather/goesr"
LOG="$HOME/evade_data/other/spaceweather/goesr_download.log"
BASE="https://data.ngdc.noaa.gov/platforms/solar-space-observing-satellites/goes"
CURL="curl -sS --fail --max-time 300 --retry 2 --retry-delay 10 -A evade-research-download"
mkdir -p "$ROOT"
log() { echo "$(date -u +%FT%TZ) $*" | tee -a "$LOG"; }
avail_gb() { df -BG --output=avail "$ROOT" | tail -1 | tr -dc 0-9; }
if [ "$(avail_gb)" -lt 60 ]; then echo "less than 60 GB free; stopping"; exit 75; fi

get() { # get <dir> <url> <file>
  [ -f "$1/$2" ] && return 0
  [ "$DRY" = 1 ] && return 0
  if $CURL -o "$1/$2.part" "$3/$2"; then mv "$1/$2.part" "$1/$2"; sleep 1; return 0; fi
  log "FAIL $3/$2"; sleep 3; return 1
}

listing() { $CURL "$1/" 2>/dev/null; sleep 1; }

sizes() { # print total of the sizes shown next to matching files in an Apache listing (K/M/G suffixes)
  grep -o "$1\">[^<]*</a></td><td[^>]*>[^<]*</td><td[^>]*>[^<]*" | sed 's/.*>//' | awk '
    { v=$1; u=substr(v,length(v)); n=v+0; if (u=="K") n*=1e3; else if (u=="M") n*=1e6; else if (u=="G") n*=1e9; s+=n }
    END { printf "%.0f", s }'
}

xrs() { # xrs <sat> <first year> <last year> : yearly 1-min science files
  local sat=$1
  local url="$BASE/goes$sat/l2/data/xrsf-l2-avg1m_science" out="$ROOT/xrs/g$sat"
  mkdir -p "$out"
  local L; L=$(listing "$url") || { log "LISTFAIL xrs g$sat"; return; }
  local files; files=$(echo "$L" | grep -o "sci_xrsf-l2-avg1m_g${sat}_y[0-9]*_v[0-9-]*\.nc" | sort -u |
                       awk -v a="$2" -v b="$3" '{ match($0, /_y[0-9]+_/); y = substr($0, RSTART+2, 4) + 0; if (y >= a && y <= b) print }')
  log "XRS g$sat: $(echo "$files" | grep -c .) yearly files, listed bytes $(echo "$L" | sizes "_y[0-9]*_v[0-9-]*\.nc")"
  for f in $files; do get "$out" "$f" "$url"; done
}

sgps() { # sgps <sat> <first YYYY-MM> <last YYYY-MM> : daily 5-min average files
  local sat=$1 from=$2 to=$3 n=0 bytes=0
  local url="$BASE/goes$sat/l2/data/sgps-l2-avg5m"
  local d="$from-01"
  while [ "$(date -u -d "$d" +%Y-%m)" \< "$to" ] || [ "$(date -u -d "$d" +%Y-%m)" = "$to" ]; do
    local y m out L files; y=$(date -u -d "$d" +%Y); m=$(date -u -d "$d" +%m)
    out="$ROOT/sgps/g$sat/$y/$m"; mkdir -p "$out"
    if [ -f "$out/.done" ]; then d=$(date -u -d "$d +1 month" +%Y-%m-01); continue; fi
    if L=$(listing "$url/$y/$m"); then
      files=$(echo "$L" | grep -o "sci_sgps-l2-avg5m_g${sat}_d[0-9]*_v[0-9-]*\.nc" | sort -u)
      n=$((n + $(echo "$files" | grep -c .))); bytes=$((bytes + $(echo "$L" | sizes "\.nc")))
      ok=1
      for f in $files; do get "$out" "$f" "$url/$y/$m" || ok=0; done
      if [ "$DRY" = 0 ] && [ "$ok" = 1 ]; then touch "$out/.done"; log "OK sgps g$sat $y-$m"; fi
    else
      log "LISTFAIL sgps g$sat $y-$m (skipped)"
    fi
    d=$(date -u -d "$d +1 month" +%Y-%m-01)
  done
  log "SGPS g$sat $from..$to: $n daily files, listed bytes $bytes"
}

log "START dry=$DRY"
xrs 16 2017 2022; xrs 18 2022 2026
sgps 16 2020-01 2022-12          # NCEI SGPS L2 averages start 2020; GOES-16 until GOES-18 takes over
sgps 18 2022-09 2026-11          # GOES-18 (the SWPC primary proton satellite in 2026); 4-month overlap with GOES-16
log "SKIP goes16 2023-2025, goes17 and goes19: not needed for one proton series plus one overlap window (proof of concept)"
log "DONE dry=$DRY"
