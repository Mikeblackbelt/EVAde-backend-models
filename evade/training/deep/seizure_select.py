"""Pick which EDF files to download: the beta file rule of evade.training.data (chb_jobs, siena_jobs) applied to a
dataset's SHA256SUMS listing, so the download fetches exactly the files the pack will use.

  python -m evade.training.deep.seizure_select --dataset chbmit --root <data dir> --listing <hash  path file> \
         [--max-free 3]

Reads "hash  path" lines, prints the lines to keep: every non-EDF line, every EDF with a seizure, and the first
`--max-free` seizure-free EDFs per subject in sorted order (-1 keeps every EDF). Seizure files come from the
summaries already downloaded into --root (CHB-MIT chbNN-summary.txt, parsed with the pack's parser; Siena
Seizures-list-*.txt, a file counts if its name appears in the list, as siena_jobs does). A missing summary is an
error, never "no seizures". Data files are read as text only.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from ...preprocessing.eeg import parse_chb_summary


def seizure_test(dataset: str, root: Path, case: str):
    if dataset == "chbmit":
        summ = root / case / f"{case}-summary.txt"
        if not summ.exists():
            raise FileNotFoundError(f"{summ} missing: download the summaries first")
        seiz = parse_chb_summary(summ.read_text(errors="ignore"))
        return lambda name: bool(seiz.get(name))
    lst = sorted((root / case).glob("Seizures-list-*.txt"))
    if not lst:
        raise FileNotFoundError(f"no Seizures-list-*.txt in {root / case}: download the lists first")
    text = lst[0].read_text(errors="ignore")
    return lambda name: name in text


def select(dataset: str, root: Path, lines: list[str], max_free: int) -> list[str]:
    edfs: dict[str, list[tuple[str, str]]] = {}
    keep = []
    for line in lines:
        parts = line.split()
        if len(parts) != 2:
            continue
        path = parts[1]
        sub = path.split("/")
        if len(sub) == 2 and sub[1].endswith(".edf"):
            edfs.setdefault(sub[0], []).append((sub[1], line))
        else:
            keep.append(line)
    for case in sorted(edfs):
        if max_free < 0:
            keep += [ln for _, ln in sorted(edfs[case])]
            continue
        is_sz = seizure_test(dataset, root, case)
        free = 0
        for name, ln in sorted(edfs[case]):          # same order as sorted(subj.glob("*.edf"))
            if is_sz(name):
                keep.append(ln)
            elif free < max_free:
                keep.append(ln); free += 1
    return keep


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True, choices=["chbmit", "siena"])
    p.add_argument("--root", required=True, type=Path)
    p.add_argument("--listing", required=True, type=Path)
    p.add_argument("--max-free", type=int, default=3)
    a = p.parse_args(argv)
    lines = a.listing.read_text().splitlines()
    sys.stdout.write("".join(ln + "\n" for ln in select(a.dataset, a.root, lines, a.max_free)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
