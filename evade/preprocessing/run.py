"""Command line: python -m evade.preprocessing.run --dataset mitdb --root DIR --out DIR [--limit N]

Writes <out>/<dataset>_windows.parquet (schema table) plus per-record .npz files.
CPU only, single process. Run heavy jobs through the guard on the shared server.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .datasets import RUNNERS


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=sorted(RUNNERS))
    ap.add_argument("--root", required=True, type=Path, help="dataset folder")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--limit", type=int, default=None, help="max records (use for smoke runs)")
    a = ap.parse_args(argv)
    if not a.root.exists():
        print(f"missing dataset folder: {a.root}", file=sys.stderr)
        return 2
    a.out.mkdir(parents=True, exist_ok=True)
    df = RUNNERS[a.dataset](a.root, a.out, a.limit)
    df.to_parquet(a.out / f"{a.dataset}_windows.parquet")
    n = len(df)
    lq = float(df["low_quality"].mean()) if n and "low_quality" in df else float("nan")
    print(f"{a.dataset}: {n} rows, low_quality fraction {lq:.3f}, wrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
