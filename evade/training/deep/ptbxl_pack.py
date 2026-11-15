"""Pack PTB-XL 100 Hz records into one compressed npz for GPU training (no wfdb needed).

  python -m evade.training.deep.ptbxl_pack --src <extracted ptb-xl folder> --out ptbxl100.npz [--limit N]

Output arrays:
  X        float16 (N, 12, 1000)   millivolts, 10 s at 100 Hz, leads in header order (I II III aVR aVL aVF V1-V6)
  y        uint8   (N, 5)          diagnostic superclass multi-hot: NORM MI STTC CD HYP
  fold     int8    (N,)            strat_fold 1-10 (official split: train 1-8, val 9, test 10)
  patient  int64   (N,)            patient_id (bootstrap unit)
  ecg_id   int64   (N,)
Records with no diagnostic superclass are dropped, as in the PTB-XL benchmark.
The dataset files are read as data only.
"""
from __future__ import annotations

import argparse
import ast
import re
from pathlib import Path

import numpy as np
import pandas as pd

SUPER = ["NORM", "MI", "STTC", "CD", "HYP"]
LEADS = ["I", "II", "III", "AVR", "AVL", "AVF", "V1", "V2", "V3", "V4", "V5", "V6"]
_GAIN = re.compile(r"^([-+0-9.eE]+)(?:\(([-+0-9]+)\))?")


def read_record(hea: Path) -> np.ndarray:
    """Read a WFDB format-16 record (header + .dat) and return float32 (n_sig, n_samp) in physical units."""
    lines = [ln for ln in hea.read_text().splitlines() if ln.strip() and not ln.startswith("#")]
    head = lines[0].split()
    n_sig, n_samp = int(head[1]), int(head[3])
    gains, bases, fmt, dat = [], [], None, None
    for ln in lines[1:1 + n_sig]:
        f = ln.split()
        dat, fmt = f[0], f[1]
        m = _GAIN.match(f[2])
        g = float(m.group(1)) if m else 200.0
        gains.append(g if g != 0 else 200.0)
        bases.append(int(m.group(2)) if m and m.group(2) else (int(f[4]) if len(f) > 4 else 0))
    if fmt != "16":
        raise ValueError(f"{hea.name}: unsupported format {fmt}")
    raw = np.fromfile(hea.with_name(dat), dtype="<i2")
    raw = raw[: n_sig * n_samp].reshape(n_samp, n_sig).T.astype(np.float32)
    return (raw - np.asarray(bases, np.float32)[:, None]) / np.asarray(gains, np.float32)[:, None]


def superclasses(scp_codes: str, diag_map: dict) -> np.ndarray:
    y = np.zeros(len(SUPER), np.uint8)
    for code in ast.literal_eval(scp_codes):
        c = diag_map.get(code)
        if c in SUPER:
            y[SUPER.index(c)] = 1
    return y


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--src", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--limit", type=int, default=0, help="smoke: first N records only")
    a = p.parse_args()
    db = pd.read_csv(a.src / "ptbxl_database.csv", index_col="ecg_id")
    st = pd.read_csv(a.src / "scp_statements.csv", index_col=0)
    diag_map = st[st["diagnostic"] == 1]["diagnostic_class"].to_dict()
    if a.limit:
        db = db.iloc[: a.limit]
    X, Y, F, P, E = [], [], [], [], []
    for ecg_id, r in db.iterrows():
        y = superclasses(r["scp_codes"], diag_map)
        if y.sum() == 0:
            continue
        x = read_record(a.src / f"{r['filename_lr']}.hea")
        if x.shape != (12, 1000) or not np.isfinite(x).all():
            continue
        X.append(x.astype(np.float16)); Y.append(y); F.append(int(r["strat_fold"]))
        P.append(int(r["patient_id"])); E.append(int(ecg_id))
    X = np.stack(X)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(a.out, X=X, y=np.stack(Y), fold=np.asarray(F, np.int8), patient=np.asarray(P, np.int64),
                        ecg_id=np.asarray(E, np.int64), classes=np.asarray(SUPER), leads=np.asarray(LEADS))
    counts = dict(zip(SUPER, np.stack(Y).sum(0).tolist()))
    print(f"packed {len(X)} records of {len(db)} -> {a.out}; per class {counts}; "
          f"folds {np.bincount(np.asarray(F), minlength=11)[1:].tolist()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
