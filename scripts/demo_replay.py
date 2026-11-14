"""Replay a real CHB-MIT recording through a running EVAde API (demo, no clinical meaning).

  EVADE_API_TOKEN=... python scripts/demo_replay.py --api http://127.0.0.1:8000 \
      --edf ~/evade_data/eeg/chbmit/chb01/chb01_03.edf --seizure 2996 3036

Steps: create a user, send the first --baseline-s seconds as the user's EEG baseline (in 10 s calls to stay
under the 1 MB cap), then send 10 s windows from --from-s to --to-s and print each flag next to the annotation.
The dataset file is read as data only.
"""
from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import httpx
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from evade.preprocessing.datasets import read_edf  # noqa: E402
from evade.preprocessing.eeg import bipolar_from_monopolar, select_bipolar  # noqa: E402
from evade.preprocessing import config  # noqa: E402


def window(x, fs, t0, secs):
    a, b = int(t0 * fs), int((t0 + secs) * fs)
    return {"modality": "eeg", "t0_utc": datetime.now(timezone.utc).isoformat(), "fs_hz": fs, "unit": "uv",
            "channels": config.BIPOLAR_NAMES, "samples": np.round(x[:, a:b], 2).tolist(), "source": "replay"}


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--api", default="http://127.0.0.1:8000")
    p.add_argument("--edf", required=True)
    p.add_argument("--seizure", nargs=2, type=float, default=None, help="annotated onset and offset, seconds")
    p.add_argument("--baseline-s", type=float, default=120.0)
    p.add_argument("--from-s", type=float, default=2900.0)
    p.add_argument("--to-s", type=float, default=3100.0)
    p.add_argument("--win-s", type=float, default=10.0, help="seconds per request (512 Hz data: use 8 to stay under 1 MB)")
    a = p.parse_args()
    h = {"Authorization": f"Bearer {os.environ['EVADE_API_TOKEN']}"}
    data, names, fs, _ = read_edf(Path(a.edf).expanduser())
    bip = any('-' in n for n in names)
    x, present = select_bipolar(data, names) if bip else bipolar_from_monopolar(data, names)   # CHB-MIT vs Siena
    c = httpx.Client(base_url=a.api, headers=h, timeout=60)
    uid = c.post("/v1/users", json={"consent": True, "cohort": "demo"}).raise_for_status().json()["data"]["user_id"]
    W = a.win_s
    for t in np.arange(0, a.baseline_s, W):
        c.post("/v1/baseline/eeg", json={"user_id": uid, "windows": [window(x, fs, t, W)]}).raise_for_status()
    print(f"user {uid[:8]}: baseline from 0-{a.baseline_s:.0f} s; channels mapped {sum(1 for q in present if q)}/18")
    print(f"{'window (s)':>12} {'annotated':>10} {'flag':>16} {'p':>6} {'theta/beta z':>13}")
    for t in np.arange(a.from_s, a.to_s, W):
        r = c.post("/v1/predict/eeg", json={"user_id": uid, "window": window(x, fs, t, W)}).raise_for_status().json()["data"]
        sz = r.get("seizure", {})
        ann = "seizure" if a.seizure and (t < a.seizure[1] and t + W > a.seizure[0]) else ""
        tb = next((cmp.get("z") for cmp in r["comparison"] if cmp["metric"] == "theta_beta"), None)
        print(f"{t:6.0f}-{t + W:<5.0f} {ann:>10} {str(sz.get('flag')):>16} {sz.get('probability', float('nan')):6.3f} "
              f"{'' if tb is None else round(tb, 2):>13}")
    risk = c.post("/v1/predict/risk", json={"user_id": uid, "logistics_floor_days": 0.5}).raise_for_status().json()
    print("risk:", risk["data"]["risk"], "| MDTR:", risk["data"]["mdtr"]["display"], "| warnings:", risk["warnings"])
    c.delete(f"/v1/users/{uid}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
