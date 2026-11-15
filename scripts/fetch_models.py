"""Sync model folders from Cloudflare R2 into EVADE_MODEL_DIR (sha256-verified), or push one folder up.

  python scripts/fetch_models.py [--task seizure --task spaceweather] [--dest DIR]
  python scripts/fetch_models.py --push <EVADE_MODEL_DIR>/<task>

Env: CF_ACCOUNT_ID, CF_R2_BUCKET, CF_R2_ACCESS_KEY_ID, CF_R2_SECRET_ACCESS_KEY (pip install boto3), EVADE_MODEL_DIR.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evade.models.fetch import pull, push  # noqa: E402
from evade.storage import r2_from_env  # noqa: E402


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--dest", default=os.environ.get("EVADE_MODEL_DIR"))
    p.add_argument("--task", action="append", help="only these tasks (repeatable)")
    p.add_argument("--push", help="upload this local model folder instead of downloading")
    a = p.parse_args(argv)
    blobs = r2_from_env(required=True)
    if a.push:
        print(json.dumps({"pushed": push(blobs, a.push)}))
        return 0
    if not a.dest:
        p.error("set EVADE_MODEL_DIR or pass --dest")
    res = pull(blobs, a.dest, a.task)
    print(json.dumps(res, indent=2))
    return 1 if any(v.startswith("failed") for v in res.values()) else 0


if __name__ == "__main__":
    raise SystemExit(main())
