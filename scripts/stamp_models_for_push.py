"""Prepare model folders for the R2 models bucket: copy them into one output folder and give each a manifest in the
model-registry format that scripts/fetch_models.py --push (and the API's pull at start) requires.

  python scripts/stamp_models_for_push.py --out <push dir> \
      --ecg <beta ecg folder> --seizure <beta seizure folder> --spaceweather <spaceweather folder> \
      --ecg-deep <deep ECG export folder> --id ecg_deep=EVA-13

Then, with the models bucket's WRITE key in the environment (never in a file):
  for t in ecg seizure spaceweather ecg_deep; do python scripts/fetch_models.py --push <push dir>/$t; done

- ecg / seizure: the beta joblib folders have no manifest (the API serves them through a synthesized legacy entry,
  marked unverified). Here they get a real manifest with the registry's legacy specs, EVA-6 / EVA-2, and sha256 for
  the weights, feature_names.json and metrics.json.
- spaceweather: converted from its original manifest layout (sha256 per file) to the registry layout; the original
  fields are kept. evade.spaceweather.model.load_forecaster reads both layouts.
- ecg_deep (or any folder that already has a registry manifest): re-stamped with --id, every file re-hashed.
Only model files are copied (model.*, feature_names.json, metrics.json). Folder names are the task names the API
looks up. Outputs stay status "draft" (the API serves drafts only with EVADE_ALLOW_DRAFT=1).
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evade.models import manifest as MF  # noqa: E402
from evade.models.export import headline_from_metrics, restamp, write_manifest  # noqa: E402
from evade.models.registry import LEGACY  # noqa: E402

KEEP = re.compile(r"^(model\.(joblib|onnx|ts|pt)|feature_names\.json|metrics\.json|manifest\.json)$")


def copy_folder(src: Path, dst: Path) -> None:
    if not src.is_dir():
        raise SystemExit(f"not a folder: {src}")
    if dst.exists():
        shutil.rmtree(dst)
    dst.mkdir(parents=True)
    for f in src.iterdir():
        if f.is_file() and KEEP.match(f.name):
            shutil.copy2(f, dst / f.name)


def metrics_of(d: Path) -> dict:
    p = d / "metrics.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.is_file() else {}


def stamp_legacy(task: str, d: Path, model_id: str | None) -> dict:
    spec = LEGACY[task]
    (d / "manifest.json").unlink(missing_ok=True)
    return write_manifest(d, task=task, input_spec=spec["input"], output_spec=spec["output"], data=spec["data"],
                          model_id=model_id or spec["model_id"], headline=headline_from_metrics(task, metrics_of(d)))


def stamp_spaceweather(d: Path, model_id: str | None) -> dict:
    old = json.loads((d / "manifest.json").read_text(encoding="utf-8"))
    if isinstance(old.get("sha256"), str):                       # already in the registry layout
        return restamp(d, model_id=model_id)
    want = old.get("sha256") or {}
    for name in ("model.joblib", "feature_names.json", "metrics.json"):   # the copy must match the trained files
        if (d / name).is_file() and name in want and MF.sha256_file(d / name) != want[name]:
            raise SystemExit(f"spaceweather {name} does not match its original manifest; not stamping")
    keep = {k: v for k, v in old.items() if k not in ("sha256", "model_id", "status", "format", "task", "weights",
                                                      "files", "input", "output", "headline", "data")}
    keep["original_task"] = old.get("task")
    return write_manifest(d, task="spaceweather",
                          input_spec={"kind": "features", "modality": "space_weather", "features": "feature_names.json"},
                          output_spec={"type": "binary", "classes": ["spe_onset_24h"],
                                       "note": "probability that >=10 MeV protons reach 10 pfu within 24 h"},
                          data=old.get("training_data_range") or "NOAA NCEI GOES (see MODEL_REGISTRY.md)",
                          model_id=model_id or old.get("model_id"), status=old.get("status", "draft"),
                          headline=headline_from_metrics("spaceweather", metrics_of(d)),
                          commit=old.get("git_commit"), trained_at=old.get("created_utc"), extra=keep)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    for t in ("ecg", "seizure", "spaceweather", "ecg-deep", "sleep-deep", "seizure-deep"):
        ap.add_argument(f"--{t}", help=f"source folder for {t.replace('-', '_')}")
    ap.add_argument("--id", action="append", default=[], metavar="TASK=EVA-n")
    a = ap.parse_args(argv)
    ids = dict(x.split("=", 1) for x in a.id)
    out = Path(a.out)
    done = {}
    for arg, task in (("ecg", "ecg"), ("seizure", "seizure"), ("spaceweather", "spaceweather"),
                      ("ecg_deep", "ecg_deep"), ("sleep_deep", "sleep_deep"), ("seizure_deep", "seizure_deep")):
        src = getattr(a, arg)
        if not src:
            continue
        d = out / task
        copy_folder(Path(src), d)
        if task in LEGACY:
            m = stamp_legacy(task, d, ids.get(task))
        elif task == "spaceweather":
            m = stamp_spaceweather(d, ids.get(task))
        else:
            if not (d / "manifest.json").is_file():
                raise SystemExit(f"{task}: no manifest.json in {src} (export it with the training code first)")
            m = restamp(d, model_id=ids.get(task))
        MF.validate(m)                                            # refuses UNASSIGNED: every pushed folder needs an ID
        done[task] = {"model_id": m["model_id"], "format": m["format"], "files": [m["weights"], *m.get("files", {})]}
    if not done:
        ap.error("give at least one source folder")
    print(json.dumps(done, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
