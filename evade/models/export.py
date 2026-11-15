"""Write manifest.json for a trained model folder so it can be served by evade.models.registry.

Called by evade.training.train after every run. To give an existing folder its registry ID or promote it:

  python -m evade.models.export --dir <EVADE_MODEL_DIR>/<task> --model-id EVA-7 [--status validated]
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from . import manifest as MF

HASHED_EXTRAS = ("feature_names.json", "metrics.json")


def code_commit() -> str:
    """EVADE_CODE_COMMIT, else a COMMIT file at the tree root (tar uploads have no .git), else git, else unknown."""
    env = os.environ.get("EVADE_CODE_COMMIT")
    if env:
        return env.strip()
    root = Path(__file__).resolve().parents[2]
    f = root / "COMMIT"
    if f.is_file():
        return f.read_text(encoding="utf-8").strip()[:64]
    try:
        r = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"], capture_output=True, text=True, timeout=10)
        if r.returncode == 0 and r.stdout.strip():
            return r.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        pass
    return "unknown"


def write_manifest(out_dir: str | Path, *, task: str, input_spec: dict, output_spec: dict, data: object,
                   model_id: str | None = None, status: str = "draft", fmt: str = "joblib",
                   weights: str = "model.joblib", headline: dict | None = None, commit: str | None = None,
                   trained_at: str | None = None, extra: dict | None = None) -> dict:
    """Hash the weights (and feature_names.json / metrics.json if present) and write <out_dir>/manifest.json.

    model_id None writes UNASSIGNED: the registry lists the folder but will not serve it until a
    MODEL_REGISTRY.md row exists and the folder is re-stamped with --model-id.
    """
    d = Path(out_dir)
    w = d / MF.safe_name(weights, "weights")
    if not w.is_file():
        raise FileNotFoundError(f"weights file not found: {w.name}")
    files = {n: MF.sha256_file(d / n) for n in HASHED_EXTRAS if (d / n).is_file()}
    m = {"schema_version": MF.SCHEMA_VERSION, "model_id": model_id or MF.UNASSIGNED, "task": task, "status": status,
         "format": fmt, "weights": w.name, "sha256": MF.sha256_file(w), "files": files, "input": input_spec,
         "output": output_spec, "headline": headline or {}, "code_commit": commit or code_commit(),
         "trained_at": trained_at or datetime.now(timezone.utc).isoformat(timespec="seconds"), "data": data,
         "claim": "outputs are flags for human review, not diagnoses", **(extra or {})}
    MF.validate(m, allow_unassigned=True)
    tmp = d / (MF.MANIFEST + ".tmp")
    tmp.write_text(json.dumps(m, indent=2, default=float), encoding="utf-8")
    os.replace(tmp, d / MF.MANIFEST)
    return m


def restamp(out_dir: str | Path, *, model_id: str | None = None, status: str | None = None) -> dict:
    """Set model_id and/or status on an existing manifest and re-hash every file."""
    d = Path(out_dir)
    m = json.loads((d / MF.MANIFEST).read_text(encoding="utf-8"))
    keep = {k: v for k, v in m.items() if k not in ("schema_version", "model_id", "task", "status", "format", "weights",
                                                     "sha256", "files", "input", "output", "headline", "code_commit",
                                                     "trained_at", "data", "claim")}
    return write_manifest(d, task=m["task"], input_spec=m["input"], output_spec=m["output"], data=m["data"],
                          model_id=model_id or (None if m["model_id"] == MF.UNASSIGNED else m["model_id"]),
                          status=status or m["status"], fmt=m["format"], weights=m["weights"],
                          headline=m.get("headline"), commit=m["code_commit"], trained_at=m["trained_at"], extra=keep)


def headline_from_metrics(task: str, m: dict) -> dict:
    """Short metric summary for /v1/models, for the beta tasks whose metrics.json layout is known."""
    try:
        if task == "seizure":
            h = m["leave_one_subject_out_chbmit"]["hgb"]
            return {"loso_pooled_auroc": h["pooled_auroc"], "subject_auroc_ci": h["subject_auroc_ci"],
                    "external_siena_auroc": m.get("external_siena", {}).get("hgb", {}).get("auroc")}
        if task == "ecg":
            return {"macro_auroc_test_fold": m["hgb"]["macro_auroc"], "ci95": m["hgb"]["macro_auroc_ci95"]}
        if task == "sleep":
            h = m["leave_one_subject_out"]
            return {"subject_kappa_ci": h["hgb"]["subject_kappa_ci"],
                    "logreg_baseline_kappa_ci": h["logreg_baseline"]["subject_kappa_ci"]}
    except (KeyError, TypeError):
        return {}
    return {}


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Assign a registry ID or status to a model folder (re-hashes files).")
    p.add_argument("--dir", required=True)
    p.add_argument("--model-id")
    p.add_argument("--status", choices=MF.STATUSES)
    a = p.parse_args(argv)
    m = restamp(a.dir, model_id=a.model_id, status=a.status)
    print(json.dumps({k: m[k] for k in ("model_id", "task", "status", "format", "sha256")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
