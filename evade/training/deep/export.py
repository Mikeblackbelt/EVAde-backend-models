"""Export a trained deep ECG run to a servable model folder (the EVAde model artifact contract).

  python -m evade.training.deep.export --run <run dir> --out <export dir>/ecg_deep [--model-id EVA-n]

Writes model.onnx (if the ONNX exporter is available, else model.ts TorchScript), feature_names.json (leads),
metrics.json (from the run) and manifest.json with sha256 of every file, the input window, normalization and classes.
The API preprocesses with the manifest: bandless, per-lead (x - mean) / std, 2.5 s windows, sigmoid outputs averaged.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from .models import build


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def code_commit() -> str:
    env = os.environ.get("EVADE_CODE_COMMIT")
    if env:
        return env
    for c in (Path(__file__).resolve().parents[3] / "COMMIT",):
        if c.exists():
            return c.read_text().strip()
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True, type=Path, help="run dir with metrics.json and seed*/best.pt")
    p.add_argument("--arch", default="xresnet1d101")
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--model-id", default="UNASSIGNED")
    a = p.parse_args()
    m = json.loads((a.run / "metrics.json").read_text())
    seed = m["selected_seed"]
    leads, classes, win = m["leads"], m["classes"], m["window"]
    model = build(a.arch, len(leads), len(classes))
    model.load_state_dict(torch.load(a.run / f"seed{seed}" / "best.pt", map_location="cpu", weights_only=True))
    model.eval()
    a.out.mkdir(parents=True, exist_ok=True)
    x = torch.zeros(1, len(leads), win["crop_samples"])
    fmt, weights = None, None
    try:
        torch.onnx.export(model, x, a.out / "model.onnx", input_names=["x"], output_names=["logits"],
                          dynamic_axes={"x": {0: "batch"}, "logits": {0: "batch"}}, opset_version=17, dynamo=False)
        fmt, weights = "onnx", "model.onnx"
    except Exception as e:  # exporter or onnx package missing: fall back to TorchScript
        print(f"onnx export unavailable ({type(e).__name__}: {e}); writing TorchScript", flush=True)
        with torch.no_grad():
            torch.jit.trace(model, x).save(str(a.out / "model.ts"))
        fmt, weights = "torchscript", "model.ts"
    with torch.no_grad():
        ref = model(torch.randn(4, len(leads), win["crop_samples"])).shape
    (a.out / "feature_names.json").write_text(json.dumps(leads))
    (a.out / "metrics.json").write_text(json.dumps(m, indent=1))
    files = {f: sha256(a.out / f) for f in (weights, "feature_names.json", "metrics.json")}
    manifest = {
        "model_id": a.model_id, "task": "ecg_deep", "status": "draft", "format": fmt, "weights": weights,
        "sha256": files[weights], "files_sha256": files,
        "input": {"kind": "tensor", "modality": "ecg", "fs_hz": win["fs_hz"], "leads": leads,
                  "tensor_shape": ["batch", len(leads), win["crop_samples"]], "record_samples": win["record_samples"],
                  "window_stride_samples": win["stride_samples"], "unit": "mV",
                  "normalization": {"type": "per_lead_zscore", **m["normalization"]}},
        "output": {"type": "multilabel", "classes": classes, "activation": "sigmoid",
                   "aggregation": "mean over sliding windows", "logits_shape": list(ref)},
        "code_commit": code_commit(), "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "data": {"dataset": "PTB-XL 1.0.3 (100 Hz)", "split": m["protocol"], "n": m["n"]},
        "headline": {"test_macro_auroc_selected": m["selected"]["test_macro_auroc"],
                     "ci95": m["selected"]["ci95"], "test_macro_auroc_ensemble": m["ensemble"]["test_macro_auroc"],
                     "beta_eva6": m["baseline_beta_eva6_macro_auroc"]},
        "arch": a.arch, "selected_seed": seed,
    }
    (a.out / "manifest.json").write_text(json.dumps(manifest, indent=1))
    print(f"exported {fmt} to {a.out} (sha256 {files[weights][:12]})", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
