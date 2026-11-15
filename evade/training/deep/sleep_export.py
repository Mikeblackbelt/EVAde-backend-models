"""Export the deep sleep-staging model to a servable folder (the EVAde model artifact contract).

  python -m evade.training.deep.sleep_export --run <run dir> --out <run dir>/export/sleep [--model-id EVA-n]

<run> holds final/ (the exported weights, final/best.pt), kfold/metrics.json (the performance estimate) and
optionally beta5/metrics.json (the beta-comparable track). Writes model.onnx (dynamo=False; TorchScript model.ts if
the ONNX exporter fails), feature_names.json (channels), metrics.json and manifest.json with the sha256 of every file,
the input spec (sequence of epochs, normalization rule), the output spec (classes, softmax, operating points).
If onnxruntime is installed, ONNX parity with PyTorch is checked at a batch size and sequence length different from
the export trace, and the result goes in the manifest.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from .export import code_commit, sha256
from .sleep_models import build


def parity(onnx_path: Path, model, c: int, t: int) -> dict:
    try:
        import onnxruntime as ort
    except Exception as e:      # noqa: BLE001
        return {"checked": False, "reason": f"onnxruntime unavailable ({type(e).__name__})"}
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    worst = 0.0
    for b, l in ((1, 20), (3, 7), (2, 33)):
        x = torch.randn(b, l, c, t)
        with torch.no_grad():
            ref = model(x).numpy()
        got = sess.run(None, {"x": x.numpy()})[0]
        if got.shape != ref.shape:
            return {"checked": True, "ok": False, "reason": f"shape {got.shape} != {ref.shape} at batch {b} seq {l}"}
        worst = max(worst, float(np.abs(got - ref).max()))
    return {"checked": True, "ok": worst < 1e-3, "max_abs_logit_diff": worst, "shapes_tested": [[1, 20], [3, 7], [2, 33]]}


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--model-id", default="UNASSIGNED")
    a = p.parse_args(argv)
    fin = json.loads((a.run / "final" / "metrics.json").read_text())
    kf = json.loads((a.run / "kfold" / "metrics.json").read_text())
    b5p = a.run / "beta5" / "metrics.json"
    b5 = json.loads(b5p.read_text()) if b5p.exists() else None
    cfg = fin["config"]
    if cfg["channels"] != kf["config"]["channels"] or cfg["arch"] != kf["config"]["arch"]:
        raise SystemExit("final and kfold runs differ in channels or arch; the kfold estimate would not apply")
    c, t, L = len(cfg["channels"]), cfg["epoch_samples"], cfg["seq_len"]
    model = build(cfg["arch"], c, len(cfg["classes"]))
    model.load_state_dict(torch.load(a.run / "final" / "final" / "best.pt", map_location="cpu", weights_only=True))
    model.eval()
    a.out.mkdir(parents=True, exist_ok=True)
    x = torch.zeros(1, L, c, t)       # batch 1 trace, as the GRU exporter recommends; parity checks other shapes
    fmt, weights, why = None, None, None
    try:
        torch.onnx.export(model, x, a.out / "model.onnx", input_names=["x"], output_names=["logits"],
                          dynamic_axes={"x": {0: "batch", 1: "seq"}, "logits": {0: "batch", 1: "seq"}},
                          opset_version=17, dynamo=False)
        fmt, weights = "onnx", "model.onnx"
        par = parity(a.out / "model.onnx", model, c, t)
    except Exception as e:  # exporter or onnx package missing: fall back to TorchScript
        why = f"{type(e).__name__}: {e}"
        print(f"onnx export unavailable ({why}); writing TorchScript", flush=True)
        with torch.no_grad():
            torch.jit.trace(model, x).save(str(a.out / "model.ts"))
        fmt, weights = "torchscript", "model.ts"
        par = {"checked": False, "reason": "TorchScript fallback"}
    with torch.no_grad():
        ref = list(model(torch.randn(1, L, c, t)).shape)
    metrics = {"kfold": kf, "beta5": b5, "final": {k: fin.get(k) for k in ("protocol", "units", "n", "config")}}
    (a.out / "feature_names.json").write_text(json.dumps(cfg["channels"]))
    (a.out / "metrics.json").write_text(json.dumps(metrics, indent=1))
    files = {f: sha256(a.out / f) for f in (weights, "feature_names.json", "metrics.json")}
    oof = kf.get("out_of_fold", {})
    head = {"estimate_from": "subject-wise k-fold cross-validation (out-of-fold); the exported model is the "
                             "all-subject 'final' model trained with the same recipe",
            "kfold_pooled_kappa": oof.get("pooled", {}).get("kappa"), "kfold_kappa_ci95": oof.get("ci95", {}).get("kappa"),
            "kfold_macro_f1": oof.get("pooled", {}).get("macro_f1"),
            "kfold_subject_mean_kappa": oof.get("subject_mean_kappa"),
            "beta5_subject_mean_kappa": (b5 or {}).get("out_of_fold", {}).get("subject_mean_kappa"),
            "beta_baseline": kf.get("baseline_beta")}
    manifest = {
        "model_id": a.model_id, "task": "sleep_staging", "status": "draft", "format": fmt, "weights": weights,
        "sha256": files[weights], "files_sha256": files, "onnx_fallback_reason": why, "onnx_parity": par,
        "input": {"kind": "tensor", "modality": "eeg", "fs_hz": cfg["fs_hz"], "channels": cfg["channels"],
                  "edf_labels": [f"EEG {ch}" for ch in cfg["channels"]],
                  "tensor_shape": ["batch", "seq", c, t], "dtype": "float32", "epoch_s": 30,
                  "seq_len_trained": L, "seq_len_any": True, "unit": "uV before normalization",
                  "filtering": "none (raw Sleep-EDFx 100 Hz signals)",
                  "normalization": cfg["normalization"] | {
                      "how_api_computes": "per channel over all 30 s epochs of the night being staged: "
                                          "x = clip((x - median) / max(1.4826 * MAD, 0.1), -20, 20)"},
                  "sequence": "consecutive 30 s epochs of one night, in time order; never mix nights"},
        "output": {"type": "multiclass_sequence", "classes": cfg["classes"], "activation": "softmax over the last axis",
                   "logits_shape": ["batch", "seq", len(cfg["classes"])], "example_logits_shape": ref,
                   "aggregation": f"windows of {L} epochs with stride {cfg['eval_stride']}; softmax averaged per epoch",
                   "operating_points": kf.get("operating_points", []),
                   "use": "sleep-stage estimates and total sleep time flagged for human review; not a diagnosis"},
        "code_commit": code_commit(), "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "data": {"dataset": "Sleep-EDFx 1.0.0 sleep-cassette (PhysioNet)", "n_kfold": kf.get("n"),
                 "splits_sha256": kf.get("protocol", {}).get("splits_sha256"), "k": kf.get("protocol", {}).get("k"),
                 "final_val_subjects": (fin.get("units") or [{}])[0].get("val_subjects")},
        "headline": head, "arch": cfg["arch"], "smoke": bool(kf.get("smoke")),
    }
    (a.out / "manifest.json").write_text(json.dumps(manifest, indent=1))
    print(f"exported {fmt} to {a.out} (sha256 {files[weights][:12]}); parity {par}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
