"""Export the final deep seizure-suspect model to a servable folder (the EVAde model artifact contract).

  python -m evade.training.deep.seizure_export --run <final run dir> --out <export dir>/seizure [--model-id EVA-n]

Writes model.onnx (torch.onnx.export, dynamo=False, opset 17, dynamic batch; TorchScript model.ts if ONNX export
fails), feature_names.json (the 18 bipolar channels in order), metrics.json (from the run) and manifest.json with
the sha256 of every file, the input spec (montage, filter, sampling rate, window, per-person normalization) and the
output spec (sigmoid, smoothing, operating points from the CHB-MIT out-of-fold scores). If onnxruntime is installed,
the ONNX output is checked against torch on random inputs. Outputs are flags for human review, not diagnoses.
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from ...preprocessing import config
from .export import code_commit, sha256
from .seizure_models import build


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True, type=Path, help="final run dir with metrics.json and model/best.pt")
    p.add_argument("--arch", default="")
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--model-id", default="UNASSIGNED")
    p.add_argument("--parity-tol", type=float, default=1e-3)
    a = p.parse_args(argv)
    m = json.loads((a.run / "metrics.json").read_text())
    arch = a.arch or m["arch"]
    win = m["window"]
    T = int(round(win["fs_hz"] * win["window_s"]))
    channels = list(config.BIPOLAR_NAMES)
    model = build(arch, len(channels), 1)
    model.load_state_dict(torch.load(a.run / "model" / "best.pt", map_location="cpu", weights_only=True))
    model.eval()
    a.out.mkdir(parents=True, exist_ok=True)
    x = torch.zeros(1, len(channels), T)
    fmt, weights, note = None, None, None
    try:
        torch.onnx.export(model, x, a.out / "model.onnx", input_names=["x"], output_names=["logit"],
                          dynamic_axes={"x": {0: "batch"}, "logit": {0: "batch"}}, opset_version=17, dynamo=False)
        fmt, weights = "onnx", "model.onnx"
    except Exception as e:  # exporter or onnx package missing: fall back to TorchScript
        note = f"onnx export failed ({type(e).__name__}: {e}); TorchScript written instead"
        print(note, flush=True)
        with torch.no_grad():
            torch.jit.trace(model, x).save(str(a.out / "model.ts"))
        fmt, weights = "torchscript", "model.ts"
    parity = None
    xr = torch.randn(8, len(channels), T)
    with torch.no_grad():
        ref = model(xr).numpy()
    if fmt == "onnx":
        try:
            import onnxruntime as ort
            sess = ort.InferenceSession(str(a.out / "model.onnx"), providers=["CPUExecutionProvider"])
            got = sess.run(None, {"x": xr.numpy()})[0]
            diff = float(np.abs(got - ref).max())
            parity = {"max_abs_diff_logit": diff, "n": 8, "tol": a.parity_tol, "ok": diff <= a.parity_tol}
            if not parity["ok"]:
                print(f"WARNING: onnx/torch parity {diff:.2e} above {a.parity_tol}", flush=True)
        except ImportError:
            parity = {"skipped": "onnxruntime not installed"}
    (a.out / "feature_names.json").write_text(json.dumps(channels))
    (a.out / "metrics.json").write_text(json.dumps(m, indent=1))
    files = {f: sha256(a.out / f) for f in (weights, "feature_names.json", "metrics.json")}
    ops = m.get("operating_points_for_serving") or []
    ext = m.get("external_siena", {}).get("all_subjects", {}).get("at_chbmit_thresholds") or {}
    manifest = {
        "model_id": a.model_id, "task": "seizure_window", "status": "draft", "format": fmt, "weights": weights,
        "sha256": files[weights], "files_sha256": files,
        "input": {
            "kind": "tensor", "modality": "eeg", "dtype": "float32", "tensor_shape": ["batch", len(channels), T],
            "fs_hz": win["fs_hz"], "window_s": win["window_s"], "step_s": win["step_s"],
            "channels": channels, "channels_file": "feature_names.json",
            "montage": "18-channel bipolar double banana (anode minus cathode); a missing channel is set to 0 after normalization",
            "unit": "microvolt",
            "filter": {"bandpass_hz": list(config.EEG_BAND), "type": "Butterworth order 4, zero-phase (sosfiltfilt)",
                       "notch_hz": "mains (60 in the US data, 50 in Europe), Q 30, zero-phase",
                       "resample": "polyphase to fs_hz"},
            "clip_raw_uv": 2000.0,
            "normalization": {
                "type": "per_person_per_channel_robust",
                "formula": "x_norm = clip((x - median_c) / mad_c, -20, 20)",
                "median_c": "median of the person's samples on channel c",
                "mad_c": "max(1.4826 x median(|x - median_c|), 0.5 uV)",
                "statistics_from": "the person's own seizure-free baseline EEG (training: that person's non-ictal "
                                   "windows, up to 4,000 windows evenly spaced); the API computes them from the "
                                   "user's baseline recording, never from the window being scored",
                "clip": 20.0, "per_person_stats_in_model": False},
        },
        "output": {
            "type": "binary", "classes": ["seizure_suspect"], "activation": "sigmoid", "logits_shape": ["batch", 1],
            "post": "centered moving average of 3 consecutive window probabilities within a recording; the "
                    "thresholds apply to this smoothed probability (flag = smoothed probability > threshold)",
            "operating_points": [{"name": op["name"], "target_fpr": op["target_fpr"], "threshold": op["threshold"],
                                  "sensitivity_val_subject_mean": op["sensitivity_oof_subject_mean"],
                                  "event_sensitivity_val": op["event_sensitivity_oof"],
                                  "fp_windows_per_hour_val": op["fp_windows_per_hour_oof"],
                                  "false_alarms_per_hour_val": op["false_alarms_per_hour_oof"],
                                  "source": "CHB-MIT pooled out-of-fold scores (leave-one-subject-out)"}
                                 for op in ops],
            "meaning": "a flag asks a person to review the EEG; it is not a diagnosis"},
        "code_commit": code_commit(), "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "data": {"train": "CHB-MIT 1.0.0 (" + ", ".join(m.get("cases", [])) + ")",
                 "external_test": "Siena Scalp EEG 1.0.0", "thresholds_from": m.get("thresholds_from"),
                 "train_subjects": m.get("train", {}).get("train_subjects"),
                 "val_subjects": m.get("train", {}).get("val_subjects")},
        "headline": {"siena_auroc": ext.get("pooled_auroc"), "siena_auroc_ci95": ext.get("pooled_auroc_ci95"),
                     "siena_ap": ext.get("pooled_ap"), "beta_eva2": m.get("baseline_beta_eva2")},
        "arch": arch, "n_parameters": int(sum(p.numel() for p in model.parameters())),
        "onnx_parity": parity, "export_note": note,
    }
    (a.out / "manifest.json").write_text(json.dumps(manifest, indent=1, default=float))
    # sha256sum -c compatible list of every file, the manifest included
    names = sorted(list(files) + ["manifest.json"])
    (a.out / "SHA256SUMS").write_text("".join(f"{sha256(a.out / f)}  {f}\n" for f in names))
    print(f"exported {fmt} to {a.out} (sha256 {files[weights][:12]}, {manifest['n_parameters']} parameters)",
          flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
