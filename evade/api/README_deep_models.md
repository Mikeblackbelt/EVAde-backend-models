# Deep model endpoints (placeholder until trained)

Three endpoints serve the deep models: `ecg_deep`, `sleep_deep` and `seizure_deep`. Each one works **today**, before any weights exist. Until a model folder appears, an endpoint returns **placeholder** outputs that follow the exact schema of the real ones, so the frontend and backend teams can integrate now. When a valid model folder appears, the same endpoint answers from the model with no code change. Outputs are flags for human review, never diagnoses.

Code: [deep_models.py](deep_models.py) holds the router and request limits. [../models/deep.py](../models/deep.py) is the loader, and [../models/deep_tasks.py](../models/deep_tasks.py) holds the preprocessing, post-processing and mock outputs.

## Endpoints

All of them need the same auth header as every other endpoint (see [README.md](README.md)), and every response uses the standard envelope (`request_id`, `user_id`, `server_time`, `api_version`, `model_ids`, `pipeline_version`, `data`, `warnings`).

| Endpoint | Input | Output (`data`) |
|---|---|---|
| `POST /v1/predict/ecg-deep` | 12-lead ECG, 10 s (9.5-10.5 s accepted), any `fs_hz` from 50 to 1000, resampled to 100 Hz | per-class probabilities for NORM, MI, STTC, CD and HYP, the per-class thresholds, and `top_classes` (classes at or above their threshold, highest first) |
| `POST /v1/predict/sleep-deep` | Fpz-Cz and Pz-Oz at 100 Hz, 1 to 20 consecutive 30 s epochs | the stage of each epoch (W, N1, N2, N3, REM) with probabilities; a summary with total sleep time, efficiency, onset latency and minutes per stage |
| `POST /v1/predict/seizure-deep` | 18-channel bipolar EEG (double banana), 4-30 s, `fs_hz` from 100 to 1000, `mains_hz` 50 or 60 | `status`; per-window probability and level for 4 s windows at a 2 s step; `max_probability`, `flag` and thresholds |
| `GET /v1/models` | none | now also includes `deep_models`: one row per deep task with `state` (`mock`, `available` or `unavailable`), `reason_code` and `reason` |
| `POST /v1/models/reload` | empty body (send `Content-Length: 0`) | rescans the deep model folders and returns the new `deep_models` rows. Other models still load only at service start |

### Fields the frontend should check

| Field | Mock (no weights yet) | Model |
|---|---|---|
| `data.source` | `"mock"` | `"model"` |
| `data.model_id` | `null` | `"EVA-<n>"` |
| `data.model_state` | `"mock"` (no folder) or `"unavailable"` (folder present but refused) | `"available"` |
| `data.model_status` | `null` | `"draft"` or `"validated"` |
| `model_ids` (envelope) | `[]` | `["EVA-<n>"]` |
| `warnings` | contains `"placeholder output: model not trained yet; not a prediction"` | `"draft model output"` for a draft model |

**Rule for the UI:** if `data.source != "model"`, show the result as a placeholder (for example greyed out, labelled "demo data") and never as a health finding. Mock outputs are deterministic: the same input always gives the same output. They are seeded from a hash of the samples, so a demo looks stable.

### What feeds the risk score

- **Mock outputs never feed anything.** They never reach the risk engine, flags, overrides, the latest comparisons or baselines. `feed_shadow_risk` refuses any output whose `source` is not `"model"`, and the tests check this.
- **Real `seizure_deep` outputs** enter the neuro domain as a **shadow** input (metric `seizure_deep_probability`), the same way the beta seizure model does (score 0.40 for `review` and 0.70 for `suspect_seizure`). A shadow input raises R, but it cannot trip the abnormal-component flag or the extreme-domain MDTR cap on its own. A request with `source` set to `simulated` or `replay` is labelled `synthetic`, which makes the risk result provisional. `override_candidate` is information for a human. The API never raises override O4 from it.
- **Real `ecg_deep` and `sleep_deep` outputs** do not feed the risk score yet, because the beta has no precedent to follow (the beta ECG model does not feed risk either). Mapping total sleep time into B1, or a superclass into the cardiac domain, is a later decision.

## Requests and responses

### ECG

```json
POST /v1/predict/ecg-deep
{"user_id": "<uuid>", "fs_hz": 500, "leads": ["I","II","III","aVR","aVL","aVF","V1","V2","V3","V4","V5","V6"],
 "samples": [[...5000 values in mV...], ... 12 rows]}
```

Leads can come in any order, and their names are not case-sensitive (`AVR` or `aVR`). Optional fields: `t0_utc` (never in the future) and `source` (`device`, `manual`, `simulated` or `replay`).

```json
{"request_id": "...", "user_id": "<uuid>", "model_ids": [], "warnings": ["placeholder output: model not trained yet; not a prediction"],
 "data": {"task": "ecg_deep", "source": "mock", "model_id": null, "model_status": null, "model_state": "mock",
          "note": "for human review, not a diagnosis",
          "probabilities": {"NORM": 0.2484, "MI": 0.8128, "STTC": 0.2126, "CD": 0.0357, "HYP": 0.1089},
          "thresholds": {"NORM": 0.5, "MI": 0.5, "STTC": 0.5, "CD": 0.5, "HYP": 0.5}, "threshold_source": "default",
          "top_classes": [{"class": "MI", "probability": 0.8128}], "n_crops": 0,
          "input": {"fs_hz": 500.0, "n_samples": 5000, "leads": ["I", "II", "..."], "model_fs_hz": 100.0}}}
```

The probabilities are multilabel (one sigmoid per class), so they do not sum to 1. In model mode, `thresholds` come from the manifest (`threshold_source: "manifest"`): either `output.operating_points` as `{class: t}` or `{class: {"threshold": t}}`, or `output.thresholds`. If the manifest gives none, they are 0.5, labelled `threshold_source: "default"`. That is not an operating point, so the UI should not treat `top_classes` as flags. `export.py` writes no operating points yet. `n_crops` is the number of 2.5 s crops averaged (7 for the PTB-XL export). The record is resampled to 100 Hz. The central 10 s are used, and a short record is padded with each lead's training mean.

### Sleep

```json
POST /v1/predict/sleep-deep
{"user_id": "<uuid>", "fs_hz": 100, "channels": ["Fpz-Cz", "Pz-Oz"], "samples": [[...n x 3000...], [...n x 3000...]]}
```

```json
{"data": {"task": "sleep_deep", "source": "mock", "model_id": null, "model_status": null, "model_state": "mock", "note": "...",
          "epochs": [{"index": 0, "t_start_s": 0.0, "stage": "W",
                      "probabilities": {"W": 0.8693, "N1": 0.0797, "N2": 0.0305, "N3": 0.0083, "REM": 0.0122}},
                     {"index": 1, "t_start_s": 30.0, "stage": "N2",
                      "probabilities": {"W": 0.1476, "N1": 0.0133, "N2": 0.7734, "N3": 0.0485, "REM": 0.0172}}, "..."],
          "summary": {"n_epochs": 4, "recording_min": 2.0, "total_sleep_time_min": 1.5, "sleep_efficiency": 0.75,
                      "sleep_onset_latency_min": 0.5,
                      "minutes_per_stage": {"W": 0.5, "N1": 0.0, "N2": 1.5, "N3": 0.0, "REM": 0.0},
                      "scope": "submitted epochs only; send consecutive chunks and combine them for a night"},
          "input": {"fs_hz": 100.0, "channels": ["Fpz-Cz", "Pz-Oz"], "epoch_s": 30.0}}}
```

Each epoch's probabilities sum to 1. **The 1 MB body cap** allows about 20 epochs (10 minutes) per request, so send a night in consecutive chunks and add up the summaries. The model's per-night z-score then uses the statistics of each chunk, not of the whole night, which is a known limitation until a compact upload format exists.

### Seizure

```json
POST /v1/predict/seizure-deep
{"user_id": "<uuid>", "fs_hz": 256, "mains_hz": 60,
 "channels": ["FP1-F7","F7-T7","T7-P7","P7-O1","FP1-F3","F3-C3","C3-P3","P3-O1","FP2-F4","F4-C4","C4-P4","P4-O2",
              "FP2-F8","F8-T8","T8-P8","P8-O2","FZ-CZ","CZ-PZ"],
 "samples": [[...], ... 18 rows]}
```

Channel names follow the beta rules: the old names T3, T4, T5 and T6 are mapped to the new ones, and suffixes such as `-0` are ignored.

```json
{"data": {"task": "seizure_deep", "source": "mock", "model_id": null, "model_status": null, "model_state": "mock", "note": "...",
          "status": "ok", "quality_index": 1.0, "needs_baseline": false,
          "windows": [{"t_start_s": 0.0, "probability": 0.0114, "level": "none"},
                      {"t_start_s": 2.0, "probability": 0.0232, "level": "none"}, "..."],
          "max_probability": 0.0232, "flag": "none",
          "thresholds": {"review_5pct_fpr": 0.5, "suspect_1pct_fpr": 0.8}, "override_candidate": false,
          "baseline_hint": null,
          "input": {"fs_hz": 250.0, "channels": ["FP1-F7", "..."], "window_s": 4.0, "step_s": 2.0}}}
```

Before the user has an EEG baseline:

```json
{"data": {"task": "seizure_deep", "source": "mock", "model_id": null, "...": "...",
          "status": "needs_baseline", "quality_index": 1.0, "needs_baseline": true, "windows": [],
          "max_probability": null, "flag": null, "thresholds": {"review_5pct_fpr": 0.5, "suspect_1pct_fpr": 0.8},
          "override_candidate": false,
          "baseline_hint": "submit at least 30 4 s windows of EEG to /v1/baseline/eeg first", "input": {"...": "..."}}}
```

`GET /v1/models` lists each deep task under `deep_models`, for example:

```json
{"task": "ecg_deep", "state": "mock", "source": "mock", "model_id": null, "status": null, "format": null, "runtime": null,
 "reason_code": null, "reason": "no model folder yet: placeholder outputs", "sha256_verified": false,
 "input": null, "output": null, "headline": null, "code_commit": null, "trained_at": null}
```

`status` can take three values:
- `ok`;
- `needs_baseline`: the user has not yet sent at least 30 four-second EEG windows to `/v1/baseline/eeg`. This is the same gate as `/v1/predict/eeg`. `windows` is then `[]`, `flag` and `max_probability` are `null`, and `baseline_hint` says what to do;
- `low_quality`: the mean signal quality is under 0.5.

The window levels are `none`, `review` (above the 5% false-positive-rate threshold) and `suspect_seizure` (above the 1% threshold). In model mode, both thresholds come from `output.operating_points` in the manifest. In mock mode they are placeholders (0.5 and 0.8).

## Errors and limits

Validation errors return 422 with `{"error": {"code", "message", "field"}}`. The message describes the rule, never the submitted values. The codes are `bad_shape` (wrong channels, length or epoch count), `nan_in_signal`, `future_timestamp` and `validation_error` (for example too many leads, channels or samples). A body over 1 MB returns 413. An unknown `user_id` returns 404.

| Endpoint | Limits |
|---|---|
| ecg-deep | exactly 12 leads; at most 10,500 samples per lead; 9.5-10.5 s |
| sleep-deep | exactly 2 channels; 1-20 epochs of 3,000 samples; fs 100 Hz |
| seizure-deep | exactly 18 channels; 4-30 s; at most 30,000 samples per channel |

Request bodies and signals are never logged. The loader logs only folder names and reason codes.

## Making an endpoint go live: dropping in a model

1. Put the export into `<EVADE_MODEL_DIR>/<task>/`, where `<task>` is `ecg_deep`, `sleep_deep` or `seizure_deep`:
   ```
   <EVADE_MODEL_DIR>/
     ecg_deep/      manifest.json  model.onnx | model.ts | model.pt  [feature_names.json  metrics.json]
     sleep_deep/    ...
     seizure_deep/  ...
   ```
2. `manifest.json` follows [the model contract](../models/README.md), with these additions for deep models:
   - `format` is `onnx`, `torchscript`, `torch_state_dict`, or `torch`. With `torch`, a `model.pt` is auto-detected: a TorchScript archive is loaded with `torch.jit.load`, and anything else is loaded as a state_dict.
   - A state_dict also needs `"arch": "<name>"`. The architectures known are those registered with `evade.models.deep.register_arch`, plus `evade.training.deep.models.build` once that branch is merged (`xresnet1d101`, `resnet1d_wang`, and others).
   - `sha256` of the weights is required. Side files are hashed under `files` or `files_sha256`; the latter is what `evade/training/deep/export.py` writes.
   - `model_id` must be an `EVA-<n>` with a `MODEL_REGISTRY.md` row. A training export writes `UNASSIGNED`, which stays unavailable until you re-stamp it: `python -m evade.models.export --dir <folder> --model-id EVA-<n>`, or edit the field.
   - `input.tensor_shape`, `output.classes`, and the task fields:

   | Task | Manifest fields the endpoint reads |
   |---|---|
   | ecg_deep | `input.fs_hz` (100), `input.leads` (any order of the 12), `tensor_shape` `["batch", 12, crop]`, `record_samples` (1000), `window_stride_samples`, `normalization` `{"type": "per_lead_zscore", "mean": [12], "std": [12]}` or `"none"`; `output.classes` (the 5 superclasses), `activation` `sigmoid`; optional `output.thresholds` per class. The deep export's manifest already fits. |
   | sleep_deep | `input.fs_hz` (100), `channels` `["Fpz-Cz", "Pz-Oz"]`, `tensor_shape` `["batch", 2, 3000]` (per epoch) or `["batch", L, 2, 3000]` (sequence of L epochs; a name instead of a number passes the whole sequence), `normalization` `per_night_zscore` or `none`; `output.classes` (W, N1, N2, N3, REM in any order; R counts as REM), `activation` `softmax` |
   | seizure_deep | `input.fs_hz` (250), `window_s` (4), `step_s` (2), `channels` (the 18 bipolar pairs), `tensor_shape` `["batch", 18, 1000]`, `normalization` `none`, `per_window_zscore` or `per_window_robust`, `smooth_windows` (3); `output.operating_points` with thresholds at `target_fpr` 0.05 and 0.01 (or names `review` and `suspect_seizure`), `activation` `sigmoid` |
3. Install the runtime: `pip install onnxruntime` for ONNX, or `pip install torch --index-url https://download.pytorch.org/whl/cpu` for TorchScript and state_dict.
4. Set `EVADE_ALLOW_DRAFT=1` while the manifest says `"status": "draft"`.
5. Restart the service, or call `POST /v1/models/reload`. Then check that `GET /v1/models` shows the task under `deep_models` with `"state": "available"`, and that the endpoint returns `"source": "model"`.

If the state is `unavailable`, `reason_code` gives the reason, and the server log has the detail. The reason codes are those of the model registry (`manifest_invalid`, `model_id_unassigned`, `draft_not_allowed`, `runtime_missing`, `file_missing`, `sha256_mismatch`, `unexpected_model_type`, `load_failed`) plus `arch_unknown` and `spec_unsupported`. While a model is unavailable, the endpoint keeps answering with placeholders.

**Loading is safe by construction.** Each file is read once, its sha256 is checked, and only those same bytes are deserialized. State_dicts load with `torch.load(weights_only=True)`, which loads tensors and never runs code. TorchScript loads with `torch.jit.load` from a buffer, and ONNX with onnxruntime from bytes. A file whose hash does not match is never deserialized. After loading, one batch of zeros checks that the output shape matches the manifest.

**Known gap (seizure):** the training plan normalizes the seizure net's signal per person, using baseline windows (per-channel median and MAD). The API does not store per-channel signal baselines yet, so a manifest that asks for that rule is listed `spec_unsupported` and the endpoint stays on placeholders. Closing the gap needs a per-channel signal baseline, kept by `/v1/baseline/eeg`.
