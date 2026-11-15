# Deep ECG serving adapter (`evade/models/onnx_ecg.py`)

`EcgDeepModel` loads the model folder that `evade.training.deep.export` writes and scores one ECG record on CPU. It is a plain Python module: it imports numpy at import time, scipy only when resampling, and onnxruntime (or torch, for a TorchScript folder) only when a model is loaded. It has no FastAPI dependency. The API wraps it, as described under "How the API calls it" below.

**Status: draft.** The outputs are probabilities for review by a clinician, not diagnoses. The model was trained on 12-lead resting hospital ECGs (PTB-XL), not on wearable or in-flight recordings. Until the model has an `EVA-n` row in `MODEL_REGISTRY.md` with status `active`, every result carries a `draft_model` note.

## Model folder (the export contract)

```
ecg_deep/
  manifest.json        model_id, status, format (onnx | torchscript), weights, sha256, files_sha256,
                       input {fs_hz, leads, tensor_shape [batch, leads, crop], record_samples,
                              window_stride_samples, unit "mV", normalization {type per_lead_zscore, mean[], std[]}},
                       output {type multilabel, classes[], activation sigmoid, aggregation, [operating_points]}
  model.onnx | model.ts
  feature_names.json   the lead list (must equal input.leads)
  metrics.json         the training run's metrics (bootstrap CIs, beta baseline number)
```

## What loading does

1. Reads `manifest.json`. If `expected_manifest_sha256` is given (for example, the value pinned in the model registry), the manifest bytes must match it.
2. For every entry in `manifest.files_sha256`, it reads the file **once**, hashes those bytes, and keeps them. The model is built from the same bytes (`onnxruntime.InferenceSession(bytes)` or `torch.jit.load(BytesIO(bytes))`), so nothing can change between the check and the load. Loading is refused (`ModelIntegrityError`) when:
   - a listed file is missing, or its sha256 differs;
   - a listed name is unsafe (it contains `/`, `\`, `:` or `..`);
   - `weights` is not in the list, or `manifest.sha256` does not match the weights;
   - `feature_names.json` disagrees with `input.leads`;
   - the normalization is not a finite per-lead z-score with one mean and one positive std per lead;
   - the output is not multilabel sigmoid, or the format is unknown.
3. A missing runtime (no onnxruntime for an ONNX folder, or no torch for TorchScript) raises `ModelLoadError`. ONNX runs on `CPUExecutionProvider`.

The manifest cannot hash itself, so the sha256 check proves the folder is self-consistent, not that it is the folder you meant to deploy. Pin `manifest_sha256` (from `model.describe()`) in the registry or the deployment config, and pass it as `expected_manifest_sha256`.

## What `predict` does

```python
from evade.models.onnx_ecg import EcgDeepModel, EcgInputError

model = EcgDeepModel("/models/ecg_deep", expected_manifest_sha256=PINNED, intra_op_threads=2)
out = model.predict(ecg, fs_hz=500.0, leads=["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"])
out.to_dict()
```

`ecg` is an array of shape `(leads, samples)` in millivolts. Pass `unit="uV"` or `unit="V"` to scale. The steps follow `evade.training.deep.train`:

| Step | Rule |
|---|---|
| Values | NaN or inf is an error. Nothing is filled in silently. |
| Leads | With `leads=None`, the row count must equal `len(input.leads)`, and the order is assumed (note `lead_order_assumed`). With names, they are matched case-insensitively (`aVR` = `AVR` = `avr`, and `DI`/`DII`/`DIII` = `I`/`II`/`III`). Rows are reordered to the model's order (note `leads_reordered`). Extra leads are dropped (note `extra_leads_ignored`). A missing or duplicate lead, or a name count that differs from the row count, is an error. |
| Quality | These checks run on the input before resampling: flat leads (SD below 5 µV), amplitude above 10 mV (usually wrong units), and clipping (more than 2% of samples exactly at a lead's max or min). Each gives a `warning` note; none of them blocks the prediction. |
| Bounds | `fs_hz` outside 50-1000 Hz, a record longer than 60 s, or more than 64 rows is an error, checked before any copy or scan of the samples. |
| Rate | If `fs_hz` differs from `input.fs_hz` (100 Hz), the record is resampled with `scipy.signal.resample_poly` (note `resampled`). The polyphase up and down factors must both be at most 100, or the rate is refused. Upsampling also adds a warning. |
| Length | The model expects `record_samples` (10 s). The default `length_policy="pad_crop"` handles other lengths as follows. A longer record is cropped to the central 10 s (note `cropped`); it is first cut to the central 10 s plus 1 s each side before resampling, so the work does not grow with the input length. A shorter one is padded at the end with a flat line at each lead's training mean (warning `padded`). Anything shorter than one window (2.5 s) is an error. `length_policy="valid"` scores a short record on the windows inside the real signal only (warning `short_record`). `"strict"` makes any other length an error. |
| Normalization | Per lead, `(x - mean) / std`, using the train-fold statistics from the manifest. |
| Windows | Windows of `crop` samples (2.5 s), with a stride of `window_stride_samples` (1.25 s) over the 10 s, give 7 windows. All of them go through the model in one batch. |
| Output | The sigmoid of each window's logits, averaged over the windows, gives one probability per class. |

`out.to_dict()`:

```json
{
  "probabilities": {"NORM": 0.81, "MI": 0.07, "STTC": 0.10, "CD": 0.05, "HYP": 0.04},
  "flags": null,
  "operating_points": null,
  "n_windows": 7,
  "quality": {"ok": true, "notes": [{"code": "resampled", "severity": "info", "detail": "resampled from 500 Hz to 100 Hz (polyphase 1/5)"}]},
  "input": {"fs_hz_in": 500.0, "fs_hz_model": 100.0, "samples_in": 5000, "samples_after_resample": 1000,
            "seconds_in": 10.0, "lead_order": "given", "unit_in": "mV", "length_policy": "pad_crop",
            "padded_samples": 0, "cropped_samples": 0},
  "model": {"model_id": "EVA-n", "status": "draft", "task": "ecg_deep", "format": "onnx", "weights_sha256": "..."},
  "disclaimer": "Probabilities from a research model trained on hospital ECGs (PTB-XL), for review by a clinician. Not a diagnosis."
}
```

(The numbers above are illustrative, not model results.)

**Flags.** `flags` stays `null` unless the manifest has `output.operating_points`, given either as `{class: threshold}` or as `{class: {"threshold": t, ...}}`. A flag is then `probability >= threshold`. The adapter never invents a 0.5 threshold. `export.py` at commit a036ba8 writes no operating points, so for now every result is probabilities only, with the note `no_operating_points`.

Errors and notes never contain waveform values, only counts, lead names and durations.

## How the API calls it

The wiring belongs to a later branch; `evade/api/app.py` is not changed here. **The actual API wiring is `POST /v1/predict/ecg-deep` on branch `api-model-stubs`** (router in `evade/api/deep_models.py`, documented in `evade/api/README_deep_models.md`). It already returns placeholder outputs in the final schema until an `ecg_deep` model folder appears. When the branches merge, fold this adapter's preprocessing into that router rather than adding a second ECG path: the sha256-verified loading, lead matching and reordering, unit scaling, the input bounds (50-1000 Hz, up/down factors of at most 100, 60 s, 64 rows), polyphase resampling, the pad/crop length policies, the quality notes, and the train-fold z-score with the 7-window sigmoid mean. Keep the router's request schema and its stricter 9.5-10.5 s window. Two names need reconciling: the router reads per-class `output.thresholds`, while this adapter reads `output.operating_points`. The plan below predates that branch and is kept for reference.

- **Startup.** If `EVADE_MODEL_DIR/ecg_deep/manifest.json` exists, build one `EcgDeepModel` and keep it in `STATE.models["ecg_deep"]`. A `ModelIntegrityError` or `ModelLoadError` is logged by its message only; the API then starts without the model, and `/health` lists it as not loaded. The object is read-only after loading, so request threads can share it. Set `intra_op_threads` to suit the host.
- **`POST /v1/predict/ecg`.** This uses the existing `EcgRequest.window`, which carries `channels`, `samples`, `fs_hz` and `unit`. The call is:

  ```python
  try:
      out = STATE.models["ecg_deep"].predict(np.asarray(w.samples, np.float32), w.fs_hz, leads=w.channels, unit=w.unit)
  except EcgInputError as e:
      ...  # return the existing error envelope: 422, code "bad_shape" or "bad_leads", message str(e)
  data["deep_superclass"] = out.to_dict()
  ids.append(out.model["model_id"])
  warnings += [n.code for n in out.notes if n.severity == "warning"]
  ```

  The deep result is added next to the existing rhythm backstop and the per-person baseline comparison. It does not replace them. If a window has fewer than the required leads (a single-lead wearable, say), the request still succeeds without `deep_superclass`, with a warning. A single-lead model (`train.py --leads I`) exports its own folder and is loaded the same way.
- **`GET /v1/models`.** Add `model.describe()`: the id, status, format, sha256 values, input spec and the training headline (macro AUROC with its patient-bootstrap CI, next to the beta EVA-6 baseline of 0.867).
- **Requests.** The 1 MB body limit fits 12 x 5000 samples at 500 Hz as JSON, but only just; check it when wiring.

## Tests

`tests/test_onnx_ecg.py` builds a tiny `resnet1d_wang` with random weights and BatchNorm statistics, saves it in a fake run folder, and exports it with the real `export.py`. It runs once for ONNX and once with the TorchScript fallback forced. The tests check:

- torch (`train.predict`) against onnxruntime parity, within 1e-4, for `resnet1d_wang` and for a one-block-per-stage `XResNet1d` (the same ops as the served `xresnet1d101`);
- TorchScript parity;
- refusal on a sha256 mismatch for each listed file, a missing file, an unsafe name, or a pinned manifest mismatch;
- the wrong-lead-count, missing-lead and duplicate-lead errors, aliases and reordering;
- resampling from 500 Hz, and that cropping a long record before resampling gives the same model input;
- the input bounds: 49.9 and 1000.1 Hz refused, 50 and 1000 Hz accepted, a rate needing a factor above 100 refused, 60 s accepted and one sample more refused, more than 64 rows refused;
- the pad, crop, valid and strict length policies;
- NaN refusal and the quality notes, unit scaling, and operating-point flags;
- that importing the module pulls in no fastapi, torch or onnxruntime.

Run them with:

```
python -m pytest -q tests/test_onnx_ecg.py
```

They need numpy, scipy, scikit-learn, torch (CPU), onnx and onnxruntime. Without them the tests are skipped. `tests/test_deep_ecg_pipeline.py` runs the whole chain on CPU with synthetic data: a fake PTB-XL WFDB folder, `ptbxl_pack`, a 1-epoch `train`, `export`, and then this adapter.
