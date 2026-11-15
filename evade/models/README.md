# Model artifacts: folder contract and registry

Every servable model is one folder, `<EVADE_MODEL_DIR>/<task>/`, wherever it was trained (the lab server, the CUNY SLURM cluster, a laptop). The API loads any folder that follows this contract, with no task-specific code in the loader. A new model, such as `spaceweather/`, needs only its own endpoint.

```
<EVADE_MODEL_DIR>/
  seizure/        manifest.json  model.joblib  feature_names.json  metrics.json
  ecg/            ...
  spaceweather/   ...
```

| File | Content |
|---|---|
| `manifest.json` | what the model is, how to feed it, what it returns, hashes (below) |
| weights | `model.joblib`, `model.onnx` or `model.pt` (TorchScript); the name is given by `manifest.weights` |
| `feature_names.json` | ordered feature list for `input.kind = features` (the column order the model expects) |
| `metrics.json` | evaluation results written by the training run (subject-wise protocol, baseline beside the model) |

## manifest.json

| Field | Required | Meaning |
|---|---|---|
| `schema_version` | yes (1) | contract version |
| `model_id` | yes | `EVA-<n>` from [MODEL_REGISTRY.md](../../MODEL_REGISTRY.md). A training run without `--model-id` writes `UNASSIGNED`: the folder is listed but not served until it is re-stamped |
| `task` | yes | `[a-z0-9_]`; should equal the folder name (the folder name is the key the API uses) |
| `status` | yes | `draft` (served only with `EVADE_ALLOW_DRAFT=1`, labelled draft) or `validated` |
| `format` | yes | `joblib`, `onnx` or `torchscript` |
| `weights` | yes | bare file name inside the folder (no paths) |
| `sha256` | yes | SHA-256 of the weights file, checked before anything is deserialized |
| `files` | written by export | `{file name: sha256}` for `feature_names.json` and `metrics.json`, also checked |
| `input` | yes | `kind` (`features` or `tensor`), `modality`, and as relevant `fs_hz`, `window_s`, `step_s`, `channels` or `leads`, `features` (the feature list file, for `features`), `tensor_shape` (for `tensor`), `normalization` (for example `per_user_robust`) |
| `output` | yes | `type` (optional) (`binary`, `multiclass`, `multilabel`, `regression`), `classes` (labels), `thresholds` (named operating points, for example `review` and `suspect`) and `operating_points` (target FPR, threshold, sensitivity) |
| `headline` | optional | short metric summary shown by `GET /v1/models` |
| `code_commit` | recommended | git commit of the training code (`EVADE_CODE_COMMIT`, a `COMMIT` file at the tree root, or `git rev-parse HEAD`) |
| `trained_at` | recommended | UTC ISO time |
| `data` | recommended | datasets, versions, subjects or splits used |
| `claim` | written by export | "outputs are flags for human review, not diagnoses" |

Required: `model_id`, `task`, `status`, `format`, `weights`, `sha256`, `input` (with `modality`), and `output`. A missing `code_commit`, `trained_at` or `data` is logged as a warning, not refused. Hand-written manifests are read leniently:
- `input.kind` is inferred: `tensor` if `tensor_shape` is given, otherwise `features`;
- `input.feature_names` or `input.feature_list` count as `input.features`, and without any reference the folder's `feature_names.json` is used;
- `output.labels` counts as `output.classes`, and `output.type` is optional;
- `commit` counts as `code_commit`, and `data_used` counts as `data`.

`write_manifest` always writes the canonical names.

Example (abridged):

```json
{
  "schema_version": 1, "model_id": "EVA-2", "task": "seizure", "status": "draft", "format": "joblib",
  "weights": "model.joblib", "sha256": "<64 hex>",
  "files": {"feature_names.json": "<64 hex>", "metrics.json": "<64 hex>"},
  "input": {"kind": "features", "modality": "eeg", "fs_hz": 250.0, "window_s": 4.0, "step_s": 2.0,
            "channels": ["FP1-F7", "..."], "features": "feature_names.json", "normalization": "per_user_robust"},
  "output": {"type": "binary", "classes": ["seizure_suspect"], "thresholds": {"review": 0.41, "suspect": 0.77},
             "operating_points": [{"target_fpr": 0.01, "threshold": 0.77, "sensitivity_subject_mean": 0.6}]},
  "headline": {"loso_pooled_auroc": 0.98}, "code_commit": "8404bb1", "trained_at": "2026-11-14T10:00:00+00:00",
  "data": {"datasets": ["CHB-MIT", "Siena (external test)"]}
}
```

## Producing a folder

- `python -m evade.training.train --task ecg --ptbxl <dir> --out <EVADE_MODEL_DIR> --model-id EVA-<n>` writes all four files. Every run writes `manifest.json` through `evade.models.export.write_manifest`. A new model (spaceweather, a deep EEG net) calls `write_manifest(out_dir, task=..., input_spec=..., output_spec=..., data=..., model_id=..., fmt=..., weights=...)` after saving its weights.
- To give a finished folder its registry ID, or to promote it after review, run `python -m evade.models.export --dir <folder> --model-id EVA-<n> [--status validated]`. This re-hashes every file.
- A retrain is a new `EVA-<n>`. Add the `MODEL_REGISTRY.md` row in the same commit.

## Loading (`registry.py`)

`ModelRegistry.from_env()` scans `EVADE_MODEL_DIR` at service start. Every folder with a `manifest.json` goes through these checks in order: the manifest validates, the draft gate passes, the runtime is installed, the hashes match, and only then is the model loaded. A folder that fails any check is listed under `unavailable` in `GET /v1/models` with the reason, and the service still starts.

| Format | Runtime | If the runtime is missing |
|---|---|---|
| joblib | `joblib` (in requirements) | listed unavailable |
| onnx | `onnxruntime` (optional, `pip install onnxruntime`), CPU provider | listed unavailable |
| torchscript | `torch` (optional), `torch.jit.load(map_location="cpu")` | listed unavailable |

**Beta folders from before this contract:** `seizure/` and `ecg/` with `model.joblib` and no `manifest.json` still load. The registry builds a manifest from MODEL_REGISTRY.md (EVA-2, EVA-6, draft) and marks them `legacy: true`. Without a reference hash they are served as `integrity: "unverified"` in `GET /v1/models`, with a warning in the log. Pinning the weights' SHA-256 with `EVADE_LEGACY_SHA256_SEIZURE` or `EVADE_LEGACY_SHA256_ECG` (64 hex characters, computed on the machine that built it) makes a mismatch refuse to load, but the folder stays `unverified`, because its `feature_names.json` has no hash. To verify it, call `write_manifest` on the folder on the machine that built it (as `train.py` now does), or retrain.

**Read once, then load.** Each file is read into memory once. Those bytes are hashed and compared with the manifest, and the same bytes are deserialized: joblib from a `BytesIO`, ONNX from bytes, TorchScript from a buffer. Nothing can swap the file between the check and the load. A loader is never called when a hash does not match. The loaded object must look like a model (`predict` or `predict_proba`, a dict of such estimators, an `onnxruntime.InferenceSession`, or a TorchScript module), otherwise the folder is unavailable with `unexpected_model_type`. That check runs after unpickling, so it is a sanity check, not a defence.

**Unavailable folders** are listed in `GET /v1/models` with a generic `reason_code` and `reason`: `manifest_invalid`, `model_id_unassigned`, `draft_not_allowed`, `runtime_missing`, `file_missing`, `sha256_mismatch`, `unexpected_model_type`, `load_failed`. The detail (exception text, file names) goes only to the server log.

**Safety:** joblib and pickle files execute code when loaded. Only put artifacts built by `evade.training` on machines we control into `EVADE_MODEL_DIR`, or artifacts fetched from our own R2 bucket. The sha256 check catches corruption, partial copies and swapped files. It does not protect against a compromised source, because the hash travels with the artifact. Prefer ONNX for models that leave our machines.

## Syncing with Cloudflare R2

R2 layout: `models/<task>/manifest.json` and the files it names, in a **separate models bucket**, so the API's data-bucket key cannot rewrite models.

| Machine | Variables | R2 token permission |
|---|---|---|
| servers (pull) | `CF_R2_MODELS_BUCKET`, `CF_R2_MODELS_ACCESS_KEY_ID`, `CF_R2_MODELS_SECRET_ACCESS_KEY` | Object Read only, models bucket |
| training machines (push) | `CF_R2_MODELS_BUCKET`, `CF_R2_MODELS_WRITE_ACCESS_KEY_ID`, `CF_R2_MODELS_WRITE_SECRET_ACCESS_KEY` | Object Read & Write, models bucket |

The write keys never go on a server. `pull` only ever sees a list/get view of the bucket. Without `CF_R2_MODELS_BUCKET`, the script refuses to run. Only with `EVADE_ALLOW_SHARED_MODELS_BUCKET=1` (for local experiments) does it fall back to `models/` in the data bucket (`CF_R2_BUCKET` with its read-write key), with a warning.

```
python scripts/fetch_models.py                    # pull every task into EVADE_MODEL_DIR (sha256 verified, atomic swap)
python scripts/fetch_models.py --task spaceweather
python scripts/fetch_models.py --push $EVADE_MODEL_DIR/spaceweather   # upload (files first, manifest last)
```

This needs `pip install boto3`, `CF_ACCOUNT_ID` and the variables above. A failed hash leaves the current folder untouched, and so does a failed folder swap, because the old folder is renamed back. A folder is downloaded again only when the remote manifest differs from the local one (compared as parsed JSON, so hand-written manifests are not re-downloaded every time) or a local hash no longer matches.

## Why there is no generic `POST /v1/predict/{model_id}`

It was left out on purpose, for two reasons:

1. **The served models need preprocessing that a manifest cannot express.** The seizure model needs the user's own EEG feature baseline (per-person robust z), 3-window smoothing and montage mapping. The ECG model needs band-pass filtering and per-lead features. A generic endpoint would either duplicate that logic or skip it and return wrong numbers.
2. **A path-parameter route under `/v1/predict/` would shadow** every `/v1/predict/<name>` route declared after it, including the spaceweather endpoint being added on another branch.

Each model therefore gets its own small endpoint, and that endpoint reads its thresholds and input spec from `STATE.models[task]["manifest"]`.
