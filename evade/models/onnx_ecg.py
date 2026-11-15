"""CPU serving adapter for the deep ECG model folder written by evade.training.deep.export.

    from evade.models.onnx_ecg import EcgDeepModel
    model = EcgDeepModel("<export dir>/ecg_deep")
    out = model.predict(ecg, fs_hz=500.0, leads=["I", "II", ..., "V6"])    # ecg: (leads, samples), millivolts
    out.to_dict()   # per-class probabilities, optional flags, quality notes, model id and status

Loading refuses the folder unless every file listed in manifest.files_sha256 matches its sha256. Each file is read
once, hashed, and the model is built from those same bytes (onnxruntime on CPUExecutionProvider, or TorchScript
when the manifest format is "torchscript"). Preprocessing follows the training code (evade.training.deep.train):
per-lead (x - mean) / std from the train folds, sliding windows of crop_samples with stride window_stride_samples
over record_samples, sigmoid per window, mean over windows.

Input bounds: fs_hz must be 50-1000 Hz, the polyphase up/down factors at most 100, the record at most 60 s and 64
rows. A longer-than-needed record is cropped to the central record length (plus 1 s each side) before resampling.

The outputs are probabilities for review by a clinician, not diagnoses. Flags are only produced when the manifest
carries operating points (output.operating_points); no default threshold is invented here.

This module imports only numpy at import time; scipy is imported when resampling is needed, and onnxruntime or
torch when a model is loaded.
It has no web framework dependency, and it never puts waveform values in errors or notes.
"""
from __future__ import annotations

import hashlib
import io
import json
import math
from dataclasses import dataclass, field
from fractions import Fraction
from pathlib import Path
from typing import Any, Sequence

import numpy as np

DISCLAIMER = ("Probabilities from a research model trained on hospital ECGs (PTB-XL), for review by a clinician. "
              "Not a diagnosis.")

# Lead name aliases mapped to the canonical upper-case names used by the PTB-XL packer (I II III AVR AVL AVF V1-V6).
_LEAD_ALIASES = {"DI": "I", "D1": "I", "DII": "II", "D2": "II", "DIII": "III", "D3": "III",
                 "1": "I", "2": "II", "3": "III"}
_UNIT_SCALE = {"MV": 1.0, "UV": 1e-3, "µV".upper(): 1e-3, "μV".upper(): 1e-3, "V": 1e3}
# Quality heuristics (on the signal in mV after resampling, before normalization)
FLAT_STD_MV = 0.005           # a lead with SD below 5 uV is treated as flat or disconnected
HIGH_AMPLITUDE_MV = 10.0      # a 99.9th percentile |x| above 10 mV suggests wrong units or artefact
CLIP_FRACTION = 0.02          # more than 2% of samples exactly at the lead's max or min suggests clipping
# Input bounds (they cap the CPU and memory one request can use)
MIN_FS_HZ, MAX_FS_HZ = 50.0, 1000.0
MAX_RESAMPLE_FACTOR = 100     # polyphase up and down factors must both be at most this
MAX_INPUT_SECONDS = 60.0      # longer records are refused before any copy, scan or resampling
MAX_INPUT_ROWS = 64           # at most this many leads (rows) per record


class ModelIntegrityError(RuntimeError):
    """The model folder failed verification (missing file, sha256 mismatch, bad manifest). The model is not loaded."""


class ModelLoadError(RuntimeError):
    """The model passed verification but its runtime (onnxruntime or torch) is missing or could not load it."""


class EcgInputError(ValueError):
    """The ECG passed to predict() cannot be used (wrong lead count, unknown leads, too short, non-finite values)."""


def canonical_lead(name: str) -> str:
    """Canonical lead name: upper case without spaces, so aVR, AVR and 'avr' all map to AVR; DI/DII/DIII map to I/II/III."""
    s = str(name).strip().upper().replace(" ", "").replace("_", "")
    if s.startswith("LEAD"):
        s = s[4:]
    return _LEAD_ALIASES.get(s, s)


def _sha256(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def _safe_name(name: str) -> bool:
    return bool(name) and name not in (".", "..") and not any(c in name for c in "/\\:") and ".." not in name


def _expit(z: np.ndarray) -> np.ndarray:
    z = np.asarray(z, dtype=np.float64)
    out = np.empty_like(z)
    pos = z >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    ez = np.exp(z[~pos])
    out[~pos] = ez / (1.0 + ez)
    return out


@dataclass
class QualityNote:
    code: str
    detail: str
    severity: str = "info"          # info | warning

    def to_dict(self) -> dict:
        return {"code": self.code, "severity": self.severity, "detail": self.detail}


@dataclass
class Preprocessed:
    x: np.ndarray                    # float32 (leads, record_samples), normalized
    starts: list[int]                # window start samples
    notes: list[QualityNote]
    info: dict


@dataclass
class EcgPrediction:
    probabilities: dict[str, float]
    flags: dict[str, bool] | None
    operating_points: dict[str, float] | None
    n_windows: int
    notes: list[QualityNote]
    info: dict
    model: dict
    window_probabilities: np.ndarray | None = field(default=None, repr=False)

    @property
    def quality_ok(self) -> bool:
        return not any(n.severity == "warning" for n in self.notes)

    def to_dict(self) -> dict:
        return {"probabilities": self.probabilities, "flags": self.flags, "operating_points": self.operating_points,
                "n_windows": self.n_windows, "quality": {"ok": self.quality_ok,
                                                         "notes": [n.to_dict() for n in self.notes]},
                "input": self.info, "model": self.model, "disclaimer": DISCLAIMER}


class EcgDeepModel:
    """A verified, loaded deep ECG model folder (manifest.json + model.onnx or model.ts + listed files)."""

    def __init__(self, model_dir: str | Path, *, expected_manifest_sha256: str | None = None,
                 intra_op_threads: int | None = None, keep_window_probabilities: bool = False):
        self.model_dir = Path(model_dir)
        self.keep_window_probabilities = keep_window_probabilities
        mpath = self.model_dir / "manifest.json"
        if not mpath.is_file():
            raise ModelIntegrityError(f"manifest.json not found in {self.model_dir.name}")
        mbytes = mpath.read_bytes()
        self.manifest_sha256 = _sha256(mbytes)
        if expected_manifest_sha256 and self.manifest_sha256 != expected_manifest_sha256.lower():
            raise ModelIntegrityError("manifest.json sha256 does not match the pinned value")
        try:
            m = json.loads(mbytes.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as e:
            raise ModelIntegrityError(f"manifest.json is not valid JSON ({type(e).__name__})") from None
        self.manifest = m
        self._blobs = self._verify_files(m)
        self._parse_spec(m)
        self.format = m.get("format")
        self._run = self._load_runtime(self._blobs[m["weights"]], intra_op_threads)
        del self._blobs   # the weights live in the runtime now; drop the raw bytes

    # ---- loading -------------------------------------------------------------------------------------------------
    def _verify_files(self, m: dict) -> dict[str, bytes]:
        files = m.get("files_sha256")
        weights = m.get("weights")
        if not isinstance(files, dict) or not files:
            raise ModelIntegrityError("manifest.files_sha256 is missing or empty")
        if weights not in files:
            raise ModelIntegrityError("manifest.weights is not listed in manifest.files_sha256")
        blobs = {}
        for name, want in files.items():
            if not isinstance(name, str) or not _safe_name(name):
                raise ModelIntegrityError(f"unsafe file name in manifest.files_sha256: {name!r}")
            p = self.model_dir / name
            if not p.is_file():
                raise ModelIntegrityError(f"file listed in manifest is missing: {name}")
            b = p.read_bytes()                       # read once; everything below uses these bytes
            if _sha256(b) != str(want).lower():
                raise ModelIntegrityError(f"sha256 mismatch for {name}; refusing to load")
            blobs[name] = b
        if m.get("sha256") and str(m["sha256"]).lower() != _sha256(blobs[weights]):
            raise ModelIntegrityError("manifest.sha256 does not match the weights file")
        return blobs

    def _parse_spec(self, m: dict) -> None:
        try:
            inp, out = m["input"], m["output"]
            self.leads = [canonical_lead(x) for x in inp["leads"]]
            self.fs_hz = float(inp["fs_hz"])
            shape = inp["tensor_shape"]
            self.crop = int(shape[2])
            self.record = int(inp["record_samples"])
            self.stride = int(inp["window_stride_samples"])
            norm = inp["normalization"]
            self.classes = [str(c) for c in out["classes"]]
        except (KeyError, TypeError, IndexError, ValueError) as e:
            raise ModelIntegrityError(f"manifest input/output spec is incomplete ({type(e).__name__}: {e})") from None
        if int(shape[1]) != len(self.leads):
            raise ModelIntegrityError("manifest tensor_shape lead axis does not match input.leads")
        if len(set(self.leads)) != len(self.leads):
            raise ModelIntegrityError("manifest input.leads has duplicate leads")
        if not (0 < self.crop <= self.record and self.stride > 0 and self.fs_hz > 0):
            raise ModelIntegrityError("manifest window spec is invalid")
        if norm.get("type") != "per_lead_zscore":
            raise ModelIntegrityError(f"unsupported normalization type {norm.get('type')!r}")
        self.mean = np.asarray(norm["mean"], dtype=np.float32)
        self.std = np.asarray(norm["std"], dtype=np.float32)
        if self.mean.shape != (len(self.leads),) or self.std.shape != (len(self.leads),) \
                or not np.all(np.isfinite(self.mean)) or not np.all(self.std > 0):
            raise ModelIntegrityError("manifest normalization mean/std must be finite, one per lead, std > 0")
        if out.get("activation", "sigmoid") != "sigmoid" or out.get("type", "multilabel") != "multilabel":
            raise ModelIntegrityError("only multilabel sigmoid outputs are supported")
        if str(inp.get("unit", "mV")).upper() != "MV":
            raise ModelIntegrityError(f"unsupported input unit {inp.get('unit')!r}")
        if "feature_names.json" in self._blobs:      # verified bytes; must agree with the manifest leads
            try:
                fn = json.loads(self._blobs["feature_names.json"].decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError):
                raise ModelIntegrityError("feature_names.json is not valid JSON") from None
            if [canonical_lead(x) for x in fn] != self.leads:
                raise ModelIntegrityError("feature_names.json does not match manifest input.leads")
        self.operating_points = self._parse_operating_points(out.get("operating_points"))

    def _parse_operating_points(self, ops: Any) -> dict[str, float] | None:
        """Accept {class: threshold} or {class: {"threshold": t, ...}}; None when the manifest has none."""
        if not ops:
            return None
        if not isinstance(ops, dict):
            raise ModelIntegrityError("manifest output.operating_points must be a mapping of class to threshold")
        res = {}
        for c, v in ops.items():
            t = v.get("threshold") if isinstance(v, dict) else v
            if c not in self.classes or not isinstance(t, (int, float)) or not 0.0 <= float(t) <= 1.0:
                raise ModelIntegrityError(f"invalid operating point for class {c!r}")
            res[c] = float(t)
        return res

    def _load_runtime(self, weights: bytes, threads: int | None):
        fmt = self.format
        if fmt == "onnx":
            try:
                import onnxruntime as ort
            except ImportError:
                raise ModelLoadError("manifest format is onnx but onnxruntime is not installed") from None
            so = ort.SessionOptions()
            if threads:
                so.intra_op_num_threads = int(threads)
            try:
                sess = ort.InferenceSession(weights, sess_options=so, providers=["CPUExecutionProvider"])
            except Exception as e:
                raise ModelLoadError(f"onnxruntime could not load the model ({type(e).__name__})") from None
            in_name = sess.get_inputs()[0].name
            out_name = sess.get_outputs()[0].name

            def run(x: np.ndarray) -> np.ndarray:
                return np.asarray(sess.run([out_name], {in_name: x})[0])
            return run
        if fmt == "torchscript":
            try:
                import torch
            except ImportError:
                raise ModelLoadError("manifest format is torchscript but torch is not installed") from None
            try:
                mod = torch.jit.load(io.BytesIO(weights), map_location="cpu")
            except Exception as e:
                raise ModelLoadError(f"torch could not load the TorchScript model ({type(e).__name__})") from None
            mod.eval()
            if threads:
                torch.set_num_threads(int(threads))

            def run(x: np.ndarray) -> np.ndarray:
                with torch.no_grad():
                    return mod(torch.from_numpy(x)).float().numpy()
            return run
        raise ModelIntegrityError(f"unsupported model format {fmt!r}")

    # ---- metadata ------------------------------------------------------------------------------------------------
    def describe(self) -> dict:
        m = self.manifest
        return {"model_id": m.get("model_id"), "status": m.get("status"), "task": m.get("task"),
                "format": self.format, "arch": m.get("arch"), "weights_sha256": m.get("sha256"),
                "manifest_sha256": self.manifest_sha256, "code_commit": m.get("code_commit"),
                "leads": self.leads, "fs_hz": self.fs_hz, "record_samples": self.record,
                "crop_samples": self.crop, "stride_samples": self.stride, "classes": self.classes,
                "operating_points": self.operating_points, "headline": m.get("headline")}

    # ---- preprocessing -------------------------------------------------------------------------------------------
    def _order_leads(self, x: np.ndarray, leads: Sequence[str] | None, notes: list[QualityNote]) -> tuple[np.ndarray, str]:
        need = len(self.leads)
        if leads is None:
            if x.shape[0] != need:
                raise EcgInputError(f"expected {need} leads ({', '.join(self.leads)}), got {x.shape[0]}")
            notes.append(QualityNote("lead_order_assumed",
                                     f"no lead names given; assumed the order {' '.join(self.leads)}"))
            return x, "assumed"
        if len(leads) != x.shape[0]:
            raise EcgInputError(f"{len(leads)} lead names given for {x.shape[0]} rows")
        given = [canonical_lead(n) for n in leads]
        dup = sorted({g for g in given if given.count(g) > 1})
        if dup:
            raise EcgInputError(f"duplicate leads: {', '.join(dup)}")
        missing = [l for l in self.leads if l not in given]
        if missing:
            raise EcgInputError(f"missing leads required by the model: {', '.join(missing)} "
                                f"(expected {need}: {', '.join(self.leads)})")
        extra = [g for g in given if g not in self.leads]
        idx = [given.index(l) for l in self.leads]
        status = "given"
        if extra:
            notes.append(QualityNote("extra_leads_ignored", f"leads not used by the model were ignored: {', '.join(extra)}"))
            status = "selected"
        elif idx != list(range(need)):
            notes.append(QualityNote("leads_reordered", "leads were reordered to the model's order"))
            status = "reordered"
        return x[idx], status

    def _ratio(self, fs: float) -> tuple[int, int] | None:
        """Polyphase (up, down) for fs -> model rate, both at most MAX_RESAMPLE_FACTOR; None when no resampling."""
        if abs(fs - self.fs_hz) < 1e-6 * self.fs_hz:
            return None
        frac = Fraction(self.fs_hz / fs).limit_denominator(MAX_RESAMPLE_FACTOR)
        up, down = frac.numerator, frac.denominator
        if up < 1 or up > MAX_RESAMPLE_FACTOR or down > MAX_RESAMPLE_FACTOR \
                or abs(fs * up / down - self.fs_hz) > 1e-3 * self.fs_hz:
            raise EcgInputError(f"cannot resample {fs:g} Hz to {self.fs_hz:g} Hz with up/down factors of at most "
                                f"{MAX_RESAMPLE_FACTOR}")
        return up, down

    def _resample(self, x: np.ndarray, fs: float, notes: list[QualityNote]) -> np.ndarray:
        ratio = self._ratio(fs)
        if ratio is None:
            return x
        from scipy.signal import resample_poly
        up, down = ratio
        y = resample_poly(x, up, down, axis=1).astype(np.float32)
        notes.append(QualityNote("resampled", f"resampled from {fs:g} Hz to {self.fs_hz:g} Hz (polyphase {up}/{down})"))
        if fs < self.fs_hz:
            notes.append(QualityNote("upsampled", f"input rate {fs:g} Hz is below the model's {self.fs_hz:g} Hz; "
                                     "upsampling adds no information", "warning"))
        return y

    def _fit_length(self, x: np.ndarray, length_policy: str, notes: list[QualityNote],
                    n_orig: int | None = None) -> tuple[np.ndarray, dict]:
        """n_orig: the record's full length at the model rate when preprocess() already cropped it (for notes)."""
        n, rec, fs = x.shape[1], self.record, self.fs_hz
        info = {"padded_samples": 0, "cropped_samples": 0}
        if n == rec and n_orig is None:
            return x, info
        if length_policy == "strict":
            raise EcgInputError(f"expected {rec} samples ({rec / fs:g} s at {fs:g} Hz) after resampling, got {n}")
        if n >= rec:
            off, n_full = (n - rec) // 2, max(n, n_orig or 0)
            info["cropped_samples"] = n_full - rec
            notes.append(QualityNote("cropped", f"record was {n_full / fs:.2f} s; used the central {rec / fs:g} s "
                                     f"({n_full - rec} samples dropped)"))
            return x[:, off:off + rec], info
        if n < self.crop:
            raise EcgInputError(f"record is {n / fs:.2f} s; at least {self.crop / fs:g} s (one window) is required, "
                                f"{rec / fs:g} s expected")
        if length_policy == "valid":
            info["missing_samples"] = rec - n
            notes.append(QualityNote("short_record", f"record is {n / fs:.2f} s of the expected {rec / fs:g} s; "
                                     "scored on windows inside the real signal only", "warning"))
            return x, info
        info["padded_samples"] = rec - n
        # pad_crop: pad at the end with each lead's training mean (0 after normalization), i.e. a flat line
        pad = np.repeat(self.mean[:, None], rec - n, axis=1).astype(np.float32)
        notes.append(QualityNote("padded", f"record is {n / fs:.2f} s of the expected {rec / fs:g} s; padded with "
                                 f"{(rec - n) / fs:.2f} s of flat signal, probabilities are less reliable", "warning"))
        return np.concatenate([x, pad], axis=1), info

    def _quality(self, x: np.ndarray, notes: list[QualityNote]) -> None:
        sd = x.std(axis=1)
        flat = [l for l, s in zip(self.leads, sd) if s < FLAT_STD_MV]
        if flat:
            notes.append(QualityNote("flat_leads", f"flat or disconnected leads: {', '.join(flat)}", "warning"))
        p999 = np.percentile(np.abs(x), 99.9, axis=1)
        high = [l for l, v in zip(self.leads, p999) if v > HIGH_AMPLITUDE_MV]
        if high:
            notes.append(QualityNote("high_amplitude", f"amplitude above {HIGH_AMPLITUDE_MV:g} mV in "
                                     f"{', '.join(high)}; check units (mV expected) and artefact", "warning"))
        clipped = []
        for l, row, s in zip(self.leads, x, sd):       # a clipped lead sits exactly at its max or min for a while
            if s < FLAT_STD_MV or row.size == 0:
                continue
            at_rail = max(np.mean(row == row.max()), np.mean(row == row.min()))
            if at_rail > CLIP_FRACTION:
                clipped.append(l)
        if clipped:
            notes.append(QualityNote("possible_clipping", f"possible amplitude clipping in {', '.join(clipped)}",
                                     "warning"))

    def preprocess(self, ecg: np.ndarray, fs_hz: float, leads: Sequence[str] | None = None, *,
                   unit: str = "mV", length_policy: str = "pad_crop") -> Preprocessed:
        """Lead check and order, unit scaling, resampling, length fit, quality notes and per-lead z-score.

        length_policy: "pad_crop" (default; pad short records with flat signal, crop long ones to the centre),
        "valid" (short records are scored on windows inside the real signal only; long ones are cropped), or
        "strict" (anything but exactly record_samples after resampling is an error).
        """
        if length_policy not in ("pad_crop", "valid", "strict"):
            raise ValueError(f"unknown length_policy {length_policy!r}")
        x = np.asarray(ecg)
        if x.ndim != 2:
            raise EcgInputError(f"ecg must be 2-D (leads, samples); got {x.ndim}-D")
        if not np.issubdtype(x.dtype, np.number):
            raise EcgInputError("ecg must be numeric")
        try:
            fs = float(fs_hz)
        except (TypeError, ValueError):
            raise EcgInputError("fs_hz must be a number") from None
        if not math.isfinite(fs) or fs <= 0:
            raise EcgInputError("fs_hz must be a positive finite number")
        if not MIN_FS_HZ <= fs <= MAX_FS_HZ:
            raise EcgInputError(f"fs_hz must be between {MIN_FS_HZ:g} and {MAX_FS_HZ:g} Hz, got {fs:g}")
        self._ratio(fs)                          # refuse an unusable rate before any work on the samples
        scale = _UNIT_SCALE.get(str(unit).strip().upper())
        if scale is None:
            raise EcgInputError(f"unsupported unit {unit!r} (use mV, uV or V)")
        n_in = int(x.shape[1])
        if x.shape[0] > MAX_INPUT_ROWS:
            raise EcgInputError(f"ecg has {x.shape[0]} rows; at most {MAX_INPUT_ROWS} leads are accepted")
        if n_in > math.floor(MAX_INPUT_SECONDS * fs):
            raise EcgInputError(f"record is {n_in / fs:.1f} s; at most {MAX_INPUT_SECONDS:g} s is accepted")
        n_orig = None
        if length_policy != "strict":            # crop to what the model needs (+1 s each side for the resampling
            margin = math.ceil(fs)               # filter) before copying, scanning or resampling anything
            need = math.ceil(self.record * fs / self.fs_hz) + 2 * margin
            if n_in > need:
                off = (n_in - need) // 2
                x = x[:, off:off + need]         # a view, no copy
                n_orig = int(round(n_in * self.fs_hz / fs))
        x = x.astype(np.float64)
        bad = ~np.isfinite(x)
        if bad.any():
            raise EcgInputError(f"ecg has {int(bad.sum())} non-finite samples (NaN or inf); fill or drop them first")
        notes: list[QualityNote] = []
        if self.model_status() != "active":      # MODEL_REGISTRY status values: draft | active | superseded | ...
            notes.append(QualityNote("draft_model", f"model {self.manifest.get('model_id')} has status "
                                     f"{self.model_status()}, not an active registry release"))
        x, lead_status = self._order_leads(x, leads, notes)
        x = (x * scale).astype(np.float32)
        self._quality(x, notes)                  # on the real input signal, before resampling and padding
        x = self._resample(x, fs, notes)
        n_rs = x.shape[1]
        x, len_info = self._fit_length(x, length_policy, notes, n_orig)
        n_real = min(n_rs, self.record)
        x = ((x - self.mean[:, None]) / self.std[:, None]).astype(np.float32)
        last = (n_real if length_policy == "valid" else x.shape[1]) - self.crop
        starts = list(range(0, last + 1, self.stride))
        info = {"fs_hz_in": fs, "fs_hz_model": self.fs_hz, "samples_in": int(n_in), "samples_after_resample": int(n_rs),
                "seconds_in": round(n_in / fs, 3), "lead_order": lead_status, "unit_in": unit,
                "length_policy": length_policy, **len_info}
        return Preprocessed(x=x, starts=starts, notes=notes, info=info)

    def model_status(self) -> str:
        return str(self.manifest.get("status", "draft"))

    # ---- inference -----------------------------------------------------------------------------------------------
    def predict(self, ecg: np.ndarray, fs_hz: float, leads: Sequence[str] | None = None, *, unit: str = "mV",
                length_policy: str = "pad_crop") -> EcgPrediction:
        """Score one ECG record of shape (leads, samples). Returns per-class mean sigmoid probabilities."""
        pp = self.preprocess(ecg, fs_hz, leads, unit=unit, length_policy=length_policy)
        wins = np.stack([pp.x[:, s:s + self.crop] for s in pp.starts]).astype(np.float32)
        logits = self._run(np.ascontiguousarray(wins))
        if logits.shape != (len(pp.starts), len(self.classes)):
            raise ModelLoadError(f"model output shape {tuple(logits.shape)} does not match "
                                 f"({len(pp.starts)}, {len(self.classes)})")
        pw = _expit(logits)
        prob = pw.mean(axis=0)
        probs = {c: float(p) for c, p in zip(self.classes, prob)}
        flags = None
        if self.operating_points:
            flags = {c: bool(probs[c] >= t) for c, t in self.operating_points.items()}
        mid = self.manifest.get("model_id")
        model = {"model_id": mid, "status": self.model_status(), "task": self.manifest.get("task"),
                 "format": self.format, "weights_sha256": self.manifest.get("sha256")}
        notes = list(pp.notes)
        if not mid or mid == "UNASSIGNED":
            notes.append(QualityNote("no_registry_id", "model has no EVA registry id yet"))
        if flags is None:
            notes.append(QualityNote("no_operating_points",
                                     "manifest has no operating points; probabilities only, no flags"))
        return EcgPrediction(probabilities=probs, flags=flags, operating_points=self.operating_points,
                             n_windows=len(pp.starts), notes=notes, info=pp.info, model=model,
                             window_probabilities=pw.astype(np.float32) if self.keep_window_probabilities else None)
