"""Pack CHB-MIT and Siena into per-subject float16 window shards for the deep seizure-suspect model (CPU job).

  python -m evade.training.deep.seizure_pack --chb <chbmit root> --siena <siena root> --out <dir>
         [--fs 256] [--max-free 3] [--workers 4] [--siena-workers 1] [--smoke]

Reuses the beta preprocessing: evade.preprocessing.datasets.read_edf (Siena with label_prefix="EEG"), the fixed
18-channel bipolar montage (select_bipolar for CHB-MIT, bipolar_from_monopolar for Siena), preprocess_eeg
(0.5-45 Hz zero-phase band-pass, mains notch 60 Hz CHB-MIT / 50 Hz Siena, resample), the beta window labels and
buffers (label_windows_seizure) and the beta file selection and seizure parsing (evade.training.data.chb_jobs and
siena_jobs: every file with a seizure plus `--max-free` seizure-free files per subject; -1 keeps every file).

One shard per CHB-MIT case or Siena subject: <out>/<dataset>/<case>.npz with
  X          float16 (N, 18, T)   windows in microvolts, T = 4 s x fs, step 2 s, clipped to +-CLIP_UV
  y          int8    (N,)         1 ictal (>= 2 s overlap), 0 non-ictal, -1 buffer around a seizure (excluded from
                                  window metrics and training, kept for smoothing and event scoring)
  rec        int16   (N,)         index into `records`
  start_s    float32 (N,)         window start within its record
  records    str     (R,)         record ids (EDF stem); rec_dur_s float32 (R,); rec_channels int8 (R,) present
  sz_rec, sz_start_s, sz_end_s    seizure table (record index, start, end in s), used for event-level scoring
  norm_median, norm_mad  float32 (18,)   per-channel median and 1.4826 x MAD of this person's NON-ictal (label 0)
                                         windows: the personal baseline used by the model
  norm_median_all, norm_mad_all          the same over all windows, ictal included (the beta's rule, kept only to
                                         show the limitation; not used by default)
  channels, dataset, subject, fs
chb21 is chb01 1.5 years later: it gets its own shard (its own baseline session) and training groups it with chb01.
Dataset files are read as data only; nothing in them is executed.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from ...preprocessing import config
from ...preprocessing.datasets import read_edf
from ...preprocessing.eeg import (bipolar_from_monopolar, label_windows_seizure, parse_siena_seizure_list,
                                  preprocess_eeg, select_bipolar)
from ...preprocessing.signal import make_windows
from .. import data as beta_data

CHANNELS = list(config.BIPOLAR_NAMES)
CLIP_UV = 2000.0           # bipolar EEG beyond +-2 mV is artifact; also keeps float16 finite
MAD_FLOOR_UV = 0.5         # zero-filled (missing) channels get this spread instead of 0
NORM_MAX_WINDOWS = 4000    # windows used for the median / MAD (evenly spaced, deterministic)
SUBJECT_ALIAS = {"chb21": "chb01"}
SIENA6 = ["PN00", "PN01", "PN03", "PN05", "PN06", "PN07"]      # the beta's six Siena subjects
BETA5 = ["chb01", "chb02", "chb03", "chb04", "chb05"]


def siena_segments(text: str, edf_name: str, rec_dur: float) -> tuple[list, list]:
    """Seizures of one Siena file as (start_s, end_s) offsets, split into usable and inconsistent ones (beta logic)."""
    block = text.split(edf_name)[1].split("File name")[0] if edf_name in text else ""
    _, sz = parse_siena_seizure_list(block)
    segs = [((s - reg) % 86400.0, (e - reg) % 86400.0) for reg, s, e in sz]   # modulo: midnight rollover
    bad = [(s0, e0) for s0, e0 in segs if e0 > rec_dur + 5.0 or e0 < s0]
    return [sg for sg in segs if sg not in bad], bad


def _filter_per_channel(x: np.ndarray, fs: float, fs_out: float, mains: float) -> np.ndarray:
    """preprocess_eeg one channel at a time (bounds the float64 working memory on multi-hour Siena files)."""
    out = None
    for k in range(x.shape[0]):
        yk, _ = preprocess_eeg(x[k:k + 1], fs, fs_out=fs_out, mains=mains)
        if out is None:
            out = np.empty((x.shape[0], yk.shape[-1]), np.float32)
        out[k] = yk[0]
    return out


def process_record(edf: Path, dataset: str, seizures_or_text, fs_out: float) -> dict:
    """Read, montage, filter, window and label one EDF. Returns float16 windows and metadata."""
    if dataset == "chbmit":
        data, names, fs, _ = read_edf(edf)
        x, present = select_bipolar(data, names)
        mains = 60.0
    else:
        data, names, fs, _ = read_edf(edf, label_prefix="EEG")
        x, present = bipolar_from_monopolar(data, names)
        mains = 50.0                                   # recorded in Italy
    del data
    x = _filter_per_channel(x, fs, fs_out, mains)
    rec_dur = x.shape[-1] / fs_out
    np.clip(x, -CLIP_UV, CLIP_UV, out=x)
    wins, starts = make_windows(x.astype(np.float16), fs_out, config.EEG_WIN_S, config.EEG_STEP_S)
    del x
    if dataset == "chbmit":
        good, bad = list(seizures_or_text), []
    else:
        good, bad = siena_segments(seizures_or_text, edf.name, rec_dur)
    lab = label_windows_seizure(starts, config.EEG_WIN_S, good)
    for s0, _ in bad:   # inconsistent annotation: never ictal, excluded from the listed onset on
        lab[(starts >= s0 - config.SZ_PRE_BUFFER_S) & (lab != 1)] = -1
    return {"X": wins, "y": lab.astype(np.int8), "start_s": starts.astype(np.float32), "record": edf.stem,
            "rec_dur_s": float(rec_dur), "channels_present": int(sum(1 for p in present if p)),
            "seizures": [(float(a), float(b)) for a, b in good], "n_bad_annotations": len(bad), "fs_in": float(fs)}


def robust_channel_stats(X: np.ndarray, sel: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-channel median and 1.4826 x MAD over the samples of the selected windows (evenly subsampled)."""
    idx = np.flatnonzero(sel)
    if len(idx) == 0:
        idx = np.arange(len(X))
    if len(idx) > NORM_MAX_WINDOWS:
        idx = idx[np.linspace(0, len(idx) - 1, NORM_MAX_WINDOWS).astype(int)]
    med = np.empty(X.shape[1], np.float32)
    mad = np.empty(X.shape[1], np.float32)
    for c in range(X.shape[1]):
        v = X[idx, c, :].astype(np.float32).ravel()
        m = float(np.median(v))
        med[c] = m
        mad[c] = max(1.4826 * float(np.median(np.abs(v - m))), MAD_FLOOR_UV)
    return med, mad


def pack_subject(job: dict) -> dict:
    """Build one shard. job: dataset, case, files [(edf, seizures_or_text)], out, fs, smoke. Skips a finished shard."""
    out = Path(job["out"]) / job["dataset"] / f"{job['case']}.npz"
    if out.exists():
        z = np.load(out, allow_pickle=False)
        return {"case": job["case"], "dataset": job["dataset"], "skipped": True, "n": int(len(z["y"])),
                "n_ictal": int((z["y"] == 1).sum()), "n_seizures": int(len(z["sz_rec"])), "files": int(len(z["records"]))}
    t0 = time.time()
    parts, recs, durs, chans, sz_rec, sz_s, sz_e, rec_idx, log = [], [], [], [], [], [], [], [], []
    for edf, info in job["files"]:
        try:
            r = process_record(Path(edf), job["dataset"], info, job["fs"])
        except Exception as e:                 # noqa: BLE001  (partial download, unreadable header)
            log.append(f"SKIPPED {Path(edf).name}: {type(e).__name__}: {e}")
            continue
        if job["smoke"]:                       # keep every ictal window and a slice of the rest
            keep = np.flatnonzero(r["y"] != 0)
            keep = np.union1d(keep, np.arange(0, len(r["y"]), max(1, len(r["y"]) // 200)))
            for k in ("X", "y", "start_s"):
                r[k] = r[k][keep]
        i = len(recs)
        recs.append(r["record"]); durs.append(r["rec_dur_s"]); chans.append(r["channels_present"])
        for a, b in r["seizures"]:
            sz_rec.append(i); sz_s.append(a); sz_e.append(b)
        rec_idx.append(np.full(len(r["y"]), i, np.int16))
        parts.append(r)
        listed = job["dataset"] == "siena" and Path(edf).name in str(info)
        if listed and not r["seizures"]:
            log.append(f"WARNING {Path(edf).name}: listed in the seizure file but no usable seizure parsed")
        log.append(f"{Path(edf).name}: {len(r['y'])} windows, {int((r['y'] == 1).sum())} ictal, "
                   f"{len(r['seizures'])} seizures, {r['n_bad_annotations']} inconsistent, "
                   f"{r['channels_present']}/18 channels, fs {r['fs_in']:g} Hz")
    if not parts:
        return {"case": job["case"], "dataset": job["dataset"], "skipped": False, "n": 0, "log": log}
    X = np.concatenate([p["X"] for p in parts]); [p.pop("X") for p in parts]
    y = np.concatenate([p["y"] for p in parts])
    med, mad = robust_channel_stats(X, y == 0)
    med_all, mad_all = robust_channel_stats(X, np.ones(len(y), bool))
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(out.stem + ".tmp.npz")
    np.savez(tmp, X=X, y=y, rec=np.concatenate(rec_idx), start_s=np.concatenate([p["start_s"] for p in parts]),
             records=np.asarray(recs), rec_dur_s=np.asarray(durs, np.float32), rec_channels=np.asarray(chans, np.int8),
             sz_rec=np.asarray(sz_rec, np.int16), sz_start_s=np.asarray(sz_s, np.float32),
             sz_end_s=np.asarray(sz_e, np.float32), norm_median=med, norm_mad=mad, norm_median_all=med_all,
             norm_mad_all=mad_all, channels=np.asarray(CHANNELS), dataset=np.asarray(job["dataset"]),
             subject=np.asarray(SUBJECT_ALIAS.get(job["case"], job["case"])), fs=np.asarray(job["fs"], np.float32))
    os.replace(tmp, out)
    return {"case": job["case"], "dataset": job["dataset"], "skipped": False, "n": int(len(y)),
            "n_ictal": int((y == 1).sum()), "n_seizures": len(sz_rec), "files": len(recs),
            "n_norm_windows": int(min((y == 0).sum(), NORM_MAX_WINDOWS)), "sec": round(time.time() - t0, 1), "log": log}


def build_jobs(a) -> list[dict]:
    max_free = 10 ** 6 if a.max_free < 0 else a.max_free
    jobs = []
    if a.chb:
        by_case: dict[str, list] = {}
        for edf, sz, case in beta_data.chb_jobs(Path(a.chb), max_free):
            by_case.setdefault(case, []).append((str(edf), sz))
        for case, files in sorted(by_case.items()):
            if a.cases and case not in a.cases:
                continue
            jobs.append({"dataset": "chbmit", "case": case, "files": files[:3] if a.smoke else files})
    if a.siena:
        by_subj: dict[str, list] = {}
        for edf, text, subj in beta_data.siena_jobs(Path(a.siena), max_free):
            by_subj.setdefault(subj, []).append((str(edf), text))
        for subj, files in sorted(by_subj.items()):
            if a.cases and subj not in a.cases:
                continue
            jobs.append({"dataset": "siena", "case": subj, "files": files[:2] if a.smoke else files})
    for j in jobs:
        j.update(out=str(a.out), fs=float(a.fs), smoke=bool(a.smoke))
    return jobs


def _run(jobs: list[dict], workers: int) -> list[dict]:
    if workers <= 1:
        return [pack_subject(j) for j in jobs]
    with ProcessPoolExecutor(max_workers=workers) as ex:
        return list(ex.map(pack_subject, jobs))


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--chb", type=Path)
    p.add_argument("--siena", type=Path)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--fs", type=float, default=256.0, help="output rate (CHB-MIT is 256 Hz native, Siena 512 Hz)")
    p.add_argument("--max-free", type=int, default=3, help="seizure-free files per subject (beta rule: 3; -1 = all)")
    p.add_argument("--cases", default="", help="comma list to pack only these cases or subjects")
    p.add_argument("--workers", type=int, default=4, help="parallel CHB-MIT cases")
    p.add_argument("--siena-workers", type=int, default=1, help="parallel Siena subjects (files up to 1.7 GB)")
    p.add_argument("--smoke", action="store_true", help="3 files per case, 2 per Siena subject, subsampled windows")
    a = p.parse_args(argv)
    a.cases = set(c for c in a.cases.split(",") if c)
    jobs = build_jobs(a)
    t0 = time.time()
    res = _run([j for j in jobs if j["dataset"] == "chbmit"], a.workers)
    res += _run([j for j in jobs if j["dataset"] == "siena"], a.siena_workers)
    for r in res:
        for line in r.pop("log", []):
            print(f"[{r['case']}] {line}", flush=True)
    summary = {"fs": a.fs, "window_s": config.EEG_WIN_S, "step_s": config.EEG_STEP_S, "max_free": a.max_free,
               "smoke": a.smoke, "channels": CHANNELS, "shards": res, "sec": round(time.time() - t0, 1)}
    for ds in ("chbmit", "siena"):
        rs = [r for r in res if r["dataset"] == ds and r.get("n")]
        summary[ds] = {"cases": len(rs), "windows": sum(r["n"] for r in rs), "ictal": sum(r["n_ictal"] for r in rs),
                       "seizures": sum(r["n_seizures"] for r in rs)}
    b5 = [r for r in res if r["case"] in BETA5]
    summary["check_beta5"] = {"windows": sum(r.get("n", 0) for r in b5), "ictal": sum(r.get("n_ictal", 0) for r in b5),
                              "beta_reference": "97,624 windows / 992 ictal at 250 Hz (counts at 256 Hz may differ "
                                                "by a few windows per file)"}
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "pack_summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps({k: summary[k] for k in ("chbmit", "siena", "check_beta5")}), flush=True)
    if summary["siena"]["seizures"] and not a.smoke and summary["siena"]["seizures"] != 47:
        print(f"WARNING: {summary['siena']['seizures']} usable Siena seizures parsed; the dataset page lists 47",
              flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
