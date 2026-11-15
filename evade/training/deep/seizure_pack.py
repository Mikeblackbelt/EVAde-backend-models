"""Pack CHB-MIT and Siena into per-subject float16 window shards for the deep seizure-suspect model (CPU job).

  python -m evade.training.deep.seizure_pack --chb <chbmit root> --siena <siena root> --out <dir>
         [--fs 256] [--max-free 3] [--workers 4] [--siena-workers 1] [--smoke]

Reuses the beta preprocessing: evade.preprocessing.datasets.read_edf (Siena with label_prefix="EEG"), the fixed
18-channel bipolar montage (select_bipolar for CHB-MIT, bipolar_from_monopolar for Siena), preprocess_eeg
(0.5-45 Hz zero-phase band-pass, mains notch 60 Hz CHB-MIT / 50 Hz Siena, resample), the beta window labels and
buffers (label_windows_seizure) and the beta file selection and seizure parsing (evade.training.data.chb_jobs, and
siena_files = siena_jobs with a name match that tolerates the real Siena lists' typos: every file with a seizure
plus `--max-free` seizure-free files per subject; -1 keeps every file).

One shard per CHB-MIT case or Siena subject: <out>/<dataset>/<case>.npz with
  X          float16 (N, 18, T)   windows ALREADY NORMALIZED per window (per_window_robust below), T = 4 s x fs,
                                  step 2 s; this is exactly the tensor the model sees in training and in the API
  y          int8    (N,)         1 ictal (>= 2 s overlap), 0 non-ictal, -1 buffer around a seizure (excluded from
                                  window metrics and training, kept for smoothing and event scoring)
  rec        int16   (N,)         index into `records`
  start_s    float32 (N,)         window start within its record
  records    str     (R,)         record ids (EDF stem); rec_dur_s float32 (R,); rec_channels int8 (R,) present;
                                  rec_present bool (R, 18) which montage channels exist (missing ones are zero-filled
                                  and stay exactly 0 after normalization; records with fewer than --min-channels are
                                  excluded)
  sz_rec, sz_start_s, sz_end_s    seizure table (record index, start, end in s), used for event-level scoring
  norm       str                  "per_window_robust" (training refuses shards without it: stale-shard guard)
  pack_info  str (JSON)           the shard's log lines, excluded records and inconsistent-annotation count, replayed
                                  when a resumed prep skips the finished shard
  channels, dataset, subject, fs
Normalization (identical to the API's seizure_deep adapter, evade/models/deep_tasks.py, "per_window_robust"): for
each window and channel, x_norm = clip((x - median) / max(1.4826 x median(|x - median|), 1e-6), -20, 20), with the
median over the window's own samples. No per-person statistics, nothing computed from labels or other windows.
chb21 is chb01 1.5 years later: it gets its own shard and training groups it with chb01.
Dataset files are read as data only; nothing in them is executed.
"""
from __future__ import annotations

import argparse
import json
import os
import re
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
NORM = "per_window_robust"   # the API's name for the rule (evade/models/deep_tasks.py SeizureAdapter)
NORM_CLIP = 20.0
MAD_EPS = 1e-6               # the API's floor on 1.4826 x MAD (a zero-filled channel stays exactly 0)
SUBJECT_ALIAS = {"chb21": "chb01"}
SIENA6 = ["PN00", "PN01", "PN03", "PN05", "PN06", "PN07"]      # the beta's six Siena subjects
BETA5 = ["chb01", "chb02", "chb03", "chb04", "chb05"]


def siena_blocks(text: str) -> list[tuple[str, str]]:
    """Split a Siena seizure list into (file name, block text), one block per 'File name:' line.

    The list repeats 'File name:' for every seizure, so a file with several seizures has several blocks (the beta
    parser read only the first one). 'Start time:' / 'End time:' lines without the word 'Seizure' are normalized
    so evade.preprocessing.eeg.parse_siena_seizure_list reads them.
    """
    text = re.sub(r"(?im)^(\s*)(start|end)\s+time", r"\1Seizure \2 time", text)
    out = []
    parts = re.split(r"(?im)^\s*File\s+name\s*:\s*", text)
    for part in parts[1:]:
        name, _, rest = part.partition("\n")
        out.append((name.strip(), rest))
    return out


def _canon_siena(name: str) -> str:
    return re.sub(r"^PNO(?=\d)", "PN0", name.strip().upper())        # PN06 lists write "PNO6-1.edf" (letter O)


def siena_name_match(listed: str, edf_name: str) -> bool:
    """A list's 'File name:' against an EDF name, tolerating the real lists' typos: letter O for zero (PN06) and a
    bare subject name for its first file (PN01 lists 'PN01.edf' for PN01-1.edf)."""
    a, b = _canon_siena(listed), _canon_siena(edf_name)
    return a == b or ("-" not in a and b == a.replace(".EDF", "-1.EDF"))


def siena_is_listed(text: str, edf_name: str) -> bool:
    return any(siena_name_match(n, edf_name) for n, _ in siena_blocks(text))


def siena_files(root: Path, max_free: int) -> list[tuple[Path, str, str]]:
    """evade.training.data.siena_jobs with siena_name_match: every listed file + `max_free` unlisted per subject."""
    jobs = []
    for subj in sorted(p for p in root.glob("PN[0-9][0-9]") if p.is_dir()):
        lst = next(iter(sorted(subj.glob("Seizures-list-*.txt"))), None)
        text = lst.read_text(errors="ignore") if lst else ""
        free = 0
        for edf in sorted(subj.glob("*.edf")):
            if siena_is_listed(text, edf.name):
                jobs.append((edf, text, subj.name))
            elif free < max_free:
                jobs.append((edf, text, subj.name)); free += 1
    return jobs


def siena_segments(text: str, edf_name: str, rec_dur: float) -> tuple[list, list]:
    """Seizures of one Siena file as (start_s, end_s) offsets, split into usable and inconsistent ones."""
    segs = []
    for name, block in siena_blocks(text):
        if not siena_name_match(name, edf_name):
            continue
        _, sz = parse_siena_seizure_list(block)
        segs += [((s - reg) % 86400.0, (e - reg) % 86400.0) for reg, s, e in sz]   # modulo: midnight rollover
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
    wins32, starts = make_windows(x, fs_out, config.EEG_WIN_S, config.EEG_STEP_S)
    del x
    wins = np.empty(wins32.shape, np.float16)
    for i in range(0, len(wins32), 512):        # chunks bound the float32 temporaries
        wins[i:i + 512] = per_window_robust(wins32[i:i + 512])
    del wins32
    if dataset == "chbmit":
        good, bad = list(seizures_or_text), []
    else:
        good, bad = siena_segments(seizures_or_text, edf.name, rec_dur)
    lab = label_windows_seizure(starts, config.EEG_WIN_S, good)
    for s0, _ in bad:   # inconsistent annotation: never ictal, excluded from the listed onset on
        lab[(starts >= s0 - config.SZ_PRE_BUFFER_S) & (lab != 1)] = -1
    return {"X": wins, "y": lab.astype(np.int8), "start_s": starts.astype(np.float32), "record": edf.stem,
            "rec_dur_s": float(rec_dur), "channels_present": int(sum(1 for p in present if p)),
            "present": np.asarray([bool(p) for p in present]),
            "seizures": [(float(a), float(b)) for a, b in good], "n_bad_annotations": len(bad), "fs_in": float(fs)}


def per_window_robust(w: np.ndarray) -> np.ndarray:
    """(..., T) windows -> float32 robust z per window and channel, the API's per_window_robust rule exactly:
    clip((x - median) / max(1.4826 x median(|x - median|), 1e-6), -20, 20), statistics over the last axis."""
    w = np.asarray(w, np.float32)
    med = np.median(w, axis=-1, keepdims=True)
    mad = 1.4826 * np.median(np.abs(w - med), axis=-1, keepdims=True)
    return np.clip((w - med) / np.maximum(mad, MAD_EPS), -NORM_CLIP, NORM_CLIP).astype(np.float32)


def siena_listed_seizures(text: str) -> int:
    """Number of seizures in a Siena list (all blocks, every seizure in a block)."""
    return sum(len(parse_siena_seizure_list(block)[1]) for _, block in siena_blocks(text))


def _finished(out: Path, job: dict) -> dict | None:
    """Summary of a finished shard (its log replayed), or None if it is missing or stale (another normalization)."""
    if not out.exists():
        return None
    z = np.load(out, allow_pickle=False)
    if "norm" not in z.files or str(z["norm"]) != NORM or "pack_info" not in z.files:
        return None
    info = json.loads(str(z["pack_info"]))
    return {"case": job["case"], "dataset": job["dataset"], "skipped": True, "n": int(len(z["y"])),
            "n_ictal": int((z["y"] == 1).sum()), "n_seizures": int(len(z["sz_rec"])),
            "n_inconsistent_annotations": info["n_bad"], "files": int(len(z["records"])),
            "excluded": info["excluded"], "log": ["(finished shard, log replayed)"] + info["log"]}


def pack_subject(job: dict) -> dict:
    """Build one shard. job: dataset, case, files [(edf, seizures_or_text)], out, fs, smoke, min_channels.

    Skips a finished shard (replaying its log, exclusions and counts); a shard from an older normalization is
    rebuilt. Records with fewer than `min_channels` of the 18 montage channels are left out (logged and counted):
    mostly zero-filled windows would be scored with real labels.
    """
    out = Path(job["out"]) / job["dataset"] / f"{job['case']}.npz"
    done = _finished(out, job)
    if done is not None:
        return done
    t0 = time.time()
    parts, recs, durs, chans, pres, sz_rec, sz_s, sz_e, rec_idx, log = [], [], [], [], [], [], [], [], [], []
    excluded, n_bad = [], 0
    if out.exists():
        log.append("rebuilding a shard packed with an older normalization")
    for edf, info in job["files"]:
        try:
            r = process_record(Path(edf), job["dataset"], info, job["fs"])
        except Exception as e:                 # noqa: BLE001  (partial download, unreadable header)
            log.append(f"SKIPPED {Path(edf).name}: {type(e).__name__}: {e}")
            continue
        if r["channels_present"] < job["min_channels"]:
            excluded.append({"record": r["record"], "channels_present": r["channels_present"],
                             "seizures": len(r["seizures"])})
            log.append(f"EXCLUDED {Path(edf).name}: only {r['channels_present']}/18 montage channels "
                       f"({len(r['seizures'])} seizures lost)")
            continue
        if job["smoke"]:                       # keep every ictal window and a slice of the rest
            keep = np.flatnonzero(r["y"] != 0)
            keep = np.union1d(keep, np.arange(0, len(r["y"]), max(1, len(r["y"]) // 200)))
            for k in ("X", "y", "start_s"):
                r[k] = r[k][keep]
        i = len(recs)
        recs.append(r["record"]); durs.append(r["rec_dur_s"]); chans.append(r["channels_present"])
        pres.append(r["present"])
        for a, b in r["seizures"]:
            sz_rec.append(i); sz_s.append(a); sz_e.append(b)
        n_bad += r["n_bad_annotations"]
        rec_idx.append(np.full(len(r["y"]), i, np.int16))
        parts.append(r)
        listed = job["dataset"] == "siena" and siena_is_listed(str(info), Path(edf).name)
        if listed and not r["seizures"]:
            log.append(f"WARNING {Path(edf).name}: listed in the seizure file but no usable seizure parsed")
        log.append(f"{Path(edf).name}: {len(r['y'])} windows, {int((r['y'] == 1).sum())} ictal, "
                   f"{len(r['seizures'])} seizures, {r['n_bad_annotations']} inconsistent, "
                   f"{r['channels_present']}/18 channels, fs {r['fs_in']:g} Hz")
    if job["dataset"] == "siena" and job["files"] and not job["smoke"]:
        n_listed = siena_listed_seizures(str(job["files"][0][1]))
        if n_listed != len(sz_rec) + n_bad:
            log.append(f"WARNING {job['case']}: {n_listed} seizures in the list, {len(sz_rec)} usable and "
                       f"{n_bad} inconsistent parsed (files missing, excluded, or names that do not match)")
    if not parts:
        return {"case": job["case"], "dataset": job["dataset"], "skipped": False, "n": 0, "excluded": excluded,
                "log": log}
    X = np.concatenate([p["X"] for p in parts]); [p.pop("X") for p in parts]
    y = np.concatenate([p["y"] for p in parts])
    rec = np.concatenate(rec_idx)
    present = np.stack(pres)                   # (records, 18)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(out.stem + ".tmp.npz")
    info = {"log": log, "excluded": excluded, "n_bad": n_bad}
    np.savez(tmp, X=X, y=y, rec=rec, start_s=np.concatenate([p["start_s"] for p in parts]),
             records=np.asarray(recs), rec_dur_s=np.asarray(durs, np.float32), rec_channels=np.asarray(chans, np.int8),
             rec_present=present, sz_rec=np.asarray(sz_rec, np.int16), sz_start_s=np.asarray(sz_s, np.float32),
             sz_end_s=np.asarray(sz_e, np.float32), norm=np.asarray(NORM), pack_info=np.asarray(json.dumps(info)),
             channels=np.asarray(CHANNELS), dataset=np.asarray(job["dataset"]),
             subject=np.asarray(SUBJECT_ALIAS.get(job["case"], job["case"])), fs=np.asarray(job["fs"], np.float32))
    os.replace(tmp, out)
    return {"case": job["case"], "dataset": job["dataset"], "skipped": False, "n": int(len(y)),
            "n_ictal": int((y == 1).sum()), "n_seizures": len(sz_rec), "n_inconsistent_annotations": n_bad,
            "files": len(recs), "excluded": excluded, "sec": round(time.time() - t0, 1), "log": log}


def _smoke_pick(files: list, is_seizure, n_sz: int, n_free: int) -> list:
    """Smoke runs: seizure files first (so every subject has ictal windows), then a few seizure-free ones."""
    return [f for f in files if is_seizure(f)][:n_sz] + [f for f in files if not is_seizure(f)][:n_free]


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
            if a.smoke:
                files = _smoke_pick(files, lambda f: bool(f[1]), 2, 1)
            jobs.append({"dataset": "chbmit", "case": case, "files": files})
    if a.siena:
        by_subj: dict[str, list] = {}
        for edf, text, subj in siena_files(Path(a.siena), max_free):
            by_subj.setdefault(subj, []).append((str(edf), text))
        for subj, files in sorted(by_subj.items()):
            if a.cases and subj not in a.cases:
                continue
            if a.smoke:
                files = _smoke_pick(files, lambda f: siena_is_listed(f[1], Path(f[0]).name), 2, 0)
            jobs.append({"dataset": "siena", "case": subj, "files": files})
    for j in jobs:
        j.update(out=str(a.out), fs=float(a.fs), smoke=bool(a.smoke), min_channels=int(a.min_channels))
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
    p.add_argument("--min-channels", type=int, default=12,
                   help="records with fewer of the 18 montage channels present are excluded (logged, counted)")
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
               "smoke": a.smoke, "channels": CHANNELS, "normalization": NORM, "shards": res,
               "sec": round(time.time() - t0, 1)}
    for ds in ("chbmit", "siena"):
        rs = [r for r in res if r["dataset"] == ds and r.get("n")]
        ex = [e for r in res if r["dataset"] == ds for e in r.get("excluded", [])]
        summary[ds] = {"cases": len(rs), "windows": sum(r["n"] for r in rs), "ictal": sum(r["n_ictal"] for r in rs),
                       "seizures": sum(r["n_seizures"] for r in rs),
                       "inconsistent_annotations": sum(r.get("n_inconsistent_annotations", 0) for r in rs),
                       "excluded_records": len(ex), "excluded_seizures": sum(e["seizures"] for e in ex)}
    b5 = [r for r in res if r["case"] in BETA5]
    summary["check_beta5"] = {"windows": sum(r.get("n", 0) for r in b5), "ictal": sum(r.get("n_ictal", 0) for r in b5),
                              "beta_reference": "97,624 windows / 992 ictal at 250 Hz (counts at 256 Hz may differ "
                                                "by a few windows per file)"}
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "pack_summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps({k: summary[k] for k in ("chbmit", "siena", "check_beta5")}), flush=True)
    if summary["siena"]["cases"] == 14 and not a.smoke and summary["siena"]["seizures"] != 47:   # all subjects only
        print(f"WARNING: {summary['siena']['seizures']} usable Siena seizures parsed; the dataset page lists 47",
              flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
