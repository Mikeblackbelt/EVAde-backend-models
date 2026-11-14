"""Dataset loaders and runners. Heavy dependencies (wfdb, pyedflib) are imported lazily.

Each run_* function takes the dataset folder and an output folder, processes at most `limit` records,
writes <out>/<dataset>/<record>.npz (float32 windows) and returns a schema DataFrame.
Treat dataset folders as untrusted data: only data files are read, nothing in them is executed.
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

from . import config
from .ecg import PACED_RECORDS, BEAT_SYMBOLS, detect_qrs, preprocess_ecg, score_beats, window_table
from .eeg import (bipolar_from_monopolar, hypnogram_to_epochs, label_windows_seizure, parse_chb_summary,
                  parse_siena_seizure_list, preprocess_eeg, select_bipolar)
from .signal import assign_fold, resample


def _save(out: Path, dataset: str, record: str, wins: np.ndarray, extra: dict | None = None) -> None:
    d = out / dataset
    d.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(d / f"{record}.npz", windows=wins, **(extra or {}))


def read_edf(path: Path) -> tuple[np.ndarray, list[str], float, "pd.Timestamp | None"]:
    """Read all channels of an EDF with the same sampling rate (the most common one). Returns data, names, fs, start."""
    import pyedflib
    f = pyedflib.EdfReader(str(path))
    try:
        n = f.signals_in_file
        fss = [f.getSampleFrequency(i) for i in range(n)]
        fs = max(set(fss), key=fss.count)
        keep = [i for i in range(n) if fss[i] == fs]
        data = np.stack([f.readSignal(i) for i in keep])
        names = [f.getLabel(i) for i in keep]
        start = pd.Timestamp(f.getStartdatetime())
    finally:
        f._close()
    return data, names, float(fs), start


def read_edf_annotations(path: Path):
    import pyedflib
    f = pyedflib.EdfReader(str(path))
    try:
        on, du, de = f.readAnnotations()
        n_dur = f.getFileDuration()
    finally:
        f._close()
    return np.asarray(on, float), np.asarray(du, float), list(de), float(n_dur)


# ---------- ECG / PPG ----------
def run_mitdb(root: Path, out: Path, limit: int | None = None) -> pd.DataFrame:
    import wfdb
    recs = sorted({p.stem for p in root.glob("*.hea") if p.stem.isdigit()})[:limit]
    rows = []
    for rid in recs:
        rec = wfdb.rdrecord(str(root / rid))
        ann = wfdb.rdann(str(root / rid), "atr")
        x = rec.p_signal.T[:1]  # first lead (MLII in most records)
        fs = float(rec.fs)
        x_p, fs_p = preprocess_ecg(x, fs)
        wins, df = window_table(x_p, fs_p, config.ECG_WIN_S, config.ECG_STEP_S, participant_id=f"mitdb-{rid}",
                                record_id=rid, dataset="mitdb", modality="ecg")
        det = detect_qrs(x_p[0], fs_p)
        ref = np.array([s for s, sym in zip(ann.sample, ann.symbol) if sym in BEAT_SYMBOLS], dtype=float)
        ref_p = np.round(ref * fs_p / fs).astype(int)
        sc = score_beats(ref_p, det, fs_p)
        df["qrs_sensitivity_record"] = sc["sensitivity"]
        df["qrs_ppv_record"] = sc["ppv"]
        df["paced_record"] = rid in PACED_RECORDS
        _save(out, "mitdb", rid, wins)
        rows.append(df)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=config.SCHEMA_COLUMNS)


def run_bidmc(root: Path, out: Path, limit: int | None = None) -> pd.DataFrame:
    files = sorted(root.glob("**/bidmc_*_Signals.csv"))[:limit]   # CSVs sit in bidmc_csv/
    rows = []
    for p in files:
        rid = re.search(r"bidmc_(\d+)_", p.name).group(1)
        d = pd.read_csv(p)
        d.columns = [c.strip() for c in d.columns]
        t = d["Time [s]"].to_numpy()
        fs = 1.0 / float(np.median(np.diff(t)))
        for col, mod in (("PLETH", "ppg"), ("II", "ecg"), ("RESP", "resp")):
            if col not in d.columns:
                continue
            x = d[col].to_numpy(float)[None]
            if mod == "resp":
                from .signal import bandpass
                x = bandpass(np.nan_to_num(x), fs, 0.1, 0.8)
                x_p, fs_p = resample(x, fs, 25.0), 25.0
            elif mod == "ppg":
                from .signal import bandpass
                x_p, fs_p = resample(bandpass(np.nan_to_num(x), fs, 0.5, 8.0), fs, config.FS_TARGET), config.FS_TARGET
            else:
                x_p, fs_p = preprocess_ecg(x, fs)
            wins, df = window_table(x_p, fs_p, config.ECG_WIN_S, config.ECG_STEP_S, participant_id=f"bidmc-{rid}",
                                    record_id=f"{rid}_{mod}", dataset="bidmc", modality=mod)
            _save(out, "bidmc", f"{rid}_{mod}", wins)
            rows.append(df)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=config.SCHEMA_COLUMNS)


def run_ptbxl(root: Path, out: Path, limit: int | None = None) -> pd.DataFrame:
    import ast
    import wfdb
    meta = pd.read_csv(root / "ptbxl_database.csv", index_col="ecg_id")
    scp = pd.read_csv(root / "scp_statements.csv", index_col=0)
    diag = scp[scp.diagnostic == 1]
    rows = []
    for ecg_id, r in list(meta.iterrows())[:limit]:
        codes = ast.literal_eval(r.scp_codes)
        classes = sorted({diag.loc[c].diagnostic_class for c in codes if c in diag.index})
        rec = wfdb.rdrecord(str(root / r.filename_lr))
        x_p, fs_p = preprocess_ecg(rec.p_signal.T, float(rec.fs))
        wins, df = window_table(x_p, fs_p, 10.0, 10.0, participant_id=f"ptbxl-{int(r.patient_id)}",
                                record_id=str(ecg_id), dataset="ptbxl", modality="ecg12")
        df["strat_fold"] = int(r.strat_fold)   # use the official fold, not the hash fold
        df["diagnostic_classes"] = ",".join(classes)
        _save(out, "ptbxl", str(ecg_id), wins)
        rows.append(df)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=config.SCHEMA_COLUMNS)


# ---------- EEG ----------
def run_chbmit(root: Path, out: Path, limit: int | None = None) -> pd.DataFrame:
    rows = []
    done = 0
    for subj in sorted(p for p in root.glob("chb[0-9][0-9]") if p.is_dir()):
        summ = subj / f"{subj.name}-summary.txt"
        seiz = parse_chb_summary(summ.read_text(errors="ignore")) if summ.exists() else {}
        for edf in sorted(subj.glob("*.edf")):
            if limit is not None and done >= limit:
                return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=config.SCHEMA_COLUMNS)
            data, names, fs, _ = read_edf(edf)
            x, present = select_bipolar(data, names)
            x_p, fs_p = preprocess_eeg(x, fs)
            wins, df = window_table(x_p, fs_p, config.EEG_WIN_S, config.EEG_STEP_S, participant_id=subj.name,
                                    record_id=edf.stem, dataset="chbmit", modality="eeg")
            df["label"] = label_windows_seizure(df.window_start_s.to_numpy(), config.EEG_WIN_S, seiz.get(edf.name, []))
            df["n_channels_present"] = sum(1 for p in present if p)
            _save(out, "chbmit", edf.stem, wins, {"label": df.label.to_numpy()})
            rows.append(df)
            done += 1
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=config.SCHEMA_COLUMNS)


def run_siena(root: Path, out: Path, limit: int | None = None) -> pd.DataFrame:
    rows = []
    done = 0
    for subj in sorted(p for p in root.glob("PN[0-9][0-9]") if p.is_dir()):
        lst = next(iter(subj.glob("Seizures-list-*.txt")), None)
        text = lst.read_text(errors="ignore") if lst else ""
        for edf in sorted(subj.glob("*.edf")):
            if limit is not None and done >= limit:
                return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=config.SCHEMA_COLUMNS)
            data, names, fs, _ = read_edf(edf)
            x, present = bipolar_from_monopolar(data, names)
            x_p, fs_p = preprocess_eeg(x, fs, mains=50.0)   # recorded in Italy (50 Hz mains)
            wins, df = window_table(x_p, fs_p, config.EEG_WIN_S, config.EEG_STEP_S, participant_id=subj.name,
                                    record_id=edf.stem, dataset="siena", modality="eeg")
            # per-file seizure blocks: split the list text by file name when present
            block = text.split(edf.name)[1].split("File name")[0] if edf.name in text else ""
            _, sz = parse_siena_seizure_list(block)
            segs = [(max(s - reg, 0.0), e - reg) for reg, s, e in sz]
            df["label"] = label_windows_seizure(df.window_start_s.to_numpy(), config.EEG_WIN_S, segs)
            df["n_channels_present"] = sum(1 for p in present if p)
            _save(out, "siena", edf.stem, wins, {"label": df.label.to_numpy()})
            rows.append(df)
            done += 1
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=config.SCHEMA_COLUMNS)


def run_eegmat(root: Path, out: Path, limit: int | None = None) -> pd.DataFrame:
    rows = []
    info = root / "subject-info.csv"
    grp = {}
    if info.exists():
        si = pd.read_csv(info)
        si.columns = [c.strip() for c in si.columns]
        qcol = next((c for c in si.columns if "quality" in c.lower()), None)
        if qcol:
            grp = dict(zip(si.iloc[:, 0], si[qcol]))
    for edf in sorted(root.glob("Subject*_[12].edf"))[:limit]:
        m = re.match(r"(Subject\d+)_(\d)", edf.name)
        subj, cond = m.group(1), int(m.group(2))
        data, names, fs, _ = read_edf(edf)
        x, present = bipolar_from_monopolar(data, names)
        x_p, fs_p = preprocess_eeg(x, fs, mains=50.0)   # Ukrainian recordings, 50 Hz mains
        wins, df = window_table(x_p, fs_p, config.EEG_WIN_S, config.EEG_STEP_S, participant_id=subj,
                                record_id=edf.stem, dataset="eegmat", modality="eeg")
        df["label"] = 0 if cond == 1 else 1      # 0 = rest, 1 = mental arithmetic
        df["count_quality"] = grp.get(subj, np.nan)
        _save(out, "eegmat", edf.stem, wins)
        rows.append(df)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=config.SCHEMA_COLUMNS)


def run_sleepedfx(root: Path, out: Path, limit: int | None = None) -> pd.DataFrame:
    rows = []
    base = root / "sleep-cassette" if (root / "sleep-cassette").exists() else root
    for psg in sorted(base.glob("*-PSG.edf"))[:limit]:
        hyp = next(iter(base.glob(psg.name[:7] + "*-Hypnogram.edf")), None)
        if hyp is None:
            continue
        data, names, fs, _ = read_edf(psg)
        keep = [i for i, n in enumerate(names) if n.strip().startswith("EEG")]
        x_p, fs_p = preprocess_eeg(data[keep], fs, mains=50.0)
        on, du, de, _ = read_edf_annotations(hyp)
        wins, df = window_table(x_p, fs_p, config.SLEEP_EPOCH_S, config.SLEEP_EPOCH_S,
                                participant_id=psg.name[:5], record_id=psg.name[:7], dataset="sleepedfx", modality="eeg")
        df["label"] = hypnogram_to_epochs(on, du, de, len(df))
        _save(out, "sleepedfx", psg.name[:7], wins, {"label": df.label.to_numpy()})
        rows.append(df)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=config.SCHEMA_COLUMNS)


# ---------- NHANES ----------
def run_nhanes(root: Path, out: Path, limit: int | None = None) -> pd.DataFrame:
    """Merge DXA femur, spine and demographics (2017-March 2020) into a reference table by SEQN."""
    fem = pd.read_sas(root / "P_DXXFEM.xpt", format="xport")
    spn = pd.read_sas(root / "P_DXXSPN.xpt", format="xport")
    dem = pd.read_sas(root / "P_DEMO.xpt", format="xport")[["SEQN", "RIAGENDR", "RIDAGEYR", "RIDRETH3"]]
    d = dem.merge(fem[["SEQN", "DXXNKBMD", "DXXOSBMD"]], on="SEQN", how="inner").merge(
        spn[["SEQN", "DXXOSBMD"]].rename(columns={"DXXOSBMD": "DXXSPBMD"}), on="SEQN", how="left")
    d = d.rename(columns={"RIAGENDR": "sex", "RIDAGEYR": "age", "RIDRETH3": "race_eth",
                          "DXXNKBMD": "femoral_neck_bmd", "DXXOSBMD": "total_femur_bmd", "DXXSPBMD": "spine_bmd"})
    d["participant_id"] = "nhanes-" + d.SEQN.astype(int).astype(str)
    d["dataset"], d["modality"] = "nhanes", "bone_reference"
    out.mkdir(parents=True, exist_ok=True)
    d.to_parquet(out / "nhanes_bone_reference.parquet")
    return d


RUNNERS = {"mitdb": run_mitdb, "bidmc": run_bidmc, "ptbxl": run_ptbxl, "chbmit": run_chbmit,
           "siena": run_siena, "eegmat": run_eegmat, "sleepedfx": run_sleepedfx, "nhanes": run_nhanes}
