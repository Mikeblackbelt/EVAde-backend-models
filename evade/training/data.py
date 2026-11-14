"""Build feature tables from raw datasets for the beta models. Raw files are read as data only."""
from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import pandas as pd

from ..preprocessing import config
from ..preprocessing.datasets import read_edf, read_edf_annotations
from ..preprocessing.ecg import window_table
from ..preprocessing.eeg import (bipolar_from_monopolar, hypnogram_to_epochs, label_windows_seizure,
                                 parse_chb_summary, parse_siena_seizure_list, preprocess_eeg, select_bipolar)
from .features import EEG_BANDS, SLEEP_BANDS, ecg_features, spectral_features

CHANNELS = config.BIPOLAR_NAMES
WAKE_PAD_EPOCHS = 60          # 30 min of 30 s epochs kept on each side of the sleep period


def _frame(feats: np.ndarray, names: list[str], df: pd.DataFrame, extra: dict) -> pd.DataFrame:
    out = pd.DataFrame(feats, columns=names)
    for k in ("participant_id", "record_id", "window_start_s", "quality_index"):
        out[k] = df[k].to_numpy()
    for k, v in extra.items():
        out[k] = v
    return out


def chb_file_features(edf: Path, seizures: list, subj: str) -> pd.DataFrame:
    data, names, fs, _ = read_edf(edf)
    x, present = select_bipolar(data, names)
    x_p, fs_p = preprocess_eeg(x, fs)
    del data, x
    x_p = x_p.astype(np.float32)      # halves memory for long recordings
    wins, df = window_table(x_p, fs_p, config.EEG_WIN_S, config.EEG_STEP_S, participant_id=subj,
                            record_id=edf.stem, dataset="chbmit", modality="eeg")
    label = label_windows_seizure(df.window_start_s.to_numpy(), config.EEG_WIN_S, seizures)
    feats, fn = spectral_features(wins, fs_p, EEG_BANDS, CHANNELS)
    return _frame(feats, fn, df, {"label": label, "dataset": "chbmit"})


def siena_file_features(edf: Path, text: str, subj: str) -> pd.DataFrame:
    data, names, fs, _ = read_edf(edf)
    x, present = bipolar_from_monopolar(data, names)
    x_p, fs_p = preprocess_eeg(x, fs, mains=50.0)
    del data, x
    x_p = x_p.astype(np.float32)
    wins, df = window_table(x_p, fs_p, config.EEG_WIN_S, config.EEG_STEP_S, participant_id=subj,
                            record_id=edf.stem, dataset="siena", modality="eeg")
    block = text.split(edf.name)[1].split("File name")[0] if edf.name in text else ""
    _, sz = parse_siena_seizure_list(block)
    segs = [((s - reg) % 86400.0, (e - reg) % 86400.0) for reg, s, e in sz]
    rec_dur = x_p.shape[-1] / fs_p
    bad = [(s0, e0) for s0, e0 in segs if e0 > rec_dur + 5.0 or e0 < s0]   # inconsistent annotation: do not label ictal
    good = [sg for sg in segs if sg not in bad]
    label = label_windows_seizure(df.window_start_s.to_numpy(), config.EEG_WIN_S, good)
    for s0, _ in bad:
        label[(df.window_start_s.to_numpy() >= s0 - config.SZ_PRE_BUFFER_S) & (label != 1)] = -1
    feats, fn = spectral_features(wins, fs_p, EEG_BANDS, CHANNELS)
    return _frame(feats, fn, df, {"label": label, "dataset": "siena"})


def chb_jobs(root: Path, max_free: int):
    """All files with seizures plus up to `max_free` seizure-free files per subject."""
    jobs = []
    for subj in sorted(p for p in root.glob("chb[0-9][0-9]") if p.is_dir()):
        summ = subj / f"{subj.name}-summary.txt"
        seiz = parse_chb_summary(summ.read_text(errors="ignore")) if summ.exists() else {}
        free = 0
        for edf in sorted(subj.glob("*.edf")):
            sz = seiz.get(edf.name, [])
            if sz:
                jobs.append((edf, sz, subj.name))
            elif free < max_free:
                jobs.append((edf, [], subj.name)); free += 1
    return jobs


def siena_jobs(root: Path, max_free: int):
    jobs = []
    for subj in sorted(p for p in root.glob("PN[0-9][0-9]") if p.is_dir()):
        lst = next(iter(subj.glob("Seizures-list-*.txt")), None)
        text = lst.read_text(errors="ignore") if lst else ""
        free = 0
        for edf in sorted(subj.glob("*.edf")):
            if edf.name in text:
                jobs.append((edf, text, subj.name))
            elif free < max_free:
                jobs.append((edf, text, subj.name)); free += 1
    return jobs


def sleep_features(root: Path, limit: int | None = None) -> pd.DataFrame:
    base = root / "sleep-cassette" if (root / "sleep-cassette").exists() else root
    rows = []
    for psg in sorted(base.glob("*-PSG.edf"))[:limit]:
        hyp = next(iter(base.glob(psg.name[:7] + "*-Hypnogram.edf")), None)
        if hyp is None:
            continue
        try:
            data, names, fs, _ = read_edf(psg, label_prefix="EEG")
            on, du, de, _ = read_edf_annotations(hyp)
        except Exception as e:                  # noqa: BLE001  (partial download)
            print(f"SKIPPED {psg.name}: {type(e).__name__}: {e}", flush=True)
            continue
        x_p, fs_p = preprocess_eeg(data, fs, mains=50.0)
        wins, df = window_table(x_p, fs_p, config.SLEEP_EPOCH_S, config.SLEEP_EPOCH_S, participant_id=psg.name[:5],
                                record_id=psg.name[:7], dataset="sleepedfx", modality="eeg")
        label = hypnogram_to_epochs(on, du, de, len(df))
        feats, fn = spectral_features(wins, fs_p, SLEEP_BANDS, [n.strip() for n in names], nperseg=512)
        # context: features of the previous and next epoch (edge epochs repeat themselves)
        prev, nxt = np.vstack([feats[:1], feats[:-1]]), np.vstack([feats[1:], feats[-1:]])
        feats = np.hstack([feats, prev, nxt])
        fn = fn + [f"prev_{n}" for n in fn] + [f"next_{n}" for n in fn]
        frame = _frame(feats, fn, df, {"label": label, "dataset": "sleepedfx"})
        # keep 30 min of wake before the first and after the last sleep epoch (standard Sleep-EDF practice);
        # the long out-of-bed wake periods otherwise dominate and inflate accuracy and kappa
        asleep = np.where((label >= 1) & (label <= 4))[0]
        if len(asleep):
            a0, a1 = max(asleep[0] - WAKE_PAD_EPOCHS, 0), min(asleep[-1] + WAKE_PAD_EPOCHS, len(label) - 1)
            frame = frame.iloc[a0:a1 + 1]
        rows.append(frame)
    return pd.concat(rows, ignore_index=True)


LEADS = ["I", "II", "III", "AVR", "AVL", "AVF", "V1", "V2", "V3", "V4", "V5", "V6"]
SUPER = ["NORM", "MI", "STTC", "CD", "HYP"]


def _ptb_row(root: Path, rec_path: str):
    import wfdb
    from scipy.signal import butter, sosfiltfilt
    rec = wfdb.rdrecord(str(root / rec_path))
    x = np.nan_to_num(rec.p_signal.T.astype(np.float64))
    sos = butter(4, [0.5, 40.0], btype="band", fs=float(rec.fs), output="sos")
    return sosfiltfilt(sos, x, axis=-1), float(rec.fs)


def ptbxl_features(root: Path, limit: int | None = None, n_jobs: int = 8) -> pd.DataFrame:
    from joblib import Parallel, delayed
    meta = pd.read_csv(root / "ptbxl_database.csv", index_col="ecg_id")
    scp = pd.read_csv(root / "scp_statements.csv", index_col=0)
    diag = scp[scp.diagnostic == 1]
    meta = meta.iloc[:limit] if limit else meta
    res = Parallel(n_jobs=n_jobs, batch_size=64)(delayed(_ptb_row)(root, p) for p in meta.filename_lr)
    fs = res[0][1]
    wins = np.stack([r[0] for r in res])
    feats, fn = ecg_features(wins, fs, LEADS)
    cls = []
    for s in meta.scp_codes:
        codes = ast.literal_eval(s)
        cls.append({diag.loc[c].diagnostic_class for c in codes if c in diag.index})
    out = pd.DataFrame(feats, columns=fn)
    out["participant_id"] = ["ptbxl-%d" % int(p) for p in meta.patient_id]
    out["record_id"] = meta.index.astype(str)
    out["strat_fold"] = meta.strat_fold.to_numpy()
    for k in SUPER:
        out[f"y_{k}"] = [int(k in c) for c in cls]
    return out
