"""EEG preprocessing: channel mapping to a fixed bipolar montage, seizure labels, sleep labels."""
from __future__ import annotations

import re

import numpy as np

from . import config
from .signal import bandpass, notch, resample

_BANDS = {"delta": (0.5, 4), "theta": (4, 8), "alpha": (8, 13), "beta": (13, 30)}


def norm_electrode(name: str) -> str:
    """'EEG Fp1-REF' -> 'FP1', 'T3' -> 'T7'. Upper case, strip prefix and reference suffix, alias."""
    n = name.upper().strip()
    n = re.sub(r"^EEG[\s_-]*", "", n)
    n = re.split(r"[-_ ]", n)[0]
    return config.ELECTRODE_ALIASES.get(n, n)


def norm_bipolar_name(name: str) -> str:
    """'T8-P8-0' -> 'T8-P8', 't3-t5' -> 'T7-P7'. Returns '' for non-pairs."""
    parts = re.split(r"-", name.upper().strip())
    if len(parts) >= 2 and re.fullmatch(r"[A-Z]+[0-9Z]", parts[0]) and re.fullmatch(r"[A-Z]+[0-9Z]", parts[1]):
        a = config.ELECTRODE_ALIASES.get(parts[0], parts[0])
        b = config.ELECTRODE_ALIASES.get(parts[1], parts[1])
        return f"{a}-{b}"
    return ""


def select_bipolar(data: np.ndarray, names: list[str]) -> tuple[np.ndarray, list[str]]:
    """Pick the fixed 18-channel montage from a bipolar recording (CHB-MIT style).

    Ignores duplicate or extra channels (for example the '-1' repeats of T8-P8). Missing channels
    are zero-filled and reported in the returned mask names list as '' so callers can flag them.
    """
    index: dict[str, int] = {}
    for i, n in enumerate(names):
        key = norm_bipolar_name(n)
        if key and key not in index:
            index[key] = i
    out = np.zeros((len(config.BIPOLAR_NAMES), data.shape[-1]), dtype=data.dtype)
    present = []
    for k, target in enumerate(config.BIPOLAR_NAMES):
        if target in index:
            out[k] = data[index[target]]
            present.append(target)
        else:
            present.append("")
    return out, present


def bipolar_from_monopolar(data: np.ndarray, names: list[str]) -> tuple[np.ndarray, list[str]]:
    """Derive the fixed 18-channel bipolar montage from monopolar channels (Siena, EEGMAT)."""
    idx: dict[str, int] = {}
    for i, n in enumerate(names):
        key = norm_electrode(n)
        if key not in idx:
            idx[key] = i
    out = np.zeros((len(config.BIPOLAR_PAIRS), data.shape[-1]), dtype=data.dtype)
    present = []
    for k, (a, b) in enumerate(config.BIPOLAR_PAIRS):
        if a in idx and b in idx:
            out[k] = data[idx[a]] - data[idx[b]]
            present.append(config.BIPOLAR_NAMES[k])
        else:
            present.append("")
    return out, present


def preprocess_eeg(x: np.ndarray, fs: float, fs_out: float = config.FS_TARGET,
                   mains: float = config.NOTCH_HZ) -> tuple[np.ndarray, float]:
    """Band-pass 0.5-45 Hz, mains notch, resample. x is (channels, samples)."""
    x = np.nan_to_num(np.asarray(x, float))
    x = bandpass(x, fs, *config.EEG_BAND)
    x = notch(x, fs, mains)
    return resample(x, fs, fs_out), fs_out


def band_power_features(win: np.ndarray, fs: float) -> dict:
    """Mean relative band power (delta, theta, alpha, beta) over channels, plus theta/beta ratio."""
    from scipy.signal import welch
    f, p = welch(win, fs=fs, nperseg=min(win.shape[-1], int(2 * fs)), axis=-1)
    df = f[1] - f[0]
    pw = {k: float((p[..., (f >= lo) & (f < hi)].sum(-1) * df).mean()) for k, (lo, hi) in _BANDS.items()}
    tot = sum(pw.values()) or 1e-12
    out = {f"rel_{k}": v / tot for k, v in pw.items()}
    out["theta_beta"] = pw["theta"] / max(pw["beta"], 1e-12)
    return out


def label_windows_seizure(starts_s: np.ndarray, win_s: float, seizures: list[tuple[float, float]]) -> np.ndarray:
    """1 = ictal, 0 = interictal, -1 = excluded buffer around a seizure.

    A window is ictal if it overlaps a seizure by at least config.SZ_MIN_OVERLAP_S. Non-ictal windows
    inside the pre/post buffers are -1 so training can drop them.
    """
    starts = np.asarray(starts_s, float)
    ends = starts + win_s
    lab = np.zeros(len(starts), dtype=int)
    for s, e in seizures:
        overlap = np.minimum(ends, e) - np.maximum(starts, s)
        ictal = overlap >= config.SZ_MIN_OVERLAP_S
        near = (ends > s - config.SZ_PRE_BUFFER_S) & (starts < e + config.SZ_POST_BUFFER_S)
        lab[near & ~ictal & (lab == 0)] = -1
        lab[ictal] = 1
    return lab


def parse_chb_summary(text: str) -> dict[str, list[tuple[float, float]]]:
    """Parse a CHB-MIT summary file into {file_name: [(start_s, end_s), ...]}.

    Handles both 'Seizure Start Time: N seconds' and 'Seizure 1 Start Time: N seconds'.
    """
    out: dict[str, list[tuple[float, float]]] = {}
    cur = None
    starts: list[float] = []
    for line in text.splitlines():
        m = re.match(r"\s*File Name:\s*(\S+)", line)
        if m:
            cur = m.group(1)
            out.setdefault(cur, [])
            starts = []
            continue
        if cur is None:
            continue
        m = re.match(r"\s*Seizure(?:\s+\d+)?\s+Start Time:\s*(\d+)", line, re.I)
        if m:
            starts.append(float(m.group(1)))
            continue
        m = re.match(r"\s*Seizure(?:\s+\d+)?\s+End Time:\s*(\d+)", line, re.I)
        if m and starts:
            out[cur].append((starts.pop(0), float(m.group(1))))
    return out


def hms_to_s(txt: str) -> float:
    """'12.34.56' or '12:34:56' -> seconds since midnight."""
    h, m, s = [int(p) for p in re.split(r"[.:]", txt.strip())[:3]]
    return h * 3600 + m * 60 + s


def parse_siena_seizure_list(text: str) -> tuple[float | None, list[tuple[float, float, float]]]:
    """Parse Siena 'Seizures-list-PNxx.txt' into (registration_start_s, [(reg_start, seiz_start, seiz_end)]).

    Format observed in the dataset description: 'Registration start time: hh.mm.ss', 'Seizure start time:
    hh.mm.ss', 'Seizure end time: hh.mm.ss' per file. Returns times as seconds since midnight; the caller
    converts to offsets within the recording. Not validated against all files (verify on real data).
    """
    reg = None
    seizures = []
    cur_start = None
    for line in text.splitlines():
        m = re.search(r"Registration start time\s*:\s*([\d.:]+)", line, re.I)
        if m:
            reg = hms_to_s(m.group(1))
            continue
        m = re.search(r"Seizure start time\s*:\s*([\d.:]+)", line, re.I)
        if m:
            cur_start = hms_to_s(m.group(1))
            continue
        m = re.search(r"Seizure end time\s*:\s*([\d.:]+)", line, re.I)
        if m and cur_start is not None and reg is not None:
            seizures.append((reg, cur_start, hms_to_s(m.group(1))))
            cur_start = None
    return reg, seizures


SLEEP_STAGE_MAP = {"Sleep stage W": 0, "Sleep stage 1": 1, "Sleep stage 2": 2,
                   "Sleep stage 3": 3, "Sleep stage 4": 3, "Sleep stage R": 4}


def hypnogram_to_epochs(onsets_s, durations_s, descriptions, n_epochs: int, epoch_s: float = config.SLEEP_EPOCH_S) -> np.ndarray:
    """Convert EDF+ hypnogram annotations to one stage per epoch: 0=W 1=N1 2=N2 3=N3 (stages 3 and 4 merged) 4=REM, -1=unknown."""
    lab = -np.ones(n_epochs, dtype=int)
    for on, du, d in zip(onsets_s, durations_s, descriptions):
        code = SLEEP_STAGE_MAP.get(str(d).strip(), -1)
        a = int(round(on / epoch_s))
        b = int(round((on + du) / epoch_s))
        lab[max(a, 0):min(b, n_epochs)] = code
    return lab
