"""Pack Sleep-EDFx sleep-cassette nights into one npz for deep sleep staging (pure numpy EDF reader, no pyedflib).

  python -m evade.training.deep.sleep_pack --src <folder with SC*-PSG.edf and SC*-Hypnogram.edf> --out sleep_sc.npz
         [--limit N]

Per night: EEG Fpz-Cz and EEG Pz-Oz at 100 Hz (raw EDF physical values in microvolts, no filtering), cut into
30 s epochs (3,000 samples) from the start of the recording. Labels follow the beta (evade/preprocessing/eeg.py
hypnogram_to_epochs): W=0, N1=1, N2=2, N3=3 (R&K stages 3 and 4 merged), REM=4; movement time and '?' are -1.
Trimming follows the beta (evade/training/data.py sleep_features): keep the epochs from 60 epochs (30 min) before
the first sleep epoch to 60 epochs after the last one, THEN drop the -1 epochs (movement, unknown).
No normalization happens here: X stays in raw microvolts and the model normalizes each 30 s epoch on its own
(sleep_models.epoch_robust_z), so the label-based trimming only chooses which epochs are trained and scored and never
shapes any input statistic.

Output arrays (rows sorted by night, then by time; every night is one contiguous block):
  X            float16 (N, 2, 3000)   microvolts, channels in `channels` order
  y            int8    (N,)           0..4 (W N1 N2 N3 REM)
  subject      str     (N,)           e.g. SC400 (file name[:5]; bootstrap and split unit)
  night        str     (N,)           e.g. SC4001E (file name[:7]; sequences never cross nights)
  epoch_index  int32   (N,)           position of the epoch in the full recording (gaps = dropped epochs)
  classes, channels, fs, epoch_samples
The EDF files are read as data only; nothing in them is executed.

Why not pyedflib: the cluster venv (setup_env_uv.sh) has no pyedflib, and it needs a C build where no wheel exists.
This reader handles plain EDF and EDF+ (16-bit) signals and EDF+ annotation channels, which is what Sleep-EDFx uses.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

CLASSES = ["W", "N1", "N2", "N3", "REM"]
CHANNELS = ["Fpz-Cz", "Pz-Oz"]
EDF_LABELS = {"Fpz-Cz": "EEG Fpz-Cz", "Pz-Oz": "EEG Pz-Oz"}
FS = 100
EPOCH_S = 30
EPOCH_SAMPLES = FS * EPOCH_S
WAKE_PAD_EPOCHS = 60          # 30 min of 30 s epochs kept on each side of the sleep period (as the beta)
BETA_SUBJECTS = ["SC400", "SC401", "SC402", "SC403", "SC404"]
BETA_EPOCHS = 10589           # epochs the beta kept for these 5 subjects (10 nights)
# identical to evade/preprocessing/eeg.py SLEEP_STAGE_MAP (copied so this module needs only numpy)
SLEEP_STAGE_MAP = {"Sleep stage W": 0, "Sleep stage 1": 1, "Sleep stage 2": 2,
                   "Sleep stage 3": 3, "Sleep stage 4": 3, "Sleep stage R": 4}


# ---------------------------------------------------------------- EDF reading
def _fields(b: bytes, n: int, width: int) -> list[str]:
    return [b[i * width:(i + 1) * width].decode("latin-1").strip() for i in range(n)]


def read_header(path: Path) -> dict:
    with open(path, "rb") as f:
        h = f.read(256)
        if len(h) < 256:
            raise ValueError(f"{Path(path).name}: truncated header")
        ns = int(h[252:256].decode("ascii").strip())
        s = f.read(ns * 256)
    if len(s) < ns * 256:
        raise ValueError(f"{Path(path).name}: truncated signal header")
    off = 0
    cols = {}
    for name, width in (("label", 16), ("transducer", 80), ("unit", 8), ("pmin", 8), ("pmax", 8), ("dmin", 8),
                        ("dmax", 8), ("prefilter", 80), ("nsamp", 8), ("reserved", 32)):
        cols[name] = _fields(s[off:off + ns * width], ns, width)
        off += ns * width
    hdr_bytes = int(h[184:192].decode("ascii").strip())
    n_rec = int(h[236:244].decode("ascii").strip())
    rec_dur = float(h[244:252].decode("ascii").strip())
    nsamp = [int(v) for v in cols["nsamp"]]
    rec_len = sum(nsamp) * 2
    size = Path(path).stat().st_size
    n_full = (size - hdr_bytes) // rec_len if rec_len else 0
    if n_rec < 0 or n_rec > n_full:     # -1 (unknown) or an interrupted download: use the complete records present
        n_rec = int(n_full)
    return {"ns": ns, "header_bytes": hdr_bytes, "n_records": n_rec, "record_s": rec_dur,
            "reserved": h[192:236].decode("latin-1").strip(), "labels": cols["label"], "units": cols["unit"],
            "pmin": [float(v) for v in cols["pmin"]], "pmax": [float(v) for v in cols["pmax"]],
            "dmin": [float(v) for v in cols["dmin"]], "dmax": [float(v) for v in cols["dmax"]], "nsamp": nsamp}


def _records(path: Path, h: dict) -> np.ndarray:
    """All data records as int16 (n_records, samples per record over all signals)."""
    width = sum(h["nsamp"])
    raw = np.fromfile(path, dtype="<i2", count=h["n_records"] * width, offset=h["header_bytes"])
    return raw.reshape(h["n_records"], width)


def read_signals(path: Path, labels: list[str]) -> tuple[np.ndarray, float]:
    """Read the named signals (exact EDF labels, compared after strip) in physical units. Returns (len, n) and fs."""
    h = read_header(path)
    rec = _records(path, h)
    starts = np.concatenate([[0], np.cumsum(h["nsamp"])])
    out, fss = [], []
    for lab in labels:
        if lab not in h["labels"]:
            raise ValueError(f"{Path(path).name}: no signal {lab!r} (have {h['labels']})")
        i = h["labels"].index(lab)
        dig = rec[:, starts[i]:starts[i + 1]].reshape(-1).astype(np.float64)
        gain = (h["pmax"][i] - h["pmin"][i]) / (h["dmax"][i] - h["dmin"][i])
        out.append(((dig - h["dmin"][i]) * gain + h["pmin"][i]).astype(np.float32))
        fss.append(h["nsamp"][i] / h["record_s"])
    if len(set(fss)) != 1:
        raise ValueError(f"{Path(path).name}: signals have different rates {fss}")
    return np.stack(out), float(fss[0])


def read_annotations(path: Path) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """EDF+ annotations (all 'EDF Annotations' signals): onsets_s, durations_s, descriptions.

    Each data record holds TALs: '+onset[\\x15duration]\\x14text\\x14[text\\x14...]\\x00'. The first TAL of a record
    is the time-keeping one with no text; TALs without text are skipped.
    """
    h = read_header(path)
    idx = [i for i, lab in enumerate(h["labels"]) if lab == "EDF Annotations"]
    if not idx:
        raise ValueError(f"{Path(path).name}: no 'EDF Annotations' signal")
    rec = _records(path, h)
    starts = np.concatenate([[0], np.cumsum(h["nsamp"])])
    on, du, de = [], [], []
    for r in range(h["n_records"]):
        for i in idx:
            blob = rec[r, starts[i]:starts[i + 1]].astype("<i2").tobytes()
            for tal in blob.split(b"\x00"):
                if not tal or b"\x14" not in tal:
                    continue
                parts = tal.split(b"\x14")
                timing, texts = parts[0].decode("latin-1"), [t.decode("utf-8", "replace") for t in parts[1:] if t]
                if not texts:
                    continue
                if "\x15" in timing:
                    o, d = timing.split("\x15", 1)
                else:
                    o, d = timing, ""
                for t in texts:
                    on.append(float(o)); du.append(float(d) if d else 0.0); de.append(t)
    return np.asarray(on, float), np.asarray(du, float), de


# ---------------------------------------------------------------- labels
def hypnogram_to_epochs(onsets_s, durations_s, descriptions, n_epochs: int, epoch_s: float = EPOCH_S) -> np.ndarray:
    """Same rule as the beta: one stage per epoch, later annotations overwrite earlier ones, -1 = unknown."""
    lab = -np.ones(n_epochs, dtype=int)
    for on, du, d in zip(onsets_s, durations_s, descriptions):
        code = SLEEP_STAGE_MAP.get(str(d).strip(), -1)
        a = int(round(on / epoch_s))
        b = int(round((on + du) / epoch_s))
        lab[max(a, 0):min(b, n_epochs)] = code
    return lab


def trim_and_drop(label: np.ndarray) -> np.ndarray:
    """Indices kept: sleep period +/- WAKE_PAD_EPOCHS (on labels 1-4, as the beta), then -1 epochs removed."""
    asleep = np.flatnonzero((label >= 1) & (label <= 4))
    if len(asleep) == 0:
        return np.empty(0, dtype=np.int64)
    a0, a1 = max(asleep[0] - WAKE_PAD_EPOCHS, 0), min(asleep[-1] + WAKE_PAD_EPOCHS, len(label) - 1)
    idx = np.arange(a0, a1 + 1)
    return idx[label[idx] >= 0]


def pack_night(psg: Path, hyp: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    sig, fs = read_signals(psg, [EDF_LABELS[c] for c in CHANNELS])
    if abs(fs - FS) > 1e-6:
        raise ValueError(f"{psg.name}: EEG at {fs} Hz, expected {FS}")
    n_ep = sig.shape[1] // EPOCH_SAMPLES
    on, du, de = read_annotations(hyp)
    label = hypnogram_to_epochs(on, du, de, n_ep)
    keep = trim_and_drop(label)
    x = sig[:, : n_ep * EPOCH_SAMPLES].reshape(len(CHANNELS), n_ep, EPOCH_SAMPLES).transpose(1, 0, 2)[keep]
    return x.astype(np.float16), label[keep].astype(np.int8), keep.astype(np.int32)


def find_nights(src: Path) -> list[tuple[Path, Path]]:
    base = src / "sleep-cassette" if (src / "sleep-cassette").exists() else src
    out = []
    for psg in sorted(base.glob("SC*-PSG.edf")):
        hyp = next(iter(sorted(base.glob(psg.name[:7] + "*-Hypnogram.edf"))), None)
        if hyp is None:
            print(f"SKIPPED {psg.name}: no hypnogram", flush=True)
            continue
        out.append((psg, hyp))
    return out


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--src", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--limit", type=int, default=0, help="smoke: first N nights only")
    a = p.parse_args(argv)
    nights = find_nights(a.src)
    if a.limit:
        nights = nights[: a.limit]
    X, Y, S, N, E = [], [], [], [], []
    for psg, hyp in nights:
        try:
            x, y, e = pack_night(psg, hyp)
        except Exception as ex:          # noqa: BLE001  (corrupt or partial file: report and continue)
            print(f"SKIPPED {psg.name}: {type(ex).__name__}: {ex}", flush=True)
            continue
        if len(y) == 0:
            print(f"SKIPPED {psg.name}: no scored sleep", flush=True)
            continue
        X.append(x); Y.append(y); E.append(e)
        S += [psg.name[:5]] * len(y); N += [psg.name[:7]] * len(y)
        print(f"{psg.name[:7]}: {len(y)} epochs", flush=True)
    if not X:
        raise SystemExit("no nights packed")
    X, Y, E = np.concatenate(X), np.concatenate(Y), np.concatenate(E)
    S, N = np.asarray(S), np.asarray(N)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    # uncompressed: float16 EEG barely compresses and zlib on about 2 GB takes many minutes
    np.savez(a.out, X=X, y=Y, subject=S, night=N, epoch_index=E, classes=np.asarray(CLASSES),
             channels=np.asarray(CHANNELS), fs=np.asarray(FS), epoch_samples=np.asarray(EPOCH_SAMPLES))
    counts = dict(zip(CLASSES, np.bincount(Y, minlength=5).tolist()))
    beta = int(np.isin(S, BETA_SUBJECTS).sum())
    print(f"packed {len(Y)} epochs, {len(np.unique(N))} nights, {len(np.unique(S))} subjects -> {a.out}; "
          f"per class {counts}")
    print(f"beta check: SC400-SC404 kept {beta} epochs (beta: {BETA_EPOCHS}) "
          f"{'MATCH' if beta == BETA_EPOCHS else 'DIFFERENT (or not all 10 nights packed)'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
