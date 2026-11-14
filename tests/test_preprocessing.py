import os
from pathlib import Path

import numpy as np
import pytest

from evade.preprocessing import config
from evade.preprocessing import ecg, eeg
from evade.preprocessing.signal import (assign_fold, bandpass, make_windows, notch, resample, robust_stats,
                                        robust_z, signal_quality)

rng = np.random.default_rng(0)


def test_bandpass_attenuates_out_of_band():
    fs = 250
    t = np.arange(0, 20, 1 / fs)
    x = np.sin(2 * np.pi * 10 * t) + np.sin(2 * np.pi * 100 * t) + 3 * np.sin(2 * np.pi * 0.05 * t)
    y = bandpass(x, fs, 0.5, 40)
    mid = slice(500, -500)
    spec = np.abs(np.fft.rfft(y[mid]))
    f = np.fft.rfftfreq(len(y[mid]), 1 / fs)
    assert spec[np.argmin(abs(f - 10))] > 20 * spec[np.argmin(abs(f - 100))]
    assert abs(y[mid].mean()) < 0.1


def test_bandpass_rejects_bad_band():
    with pytest.raises(ValueError):
        bandpass(np.zeros(1000), 250, 50, 10)


def test_notch_removes_mains_and_skips_above_nyquist():
    fs = 250
    t = np.arange(0, 10, 1 / fs)
    x = np.sin(2 * np.pi * 10 * t) + np.sin(2 * np.pi * 60 * t)
    y = notch(x, fs, 60)
    f = np.fft.rfftfreq(len(y), 1 / fs)
    s = np.abs(np.fft.rfft(y))
    assert s[np.argmin(abs(f - 60))] < 0.05 * s[np.argmin(abs(f - 10))]
    z = np.zeros(100)
    assert notch(z, 100, 60) is z  # 60 Hz >= Nyquist 50 Hz: untouched


@pytest.mark.parametrize("fs_in,fs_out", [(360, 250), (125, 250), (512, 250), (500, 250), (256, 250)])
def test_resample_length_and_tone(fs_in, fs_out):
    t = np.arange(0, 10, 1 / fs_in)
    x = np.sin(2 * np.pi * 5 * t)
    y = resample(x, fs_in, fs_out)
    assert abs(len(y) - 10 * fs_out) <= 2
    f = np.fft.rfftfreq(len(y), 1 / fs_out)
    assert abs(f[np.argmax(np.abs(np.fft.rfft(y)))] - 5) < 0.3


def test_resample_noop():
    x = rng.normal(size=100)
    assert resample(x, 250, 250) is x


def test_make_windows_counts_and_starts():
    x = rng.normal(size=(2, 250 * 10))
    w, s = make_windows(x, 250, 4, 2)
    assert w.shape == (4, 2, 1000)
    assert list(s) == [0, 2, 4, 6]
    w2, s2 = make_windows(x[:, :500], 250, 4, 2)
    assert len(w2) == 0 and len(s2) == 0


def test_quality_flat_vs_noise_vs_nan():
    good = rng.normal(size=1000)
    assert signal_quality(good) > 0.9
    assert signal_quality(np.zeros(1000)) < 0.1
    bad = good.copy()
    bad[:500] = np.nan
    assert signal_quality(bad) <= 0.5
    spiky = good.copy()
    spiky[::5] = 500
    assert signal_quality(spiky) < 0.6


def test_robust_stats_resist_outlier_and_floor():
    x = np.concatenate([np.full(50, 10.0) + rng.normal(0, 0.1, 50), [1000.0]])
    c, s = robust_stats(x)
    assert abs(c - 10) < 0.2
    assert s >= 1.5  # floor is 15% of the median
    assert np.all(np.abs(robust_z([1e9, -1e9], c, s)) == 5.0)
    with pytest.raises(ValueError):
        robust_stats([np.nan])


def test_fold_is_deterministic_and_subjectwise():
    assert assign_fold("chb01") == assign_fold("chb01")
    folds = {assign_fold(f"p{i}") for i in range(200)}
    assert folds == set(range(config.N_FOLDS))


def _synthetic_ecg(fs=250, seconds=60, hr=72):
    t = np.arange(0, seconds, 1 / fs)
    x = rng.normal(0, 0.02, len(t))
    beats = np.arange(0.5, seconds - 0.5, 60 / hr)
    for b in beats:
        i = int(b * fs)
        x[i - 2:i + 3] += np.array([0.1, 0.5, 1.5, 0.5, 0.1])
    return x, (beats * fs).astype(int)


def test_qrs_detection_on_synthetic_ecg():
    x, ref = _synthetic_ecg()
    det = ecg.detect_qrs(x, 250)
    sc = ecg.score_beats(ref, det, 250)
    assert sc["sensitivity"] > 0.97 and sc["ppv"] > 0.97


def test_qrs_empty_and_flat():
    assert len(ecg.detect_qrs(np.zeros(2500), 250)) == 0


def test_score_beats_exact_cases():
    ref = np.array([100, 300, 500])
    assert ecg.score_beats(ref, ref, 250)["sensitivity"] == 1.0
    sc = ecg.score_beats(ref, np.array([100, 500, 900]), 250)
    assert (sc["tp"], sc["fn"], sc["fp"]) == (2, 1, 1)
    sc = ecg.score_beats(ref, np.array([100 + 60]), 250)  # 240 ms off, outside 150 ms
    assert sc["tp"] == 0


def test_hrv_constant_rr_and_filtering():
    f = ecg.hrv_features(np.full(50, 0.8))
    assert abs(f["hr_bpm"] - 75) < 1e-6 and f["rmssd_ms"] == 0 and f["valid_frac"] == 1
    rr = np.array([0.8] * 20 + [5.0, 0.1])
    g = ecg.hrv_features(rr)
    assert g["n_rr"] == 20 and abs(g["valid_frac"] - 20 / 22) < 1e-9
    assert np.isnan(ecg.hrv_features(np.array([0.8, 0.8]))["rmssd_ms"])


def test_window_table_schema_and_labels():
    x, _ = _synthetic_ecg(seconds=65)
    xp, fs = ecg.preprocess_ecg(x, 250)
    wins, df = ecg.window_table(xp, fs, 30, 15, participant_id="p1", record_id="r1", dataset="syn", modality="ecg")
    assert wins.dtype == np.float32 and wins.shape[0] == len(df) == 3
    assert list(df.columns) == config.SCHEMA_COLUMNS
    assert df.fold.nunique() == 1 and df.pipeline_version.iloc[0].startswith("pp-")


def test_bipolar_from_monopolar_is_difference_and_aliases():
    names = ["EEG Fp1-REF", "EEG F7-REF", "T3", "T5", "O1", "Fp2", "F8", "T4", "T6", "O2",
             "F3", "C3", "P3", "F4", "C4", "P4", "Fz", "Cz", "Pz"]
    data = rng.normal(size=(len(names), 500))
    out, present = eeg.bipolar_from_monopolar(data, names)
    assert out.shape == (18, 500)
    assert np.allclose(out[0], data[0] - data[1])       # FP1-F7
    assert np.allclose(out[1], data[1] - data[2])       # F7-T7 (T3 aliased)
    assert all(present)
    out2, present2 = eeg.bipolar_from_monopolar(data[:5], names[:5])
    assert not all(present2) and np.all(out2[-1] == 0)


def test_select_bipolar_handles_duplicates_and_extras():
    names = ["FP1-F7", "F7-T7", "T7-P7", "P7-O1", "FP1-F3", "F3-C3", "C3-P3", "P3-O1", "FP2-F4", "F4-C4",
             "C4-P4", "P4-O2", "FP2-F8", "F8-T8", "T8-P8-0", "P8-O2", "FZ-CZ", "CZ-PZ", "P7-T7", "T8-P8-1", "VNS", "ECG"]
    data = np.arange(len(names))[:, None] * np.ones((1, 100))
    out, present = eeg.select_bipolar(data, names)
    assert all(present)
    assert out[14, 0] == names.index("T8-P8-0")          # first duplicate wins
    assert out[2, 0] == 2                                # T7-P7, not the reversed P7-T7


def test_label_windows_seizure_buffers():
    starts = np.arange(0, 600, 2.0)
    lab = eeg.label_windows_seizure(starts, 4.0, [(300.0, 330.0)])
    assert lab[starts == 300][0] == 1 and lab[starts == 328][0] == 1
    assert lab[starts == 100][0] == 0
    assert lab[starts == 270][0] == -1 and lab[starts == 400][0] == -1
    assert lab[starts == 500][0] == 0
    assert lab[starts == 298][0] == 1   # exactly 2 s overlap counts as ictal
    assert set(np.unique(lab)) <= {-1, 0, 1}


CHB_NEW = """File Name: chb01_03.edf
File Start Time: 13:43:04
File End Time: 14:43:04
Number of Seizures in File: 1
Seizure Start Time: 2996 seconds
Seizure End Time: 3036 seconds

File Name: chb01_04.edf
Number of Seizures in File: 0
"""
CHB_OLD = """File Name: chb02_16.edf
Number of Seizures in File: 2
Seizure 1 Start Time: 130 seconds
Seizure 1 End Time: 212 seconds
Seizure 2 Start Time: 2972 seconds
Seizure 2 End Time: 3053 seconds
"""


def test_parse_chb_summary_both_formats():
    a = eeg.parse_chb_summary(CHB_NEW)
    assert a["chb01_03.edf"] == [(2996.0, 3036.0)] and a["chb01_04.edf"] == []
    b = eeg.parse_chb_summary(CHB_OLD)
    assert b["chb02_16.edf"] == [(130.0, 212.0), (2972.0, 3053.0)]


def test_parse_siena_list():
    txt = """Seizure n. 1
File name: PN00-1.edf
Registration start time: 19.39.33
Registration end time: 20.50.09
Seizure start time: 19.58.36
Seizure end time: 19.59.46
"""
    reg, sz = eeg.parse_siena_seizure_list(txt)
    assert reg == 19 * 3600 + 39 * 60 + 33
    assert len(sz) == 1 and sz[0][2] - sz[0][1] == 70


def test_hypnogram_epochs():
    on, du = [0.0, 60.0, 120.0, 150.0], [60.0, 60.0, 30.0, 30.0]
    de = ["Sleep stage W", "Sleep stage 2", "Sleep stage 4", "Sleep stage ?"]
    lab = eeg.hypnogram_to_epochs(on, du, de, 6)
    assert list(lab) == [0, 0, 2, 2, 3, -1]


def test_band_power_alpha_peak():
    fs = 250
    t = np.arange(0, 4, 1 / fs)
    w = np.stack([np.sin(2 * np.pi * 10 * t) + 0.05 * rng.normal(size=len(t)) for _ in range(4)])
    f = eeg.band_power_features(w, fs)
    assert f["rel_alpha"] > 0.8
    assert abs(sum(f[k] for k in ("rel_delta", "rel_theta", "rel_alpha", "rel_beta")) - 1) < 1e-9


# ---- optional smoke tests against real downloaded data (set EVADE_DATA to the data root) ----
DATA = os.environ.get("EVADE_DATA")
needs_data = pytest.mark.skipif(not DATA, reason="EVADE_DATA not set")


@needs_data
def test_mitdb_smoke(tmp_path):
    pytest.importorskip("wfdb")
    root = Path(DATA) / "ecg_ppg" / "mitdb"
    if not root.exists():
        pytest.skip("mitdb not downloaded")
    from evade.preprocessing.datasets import run_mitdb
    df = run_mitdb(root, tmp_path, limit=2)
    assert len(df) > 50 and df.qrs_sensitivity_record.min() > 0.9 and df.qrs_ppv_record.min() > 0.9
    assert list(df.columns[:len(config.SCHEMA_COLUMNS)]) == config.SCHEMA_COLUMNS


@needs_data
def test_bidmc_smoke(tmp_path):
    root = Path(DATA) / "ecg_ppg" / "bidmc"
    if not root.exists():
        pytest.skip("bidmc not downloaded")
    from evade.preprocessing.datasets import run_bidmc
    df = run_bidmc(root, tmp_path, limit=1)
    assert set(df.modality) == {"ppg", "ecg", "resp"} and len(df) > 20
