"""End-to-end test of the deep seizure pipeline on tiny synthetic EDF files (CPU, no real data):
pack (CHB-MIT-like and Siena-like EDFs) -> LOSO smoke train with a fake SIGUSR1 and resume -> final model with the
external set -> ONNX export and manifest. Plus unit tests of the metric helpers."""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
try:                       # real EDF files when pyedflib is installed (the cluster venv); else an in-memory reader
    import pyedflib  # noqa: F401
    HAVE_EDF = True
except ImportError:
    HAVE_EDF = False
FAKE: dict = {}

from evade.preprocessing import config  # noqa: E402
from evade.training import metrics as M  # noqa: E402
from evade.training.deep import seizure_eval as E  # noqa: E402
from evade.training.deep import seizure_export, seizure_pack, seizure_train  # noqa: E402


def _signal(n, fs, rng, seizures=(), amp=300.0):
    t = np.arange(n) / fs
    x = rng.normal(0, 25.0, n) + 10 * np.sin(2 * np.pi * 10 * t)
    for a, b in seizures:
        m = (t >= a) & (t < b)
        x[m] += amp * np.sin(2 * np.pi * 3 * t[m])
    return x


def _write_edf(path, labels, sigs, fs):
    if not HAVE_EDF:
        path.write_bytes(b"")
        FAKE[str(path)] = (np.stack(sigs), list(labels), float(fs))
        return
    from pyedflib import highlevel
    hdr = highlevel.make_signal_headers(labels, dimension="uV", sample_frequency=fs, physical_min=-3000,
                                        physical_max=3000)
    highlevel.write_edf(str(path), [np.clip(s, -2999, 2999) for s in sigs], hdr)


def make_chb(root, rng, cases=("chb01", "chb02", "chb03"), dur=300, fs=256):
    for c in cases:
        d = root / c
        d.mkdir(parents=True)
        lines = []
        files = [(1, [(120.0, 170.0)]), (2, [])]
        if c == "chb03":
            files.append((3, []))          # chb03_02 lacks FZ-CZ; chb03_03 has 5 channels only (excluded)
        for k, sz in files:
            name = f"{c}_{k:02d}.edf"
            labels = list(config.BIPOLAR_NAMES) + ["ECG"]
            if c == "chb03" and k == 2:
                labels.remove("FZ-CZ")
            if c == "chb03" and k == 3:
                labels = labels[:5]
            sigs = [_signal(dur * fs, fs, rng, sz) for _ in labels]
            _write_edf(d / name, labels, sigs, fs)
            lines += [f"File Name: {name}", "File Start Time: 10:00:00", "File End Time: 10:05:00",
                      f"Number of Seizures in File: {len(sz)}"]
            for a, b in sz:
                lines += [f"Seizure Start Time: {int(a)} seconds", f"Seizure End Time: {int(b)} seconds"]
            lines.append("")
        (d / f"{c}-summary.txt").write_text("\n".join(lines))


def make_siena(root, rng, subj="PN00", dur=240, fs=512):
    d = root / subj
    d.mkdir(parents=True)
    elec = sorted({e for pair in config.BIPOLAR_PAIRS for e in pair})
    labels = ["EEG " + e.capitalize() for e in elec] + ["EKG EKG"]
    sz = [(40.0, 70.0), (150.0, 190.0)]
    sigs = [_signal(dur * fs, fs, rng, sz, amp=150.0 * (1 + i % 3)) for i, _ in enumerate(labels)]
    _write_edf(d / f"{subj}-1.edf", labels, sigs, fs)
    # two seizures in one file: the list repeats "File name:" per seizure (the second block uses "Start time:")
    (d / f"Seizures-list-{subj}.txt").write_text(
        "Seizures list\n\nSeizure n 1\nFile name: PN00-1.edf\nRegistration start time: 10.00.00\n"
        "Registration end time: 10.04.00\nSeizure start time: 10.00.40\nSeizure end time: 10.01.10\n\n"
        "Seizure n 2\nFile name: PN00-1.edf\nRegistration start time: 10.00.00\nRegistration end time: 10.04.00\n"
        "Start time: 10.02.30\nEnd time: 10.03.10\n")


def _fake_read_edf(path, label_prefix=None):
    """Stand-in for evade.preprocessing.datasets.read_edf (same return contract) when pyedflib is missing."""
    data, labels, fs = FAKE[str(path)]
    keep = [i for i, n in enumerate(labels) if label_prefix is None or n.startswith(label_prefix)]
    return data[keep].astype(float), [labels[i] for i in keep], fs, None


@pytest.fixture(scope="module")
def packed(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("sz")
    rng = np.random.default_rng(0)
    make_chb(tmp / "chb", rng)
    make_siena(tmp / "siena", rng)
    out = tmp / "pack"
    with pytest.MonkeyPatch.context() as mp:
        if not HAVE_EDF:
            mp.setattr(seizure_pack, "read_edf", _fake_read_edf)
        assert seizure_pack.main(["--chb", str(tmp / "chb"), "--siena", str(tmp / "siena"), "--out", str(out),
                                  "--workers", "1"]) == 0
    return tmp, out


def test_pack_shards(packed):
    _, out = packed
    z = np.load(out / "chbmit" / "chb01.npz", allow_pickle=False)
    assert z["X"].dtype == np.float16 and z["X"].shape[1:] == (18, 1024)
    assert len(z["y"]) == len(z["rec"]) == len(z["start_s"]) == z["X"].shape[0]
    assert set(np.unique(z["y"])) <= {-1, 0, 1} and (z["y"] == 1).sum() > 0
    assert list(z["records"]) == ["chb01_01", "chb01_02"] and len(z["sz_rec"]) == 1
    # stored windows are per-window robust z: no per-person statistics in the shard
    assert str(z["norm"]) == "per_window_robust" and not any(k.startswith("norm_m") for k in z.files)
    X = z["X"].astype(np.float32)
    assert np.abs(X).max() <= 20.0
    np.testing.assert_allclose(np.median(X, axis=2), 0.0, atol=0.02)
    np.testing.assert_allclose(1.4826 * np.median(np.abs(X), axis=2), 1.0, atol=0.02)
    # missing channel stays exactly 0; the 5-channel record is excluded
    c3 = np.load(out / "chbmit" / "chb03.npz", allow_pickle=False)
    k = list(config.BIPOLAR_NAMES).index("FZ-CZ")
    assert list(c3["records"]) == ["chb03_01", "chb03_02"] and not c3["rec_present"][1, k]
    assert np.all(c3["X"][c3["rec"] == 1, k] == 0) and np.abs(c3["X"][c3["rec"] == 0, k]).max() > 1.0
    s = np.load(out / "siena" / "PN00.npz", allow_pickle=False)
    assert s["X"].shape[1:] == (18, 1024) and float(s["fs"]) == 256.0
    np.testing.assert_allclose(np.c_[s["sz_start_s"], s["sz_end_s"]], [[40, 70], [150, 190]])
    ict = s["start_s"][s["y"] == 1]
    assert ict.min() >= 36.0 and ict.max() <= 190.0 and ((ict > 70) & (ict < 146)).sum() == 0
    summ = json.loads((out / "pack_summary.json").read_text())
    assert summ["chbmit"]["cases"] == 3 and summ["siena"]["seizures"] == 2
    assert summ["chbmit"]["excluded_records"] == 1


def test_per_window_robust_matches_api():
    """Same numbers as the API adapter's per_window_robust (evade/models/deep_tasks.py SeizureAdapter.predict)."""
    rng = np.random.default_rng(3)
    w = rng.normal(0, 40, (5, 18, 1024)) + rng.standard_t(2, (5, 18, 1024)) * 30
    w[1, 4] = 0.0                                                  # a zero-filled (missing) channel
    med = np.median(w, axis=2, keepdims=True)                      # the API's lines, verbatim
    api = np.clip((w - med) / np.maximum(1.4826 * np.median(np.abs(w - med), axis=2, keepdims=True), 1e-6), -20, 20)
    np.testing.assert_allclose(seizure_pack.per_window_robust(w), api, atol=1e-4)
    assert np.all(seizure_pack.per_window_robust(w)[1, 4] == 0)


def test_resumed_pack_replays_log_and_rebuilds_stale(packed, capsys):
    tmp, out = packed
    replay = tmp / "replay"
    (replay / "chbmit").mkdir(parents=True)
    shutil.copy(out / "chbmit" / "chb03.npz", replay / "chbmit" / "chb03.npz")
    with pytest.MonkeyPatch.context() as mp:
        if not HAVE_EDF:
            mp.setattr(seizure_pack, "read_edf", _fake_read_edf)
        mp.setattr(seizure_pack, "process_record", None)                 # a finished shard must not be re-read
        assert seizure_pack.main(["--chb", str(tmp / "chb"), "--out", str(replay), "--workers", "1",
                                  "--cases", "chb03"]) == 0
        assert "EXCLUDED chb03_03.edf" in capsys.readouterr().out          # replayed from the finished shard
        summ = json.loads((replay / "pack_summary.json").read_text())
        mp.undo()
        if not HAVE_EDF:
            mp.setattr(seizure_pack, "read_edf", _fake_read_edf)
        assert summ["chbmit"]["excluded_records"] == 1
        # a shard from the old normalization (no norm field) is rebuilt, never reused
        stale = out / "stale"
        (stale / "chbmit").mkdir(parents=True)
        np.savez(stale / "chbmit" / "chb02.npz", y=np.zeros(1, np.int8))
        assert seizure_pack.main(["--chb", str(tmp / "chb"), "--out", str(stale), "--workers", "1",
                                  "--cases", "chb02"]) == 0
    z = np.load(stale / "chbmit" / "chb02.npz", allow_pickle=False)
    assert str(z["norm"]) == "per_window_robust" and len(z["y"]) > 1


def test_download_selection_matches_pack_rule(packed):
    from evade.training import data as beta_data
    from evade.training.deep import seizure_select
    tmp, _ = packed
    root = tmp / "chb"
    lines = [f"{'0' * 64}  {p.relative_to(root).as_posix()}" for p in sorted(root.rglob("*")) if p.is_file()]
    for max_free in (0, 1, -1):
        got = [ln.split()[1] for ln in seizure_select.select("chbmit", root, lines, max_free) if ln.endswith(".edf")]
        want = [f"{case}/{Path(edf).name}" for edf, _, case in beta_data.chb_jobs(root, 10 ** 6 if max_free < 0
                                                                                   else max_free)]
        assert got == want
    sroot = tmp / "siena"
    slines = [f"{'0' * 64}  {p.relative_to(sroot).as_posix()}" for p in sorted(sroot.rglob("*")) if p.is_file()]
    got = [ln.split()[1] for ln in seizure_select.select("siena", sroot, slines, 0)]
    want = [f"{s}/{Path(edf).name}" for edf, _, s in beta_data.siena_jobs(sroot, 0)]
    assert [g for g in got if g.endswith(".edf")] == want and "PN00/Seizures-list-PN00.txt" in got
    with pytest.raises(FileNotFoundError):                 # a missing summary is an error, never "no seizures"
        seizure_select.select("chbmit", tmp / "nowhere", lines, 3)


def test_sigusr1_on_last_step_validates_then_resumes(packed):
    """1-epoch run (the smoke setting): a signal on the last step validates, saves best.pt, exits 99; resume ends."""
    tmp, out = packed
    run = tmp / "laststep"
    args = ["--data", str(out), "--arch", "seizure_tiny", "--epochs", "1", "--epoch-windows", "96", "--bs", "32",
            "--n-boot", "10", "--inner-val", "1", "--cases", "chb01,chb02,chb03", "--protocol", "loso",
            "--run", str(run), "--resume"]
    real_augment = seizure_train.augment
    calls = {"n": 0}

    def augment_signal_on_third(*a):                     # 96 // 32 = 3 steps: signal on the last one
        calls["n"] += 1
        if calls["n"] == 3:
            seizure_train.STOP["flag"] = True
        return real_augment(*a)

    try:
        with pytest.MonkeyPatch.context() as mp, pytest.raises(SystemExit) as e:
            mp.setattr(seizure_train, "augment", augment_signal_on_third)
            seizure_train.main(args)
    finally:
        seizure_train.STOP["flag"] = False
    assert e.value.code == 99
    fd = run / "fold_chb01"
    ck = torch.load(fd / "last.pt", weights_only=True)
    assert (ck["epoch"], ck["step"]) == (0, 3) and len(ck["hist"]) == 1 and (fd / "best.pt").exists()
    assert seizure_train.main(args) == 0
    assert json.loads((fd / "result.json").read_text())["epochs_run"] == 1


def test_end_to_end_train_resume_export(packed):
    tmp, out = packed
    common = ["--data", str(out), "--arch", "seizure_tiny", "--epochs", "2", "--epoch-windows", "96", "--bs", "32",
              "--n-boot", "20", "--inner-val", "1"]
    loso = tmp / "loso"
    real_augment = seizure_train.augment

    def augment_then_signal(*args):                      # fake SIGUSR1 during the first training step
        seizure_train.STOP["flag"] = True
        return real_augment(*args)

    try:
        with pytest.MonkeyPatch.context() as mp, pytest.raises(SystemExit) as e:
            mp.setattr(seizure_train, "augment", augment_then_signal)
            seizure_train.main(common + ["--protocol", "loso", "--run", str(loso), "--resume"])
    finally:
        seizure_train.STOP["flag"] = False
    assert e.value.code == 99
    ck = torch.load(loso / "fold_chb01" / "last.pt", weights_only=True)
    assert (ck["epoch"], ck["step"]) == (0, 1)           # mid-epoch checkpoint: resumes at step 2 of epoch 1
    assert seizure_train.main(common + ["--protocol", "loso", "--run", str(loso), "--resume"]) == 0
    m = json.loads((loso / "metrics.json").read_text())
    oof = m["chbmit_oof"]
    assert [f["test_subject"] for f in m["folds"]] == ["chb01", "chb02", "chb03"]
    assert all(len(f["val_subjects"]) == 1 and f["test_subject"] not in f["val_subjects"] for f in m["folds"])
    assert 0.0 <= oof["pooled_auroc"] <= 1.0 and "lo" in oof["pooled_auroc_ci95"]
    assert [op["target_fpr"] for op in oof["operating_points"]] == [0.01, 0.05]
    assert oof["operating_points"][0]["events_total"] == 3
    assert m["baseline_beta_eva2"]["pooled_auroc"] == 0.967
    assert not (loso / "fold_chb01" / "last.pt").exists()

    final = tmp / "final"
    assert seizure_train.main(common + ["--protocol", "final", "--external", "siena", "--thresholds-from",
                                        str(loso / "metrics.json"), "--run", str(final), "--resume"]) == 0
    fm = json.loads((final / "metrics.json").read_text())
    ext = fm["external_siena"]["all_subjects"]["at_chbmit_thresholds"]
    assert ext["operating_points"][0]["threshold"] == oof["operating_points"][0]["threshold"]
    assert ext["n_seizures"] == 2 and "siena6_beta_subjects" in fm["external_siena"]

    exp = tmp / "export" / "seizure"
    assert seizure_export.main(["--run", str(final), "--out", str(exp)]) == 0
    man = json.loads((exp / "manifest.json").read_text())
    assert man["format"] in ("onnx", "torchscript") and man["task"] == "seizure_deep"
    assert man["input"]["tensor_shape"] == ["batch", 18, 1024] and man["input"]["fs_hz"] == 256.0
    assert man["input"]["normalization"]["type"] == "per_window_robust" and man["input"]["smooth_windows"] == 3
    ops = man["output"]["operating_points"]
    assert [(o["name"], o["target_fpr"]) for o in ops] == [("review", 0.05), ("suspect_seizure", 0.01)]
    assert 0.0 <= ops[0]["threshold"] <= ops[1]["threshold"] <= 1.0
    assert "OOF" in ops[0]["estimate"] and m["per_fold_thresholds"]["spread"]["0.01"]["min"] <= 1.0
    for f, h in man["files_sha256"].items():
        assert seizure_export.sha256(exp / f) == h
    assert (exp / "SHA256SUMS").read_text().count("\n") == 4
    if man["format"] == "onnx" and "ok" in (man["onnx_parity"] or {}):
        assert man["onnx_parity"]["ok"]


def test_smoke_flag_runs(packed):
    tmp, out = packed
    run = tmp / "smoke"
    assert seizure_train.main(["--data", str(out), "--arch", "seizure_tiny", "--bs", "16", "--protocol", "loso",
                               "--smoke", "--run", str(run)]) == 0
    m = json.loads((run / "metrics.json").read_text())
    assert m["smoke"] and m["chbmit_oof"]["bootstrap"]["n_boot"] == 50


def test_weighted_metrics_match_sklearn():
    from sklearn.metrics import average_precision_score, roc_auc_score
    rng = np.random.default_rng(1)
    y = rng.integers(0, 2, 500)
    p = np.round(rng.random(500) + 0.3 * y, 2)            # rounded: ties
    w = rng.integers(0, 3, 500).astype(float)
    assert E.weighted_auroc(y, p) == pytest.approx(roc_auc_score(y, p))
    assert E.weighted_auroc(y, p, w) == pytest.approx(roc_auc_score(y, p, sample_weight=w))
    assert E.weighted_ap(y, p) == pytest.approx(average_precision_score(y, p))
    assert E.weighted_ap(y, p, w) == pytest.approx(average_precision_score(y, p, sample_weight=w))


def test_events_and_false_alarms():
    start = np.arange(0, 100, 2.0)
    rec = np.zeros(len(start), int)
    y = np.zeros(len(start), int)
    y[(start >= 40) & (start < 50)] = 1
    ev = E.events_table(np.array(["a"] * len(start)), rec, start, ["a"], [0], [41.0], [51.0])
    assert set(start[ev[0][1]]) == set(np.arange(38, 51, 2.0))     # windows overlapping [41, 51)
    flag = np.zeros(len(start), bool)
    flag[[2, 3, 4, 30]] = True                                     # one run, then a lone flag far away
    assert E.false_alarms(flag, y, rec, start, refractory_s=30.0) == 2
    thr = M.threshold_at_fpr(y, np.where(flag, 1.0, 0.0), 0.01)
    assert thr == 1.0      # tie-safe: 4 of 45 negatives at 1.0 > 1% FPR, so nothing is flagged
