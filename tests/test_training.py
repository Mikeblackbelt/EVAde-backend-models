"""Unit tests for the beta training pieces, on synthetic data (no datasets needed)."""
import numpy as np
import pytest

pytest.importorskip("sklearn")

from evade.training import metrics as M
from evade.training.features import EEG_BANDS, ecg_features, spectral_features
from evade.training.train import fit, make_hgb, make_lr


def _sine(freq, fs=250.0, n_win=6, ch=3, secs=4.0, seed=0):
    t = np.arange(int(fs * secs)) / fs
    rng = np.random.default_rng(seed)
    return np.stack([[np.sin(2 * np.pi * freq * t) + 0.05 * rng.standard_normal(t.size) for _ in range(ch)]
                     for _ in range(n_win)])


def test_feature_shape_and_names():
    w = _sine(10.0)
    f, names = spectral_features(w, 250.0, EEG_BANDS, ["a", "b", "c"])
    assert f.shape == (6, len(names)) and np.isfinite(f).all()
    assert names[0] == "a_logp_delta"


def test_alpha_band_dominates_for_10hz():
    w = _sine(10.0)
    f, names = spectral_features(w, 250.0, EEG_BANDS, ["a", "b", "c"])
    row = dict(zip(names, f[0]))
    assert row["a_rel_alpha"] > 0.8


def test_feature_separates_classes_and_model_learns():
    x = np.concatenate([_sine(10.0, n_win=30, seed=1), _sine(22.0, n_win=30, seed=2)])   # HGB needs >= 20 samples per leaf
    y = np.array([0] * 30 + [1] * 30)
    f, _ = spectral_features(x, 250.0, EEG_BANDS, ["a", "b", "c"])
    for mk in (make_hgb, make_lr):
        m = fit(mk(), f, y)
        assert (m.predict(f) == y).mean() == 1.0


def test_threshold_and_sensitivity():
    y = np.array([0] * 100 + [1] * 10)
    p = np.concatenate([np.linspace(0, 0.5, 100), np.linspace(0.6, 1.0, 10)])
    thr = M.threshold_at_fpr(y, p, 0.01)
    sens, fpr = M.sens_at(y, p, thr)
    assert fpr <= 0.011 and sens == 1.0


def test_auroc_nan_when_one_class():
    assert np.isnan(M.safe_auroc([0, 0, 0], [0.1, 0.2, 0.3]))


def test_bootstrap_ci_contains_mean():
    ci = M.subject_bootstrap_ci([0.7, 0.8, 0.9, 0.85])
    assert ci["lo"] <= ci["mean"] <= ci["hi"] and ci["n_subjects"] == 4


def test_ecg_features_shape():
    rng = np.random.default_rng(0)
    w = rng.standard_normal((4, 12, 1000))
    f, names = ecg_features(w, 100.0, [f"L{i}" for i in range(12)])
    assert f.shape == (4, len(names))
