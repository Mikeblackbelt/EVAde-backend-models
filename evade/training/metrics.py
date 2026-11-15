"""Metrics and subject-level confidence intervals for the beta models."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import (average_precision_score, cohen_kappa_score, f1_score, roc_auc_score)


def safe_auroc(y, p) -> float:
    y = np.asarray(y)
    return float("nan") if len(np.unique(y)) < 2 else float(roc_auc_score(y, p))


def safe_ap(y, p) -> float:
    y = np.asarray(y)
    return float("nan") if y.sum() == 0 else float(average_precision_score(y, p))


def threshold_at_fpr(y, p, fpr: float = 0.01) -> float:
    """Threshold t such that the fraction of negatives with score > t is at most `fpr` (flag = score > t).

    Uses a strict inequality so that ties (for example many scores at exactly 1.0) cannot push the
    false-positive rate above the target; if ties make the target unreachable, nothing above t is flagged.
    """
    neg = np.asarray(p)[np.asarray(y) == 0]
    if len(neg) == 0:
        return 0.5
    return float(np.quantile(neg, 1 - fpr, method="higher"))


def sens_at(y, p, thr: float) -> tuple[float, float]:
    """Return (sensitivity, false-positive rate) for flag = score > thr."""
    y = np.asarray(y)
    pred = np.asarray(p) > thr
    sens = float(pred[y == 1].mean()) if (y == 1).any() else float("nan")
    fpr = float(pred[y == 0].mean()) if (y == 0).any() else float("nan")
    return sens, fpr


def multiclass_scores(y, pred) -> dict:
    return {"accuracy": float((np.asarray(y) == np.asarray(pred)).mean()),
            "macro_f1": float(f1_score(y, pred, average="macro")),
            "kappa": float(cohen_kappa_score(y, pred))}


def subject_bootstrap_ci(values, n_boot: int = 2000, seed: int = 0) -> dict:
    """Mean and 95% percentile bootstrap interval over subjects (the unit of generalization)."""
    v = np.asarray([x for x in values if np.isfinite(x)], dtype=float)
    if len(v) == 0:
        return {"mean": float("nan"), "lo": float("nan"), "hi": float("nan"), "n_subjects": 0}
    rng = np.random.default_rng(seed)
    means = np.array([v[rng.integers(0, len(v), len(v))].mean() for _ in range(n_boot)])
    return {"mean": float(v.mean()), "lo": float(np.percentile(means, 2.5)), "hi": float(np.percentile(means, 97.5)),
            "n_subjects": int(len(v))}
