"""Seizure-suspect metrics from saved window scores: window AUROC / AP, operating points, false-positive windows
per hour, event-level sensitivity, false alarms per hour, and subject-level bootstrap CIs. Pure numpy.

Conventions (same as the beta, evade/training/train.py):
- scores are smoothed with a centered 3-window moving average inside each recording before scoring;
- label -1 windows (buffers around seizures) are left out of window metrics but kept for smoothing and events;
- a threshold for a target FPR comes from evade.training.metrics.threshold_at_fpr (tie-safe "higher" quantile of
  the negative scores; flag = score > threshold), computed on pooled out-of-fold CHB-MIT scores and applied
  unchanged to every test set, Siena included;
- false-positive windows per hour = flagged non-ictal windows / (non-ictal windows x 2 s step / 3600). Windows
  overlap by 2 s, so one minute of false activity is about 30 windows;
- an event (seizure) counts as detected if any window overlapping it (any label) is flagged;
- a false alarm is a run of flagged non-ictal windows in one recording; flags within `refractory_s` of the previous
  one join the same alarm.
CIs resample subjects with replacement (the unit of generalization); thresholds stay fixed inside the bootstrap.
"""
from __future__ import annotations

import numpy as np

from .. import metrics as M
from ..train import smooth_by_record

STEP_S = 2.0
WIN_S = 4.0


def weighted_auroc(y: np.ndarray, p: np.ndarray, w: np.ndarray | None = None) -> float:
    """Mann-Whitney AUROC with sample weights (ties count one half)."""
    y = np.asarray(y); p = np.asarray(p, float)
    w = np.ones(len(y)) if w is None else np.asarray(w, float)
    u, inv = np.unique(p, return_inverse=True)
    gp = np.bincount(inv, w * (y == 1), len(u))
    gn = np.bincount(inv, w * (y == 0), len(u))
    wp, wn = gp.sum(), gn.sum()
    if wp == 0 or wn == 0:
        return float("nan")
    below = np.cumsum(gn) - gn
    return float((gp * (below + 0.5 * gn)).sum() / (wp * wn))


def weighted_ap(y: np.ndarray, p: np.ndarray, w: np.ndarray | None = None) -> float:
    """Average precision (step-wise, as sklearn) with sample weights, ties grouped."""
    y = np.asarray(y); p = np.asarray(p, float)
    w = np.ones(len(y)) if w is None else np.asarray(w, float)
    u, inv = np.unique(-p, return_inverse=True)           # descending score groups
    gp = np.bincount(inv, w * (y == 1), len(u))
    gn = np.bincount(inv, w * (y == 0), len(u))
    if gp.sum() == 0:
        return float("nan")
    tp, fp = np.cumsum(gp), np.cumsum(gn)
    prec = tp / np.maximum(tp + fp, 1e-12)
    return float((gp * prec).sum() / gp.sum())


def subject_multiplicity(subjects: np.ndarray, n_boot: int, seed: int):
    """Yield, per bootstrap resample, a per-window weight = how often its subject was drawn."""
    uniq, inv = np.unique(subjects, return_inverse=True)
    rng = np.random.default_rng(seed)
    for _ in range(n_boot):
        cnt = np.bincount(rng.integers(0, len(uniq), len(uniq)), minlength=len(uniq)).astype(float)
        yield cnt, cnt[inv]


def _ci(vals) -> dict:
    v = np.asarray([x for x in vals if np.isfinite(x)], float)
    if len(v) == 0:
        return {"lo": float("nan"), "hi": float("nan")}
    return {"lo": float(np.percentile(v, 2.5)), "hi": float(np.percentile(v, 97.5))}


def events_table(subject, rec_key, start_s, sz_subject, sz_rec_key, sz_start, sz_end):
    """For every seizure, the indices of the windows that overlap it (window [s, s+4) vs seizure [a, b))."""
    out = []
    order = {}
    for i, k in enumerate(rec_key):
        order.setdefault(k, []).append(i)
    order = {k: np.asarray(v) for k, v in order.items()}
    for s, k, a, b in zip(sz_subject, sz_rec_key, sz_start, sz_end):
        idx = order.get(k, np.empty(0, int))
        if len(idx):
            st = start_s[idx]
            idx = idx[(st < b) & (st + WIN_S > a)]
        out.append((s, idx))
    return out


def false_alarms(flag: np.ndarray, y: np.ndarray, rec_key: np.ndarray, start_s: np.ndarray,
                 refractory_s: float = 30.0) -> int:
    """Number of false alarms: runs of flagged label-0 windows per recording, merged within refractory_s."""
    n = 0
    sel = np.flatnonzero(flag & (y == 0))
    if len(sel) == 0:
        return 0
    keys, t = rec_key[sel], start_s[sel]
    o = np.lexsort((t, keys))
    keys, t = keys[o], t[o]
    new = np.ones(len(t), bool)
    new[1:] = (keys[1:] != keys[:-1]) | (t[1:] - t[:-1] > refractory_s)
    n = int(new.sum())
    return n


def evaluate(scores: np.ndarray, y: np.ndarray, subject: np.ndarray, rec_key: np.ndarray, start_s: np.ndarray,
             seizures: dict, thresholds: dict | None = None, fprs=(0.01, 0.05), n_boot: int = 2000, seed: int = 0,
             smooth: int = 3, refractory_s: float = 30.0) -> dict:
    """Full metric block for one set of window scores.

    seizures: {"subject": [...], "rec_key": [...], "start_s": [...], "end_s": [...]}.
    thresholds: {"0.01": t, ...} to apply fixed thresholds (external test); None derives them from these scores.
    """
    y = np.asarray(y); subject = np.asarray(subject); rec_key = np.asarray(rec_key)
    start_s = np.asarray(start_s, float); raw = np.asarray(scores, float)
    ps = smooth_by_record(raw, rec_key, start_s, smooth) if smooth > 1 else raw.copy()
    m = y >= 0
    yw, pw, sw = y[m], ps[m], subject[m]
    subs = sorted(set(sw))
    res = {"n_windows": int(m.sum()), "n_ictal": int((yw == 1).sum()), "n_subjects": len(subs),
           "subjects": subs, "smoothing_windows": smooth,
           "n_seizures": int(len(seizures["subject"])),
           "nonictal_hours": float((yw == 0).sum() * STEP_S / 3600.0)}
    res["pooled_auroc"] = weighted_auroc(yw, pw)
    res["pooled_ap"] = weighted_ap(yw, pw)
    res["pooled_auroc_unsmoothed"] = weighted_auroc(yw, raw[m])
    res["pooled_ap_unsmoothed"] = weighted_ap(yw, raw[m])
    per = {}
    for s in subs:
        t = sw == s
        per[s] = {"auroc": M.safe_auroc(yw[t], pw[t]), "ap": M.safe_ap(yw[t], pw[t]),
                  "n_windows": int(t.sum()), "n_ictal": int((yw[t] == 1).sum())}
    ev = events_table(subject, rec_key, start_s, seizures["subject"], seizures["rec_key"], seizures["start_s"],
                      seizures["end_s"])
    if thresholds is None:
        thresholds = {f"{f:g}": M.threshold_at_fpr(yw, pw, f) for f in fprs}
    ops = []
    boot_ops = []
    for key, thr in thresholds.items():
        flag_all = ps > thr
        flag = flag_all[m]
        op = {"target_fpr": float(key), "threshold": float(thr)}
        sens_p, fpr_p = M.sens_at(yw, pw, thr)
        op["sensitivity_pooled"], op["fpr_pooled"] = sens_p, fpr_p
        op["fp_windows_per_hour_pooled"] = fpr_p * 3600.0 / STEP_S if np.isfinite(fpr_p) else float("nan")
        sens_s, fp_s = {}, {}
        for s in subs:
            t = sw == s
            se, fr = M.sens_at(yw[t], pw[t], thr)
            sens_s[s], fp_s[s] = se, fr * 3600.0 / STEP_S if np.isfinite(fr) else float("nan")
            per[s][f"sensitivity_at_{key}"] = se
            per[s][f"fp_windows_per_hour_at_{key}"] = fp_s[s]
        op["sensitivity_subject_mean"] = M.subject_bootstrap_ci([v for v in sens_s.values()], n_boot, seed)
        fin = [v for v in fp_s.values() if np.isfinite(v)]
        op["fp_windows_per_hour_subject_mean"] = float(np.mean(fin)) if fin else float("nan")
        det = {s: [0, 0] for s in set(seizures["subject"])}
        for s, idx in ev:
            det[s][1] += 1
            det[s][0] += int(len(idx) > 0 and flag_all[idx].any())
        n_det, n_ev = sum(v[0] for v in det.values()), sum(v[1] for v in det.values())
        op["event_sensitivity"] = n_det / n_ev if n_ev else float("nan")
        op["events_detected"], op["events_total"] = n_det, n_ev
        for s in subs:
            if s in det:
                per[s][f"events_detected_at_{key}"] = det[s][0]; per[s]["events_total"] = det[s][1]
        fa = false_alarms(flag, yw, rec_key[m], start_s[m], refractory_s)
        op["false_alarms_per_hour"] = fa / res["nonictal_hours"] if res["nonictal_hours"] else float("nan")
        op["refractory_s"] = refractory_s
        ops.append(op)
        boot_ops.append((op, flag, det))
    res["operating_points"] = ops
    res["per_subject"] = per
    res["subject_auroc_mean"] = M.subject_bootstrap_ci([v["auroc"] for v in per.values()], n_boot, seed)
    res["subject_ap_mean"] = M.subject_bootstrap_ci([v["ap"] for v in per.values()], n_boot, seed)

    # subject bootstrap of the pooled metrics (same resamples for every metric)
    auc_b, ap_b = [], []
    op_b = [{"sens": [], "fph": [], "ev": []} for _ in ops]
    sub_list = np.unique(sw)
    for cnt, w in subject_multiplicity(sw, n_boot, seed):
        auc_b.append(weighted_auroc(yw, pw, w)); ap_b.append(weighted_ap(yw, pw, w))
        c_of = dict(zip(sub_list, cnt))
        for j, (op, flag, det) in enumerate(boot_ops):
            wp, wn = (w * (yw == 1)).sum(), (w * (yw == 0)).sum()
            op_b[j]["sens"].append((w * (yw == 1) * flag).sum() / wp if wp else np.nan)
            op_b[j]["fph"].append((w * (yw == 0) * flag).sum() / wn * 3600.0 / STEP_S if wn else np.nan)
            nd = sum(c_of.get(s, 0) * v[0] for s, v in det.items())
            nt = sum(c_of.get(s, 0) * v[1] for s, v in det.items())
            op_b[j]["ev"].append(nd / nt if nt else np.nan)
    res["pooled_auroc_ci95"] = _ci(auc_b)
    res["pooled_ap_ci95"] = _ci(ap_b)
    for op, b in zip(ops, op_b):
        op["sensitivity_pooled_ci95"] = _ci(b["sens"])
        op["fp_windows_per_hour_pooled_ci95"] = _ci(b["fph"])
        op["event_sensitivity_ci95"] = _ci(b["ev"])
    res["bootstrap"] = {"unit": "subject", "n_boot": n_boot, "seed": seed,
                        "note": "thresholds fixed; subjects resampled with replacement"}
    return res
