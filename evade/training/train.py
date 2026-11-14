"""Beta training: three CPU models with subject-wise evaluation.

  python -m evade.training.train --task seizure --chb <root> --siena <root> --out <dir> [--smoke]
  python -m evade.training.train --task sleep   --sleep <root> --out <dir> [--smoke]
  python -m evade.training.train --task ecg     --ptbxl <root> --out <dir> [--smoke]

Each task writes <out>/<task>/metrics.json, model.joblib, feature_names.json and a features parquet cache.
Design: gradient boosting on band-power features, with a logistic-regression baseline in the same protocol.
Beta only: not tuned, not externally validated beyond what is stated in metrics.json.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

META = {"participant_id", "record_id", "window_start_s", "quality_index", "label", "dataset", "strat_fold"}


def feat_cols(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in META and not c.startswith("y_")]


def make_hgb(seed: int = 0):
    from sklearn.ensemble import HistGradientBoostingClassifier
    return HistGradientBoostingClassifier(max_iter=150, learning_rate=0.1, max_depth=6, l2_regularization=1.0,
                                          early_stopping=False, random_state=seed)


def make_lr():
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    return make_pipeline(StandardScaler(), LogisticRegression(max_iter=300, C=0.1, class_weight="balanced"))


def fit(model, X, y, balance: bool = True):
    from sklearn.utils.class_weight import compute_sample_weight
    if hasattr(model, "steps"):          # pipeline: LR handles class_weight itself
        model.fit(X, y)
    else:
        model.fit(X, y, sample_weight=compute_sample_weight("balanced", y) if balance else None)
    return model


def _write(out: Path, task: str, metrics: dict, model, cols: list[str]) -> None:
    import joblib
    d = out / task
    d.mkdir(parents=True, exist_ok=True)
    (d / "metrics.json").write_text(json.dumps(metrics, indent=2, default=float))
    (d / "feature_names.json").write_text(json.dumps(cols))
    joblib.dump(model, d / "model.joblib")     # our own artifact; never load a joblib file you did not build


# ---------------- seizure ----------------
def _safe(fn, *args):
    """Run a per-file feature job; return None (and say why) if the file is unreadable, e.g. a partial download."""
    try:
        return fn(*args)
    except Exception as e:                      # noqa: BLE001
        print(f"SKIPPED {args[0]}: {type(e).__name__}: {e}", flush=True)
        return None


def run_seizure(a) -> dict:
    from joblib import Parallel, delayed
    from . import data, metrics as M
    out = Path(a.out); d = out / "seizure"; d.mkdir(parents=True, exist_ok=True)
    cache_c, cache_s = d / "features_chbmit.parquet", d / "features_siena.parquet"
    t0 = time.time()
    if not cache_c.exists():
        jobs = data.chb_jobs(Path(a.chb), a.max_free)[: a.file_limit]
        parts = Parallel(n_jobs=a.threads)(delayed(_safe)(data.chb_file_features, e, s, sub) for e, s, sub in jobs)
        pd.concat([x for x in parts if x is not None], ignore_index=True).to_parquet(cache_c)
    if a.siena and not cache_s.exists():
        jobs = data.siena_jobs(Path(a.siena), a.max_free)[: a.file_limit]
        parts = Parallel(n_jobs=a.threads)(delayed(_safe)(data.siena_file_features, e, t, sub) for e, t, sub in jobs)
        pd.concat([x for x in parts if x is not None], ignore_index=True).to_parquet(cache_s)
    chb = pd.read_parquet(cache_c)
    chb = chb[chb.label >= 0].reset_index(drop=True)
    cols = feat_cols(chb)
    X, y, g = chb[cols].to_numpy(np.float32), chb.label.to_numpy(), chb.participant_id.to_numpy()
    res = {"task": "seizure", "model": "HistGradientBoosting on band-power and line-length features",
           "n_windows": int(len(chb)), "n_ictal": int(y.sum()), "subjects": sorted(set(g)),
           "window_s": 4.0, "step_s": 2.0, "feature_time_s": round(time.time() - t0, 1)}
    subs = [s for s in sorted(set(g)) if y[g == s].sum() > 0 and (y[g == s] == 0).sum() > 0]
    out_cv = {}
    for name, mk in (("hgb", make_hgb), ("logreg_baseline", make_lr)):
        oof = np.full(len(y), np.nan)
        for s in subs:
            tr, te = g != s, g == s
            if y[tr].sum() == 0:
                continue
            oof[te] = fit(mk(), X[tr], y[tr]).predict_proba(X[te])[:, 1]
        m = ~np.isnan(oof)
        thr = M.threshold_at_fpr(y[m], oof[m], 0.01)
        per = {}
        for s in subs:
            te = (g == s) & m
            if not te.any():
                continue
            sens, fpr = M.sens_at(y[te], oof[te], thr)
            per[s] = {"auroc": M.safe_auroc(y[te], oof[te]), "ap": M.safe_ap(y[te], oof[te]),
                      "sensitivity_at_thr": sens, "fpr_at_thr": fpr, "n_ictal": int(y[te].sum())}
        out_cv[name] = {
            "pooled_auroc": M.safe_auroc(y[m], oof[m]), "pooled_ap": M.safe_ap(y[m], oof[m]),
            "threshold_at_1pct_fpr": thr,
            "per_subject": per,
            "subject_auroc_ci": M.subject_bootstrap_ci([v["auroc"] for v in per.values()]),
            "subject_sensitivity_ci": M.subject_bootstrap_ci([v["sensitivity_at_thr"] for v in per.values()]),
            "fp_windows_per_hour_at_thr": float(np.nanmean([v["fpr_at_thr"] for v in per.values()]) * 1800.0)}
    res["leave_one_subject_out_chbmit"] = out_cv
    final = fit(make_hgb(), X, y)
    if cache_s.exists():
        sie = pd.read_parquet(cache_s)
        sie = sie[sie.label >= 0].reset_index(drop=True)
        ys, ps = sie.label.to_numpy(), final.predict_proba(sie[cols].to_numpy(np.float32))[:, 1]
        thr = out_cv["hgb"]["threshold_at_1pct_fpr"]
        sens, fpr = M.sens_at(ys, ps, thr)
        lr = fit(make_lr(), X, y).predict_proba(sie[cols].to_numpy(np.float32))[:, 1]
        res["external_siena"] = {
            "note": "trained on CHB-MIT (pediatric, bipolar montage), tested on Siena (adults, monopolar converted to bipolar)",
            "n_windows": int(len(sie)), "n_ictal": int(ys.sum()), "subjects": sorted(set(sie.participant_id)),
            "hgb": {"auroc": M.safe_auroc(ys, ps), "ap": M.safe_ap(ys, ps), "sensitivity_at_chb_thr": sens, "fpr_at_chb_thr": fpr},
            "logreg_baseline": {"auroc": M.safe_auroc(ys, lr), "ap": M.safe_ap(ys, lr)}}
    res["limits"] = ["window-level metrics, not event-based SzCORE scoring",
                     "false-positive windows per hour counts overlapping windows, not alarms",
                     "seizure-free files are subsampled per subject, so absolute false-alarm rates are optimistic or noisy",
                     "small number of subjects; intervals are wide"]
    _write(out, "seizure", res, final, cols)
    return res


# ---------------- sleep ----------------
def run_sleep(a) -> dict:
    from . import data, metrics as M
    out = Path(a.out); d = out / "sleep"; d.mkdir(parents=True, exist_ok=True)
    cache = d / "features_sleepedfx.parquet"
    if not cache.exists():
        data.sleep_features(Path(a.sleep), a.file_limit).to_parquet(cache)
    df = pd.read_parquet(cache)
    df = df[df.label >= 0].reset_index(drop=True)
    cols = feat_cols(df)
    X, y, g = df[cols].to_numpy(np.float32), df.label.to_numpy(), df.participant_id.to_numpy()
    res = {"task": "sleep_staging", "model": "HistGradientBoosting on band-power features of each 30 s epoch plus its neighbours (2 EEG channels)",
           "epochs_kept": "sleep period plus 30 min of wake on each side",
           "classes": {"0": "W", "1": "N1", "2": "N2", "3": "N3", "4": "REM"},
           "n_epochs": int(len(df)), "subjects": sorted(set(g)), "class_counts": {int(k): int(v) for k, v in zip(*np.unique(y, return_counts=True))}}
    subs = sorted(set(g))
    cv = {}
    for name, mk in (("hgb", make_hgb), ("logreg_baseline", make_lr)):
        per, preds = {}, np.full(len(y), -1)
        for s in subs:
            tr, te = g != s, g == s
            if len(set(y[tr])) < 2:
                continue
            preds[te] = fit(mk(), X[tr], y[tr]).predict(X[te])
            per[s] = M.multiclass_scores(y[te], preds[te])
        m = preds >= 0
        cv[name] = {"pooled": M.multiclass_scores(y[m], preds[m]), "per_subject": per,
                    "subject_kappa_ci": M.subject_bootstrap_ci([v["kappa"] for v in per.values()]),
                    "subject_accuracy_ci": M.subject_bootstrap_ci([v["accuracy"] for v in per.values()])}
    res["leave_one_subject_out"] = cv
    res["limits"] = ["few subjects (healthy sleep-cassette recordings, ages 25-101 in the dataset)",
                     "single dataset, no external validation", "two EEG channels only; wearable headbands differ"]
    final = fit(make_hgb(), X, y)
    _write(out, "sleep", res, final, cols)
    return res


# ---------------- ecg ----------------
def run_ecg(a) -> dict:
    from . import data, metrics as M
    out = Path(a.out); d = out / "ecg"; d.mkdir(parents=True, exist_ok=True)
    cache = d / "features_ptbxl.parquet"
    if not cache.exists():
        data.ptbxl_features(Path(a.ptbxl), a.row_limit, a.threads).to_parquet(cache)
    df = pd.read_parquet(cache)
    cols = feat_cols(df)
    tr, te = df.strat_fold <= 8, df.strat_fold == 10
    Xtr, Xte = df.loc[tr, cols].to_numpy(np.float32), df.loc[te, cols].to_numpy(np.float32)
    res = {"task": "ptbxl_superclass", "model": "one-vs-rest HistGradientBoosting on per-lead spectral and shape features",
           "split": "official PTB-XL folds: train 1-8, test 10 (fold 9 unused in beta)", "n_train": int(tr.sum()), "n_test": int(te.sum()),
           "classes": data.SUPER}
    final = {}
    for name, mk in (("hgb", make_hgb), ("logreg_baseline", make_lr)):
        aucs, probs = {}, {}
        for k in data.SUPER:
            ytr, yte = df.loc[tr, f"y_{k}"].to_numpy(), df.loc[te, f"y_{k}"].to_numpy()
            if ytr.sum() == 0 or yte.sum() == 0:
                continue
            mdl = fit(mk(), Xtr, ytr)
            probs[k] = mdl.predict_proba(Xte)[:, 1]
            aucs[k] = M.safe_auroc(yte, probs[k])
            if name == "hgb":
                final[k] = mdl
        macro = float(np.nanmean(list(aucs.values()))) if aucs else float("nan")
        # patient-level bootstrap of macro AUROC on the test fold
        pid = df.loc[te, "participant_id"].to_numpy()
        up = np.unique(pid)
        idx = {p: np.where(pid == p)[0] for p in up}
        rng = np.random.default_rng(0)
        bs = []
        for _ in range(200):
            samp = np.concatenate([idx[p] for p in rng.choice(up, len(up))])
            v = [M.safe_auroc(df.loc[te, f"y_{k}"].to_numpy()[samp], probs[k][samp]) for k in probs]
            bs.append(np.nanmean(v))
        res[name] = {"auroc_per_class": aucs, "macro_auroc": macro,
                     "macro_auroc_ci95": [float(np.nanpercentile(bs, 2.5)), float(np.nanpercentile(bs, 97.5))]}
    res["limits"] = ["hand-built spectral features, no beat-level morphology; published deep models do better",
                     "labels are PTB-XL cardiologist annotations (diagnostic superclasses), single-site data",
                     "12-lead hospital ECG, not single-lead wearable"]
    _write(out, "ecg", res, final, cols)
    return res


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--task", required=True, choices=["seizure", "sleep", "ecg"])
    p.add_argument("--out", required=True)
    p.add_argument("--chb"); p.add_argument("--siena"); p.add_argument("--sleep"); p.add_argument("--ptbxl")
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--max-free", type=int, default=3, help="seizure-free files kept per subject (seizure task)")
    p.add_argument("--smoke", action="store_true", help="tiny run to check the pipeline end to end")
    a = p.parse_args(argv)
    a.file_limit = 4 if a.smoke else None
    a.row_limit = 600 if a.smoke else None
    if a.smoke:
        a.max_free = 1
    r = {"seizure": run_seizure, "sleep": run_sleep, "ecg": run_ecg}[a.task](a)
    print(json.dumps({k: v for k, v in r.items() if k in ("task", "n_windows", "n_epochs", "n_train", "n_test")}, default=float))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
