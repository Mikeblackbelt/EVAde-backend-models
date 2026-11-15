"""Train and evaluate the SPE onset forecast (EVA-7 arch, EVA-8 model) on historical GOES data.

Data: NOAA NCEI GOES 5-min integral protons and XRS averages (scripts/download_goes_history.sh), 1998-2019.
Task: at each quiet hour (>=10 MeV hourly flux below 10 pfu), will the hourly flux reach 10 pfu within 24 h?
Protocol A (headline, time-forward): train 1998-2010, Platt recalibration 2011-2012, test 2013-2019, 72 h embargo
around every boundary (48 h feature lookback + 24 h label horizon) so no storm straddles a split.
Model selection (logistic vs gradient boosting) uses blocked CV inside the pre-test period only.
Protocol B (robustness): leave-one-block-out over all years with the same embargo, pooled out-of-fold metrics.
Baselines: climatology (training base rate) and persistence (logistic on the current log flux only).

Run:  python -I -m evade.spaceweather.train --raw ~/evade_data/other/spaceweather/raw --out <model_dir> [--smoke]
"""
from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from .features import FEATURES, LOOKBACK_H, HORIZON_H, feature_frame, labels, onset_times, to_hourly
from .model import ARCH_ID, MODEL_ID, TASK, sha256

XRS_UNSCALE = {"long": 0.7, "short": 0.85}   # SWPC scaling on GOES-8..15 XRS; divide to get true irradiance
EMBARGO_H = LOOKBACK_H + HORIZON_H
BOOT_BLOCK_DAYS = 27                          # one solar rotation


# ---------------- parsing (files are untrusted text; parsed as CSV only) ----------------
def _data_offset(path: Path) -> int:
    with open(path, "r", encoding="ascii", errors="replace") as f:
        for i, line in enumerate(f):
            if line.startswith("data:"):
                return i + 1
            if i > 5000:
                break
    raise ValueError(f"no data: marker in {path.name}")


def parse_goes_csv(path: Path) -> pd.DataFrame:
    """One NCEI GOES CSV -> normalized columns indexed by UTC time. Fill values and flagged samples become NaN."""
    name = path.name
    df = pd.read_csv(path, skiprows=_data_offset(path), low_memory=False)
    t = pd.to_datetime(df["time_tag"], utc=True, errors="coerce")
    num = lambda c: pd.to_numeric(df[c], errors="coerce") if c in df.columns else pd.Series(np.nan, index=df.index)
    pos = lambda s: s.where(s >= 0)

    def det_mean(base: str) -> pd.Series:
        vals = []
        for side in ("E", "W"):
            v, q = num(f"{base}{side}"), num(f"{base}{side}_QUAL_FLAG")
            vals.append(pos(v).where(q.fillna(1) == 0))
        return pd.concat(vals, axis=1).mean(axis=1)          # mean of the two detectors that pass QC

    if "_eps_5m_" in name:                                   # GOES-8..12 EPS, corrected integral channels
        out = {"proton_10mev_pfu": pos(num("p3_flux_ic")), "proton_50mev_pfu": pos(num("p5_flux_ic")),
               "proton_100mev_pfu": pos(num("p7_flux_ic"))}
    elif "_epead_cpflux_5m_" in name:                        # GOES-13..15 EPEAD corrected integral, both detectors
        out = {"proton_10mev_pfu": det_mean("ZPGT10"), "proton_50mev_pfu": det_mean("ZPGT50"),
               "proton_100mev_pfu": det_mean("ZPGT100")}
    elif "_xrs_5m_" in name:                                 # GOES-8..12 XRS
        xl, xs = num("xl"), num("xs")
        out = {"xray_long_wm2": xl.where(xl > 0) / XRS_UNSCALE["long"], "xray_short_wm2": xs.where(xs > 0) / XRS_UNSCALE["short"]}
    elif "_xrs_1m_" in name:                                 # GOES-13..15 XRS
        b, a = num("B_AVG"), num("A_AVG")
        b = b.where((b > 0) & (num("B_QUAL_FLAG").fillna(1) == 0))
        a = a.where((a > 0) & (num("A_QUAL_FLAG").fillna(1) == 0))
        out = {"xray_long_wm2": b / XRS_UNSCALE["long"], "xray_short_wm2": a / XRS_UNSCALE["short"]}
    else:
        raise ValueError(f"unknown file type {name}")
    res = pd.DataFrame(out)
    res.index = t
    res = res[res.index.notna()]
    return res[~res.index.duplicated(keep="last")].sort_index()


def build_hourly(raw: Path, years: tuple[int, int]) -> tuple[pd.DataFrame, dict]:
    """Hourly table for every downloaded month. X-rays: per month the satellite with most valid hours, gaps filled
    from the others. Protons: the one satellite the downloader picked."""
    parts, prov = [], {}
    for mdir in sorted(raw.glob("*/*")):
        try:
            y = int(mdir.parent.name)
        except ValueError:
            continue
        if not (years[0] <= y <= years[1]) or not mdir.is_dir():
            continue
        pfiles = sorted(list(mdir.glob("*_eps_5m_*.csv")) + list(mdir.glob("*_epead_cpflux_5m_*.csv")))
        xfiles = sorted(list(mdir.glob("*_xrs_5m_*.csv")) + list(mdir.glob("*_xrs_1m_*.csv")))
        if not pfiles:
            continue
        hp = to_hourly(parse_goes_csv(pfiles[0]))[["p10", "p50", "p100"]]
        xs = []
        for xf in xfiles:
            try:
                hx = to_hourly(parse_goes_csv(xf))[["xmax", "xmean"]]
                xs.append((int(hx["xmax"].notna().sum()), xf.name, hx))
            except Exception as e:                         # a bad month file is skipped, not fatal
                print(f"skip {xf.name}: {type(e).__name__}", flush=True)
        xs.sort(key=lambda r: -r[0])
        hx = None
        for _, _, part in xs:
            hx = part if hx is None else hx.combine_first(part)
        h = hp.join(hx, how="outer") if hx is not None else hp.assign(xmax=np.nan, xmean=np.nan)
        parts.append(h)
        prov[f"{mdir.parent.name}-{mdir.name}"] = {"protons": pfiles[0].name, "xrs": [r[1] for r in xs]}
    if not parts:
        raise SystemExit(f"no GOES files under {raw}")
    h = pd.concat(parts).sort_index()
    h = h[~h.index.duplicated(keep="first")]
    h = h.reindex(pd.date_range(h.index.min(), h.index.max(), freq="1h", tz="UTC"))
    return h, prov


# ---------------- models ----------------
def make_lr():
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), LogisticRegression(C=1.0, max_iter=5000))


def make_hgb():
    from sklearn.ensemble import HistGradientBoostingClassifier
    return HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_leaf_nodes=15, min_samples_leaf=200,
                                          l2_regularization=1.0, early_stopping=False, random_state=0)


def make_persistence():
    from sklearn.linear_model import LogisticRegression
    return LogisticRegression(max_iter=1000)


MAKERS = {"logistic": make_lr, "hgb": make_hgb}


def logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def fit_platt(p, y) -> list[float]:
    from sklearn.linear_model import LogisticRegression
    m = LogisticRegression(C=1e6, max_iter=1000).fit(logit(p)[:, None], y)
    return [float(m.coef_[0, 0]), float(m.intercept_[0])]


def apply_platt(p, ab):
    return 1 / (1 + np.exp(-(ab[0] * logit(p) + ab[1])))


# ---------------- metrics ----------------
def metrics(y, p, clim: float) -> dict:
    from sklearn.metrics import average_precision_score, roc_auc_score
    y, p = np.asarray(y), np.asarray(p)
    b = float(np.mean((p - y) ** 2))
    bc = float(np.mean((clim - y) ** 2))
    two = len(np.unique(y)) == 2
    return {"n": int(len(y)), "positives": int(y.sum()), "brier": round(b, 5), "brier_climatology": round(bc, 5),
            "bss": round(1 - b / bc, 4) if bc > 0 else None,
            "auroc": round(float(roc_auc_score(y, p)), 4) if two else None,
            "ap": round(float(average_precision_score(y, p)), 4) if two else None}


def reliability(y, p, edges=(0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0001)) -> list[dict]:
    y, p = np.asarray(y), np.asarray(p)
    out = []
    for lo, hi in zip(edges, edges[1:]):
        m = (p >= lo) & (p < hi)
        if m.sum():
            out.append({"bin": [lo, round(min(hi, 1.0), 3)], "n": int(m.sum()), "mean_p": round(float(p[m].mean()), 4),
                        "observed": round(float(y[m].mean()), 4)})
    return out


def block_bootstrap(t: pd.DatetimeIndex, y, preds: dict, clim: float, reps: int = 500, seed: int = 0) -> dict:
    """95% CIs from resampling 27-day blocks (keeps each storm's hours together), incl. model-minus-baseline Brier."""
    from sklearn.metrics import roc_auc_score
    rng = np.random.default_rng(seed)
    y = np.asarray(y)
    blk = ((t - t.min()) / pd.Timedelta(days=BOOT_BLOCK_DAYS)).astype(int).to_numpy()
    ub = np.unique(blk)
    idx = {b: np.flatnonzero(blk == b) for b in ub}
    acc = {k: {"brier": [], "auroc": [], "bss": []} for k in preds}
    acc["diff_brier_model_minus_persistence"] = {"v": []}
    for _ in range(reps):
        ii = np.concatenate([idx[b] for b in rng.choice(ub, size=len(ub), replace=True)])
        yy = y[ii]
        bc = np.mean((clim - yy) ** 2)
        br = {}
        for k, p in preds.items():
            pp = np.asarray(p)[ii]
            br[k] = np.mean((pp - yy) ** 2)
            acc[k]["brier"].append(br[k])
            acc[k]["bss"].append(1 - br[k] / bc if bc > 0 else np.nan)
            if len(np.unique(yy)) == 2:
                acc[k]["auroc"].append(roc_auc_score(yy, pp))
        if "model" in br and "persistence" in br:
            acc["diff_brier_model_minus_persistence"]["v"].append(br["model"] - br["persistence"])
    ci = lambda v: [round(float(np.nanpercentile(v, 2.5)), 5), round(float(np.nanpercentile(v, 97.5)), 5)] if len(v) else None
    return {k: {m: ci(v) for m, v in d.items()} for k, d in acc.items()} | {"reps": reps, "block_days": BOOT_BLOCK_DAYS}


def thresholds_table(t, y, p, onsets, thr=(0.1, 0.25, 0.5)) -> list[dict]:
    """Hourly hit and false-alarm rates, plus the share of independent onsets with p >= thr in the prior 24 h."""
    y, p = np.asarray(y), np.asarray(p)
    s = pd.Series(p, index=t)
    rows = []
    for th in thr:
        hit = p >= th
        tp, fp = int((hit & (y == 1)).sum()), int((hit & (y == 0)).sum())
        ev = [bool((s[(s.index >= o - pd.Timedelta(hours=HORIZON_H)) & (s.index < o)] >= th).any()) for o in onsets]
        quiet_days = max((y == 0).sum() / 24, 1)
        rows.append({"threshold": th, "hourly_tpr": round(tp / max(int(y.sum()), 1), 3),
                     "hourly_fpr": round(fp / max(int((y == 0).sum()), 1), 4), "precision": round(tp / max(tp + fp, 1), 3),
                     "onsets_warned": f"{sum(ev)}/{len(ev)}", "false_alarm_hours_per_quiet_day": round(fp / quiet_days, 3)})
    return rows


# ---------------- splits ----------------
def embargo(t: pd.DatetimeIndex, boundaries: list[pd.Timestamp], hours: int = EMBARGO_H) -> np.ndarray:
    """True where a sample is at least `hours` away from every boundary."""
    keep = np.ones(len(t), bool)
    for b in boundaries:
        keep &= np.abs((t - b) / pd.Timedelta(hours=1)) >= hours
    return keep


def equal_blocks(t0: pd.Timestamp, t1: pd.Timestamp, k: int) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    edges = pd.date_range(t0, t1, periods=k + 1)
    return list(zip(edges[:-1], edges[1:]))


def blocked_oof(D: pd.DataFrame, make, blocks, cols) -> np.ndarray:
    """Leave-one-block-out predictions; training rows within the embargo of the held-out block are dropped."""
    t = D.index
    oof = np.full(len(D), np.nan)
    for i, (a, b) in enumerate(blocks):
        last = i == len(blocks) - 1
        te = (t >= a) & ((t <= b) if last else (t < b))
        tr = ~te & embargo(t, [a, b])
        if te.sum() == 0 or D["y"][tr].nunique() < 2:
            continue
        m = make().fit(D.loc[tr, cols].to_numpy(float), D["y"][tr].to_numpy())
        oof[te] = m.predict_proba(D.loc[te, cols].to_numpy(float))[:, 1]
    return oof


# ---------------- main ----------------
def run(a) -> dict:
    t_start = time.time()
    raw = Path(a.raw).expanduser()
    h, prov = build_hourly(raw, (a.first_year, a.last_year))
    print(f"hourly rows {len(h)}  {h.index.min()} .. {h.index.max()}  p10 valid {h['p10'].notna().mean():.3f} "
          f"xray valid {h['xmax'].notna().mean():.3f}", flush=True)
    F, L = feature_frame(h), labels(h)
    D = F.join(L)
    D = D[D["eligible"]].drop(columns="eligible")
    t = D.index
    onsets_all = onset_times(h)
    train_end, calib_end = pd.Timestamp(a.train_end, tz="UTC"), pd.Timestamp(a.calib_end, tz="UTC")
    t0, t1 = h.index.min(), h.index.max()
    print(f"eligible quiet hours {len(D)}, positives {int(D['y'].sum())}, onsets {len(onsets_all)}", flush=True)

    # model selection: blocked CV inside the pre-test period
    pre = D[t < calib_end]
    sel_blocks = equal_blocks(pre.index.min(), pre.index.max(), a.sel_blocks)
    sel = {}
    for name, mk in MAKERS.items():
        oof = blocked_oof(pre, mk, sel_blocks, FEATURES)
        ok = ~np.isnan(oof)
        sel[name] = metrics(pre["y"][ok], oof[ok], float(pre["y"].mean()))
        print(f"selection CV {name}: {sel[name]}", flush=True)
    chosen = min(sel, key=lambda k: sel[k]["brier"])

    # protocol A: time-forward
    keep = embargo(t, [train_end, calib_end])
    tr, ca, te = keep & (t < train_end), keep & (t >= train_end) & (t < calib_end), keep & (t >= calib_end)
    Xtr, ytr = D.loc[tr, FEATURES].to_numpy(float), D["y"][tr].to_numpy()
    est = MAKERS[chosen]().fit(Xtr, ytr)
    p_ca = est.predict_proba(D.loc[ca, FEATURES].to_numpy(float))[:, 1]
    platt = fit_platt(p_ca, D["y"][ca].to_numpy()) if D["y"][ca].nunique() == 2 else None
    raw_te = est.predict_proba(D.loc[te, FEATURES].to_numpy(float))[:, 1]
    p_te = apply_platt(raw_te, platt) if platt else raw_te
    pers = make_persistence().fit(D.loc[tr, ["lp10"]].to_numpy(float), ytr)
    pers_te = pers.predict_proba(D.loc[te, ["lp10"]].to_numpy(float))[:, 1]
    clim = float(ytr.mean())
    yte, tte = D["y"][te].to_numpy(), t[te]
    onsets_te = [o for o in onsets_all if o >= calib_end + pd.Timedelta(hours=EMBARGO_H)]
    A = {"split": {"train": [str(t0), str(train_end)], "calibration": [str(train_end), str(calib_end)],
                   "test": [str(calib_end), str(t1)], "embargo_hours": EMBARGO_H,
                   "n_train": int(tr.sum()), "n_calibration": int(ca.sum()), "n_test": int(te.sum()),
                   "independent_onsets": {"train": sum(o < train_end for o in onsets_all),
                                          "calibration": sum(train_end <= o < calib_end for o in onsets_all),
                                          "test": len(onsets_te)}},
         "model": metrics(yte, p_te, clim), "model_uncalibrated": metrics(yte, raw_te, clim),
         "persistence": metrics(yte, pers_te, clim), "climatology": metrics(yte, np.full(len(yte), clim), clim),
         "reliability_model": reliability(yte, p_te), "reliability_persistence": reliability(yte, pers_te),
         "thresholds_model": thresholds_table(tte, yte, p_te, onsets_te),
         "thresholds_persistence": thresholds_table(tte, yte, pers_te, onsets_te)}
    A["bootstrap_ci95"] = block_bootstrap(tte, yte, {"model": p_te, "persistence": pers_te}, clim, reps=a.boot)
    print(f"protocol A test: model {A['model']} | persistence {A['persistence']}", flush=True)

    # protocol B: leave-one-block-out over all data with the chosen model type
    blocks = equal_blocks(D.index.min(), D.index.max(), a.cv_blocks)
    oof = blocked_oof(D, MAKERS[chosen], blocks, FEATURES)
    oof_p = blocked_oof(D, make_persistence, blocks, ["lp10"])
    ok = ~np.isnan(oof) & ~np.isnan(oof_p)
    clim_b = np.full(len(D), np.nan)
    for i, (b0, b1) in enumerate(blocks):
        te_b = (t >= b0) & ((t <= b1) if i == len(blocks) - 1 else (t < b1))
        tr_b = ~te_b & embargo(t, [b0, b1])
        clim_b[te_b] = D["y"][tr_b].mean()
    B = {"blocks": [[str(x), str(y)] for x, y in blocks], "embargo_hours": EMBARGO_H,
         "model": metrics(D["y"][ok], oof[ok], float(D["y"].mean())),
         "persistence": metrics(D["y"][ok], oof_p[ok], float(D["y"].mean())),
         "climatology_per_fold": metrics(D["y"][ok], clim_b[ok], float(D["y"].mean())),
         "note": "pooled out-of-fold, no Platt step; BSS here is against the pooled base rate"}
    print(f"protocol B: model {B['model']} | persistence {B['persistence']}", flush=True)

    out = Path(a.out).expanduser() / "spaceweather"
    out.mkdir(parents=True, exist_ok=True)
    import joblib
    import sklearn
    joblib.dump({"estimator": est, "platt": platt}, out / "model.joblib")   # our own artifact; load only what you built
    (out / "feature_names.json").write_text(json.dumps(FEATURES))
    data_version = pd.util.hash_pandas_object(h.round(12), index=True).sum()
    m = {"task": TASK, "model_id": MODEL_ID, "arch_id": ARCH_ID, "chosen_model": chosen, "selection_cv": sel,
         "selection_blocks": [[str(x), str(y)] for x, y in sel_blocks], "protocol_a_time_forward": A,
         "protocol_b_blocked_cv": B, "climatology_rate": round(clim, 5), "platt": platt,
         "label": "hourly mean >=10 MeV flux reaches 10 pfu in any of the next 24 h; samples: hours below 10 pfu now",
         "data": {"source": "NOAA NCEI GOES SEM 5-min averages (EPS/EPEAD integral protons, XRS)", "rows_hourly": len(h),
                  "range": [str(t0), str(t1)], "hourly_table_hash": str(int(data_version) & ((1 << 64) - 1)),
                  "months": len(prov), "xrs_unscaled_by": XRS_UNSCALE},
         "runtime_s": round(time.time() - t_start, 1), "smoke": bool(a.smoke)}
    (out / "metrics.json").write_text(json.dumps(m, indent=1, default=str))
    (out / "provenance.json").write_text(json.dumps(prov, indent=0))
    man = {"task": TASK, "model_id": MODEL_ID, "arch_id": ARCH_ID, "status": "draft", "format": "joblib",
           "contents": "dict(estimator=sklearn %s, platt=[a, b] or None)" % chosen, "version": "0.1.0",
           "feature_count": len(FEATURES), "training_data_range": {"train": A["split"]["train"],
                                                                  "calibration": A["split"]["calibration"],
                                                                  "test_held_out": A["split"]["test"]},
           "created_utc": datetime.now(timezone.utc).isoformat(), "git_commit": a.commit,
           "python": platform.python_version(), "sklearn": sklearn.__version__, "numpy": np.__version__,
           "sha256": {f: sha256(out / f) for f in ("model.joblib", "feature_names.json", "metrics.json")}}
    (out / "manifest.json").write_text(json.dumps(man, indent=1))
    print(f"wrote {out}", flush=True)
    return m


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--raw", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--first-year", type=int, default=1998)
    p.add_argument("--last-year", type=int, default=2019)
    p.add_argument("--train-end", default="2011-01-01")
    p.add_argument("--calib-end", default="2013-01-01")
    p.add_argument("--sel-blocks", type=int, default=5)
    p.add_argument("--cv-blocks", type=int, default=9)
    p.add_argument("--boot", type=int, default=500)
    p.add_argument("--commit", default=None)
    p.add_argument("--smoke", action="store_true", help="one year (2003), quarter splits, few bootstrap reps")
    a = p.parse_args(argv)
    if a.smoke:
        a.first_year = a.last_year = 2003
        a.train_end, a.calib_end, a.sel_blocks, a.cv_blocks, a.boot = "2003-08-01", "2003-10-01", 3, 3, 20
    run(a)
    return 0


if __name__ == "__main__":
    sys.exit(main())
