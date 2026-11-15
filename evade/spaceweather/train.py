"""Train the SPE onset forecast, version 2 (EVA-9 GOES-only; EVA-10/EVA-11 GOES + CME), on GOES 1998-2026.

Data: NOAA NCEI GOES-8..15 5-min averages (CSV, 1998-2020-03) and GOES-16/18 SGPS + XRS L2 (netCDF, 2020-11 on),
the SOHO/LASCO CDAW CME catalogue (historical stand-in for DONKI CMEs), per-satellite background correction.
Task: at each quiet hour (>=10 MeV hourly flux below 10 pfu), will the hourly flux reach 10 pfu within 24 h?
Split (proof of concept, one split): train 1998-2016, test 2017 onward with a 72 h embargo at the boundary;
results overall, for the GOES-15 era (2017-2020-03) and the GOES-R era (2020-11 on). Logistic regression, no
recalibration step. Baselines: persistence (current log flux only) and climatology (training base rate).

Run:  python -I -c "import sys; sys.path.insert(0, '<code>'); from evade.spaceweather.train import main; main()" \
        --raw <goes csv root> --goesr <goes-r root> --cdaw <univ_all.txt> --out <model_dir> [--smoke]
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

from . import background as BG
from .features import (CME_FEATURES, FEATURES, HORIZON_H, LOOKBACK_H, cme_frame, feature_frame, labels, onset_times,
                       to_hourly)
from .model import TASK, sha256

XRS_UNSCALE = {"long": 0.7, "short": 0.85}   # SWPC scaling on GOES-8..15 XRS; divide to get true irradiance
EMBARGO_H = LOOKBACK_H + HORIZON_H
BOOT_BLOCK_DAYS = 27                          # one solar rotation
CME_LAG_H = 3                                 # assume a CME is known 3 h after its first coronagraph image
GOESR_SWITCH = "2023-01-01"                   # GOES-16 protons/X-rays before, GOES-18 after
OVERLAP = ("2022-09-01", "2023-01-01")        # GOES-16 / GOES-18 overlap kept back for validating the correction


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
        hp["sat"] = float(pfiles[0].name[1:3])                  # g08_..., g15_... -> 8, 15
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


# ---------------- GOES-R ----------------
def build_goesr_sat(root: Path, sat: int, first: str | None = None, last: str | None = None) -> pd.DataFrame:
    """Hourly table for one GOES-R satellite: SGPS integral protons (integrated from differential) and XRS."""
    from .archive import read_sgps_day, read_xrs_year
    parts = []
    for f in sorted((root / "sgps" / f"g{sat}").glob("*/*/*.nc")):
        day = f.name.split("_d")[1][:8]
        if (first and day < first.replace("-", "")) or (last and day >= last.replace("-", "")):
            continue
        try:
            parts.append(read_sgps_day(f))
        except Exception as e:                              # a bad daily file is skipped, not fatal
            print(f"skip {f.name}: {type(e).__name__}", flush=True)
    xr = []
    for f in sorted((root / "xrs" / f"g{sat}").glob("*.nc")):
        try:
            xr.append(read_xrs_year(f))
        except Exception as e:
            print(f"skip {f.name}: {type(e).__name__}", flush=True)
    if not parts:
        return pd.DataFrame(columns=["p10", "p50", "p100", "xmax", "xmean", "sat"])
    p = pd.concat(parts).sort_index()
    hp = to_hourly(p[~p.index.duplicated()])[["p10", "p50", "p100"]]
    if xr:
        x = pd.concat(xr).sort_index()
        x = x[(x.index >= hp.index.min()) & (x.index < hp.index.max() + pd.Timedelta(hours=1))]
        hx = to_hourly(x[~x.index.duplicated()])[["xmax", "xmean"]]
        hp = hp.join(hx, how="left")
    else:
        hp = hp.assign(xmax=np.nan, xmean=np.nan)
    hp["sat"] = float(sat)
    return hp


def combine_goesr(g16: pd.DataFrame, g18: pd.DataFrame) -> pd.DataFrame:
    """GOES-16 before GOESR_SWITCH, GOES-18 after; gaps filled from the other satellite (with its own `sat`)."""
    if not len(g16) or not len(g18):
        return g16 if len(g16) else g18
    sw = pd.Timestamp(GOESR_SWITCH, tz="UTC")
    primary = pd.concat([g16[g16.index < sw], g18[g18.index >= sw]])
    backup = pd.concat([g18[g18.index < sw], g16[g16.index >= sw]])
    idx = primary.index.union(backup.index)
    primary, backup = primary.reindex(idx), backup.reindex(idx)
    gap = primary["p10"].isna() & backup["p10"].notna()
    primary.loc[gap, ["p10", "p50", "p100", "sat"]] = backup.loc[gap, ["p10", "p50", "p100", "sat"]]
    xgap = primary["xmax"].isna()
    primary.loc[xgap, ["xmax", "xmean"]] = backup.loc[xgap, ["xmax", "xmean"]]
    return primary


# ---------------- models and metrics ----------------
def make_lr():
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), LogisticRegression(C=1.0, max_iter=5000))


def make_persistence():
    from sklearn.linear_model import LogisticRegression
    return LogisticRegression(max_iter=1000)


def metrics(y, p) -> dict:
    from sklearn.metrics import roc_auc_score
    y, p = np.asarray(y), np.asarray(p)
    two = len(np.unique(y)) == 2
    return {"n": int(len(y)), "positives": int(y.sum()), "brier": round(float(np.mean((p - y) ** 2)), 5),
            "auroc": round(float(roc_auc_score(y, p)), 4) if two else None,
            "mean_p": round(float(p.mean()), 4) if len(p) else None,
            "base_rate": round(float(y.mean()), 4) if len(y) else None}


def reliability(y, p, edges=(0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 1.0001)) -> list[dict]:
    y, p = np.asarray(y), np.asarray(p)
    out = []
    for lo, hi in zip(edges, edges[1:]):
        m = (p >= lo) & (p < hi)
        if m.sum():
            out.append({"bin": [lo, round(min(hi, 1.0), 3)], "n": int(m.sum()), "mean_p": round(float(p[m].mean()), 4),
                        "observed": round(float(y[m].mean()), 4)})
    return out


def diff_ci(t: pd.DatetimeIndex, y, pa, pb, reps: int = 300, seed: int = 0) -> dict:
    """95% CI of (b - a) for Brier and AUROC, resampling 27-day blocks (keeps each storm's hours together)."""
    from sklearn.metrics import roc_auc_score
    rng = np.random.default_rng(seed)
    y, pa, pb = np.asarray(y), np.asarray(pa), np.asarray(pb)
    blk = ((t - t.min()) / pd.Timedelta(days=BOOT_BLOCK_DAYS)).astype(int).to_numpy()
    ub = np.unique(blk)
    idx = {b: np.flatnonzero(blk == b) for b in ub}
    db, da = [], []
    for _ in range(reps):
        ii = np.concatenate([idx[b] for b in rng.choice(ub, size=len(ub), replace=True)])
        db.append(np.mean((pb[ii] - y[ii]) ** 2) - np.mean((pa[ii] - y[ii]) ** 2))
        if len(np.unique(y[ii])) == 2:
            da.append(roc_auc_score(y[ii], pb[ii]) - roc_auc_score(y[ii], pa[ii]))
    q = lambda v: [round(float(np.percentile(v, 2.5)), 5), round(float(np.percentile(v, 97.5)), 5)] if v else None
    return {"brier_b_minus_a": q(db), "auroc_b_minus_a": q(da), "reps": reps, "block_days": BOOT_BLOCK_DAYS}


def embargo(t: pd.DatetimeIndex, boundaries: list[pd.Timestamp], hours: int = EMBARGO_H) -> np.ndarray:
    keep = np.ones(len(t), bool)
    for b in boundaries:
        keep &= np.abs((t - b) / pd.Timedelta(hours=1)) >= hours
    return keep


def live_swpc_background(hours: int = 168) -> dict | None:
    """Quiet floor of the live SWPC operational integral feed (7 days), for the 'swpc:<sat>' table entries."""
    from .features import records_frame
    from .providers import NoaaSwpcProvider
    try:
        obs = NoaaSwpcProvider().fetch(hours)
    except Exception as e:
        print(f"live SWPC background skipped: {type(e).__name__}", flush=True)
        return None
    h = to_hourly(records_frame(obs))
    sats = h["sat"].dropna()
    if not len(sats):
        return None
    sat = str(int(sats.mode().iloc[0]))
    e = BG.estimate({sat: h})["satellites"][sat]
    e["estimated_from"] = [str(h.index.min()), str(h.index.max())]
    return {f"swpc:{sat}": e, "_hourly": h}


# ---------------- main ----------------
def run(a) -> dict:
    t_start = time.time()
    raw, goesr = Path(a.raw).expanduser(), Path(a.goesr).expanduser()
    h_old, prov = build_hourly(raw, (a.first_year, a.last_year))
    g16 = build_goesr_sat(goesr, 16, a.goesr_first, a.goesr_last)
    g18 = build_goesr_sat(goesr, 18, a.goesr_first, a.goesr_last)
    print(f"old {len(h_old)} h; g16 {len(g16)} h; g18 {len(g18)} h", flush=True)

    # per-satellite background, the GOES-16/18 overlap held out for validation
    by_sat = {str(int(s)): g.drop(columns="sat") for s, g in h_old.groupby("sat")}
    for s, g in (("16", g16), ("18", g18)):
        if len(g):
            by_sat[s] = g
    table = BG.estimate(by_sat, exclude=[OVERLAP])
    check = {}
    live = live_swpc_background() if a.swpc_live else None
    if live:
        h_live = live.pop("_hourly")
        table["satellites"].update(live)
        if len(g18):         # our archive integration vs the SWPC operational integral for GOES-18, common hours
            j = h_live[["p10", "p50", "p100"]].join(g18[["p10", "p50", "p100"]].add_suffix("_ncei"), how="inner").dropna()
            check["swpc_vs_ncei_integration_g18"] = {
                "hours": int(len(j)),
                **{c: round(float((np.log10(j[c].clip(lower=1e-4)) - np.log10(j[f"{c}_ncei"].clip(lower=1e-4))).median()), 3)
                   if len(j) else None for c in ("p10", "p50", "p100")},
                "note": "median log10(SWPC / NCEI-integrated); 0 = identical"}
    table["note"] = "estimated by evade.spaceweather.train; quiet = hourly >=10 MeV < 10 pfu, no S1 within 72 h"
    agree = {}
    if len(g16) and len(g18):
        w = lambda g: g[(g.index >= pd.Timestamp(OVERLAP[0], tz="UTC")) & (g.index < pd.Timestamp(OVERLAP[1], tz="UTC"))]
        agree["g16_vs_g18_2022_09_12"] = {"before": BG.overlap_agreement(w(g16), w(g18), None, "16", "18"),
                                          "after": BG.overlap_agreement(w(g16), w(g18), table, "16", "18")}
    print("background", json.dumps(table["satellites"]), json.dumps(agree), json.dumps(check), flush=True)

    gr = combine_goesr(g16, g18) if (len(g16) or len(g18)) else pd.DataFrame()
    h = pd.concat([h_old, gr]).sort_index() if len(gr) else h_old
    h = h[~h.index.duplicated(keep="first")]
    h = h.reindex(pd.date_range(h.index.min(), h.index.max(), freq="1h", tz="UTC"))
    hc, _ = BG.correct_hourly(h, table)

    from .archive import read_cdaw
    cd = read_cdaw(Path(a.cdaw).expanduser())
    cd["t"] = cd["t"] + pd.Timedelta(hours=CME_LAG_H)
    F = feature_frame(hc).join(cme_frame(h.index, cd))
    F_raw = feature_frame(h)                                   # same features without background correction
    L = labels(h)                                              # labels on the satellite's own flux
    ok = L["eligible"].to_numpy()
    D, Draw, y_all = F[ok], F_raw[ok], L["y"][ok]
    t = D.index
    split = pd.Timestamp(a.split, tz="UTC")
    keep = embargo(t, [split])
    tr, te = keep & (t < split), keep & (t >= split)
    ytr, yte = y_all[tr].to_numpy(), y_all[te].to_numpy()
    onsets = onset_times(h)
    print(f"eligible {len(D)}, train {tr.sum()} (pos {ytr.sum()}), test {te.sum()} (pos {yte.sum()}), "
          f"onsets {len(onsets)}", flush=True)

    cols = {"goes_bgcorr": FEATURES, "goes_cme_bgcorr": FEATURES + CME_FEATURES}
    models, preds = {}, {}
    for name, c in cols.items():
        models[name] = make_lr().fit(D.loc[tr, c].to_numpy(float), ytr)
        preds[name] = models[name].predict_proba(D.loc[te, c].to_numpy(float))[:, 1]
    m_raw = make_lr().fit(Draw.loc[tr, FEATURES].to_numpy(float), ytr)
    preds["goes_no_bgcorr"] = m_raw.predict_proba(Draw.loc[te, FEATURES].to_numpy(float))[:, 1]
    pers = make_persistence().fit(D.loc[tr, ["lp10"]].to_numpy(float), ytr)
    preds["persistence"] = pers.predict_proba(D.loc[te, ["lp10"]].to_numpy(float))[:, 1]
    preds["climatology"] = np.full(int(te.sum()), float(ytr.mean()))

    tte = t[te]
    goesr_start = gr.index.min() if len(gr) else pd.Timestamp("2100-01-01", tz="UTC")
    eras = {"test_all": np.ones(len(tte), bool), "goes15_era": np.asarray(tte < goesr_start),
            "goesr_era": np.asarray(tte >= goesr_start)}
    first_ok = split + pd.Timedelta(hours=EMBARGO_H)
    in_era = {"test_all": lambda o: True, "goes15_era": lambda o: o < goesr_start, "goesr_era": lambda o: o >= goesr_start}
    table_m = {}
    for era, msk in eras.items():
        table_m[era] = {"onsets": int(sum(o >= first_ok and in_era[era](o) for o in onsets)),
                        **{k: metrics(yte[msk], p[msk]) for k, p in preds.items()}}
    gm = eras["goesr_era"]
    rel = {"test_all_goes_bgcorr": reliability(yte, preds["goes_bgcorr"]),
           "goesr_era_goes_bgcorr": reliability(yte[gm], preds["goes_bgcorr"][gm]) if gm.any() else None,
           "goesr_era_goes_cme_bgcorr": reliability(yte[gm], preds["goes_cme_bgcorr"][gm]) if gm.any() else None}
    cme_ci = {"test_all": diff_ci(tte, yte, preds["goes_bgcorr"], preds["goes_cme_bgcorr"], reps=a.boot)}
    if gm.sum() > 100 and yte[gm].sum() > 0:
        cme_ci["goesr_era"] = diff_ci(tte[gm], yte[gm], preds["goes_bgcorr"][gm], preds["goes_cme_bgcorr"][gm],
                                      reps=a.boot)
    for era, d in table_m.items():
        print(era, json.dumps({k: (v if not isinstance(v, dict) else {kk: v[kk] for kk in ("brier", "auroc")})
                               for k, v in d.items()}), flush=True)
    print("cme_minus_goes_ci", json.dumps(cme_ci), flush=True)

    # pre-stated serving rule: the CME model is served only if the CI shows a Brier gain on the whole test period
    ci_b = cme_ci["test_all"]["brier_b_minus_a"]
    served = "goes_cme_bgcorr" if ci_b and ci_b[1] < 0 else "goes_bgcorr"
    ids = dict(zip(("goes_bgcorr", "cme_arch", "goes_cme_bgcorr"), a.ids.split(",")))
    out = Path(a.out).expanduser() / "spaceweather"
    out.mkdir(parents=True, exist_ok=True)
    import joblib
    import sklearn
    joblib.dump({"estimator": models[served], "platt": None, "background": table}, out / "model.joblib")  # own artifact
    (out / "feature_names.json").write_text(json.dumps(cols[served]))
    (out / "background.json").write_text(json.dumps(table, indent=1))
    m = {"task": TASK, "model_id": ids[served], "served": served, "ids": ids, "split": a.split,
         "test_range": [str(split), str(h.index.max())], "goesr_start": str(goesr_start),
         "metrics": table_m, "reliability": rel, "cme_minus_goes_ci95": cme_ci, "background": table,
         "background_agreement": agree, "integration_check": check, "climatology_rate": round(float(ytr.mean()), 5),
         "label": "hourly mean >=10 MeV flux reaches 10 pfu in any of the next 24 h; samples: hours below 10 pfu now",
         "data": {"goes_csv_months": len(prov), "hourly_rows": len(h), "range": [str(h.index.min()), str(h.index.max())],
                  "cdaw_cmes": len(cd), "cme_lag_hours": CME_LAG_H, "goesr_switch": GOESR_SWITCH},
         "serving_note": f"draft model {ids[served]} trained on GOES-8..15 1998-2016, tested 2017 on incl. GOES-R; "
                         "a probability for review, not a warning service",
         "runtime_s": round(time.time() - t_start, 1), "smoke": bool(a.smoke)}
    (out / "metrics.json").write_text(json.dumps(m, indent=1, default=str))
    (out / "provenance.json").write_text(json.dumps(prov, indent=0))
    man = {"task": TASK, "model_id": ids[served], "arch_id": "EVA-7" if served == "goes_bgcorr" else ids["cme_arch"],
           "status": "draft", "format": "joblib",
           "contents": "dict(estimator=sklearn logistic pipeline, platt=None, background=table)", "version": "0.2.0",
           "feature_count": len(cols[served]),
           "training_data_range": {"train": [str(h.index.min()), str(split)], "test_held_out": m["test_range"]},
           "created_utc": datetime.now(timezone.utc).isoformat(), "git_commit": a.commit,
           "python": platform.python_version(), "sklearn": sklearn.__version__, "numpy": np.__version__,
           "sha256": {f: sha256(out / f) for f in ("model.joblib", "feature_names.json", "metrics.json", "background.json")}}
    (out / "manifest.json").write_text(json.dumps(man, indent=1))
    print(f"served {served} as {ids[served]}; wrote {out}", flush=True)
    return m


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--raw", required=True)
    p.add_argument("--goesr", required=True)
    p.add_argument("--cdaw", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--first-year", type=int, default=1998)
    p.add_argument("--last-year", type=int, default=2020)
    p.add_argument("--goesr-first", default=None)
    p.add_argument("--goesr-last", default=None)
    p.add_argument("--split", default="2017-01-01")
    p.add_argument("--boot", type=int, default=300)
    p.add_argument("--ids", default="EVA-9,EVA-10,EVA-11", help="GOES-only model, CME arch, CME model")
    p.add_argument("--swpc-live", action="store_true", help="estimate the live SWPC feed background (network)")
    p.add_argument("--commit", default=None)
    p.add_argument("--smoke", action="store_true", help="2003 plus a few GOES-R months, few bootstrap reps")
    a = p.parse_args(argv)
    if a.smoke:
        a.first_year = a.last_year = 2003
        a.goesr_first = a.goesr_first or "2021-06-01"
        a.goesr_last = a.goesr_last or "2021-10-01"
        a.split, a.boot = "2003-10-01", 20
    run(a)
    return 0


if __name__ == "__main__":
    sys.exit(main())
