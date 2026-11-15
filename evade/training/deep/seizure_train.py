"""Train and evaluate the deep seizure-suspect window model on packed CHB-MIT shards, resumable on SLURM.

  python -m evade.training.deep.seizure_train --data <pack dir> --protocol loso  --cases beta5 --run <dir> --resume
  python -m evade.training.deep.seizure_train --data <pack dir> --protocol final --cases all --external siena \
         --thresholds-from <loso run>/metrics.json --run <dir> --resume

Protocols:
  loso   leave one subject out over the subjects of --cases (chb21 is grouped with chb01). In each fold,
         `--inner-val` training subjects with seizures are held out for checkpoint selection and early stopping;
         the test subject is scored once with the selected checkpoint. Out-of-fold scores give the metrics and
         the operating points (thresholds at 1% and 5% pooled out-of-fold FPR).
  final  train on every subject of --cases (minus the inner validation subjects), then score the external set
         (Siena) once at the thresholds of --thresholds-from (the CHB-MIT out-of-fold operating points).
Input: each window is robust-scaled per channel with its person's median and 1.4826 x MAD from that person's
non-ictal windows (seizure_pack.py), clipped to +-20. --norm all reproduces the beta's all-windows rule.
Training: balanced batches (half ictal, half non-ictal, ictal drawn with replacement) and BCE by default, or
--sampler random with --loss focal; augmentation: +-0.5 s circular shift, x0.8-1.2 amplitude; fp16 autocast on
CUDA (V100). SIGUSR1 (sbatch --signal=B:USR1@N) saves last.pt and exits 99; resubmitting resumes and skips
finished folds. Outputs are seizure-suspect flags for human review, not diagnoses.
"""
from __future__ import annotations

import argparse
import json
import os
import signal
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

from . import seizure_eval as E
from .seizure_models import build
from .seizure_pack import BETA5, SIENA6

STOP = {"flag": False}
NORM_CLIP = 20.0
BETA = {"model": "EVA-2 beta HistGradientBoosting on band-power features (chb01-05 LOSO, Siena-6 external)",
        "pooled_auroc": 0.967, "subject_auroc_mean": 0.986, "subject_auroc_ci95": [0.969, 0.997],
        "sensitivity_at_1pct_fpr_subject_mean": 0.35, "sensitivity_at_1pct_fpr_ci95": [0.20, 0.51],
        "fp_windows_per_hour_at_1pct": 8.8, "siena_auroc": 0.920, "siena_ap": 0.173,
        "normalization": "per-person robust z over ALL of the person's windows, ictal included (a limitation: "
                         "the API can only use a seizure-free baseline)",
        "smoothing": "3-window centered moving average"}


def _on_usr1(signum, frame):
    STOP["flag"] = True


def check_stop():
    """Between folds or before external scoring: everything finished is on disk, so just exit 99."""
    if STOP["flag"]:
        print("SIGUSR1 between steps; finished folds are saved; exiting 99", flush=True)
        sys.exit(99)


class Shards:
    """Windows of several shards in one float16 array, plus per-window subject, record and label arrays."""

    def __init__(self, files: list[Path], norm: str, smoke: bool, rec_offset: int = 0):
        meta, total = [], 0
        for f in files:
            z = np.load(f, allow_pickle=False)
            y = z["y"]
            keep = np.arange(len(y))
            if smoke:
                rest = np.flatnonzero(y == 0)
                keep = np.union1d(np.flatnonzero(y != 0), rest[:: max(1, len(rest) // 300)])
            meta.append((f, keep)); total += len(keep)
            T = int(round(float(z["fs"]) * E.WIN_S))      # (reading z["X"] here would load the whole shard twice)
        self.X = np.empty((total, 18, T), np.float16)
        ys, sh, rec, st, self.subjects, self.cases, self.med, self.mad = [], [], [], [], [], [], [], []
        pres = []
        self.records, sz = [], {"subject": [], "rec_key": [], "start_s": [], "end_s": []}
        pos, r0 = 0, rec_offset
        for i, (f, keep) in enumerate(meta):
            z = np.load(f, allow_pickle=False)
            self.X[pos:pos + len(keep)] = z["X"][keep]
            pos += len(keep)
            ys.append(z["y"][keep]); sh.append(np.full(len(keep), i, np.int32))
            rec.append(z["rec"][keep].astype(np.int64) + r0); st.append(z["start_s"][keep])
            subj = str(z["subject"])
            self.subjects.append(subj); self.cases.append(f.stem)
            self.med.append(z["norm_median" if norm == "nonictal" else "norm_median_all"])
            self.mad.append(z["norm_mad" if norm == "nonictal" else "norm_mad_all"])
            self.records += [f"{f.stem}/{r}" for r in z["records"]]
            pres.append(z["rec_present"] if "rec_present" in z.files else np.ones((len(z["records"]), 18), bool))
            for k, a, b in zip(z["sz_rec"], z["sz_start_s"], z["sz_end_s"]):
                sz["subject"].append(subj); sz["rec_key"].append(int(k) + r0)
                sz["start_s"].append(float(a)); sz["end_s"].append(float(b))
            r0 += len(z["records"])
            self.fs = float(z["fs"])
        self.y = np.concatenate(ys).astype(np.int8)
        self.shard = np.concatenate(sh)
        self.rec = np.concatenate(rec)
        self.start_s = np.concatenate(st).astype(np.float64)
        self.subj = np.asarray(self.subjects)[self.shard]
        self.med = np.stack(self.med).astype(np.float32)
        self.mad = np.stack(self.mad).astype(np.float32)
        self.seizures = sz
        self.n_records = r0 - rec_offset
        self.present = np.concatenate(pres)          # (records, 18), indexed by rec - rec_offset
        self.rec_offset = rec_offset

    def batch(self, idx: np.ndarray, device) -> torch.Tensor:
        idx = np.sort(idx)
        x = torch.from_numpy(self.X[idx]).to(device, non_blocking=True).float()
        s = self.shard[idx]
        med = torch.from_numpy(self.med[s]).to(device)[:, :, None]
        mad = torch.from_numpy(self.mad[s]).to(device)[:, :, None]
        # channels missing from a record stay exactly 0 after normalization (as the API does)
        keep = torch.from_numpy(self.present[self.rec[idx] - self.rec_offset]).to(device)[:, :, None]
        return ((x - med) / mad).clamp_(-NORM_CLIP, NORM_CLIP) * keep, idx


def shard_files(root: Path, dataset: str, cases: str) -> list[Path]:
    files = sorted((root / dataset).glob("*.npz"))
    files = [f for f in files if not f.name.endswith(".tmp.npz")]
    if cases in ("", "all"):
        return files
    want = BETA5 if cases == "beta5" else SIENA6 if cases == "siena6" else cases.split(",")
    return [f for f in files if f.stem in want]


@torch.no_grad()
def predict(model, d: Shards, idx: np.ndarray, device, bs: int = 1024) -> np.ndarray:
    model.eval()
    out = np.empty(len(idx), np.float32)
    for i in range(0, len(idx), bs):
        sel = idx[i:i + bs]
        xb, order = d.batch(sel, device)
        with torch.autocast("cuda", dtype=torch.float16, enabled=device.type == "cuda"):
            p = torch.sigmoid(model(xb).float()).squeeze(1).cpu().numpy()
        # d.batch sorts; map back to the order of sel
        pos = np.searchsorted(order, sel)
        out[i:i + len(sel)] = p[pos]
    return out


class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=0.25):
        super().__init__()
        self.gamma, self.alpha = gamma, alpha

    def forward(self, logits, y):
        bce = nn.functional.binary_cross_entropy_with_logits(logits, y, reduction="none")
        pt = torch.exp(-bce)
        a = self.alpha * y + (1 - self.alpha) * (1 - y)
        return (a * (1 - pt) ** self.gamma * bce).mean()


def augment(x: torch.Tensor, gen: torch.Generator, fs: float) -> torch.Tensor:
    b = x.shape[0]
    scale = (0.8 + 0.4 * torch.rand(b, 1, 1, generator=gen)).to(x.device)
    shifts = torch.randint(-int(0.5 * fs), int(0.5 * fs) + 1, (b,), generator=gen)
    t = x.shape[2]
    idx = ((torch.arange(t)[None, :] - shifts[:, None]) % t).to(x.device)    # move (b, t), then expand on device
    x = torch.gather(x, 2, idx[:, None, :].expand(b, x.shape[1], t))
    return x * scale


def pick_val(train_subjects: list[str], d: Shards, n_val: int, seed: int) -> list[str]:
    with_sz = [s for s in sorted(train_subjects) if (d.y[d.subj == s] == 1).any()]
    if n_val <= 0 or len(with_sz) <= n_val:
        return []
    rng = np.random.default_rng(seed)
    return sorted(rng.choice(with_sz, n_val, replace=False).tolist())


def train_fold(a, d: Shards, train_subjects: list[str], val_subjects: list[str], outdir: Path, device,
               fold_seed: int):
    """Train one model; returns (model loaded with best.pt, info dict). Resumes from last.pt."""
    outdir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(fold_seed); np.random.seed(fold_seed)
    gen = torch.Generator(); gen.manual_seed(fold_seed)
    tr_mask = np.isin(d.subj, train_subjects) & ~np.isin(d.subj, val_subjects)
    pos_idx = np.flatnonzero(tr_mask & (d.y == 1))
    neg_idx = np.flatnonzero(tr_mask & (d.y == 0))
    lab_idx = np.flatnonzero(tr_mask & (d.y >= 0))
    val_idx = np.flatnonzero(np.isin(d.subj, val_subjects) & (d.y >= 0))
    if len(pos_idx) == 0:
        raise RuntimeError("no ictal windows in the training subjects")
    model = build(a.arch, 18, 1).to(device)
    epochs = 1 if a.smoke else a.epochs
    steps = max(1, min(len(lab_idx), a.epoch_windows) // a.bs)
    if a.smoke:
        steps = min(steps, 4)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.wd)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=epochs * steps)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    loss_fn = FocalLoss() if a.loss == "focal" else nn.BCEWithLogitsLoss()
    # checkpoint = (epoch in progress, steps done in it): a SIGUSR1 mid-epoch resumes at the next step, so the
    # one-cycle schedule and the sampler stream continue exactly where they stopped
    start, start_step, tot0, best, bad, hist = 0, 0, 0.0, -1.0, 0, []
    last = outdir / "last.pt"
    if a.resume and last.exists():
        ck = torch.load(last, map_location=device, weights_only=True)
        model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"]); sched.load_state_dict(ck["sched"])
        scaler.load_state_dict(ck["scaler"]); gen.set_state(ck["gen"].cpu())
        start, start_step, tot0 = ck["epoch"], ck["step"], ck["tot"]
        best, bad, hist = ck["best"], ck["bad"], ck["hist"]
        if start_step >= steps:
            start, start_step, tot0 = start + 1, 0, 0.0
        print(f"  resumed at epoch {start} step {start_step} (best val {best:.4f})", flush=True)

    def save(epoch, step, tot):
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                    "scaler": scaler.state_dict(), "gen": gen.get_state(), "epoch": int(epoch), "step": int(step),
                    "tot": float(tot), "best": float(best), "bad": int(bad), "hist": hist}, last.with_suffix(".tmp"))
        os.replace(last.with_suffix(".tmp"), last)

    half = a.bs // 2
    for ep in range(start, epochs):
        if bad >= a.patience:
            break
        model.train(); t0 = time.time()
        first = start_step if ep == start else 0
        tot = tot0 if ep == start else 0.0
        for i in range(first, steps):
            if a.sampler == "balanced":
                b = np.concatenate([pos_idx[torch.randint(0, len(pos_idx), (half,), generator=gen).numpy()],
                                    neg_idx[torch.randint(0, len(neg_idx), (a.bs - half,), generator=gen).numpy()]])
            else:
                b = lab_idx[torch.randint(0, len(lab_idx), (a.bs,), generator=gen).numpy()]
            xb, order = d.batch(b, device)
            yb = torch.from_numpy((d.y[order] == 1).astype(np.float32)).to(device)[:, None]
            xb = augment(xb, gen, d.fs)
            with torch.autocast("cuda", dtype=torch.float16, enabled=device.type == "cuda"):
                loss = loss_fn(model(xb).float(), yb)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); sched.step()
            tot += float(loss.detach())
            if STOP["flag"]:
                save(ep, i + 1, tot)
                print(f"  SIGUSR1 at epoch {ep} step {i + 1}/{steps}, checkpoint saved; exiting 99", flush=True)
                sys.exit(99)
        if len(val_idx):
            pv = predict(model, d, val_idx, device)
            val = E.weighted_auroc(d.y[val_idx], pv)
        else:                      # no validation subjects: keep the last epoch
            val = float(ep)
        hist.append({"epoch": int(ep), "loss": tot / steps, "val_auroc": float(val), "sec": round(time.time() - t0, 1)})
        if val > best:
            best, bad = float(val), 0
            torch.save(model.state_dict(), outdir / "best.pt")
        else:
            bad += 1
        save(ep, steps, tot)
        print(f"  epoch {ep + 1}/{epochs} loss {tot / steps:.4f} val {val:.4f} best {best:.4f} "
              f"({time.time() - t0:.0f}s)", flush=True)
    model.load_state_dict(torch.load(outdir / "best.pt", map_location=device, weights_only=True))
    info = {"train_subjects": sorted(set(train_subjects) - set(val_subjects)), "val_subjects": val_subjects,
            "best_val_auroc": best, "epochs_run": len(hist), "steps_per_epoch": steps,
            "n_train_windows": int(len(lab_idx)), "n_train_ictal": int(len(pos_idx)), "history": hist}
    return model, info


def eval_block(d: Shards, scores: np.ndarray, idx: np.ndarray, thresholds, a) -> dict:
    sel = set(np.unique(d.subj[idx]).tolist())
    keep = [i for i, s in enumerate(d.seizures["subject"]) if s in sel]
    sz = {k: [v[i] for i in keep] for k, v in d.seizures.items()}
    return E.evaluate(scores, d.y[idx], d.subj[idx], d.rec[idx], d.start_s[idx], sz, thresholds=thresholds,
                      fprs=a.fprs, n_boot=a.n_boot, seed=0, smooth=a.smooth)


def run_loso(a, device) -> dict:
    d = Shards(shard_files(a.data, "chbmit", a.cases), a.norm, a.smoke)
    subjects = sorted(s for s in set(d.subj.tolist()) if (d.y[d.subj == s] == 1).any())
    if a.smoke:
        subjects = subjects[:3]
    print(f"loso: {len(d.y)} windows, {int((d.y == 1).sum())} ictal, subjects {subjects}", flush=True)
    scores = np.full(len(d.y), np.nan, np.float32)
    folds = []
    for k, s in enumerate(subjects):
        fd = a.run / f"fold_{s}"
        te = np.flatnonzero(d.subj == s)
        if (fd / "result.json").exists():
            print(f"fold {s}: done, skipping", flush=True)
            folds.append(json.loads((fd / "result.json").read_text()))
            scores[te] = np.load(fd / "preds.npz")["score"]
            continue
        check_stop()
        train_s = [t for t in subjects if t != s]
        val_s = pick_val(train_s, d, a.inner_val, a.seed + k)
        print(f"fold {s}: train {len(train_s) - len(val_s)} subjects, val {val_s}", flush=True)
        model, info = train_fold(a, d, train_s, val_s, fd, device, a.seed + k)
        p = predict(model, d, te, device)
        np.savez(fd / "preds.npz", score=p, y=d.y[te], start_s=d.start_s[te], rec=d.rec[te])
        info.update({"test_subject": s, "n_test_windows": int((d.y[te] >= 0).sum()),
                     "test_auroc": E.weighted_auroc(d.y[te][d.y[te] >= 0], p[d.y[te] >= 0])})
        (fd / "result.json").write_text(json.dumps(info, indent=1))
        (fd / "last.pt").unlink(missing_ok=True)
        folds.append(info)
        scores[te] = p
        print(f"fold {s}: test AUROC {info['test_auroc']:.4f}", flush=True)
    idx = np.flatnonzero(np.isin(d.subj, subjects))
    res = eval_block(d, scores[idx], idx, None, a)
    return {"protocol": "leave-one-subject-out", "cases": d.cases, "subject_alias": {"chb21": "chb01"},
            "folds": [{k: f[k] for k in ("test_subject", "val_subjects", "epochs_run", "best_val_auroc", "test_auroc")}
                      for f in folds],
            "chbmit_oof": res,
            "operating_points_for_serving": [{"name": "suspect_seizure" if op["target_fpr"] <= 0.01 else "review",
                                              "target_fpr": op["target_fpr"], "threshold": op["threshold"],
                                              "sensitivity_oof_subject_mean": op["sensitivity_subject_mean"]["mean"],
                                              "sensitivity_oof_pooled": op["sensitivity_pooled"],
                                              "event_sensitivity_oof": op["event_sensitivity"],
                                              "fp_windows_per_hour_oof": op["fp_windows_per_hour_pooled"],
                                              "false_alarms_per_hour_oof": op["false_alarms_per_hour"]}
                                             for op in res["operating_points"]]}


def run_final(a, device) -> dict:
    d = Shards(shard_files(a.data, "chbmit", a.cases), a.norm, a.smoke)
    subjects = sorted(s for s in set(d.subj.tolist()) if (d.y[d.subj == s] == 1).any())
    val_s = pick_val(subjects, d, a.inner_val, a.seed)
    md = a.run / "model"
    print(f"final: train {len(subjects) - len(val_s)} subjects, val {val_s}", flush=True)
    if (md / "result.json").exists() and (md / "best.pt").exists():
        info = json.loads((md / "result.json").read_text())
        model = build(a.arch, 18, 1).to(device)
        model.load_state_dict(torch.load(md / "best.pt", map_location=device, weights_only=True))
        print("final model: done, loaded best.pt", flush=True)
    else:
        model, info = train_fold(a, d, subjects, val_s, md, device, a.seed)
        (md / "result.json").write_text(json.dumps(info, indent=1))
        (md / "last.pt").unlink(missing_ok=True)
    norm = {"median_rule": "per person, per channel, from that person's non-ictal windows" if a.norm == "nonictal"
            else "per person, per channel, from all of that person's windows (beta rule)"}
    out = {"protocol": "final (train on all --cases subjects, external test once)", "cases": d.cases,
           "train": {k: info[k] for k in ("train_subjects", "val_subjects", "epochs_run", "best_val_auroc",
                                          "n_train_windows", "n_train_ictal")}, "normalization": norm}
    thr_src = json.loads(a.thresholds_from.read_text()) if a.thresholds_from else None
    if thr_src:
        out["operating_points_for_serving"] = thr_src["operating_points_for_serving"]
        out["thresholds_from"] = {"file": a.thresholds_from.name, "protocol": thr_src["protocol"],
                                  "cases": thr_src["cases"]}
    del d
    check_stop()
    if a.external:
        files = shard_files(a.data, a.external, "all")
        if a.smoke:
            files = files[:2]
        e = Shards(files, a.norm, a.smoke)
        idx = np.arange(len(e.y))
        p = predict(model, e, idx, device)
        np.savez(a.run / f"preds_{a.external}.npz", score=p, y=e.y, subject=e.subj, start_s=e.start_s, rec=e.rec)
        thr = {f"{op['target_fpr']:g}": op["threshold"] for op in thr_src["operating_points_for_serving"]} \
            if thr_src else None
        ext = {"note": "trained on CHB-MIT (pediatric, native bipolar), tested on Siena (adults, monopolar converted "
                       "to the same bipolar montage, 512 Hz resampled to 256 Hz, 50 Hz notch)"}
        ext["all_subjects"] = {"at_chbmit_thresholds": eval_block(e, p, idx, thr, a) if thr else None,
                               "at_own_thresholds_oracle": eval_block(e, p, idx, None, a)}
        six = np.flatnonzero(np.isin(e.subj, SIENA6))
        if len(six):
            ext["siena6_beta_subjects"] = {"at_chbmit_thresholds": eval_block(e, p[six], six, thr, a) if thr else None}
        ext["oracle_note"] = "own thresholds are set on the external scores themselves; descriptive only, never served"
        out[f"external_{a.external}"] = ext
    return out


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True, type=Path, help="seizure_pack output folder")
    p.add_argument("--protocol", required=True, choices=["loso", "final"])
    p.add_argument("--cases", default="all", help="all, beta5, or a comma list of CHB-MIT cases")
    p.add_argument("--external", default="", help="final protocol: external shard folder name (siena)")
    p.add_argument("--thresholds-from", type=Path, help="metrics.json of a loso run (its operating points)")
    p.add_argument("--arch", default="seizure_resnet1d")
    p.add_argument("--norm", default="nonictal", choices=["nonictal", "all"])
    p.add_argument("--sampler", default="balanced", choices=["balanced", "random"])
    p.add_argument("--loss", default="bce", choices=["bce", "focal"])
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--patience", type=int, default=5)
    p.add_argument("--epoch-windows", type=int, default=200_000, help="windows drawn per epoch (cap)")
    p.add_argument("--bs", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--wd", type=float, default=1e-2)
    p.add_argument("--inner-val", type=int, default=1, help="training subjects held out for model selection")
    p.add_argument("--smooth", type=int, default=3)
    p.add_argument("--fprs", default="0.01,0.05")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n-boot", type=int, default=2000)
    p.add_argument("--run", required=True, type=Path)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--smoke", action="store_true")
    a = p.parse_args(argv)
    a.fprs = tuple(float(f) for f in a.fprs.split(","))
    if a.smoke:
        a.n_boot = min(a.n_boot, 50)
    if hasattr(signal, "SIGUSR1"):
        signal.signal(signal.SIGUSR1, _on_usr1)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True
    print(f"device {device} {torch.cuda.get_device_name(0) if device.type == 'cuda' else ''} torch {torch.__version__}",
          flush=True)
    a.run.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    res = run_loso(a, device) if a.protocol == "loso" else run_final(a, device)
    pack = a.data / "pack_summary.json"
    res.update({"task": "seizure_window", "arch": a.arch, "norm": a.norm, "sampler": a.sampler, "loss": a.loss,
                "inner_val": a.inner_val, "max_epochs": a.epochs, "patience": a.patience, "seed": a.seed,
                "window": {"fs_hz": 256.0, "window_s": E.WIN_S, "step_s": E.STEP_S, "channels": 18},
                "baseline_beta_eva2": BETA, "smoke": a.smoke, "sec": round(time.time() - t0, 1),
                "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu",
                "code_commit": os.environ.get("EVADE_CODE_COMMIT", "unknown"),
                "pack_summary": json.loads(pack.read_text()) if pack.exists() else None,
                "claim": "seizure-suspect flags for human review; not a diagnosis or a seizure detector",
                "limits": ["window and simple event metrics; not a certified SzCORE evaluation",
                           "seizure-free files subsampled per subject (beta rule), so false alarms per hour are "
                           "optimistic or noisy",
                           "normalization uses the person's non-ictal windows, which assumes a known seizure-free "
                           "baseline (the API's /baseline); the beta used all windows, ictal included"]})
    if res.get("pack_summary"):
        res["window"]["fs_hz"] = res["pack_summary"]["fs"]
    (a.run / "metrics.json").write_text(json.dumps(res, indent=1, default=float))
    head = res.get("chbmit_oof") or (res.get(f"external_{a.external}", {}).get("all_subjects", {})
                                     .get("at_chbmit_thresholds") or {})
    if head:
        op1 = head["operating_points"][0]
        print(f"DONE {a.protocol} pooled AUROC {head['pooled_auroc']:.4f} "
              f"[{head['pooled_auroc_ci95']['lo']:.3f}, {head['pooled_auroc_ci95']['hi']:.3f}] "
              f"subject AUROC {head['subject_auroc_mean']['mean']:.4f} AP {head['pooled_ap']:.4f} "
              f"sens@{op1['target_fpr']:g} {op1['sensitivity_subject_mean']['mean']:.3f} "
              f"events {op1['events_detected']}/{op1['events_total']}", flush=True)
    else:
        print(f"DONE {a.protocol}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
