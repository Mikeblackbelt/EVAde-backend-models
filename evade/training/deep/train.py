"""Train a deep ECG model on packed PTB-XL (official folds: train 1-8, validate 9, test 10), resumable on SLURM.

  python -m evade.training.deep.train --data ptbxl100.npz --arch xresnet1d101 --seeds 0,1,2,3,4 --run <dir> --resume

Protocol (PTB-XL benchmark style): random 2.5 s crops for training; at evaluation, sliding 2.5 s windows with
1.25 s stride over the 10 s record, sigmoid outputs averaged. Model selection on fold 9 (val) macro AUROC per epoch;
fold 10 (test) is scored once per seed with the selected checkpoint. Per-lead standardization from the train folds.
SIGUSR1 (sbatch --signal=B:USR1@N) saves a checkpoint and exits 99, so the same job file can be resubmitted.
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
from sklearn.metrics import roc_auc_score
from torch import nn

from .models import build

STOP = {"flag": False}


def _on_usr1(signum, frame):
    STOP["flag"] = True


def macro_auroc(y: np.ndarray, p: np.ndarray) -> tuple[float, list[float]]:
    per = []
    for k in range(y.shape[1]):
        per.append(float(roc_auc_score(y[:, k], p[:, k])) if 0 < y[:, k].sum() < len(y) else float("nan"))
    return float(np.nanmean(per)), per


def patient_bootstrap(y, p, patient, n_boot=1000, seed=0) -> dict:
    rng = np.random.default_rng(seed)
    uniq, inv = np.unique(patient, return_inverse=True)
    groups = [np.flatnonzero(inv == i) for i in range(len(uniq))]
    vals = []
    for _ in range(n_boot):
        idx = np.concatenate([groups[i] for i in rng.integers(0, len(groups), len(groups))])
        m, _ = macro_auroc(y[idx], p[idx])
        vals.append(m)
    vals = np.asarray(vals)
    return {"lo": float(np.nanpercentile(vals, 2.5)), "hi": float(np.nanpercentile(vals, 97.5)), "n_boot": n_boot,
            "unit": "patient", "n_patients": int(len(uniq))}


class Data:
    def __init__(self, path: Path, leads: str, smoke: bool, device):
        z = np.load(path, allow_pickle=False)
        X, y, fold, pat = z["X"], z["y"].astype(np.float32), z["fold"], z["patient"]
        self.classes = [str(c) for c in z["classes"]]
        all_leads = [str(c) for c in z["leads"]]
        li = list(range(12)) if leads == "all" else [all_leads.index(l.upper()) for l in leads.split(",")]
        self.leads = [all_leads[i] for i in li]
        X = X[:, li, :]
        tr, va, te = np.flatnonzero(fold <= 8), np.flatnonzero(fold == 9), np.flatnonzero(fold == 10)
        if smoke:
            tr, va, te = tr[:1024], va[:384], te[:384]
        xtr = X[tr].astype(np.float32)
        self.mean = xtr.mean(axis=(0, 2)).astype(np.float32)
        self.std = (xtr.std(axis=(0, 2)) + 1e-6).astype(np.float32)
        del xtr

        store = torch.float16 if device.type == "cuda" else torch.float32   # CPU convs have no fp16 path (smoke tests)

        def to_dev(idx):
            x = (X[idx].astype(np.float32) - self.mean[None, :, None]) / self.std[None, :, None]
            return torch.from_numpy(x).to(device=device, dtype=store)

        self.xtr, self.ytr = to_dev(tr), torch.from_numpy(y[tr]).to(device)
        self.xva, self.yva, self.pva = to_dev(va), y[va], pat[va]
        self.xte, self.yte, self.pte = to_dev(te), y[te], pat[te]
        self.n = {"train": len(tr), "val": len(va), "test": len(te)}


def crops(x: torch.Tensor, crop: int, gen: torch.Generator) -> torch.Tensor:
    b, c, t = x.shape
    off = torch.randint(0, t - crop + 1, (b,), device=x.device, generator=gen)
    idx = (off[:, None] + torch.arange(crop, device=x.device)[None, :])[:, None, :].expand(b, c, crop)
    return torch.gather(x, 2, idx)


@torch.no_grad()
def predict(model, x: torch.Tensor, crop: int, stride: int, bs: int = 512) -> np.ndarray:
    model.eval()
    starts = list(range(0, x.shape[2] - crop + 1, stride))
    out = []
    for i in range(0, len(x), bs):
        xb = x[i:i + bs]
        with torch.autocast("cuda", dtype=torch.float16, enabled=xb.is_cuda):
            ps = [torch.sigmoid(model(xb[:, :, s:s + crop]).float()) for s in starts]
        out.append(torch.stack(ps).mean(0).cpu().numpy())
    return np.concatenate(out)


def train_seed(a, d: Data, seed: int, outdir: Path, device) -> dict:
    outdir.mkdir(parents=True, exist_ok=True)
    done = outdir / "result.json"
    if done.exists():
        print(f"seed {seed}: already done, skipping", flush=True)
        return json.loads(done.read_text())
    torch.manual_seed(seed); np.random.seed(seed)
    gen = torch.Generator(device=device); gen.manual_seed(seed)
    model = build(a.arch, len(d.leads), len(d.classes)).to(device)
    epochs = 2 if a.smoke else a.epochs
    steps = max(1, d.n["train"] // a.bs)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.wd)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=epochs * steps)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    loss_fn = nn.BCEWithLogitsLoss()
    start, best, hist = 0, -1.0, []
    last = outdir / "last.pt"
    if a.resume and last.exists():
        ck = torch.load(last, map_location=device, weights_only=True)
        model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"]); sched.load_state_dict(ck["sched"])
        scaler.load_state_dict(ck["scaler"]); gen.set_state(ck["gen"].cpu())
        start, best, hist = ck["epoch"] + 1, ck["best"], ck["hist"]
        print(f"seed {seed}: resumed at epoch {start} (best val {best:.4f})", flush=True)

    def save(epoch):
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                    "scaler": scaler.state_dict(), "gen": gen.get_state(), "epoch": epoch, "best": best,
                    "hist": hist}, last.with_suffix(".tmp"))
        os.replace(last.with_suffix(".tmp"), last)

    for ep in range(start, epochs):
        model.train(); t0 = time.time(); tot = 0.0
        perm = torch.randperm(d.n["train"], device=device, generator=gen)
        for i in range(steps):
            idx = perm[i * a.bs:(i + 1) * a.bs]
            xb, yb = crops(d.xtr[idx], a.crop, gen), d.ytr[idx]
            with torch.autocast("cuda", dtype=torch.float16, enabled=device.type == "cuda"):
                loss = loss_fn(model(xb).float(), yb)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); sched.step()
            tot += float(loss)
            if STOP["flag"]:
                save(ep - 1)
                print(f"seed {seed}: SIGUSR1 at epoch {ep}, checkpoint saved; exiting 99", flush=True)
                sys.exit(99)
        val_auc, _ = macro_auroc(d.yva, predict(model, d.xva, a.crop, a.stride))
        hist.append({"epoch": ep, "loss": tot / steps, "val_macro_auroc": val_auc, "sec": round(time.time() - t0, 1)})
        if val_auc > best:
            best = val_auc
            torch.save(model.state_dict(), outdir / "best.pt")
        save(ep)
        print(f"seed {seed} epoch {ep + 1}/{epochs} loss {tot / steps:.4f} val {val_auc:.4f} best {best:.4f} "
              f"({time.time() - t0:.0f}s)", flush=True)

    model.load_state_dict(torch.load(outdir / "best.pt", map_location=device, weights_only=True))
    pv, pt = predict(model, d.xva, a.crop, a.stride), predict(model, d.xte, a.crop, a.stride)
    np.savez_compressed(outdir / "preds.npz", val=pv, test=pt)
    v, _ = macro_auroc(d.yva, pv)
    t, per = macro_auroc(d.yte, pt)
    res = {"seed": seed, "val_macro_auroc": v, "test_macro_auroc": t, "test_per_class": dict(zip(d.classes, per)),
           "epochs": epochs, "history": hist}
    done.write_text(json.dumps(res, indent=1))
    last.unlink(missing_ok=True)
    print(f"seed {seed}: val {v:.4f} test {t:.4f}", flush=True)
    return res


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True, type=Path)
    p.add_argument("--arch", default="xresnet1d101")
    p.add_argument("--leads", default="all", help="all, or comma list such as I")
    p.add_argument("--seeds", default="0")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--bs", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-2)
    p.add_argument("--wd", type=float, default=1e-2)
    p.add_argument("--crop", type=int, default=250)
    p.add_argument("--stride", type=int, default=125)
    p.add_argument("--run", required=True, type=Path)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--n-boot", type=int, default=1000)
    a = p.parse_args()
    signal.signal(signal.SIGUSR1, _on_usr1)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True
    print(f"device {device} {torch.cuda.get_device_name(0) if device.type == 'cuda' else ''} torch {torch.__version__}",
          flush=True)
    a.run.mkdir(parents=True, exist_ok=True)
    d = Data(a.data, a.leads, a.smoke, device)
    print(f"data {d.n} leads {d.leads} classes {d.classes}", flush=True)
    seeds = [int(s) for s in a.seeds.split(",")][: 1 if a.smoke else None]
    results = [train_seed(a, d, s, a.run / f"seed{s}", device) for s in seeds]

    preds = [np.load(a.run / f"seed{r['seed']}" / "preds.npz") for r in results]
    best_i = int(np.argmax([r["val_macro_auroc"] for r in results]))
    sel = preds[best_i]["test"]
    ens = np.mean([q["test"] for q in preds], axis=0)
    t_sel, per_sel = macro_auroc(d.yte, sel)
    t_ens, per_ens = macro_auroc(d.yte, ens)
    summary = {
        "task": "ecg_ptbxl_superclass", "arch": a.arch, "leads": d.leads, "classes": d.classes,
        "protocol": "official folds: train 1-8, val 9 (selection), test 10 (scored once per seed)",
        "n": d.n, "seeds": seeds, "per_seed": [{k: r[k] for k in ("seed", "val_macro_auroc", "test_macro_auroc")}
                                               for r in results],
        "selected_seed": results[best_i]["seed"],
        "selected": {"test_macro_auroc": t_sel, "per_class": dict(zip(d.classes, per_sel)),
                     "ci95": patient_bootstrap(d.yte, sel, d.pte, a.n_boot)},
        "ensemble": {"test_macro_auroc": t_ens, "per_class": dict(zip(d.classes, per_ens)),
                     "ci95": patient_bootstrap(d.yte, ens, d.pte, a.n_boot)},
        "baseline_beta_eva6_macro_auroc": 0.867,
        "normalization": {"mean": d.mean.tolist(), "std": d.std.tolist()},
        "window": {"fs_hz": 100, "crop_samples": a.crop, "stride_samples": a.stride, "record_samples": 1000},
        "smoke": a.smoke,
    }
    (a.run / "metrics.json").write_text(json.dumps(summary, indent=1))
    print(f"DONE selected seed {summary['selected_seed']} test {t_sel:.4f} "
          f"[{summary['selected']['ci95']['lo']:.3f}, {summary['selected']['ci95']['hi']:.3f}]; "
          f"ensemble test {t_ens:.4f}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
