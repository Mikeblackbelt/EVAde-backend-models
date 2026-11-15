"""Train the deep sleep-staging model (per-epoch CNN + bi-GRU over sequences of epochs) on packed Sleep-EDFx SC.

  python -m evade.training.deep.sleep_train --data sleep_sc.npz --protocol kfold --run <dir> --resume
  python -m evade.training.deep.sleep_train --data sleep_sc.npz --protocol beta5 --val-frac 0 --run <dir> --resume
  python -m evade.training.deep.sleep_train --data sleep_sc.npz --protocol final --run <dir> --resume

Protocols (each a list of units run in sequence; a unit is one training run in <run>/<unit>/):
  kfold  subject-wise k-fold (default 5) over all subjects; both nights of a subject stay in one fold. The fold map
         is a seeded permutation of the sorted subjects (or --splits <csv subject,fold>), written to splits.csv with
         its sha256. Inside each training fold, --val-frac of the training subjects are held out; best.pt is the
         epoch with the highest inner-validation Cohen's kappa. The test fold is scored once, with best.pt.
  beta5  the beta's protocol for a fair comparison: leave-one-subject-out on SC400-SC404 only (the beta's 5
         subjects, 10 nights). With --val-frac 0 no subject is held out and the last epoch is used (the beta
         models had no selection step either).
  final  one model on all subjects minus --val-frac held out for checkpoint selection; this is what gets exported.
         Its expected performance is the kfold estimate, not its own validation score.

Input per epoch: Fpz-Cz and Pz-Oz in raw microvolts. The model normalizes inside its graph (sleep_models.epoch_robust_z):
a robust z-score per 30 s epoch and channel (median and 1.4826 x MAD over that epoch only, floor 0.1, clip +/-20). No
statistic depends on the night, the trimming or the labels, so the API reproduces it exactly from any chunk it gets.
Training samples random windows of --seq-len consecutive epochs inside one night; weighted cross-entropy (balanced
class weights from the training subjects); AdamW + one-cycle; fp16 autocast on GPU. Evaluation encodes every epoch
once and averages softmax over GRU windows of --seq-len with --eval-stride.
Metrics: pooled Cohen's kappa, macro F1, accuracy and per-class F1 with subject bootstrap 95% CIs (resampling
subjects' confusion matrices), plus the beta's statistic (mean of per-subject kappa with a subject bootstrap).
SIGUSR1 (sbatch --signal=B:USR1@N) exits 99 at once, keeping last.pt from the end of the last finished epoch, so the
same job file can be resubmitted and the one-cycle schedule resumes at the right step (dropout RNG is not restored).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import signal
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

from .sleep_models import CLIP, SCALE_FLOOR, build
from .sleep_pack import BETA_SUBJECTS, CLASSES

STOP = {"flag": False}
K = len(CLASSES)
# beta (EVA-3 HGB, EVA-4 logistic) on SC400-SC404, leave-one-subject-out, 10,589 epochs (evade/training/README.md)
BETA_BASELINE = {
    "protocol": "leave-one-subject-out on SC400-SC404 (5 subjects, 10 nights, 10,589 epochs), band-power features",
    "eva3_hgb": {"subject_mean_kappa": 0.669, "ci95": [0.545, 0.785], "pooled_kappa": 0.668, "macro_f1": 0.712,
                 "accuracy": 0.758},
    "eva4_logreg": {"subject_mean_kappa": 0.683, "ci95": [0.616, 0.745], "pooled_kappa": 0.685, "macro_f1": 0.730,
                    "accuracy": 0.767},
}


# the rule the model applies inside its graph (sleep_models.epoch_robust_z); the API sends raw uV ("none")
NORMALIZATION = {"type": "none", "in_model": {
    "type": "per_epoch_channel_robust_z", "center": "median", "scale": "1.4826 x MAD", "scale_floor": SCALE_FLOOR,
    "clip": CLIP, "over": "the 3,000 samples of each 30 s epoch and channel on their own (median of an even count = "
                          "mean of the two middle values); nothing depends on the night, the trimming or the labels"}}


def _on_usr1(signum, frame):
    STOP["flag"] = True


def _clean(o):
    """JSON-safe: numpy scalars to Python, NaN to None."""
    if isinstance(o, dict):
        return {str(k): _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, np.ndarray):
        return _clean(o.tolist())
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (float, np.floating)):
        return None if not math.isfinite(float(o)) else float(o)
    return o


# ---------------------------------------------------------------- metrics from confusion matrices
def confusion(y: np.ndarray, pred: np.ndarray) -> np.ndarray:
    return np.bincount(np.asarray(y) * K + np.asarray(pred), minlength=K * K).reshape(K, K)


def scores(c: np.ndarray) -> dict:
    """Accuracy, Cohen's kappa, macro F1 (over classes present in truth or prediction, as sklearn) and per-class F1."""
    n = c.sum()
    if n == 0:
        return {"accuracy": float("nan"), "kappa": float("nan"), "macro_f1": float("nan"),
                "per_class_f1": {k: float("nan") for k in CLASSES}, "n": 0}
    tp = np.diag(c).astype(float)
    po = tp.sum() / n
    pe = float(c.sum(1) @ c.sum(0)) / n ** 2
    kappa = (po - pe) / (1 - pe) if pe < 1 else float("nan")
    denom = 2 * tp + (c.sum(0) - tp) + (c.sum(1) - tp)
    f1 = np.where(denom > 0, 2 * tp / np.maximum(denom, 1), np.nan)
    return {"accuracy": float(po), "kappa": float(kappa), "macro_f1": float(np.nanmean(f1)),
            "per_class_f1": dict(zip(CLASSES, [float(v) for v in f1])), "n": int(n)}


def subject_bootstrap(confs: np.ndarray, n_boot: int = 2000, seed: int = 0) -> dict:
    """95% percentile CIs of pooled metrics, resampling subjects (their summed confusion matrices) with replacement."""
    rng = np.random.default_rng(seed)
    s = len(confs)
    rows = []
    for _ in range(n_boot):
        r = scores(confs[rng.integers(0, s, s)].sum(0))
        rows.append([r["kappa"], r["macro_f1"], r["accuracy"]] + [r["per_class_f1"][k] for k in CLASSES])
    v = np.asarray(rows, float)
    lo, hi = np.nanpercentile(v, 2.5, axis=0), np.nanpercentile(v, 97.5, axis=0)
    ci = lambda i: {"lo": float(lo[i]), "hi": float(hi[i])}   # noqa: E731
    return {"kappa": ci(0), "macro_f1": ci(1), "accuracy": ci(2),
            "per_class_f1": {k: ci(3 + i) for i, k in enumerate(CLASSES)},
            "unit": "subject", "n_subjects": int(s), "n_boot": n_boot}


def mean_ci(values, n_boot: int = 2000, seed: int = 0) -> dict:
    """The beta's statistic: mean over subjects with a subject bootstrap interval (evade.training.metrics)."""
    from ..metrics import subject_bootstrap_ci
    return subject_bootstrap_ci(values, n_boot=n_boot, seed=seed)


def evaluate(y, probs, subject, night, n_boot: int) -> dict:
    pred = probs.argmax(1)
    subs = sorted(set(subject))
    confs = np.stack([confusion(y[subject == s], pred[subject == s]) for s in subs])
    per = {s: scores(c) for s, c in zip(subs, confs)}
    tst = []
    for nt in sorted(set(night)):
        m = night == nt
        tst.append(((pred[m] != 0).sum() - (y[m] != 0).sum()) * 0.5)    # minutes (30 s epochs)
    tst = np.asarray(tst, float)
    return {"pooled": scores(confs.sum(0)), "confusion": confs.sum(0).tolist(),
            "ci95": subject_bootstrap(confs, n_boot),
            "subject_mean_kappa": mean_ci([per[s]["kappa"] for s in subs], n_boot),
            "subject_mean_macro_f1": mean_ci([per[s]["macro_f1"] for s in subs], n_boot),
            "per_subject": {s: {k: per[s][k] for k in ("kappa", "macro_f1", "accuracy", "n")} for s in subs},
            "night_total_sleep_time_error_min": {"mean_abs": float(np.abs(tst).mean()), "mean_bias": float(tst.mean()),
                                                 "n_nights": int(len(tst))}}


def operating_points(y, probs, fractions=(0.05, 0.10, 0.20)) -> list[dict]:
    """Stage = argmax. A low-confidence epoch (max softmax below a threshold) is marked for review; thresholds are
    set on the out-of-fold scores so that the given fraction of epochs is marked."""
    conf, pred = probs.max(1), probs.argmax(1)
    ok = pred == y
    out = [{"rule": "stage = argmax softmax", "accuracy": float(ok.mean()), "level": "stage estimate (human review)"}]
    for f in fractions:
        t = float(np.quantile(conf, f))
        flag = conf < t
        out.append({"rule": "review if max softmax < threshold", "threshold": t, "target_review_fraction": f,
                    "review_fraction": float(flag.mean()),
                    "accuracy_not_flagged": float(ok[~flag].mean()) if (~flag).any() else float("nan"),
                    "accuracy_flagged": float(ok[flag].mean()) if flag.any() else float("nan"), "level": "review"})
    return out


# ---------------------------------------------------------------- data
class Data:
    def __init__(self, path: Path, channels: list[str], device):
        z = np.load(path, allow_pickle=False)
        all_ch = [str(c) for c in z["channels"]]
        self.channels = channels
        X = z["X"][:, [all_ch.index(c) for c in channels], :]       # raw uV; the model normalizes each epoch
        self.samples = X.shape[2]
        self.y = z["y"].astype(np.int64)
        self.subject = z["subject"].astype(str)
        self.night = z["night"].astype(str)
        if [str(c) for c in z["classes"]] != CLASSES:
            raise ValueError(f"classes {list(z['classes'])} != {CLASSES}")
        cut = np.flatnonzero(self.night[1:] != self.night[:-1]) + 1
        self.n_start = np.concatenate([[0], cut]).astype(np.int64)
        self.n_len = np.diff(np.concatenate([self.n_start, [len(self.night)]])).astype(np.int64)
        self.night_names = self.night[self.n_start]
        if len(set(self.night_names)) != len(self.night_names):
            raise ValueError("nights are not contiguous blocks in the npz")
        self.night_subject = self.subject[self.n_start]
        self.x = torch.from_numpy(np.ascontiguousarray(X, dtype=np.float16)).to(device)
        self.subjects = sorted(set(self.night_subject))

    def nights_of(self, subjects) -> np.ndarray:
        return np.flatnonzero(np.isin(self.night_subject, list(subjects)))

    def epochs_of(self, nights) -> np.ndarray:
        return np.concatenate([np.arange(self.n_start[i], self.n_start[i] + self.n_len[i]) for i in nights]) \
            if len(nights) else np.empty(0, np.int64)


# ---------------------------------------------------------------- splits
def _inner(train: list[str], frac: float, seed: int) -> list[str]:
    if frac <= 0 or len(train) < 2:
        return []
    n = min(max(1, int(round(frac * len(train)))), len(train) - 1)
    return sorted(np.random.default_rng(seed).choice(sorted(train), n, replace=False).tolist())


def make_units(d: Data, a) -> tuple[list[dict], dict]:
    subs = d.subjects
    info: dict = {"protocol": a.protocol, "val_frac": a.val_frac, "split_seed": a.split_seed}
    units = []
    if a.protocol == "kfold":
        if a.splits:
            rows = [ln.split(",") for ln in a.splits.read_text().splitlines()[1:] if ln.strip()]
            fold = {r[0].strip(): int(r[1]) for r in rows}
            missing = [s for s in subs if s not in fold]
            if missing:
                raise ValueError(f"split file lacks subjects {missing[:5]}")
        else:
            perm = np.random.default_rng(a.split_seed).permutation(subs).tolist()
            fold = {s: i % a.folds for i, s in enumerate(perm)}
        k = max(fold[s] for s in subs) + 1
        text = "subject,fold\n" + "".join(f"{s},{fold[s]}\n" for s in subs)
        (a.run / "splits.csv").write_text(text)
        info.update({"k": k, "splits_file": str(a.splits) if a.splits else "generated: splits.csv",
                     "splits_sha256": hashlib.sha256(text.encode()).hexdigest()})
        for f in range(k):
            test = [s for s in subs if fold[s] == f]
            train = [s for s in subs if fold[s] != f]
            val = _inner(train, a.val_frac, a.split_seed + 1000 + f)
            units.append({"name": f"fold{f}", "train": [s for s in train if s not in val], "val": val, "test": test})
    elif a.protocol == "beta5":
        b = [s for s in BETA_SUBJECTS if s in subs]
        if len(b) < 2:
            raise ValueError(f"beta5 needs at least 2 of {BETA_SUBJECTS}; found {b}")
        for s in b:
            train = [t for t in b if t != s]
            val = _inner(train, a.val_frac, a.split_seed + 2000)
            units.append({"name": f"loso_{s}", "train": [t for t in train if t not in val], "val": val, "test": [s]})
        info["subjects"] = b
    elif a.protocol == "final":
        val = _inner(subs, a.val_frac, a.split_seed + 3000)
        units.append({"name": "final", "train": [s for s in subs if s not in val], "val": val, "test": []})
    else:
        raise ValueError(a.protocol)
    return units, info


# ---------------------------------------------------------------- train / predict
@torch.no_grad()
def predict(model, d: Data, nights, seq_len: int, stride: int) -> dict[int, np.ndarray]:
    """Softmax per epoch for each night: encode every epoch once, average GRU windows that contain it."""
    model.eval()
    dev = d.x.device
    cuda = dev.type == "cuda"
    out = {}
    for ni in nights:
        s, n = int(d.n_start[ni]), int(d.n_len[ni])
        x = d.x[s:s + n]
        with torch.autocast("cuda", dtype=torch.float16, enabled=cuda):
            f = torch.cat([model.encode(x[i:i + 512].float()).float() for i in range(0, n, 512)])
        L = min(seq_len, n)
        starts = list(range(0, n - L + 1, max(1, stride)))
        if starts[-1] != n - L:
            starts.append(n - L)
        st = torch.tensor(starts, device=dev)
        ar = torch.arange(L, device=dev)
        acc = torch.zeros(n, K, device=dev)
        cnt = torch.zeros(n, 1, device=dev)
        for j in range(0, len(starts), 256):
            idx = st[j:j + 256, None] + ar[None, :]
            with torch.autocast("cuda", dtype=torch.float16, enabled=cuda):
                pr = torch.softmax(model.context(f[idx]).float(), -1)
            acc.index_add_(0, idx.reshape(-1), pr.reshape(-1, K))
            cnt.index_add_(0, idx.reshape(-1), torch.ones(idx.numel(), 1, device=dev))
        out[int(ni)] = (acc / cnt).cpu().numpy()
    return out


def _stack(pred: dict, d: Data, nights) -> tuple[np.ndarray, np.ndarray]:
    idx = d.epochs_of(nights)
    return idx, np.concatenate([pred[int(i)] for i in nights]) if len(nights) else np.empty((0, K))


def train_unit(a, d: Data, unit: dict, outdir: Path, device) -> dict:
    outdir.mkdir(parents=True, exist_ok=True)
    done = outdir / "result.json"
    if done.exists():
        print(f"{unit['name']}: already done, skipping", flush=True)
        return json.loads(done.read_text())
    seed = a.seed
    torch.manual_seed(seed); np.random.seed(seed)
    gen = torch.Generator(device=device); gen.manual_seed(seed)
    L = a.seq_len
    tr_n = [i for i in d.nights_of(unit["train"]) if d.n_len[i] >= L]
    va_n, te_n = d.nights_of(unit["val"]), d.nights_of(unit["test"])
    if not tr_n:
        raise ValueError(f"{unit['name']}: no training night with at least {L} epochs")
    starts = torch.from_numpy(np.concatenate([d.n_start[i] + np.arange(d.n_len[i] - L + 1) for i in tr_n])).to(device)
    ytr = d.y[d.epochs_of(tr_n)]
    cnt = np.bincount(ytr, minlength=K).astype(float)
    w = np.where(cnt > 0, len(ytr) / (K * np.maximum(cnt, 1)), 0.0)
    ytorch = torch.from_numpy(d.y).to(device)
    model = build(a.arch, len(d.channels), K).to(device)
    epochs = 1 if a.smoke else a.epochs
    steps = max(1, len(ytr) // (L * a.bs))
    if a.smoke:
        steps = min(steps, 3)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.wd)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=epochs * steps)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    loss_fn = nn.CrossEntropyLoss(weight=torch.tensor(w, dtype=torch.float32, device=device))
    start, best, bad, hist = 0, -1.0, 0, []
    last = outdir / "last.pt"
    if a.resume and last.exists():
        ck = torch.load(last, map_location=device, weights_only=True)
        model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"]); sched.load_state_dict(ck["sched"])
        scaler.load_state_dict(ck["scaler"]); gen.set_state(ck["gen"].cpu())
        start, best, bad, hist = ck["epoch"] + 1, ck["best"], ck["bad"], ck["hist"]
        print(f"{unit['name']}: resumed at epoch {start} (best val kappa {best:.4f})", flush=True)

    def save(epoch):
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                    "scaler": scaler.state_dict(), "gen": gen.get_state(), "epoch": int(epoch), "best": float(best),
                    "bad": int(bad), "hist": hist}, last.with_suffix(".tmp"))
        last.with_suffix(".tmp").replace(last)

    ar = torch.arange(L, device=device)
    for ep in range(start, epochs):
        if a.patience and bad >= a.patience:
            break
        model.train(); t0 = time.time(); tot = 0.0
        for i in range(steps):
            sel = starts[torch.randint(0, len(starts), (a.bs,), device=device, generator=gen)]
            idx = sel[:, None] + ar[None, :]                      # (bs, L) consecutive epochs of one night
            xb, yb = d.x[idx].float(), ytorch[idx]
            with torch.autocast("cuda", dtype=torch.float16, enabled=device.type == "cuda"):
                logits = model(xb)
            loss = loss_fn(logits.float().reshape(-1, K), yb.reshape(-1))
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); sched.step()
            tot += float(loss)
            if STOP["flag"]:   # last.pt is from the end of the previous epoch: resuming repeats this epoch exactly
                print(f"{unit['name']}: SIGUSR1 in epoch {ep}; last.pt kept from epoch {ep - 1}; exiting 99", flush=True)
                sys.exit(99)
        rec = {"epoch": ep, "loss": tot / steps, "sec": round(time.time() - t0, 1)}
        if len(va_n):
            iv, pv = _stack(predict(model, d, va_n, L, a.eval_stride), d, va_n)
            vk = scores(confusion(d.y[iv], pv.argmax(1)))["kappa"]
            rec["val_kappa"] = float(vk) if math.isfinite(vk) else -1.0
            if rec["val_kappa"] > best:
                best, bad = rec["val_kappa"], 0
                torch.save(model.state_dict(), outdir / "best.pt")
            else:
                bad += 1
        else:                                                     # no selection: the last epoch is the model
            torch.save(model.state_dict(), outdir / "best.pt")
        hist.append(rec)
        save(ep)
        print(f"{unit['name']} epoch {ep + 1}/{epochs} loss {rec['loss']:.4f} val kappa {rec.get('val_kappa', float('nan')):.4f} "
              f"best {best:.4f} ({time.time() - t0:.0f}s)", flush=True)
        if STOP["flag"]:
            print(f"{unit['name']}: SIGUSR1 after epoch {ep}; exiting 99", flush=True)
            sys.exit(99)

    model.load_state_dict(torch.load(outdir / "best.pt", map_location=device, weights_only=True))
    res = {"unit": unit["name"], "train_subjects": unit["train"], "val_subjects": unit["val"],
           "test_subjects": unit["test"], "epochs_run": len(hist), "steps_per_epoch": steps,
           "best_val_kappa": best if len(va_n) else None,
           "best_epoch": (max(hist, key=lambda r: r.get("val_kappa", -1))["epoch"] if len(va_n) else hist[-1]["epoch"])
           if hist else None, "class_weights": dict(zip(CLASSES, w.tolist())), "history": hist}
    if len(te_n):
        it, pt = _stack(predict(model, d, te_n, L, a.eval_stride), d, te_n)
        np.savez_compressed(outdir / "preds.npz", idx=it, probs=pt.astype(np.float32))
        res["test"] = scores(confusion(d.y[it], pt.argmax(1)))
        print(f"{unit['name']}: test kappa {res['test']['kappa']:.4f} macro F1 {res['test']['macro_f1']:.4f}", flush=True)
    done.with_suffix(".tmp").write_text(json.dumps(_clean(res), indent=1))
    done.with_suffix(".tmp").replace(done)                       # atomic: resume trusts result.json
    last.unlink(missing_ok=True)
    return res


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True, type=Path)
    p.add_argument("--run", required=True, type=Path)
    p.add_argument("--protocol", required=True, choices=["kfold", "beta5", "final"])
    p.add_argument("--arch", default="cnn_bigru")
    p.add_argument("--channels", default="Fpz-Cz,Pz-Oz")
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--splits", type=Path, default=None, help="optional frozen csv: subject,fold")
    p.add_argument("--split-seed", type=int, default=0)
    p.add_argument("--val-frac", type=float, default=0.1)
    p.add_argument("--seq-len", type=int, default=20)
    p.add_argument("--eval-stride", type=int, default=5)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--patience", type=int, default=8, help="early stop after N epochs without val gain (0: off)")
    p.add_argument("--bs", type=int, default=32, help="sequences per step")
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--wd", type=float, default=1e-2)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--n-boot", type=int, default=2000)
    a = p.parse_args(argv)
    if hasattr(signal, "SIGUSR1"):
        signal.signal(signal.SIGUSR1, _on_usr1)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True
    print(f"device {device} {torch.cuda.get_device_name(0) if device.type == 'cuda' else ''} torch {torch.__version__}",
          flush=True)
    a.run.mkdir(parents=True, exist_ok=True)
    d = Data(a.data, [c.strip() for c in a.channels.split(",")], device)
    print(f"data {len(d.y)} epochs, {len(d.n_start)} nights, {len(d.subjects)} subjects, channels {d.channels}",
          flush=True)
    units, info = make_units(d, a)
    if a.smoke:
        units = units[:2]
    results = [train_unit(a, d, u, a.run / u["name"], device) for u in units]

    config = {"arch": a.arch, "channels": d.channels, "classes": CLASSES, "fs_hz": 100, "epoch_samples": d.samples,
              "seq_len": a.seq_len, "eval_stride": a.eval_stride,
              "normalization": NORMALIZATION,
              "hyper": {k: getattr(a, k) for k in ("epochs", "patience", "bs", "lr", "wd", "seed", "val_frac")}}
    summary = {"task": "sleep_staging", "protocol": info, "config": config, "smoke": a.smoke,
               "n": {"epochs": int(len(d.y)), "nights": int(len(d.n_start)), "subjects": len(d.subjects)},
               "class_counts": dict(zip(CLASSES, np.bincount(d.y, minlength=K).tolist())),
               "units": [{k: r.get(k) for k in ("unit", "best_epoch", "best_val_kappa", "epochs_run", "val_subjects",
                                                "test_subjects")}
                         | {"test_kappa": (r.get("test") or {}).get("kappa")} for r in results],
               "baseline_beta": BETA_BASELINE}
    tested = [u for u in units if u["test"]]
    if tested:
        idx = np.concatenate([np.load(a.run / u["name"] / "preds.npz")["idx"] for u in tested])
        probs = np.concatenate([np.load(a.run / u["name"] / "preds.npz")["probs"] for u in tested])
        y, sub, nt = d.y[idx], d.subject[idx], d.night[idx]
        summary["out_of_fold"] = evaluate(y, probs, sub, nt, a.n_boot)
        summary["operating_points"] = operating_points(y, probs)
        b = np.isin(sub, BETA_SUBJECTS)
        if a.protocol == "kfold" and b.any():
            summary["beta_subjects_within_kfold"] = evaluate(y[b], probs[b], sub[b], nt[b], a.n_boot)
        oof = summary["out_of_fold"]
        print(f"DONE {a.protocol}: pooled kappa {oof['pooled']['kappa']:.4f} "
              f"[{oof['ci95']['kappa']['lo']:.3f}, {oof['ci95']['kappa']['hi']:.3f}], macro F1 "
              f"{oof['pooled']['macro_f1']:.4f}, subject-mean kappa {oof['subject_mean_kappa']['mean']:.4f} "
              f"(beta HGB 0.669, logistic 0.683)", flush=True)
    else:
        print(f"DONE {a.protocol}: best val kappa {results[0].get('best_val_kappa')}", flush=True)
    (a.run / "metrics.json").write_text(json.dumps(_clean(summary), indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
