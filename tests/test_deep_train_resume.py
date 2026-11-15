"""SIGUSR1 during an epoch must exit 99 without a mid-epoch save, so a resume finishes OneCycleLR exactly."""
from __future__ import annotations

from types import SimpleNamespace
from unittest import mock

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("sklearn")

from evade.training.deep import train as TR  # noqa: E402

N_TR, N_VA, BS, EPOCHS = 32, 16, 8, 3
STEPS = N_TR // BS


def _data():
    g = torch.Generator().manual_seed(0)
    yva = np.tile(np.array([[0, 1], [1, 0]], dtype=np.float32), (N_VA // 2, 1))   # both labels in every class
    return SimpleNamespace(
        n={"train": N_TR, "val": N_VA, "test": N_VA}, leads=["I"], classes=["A", "B"],
        xtr=torch.randn(N_TR, 1, 500, generator=g), ytr=torch.randint(0, 2, (N_TR, 2), generator=g).float(),
        xva=torch.randn(N_VA, 1, 500, generator=g), yva=yva,
        xte=torch.randn(N_VA, 1, 500, generator=g), yte=yva.copy())


def _args():
    return SimpleNamespace(arch="resnet1d_wang", smoke=False, epochs=EPOCHS, bs=BS, lr=1e-3, wd=1e-2,
                           crop=250, stride=125, resume=True)


def test_sigusr1_mid_epoch_exits_99_and_resume_completes(tmp_path):
    d, a, out = _data(), _args(), tmp_path / "seed0"
    real_crops, calls = TR.crops, {"n": 0}

    def crops_then_signal(*args, **kw):      # no SIGUSR1 on Windows: raise the flag during epoch 1, step 2
        calls["n"] += 1
        if calls["n"] == STEPS + 2:
            TR.STOP["flag"] = True
        return real_crops(*args, **kw)

    try:
        with mock.patch.object(TR, "crops", crops_then_signal), pytest.raises(SystemExit) as ex:
            TR.train_seed(a, d, 0, out, torch.device("cpu"))
        assert ex.value.code == 99
        ck = torch.load(out / "last.pt", map_location="cpu", weights_only=True)
        assert ck["epoch"] == 0 and ck["sched"]["_step_count"] - 1 == STEPS   # last finished epoch only
    finally:
        TR.STOP["flag"] = False
    res = TR.train_seed(a, d, 0, out, torch.device("cpu"))          # resumes at epoch 1 and must not overshoot
    assert [h["epoch"] for h in res["history"]] == list(range(EPOCHS))
    assert not (out / "last.pt").exists()
