"""Sleep staging network: a small per-epoch 1D CNN (TinySleepNet-style front end) plus a bidirectional GRU over a
sequence of epochs. Input (batch, seq, channels, 3000) at 100 Hz; output logits (batch, seq, 5) for W N1 N2 N3 REM.

The encoder sees one 30 s epoch at a time, so evaluation can encode every epoch of a night once and run only the
GRU over overlapping windows (encode + context); forward() is the same computation in one call (used for export).
"""
from __future__ import annotations

import torch
from torch import nn


def conv_bn(ni, nf, k, stride=1):
    return nn.Sequential(nn.Conv1d(ni, nf, k, stride=stride, padding=k // 2, bias=False), nn.BatchNorm1d(nf),
                         nn.ReLU(inplace=True))


class EpochEncoder(nn.Module):
    """(N, C, 3000) -> (N, nf * 15). First conv: 0.49 s kernel, 60 ms stride (500 steps per epoch)."""

    def __init__(self, c_in: int = 2, nf: int = 128, drop: float = 0.5, samples: int = 3000):
        super().__init__()
        self.net = nn.Sequential(conv_bn(c_in, nf, 49, stride=6), nn.MaxPool1d(8, 8), nn.Dropout(drop),
                                 conv_bn(nf, nf, 7), conv_bn(nf, nf, 7), conv_bn(nf, nf, 7),
                                 nn.MaxPool1d(4, 4), nn.Flatten(), nn.Dropout(drop))
        with torch.no_grad():
            self.out_dim = int(self.net(torch.zeros(2, c_in, samples)).shape[1])

    def forward(self, x):
        return self.net(x)


class SleepSeqNet(nn.Module):
    def __init__(self, c_in: int = 2, n_out: int = 5, nf: int = 128, hidden: int = 128, drop: float = 0.5,
                 samples: int = 3000):
        super().__init__()
        self.enc = EpochEncoder(c_in, nf, drop, samples)
        self.gru = nn.GRU(self.enc.out_dim, hidden, batch_first=True, bidirectional=True)
        self.drop = nn.Dropout(drop)
        self.fc = nn.Linear(2 * hidden, n_out)

    def encode(self, x: torch.Tensor) -> torch.Tensor:          # (N, C, T) -> (N, F)
        return self.enc(x)

    def context(self, f: torch.Tensor) -> torch.Tensor:         # (B, L, F) -> (B, L, K)
        h, _ = self.gru(f)
        return self.fc(self.drop(h))

    def forward(self, x: torch.Tensor) -> torch.Tensor:         # (B, L, C, T) -> (B, L, K)
        b, l = x.shape[0], x.shape[1]
        f = self.encode(x.flatten(0, 1))
        return self.context(f.reshape(b, l, f.shape[-1]))


def build(arch: str, c_in: int, n_out: int = 5) -> nn.Module:
    if arch == "cnn_bigru":
        return SleepSeqNet(c_in, n_out)
    raise ValueError(f"unknown arch {arch}")
