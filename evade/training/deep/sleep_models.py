"""Sleep staging network: a small per-epoch 1D CNN (TinySleepNet-style front end) plus a bidirectional GRU over a
sequence of epochs. Input (batch, seq, channels, 3000) at 100 Hz; output logits (batch, seq, 5) for W N1 N2 N3 REM.

The encoder sees one 30 s epoch at a time, so evaluation can encode every epoch of a night once and run only the
GRU over overlapping windows (encode + context); forward() is the same computation in one call (used for export).

Normalization is part of the model: encode() first applies epoch_robust_z to each 30 s epoch and channel on its own
(median and 1.4826 x MAD over that epoch's 3,000 samples, floor 0.1, clip +/-20). The statistics depend only on the
epoch itself, so training, evaluation, the ONNX export and the API (which sends raw microvolts and is told
normalization "none") all compute exactly the same thing, whatever chunk of the night is submitted.
"""
from __future__ import annotations

import torch
from torch import nn

SCALE_FLOOR = 0.1     # microvolts: a flat (disconnected) epoch is divided by this, then clipped
CLIP = 20.0


def _median_last(x: torch.Tensor) -> torch.Tensor:
    """Median over the last axis, the numpy way (mean of the two middle values for an even length). Built from sort
    so it exports to ONNX (TopK) at any opset the exporter supports."""
    n = x.shape[-1]
    s = torch.sort(x, dim=-1).values
    return 0.5 * (s[..., (n - 1) // 2] + s[..., n // 2]).unsqueeze(-1)


def epoch_robust_z(x: torch.Tensor) -> torch.Tensor:
    """(N, C, T) raw signal -> per-epoch, per-channel robust z-score, in float32."""
    x = x.float()
    med = _median_last(x)
    mad = _median_last((x - med).abs())
    return torch.clamp((x - med) / torch.clamp(1.4826 * mad, min=SCALE_FLOOR), -CLIP, CLIP)


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

    def encode(self, x: torch.Tensor) -> torch.Tensor:          # (N, C, T) raw uV -> (N, F)
        return self.enc(epoch_robust_z(x))

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
