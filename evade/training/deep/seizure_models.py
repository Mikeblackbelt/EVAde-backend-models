"""1D CNN for seizure-suspect windows over the raw 18-channel bipolar EEG window (B, 18, T), T = 4 s x 256 Hz.

Layout (design, not tuned):
  1. channel-wise temporal conv: a depthwise Conv1d (groups = 18), `per_ch` filters per bipolar channel, so each
     channel is filtered on its own first (no mixing of channels yet);
  2. spatial mixing: a 1x1 conv across all channel-filters (learned montage combinations);
  3. temporal residual stack: basic blocks (two k=7 convs, BN, ReLU, identity or 1x1 shortcut), downsampling 2x
     at stage starts;
  4. global average + max pooling, dropout, one logit (sigmoid outside the model; the API applies it).
"""
from __future__ import annotations

import torch
from torch import nn

from .models import conv_bn


class BasicBlock(nn.Module):
    def __init__(self, ni, nf, stride=1, k=7):
        super().__init__()
        self.convs = nn.Sequential(conv_bn(ni, nf, k, stride=stride), conv_bn(nf, nf, k, act=False, zero_bn=True))
        sc = []
        if stride != 1:
            sc.append(nn.AvgPool1d(stride, stride, ceil_mode=True))
        if ni != nf:
            sc.append(conv_bn(ni, nf, 1, act=False))
        self.short = nn.Sequential(*sc)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.act(self.convs(x) + self.short(x))


class SeizureNet(nn.Module):
    def __init__(self, c_in=18, n_out=1, per_ch=4, widths=(64, 64, 96, 96, 128, 128, 128, 128),
                 strides=(1, 1, 2, 1, 2, 1, 2, 1), k=7, p_drop=0.3):
        super().__init__()
        nc = c_in * per_ch
        self.channelwise = nn.Sequential(nn.Conv1d(c_in, nc, 15, stride=2, padding=7, groups=c_in, bias=False),
                                         nn.BatchNorm1d(nc), nn.ReLU(inplace=True))
        self.mix = conv_bn(nc, widths[0], 1)
        blocks, ni = [], widths[0]
        for nf, s in zip(widths, strides):
            blocks.append(BasicBlock(ni, nf, s, k)); ni = nf
        self.body = nn.Sequential(*blocks)
        self.avg, self.max = nn.AdaptiveAvgPool1d(1), nn.AdaptiveMaxPool1d(1)
        self.fc = nn.Sequential(nn.Flatten(), nn.Dropout(p_drop), nn.Linear(2 * ni, n_out))

    def forward(self, x):
        h = self.body(self.mix(self.channelwise(x)))
        return self.fc(torch.cat([self.avg(h), self.max(h)], 1))


def build(arch: str, c_in: int = 18, n_out: int = 1) -> nn.Module:
    if arch == "seizure_resnet1d":
        return SeizureNet(c_in, n_out)
    if arch == "seizure_tiny":            # tests and smoke runs only
        return SeizureNet(c_in, n_out, per_ch=2, widths=(16, 16, 24), strides=(1, 2, 2))
    raise ValueError(f"unknown arch {arch}")
