"""1D CNNs for ECG, following the PTB-XL benchmark (Strodthoff et al., 2021): xresnet1d and resnet1d_wang."""
from __future__ import annotations

import torch
from torch import nn


def conv_bn(ni, nf, k, stride=1, act=True, zero_bn=False):
    bn = nn.BatchNorm1d(nf)
    nn.init.constant_(bn.weight, 0.0 if zero_bn else 1.0)
    layers = [nn.Conv1d(ni, nf, k, stride=stride, padding=k // 2, bias=False), bn]
    if act:
        layers.append(nn.ReLU(inplace=True))
    return nn.Sequential(*layers)


class XBlock(nn.Module):
    """Bottleneck residual block (expansion 4) with an avg-pool shortcut, as in fastai's XResNet."""

    def __init__(self, ni, nh, stride, k=5, expansion=4):
        super().__init__()
        nf = nh * expansion
        self.convs = nn.Sequential(conv_bn(ni, nh, 1), conv_bn(nh, nh, k, stride=stride),
                                   conv_bn(nh, nf, 1, act=False, zero_bn=True))
        sc = []
        if stride != 1:
            sc.append(nn.AvgPool1d(stride, stride, ceil_mode=True))
        if ni != nf:
            sc.append(conv_bn(ni, nf, 1, act=False))
        self.short = nn.Sequential(*sc)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.act(self.convs(x) + self.short(x))


class Head(nn.Module):
    def __init__(self, nf, n_out, ps=0.5):
        super().__init__()
        self.avg, self.max = nn.AdaptiveAvgPool1d(1), nn.AdaptiveMaxPool1d(1)
        self.fc = nn.Sequential(nn.Flatten(), nn.BatchNorm1d(2 * nf), nn.Dropout(ps / 2), nn.Linear(2 * nf, 128),
                                nn.ReLU(inplace=True), nn.BatchNorm1d(128), nn.Dropout(ps), nn.Linear(128, n_out))

    def forward(self, x):
        return self.fc(torch.cat([self.avg(x), self.max(x)], 1))


class XResNet1d(nn.Module):
    def __init__(self, layers, c_in=12, n_out=5, k=5):
        super().__init__()
        stem = [conv_bn(c_in, 32, k, stride=2), conv_bn(32, 32, k), conv_bn(32, 64, k)]
        self.stem = nn.Sequential(*stem, nn.MaxPool1d(3, 2, 1))
        widths, ni, blocks = [64, 128, 256, 512], 64, []
        for i, (nh, n) in enumerate(zip(widths, layers)):
            for j in range(n):
                blocks.append(XBlock(ni, nh, stride=2 if (j == 0 and i > 0) else 1, k=k))
                ni = nh * 4
        self.body = nn.Sequential(*blocks)
        self.head = Head(ni, n_out)

    def forward(self, x):
        return self.head(self.body(self.stem(x)))


class WangBlock(nn.Module):
    def __init__(self, ni, nf):
        super().__init__()
        self.convs = nn.Sequential(conv_bn(ni, nf, 7), conv_bn(nf, nf, 5), conv_bn(nf, nf, 3, act=False))
        self.short = conv_bn(ni, nf, 1, act=False) if ni != nf else nn.Identity()
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.act(self.convs(x) + self.short(x))


class ResNet1dWang(nn.Module):
    def __init__(self, c_in=12, n_out=5):
        super().__init__()
        self.body = nn.Sequential(WangBlock(c_in, 64), WangBlock(64, 128), WangBlock(128, 128))
        self.head = Head(128, n_out)

    def forward(self, x):
        return self.head(self.body(x))


def build(arch: str, c_in: int, n_out: int) -> nn.Module:
    if arch == "xresnet1d101":
        return XResNet1d([3, 4, 23, 3], c_in, n_out)
    if arch == "xresnet1d50":
        return XResNet1d([3, 4, 6, 3], c_in, n_out)
    if arch == "resnet1d_wang":
        return ResNet1dWang(c_in, n_out)
    raise ValueError(f"unknown arch {arch}")
