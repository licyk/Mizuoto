"""CAM++ as 3D-Speaker's torch model has it (``DTDNN.py``, ``layers.py``; Apache-2.0), batch norms unfolded:
the form IndexTTS-2 loads from ``campplus_cn_common.bin`` (CosyVoice's ONNX export is ``campplus.py``).

(batch, frames, 80) mean-normalised Kaldi fbank → (batch, embedding). Parameter names are upstream's.
"""

from __future__ import annotations

from collections import OrderedDict
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from mizuoto.runtime import ops as default_ops


def nonlinear(config: str, channels: int, ops: Any) -> nn.Sequential:
    seq = nn.Sequential()
    for name in config.split("-"):
        if name == "relu":
            seq.add_module("relu", nn.ReLU())
        elif name == "batchnorm":
            seq.add_module("batchnorm", ops.BatchNorm1d(channels))
        elif name == "batchnorm_":
            seq.add_module("batchnorm", ops.BatchNorm1d(channels, affine=False))
        else:
            raise ValueError(f"Unexpected module {name!r}")
    return seq


class BasicResBlock(nn.Module):
    def __init__(self, in_planes: int, planes: int, stride: int, ops: Any) -> None:
        super().__init__()
        self.conv1 = ops.Conv2d(in_planes, planes, kernel_size=3, stride=(stride, 1), padding=1, bias=False)
        self.bn1 = ops.BatchNorm2d(planes)
        self.conv2 = ops.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = ops.BatchNorm2d(planes)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(ops.Conv2d(in_planes, planes, kernel_size=1, stride=(stride, 1), bias=False), ops.BatchNorm2d(planes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        return F.relu(out)


class FCM(nn.Module):
    def __init__(self, ops: Any, m_channels: int = 32, feat_dim: int = 80) -> None:
        super().__init__()
        self.conv1 = ops.Conv2d(1, m_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = ops.BatchNorm2d(m_channels)
        self.layer1 = nn.Sequential(BasicResBlock(m_channels, m_channels, 2, ops), BasicResBlock(m_channels, m_channels, 1, ops))
        self.layer2 = nn.Sequential(BasicResBlock(m_channels, m_channels, 2, ops), BasicResBlock(m_channels, m_channels, 1, ops))
        self.conv2 = ops.Conv2d(m_channels, m_channels, kernel_size=3, stride=(2, 1), padding=1, bias=False)
        self.bn2 = ops.BatchNorm2d(m_channels)
        self.out_channels = m_channels * (feat_dim // 8)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(self.conv1(x.unsqueeze(1))))
        out = self.layer2(self.layer1(out))
        out = F.relu(self.bn2(self.conv2(out)))
        return out.reshape(out.shape[0], out.shape[1] * out.shape[2], out.shape[3])


class TDNNLayer(nn.Module):
    def __init__(self, cin: int, cout: int, kernel_size: int, stride: int, ops: Any) -> None:
        super().__init__()
        self.linear = ops.Conv1d(cin, cout, kernel_size, stride=stride, padding=(kernel_size - 1) // 2, bias=False)
        self.nonlinear = nonlinear("batchnorm-relu", cout, ops)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.nonlinear(self.linear(x))


class CAMLayer(nn.Module):
    def __init__(self, bn_channels: int, cout: int, kernel_size: int, padding: int, dilation: int, ops: Any) -> None:
        super().__init__()
        self.linear_local = ops.Conv1d(bn_channels, cout, kernel_size, padding=padding, dilation=dilation, bias=False)
        self.linear1 = ops.Conv1d(bn_channels, bn_channels // 2, 1)
        self.linear2 = ops.Conv1d(bn_channels // 2, cout, 1)

    @staticmethod
    def seg_pooling(x: torch.Tensor, seg_len: int = 100) -> torch.Tensor:
        seg = F.avg_pool1d(x, kernel_size=seg_len, stride=seg_len, ceil_mode=True)
        shape = seg.shape
        seg = seg.unsqueeze(-1).expand(*shape, seg_len).reshape(*shape[:-1], -1)
        return seg[..., : x.shape[-1]]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.linear_local(x)
        context = x.mean(-1, keepdim=True) + self.seg_pooling(x)
        context = F.relu(self.linear1(context))
        return y * torch.sigmoid(self.linear2(context))


class CAMDenseTDNNLayer(nn.Module):
    def __init__(self, cin: int, cout: int, bn_channels: int, kernel_size: int, dilation: int, ops: Any) -> None:
        super().__init__()
        self.nonlinear1 = nonlinear("batchnorm-relu", cin, ops)
        self.linear1 = ops.Conv1d(cin, bn_channels, 1, bias=False)
        self.nonlinear2 = nonlinear("batchnorm-relu", bn_channels, ops)
        self.cam_layer = CAMLayer(bn_channels, cout, kernel_size, (kernel_size - 1) // 2 * dilation, dilation, ops)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.cam_layer(self.nonlinear2(self.linear1(self.nonlinear1(x))))


class CAMDenseTDNNBlock(nn.ModuleList):
    def __init__(self, num_layers: int, cin: int, cout: int, bn_channels: int, kernel_size: int, dilation: int, ops: Any) -> None:
        super().__init__()
        for i in range(num_layers):
            self.add_module(f"tdnnd{i + 1}", CAMDenseTDNNLayer(cin + i * cout, cout, bn_channels, kernel_size, dilation, ops))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self:
            x = torch.cat([x, layer(x)], dim=1)
        return x


class TransitLayer(nn.Module):
    def __init__(self, cin: int, cout: int, ops: Any) -> None:
        super().__init__()
        self.nonlinear = nonlinear("batchnorm-relu", cin, ops)
        self.linear = ops.Conv1d(cin, cout, 1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(self.nonlinear(x))


class StatsPool(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.cat([x.mean(dim=-1), x.std(dim=-1, unbiased=True)], dim=-1)


class DenseLayer(nn.Module):
    def __init__(self, cin: int, cout: int, ops: Any) -> None:
        super().__init__()
        self.linear = ops.Conv1d(cin, cout, 1, bias=False)
        self.nonlinear = nonlinear("batchnorm_", cout, ops)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear(x.unsqueeze(dim=-1)).squeeze(dim=-1) if x.dim() == 2 else self.linear(x)
        return self.nonlinear(x)


class CAMPPlus(nn.Module):
    def __init__(self, feat_dim: int = 80, embedding_size: int = 192, growth_rate: int = 32, bn_size: int = 4, init_channels: int = 128, ops: Any = default_ops.default) -> None:
        super().__init__()
        self.head = FCM(ops, feat_dim=feat_dim)
        channels = self.head.out_channels
        self.xvector = nn.Sequential(OrderedDict([("tdnn", TDNNLayer(channels, init_channels, 5, 2, ops))]))
        channels = init_channels
        for i, (num_layers, kernel_size, dilation) in enumerate(zip((12, 24, 16), (3, 3, 3), (1, 2, 2), strict=True)):
            self.xvector.add_module(f"block{i + 1}", CAMDenseTDNNBlock(num_layers, channels, growth_rate, bn_size * growth_rate, kernel_size, dilation, ops))
            channels = channels + num_layers * growth_rate
            self.xvector.add_module(f"transit{i + 1}", TransitLayer(channels, channels // 2, ops))
            channels //= 2
        self.xvector.add_module("out_nonlinear", nonlinear("batchnorm-relu", channels, ops))
        self.xvector.add_module("stats", StatsPool())
        self.xvector.add_module("dense", DenseLayer(channels * 2, embedding_size, ops))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.xvector(self.head(x.permute(0, 2, 1)))
