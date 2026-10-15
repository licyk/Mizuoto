"""CAM++, the speaker embedding of CosyVoice and IndexTTS (3D-Speaker's ``CAMPPlus``, Apache-2.0).

80-bin Kaldi fbank frames (mean-normalised) → a 192-dimension embedding. The module mirrors the graph
CosyVoice ships as ``campplus.onnx``: every batch norm that directly follows a convolution is folded into
it (so those convolutions have a bias and the norm is gone: the head's, the TDNN's, each dense layer's
second, and ``out_nonlinear``'s into the last transit layer), while the norms that precede a convolution
and the final affine-free one remain. ``state_dict_from_onnx`` names the graph's weights after these modules.

The ONNX runtime computes with its own kernels, so embeddings agree to float precision, not bit for bit.
"""

from __future__ import annotations

from collections import OrderedDict
from typing import TYPE_CHECKING, Any

import torch
from torch import nn
from torch.nn import functional as F

from mizuoto.runtime import ops as default_ops

if TYPE_CHECKING:
    from mizuoto.runtime.convert.onnx import OnnxGraph


def _bn_relu(channels: int, ops: Any) -> nn.Sequential:
    seq = nn.Sequential()
    seq.add_module("batchnorm", ops.BatchNorm1d(channels))
    seq.add_module("relu", nn.ReLU())
    return seq


class BasicResBlock(nn.Module):
    def __init__(self, in_planes: int, planes: int, stride: int, ops: Any) -> None:
        super().__init__()
        self.conv1 = ops.Conv2d(in_planes, planes, kernel_size=3, stride=(stride, 1), padding=1)
        self.conv2 = ops.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut.add_module("0", ops.Conv2d(in_planes, planes, kernel_size=1, stride=(stride, 1)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv2(F.relu(self.conv1(x)))
        return F.relu(out + self.shortcut(x))


class FCM(nn.Module):
    def __init__(self, ops: Any, m_channels: int = 32, feat_dim: int = 80) -> None:
        super().__init__()
        self.conv1 = ops.Conv2d(1, m_channels, kernel_size=3, stride=1, padding=1)
        self.layer1 = nn.Sequential(BasicResBlock(m_channels, m_channels, 2, ops), BasicResBlock(m_channels, m_channels, 1, ops))
        self.layer2 = nn.Sequential(BasicResBlock(m_channels, m_channels, 2, ops), BasicResBlock(m_channels, m_channels, 1, ops))
        self.conv2 = ops.Conv2d(m_channels, m_channels, kernel_size=3, stride=(2, 1), padding=1)
        self.out_channels = m_channels * (feat_dim // 8)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.conv1(x.unsqueeze(1)))
        out = F.relu(self.conv2(self.layer2(self.layer1(out))))
        shape = out.shape
        return out.reshape(shape[0], shape[1] * shape[2], shape[3])


class TDNNLayer(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int, ops: Any) -> None:
        super().__init__()
        self.linear = ops.Conv1d(in_channels, out_channels, kernel_size, stride=stride, padding=(kernel_size - 1) // 2)
        self.nonlinear = nn.Sequential(OrderedDict([("relu", nn.ReLU())]))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.nonlinear(self.linear(x))


class CAMLayer(nn.Module):
    def __init__(self, bn_channels: int, out_channels: int, kernel_size: int, padding: int, dilation: int, ops: Any, reduction: int = 2) -> None:
        super().__init__()
        self.linear_local = ops.Conv1d(bn_channels, out_channels, kernel_size, padding=padding, dilation=dilation, bias=False)
        self.linear1 = ops.Conv1d(bn_channels, bn_channels // reduction, 1)
        self.linear2 = ops.Conv1d(bn_channels // reduction, out_channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.linear_local(x)
        context = x.mean(-1, keepdim=True) + self._seg_pooling(x)
        m = torch.sigmoid(self.linear2(F.relu(self.linear1(context))))
        return y * m

    @staticmethod
    def _seg_pooling(x: torch.Tensor, seg_len: int = 100) -> torch.Tensor:
        seg = F.avg_pool1d(x, kernel_size=seg_len, stride=seg_len, ceil_mode=True)
        shape = seg.shape
        seg = seg.unsqueeze(-1).expand(*shape, seg_len).reshape(*shape[:-1], -1)
        return seg[..., : x.shape[-1]]


class CAMDenseTDNNLayer(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, bn_channels: int, kernel_size: int, dilation: int, ops: Any) -> None:
        super().__init__()
        self.nonlinear1 = _bn_relu(in_channels, ops)
        self.linear1 = ops.Conv1d(in_channels, bn_channels, 1)  # nonlinear2's batch norm folded in
        self.cam_layer = CAMLayer(bn_channels, out_channels, kernel_size, (kernel_size - 1) // 2 * dilation, dilation, ops)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.cam_layer(F.relu(self.linear1(self.nonlinear1(x))))


class CAMDenseTDNNBlock(nn.ModuleList):
    def __init__(self, num_layers: int, in_channels: int, out_channels: int, bn_channels: int, kernel_size: int, dilation: int, ops: Any) -> None:
        super().__init__()
        for i in range(num_layers):
            self.add_module(f"tdnnd{i + 1}", CAMDenseTDNNLayer(in_channels + i * out_channels, out_channels, bn_channels, kernel_size, dilation, ops))

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        for layer in self:
            x = torch.cat([x, layer(x)], dim=1)
        return x


class TransitLayer(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, ops: Any, bias: bool = False) -> None:
        super().__init__()
        self.nonlinear = _bn_relu(in_channels, ops)
        self.linear = ops.Conv1d(in_channels, out_channels, 1, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(self.nonlinear(x))


class StatsPool(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.cat([x.mean(dim=-1), x.std(dim=-1, unbiased=True)], dim=-1)


class DenseLayer(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, ops: Any) -> None:
        super().__init__()
        self.linear = ops.Conv1d(in_channels, out_channels, 1, bias=False)
        self.nonlinear = nn.Sequential(OrderedDict([("batchnorm", ops.BatchNorm1d(out_channels))]))  # affine-free: the export stores 1 and 0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.nonlinear(self.linear(x.unsqueeze(dim=-1)).squeeze(dim=-1))


class CAMPPlus(nn.Module):
    """(batch, frames, 80) fbank → (batch, 192)."""

    def __init__(self, feat_dim: int = 80, embedding_size: int = 192, growth_rate: int = 32, bn_size: int = 4, init_channels: int = 128, ops: Any = default_ops.default) -> None:
        super().__init__()
        self.head = FCM(ops, feat_dim=feat_dim)
        channels = self.head.out_channels
        self.xvector = nn.Sequential(OrderedDict([("tdnn", TDNNLayer(channels, init_channels, 5, stride=2, ops=ops))]))
        channels = init_channels
        for i, (num_layers, kernel_size, dilation) in enumerate(zip((12, 24, 16), (3, 3, 3), (1, 2, 2), strict=True)):
            self.xvector.add_module(f"block{i + 1}", CAMDenseTDNNBlock(num_layers, channels, growth_rate, bn_size * growth_rate, kernel_size, dilation, ops))
            channels = channels + num_layers * growth_rate
            # the last transit layer takes out_nonlinear's batch norm, which follows it
            self.xvector.add_module(f"transit{i + 1}", TransitLayer(channels, channels // 2, ops, bias=i == 2))
            channels //= 2
        self.xvector.add_module("out_nonlinear", nn.Sequential(OrderedDict([("relu", nn.ReLU())])))
        self.xvector.add_module("stats", StatsPool())
        self.xvector.add_module("dense", DenseLayer(channels * 2, embedding_size, ops))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.xvector(self.head(x.permute(0, 2, 1)))


def _module_path(node_name: str) -> str:
    """``/head/layer1/layer1.0/conv1/Conv`` → ``head.layer1.0.conv1``."""
    parts = [p for p in node_name.split("/") if p][:-1]
    out: list[str] = []
    for i, p in enumerate(parts):
        if i and p.startswith(parts[i - 1] + "."):
            p = p[len(parts[i - 1]) + 1 :]
        out.append(p)
    return ".".join(out)


def state_dict_from_onnx(graph: OnnxGraph) -> dict[str, torch.Tensor]:
    """``campplus.onnx``'s weights, named after ``CAMPPlus``'s modules (from the nodes that use them)."""
    sd: dict[str, torch.Tensor] = {}
    for node in graph.nodes:
        if node.op_type == "Conv":
            path = _module_path(node.name)
            sd[f"{path}.weight"] = torch.from_numpy(graph.tensor(node.inputs[1]))
            if len(node.inputs) > 2 and node.inputs[2]:
                sd[f"{path}.bias"] = torch.from_numpy(graph.tensor(node.inputs[2]))
        elif node.op_type == "BatchNormalization":
            path = _module_path(node.name)
            for key, name in zip(("weight", "bias", "running_mean", "running_var"), node.inputs[1:5], strict=True):
                sd[f"{path}.{key}"] = torch.from_numpy(graph.tensor(name))
    return sd
