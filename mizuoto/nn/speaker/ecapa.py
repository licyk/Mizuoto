"""ECAPA-TDNN (SpeechBrain, Apache-2.0) as IndexTTS-1.5's BigVGAN keeps it (``BigVGAN/ECAPA_TDNN.py``, ``nnet/``).

Mel frames (batch, frames, mels) → (batch, 1, embedding). SpeechBrain's convolutions pad "same" with
reflection (the amount computed from the channel count, which gives ``dilation·(kernel−1)/2`` either way);
batch norms follow the activation; attentive statistics pooling with global context. Parameter names are
upstream's (``blocks.0.conv.conv.weight``, ``asp_bn.norm.running_mean``…).
"""

from __future__ import annotations

import math
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from mizuoto.runtime import ops as default_ops


class SbConv1d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, ops: Any, dilation: int = 1, groups: int = 1) -> None:
        super().__init__()
        self.conv = ops.Conv1d(in_channels, out_channels, kernel_size, dilation=dilation, groups=groups)
        self.in_channels, self.kernel_size, self.dilation = in_channels, kernel_size, dilation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        l_out = math.floor((self.in_channels - self.dilation * (self.kernel_size - 1) - 1) / 1) + 1
        pad = math.floor((self.in_channels - l_out) / 2)
        return self.conv(F.pad(x, [pad, pad], mode="reflect"))


class SbBatchNorm1d(nn.Module):
    def __init__(self, size: int, ops: Any) -> None:
        super().__init__()
        self.norm = ops.BatchNorm1d(size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(x)


class TDNNBlock(nn.Module):
    def __init__(self, cin: int, cout: int, kernel_size: int, dilation: int, ops: Any, groups: int = 1) -> None:
        super().__init__()
        self.conv = SbConv1d(cin, cout, kernel_size, ops, dilation=dilation, groups=groups)
        self.norm = SbBatchNorm1d(cout, ops)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(F.relu(self.conv(x)))


class Res2NetBlock(nn.Module):
    def __init__(self, channels: int, scale: int, kernel_size: int, dilation: int, ops: Any) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(TDNNBlock(channels // scale, channels // scale, kernel_size, dilation, ops) for _ in range(scale - 1))
        self.scale = scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y: list[torch.Tensor] = []
        y_i = x
        for i, x_i in enumerate(torch.chunk(x, self.scale, dim=1)):
            if i == 0:
                y_i = x_i
            elif i == 1:
                y_i = self.blocks[i - 1](x_i)
            else:
                y_i = self.blocks[i - 1](x_i + y_i)
            y.append(y_i)
        return torch.cat(y, dim=1)


class SEBlock(nn.Module):
    def __init__(self, channels: int, se_channels: int, ops: Any) -> None:
        super().__init__()
        self.conv1 = SbConv1d(channels, se_channels, 1, ops)
        self.conv2 = SbConv1d(se_channels, channels, 1, ops)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        s = x.mean(dim=2, keepdim=True)
        s = torch.sigmoid(self.conv2(F.relu(self.conv1(s))))
        return s * x


class SERes2NetBlock(nn.Module):
    def __init__(self, cin: int, cout: int, scale: int, se_channels: int, kernel_size: int, dilation: int, ops: Any) -> None:
        super().__init__()
        self.tdnn1 = TDNNBlock(cin, cout, 1, 1, ops)
        self.res2net_block = Res2NetBlock(cout, scale, kernel_size, dilation, ops)
        self.tdnn2 = TDNNBlock(cout, cout, 1, 1, ops)
        self.se_block = SEBlock(cout, se_channels, ops)
        self.shortcut: nn.Module | None = SbConv1d(cin, cout, 1, ops) if cin != cout else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.shortcut(x) if self.shortcut is not None else x
        x = self.se_block(self.tdnn2(self.res2net_block(self.tdnn1(x))))
        return x + residual


class AttentiveStatisticsPooling(nn.Module):
    def __init__(self, channels: int, attention_channels: int, ops: Any) -> None:
        super().__init__()
        self.eps = 1e-12
        self.tdnn = TDNNBlock(channels * 3, attention_channels, 1, 1, ops)
        self.conv = SbConv1d(attention_channels, channels, 1, ops)

    def _stats(self, x: torch.Tensor, m: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mean = (m * x).sum(2)
        std = torch.sqrt((m * (x - mean.unsqueeze(2)).pow(2)).sum(2).clamp(self.eps))
        return mean, std

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        length = x.shape[-1]
        lengths = torch.ones(x.shape[0], device=x.device)
        scaled = lengths * length
        mask = torch.arange(length, device=x.device, dtype=scaled.dtype).expand(len(scaled), length) < scaled.unsqueeze(1)
        mask = torch.as_tensor(mask, dtype=scaled.dtype, device=x.device).unsqueeze(1)
        total = mask.sum(dim=2, keepdim=True).float()
        mean, std = self._stats(x, mask / total)
        attn = torch.cat([x, mean.unsqueeze(2).repeat(1, 1, length), std.unsqueeze(2).repeat(1, 1, length)], dim=1)
        attn = self.conv(torch.tanh(self.tdnn(attn)))
        attn = attn.masked_fill(mask == 0, float("-inf"))
        attn = F.softmax(attn, dim=2)
        mean, std = self._stats(x, attn)
        return torch.cat((mean, std), dim=1).unsqueeze(2)


class ECAPA_TDNN(nn.Module):
    def __init__(
        self,
        input_size: int,
        lin_neurons: int = 192,
        channels: tuple[int, ...] = (512, 512, 512, 512, 1536),
        kernel_sizes: tuple[int, ...] = (5, 3, 3, 3, 1),
        dilations: tuple[int, ...] = (1, 2, 3, 4, 1),
        attention_channels: int = 128,
        res2net_scale: int = 8,
        se_channels: int = 128,
        ops: Any = default_ops.default,
    ) -> None:
        super().__init__()
        self.blocks = nn.ModuleList([TDNNBlock(input_size, channels[0], kernel_sizes[0], dilations[0], ops)])
        for i in range(1, len(channels) - 1):
            self.blocks.append(SERes2NetBlock(channels[i - 1], channels[i], res2net_scale, se_channels, kernel_sizes[i], dilations[i], ops))
        self.mfa = TDNNBlock(channels[-2] * (len(channels) - 2), channels[-1], kernel_sizes[-1], dilations[-1], ops)
        self.asp = AttentiveStatisticsPooling(channels[-1], attention_channels, ops)
        self.asp_bn = SbBatchNorm1d(channels[-1] * 2, ops)
        self.fc = SbConv1d(channels[-1] * 2, lin_neurons, 1, ops)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(1, 2)
        xl = []
        for layer in self.blocks:
            x = layer(x)
            xl.append(x)
        x = self.mfa(torch.cat(xl[1:], dim=1))
        x = self.fc(self.asp_bn(self.asp(x)))
        return x.transpose(1, 2)
