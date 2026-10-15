"""HiFi-GAN's generator as VITS has it (``models.py``: ``Generator``; ``modules.py``: ``ResBlock1``, ``ResBlock2``),
in GPT-SoVITS' form (MIT): an optional global condition (``cond``, added after ``conv_pre``) and an optional
bias on ``conv_post``. GPT-SoVITS v2 decodes its latent with it at 32 kHz; v4's 48 kHz vocoder is the same
class without the condition.

Weight norm (``ups``, the residual blocks' convolutions) is folded at load (``mizuoto.runtime.loading.fold_weight_norm``):
upstream recomputes ``_weight_norm(v, g)`` before every forward, which in float32 is the same weight.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from mizuoto.runtime import ops as default_ops

LRELU_SLOPE = 0.1


def get_padding(kernel_size: int, dilation: int = 1) -> int:
    return int((kernel_size * dilation - dilation) / 2)


class ResBlock1(nn.Module):
    def __init__(self, channels: int, kernel_size: int = 3, dilation: Sequence[int] = (1, 3, 5), ops: Any = default_ops.default) -> None:
        super().__init__()
        self.convs1 = nn.ModuleList(ops.Conv1d(channels, channels, kernel_size, 1, dilation=d, padding=get_padding(kernel_size, d)) for d in dilation[:3])
        self.convs2 = nn.ModuleList(ops.Conv1d(channels, channels, kernel_size, 1, dilation=1, padding=get_padding(kernel_size, 1)) for _ in range(3))

    def forward(self, x: torch.Tensor, x_mask: torch.Tensor | None = None) -> torch.Tensor:
        for c1, c2 in zip(self.convs1, self.convs2, strict=True):
            xt = F.leaky_relu(x, LRELU_SLOPE)
            if x_mask is not None:
                xt = xt * x_mask
            xt = c1(xt)
            xt = F.leaky_relu(xt, LRELU_SLOPE)
            if x_mask is not None:
                xt = xt * x_mask
            xt = c2(xt)
            x = xt + x
        if x_mask is not None:
            x = x * x_mask
        return x


class ResBlock2(nn.Module):
    def __init__(self, channels: int, kernel_size: int = 3, dilation: Sequence[int] = (1, 3), ops: Any = default_ops.default) -> None:
        super().__init__()
        self.convs = nn.ModuleList(ops.Conv1d(channels, channels, kernel_size, 1, dilation=d, padding=get_padding(kernel_size, d)) for d in dilation[:2])

    def forward(self, x: torch.Tensor, x_mask: torch.Tensor | None = None) -> torch.Tensor:
        for c in self.convs:
            xt = F.leaky_relu(x, LRELU_SLOPE)
            if x_mask is not None:
                xt = xt * x_mask
            xt = c(xt)
            x = xt + x
        if x_mask is not None:
            x = x * x_mask
        return x


class Generator(nn.Module):
    def __init__(
        self,
        initial_channel: int,
        resblock: str,
        resblock_kernel_sizes: Sequence[int],
        resblock_dilation_sizes: Sequence[Sequence[int]],
        upsample_rates: Sequence[int],
        upsample_initial_channel: int,
        upsample_kernel_sizes: Sequence[int],
        gin_channels: int = 0,
        is_bias: bool = False,
        ops: Any = default_ops.default,
    ) -> None:
        super().__init__()
        self.num_kernels = len(resblock_kernel_sizes)
        self.num_upsamples = len(upsample_rates)
        self.conv_pre = ops.Conv1d(initial_channel, upsample_initial_channel, 7, 1, padding=3)
        block = ResBlock1 if str(resblock) == "1" else ResBlock2
        self.ups = nn.ModuleList(
            ops.ConvTranspose1d(upsample_initial_channel // (2**i), upsample_initial_channel // (2 ** (i + 1)), k, u, padding=(k - u) // 2)
            for i, (u, k) in enumerate(zip(upsample_rates, upsample_kernel_sizes, strict=True))
        )
        self.resblocks = nn.ModuleList()
        ch = upsample_initial_channel
        for i in range(len(self.ups)):
            ch = upsample_initial_channel // (2 ** (i + 1))
            for k, d in zip(resblock_kernel_sizes, resblock_dilation_sizes, strict=True):
                self.resblocks.append(block(ch, k, d, ops=ops))
        self.conv_post = ops.Conv1d(ch, 1, 7, 1, padding=3, bias=is_bias)
        if gin_channels != 0:
            self.cond = ops.Conv1d(gin_channels, upsample_initial_channel, 1)

    def forward(self, x: torch.Tensor, g: torch.Tensor | None = None) -> torch.Tensor:
        x = self.conv_pre(x)
        if g is not None:
            x = x + self.cond(g)
        for i in range(self.num_upsamples):
            x = F.leaky_relu(x, LRELU_SLOPE)
            x = self.ups[i](x)
            xs: Any = None
            for j in range(self.num_kernels):
                if xs is None:
                    xs = self.resblocks[i * self.num_kernels + j](x)
                else:
                    xs += self.resblocks[i * self.num_kernels + j](x)
            x = xs / self.num_kernels
        x = F.leaky_relu(x)
        x = self.conv_post(x)
        return torch.tanh(x)
