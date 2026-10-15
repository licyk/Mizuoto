"""DAC's convolutional encoder and decoder, as transformers' ``models/dac`` writes them (Apache-2.0).

Parameter names are transformers' (``conv1``, ``block.N.res_unit1.snake1.alpha``…), so the encoder and
decoder of a checkpoint saved by transformers load as they are. Weight norm is not applied: the
checkpoints Mizuoto reads store plain weights. The quantizer is the caller's (HiggsAudio has its own).

Two variations for HiggsAudio v2's tokenizer (``_adjust_dac_decoder``): ``odd_output_padding`` gives each
transposed convolution an output padding of ``stride % 2``, and ``tanh=False`` drops the final tanh.
"""

from __future__ import annotations

import math
from typing import Any

import torch
from torch import nn

from mizuoto.runtime import ops as default_ops


class Snake1d(nn.Module):
    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.alpha = nn.Parameter(torch.empty(1, hidden_dim, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shape = x.shape
        x = x.reshape(shape[0], shape[1], -1)
        x = x + (self.alpha + 1e-9).reciprocal() * torch.sin(self.alpha * x).pow(2)
        return x.reshape(shape)


class ResidualUnit(nn.Module):
    def __init__(self, dimension: int, dilation: int, ops: Any = default_ops.default) -> None:
        super().__init__()
        pad = ((7 - 1) * dilation) // 2
        self.snake1 = Snake1d(dimension)
        self.conv1 = ops.Conv1d(dimension, dimension, kernel_size=7, dilation=dilation, padding=pad)
        self.snake2 = Snake1d(dimension)
        self.conv2 = ops.Conv1d(dimension, dimension, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.conv2(self.snake2(self.conv1(self.snake1(x))))
        padding = (x.shape[-1] - y.shape[-1]) // 2
        if padding > 0:
            x = x[..., padding:-padding]
        return x + y


class EncoderBlock(nn.Module):
    def __init__(self, dimension: int, stride: int, ops: Any = default_ops.default) -> None:
        super().__init__()
        self.res_unit1 = ResidualUnit(dimension // 2, 1, ops)
        self.res_unit2 = ResidualUnit(dimension // 2, 3, ops)
        self.res_unit3 = ResidualUnit(dimension // 2, 9, ops)
        self.snake1 = Snake1d(dimension // 2)
        self.conv1 = ops.Conv1d(dimension // 2, dimension, kernel_size=2 * stride, stride=stride, padding=math.ceil(stride / 2))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.res_unit2(self.res_unit1(x))
        return self.conv1(self.snake1(self.res_unit3(x)))


class DecoderBlock(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, stride: int, odd_output_padding: bool = False, ops: Any = default_ops.default) -> None:
        super().__init__()
        self.snake1 = Snake1d(input_dim)
        self.conv_t1 = ops.ConvTranspose1d(
            input_dim, output_dim, kernel_size=2 * stride, stride=stride, padding=math.ceil(stride / 2), output_padding=stride % 2 if odd_output_padding else 0
        )
        self.res_unit1 = ResidualUnit(output_dim, 1, ops)
        self.res_unit2 = ResidualUnit(output_dim, 3, ops)
        self.res_unit3 = ResidualUnit(output_dim, 9, ops)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_t1(self.snake1(x))
        return self.res_unit3(self.res_unit2(self.res_unit1(x)))


class DacEncoder(nn.Module):
    """(batch, 1, samples) → (batch, hidden_size, frames)."""

    def __init__(self, encoder_hidden_size: int, downsampling_ratios: list[int], hidden_size: int, ops: Any = default_ops.default) -> None:
        super().__init__()
        self.conv1 = ops.Conv1d(1, encoder_hidden_size, kernel_size=7, padding=3)
        self.block = nn.ModuleList(EncoderBlock(encoder_hidden_size * 2 ** (i + 1), stride, ops) for i, stride in enumerate(downsampling_ratios))
        d_model = encoder_hidden_size * 2 ** len(downsampling_ratios)
        self.snake1 = Snake1d(d_model)
        self.conv2 = ops.Conv1d(d_model, hidden_size, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        for block in self.block:
            x = block(x)
        return self.conv2(self.snake1(x))

    def output_length(self, length: int) -> int:
        """transformers' ``_get_conv1d_output_lengths``: every Conv1d in module order, in its float arithmetic."""
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                padding, dilation, kernel, stride = (int(v[0]) for v in (m.padding, m.dilation, m.kernel_size, m.stride))
                length = int((length + 2 * padding - dilation * (kernel - 1) - 1) / stride + 1)
        return length


class DacDecoder(nn.Module):
    """(batch, hidden_size, frames) → (batch, 1, samples)."""

    def __init__(
        self,
        hidden_size: int,
        decoder_hidden_size: int,
        upsampling_ratios: list[int],
        odd_output_padding: bool = False,
        tanh: bool = True,
        ops: Any = default_ops.default,
    ) -> None:
        super().__init__()
        self.conv1 = ops.Conv1d(hidden_size, decoder_hidden_size, kernel_size=7, padding=3)
        self.block = nn.ModuleList(
            DecoderBlock(decoder_hidden_size // 2**i, decoder_hidden_size // 2 ** (i + 1), stride, odd_output_padding, ops) for i, stride in enumerate(upsampling_ratios)
        )
        output_dim = decoder_hidden_size // 2 ** len(upsampling_ratios)
        self.snake1 = Snake1d(output_dim)
        self.conv2 = ops.Conv1d(output_dim, 1, kernel_size=7, padding=3)
        self.tanh = tanh

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        for block in self.block:
            x = block(x)
        x = self.conv2(self.snake1(x))
        return torch.tanh(x) if self.tanh else x
