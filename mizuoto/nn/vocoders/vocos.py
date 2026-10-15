"""Vocos (charactr/vocos-mel-24khz as F5-TTS uses it): ConvNeXt backbone, iSTFT head (MIT; vocos 0.1.0).

Parameter names are those of the ``pytorch_model.bin`` checkpoint (``backbone.*``, ``head.*``); the
checkpoint's ``feature_extractor.*`` buffers are not needed to decode and are ignored.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from mizuoto.runtime import ops as default_ops


class ConvNeXtBlock(nn.Module):
    def __init__(self, dim: int, intermediate_dim: int, ops: Any = default_ops.default) -> None:
        super().__init__()
        self.dwconv = ops.Conv1d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.norm = ops.LayerNorm(dim, eps=1e-6)
        self.pwconv1 = ops.Linear(dim, intermediate_dim)
        self.pwconv2 = ops.Linear(intermediate_dim, dim)
        self.gamma = nn.Parameter(torch.empty(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.dwconv(x).transpose(1, 2)
        x = self.pwconv2(F.gelu(self.pwconv1(self.norm(x))))
        x = (self.gamma * x).transpose(1, 2)
        return residual + x


class VocosBackbone(nn.Module):
    def __init__(self, input_channels: int, dim: int, intermediate_dim: int, num_layers: int, ops: Any = default_ops.default) -> None:
        super().__init__()
        self.embed = ops.Conv1d(input_channels, dim, kernel_size=7, padding=3)
        self.norm = ops.LayerNorm(dim, eps=1e-6)
        self.convnext = nn.ModuleList([ConvNeXtBlock(dim, intermediate_dim, ops) for _ in range(num_layers)])
        self.final_layer_norm = ops.LayerNorm(dim, eps=1e-6)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.embed(x)
        x = self.norm(x.transpose(1, 2)).transpose(1, 2)
        for block in self.convnext:
            x = block(x)
        return self.final_layer_norm(x.transpose(1, 2))


class ISTFT(nn.Module):
    def __init__(self, n_fft: int, hop_length: int, win_length: int) -> None:
        super().__init__()
        self.n_fft, self.hop_length, self.win_length = n_fft, hop_length, win_length
        self.register_buffer("window", torch.hann_window(win_length))

    def forward(self, spec: torch.Tensor) -> torch.Tensor:
        return torch.istft(spec, self.n_fft, self.hop_length, self.win_length, self.get_buffer("window"), center=True)


class ISTFTHead(nn.Module):
    def __init__(self, dim: int, n_fft: int, hop_length: int, ops: Any = default_ops.default) -> None:
        super().__init__()
        self.out = ops.Linear(dim, n_fft + 2)
        self.istft = ISTFT(n_fft, hop_length, n_fft)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.out(x).transpose(1, 2)
        mag, p = x.chunk(2, dim=1)
        mag = torch.clip(torch.exp(mag), max=1e2)
        s = mag * (torch.cos(p) + 1j * torch.sin(p))
        return self.istft(s)


class Vocos(nn.Module):
    """``decode`` only: (batch, mels, frames) log-mel → (batch, samples)."""

    def __init__(self, config: dict[str, Any], ops: Any = default_ops.default) -> None:
        super().__init__()
        b = config["backbone"]["init_args"]
        h = config["head"]["init_args"]
        if h.get("padding", "same") != "center":
            raise ValueError("Only Vocos with center padding is supported")
        self.backbone = VocosBackbone(b["input_channels"], b["dim"], b["intermediate_dim"], b["num_layers"], ops)
        self.head = ISTFTHead(h["dim"], h["n_fft"], h["hop_length"], ops)

    def decode(self, features: torch.Tensor) -> torch.Tensor:
        return self.head(self.backbone(features))

    @staticmethod
    def ignorable() -> list[str]:
        return ["feature_extractor.*"]
