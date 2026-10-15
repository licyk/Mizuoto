"""BigVGAN (NVIDIA, MIT): mel → waveform with anti-aliased periodic activations (``bigvgan.py``,
``activations.py``, ``alias_free_activation/torch``), configured by the model's ``config.json``.

- AMP blocks of type 1 (dilated + plain convolutions) or 2, Snake or SnakeBeta (optionally log-scale), each
  activation run at twice the rate: replicate pad, ×2 transposed convolution with a Kaiser-windowed sinc, the
  activation, the same filter down.
- The output is ``tanh``, or a clamp to [-1, 1] (``use_tanh_at_final = false``).

Weight norm is folded at load, as upstream's ``remove_weight_norm`` does before inference. The filters are
the checkpoint's buffers (computed the same way). The CUDA activation kernel (``use_cuda_kernel``: fast-math,
not bit-identical) is not ported.
"""

from __future__ import annotations

import math
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from mizuoto.runtime import ops as default_ops


def get_padding(kernel_size: int, dilation: int = 1) -> int:
    return int((kernel_size * dilation - dilation) / 2)


def kaiser_sinc_filter1d(cutoff: float, half_width: float, kernel_size: int) -> torch.Tensor:
    even = kernel_size % 2 == 0
    half_size = kernel_size // 2
    delta_f = 4 * half_width
    a = 2.285 * (half_size - 1) * math.pi * delta_f + 7.95
    if a > 50.0:
        beta = 0.1102 * (a - 8.7)
    elif a >= 21.0:
        beta = 0.5842 * (a - 21) ** 0.4 + 0.07886 * (a - 21.0)
    else:
        beta = 0.0
    window = torch.kaiser_window(kernel_size, beta=beta, periodic=False)
    time = torch.arange(-half_size, half_size) + 0.5 if even else torch.arange(kernel_size) - half_size
    if cutoff == 0:
        return torch.zeros_like(time).view(1, 1, kernel_size)
    f = 2 * cutoff * window * torch.sinc(2 * cutoff * time)
    f /= f.sum()
    return f.view(1, 1, kernel_size)


class Snake(nn.Module):
    def __init__(self, channels: int, logscale: bool) -> None:
        super().__init__()
        self.alpha = nn.Parameter(torch.empty(channels))
        self.logscale = logscale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        alpha = self.alpha.unsqueeze(0).unsqueeze(-1)
        if self.logscale:
            alpha = torch.exp(alpha)
        return x + (1.0 / (alpha + 0.000000001)) * torch.pow(torch.sin(x * alpha), 2)


class SnakeBeta(nn.Module):
    def __init__(self, channels: int, logscale: bool) -> None:
        super().__init__()
        self.alpha = nn.Parameter(torch.empty(channels))
        self.beta = nn.Parameter(torch.empty(channels))
        self.logscale = logscale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        alpha = self.alpha.unsqueeze(0).unsqueeze(-1)
        beta = self.beta.unsqueeze(0).unsqueeze(-1)
        if self.logscale:
            alpha, beta = torch.exp(alpha), torch.exp(beta)
        return x + (1.0 / (beta + 0.000000001)) * torch.pow(torch.sin(x * alpha), 2)


class UpSample1d(nn.Module):
    def __init__(self, ratio: int = 2, kernel_size: int = 12) -> None:
        super().__init__()
        self.ratio = ratio
        self.kernel_size = kernel_size
        self.stride = ratio
        self.pad = kernel_size // ratio - 1
        self.pad_left = self.pad * self.stride + (kernel_size - self.stride) // 2
        self.pad_right = self.pad * self.stride + (kernel_size - self.stride + 1) // 2
        self.register_buffer("filter", kaiser_sinc_filter1d(0.5 / ratio, 0.6 / ratio, kernel_size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        channels = x.shape[1]
        x = F.pad(x, (self.pad, self.pad), mode="replicate")
        x = self.ratio * F.conv_transpose1d(x, self.get_buffer("filter").expand(channels, -1, -1), stride=self.stride, groups=channels)
        return x[..., self.pad_left : -self.pad_right]


class LowPassFilter1d(nn.Module):
    def __init__(self, cutoff: float, half_width: float, stride: int, kernel_size: int) -> None:
        super().__init__()
        even = kernel_size % 2 == 0
        self.pad_left = kernel_size // 2 - int(even)
        self.pad_right = kernel_size // 2
        self.stride = stride
        self.register_buffer("filter", kaiser_sinc_filter1d(cutoff, half_width, kernel_size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        channels = x.shape[1]
        x = F.pad(x, (self.pad_left, self.pad_right), mode="replicate")
        return F.conv1d(x, self.get_buffer("filter").expand(channels, -1, -1), stride=self.stride, groups=channels)


class DownSample1d(nn.Module):
    def __init__(self, ratio: int = 2, kernel_size: int = 12) -> None:
        super().__init__()
        self.lowpass = LowPassFilter1d(0.5 / ratio, 0.6 / ratio, ratio, kernel_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lowpass(x)


class Activation1d(nn.Module):
    def __init__(self, activation: nn.Module) -> None:
        super().__init__()
        self.act = activation
        self.upsample = UpSample1d(2, 12)
        self.downsample = DownSample1d(2, 12)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.downsample(self.act(self.upsample(x)))


def _activation(kind: str, channels: int, logscale: bool) -> nn.Module:
    if kind == "snake":
        return Activation1d(Snake(channels, logscale))
    if kind == "snakebeta":
        return Activation1d(SnakeBeta(channels, logscale))
    raise ValueError(f"Unknown BigVGAN activation {kind!r}")


class AMPBlock1(nn.Module):
    def __init__(self, channels: int, kernel_size: int, dilation: list[int], activation: str, logscale: bool, ops: Any) -> None:
        super().__init__()
        self.convs1 = nn.ModuleList(ops.Conv1d(channels, channels, kernel_size, 1, dilation=d, padding=get_padding(kernel_size, d)) for d in dilation)
        self.convs2 = nn.ModuleList(ops.Conv1d(channels, channels, kernel_size, 1, dilation=1, padding=get_padding(kernel_size, 1)) for _ in dilation)
        self.activations = nn.ModuleList(_activation(activation, channels, logscale) for _ in range(2 * len(dilation)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        acts1, acts2 = self.activations[::2], self.activations[1::2]
        for c1, c2, a1, a2 in zip(self.convs1, self.convs2, acts1, acts2, strict=True):
            x = c2(a2(c1(a1(x)))) + x
        return x


class AMPBlock2(nn.Module):
    def __init__(self, channels: int, kernel_size: int, dilation: list[int], activation: str, logscale: bool, ops: Any) -> None:
        super().__init__()
        self.convs = nn.ModuleList(ops.Conv1d(channels, channels, kernel_size, 1, dilation=d, padding=get_padding(kernel_size, d)) for d in dilation)
        self.activations = nn.ModuleList(_activation(activation, channels, logscale) for _ in dilation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for c, a in zip(self.convs, self.activations, strict=True):
            x = c(a(x)) + x
        return x


class BigVGAN(nn.Module):
    def __init__(self, h: dict[str, Any], ops: Any = default_ops.default) -> None:
        super().__init__()
        self.h = h
        self.num_kernels = len(h["resblock_kernel_sizes"])
        self.num_upsamples = len(h["upsample_rates"])
        init = h["upsample_initial_channel"]
        self.conv_pre = ops.Conv1d(h["num_mels"], init, 7, 1, padding=3)
        block = {"1": AMPBlock1, "2": AMPBlock2}.get(str(h["resblock"]))
        if block is None:
            raise ValueError(f"Unknown BigVGAN resblock {h['resblock']!r}")
        self.ups = nn.ModuleList(
            nn.ModuleList([ops.ConvTranspose1d(init // (2**i), init // (2 ** (i + 1)), k, u, padding=(k - u) // 2)])
            for i, (u, k) in enumerate(zip(h["upsample_rates"], h["upsample_kernel_sizes"], strict=True))
        )
        logscale = bool(h.get("snake_logscale", False))
        self.resblocks = nn.ModuleList()
        ch = init
        for i in range(len(self.ups)):
            ch = init // (2 ** (i + 1))
            for k, d in zip(h["resblock_kernel_sizes"], h["resblock_dilation_sizes"], strict=True):
                self.resblocks.append(block(ch, k, d, h["activation"], logscale, ops))
        self.activation_post = _activation(h["activation"], ch, logscale)
        self.conv_post = ops.Conv1d(ch, 1, 7, 1, padding=3, bias=bool(h.get("use_bias_at_final", True)))
        self.use_tanh_at_final = bool(h.get("use_tanh_at_final", True))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_pre(x)
        for i in range(self.num_upsamples):
            ups: Any = self.ups[i]
            for up in ups:
                x = up(x)
            xs: Any = None
            for j in range(self.num_kernels):
                y = self.resblocks[i * self.num_kernels + j](x)
                xs = y if xs is None else xs + y
            x = xs / self.num_kernels
        x = self.conv_post(self.activation_post(x))
        return torch.tanh(x) if self.use_tanh_at_final else torch.clamp(x, min=-1.0, max=1.0)


class SpeakerBigVGAN(nn.Module):
    """IndexTTS-1.5's BigVGAN (``indextts/BigVGAN/models.py``): GPT latents in, conditioned on an ECAPA-TDNN
    embedding of the reference's mels (added after the first convolution and after every upsampling)."""

    def __init__(self, h: dict[str, Any], ops: Any = default_ops.default) -> None:
        super().__init__()
        from mizuoto.nn.speaker.ecapa import ECAPA_TDNN

        if h.get("feat_upsample"):
            raise ValueError("feat_upsample is not supported (IndexTTS-1.5 does not use it)")
        self.num_kernels = len(h["resblock_kernel_sizes"])
        self.num_upsamples = len(h["upsample_rates"])
        self.cond_in_each_up_layer = bool(h["cond_d_vector_in_each_upsampling_layer"])
        init = h["upsample_initial_channel"]
        self.conv_pre = ops.Conv1d(h["gpt_dim"], init, 7, 1, padding=3)
        block = {"1": AMPBlock1, "2": AMPBlock2}[str(h["resblock"])]
        self.ups = nn.ModuleList(
            nn.ModuleList([ops.ConvTranspose1d(init // (2**i), init // (2 ** (i + 1)), k, u, padding=(k - u) // 2)])
            for i, (u, k) in enumerate(zip(h["upsample_rates"], h["upsample_kernel_sizes"], strict=True))
        )
        logscale = bool(h.get("snake_logscale", False))
        self.resblocks = nn.ModuleList()
        ch = init
        for i in range(len(self.ups)):
            ch = init // (2 ** (i + 1))
            for k, d in zip(h["resblock_kernel_sizes"], h["resblock_dilation_sizes"], strict=True):
                self.resblocks.append(block(ch, k, d, h["activation"], logscale, ops))
        self.activation_post = _activation(h["activation"], ch, logscale)
        self.conv_post = ops.Conv1d(ch, 1, 7, 1, padding=3)
        self.speaker_encoder = ECAPA_TDNN(h["num_mels"], lin_neurons=h["speaker_embedding_dim"], ops=ops)
        self.cond_layer = ops.Conv1d(h["speaker_embedding_dim"], init, 1)
        if self.cond_in_each_up_layer:
            self.conds = nn.ModuleList(ops.Conv1d(h["speaker_embedding_dim"], init // (2 ** (i + 1)), 1) for i in range(len(self.ups)))

    def forward(self, x: torch.Tensor, mel_ref: torch.Tensor) -> torch.Tensor:
        """``x``: latents (B, T, gpt_dim); ``mel_ref``: (B, frames, mels) → (B, 1, samples)."""
        speaker = self.speaker_encoder(mel_ref).transpose(1, 2)
        x = self.conv_pre(x.transpose(1, 2))
        x = x + self.cond_layer(speaker)
        for i in range(self.num_upsamples):
            ups: Any = self.ups[i]
            for up in ups:
                x = up(x)
            if self.cond_in_each_up_layer:
                x = x + self.conds[i](speaker)
            xs: Any = None
            for j in range(self.num_kernels):
                y = self.resblocks[i * self.num_kernels + j](x)
                xs = y if xs is None else xs + y
            x = xs / self.num_kernels
        return torch.tanh(self.conv_post(self.activation_post(x)))
