"""AudioVAE (upstream ``modules/audiovae``): v1 (VoxCPM 0.5B and 1.5) and v2 (VoxCPM2, a 16 kHz encoder and a decoder
conditioned on the output rate, 48 kHz).

A DAC-like causal codec: Snake activations, weight-normed causal convolutions (left padding only), depthwise
convolutions on the decoder's input, residual units with dilations 1, 3, 9. The encoder's ``mu`` is the latent.

- Weight norm (``torch.nn.utils.weight_norm``, recomputed by upstream at each forward) is folded once at load with
  the same ``torch._weight_norm`` call (``runtime.loading.fold_weight_norm``), on the device the VAE runs on, so a
  GPU's reduction gives the bits upstream's does there.
- Snake is the TorchScript function upstream scripts.
- A causal convolution pads ``2 × padding`` on the left. v2 subtracts ``output_padding`` (``stride % 2`` in the
  encoder's down-sampling convolutions); v1's convolution has no such argument.
- ``streaming()`` is v2's ``StreamingVAEDecoder``: each causal convolution of the decoder keeps its last inputs and
  each transposed convolution its last frames, so a patch decodes alone exactly as upstream's patched forwards do.
"""

from __future__ import annotations

import math
from collections.abc import Generator
from contextlib import contextmanager
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from mizuoto.engines.voxcpm.config import VaeConfig
from mizuoto.runtime import ops


def snake(x: torch.Tensor, alpha: torch.Tensor) -> torch.Tensor:
    shape = x.shape
    x = x.reshape(shape[0], shape[1], -1)
    x = x + (alpha + 1e-9).reciprocal() * torch.sin(alpha * x).pow(2)
    x = x.reshape(shape)
    return x


class Snake1d(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.alpha = nn.Parameter(torch.ones(1, channels, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return snake(x, self.alpha)


class CausalConv1d(ops.default.Conv1d):
    """``CausalConv1d``: left padding ``2 × padding − output_padding`` (v1: ``2 × padding``)."""

    def __init__(self, *args: Any, padding: int = 0, output_padding: int = 0, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.causal_pad = padding * 2 - output_padding
        self.stream: dict[int, torch.Tensor] | None = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.stream is None or self.causal_pad <= 0:
            return super().forward(F.pad(x, (self.causal_pad, 0)))
        state, p, key = self.stream, self.causal_pad, id(self)
        x_pad = torch.cat([state[key], x], dim=-1) if key in state else F.pad(x, (p, 0))
        if x.shape[-1] >= p:
            state[key] = x[:, :, -p:].detach()
        else:
            prev = state.get(key, torch.zeros(x.shape[0], x.shape[1], p, device=x.device, dtype=x.dtype))
            state[key] = torch.cat([prev, x], dim=-1)[:, :, -p:].detach()
        return super().forward(x_pad)


class CausalTransposeConv1d(ops.default.ConvTranspose1d):
    def __init__(self, *args: Any, padding: int = 0, output_padding: int = 0, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.trim = padding * 2 - output_padding
        self.stream: dict[int, torch.Tensor] | None = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        context = (self.kernel_size[0] - 1) // self.stride[0]
        if self.stream is None or context <= 0:
            return super().forward(x)[..., : -self.trim]
        state, key, trim = self.stream, id(self), self.trim
        x_full = torch.cat([state[key], x], dim=-1) if key in state else F.pad(x, (context, 0))
        state[key] = x[:, :, -context:].detach()
        out = super().forward(x_full)
        left = context * self.stride[0]
        return out[..., left:-trim] if trim > 0 else out[..., left:]


class ResidualUnit(nn.Module):
    def __init__(self, dim: int, dilation: int, kernel: int = 7, groups: int = 1) -> None:
        super().__init__()
        pad = ((7 - 1) * dilation) // 2
        self.block = nn.Sequential(
            Snake1d(dim),
            CausalConv1d(dim, dim, kernel_size=kernel, dilation=dilation, padding=pad, groups=groups),
            Snake1d(dim),
            CausalConv1d(dim, dim, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)


class EncoderBlock(nn.Module):
    def __init__(self, output_dim: int, stride: int, groups: int, version: int) -> None:
        super().__init__()
        input_dim = output_dim // 2
        self.block = nn.Sequential(
            ResidualUnit(input_dim, dilation=1, groups=groups),
            ResidualUnit(input_dim, dilation=3, groups=groups),
            ResidualUnit(input_dim, dilation=9, groups=groups),
            Snake1d(input_dim),
            CausalConv1d(input_dim, output_dim, kernel_size=2 * stride, stride=stride, padding=math.ceil(stride / 2), output_padding=stride % 2 if version == 2 else 0),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class Encoder(nn.Module):
    def __init__(self, d_model: int, latent_dim: int, strides: list[int], depthwise: bool, version: int) -> None:
        super().__init__()
        blocks: list[nn.Module] = [CausalConv1d(1, d_model, kernel_size=7, padding=3)]
        for stride in strides:
            d_model *= 2
            blocks.append(EncoderBlock(d_model, stride, d_model // 2 if depthwise else 1, version))
        self.block = nn.Sequential(*blocks)
        self.fc_mu = CausalConv1d(d_model, latent_dim, kernel_size=3, padding=1)
        self.fc_logvar = CausalConv1d(d_model, latent_dim, kernel_size=3, padding=1)  # loaded, never run (encode gives mu)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc_mu(self.block(x))


class DecoderBlock(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, stride: int, groups: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            Snake1d(input_dim),
            CausalTransposeConv1d(input_dim, output_dim, kernel_size=2 * stride, stride=stride, padding=math.ceil(stride / 2), output_padding=stride % 2),
            ResidualUnit(output_dim, dilation=1, groups=groups),
            ResidualUnit(output_dim, dilation=3, groups=groups),
            ResidualUnit(output_dim, dilation=9, groups=groups),
        )
        self.input_channels = input_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class RateCondition(nn.Module):
    """``SampleRateConditionLayer`` with ``cond_type`` ``scale_bias`` (the only one the models use)."""

    def __init__(self, input_dim: int, buckets: int) -> None:
        super().__init__()
        self.scale_embed = ops.default.Embedding(buckets, input_dim)
        self.bias_embed = ops.default.Embedding(buckets, input_dim)

    def forward(self, x: torch.Tensor, sr_cond: torch.Tensor) -> torch.Tensor:
        return x * self.scale_embed(sr_cond).unsqueeze(-1) + self.bias_embed(sr_cond).unsqueeze(-1)


class Decoder(nn.Module):
    def __init__(self, config: VaeConfig) -> None:
        super().__init__()
        channels, latent = config.decoder_dim, config.latent_dim
        layers: list[nn.Module] = []
        if config.depthwise:
            layers += [CausalConv1d(latent, latent, kernel_size=7, padding=3, groups=latent), CausalConv1d(latent, channels, kernel_size=1)]
        else:
            layers.append(CausalConv1d(latent, channels, kernel_size=7, padding=3))
        output_dim = channels
        for i, stride in enumerate(config.decoder_rates):
            input_dim, output_dim = channels // 2**i, channels // 2 ** (i + 1)
            layers.append(DecoderBlock(input_dim, output_dim, stride, output_dim if config.depthwise else 1))
        layers += [Snake1d(output_dim), CausalConv1d(output_dim, 1, kernel_size=7, padding=3), nn.Tanh()]
        self.conditioned = config.sr_bin_boundaries is not None
        self.sr_bin_boundaries: torch.Tensor | None = None
        if not self.conditioned:
            self.model: nn.Module = nn.Sequential(*layers)
            return
        self.model = nn.ModuleList(layers)
        del self.sr_bin_boundaries
        self.register_buffer("sr_bin_boundaries", torch.tensor(config.sr_bin_boundaries, dtype=torch.int32))
        buckets = len(config.sr_bin_boundaries or []) + 1
        conds: list[Any] = [RateCondition(layer.input_channels, buckets) if isinstance(layer, DecoderBlock) else None for layer in layers]
        self.sr_cond_model = nn.ModuleList(conds)  # None where a layer takes no condition, as upstream

    def forward(self, x: torch.Tensor, sr_cond: torch.Tensor | None = None) -> torch.Tensor:
        if not self.conditioned:
            return self.model(x)
        assert sr_cond is not None and self.sr_bin_boundaries is not None
        index = torch.bucketize(sr_cond, self.sr_bin_boundaries)
        assert isinstance(self.model, nn.ModuleList)
        for layer, cond in zip(self.model, self.sr_cond_model):
            if cond is not None:
                x = cond(x, index)
            x = layer(x)
        return x


class AudioVAE(nn.Module):
    def __init__(self, config: VaeConfig) -> None:
        super().__init__()
        if config.use_noise_block:
            raise ValueError("AudioVAE with noise blocks is not supported (no released model uses them)")
        self.config = config
        self.latent_dim = config.latent_dim
        self.hop_length = math.prod(config.encoder_rates)
        self.chunk_size = math.prod(config.encoder_rates)
        self.decode_chunk_size = math.prod(config.decoder_rates)
        self.sample_rate = config.sample_rate
        self.out_sample_rate = config.output_rate
        self.encoder = Encoder(config.encoder_dim, config.latent_dim, config.encoder_rates, config.depthwise, config.version)
        self.decoder = Decoder(config)

    def encode(self, audio: torch.Tensor) -> torch.Tensor:
        """(B, T) or (B, 1, T) at ``sample_rate`` → the latent (B, D, T / hop), right-padded to a hop as upstream."""
        if audio.ndim == 2:
            audio = audio.unsqueeze(1)
        length = audio.shape[-1]
        right_pad = math.ceil(length / self.hop_length) * self.hop_length - length
        return self.encoder(F.pad(audio, (0, right_pad)))

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        if self.decoder.conditioned:
            return self.decoder(z, torch.tensor([self.out_sample_rate], device=z.device, dtype=torch.int32))
        return self.decoder(z)

    @contextmanager
    def streaming(self) -> Generator[AudioVAE, None, None]:
        """Decode patch by patch, carrying each causal layer's context between calls (v2's ``streaming_decode``)."""
        state: dict[int, torch.Tensor] = {}
        layers = [m for m in self.decoder.modules() if isinstance(m, (CausalConv1d, CausalTransposeConv1d))]
        for layer in layers:
            layer.stream = state
        try:
            yield self
        finally:
            for layer in layers:
                layer.stream = None
            state.clear()
