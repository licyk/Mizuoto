"""Fish Speech S2's codec at 44.1 kHz (fish-speech ``models/dac``, Fish Audio Research License; DAC parts after
descript-audio-codec, MIT): ``encode`` audio → 10 codebooks at ~21.5 frames/s, ``from_indices`` codes → audio.

- Encoder and decoder are DAC's, made causal, with a windowed causal transformer (RoPE, layer scale) in the
  encoder's last block. (Upstream also builds one for the decoder's first block but never attaches it: the
  checkpoint has no weights for it and the decoder does not run it.)
- The quantizer downsamples 4× (causal convolutions + ConvNeXt blocks), runs an 8-layer transformer, then a
  semantic codebook (4096) and 9 residual codebooks (1024, factorised through 8 dimensions); decoding reverses it.

Weight norm is kept as upstream keeps it (``weight_g``/``weight_v`` and ``parametrizations.weight.original0/1``)
and recomputed on every forward, so reduced-precision runs match upstream's too.
Attention masks are built per call (upstream allocates 32k × 32k boolean masks per transformer).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from mizuoto.runtime import ops as default_ops
from mizuoto.runtime.accel import Masked, MaskInfo, Site, attention

SITE = Site("fishspeech.codec")


class WN(nn.Module):
    """A convolution whose weight is ``_weight_norm(v, g, dim 0)`` at every call; parameters ``weight_g``, ``weight_v``, ``bias``."""

    def __init__(self, conv: nn.Module) -> None:
        super().__init__()
        w: Any = conv.weight
        self.weight_g = nn.Parameter(torch.empty(w.shape[0], *([1] * (w.dim() - 1))))
        self.weight_v = nn.Parameter(torch.empty_like(w))
        self.bias = conv.bias
        del conv.weight
        del conv.bias
        self.__dict__["_conv"] = conv  # its settings only, not a submodule

    def weight(self) -> torch.Tensor:
        return torch._weight_norm(self.weight_v, self.weight_g, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        conv: Any = self.__dict__["_conv"]
        if isinstance(conv, nn.ConvTranspose1d):
            return F.conv_transpose1d(x, self.weight(), self.bias, conv.stride, conv.padding, conv.output_padding, conv.groups, conv.dilation)
        return F.conv1d(x, self.weight(), self.bias, conv.stride, conv.padding, conv.dilation, conv.groups)


def snake(x: torch.Tensor, alpha: torch.Tensor) -> torch.Tensor:
    """``x + 1 / (alpha + 1e-9) · sin(alpha · x)²``. Without gradients and without autocast the same operations run in
    place on one buffer (a product or sum with its operands swapped is the same number). Under autocast (the codec in
    bf16 on a GPU) ``pow`` and ``reciprocal`` compute in float32 and the rest follows, which in-place operations, keeping
    their buffer's dtype, would not."""
    shape = x.shape
    x = x.reshape(shape[0], shape[1], -1)
    if (torch.is_grad_enabled() and (x.requires_grad or alpha.requires_grad)) or torch.is_autocast_enabled(x.device.type):
        x = x + (alpha + 1e-9).reciprocal() * torch.sin(alpha * x).pow(2)
        return x.reshape(shape)
    t = alpha * x
    t.sin_().pow_(2).mul_((alpha + 1e-9).reciprocal()).add_(x)
    return t.reshape(shape)


class Snake1d(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.alpha = nn.Parameter(torch.empty(1, channels, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return snake(x, self.alpha)


def extra_padding(length: int, kernel_size: int, stride: int, padding_total: int) -> int:
    n_frames = (length - kernel_size + padding_total) / stride + 1
    return (math.ceil(n_frames) - 1) * stride + (kernel_size - padding_total) - length


class CausalConv(nn.Module):
    """``CausalConvNet`` (optionally weight-normed: ``CausalWNConv1d``)."""

    def __init__(self, cin: int, cout: int, kernel_size: int, ops: Any, dilation: int = 1, stride: int = 1, groups: int = 1, wn: bool = False) -> None:
        super().__init__()
        conv = ops.Conv1d(cin, cout, kernel_size, stride=stride, dilation=dilation, groups=groups)
        self.conv: nn.Module = WN(conv) if wn else conv
        self.stride = stride
        self.kernel_size = (kernel_size - 1) * dilation + 1
        self.padding = self.kernel_size - stride

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pad = self.padding
        x = F.pad(x, (pad, extra_padding(x.shape[-1], self.kernel_size, self.stride, pad)), "constant", 0)
        return self.conv(x).contiguous()


class CausalTransConv(nn.Module):
    def __init__(self, cin: int, cout: int, kernel_size: int, ops: Any, stride: int = 1, wn: bool = False) -> None:
        super().__init__()
        conv = ops.ConvTranspose1d(cin, cout, kernel_size, stride=stride)
        self.conv: nn.Module = WN(conv) if wn else conv
        self.stride = stride
        self.kernel_size = kernel_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        pad = self.kernel_size - self.stride
        right = math.ceil(pad)
        left = pad - right
        return x[..., left : x.shape[-1] - right].contiguous()


# -- the codec's transformers --------------------------------------------------------------------------


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.empty(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = x.float()
        return (h * torch.rsqrt(torch.mean(h * h, dim=-1, keepdim=True) + self.eps)).type_as(x) * self.weight


def freqs(seq_len: int, n_elem: int, base: float = 10000) -> torch.Tensor:
    f = 1.0 / (base ** (torch.arange(0, n_elem, 2)[: (n_elem // 2)].float() / n_elem))
    f = torch.outer(torch.arange(seq_len, device=f.device), f)
    cis = torch.polar(torch.ones_like(f), f)
    return torch.stack([cis.real, cis.imag], dim=-1).to(dtype=torch.bfloat16)


def rotate(x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
    xs = x.float().reshape(*x.shape[:-1], -1, 2)
    f = freqs_cis.view(1, xs.size(1), 1, xs.size(3), 2)
    out = torch.stack([xs[..., 0] * f[..., 0] - xs[..., 1] * f[..., 1], xs[..., 1] * f[..., 0] + xs[..., 0] * f[..., 1]], -1)
    return out.flatten(3).type_as(x)


class Attention(nn.Module):
    def __init__(self, dim: int, n_head: int, head_dim: int, ops: Any) -> None:
        super().__init__()
        self.n_head, self.head_dim = n_head, head_dim
        size = n_head * head_dim
        self.wqkv = ops.Linear(dim, 3 * size, bias=False)
        self.wo = ops.Linear(size, dim, bias=False)

    def forward(self, x: torch.Tensor, freqs_cis: torch.Tensor, mask: Masked) -> torch.Tensor:
        bsz, seqlen, _ = x.shape
        size = self.n_head * self.head_dim
        q, k, v = self.wqkv(x).split([size, size, size], dim=-1)
        q = rotate(q.view(bsz, seqlen, self.n_head, self.head_dim), freqs_cis)
        k = rotate(k.view(bsz, seqlen, self.n_head, self.head_dim), freqs_cis)
        v = v.view(bsz, seqlen, self.n_head, self.head_dim)
        q, k, v = (t.transpose(1, 2) for t in (q, k, v))
        y = attention(q, k.repeat_interleave(1, dim=1), v.repeat_interleave(1, dim=1), site=SITE, mask=mask.dense, mask_info=mask.info)
        return self.wo(y.transpose(1, 2).contiguous().view(bsz, seqlen, size))


class LayerScale(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.gamma = nn.Parameter(torch.empty(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.mul_(self.gamma)


class Block(nn.Module):
    def __init__(self, dim: int, n_head: int, head_dim: int, intermediate: int, eps: float, ops: Any) -> None:
        super().__init__()
        self.attention = Attention(dim, n_head, head_dim, ops)
        self.feed_forward = nn.Module()
        self.feed_forward.add_module("w1", ops.Linear(dim, intermediate, bias=False))
        self.feed_forward.add_module("w3", ops.Linear(dim, intermediate, bias=False))
        self.feed_forward.add_module("w2", ops.Linear(intermediate, dim, bias=False))
        self.ffn_norm = RMSNorm(dim, eps)
        self.attention_norm = RMSNorm(dim, eps)
        self.attention_layer_scale = LayerScale(dim)
        self.ffn_layer_scale = LayerScale(dim)

    def _ff(self, x: torch.Tensor) -> torch.Tensor:
        ff: Any = self.feed_forward
        return ff.w2(F.silu(ff.w1(x)) * ff.w3(x))

    def forward(self, x: torch.Tensor, freqs_cis: torch.Tensor, mask: Masked) -> torch.Tensor:
        h = x + self.attention_layer_scale(self.attention(self.attention_norm(x), freqs_cis, mask))
        return h + self.ffn_layer_scale(self._ff(self.ffn_norm(h)))


class WindowLimitedTransformer(nn.Module):
    """Channels-first causal transformer; ``window_size`` limits each position to the last ``window_size``."""

    def __init__(
        self,
        input_dim: int,
        n_layer: int,
        n_head: int,
        dim: int,
        intermediate: int,
        window_size: int | None,
        ops: Any,
        head_dim: int = 64,
        eps: float = 1e-5,
        rope_base: float = 10000,
    ) -> None:
        super().__init__()
        self.window_size = window_size
        self.head_dim, self.rope_base = head_dim, rope_base
        self.layers = nn.ModuleList(Block(dim, n_head, head_dim, intermediate, eps, ops) for _ in range(n_layer))
        self.norm = RMSNorm(dim, eps)
        self.input_proj: nn.Module = ops.Linear(input_dim, dim) if input_dim != dim else nn.Identity()
        self.output_proj: nn.Module = ops.Linear(dim, input_dim) if input_dim != dim else nn.Identity()
        self._freqs: torch.Tensor | None = None

    def _freqs_cis(self, n: int, device: torch.device) -> torch.Tensor:
        # upstream precomputes 327680 positions; the first n are the same numbers
        if self._freqs is None or self._freqs.shape[0] < n or self._freqs.device != device:
            self._freqs = freqs(max(n, 4096), self.head_dim, self.rope_base).to(device)
        return self._freqs[:n]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_proj(x.transpose(1, 2))
        n = x.shape[1]
        mask = torch.tril(torch.ones(n, n))
        if self.window_size is not None:
            rows = torch.arange(n).view(-1, 1)
            valid = (rows - self.window_size + 1).clamp(min=0)
            mask = (torch.arange(n) >= valid) & mask.bool()
        mask = Masked(mask.bool()[None, None].to(x.device), MaskInfo(causal=True, window=self.window_size))
        freqs_cis = self._freqs_cis(n, x.device).to(x.dtype if x.dtype != torch.float32 else torch.float32)
        for layer in self.layers:
            x = layer(x, freqs_cis, mask)
        return self.output_proj(self.norm(x)).transpose(1, 2)


# -- DAC encoder and decoder ----------------------------------------------------------------------------


class ResidualUnit(nn.Module):
    def __init__(self, dim: int, dilation: int, ops: Any) -> None:
        super().__init__()
        self.block = nn.Sequential(Snake1d(dim), CausalConv(dim, dim, 7, ops, dilation=dilation, wn=True), Snake1d(dim), CausalConv(dim, dim, 1, ops, wn=True))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.block(x)
        pad = x.shape[-1] - y.shape[-1]
        if pad > 0:
            x = x[..., :-pad]
        return x + y


class EncoderBlock(nn.Module):
    def __init__(self, dim: int, stride: int, n_t_layer: int, ops: Any) -> None:
        super().__init__()
        transformer: nn.Module = nn.Identity() if n_t_layer == 0 else WindowLimitedTransformer(dim, n_t_layer, dim // 64, dim, dim * 3, 512, ops)
        self.block = nn.Sequential(
            ResidualUnit(dim // 2, 1, ops),
            ResidualUnit(dim // 2, 3, ops),
            ResidualUnit(dim // 2, 9, ops),
            Snake1d(dim // 2),
            CausalConv(dim // 2, dim, 2 * stride, ops, stride=stride, wn=True),
            transformer,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class Encoder(nn.Module):
    def __init__(self, d_model: int, strides: list[int], d_latent: int, n_transformer_layers: list[int], ops: Any) -> None:
        super().__init__()
        blocks: list[nn.Module] = [CausalConv(1, d_model, 7, ops, wn=True)]
        for stride, n_t in zip(strides, n_transformer_layers, strict=True):
            d_model *= 2
            blocks.append(EncoderBlock(d_model, stride, n_t, ops))
        blocks += [Snake1d(d_model), CausalConv(d_model, d_latent, 3, ops, wn=True)]
        self.block = nn.Sequential(*blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class DecoderBlock(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, stride: int, ops: Any) -> None:
        super().__init__()
        self.block = nn.Sequential(
            Snake1d(input_dim),
            CausalTransConv(input_dim, output_dim, 2 * stride, ops, stride=stride, wn=True),
            ResidualUnit(output_dim, 1, ops),
            ResidualUnit(output_dim, 3, ops),
            ResidualUnit(output_dim, 9, ops),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class Decoder(nn.Module):
    def __init__(self, input_channel: int, channels: int, rates: list[int], ops: Any) -> None:
        super().__init__()
        layers: list[nn.Module] = [CausalConv(input_channel, channels, 7, ops, wn=True)]
        output_dim = channels
        for i, stride in enumerate(rates):
            input_dim, output_dim = channels // 2**i, channels // 2 ** (i + 1)
            layers.append(DecoderBlock(input_dim, output_dim, stride, ops))
        layers += [Snake1d(output_dim), CausalConv(output_dim, 1, 7, ops, wn=True), nn.Tanh()]
        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


# -- quantizer --------------------------------------------------------------------------------------------


class VectorQuantize(nn.Module):
    """descript-audio-codec's: weight-normed 1×1 projections around a cosine-similarity codebook."""

    def __init__(self, input_dim: int, codebook_size: int, codebook_dim: int, ops: Any) -> None:
        super().__init__()
        self.in_proj = WN(ops.Conv1d(input_dim, codebook_dim, kernel_size=1))
        self.out_proj = WN(ops.Conv1d(codebook_dim, input_dim, kernel_size=1))
        self.codebook = ops.Embedding(codebook_size, codebook_dim)

    def encode(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """(z_q after out_proj, indices)."""
        latents = self.in_proj(z)
        b = latents.size(0)
        enc = F.normalize(latents.permute(0, 2, 1).reshape(-1, latents.shape[1]))
        book = F.normalize(self.codebook.weight)
        dist = enc.pow(2).sum(1, keepdim=True) - 2 * enc @ book.t() + book.pow(2).sum(1, keepdim=True).t()
        indices = (-dist).max(1)[1].reshape(b, -1)
        z_q = self.decode_code(indices)
        z_q = latents + (z_q - latents).detach()
        return self.out_proj(z_q), indices

    def decode_code(self, ids: torch.Tensor) -> torch.Tensor:
        return F.embedding(ids, self.codebook.weight).transpose(1, 2)


class ResidualVectorQuantize(nn.Module):
    def __init__(self, input_dim: int, n_codebooks: int, codebook_size: int, codebook_dim: int, ops: Any) -> None:
        super().__init__()
        self.codebook_size = codebook_size
        self.quantizers = nn.ModuleList(VectorQuantize(input_dim, codebook_size, codebook_dim, ops) for _ in range(n_codebooks))

    def encode(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z_q: Any = 0
        residual = z
        codes = []
        for i, q in enumerate(self.quantizers):
            z_q_i, idx = q.encode(residual)  # type: ignore[operator]
            mask = torch.full((z.shape[0],), fill_value=i, device=z.device) < len(self.quantizers)
            z_q = z_q + z_q_i * mask[:, None, None]
            residual = residual - z_q_i
            codes.append(idx)
        return z_q, torch.stack(codes, dim=1)

    def from_codes(self, codes: torch.Tensor) -> torch.Tensor:
        z_q: Any = 0.0
        for i in range(codes.shape[1]):
            q: Any = self.quantizers[i]
            z_q = z_q + q.out_proj(q.decode_code(codes[:, i, :]))
        return z_q


class ConvNeXtBlock(nn.Module):
    def __init__(self, dim: int, ops: Any) -> None:
        super().__init__()
        self.dwconv = CausalConv(dim, dim, 7, ops, groups=dim)
        self.norm = ops.LayerNorm(dim, eps=1e-6)
        self.pwconv1 = ops.Linear(dim, 4 * dim)
        self.pwconv2 = ops.Linear(4 * dim, dim)
        self.gamma = nn.Parameter(torch.empty(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.dwconv(x).permute(0, 2, 1)
        y = self.pwconv2(F.gelu(self.pwconv1(self.norm(y))))
        return x + (self.gamma * y).permute(0, 2, 1)


class DownsampleRVQ(nn.Module):
    def __init__(self, c: CodecConfig, ops: Any) -> None:
        super().__init__()
        d = c.latent_dim
        self.semantic_quantizer = ResidualVectorQuantize(d, 1, c.semantic_codebook_size, c.codebook_dim, ops)
        self.quantizer = ResidualVectorQuantize(d, c.n_codebooks, c.codebook_size, c.codebook_dim, ops)
        self.downsample = nn.Sequential(*[nn.Sequential(CausalConv(d, d, f, ops, stride=f), ConvNeXtBlock(d, ops)) for f in c.downsample_factor])
        self.upsample = nn.Sequential(*[nn.Sequential(CausalTransConv(d, d, f, ops, stride=f), ConvNeXtBlock(d, ops)) for f in reversed(c.downsample_factor)])
        t = c.quantizer_transformer
        self.pre_module = WindowLimitedTransformer(d, t.n_layer, t.n_head, t.dim, t.intermediate_size, t.window_size, ops)
        self.post_module = WindowLimitedTransformer(d, t.n_layer, t.n_head, t.dim, t.intermediate_size, t.window_size, ops)

    def encode(self, z: torch.Tensor) -> torch.Tensor:
        z = self.pre_module(self.downsample(z))
        semantic_z, semantic_codes = self.semantic_quantizer.encode(z)
        _, codes = self.quantizer.encode(z - semantic_z)
        return torch.cat([semantic_codes, codes], dim=1)

    def decode(self, indices: torch.Tensor) -> torch.Tensor:
        indices[:, 0] = torch.clamp(indices[:, 0], max=self.semantic_quantizer.codebook_size - 1)
        indices[:, 1:] = torch.clamp(indices[:, 1:], max=self.quantizer.codebook_size - 1)
        z_q = self.semantic_quantizer.from_codes(indices[:, :1]) + self.quantizer.from_codes(indices[:, 1:])
        return self.upsample(self.post_module(z_q))


@dataclass
class TransformerConfig:
    n_layer: int = 8
    n_head: int = 16
    dim: int = 1024
    intermediate_size: int = 3072
    window_size: int = 128


@dataclass
class CodecConfig:
    """``modded_dac_vq.yaml``; the defaults are S2's codec."""

    sample_rate: int = 44100
    encoder_dim: int = 64
    encoder_rates: list[int] = field(default_factory=lambda: [2, 4, 8, 8])
    encoder_transformer_layers: list[int] = field(default_factory=lambda: [0, 0, 0, 4])
    latent_dim: int = 1024
    decoder_dim: int = 1536
    decoder_rates: list[int] = field(default_factory=lambda: [8, 8, 4, 2])
    n_codebooks: int = 9
    codebook_size: int = 1024
    codebook_dim: int = 8
    semantic_codebook_size: int = 4096
    downsample_factor: list[int] = field(default_factory=lambda: [2, 2])
    quantizer_transformer: TransformerConfig = field(default_factory=TransformerConfig)


class FishCodec(nn.Module):
    """``modded_dac_vq.yaml``'s DAC."""

    def __init__(self, config: CodecConfig | None = None, ops: Any = default_ops.default) -> None:
        super().__init__()
        c = self.config = config or CodecConfig()
        self.sample_rate = c.sample_rate
        self.hop_length = int(math.prod(c.encoder_rates))
        self.frame_length = self.hop_length * 4
        self.encoder = Encoder(c.encoder_dim, c.encoder_rates, c.latent_dim, c.encoder_transformer_layers, ops)
        self.quantizer = DownsampleRVQ(c, ops)
        self.decoder = Decoder(c.latent_dim, c.decoder_dim, c.decoder_rates, ops)

    def encode(self, audio: torch.Tensor, lengths: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if audio.ndim == 2:
            audio = audio.unsqueeze(1)
        length = audio.shape[-1]
        audio = F.pad(audio, (0, math.ceil(length / self.frame_length) * self.frame_length - length))
        codes = self.quantizer.encode(self.encoder(audio))
        return codes, torch.ceil(lengths / self.frame_length).long()

    def from_indices(self, indices: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.quantizer.decode(indices))


IGNORED = ("*.freqs_cis", "*.causal_mask")
"""Buffers an older upstream saved with the weights; upstream no longer loads them (they are computed)."""


def codec_state(state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """``codec.pth`` (``generator.`` prefixes dropped) in this module's names: both weight-norm formats as ``weight_g``/``weight_v``."""
    if any("generator." in k for k in state):
        state = {k.replace("generator.", ""): v for k, v in state.items() if "generator." in k}
    out = {}
    for k, v in state.items():
        k = k.replace(".parametrizations.weight.original0", ".weight_g").replace(".parametrizations.weight.original1", ".weight_v")
        out[k] = v
    return out
