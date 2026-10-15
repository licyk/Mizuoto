"""The F5-TTS diffusion transformer (``f5_tts.model.backbones.dit``, MIT), for F5-TTS and its relatives.

AdaLN-zero blocks over [noisy mel | condition mel | text embedding], a ConvNeXtV2 text encoder,
convolutional position embedding, and x-transformers' interleaved rotary embedding. ``pe_attn_head``
limits RoPE to the first heads (F5-TTS v0). Parameter names are upstream's (``transformer.*`` after the
EMA prefix).

The arithmetic follows upstream operation by operation:
- the rotary angles are computed in the dtype of the checkpoint's ``inv_freq`` buffer (fp16 when the
  model runs in fp16, as upstream does on a GPU);
- the time embedding is computed in float32 and cast to the timestep's dtype;
- attention is SDPA without a mask for single requests.
"""

from __future__ import annotations

import math
from typing import Any, NamedTuple

import torch
from torch import nn
from torch.nn import functional as F

from mizuoto.runtime import ops as default_ops
from mizuoto.runtime.accel import Site, attention

SITE = Site("dit")


def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0, theta_rescale_factor: float = 1.0) -> torch.Tensor:
    theta *= theta_rescale_factor ** (dim / (dim - 2))
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)
    freqs = torch.outer(t, freqs).float()
    return torch.cat([torch.cos(freqs), torch.sin(freqs)], dim=-1)


class SinusPositionEmbedding(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor, scale: float = 1000) -> torch.Tensor:
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=x.device).float() * -emb)
        emb = scale * x.unsqueeze(1) * emb.unsqueeze(0)
        return torch.cat((emb.sin(), emb.cos()), dim=-1)


class TimestepEmbedding(nn.Module):
    def __init__(self, dim: int, freq_embed_dim: int = 256, ops: Any = default_ops.default) -> None:
        super().__init__()
        self.time_embed = SinusPositionEmbedding(freq_embed_dim)
        self.time_mlp = nn.Sequential(ops.Linear(freq_embed_dim, dim), nn.SiLU(), ops.Linear(dim, dim))

    def forward(self, timestep: torch.Tensor) -> torch.Tensor:
        hidden = self.time_embed(timestep).to(timestep.dtype)
        return self.time_mlp(hidden)


class GRN(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.gamma = nn.Parameter(torch.empty(1, 1, dim))
        self.beta = nn.Parameter(torch.empty(1, 1, dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gx = torch.norm(x, p=2, dim=1, keepdim=True)
        nx = gx / (gx.mean(dim=-1, keepdim=True) + 1e-6)
        return self.gamma * (x * nx) + self.beta + x


class ConvNeXtV2Block(nn.Module):
    def __init__(self, dim: int, intermediate_dim: int, dilation: int = 1, ops: Any = default_ops.default) -> None:
        super().__init__()
        padding = (dilation * (7 - 1)) // 2
        self.dwconv = ops.Conv1d(dim, dim, kernel_size=7, padding=padding, groups=dim, dilation=dilation)
        self.norm = ops.LayerNorm(dim, eps=1e-6)
        self.pwconv1 = ops.Linear(dim, intermediate_dim)
        self.grn = GRN(intermediate_dim)
        self.pwconv2 = ops.Linear(intermediate_dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.dwconv(x.transpose(1, 2)).transpose(1, 2)
        x = self.pwconv2(self.grn(F.gelu(self.pwconv1(self.norm(x)))))
        return residual + x


class TextEmbedding(nn.Module):
    def __init__(self, text_num_embeds: int, text_dim: int, mask_padding: bool = True, conv_layers: int = 0, conv_mult: int = 2, ops: Any = default_ops.default) -> None:
        super().__init__()
        self.text_embed = ops.Embedding(text_num_embeds + 1, text_dim)  # 0 is the filler token
        self.mask_padding = mask_padding
        self.extra_modeling = conv_layers > 0
        if self.extra_modeling:
            self.precompute_max_pos = 8192
            self.register_buffer("freqs_cis", precompute_freqs_cis(text_dim, self.precompute_max_pos), persistent=False)
            self.text_blocks = nn.Sequential(*[ConvNeXtV2Block(text_dim, text_dim * conv_mult, ops=ops) for _ in range(conv_layers)])

    def forward(self, text: torch.Tensor, seq_len: int, drop_text: bool = False) -> torch.Tensor:
        text = text + 1  # batch padding is -1
        text = text[:, :seq_len]
        text = F.pad(text, (0, seq_len - text.shape[1]), value=0)
        text_mask = text == 0
        if drop_text:
            text = torch.zeros_like(text)
        text = self.text_embed(text)
        if self.extra_modeling:
            text = text + self.get_buffer("freqs_cis")[:seq_len, :]
            if self.mask_padding:
                text = text.masked_fill(text_mask.unsqueeze(-1).expand(-1, -1, text.size(-1)), 0.0)
                for block in self.text_blocks:
                    text = block(text)
                    text = text.masked_fill(text_mask.unsqueeze(-1).expand(-1, -1, text.size(-1)), 0.0)
            else:
                text = self.text_blocks(text)
        return text


class ConvPositionEmbedding(nn.Module):
    def __init__(self, dim: int, kernel_size: int = 31, groups: int = 16, ops: Any = default_ops.default) -> None:
        super().__init__()
        self.conv1d = nn.Sequential(
            ops.Conv1d(dim, dim, kernel_size, groups=groups, padding=kernel_size // 2),
            nn.Mish(),
            ops.Conv1d(dim, dim, kernel_size, groups=groups, padding=kernel_size // 2),
            nn.Mish(),
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        x = x.permute(0, 2, 1)
        m = mask.unsqueeze(1) if mask is not None else None
        if m is not None:
            x = x.masked_fill(~m, 0.0)
        for i, block in enumerate(self.conv1d):
            x = block(x)
            if m is not None and i in (0, 2):
                x = x.masked_fill(~m, 0.0)
        return x.permute(0, 2, 1)


class InputEmbedding(nn.Module):
    def __init__(self, mel_dim: int, text_dim: int, out_dim: int, ops: Any = default_ops.default) -> None:
        super().__init__()
        self.proj = ops.Linear(mel_dim * 2 + text_dim, out_dim)
        self.conv_pos_embed = ConvPositionEmbedding(out_dim, ops=ops)

    def forward(self, x: torch.Tensor, cond: torch.Tensor, text_embed: torch.Tensor, drop_audio_cond: bool = False, mask: torch.Tensor | None = None) -> torch.Tensor:
        if drop_audio_cond:
            cond = torch.zeros_like(cond)
        x = self.proj(torch.cat((x, cond, text_embed), dim=-1))
        return self.conv_pos_embed(x, mask=mask) + x


class RotaryEmbedding(nn.Module):
    """x-transformers' ``RotaryEmbedding`` (no xpos): pairs of adjacent channels rotate together."""

    def __init__(self, dim: int, base: float = 10000) -> None:
        super().__init__()
        self.register_buffer("inv_freq", 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim)))

    def forward_from_seq_len(self, seq_len: int) -> torch.Tensor:
        inv_freq = self.get_buffer("inv_freq")
        t = torch.arange(seq_len, device=inv_freq.device)
        freqs = torch.einsum("b i , j -> b i j", t[None].type_as(inv_freq), inv_freq) / 1.0
        return torch.stack((freqs, freqs), dim=-1).reshape(*freqs.shape[:-1], -1)


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x = x.reshape(*x.shape[:-1], -1, 2)
    x1, x2 = x.unbind(dim=-1)
    return torch.stack((-x2, x1), dim=-1).reshape(*x.shape[:-2], -1)


class Rope(NamedTuple):
    """Rotary frequencies with their cos and sin, made once per forward for every attention layer (``rope_tables``)."""

    freqs: torch.Tensor
    cos: torch.Tensor
    sin: torch.Tensor


def rope_tables(freqs: torch.Tensor) -> Rope:
    return Rope(freqs, freqs.cos(), freqs.sin())


def apply_rotary(t: torch.Tensor, freqs: torch.Tensor | Rope, scale: float = 1.0) -> torch.Tensor:
    """x-transformers' ``apply_rotary_pos_emb`` (``freqs``: (1, n, d), ``t``: (b, h, n, dh)), or ``rope_tables(freqs)``.

    The same arithmetic, without the work that changes nothing: a ``Rope``'s cos and sin are those of its whole
    ``freqs`` (used when ``t`` takes every position; a slice's are computed as upstream does), a scale of 1 is not
    multiplied (x · 1 is x) and nothing is concatenated when every channel turns."""
    table = freqs.freqs if isinstance(freqs, Rope) else freqs
    rot_dim, seq_len, orig_dtype = table.shape[-1], t.shape[-2], t.dtype
    if isinstance(freqs, Rope) and table.shape[-2] == seq_len:
        cos, sin = freqs.cos, freqs.sin
    else:
        table = table[:, -seq_len:, :]
        cos, sin = table.cos(), table.sin()
    if t.ndim == 4 and cos.ndim == 3:
        cos, sin = cos.unsqueeze(1), sin.unsqueeze(1)
    t, t_unrotated = t[..., :rot_dim], t[..., rot_dim:]
    if scale == 1.0:
        t = (t * cos) + (_rotate_half(t) * sin)
    else:
        t = (t * cos * scale) + (_rotate_half(t) * sin * scale)
    if t_unrotated.shape[-1]:
        t = torch.cat((t, t_unrotated), dim=-1)
    return t.type(orig_dtype)


class Attention(nn.Module):
    def __init__(self, dim: int, heads: int, dim_head: int, pe_attn_head: int | None = None, ops: Any = default_ops.default) -> None:
        super().__init__()
        self.heads = heads
        self.pe_attn_head = pe_attn_head
        inner = dim_head * heads
        self.to_q = ops.Linear(dim, inner)
        self.to_k = ops.Linear(dim, inner)
        self.to_v = ops.Linear(dim, inner)
        self.to_out = nn.ModuleList([ops.Linear(inner, dim), nn.Dropout(0.0)])

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None, rope: torch.Tensor | Rope | None) -> torch.Tensor:
        b = x.shape[0]
        q, k, v = self.to_q(x), self.to_k(x), self.to_v(x)
        head_dim = k.shape[-1] // self.heads
        q = q.view(b, -1, self.heads, head_dim).transpose(1, 2)
        k = k.view(b, -1, self.heads, head_dim).transpose(1, 2)
        v = v.view(b, -1, self.heads, head_dim).transpose(1, 2)
        if rope is not None:
            if self.pe_attn_head is not None:
                pn = self.pe_attn_head
                q[:, :pn, :, :] = apply_rotary(q[:, :pn, :, :], rope, 1.0)
                k[:, :pn, :, :] = apply_rotary(k[:, :pn, :, :], rope, 1.0)
            else:
                q = apply_rotary(q, rope, 1.0)
                k = apply_rotary(k, rope, 1.0)
        out = attention(q, k, v, site=SITE)
        out = out.transpose(1, 2).reshape(b, -1, self.heads * head_dim).to(q.dtype)
        out = self.to_out[0](out)
        if mask is not None:
            out = out.masked_fill(~mask.unsqueeze(-1), 0.0)
        return out


class AdaLayerNorm(nn.Module):
    def __init__(self, dim: int, ops: Any = default_ops.default) -> None:
        super().__init__()
        self.linear = ops.Linear(dim, dim * 6)
        self.norm = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)

    def forward(self, x: torch.Tensor, emb: torch.Tensor) -> tuple[torch.Tensor, ...]:
        emb = self.linear(F.silu(emb))
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = torch.chunk(emb, 6, dim=1)
        x = self.norm(x) * (1 + scale_msa[:, None]) + shift_msa[:, None]
        return x, gate_msa, shift_mlp, scale_mlp, gate_mlp


class AdaLayerNormFinal(nn.Module):
    def __init__(self, dim: int, ops: Any = default_ops.default) -> None:
        super().__init__()
        self.linear = ops.Linear(dim, dim * 2)
        self.norm = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)

    def forward(self, x: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        emb = self.linear(F.silu(emb))
        scale, shift = torch.chunk(emb, 2, dim=1)
        return self.norm(x) * (1 + scale)[:, None, :] + shift[:, None, :]


class FeedForward(nn.Module):
    def __init__(self, dim: int, mult: float = 4, ops: Any = default_ops.default) -> None:
        super().__init__()
        inner = int(dim * mult)
        self.ff = nn.Sequential(nn.Sequential(ops.Linear(dim, inner), nn.GELU(approximate="tanh")), nn.Dropout(0.0), ops.Linear(inner, dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.ff(x)


class DiTBlock(nn.Module):
    def __init__(self, dim: int, heads: int, dim_head: int, ff_mult: float = 4, pe_attn_head: int | None = None, ops: Any = default_ops.default) -> None:
        super().__init__()
        self.attn_norm = AdaLayerNorm(dim, ops)
        self.attn = Attention(dim, heads, dim_head, pe_attn_head, ops)
        self.ff_norm = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.ff = FeedForward(dim, ff_mult, ops)

    def forward(self, x: torch.Tensor, t: torch.Tensor, mask: torch.Tensor | None, rope: torch.Tensor | Rope | None) -> torch.Tensor:
        norm, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.attn_norm(x, emb=t)
        x = x + gate_msa.unsqueeze(1) * self.attn(norm, mask, rope)
        norm = self.ff_norm(x) * (1 + scale_mlp[:, None]) + shift_mlp[:, None]
        return x + gate_mlp.unsqueeze(1) * self.ff(norm)


class DiT(nn.Module):
    def __init__(
        self,
        *,
        dim: int,
        depth: int = 8,
        heads: int = 8,
        dim_head: int = 64,
        ff_mult: float = 4,
        mel_dim: int = 100,
        text_num_embeds: int = 256,
        text_dim: int | None = None,
        text_mask_padding: bool = True,
        conv_layers: int = 0,
        pe_attn_head: int | None = None,
        long_skip_connection: bool = False,
        ops: Any = default_ops.default,
        **_unused: Any,
    ) -> None:
        super().__init__()
        text_dim = text_dim if text_dim is not None else mel_dim
        self.dim = dim
        self.time_embed = TimestepEmbedding(dim, ops=ops)
        self.text_embed = TextEmbedding(text_num_embeds, text_dim, mask_padding=text_mask_padding, conv_layers=conv_layers, ops=ops)
        self.input_embed = InputEmbedding(mel_dim, text_dim, dim, ops)
        self.rotary_embed = RotaryEmbedding(dim_head)
        self.transformer_blocks = nn.ModuleList([DiTBlock(dim, heads, dim_head, ff_mult, pe_attn_head, ops) for _ in range(depth)])
        self.long_skip_connection = ops.Linear(dim * 2, dim, bias=False) if long_skip_connection else None
        self.norm_out = AdaLayerNormFinal(dim, ops)
        self.proj_out = ops.Linear(dim, mel_dim)

    def text_condition(self, text: torch.Tensor, seq_len: int) -> tuple[torch.Tensor, torch.Tensor]:
        """The text embedding with and without the text (CFG), computed once per sample as upstream caches it."""
        return self.text_embed(text, seq_len, drop_text=False), self.text_embed(text, seq_len, drop_text=True)

    def forward(
        self, x: torch.Tensor, cond: torch.Tensor, text_embeds: tuple[torch.Tensor, torch.Tensor], time: torch.Tensor, mask: torch.Tensor | None = None, cfg_infer: bool = True
    ) -> torch.Tensor:
        batch, seq_len = x.shape[0], x.shape[1]
        if time.ndim == 0:
            time = time.repeat(batch)
        t = self.time_embed(time)
        text_cond, text_uncond = text_embeds
        if cfg_infer:
            x_cond = self.input_embed(x, cond, text_cond, drop_audio_cond=False, mask=mask)
            x_uncond = self.input_embed(x, cond, text_uncond, drop_audio_cond=True, mask=mask)
            x = torch.cat((x_cond, x_uncond), dim=0)
            t = torch.cat((t, t), dim=0)
            mask = torch.cat((mask, mask), dim=0) if mask is not None else None
        else:
            x = self.input_embed(x, cond, text_cond, drop_audio_cond=False, mask=mask)
        freqs = rope_tables(self.rotary_embed.forward_from_seq_len(seq_len))
        residual = x
        for block in self.transformer_blocks:
            x = block(x, t, mask, freqs)
        if self.long_skip_connection is not None:
            x = self.long_skip_connection(torch.cat((x, residual), dim=-1))
        return self.proj_out(self.norm_out(x, t))
