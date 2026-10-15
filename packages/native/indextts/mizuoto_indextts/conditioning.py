"""The GPT's conditioning encoders (index-tts ``gpt/conformer_encoder.py``, ``gpt/conformer/*``, ``gpt/perceiver.py``).

- ``ConformerEncoder``: WeNet's Conformer as IndexTTS keeps it — ``conv2d2`` subsampling (one 3×3 stride-2
  convolution), relative positional encoding *without* the rel-shift (``matrix_bd = (q + v_bias) · p``), a
  convolution module with LayerNorm and SiLU, pre-norm, no macaron FFN, masks filled with ``-inf`` and the
  softmax zeroed where masked.
- ``PerceiverResampler``: lucidrains' perceiver (MIT) — learned latents cross-attending to the latents and the
  context together, GEGLU feed-forward, a final RMSNorm (``F.normalize · √dim · γ``).
"""

from __future__ import annotations

import math
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from mizuoto.runtime import ops as default_ops
from mizuoto.runtime.accel import MaskInfo, Site, attention


def make_pad_mask(lengths: torch.Tensor, max_len: int = 0) -> torch.Tensor:
    batch = lengths.size(0)
    max_len = max_len if max_len > 0 else int(lengths.max().item())
    seq = torch.arange(0, max_len, dtype=torch.int64, device=lengths.device).unsqueeze(0).expand(batch, max_len)
    return seq >= lengths.unsqueeze(-1)


class RelPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000) -> None:
        super().__init__()
        self.xscale = math.sqrt(d_model)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))  # upstream's checkpoints carry it; theirs is used
        self.max_len = max_len

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if x.size(1) >= self.max_len:
            raise ValueError(f"The conditioning clip is too long ({x.size(1)} steps)")
        return x * self.xscale, self.get_buffer("pe")[:, : x.size(1)].to(x.device)


class Conv2dSubsampling2(nn.Module):
    def __init__(self, idim: int, odim: int, ops: Any) -> None:
        super().__init__()
        self.conv = nn.Sequential(ops.Conv2d(1, odim, 3, 2), nn.ReLU())
        self.out = nn.Sequential(ops.Linear(odim * ((idim - 1) // 2), odim))
        self.pos_enc = RelPositionalEncoding(odim)

    def forward(self, x: torch.Tensor, x_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = self.conv(x.unsqueeze(1))
        b, c, t, f = x.size()
        x = self.out(x.transpose(1, 2).contiguous().view(b, t, c * f))
        x, pos_emb = self.pos_enc(x)
        return x, pos_emb, x_mask[:, :, 2::2]


class RelPositionMultiHeadedAttention(nn.Module):
    def __init__(self, n_head: int, n_feat: int, ops: Any) -> None:
        super().__init__()
        self.d_k = n_feat // n_head
        self.h = n_head
        self.linear_q = ops.Linear(n_feat, n_feat)
        self.linear_k = ops.Linear(n_feat, n_feat)
        self.linear_v = ops.Linear(n_feat, n_feat)
        self.linear_out = ops.Linear(n_feat, n_feat)
        self.linear_pos = ops.Linear(n_feat, n_feat, bias=False)
        self.pos_bias_u = nn.Parameter(torch.empty(self.h, self.d_k))
        self.pos_bias_v = nn.Parameter(torch.empty(self.h, self.d_k))

    def forward(self, x: torch.Tensor, mask: torch.Tensor, pos_emb: torch.Tensor) -> torch.Tensor:
        n = x.size(0)
        q = self.linear_q(x).view(n, -1, self.h, self.d_k)
        k = self.linear_k(x).view(n, -1, self.h, self.d_k).transpose(1, 2)
        v = self.linear_v(x).view(n, -1, self.h, self.d_k).transpose(1, 2)
        p = self.linear_pos(pos_emb).view(pos_emb.size(0), -1, self.h, self.d_k).transpose(1, 2)
        q_u = (q + self.pos_bias_u).transpose(1, 2)
        q_v = (q + self.pos_bias_v).transpose(1, 2)
        scores = (torch.matmul(q_u, k.transpose(-2, -1)) + torch.matmul(q_v, p.transpose(-2, -1))) / math.sqrt(self.d_k)
        if mask.size(2) > 0:
            masked = mask.unsqueeze(1).eq(0)[:, :, :, : scores.size(-1)]
            scores = scores.masked_fill(masked, -float("inf"))
            attn = torch.softmax(scores, dim=-1).masked_fill(masked, 0.0)
        else:
            attn = torch.softmax(scores, dim=-1)
        out = torch.matmul(attn, v).transpose(1, 2).contiguous().view(n, -1, self.h * self.d_k)
        return self.linear_out(out)


class PositionwiseFeedForward(nn.Module):
    def __init__(self, idim: int, hidden: int, ops: Any) -> None:
        super().__init__()
        self.w_1 = ops.Linear(idim, hidden)
        self.w_2 = ops.Linear(hidden, idim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w_2(F.silu(self.w_1(x)))


class ConvolutionModule(nn.Module):
    def __init__(self, channels: int, kernel_size: int, ops: Any) -> None:
        super().__init__()
        self.pointwise_conv1 = ops.Conv1d(channels, 2 * channels, kernel_size=1)
        self.depthwise_conv = ops.Conv1d(channels, channels, kernel_size, padding=(kernel_size - 1) // 2, groups=channels)
        self.norm = ops.LayerNorm(channels)
        self.pointwise_conv2 = ops.Conv1d(channels, channels, kernel_size=1)

    def forward(self, x: torch.Tensor, mask_pad: torch.Tensor) -> torch.Tensor:
        x = x.transpose(1, 2)
        if mask_pad.size(2) > 0:
            x = x.masked_fill(~mask_pad, 0.0)
        x = F.glu(self.pointwise_conv1(x), dim=1)
        x = self.depthwise_conv(x)
        x = F.silu(self.norm(x.transpose(1, 2))).transpose(1, 2)
        x = self.pointwise_conv2(x)
        if mask_pad.size(2) > 0:
            x = x.masked_fill(~mask_pad, 0.0)
        return x.transpose(1, 2)


class ConformerEncoderLayer(nn.Module):
    def __init__(self, size: int, heads: int, linear_units: int, kernel_size: int, ops: Any) -> None:
        super().__init__()
        self.self_attn = RelPositionMultiHeadedAttention(heads, size, ops)
        self.feed_forward = PositionwiseFeedForward(size, linear_units, ops)
        self.conv_module = ConvolutionModule(size, kernel_size, ops)
        self.norm_ff = ops.LayerNorm(size, eps=1e-5)
        self.norm_mha = ops.LayerNorm(size, eps=1e-5)
        self.norm_conv = ops.LayerNorm(size, eps=1e-5)
        self.norm_final = ops.LayerNorm(size, eps=1e-5)

    def forward(self, x: torch.Tensor, mask: torch.Tensor, pos_emb: torch.Tensor, mask_pad: torch.Tensor) -> torch.Tensor:
        x = x + self.self_attn(self.norm_mha(x), mask, pos_emb)
        x = x + self.conv_module(self.norm_conv(x), mask_pad)
        x = x + 1.0 * self.feed_forward(self.norm_ff(x))
        return self.norm_final(x)


class ConformerEncoder(nn.Module):
    def __init__(
        self,
        input_size: int,
        output_size: int,
        attention_heads: int,
        linear_units: int,
        num_blocks: int,
        input_layer: str,
        ops: Any = default_ops.default,
        cnn_module_kernel: int = 15,
    ) -> None:
        super().__init__()
        if input_layer != "conv2d2":
            raise ValueError(f"Only the conv2d2 input layer IndexTTS uses is supported, not {input_layer!r}")
        self.embed = Conv2dSubsampling2(input_size, output_size, ops)
        self.after_norm = ops.LayerNorm(output_size, eps=1e-5)
        self.encoders = nn.ModuleList(ConformerEncoderLayer(output_size, attention_heads, linear_units, cnn_module_kernel, ops) for _ in range(num_blocks))

    def forward(self, xs: torch.Tensor, xs_lens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """(B, T, input) and lengths → (B, T', output), mask (B, 1, T')."""
        masks = ~make_pad_mask(xs_lens, xs.size(1)).unsqueeze(1)
        xs, pos_emb, masks = self.embed(xs, masks)
        for layer in self.encoders:
            xs = layer(xs, masks, pos_emb, masks)
        return self.after_norm(xs), masks


# -- perceiver -----------------------------------------------------------------------------------------


def _perceiver(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: torch.Tensor | None, causal: bool, scale: float | None, gqa: bool, dropout: float) -> torch.Tensor:
    """Upstream's einsum attention: masked scores filled with the dtype's lowest value, not -inf."""
    sim = torch.einsum("b h i d, b h j d -> b h i j", q, k) * (q.shape[-1] ** -0.5)
    if mask is not None:
        sim = sim.masked_fill(~mask, -torch.finfo(sim.dtype).max)
    attn = sim.softmax(dim=-1)
    return torch.einsum("b h i j, b h j d -> b h i d", attn, v)


PERCEIVER = Site("indextts.perceiver", reference=_perceiver)


class PerceiverAttention(nn.Module):
    def __init__(self, dim: int, dim_head: int, heads: int, ops: Any) -> None:
        super().__init__()
        self.heads = heads
        inner = dim_head * heads
        self.to_q = ops.Linear(dim, inner, bias=False)
        self.to_kv = ops.Linear(dim, inner * 2, bias=False)
        self.to_out = ops.Linear(inner, dim, bias=False)

    def forward(self, x: torch.Tensor, context: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
        context = torch.cat((x, context), dim=-2)
        q = self.to_q(x)
        k, v = self.to_kv(context).chunk(2, dim=-1)
        b = x.shape[0]
        q, k, v = (t.reshape(b, t.shape[1], self.heads, -1).permute(0, 2, 1, 3) for t in (q, k, v))
        m = mask[:, None, None, :] if mask is not None else None
        out = attention(q, k, v, site=PERCEIVER, mask=m, mask_info=MaskInfo(keys=mask, padding="right") if mask is not None else None)
        return self.to_out(out.permute(0, 2, 1, 3).reshape(b, out.shape[2], -1))


class PerceiverRMSNorm(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.scale = dim**0.5
        self.gamma = nn.Parameter(torch.empty(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(x, dim=-1) * self.scale * self.gamma


class GEGLUFeedForward(nn.Sequential):
    def __init__(self, dim: int, mult: float, ops: Any) -> None:
        inner = int(dim * mult * 2 / 3)
        super().__init__(ops.Linear(dim, inner * 2), _GEGLU(), ops.Linear(inner, dim))


class _GEGLU(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, gate = x.chunk(2, dim=-1)
        return F.gelu(gate) * x


class PerceiverResampler(nn.Module):
    def __init__(
        self, dim: int, dim_context: int, num_latents: int, heads: int = 8, ff_mult: float = 4, depth: int = 2, dim_head: int = 64, ops: Any = default_ops.default
    ) -> None:
        super().__init__()
        self.proj_context: nn.Module = ops.Linear(dim_context, dim) if dim_context != dim else nn.Identity()
        self.latents = nn.Parameter(torch.empty(num_latents, dim))
        self.layers = nn.ModuleList(nn.ModuleList([PerceiverAttention(dim, dim_head, heads, ops), GEGLUFeedForward(dim, ff_mult, ops)]) for _ in range(depth))
        self.norm = PerceiverRMSNorm(dim)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        x = self.proj_context(x)
        latents = self.latents.unsqueeze(0).expand(x.shape[0], -1, -1)
        for attn, ff in self.layers:  # type: ignore[misc]
            latents = attn(latents, x, mask) + latents
            latents = ff(latents) + latents
        return self.norm(latents)
