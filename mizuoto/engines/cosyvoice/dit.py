"""CosyVoice 3's flow estimator (upstream ``flow/DiT``, Apache-2.0, after F5-TTS's DiT, MIT).

F5-TTS's AdaLN-zero blocks over [x | condition | mu | speaker], with causal convolutional position embedding
and a 4-D attention mask (chunked when streaming). Rotary embedding is applied to the queries and keys
before they are split into heads, so only the first head's channels rotate (x-transformers' partial rotary
embedding over the whole projection). Parameter names are upstream's (``decoder.estimator.*`` in ``flow.pt``).
"""

from __future__ import annotations

from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from mizuoto.nn.transformer.dit import AdaLayerNorm, AdaLayerNormFinal, FeedForward, Rope, RotaryEmbedding, TimestepEmbedding, apply_rotary, rope_tables
from mizuoto.nn.transformer.wenet import add_optional_chunk_mask
from mizuoto.runtime import ops as default_ops
from mizuoto.runtime.accel import Site, attention

SITE = Site("cosyvoice.dit")


class CausalConvPositionEmbedding(nn.Module):
    def __init__(self, dim: int, ops: Any, kernel_size: int = 31, groups: int = 16) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.conv1 = nn.Sequential(ops.Conv1d(dim, dim, kernel_size, groups=groups, padding=0), nn.Mish())
        self.conv2 = nn.Sequential(ops.Conv1d(dim, dim, kernel_size, groups=groups, padding=0), nn.Mish())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 2, 1)
        x = self.conv1(F.pad(x, (self.kernel_size - 1, 0, 0, 0)))
        x = self.conv2(F.pad(x, (self.kernel_size - 1, 0, 0, 0)))
        return x.permute(0, 2, 1)


class InputEmbedding(nn.Module):
    def __init__(self, mel_dim: int, mu_dim: int, out_dim: int, spk_dim: int, ops: Any) -> None:
        super().__init__()
        self.spk_dim = spk_dim
        self.proj = ops.Linear(mel_dim * 2 + mu_dim + spk_dim, out_dim)
        self.conv_pos_embed = CausalConvPositionEmbedding(out_dim, ops)

    def forward(self, x: torch.Tensor, cond: torch.Tensor, mu: torch.Tensor, spks: torch.Tensor) -> torch.Tensor:
        parts = [x, cond, mu]
        if self.spk_dim > 0:
            parts.append(spks.unsqueeze(1).expand(spks.shape[0], x.shape[1], spks.shape[-1]))
        x = self.proj(torch.cat(parts, dim=-1))
        return self.conv_pos_embed(x) + x


class Attention(nn.Module):
    def __init__(self, dim: int, heads: int, dim_head: int, ops: Any) -> None:
        super().__init__()
        self.heads = heads
        inner = dim_head * heads
        self.to_q = ops.Linear(dim, inner)
        self.to_k = ops.Linear(dim, inner)
        self.to_v = ops.Linear(dim, inner)
        self.to_out = nn.ModuleList([ops.Linear(inner, dim), nn.Dropout(0.0)])

    def forward(self, x: torch.Tensor, mask: torch.Tensor, rope: torch.Tensor | Rope) -> torch.Tensor:
        b = x.shape[0]
        q = apply_rotary(self.to_q(x), rope, 1.0)
        k = apply_rotary(self.to_k(x), rope, 1.0)
        v = self.to_v(x)
        head_dim = k.shape[-1] // self.heads
        q = q.view(b, -1, self.heads, head_dim).transpose(1, 2)
        k = k.view(b, -1, self.heads, head_dim).transpose(1, 2)
        v = v.view(b, -1, self.heads, head_dim).transpose(1, 2)
        out = attention(q, k, v, site=SITE, mask=mask)
        out = out.transpose(1, 2).reshape(b, -1, self.heads * head_dim).to(q.dtype)
        out = self.to_out[0](out)
        return out.masked_fill(~mask[:, 0, -1].unsqueeze(-1), 0.0)


class DiTBlock(nn.Module):
    def __init__(self, dim: int, heads: int, dim_head: int, ff_mult: float, ops: Any) -> None:
        super().__init__()
        self.attn_norm = AdaLayerNorm(dim, ops)
        self.attn = Attention(dim, heads, dim_head, ops)
        self.ff_norm = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.ff = FeedForward(dim, ff_mult, ops)

    def forward(self, x: torch.Tensor, t: torch.Tensor, mask: torch.Tensor, rope: torch.Tensor | Rope) -> torch.Tensor:
        norm, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.attn_norm(x, emb=t)
        x = x + gate_msa.unsqueeze(1) * self.attn(norm, mask, rope)
        norm = self.ff_norm(x) * (1 + scale_mlp[:, None]) + shift_mlp[:, None]
        return x + gate_mlp.unsqueeze(1) * self.ff(norm)


class DiT(nn.Module):
    def __init__(
        self,
        dim: int = 1024,
        depth: int = 22,
        heads: int = 16,
        dim_head: int = 64,
        ff_mult: float = 2,
        mel_dim: int = 80,
        mu_dim: int = 80,
        spk_dim: int = 80,
        static_chunk_size: int = 50,
        ops: Any = default_ops.default,
    ) -> None:
        super().__init__()
        self.time_embed = TimestepEmbedding(dim, ops=ops)
        self.input_embed = InputEmbedding(mel_dim, mu_dim, dim, spk_dim, ops)
        self.rotary_embed = RotaryEmbedding(dim_head)
        self.transformer_blocks = nn.ModuleList(DiTBlock(dim, heads, dim_head, ff_mult, ops) for _ in range(depth))
        self.norm_out = AdaLayerNormFinal(dim, ops)
        self.proj_out = ops.Linear(dim, mel_dim)
        self.static_chunk_size = static_chunk_size

    def forward(self, x: torch.Tensor, mask: torch.Tensor, mu: torch.Tensor, t: torch.Tensor, spks: torch.Tensor, cond: torch.Tensor, streaming: bool = False) -> torch.Tensor:
        x, mu, cond = x.transpose(1, 2), mu.transpose(1, 2), cond.transpose(1, 2)
        batch, seq_len = x.shape[0], x.shape[1]
        if t.ndim == 0:
            t = t.repeat(batch)
        t = self.time_embed(t)
        x = self.input_embed(x, cond, mu, spks)
        rope = rope_tables(self.rotary_embed.forward_from_seq_len(seq_len))
        if streaming:
            attn_mask = add_optional_chunk_mask(x, mask.bool(), self.static_chunk_size).unsqueeze(dim=1)
        else:
            attn_mask = add_optional_chunk_mask(x, mask.bool(), 0).repeat(1, x.size(1), 1).unsqueeze(dim=1)
        for block in self.transformer_blocks:
            x = block(x, t, attn_mask.bool(), rope)
        return self.proj_out(self.norm_out(x, t)).transpose(1, 2)
