"""SAN-M: self-attention with a memory block (FunASR's ``funasr.models.sanm``), Paraformer's and CT-Transformer's layers.

Ported from FunASR 1.4.21 (MIT, Copyright (c) 2022 Alibaba Group), computing what its modules compute in the same
order: ``MultiHeadedAttentionSANM`` (q, k and v from one product; the values also pass a depthwise convolution over time,
the "memory", added to the attention's output), ``EncoderLayerSANM`` and ``SANMEncoder`` (pre-norm, the first layer from
the input width to the model's without a residual), ``MultiHeadedAttentionSANMDecoder`` (the memory alone),
``MultiHeadedAttentionCrossAtt``, ``DecoderLayerSANM``, the feed-forward blocks, ``SinusoidalPositionEncoder`` and
LayerNorm with eps 1e-12. Attention is hand-written upstream (q scaled before the product, masked scores -inf, the
softmax's masked weights zeroed): it is this module's attention site's reference (``mizuoto.runtime.accel``).
Inference only: no dropout, no streaming chunks.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from mizuoto.runtime import ops as default_ops
from mizuoto.runtime.accel import Site, attention


def sanm_attention(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: torch.Tensor | None, causal: bool, scale: float | None, enable_gqa: bool, dropout_p: float
) -> torch.Tensor:
    """FunASR's hand-written attention: ``q·scale`` first, then ``q·kᵀ``, masked keys ``-inf``, the softmax with the
    masked weights set to zero (a row with every key masked gives zeros, not NaN). ``mask``: bool, True attends."""
    q = q * (scale if scale is not None else q.shape[-1] ** -0.5)
    scores = torch.matmul(q, k.transpose(-2, -1))
    if mask is None:
        return torch.matmul(torch.softmax(scores, dim=-1), v)
    masked = ~mask
    weights = torch.softmax(scores.masked_fill(masked, -float("inf")), dim=-1).masked_fill(masked, 0.0)
    return torch.matmul(weights, v)


SITE = Site("paraformer", reference=sanm_attention, approx_ok=False)


def layer_norm(size: int) -> nn.LayerNorm:
    return default_ops.default.LayerNorm(size, eps=1e-12)


def sinusoidal_encoding(length: int, depth: int, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    """``SinusoidalPositionEncoder.encode`` for positions 1…length: (1, length, depth)."""
    positions = torch.arange(1, length + 1, device=device)[None, :].type(dtype)
    increment = torch.log(torch.tensor([10000], dtype=dtype, device=device)) / (depth / 2 - 1)
    inv_timescales = torch.exp(torch.arange(depth / 2, device=device).type(dtype) * (-increment))
    scaled = torch.reshape(positions, [1, -1, 1]) * torch.reshape(inv_timescales.reshape(1, -1), [1, 1, -1])
    return torch.cat([torch.sin(scaled), torch.cos(scaled)], dim=2).type(dtype)


class SelfAttention(nn.Module):
    """``MultiHeadedAttentionSANM``."""

    def __init__(self, heads: int, in_size: int, size: int, kernel_size: int, ops: Any = default_ops.default) -> None:
        super().__init__()
        self.h, self.d_k = heads, size // heads
        self.linear_out = ops.Linear(size, size)
        self.linear_q_k_v = ops.Linear(in_size, size * 3)
        self.fsmn_block = ops.Conv1d(size, size, kernel_size, stride=1, padding=0, groups=size, bias=False)
        left = (kernel_size - 1) // 2
        self.pad = (left, kernel_size - 1 - left)

    def memory(self, v: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
        """``forward_fsmn``: the values through the depthwise convolution, plus themselves, masked."""
        b = v.size(0)
        if mask is not None:
            mask = torch.reshape(mask, (b, -1, 1))
            v = v * mask
        x = torch.nn.functional.pad(v.transpose(1, 2), self.pad, value=0.0)
        x = self.fsmn_block(x).transpose(1, 2)
        x = x + v
        return x * mask if mask is not None else x

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
        """``mask``: (batch, 1, time), 1 for real frames (float or bool)."""
        b, t, _ = x.size()
        q, k, v = torch.split(self.linear_q_k_v(x), self.h * self.d_k, dim=-1)
        q_h, k_h, v_h = (torch.reshape(y, (b, t, self.h, self.d_k)).transpose(1, 2) for y in (q, k, v))
        memory = self.memory(v, mask)
        keys = None if mask is None else mask.bool().unsqueeze(1)  # (batch, 1, 1, time)
        out = attention(q_h, k_h, v_h, site=SITE, mask=keys, scale=self.d_k**-0.5)
        out = out.transpose(1, 2).contiguous().view(b, -1, self.h * self.d_k)
        return self.linear_out(out) + memory


class FeedForward(nn.Module):
    """``PositionwiseFeedForward``: w_2(relu(w_1(x)))."""

    def __init__(self, size: int, hidden: int, ops: Any = default_ops.default) -> None:
        super().__init__()
        self.w_1 = ops.Linear(size, hidden)
        self.w_2 = ops.Linear(hidden, size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w_2(torch.relu(self.w_1(x)))


class EncoderLayer(nn.Module):
    """``EncoderLayerSANM`` (pre-norm; a layer that changes the width has no residual around its attention)."""

    def __init__(self, in_size: int, size: int, heads: int, hidden: int, kernel_size: int, ops: Any = default_ops.default) -> None:
        super().__init__()
        self.self_attn = SelfAttention(heads, in_size, size, kernel_size, ops)
        self.feed_forward = FeedForward(size, hidden, ops)
        self.norm1 = layer_norm(in_size)
        self.norm2 = layer_norm(size)
        self.residual = in_size == size

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
        attended = self.self_attn(self.norm1(x), mask)
        x = x + attended if self.residual else attended
        return x + self.feed_forward(self.norm2(x))


class Encoder(nn.Module):
    """``SANMEncoder`` with ``input_layer: pe``: inputs scaled by √width, sinusoidal positions added, the layers, a norm."""

    def __init__(self, input_size: int, size: int, heads: int, hidden: int, blocks: int, kernel_size: int, ops: Any = default_ops.default) -> None:
        super().__init__()
        self.size = size
        self.encoders0 = nn.ModuleList([EncoderLayer(input_size, size, heads, hidden, kernel_size, ops)])
        self.encoders = nn.ModuleList([EncoderLayer(size, size, heads, hidden, kernel_size, ops) for _ in range(blocks - 1)])
        self.after_norm = layer_norm(size)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
        """``x`` (batch, time, input_size); ``mask`` (batch, 1, time) 1 for real frames, or None: (batch, time, size)."""
        x = x * self.size**0.5
        x = x + sinusoidal_encoding(x.size(1), x.size(2), x.dtype, x.device)
        for layer in (*self.encoders0, *self.encoders):
            x = layer(x, mask)
        return self.after_norm(x)


class DecoderMemory(nn.Module):
    """``MultiHeadedAttentionSANMDecoder``: the decoder's "self-attention" is the memory block alone."""

    def __init__(self, size: int, kernel_size: int, ops: Any = default_ops.default) -> None:
        super().__init__()
        self.fsmn_block = ops.Conv1d(size, size, kernel_size, stride=1, padding=0, groups=size, bias=False)
        left = (kernel_size - 1) // 2
        self.pad = (left, kernel_size - 1 - left)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        b = x.size(0)
        mask = torch.reshape(mask, (b, -1, 1))
        x = x * mask
        out = torch.nn.functional.pad(x.transpose(1, 2), self.pad, value=0.0)
        out = self.fsmn_block(out).transpose(1, 2)
        return (out + x) * mask


class CrossAttention(nn.Module):
    """``MultiHeadedAttentionCrossAtt``."""

    def __init__(self, heads: int, size: int, ops: Any = default_ops.default) -> None:
        super().__init__()
        self.h, self.d_k = heads, size // heads
        self.linear_q = ops.Linear(size, size)
        self.linear_k_v = ops.Linear(size, size * 2)
        self.linear_out = ops.Linear(size, size)

    def forward(self, x: torch.Tensor, memory: torch.Tensor, memory_mask: torch.Tensor) -> torch.Tensor:
        b = x.size(0)
        q_h = torch.reshape(self.linear_q(x), (b, -1, self.h, self.d_k)).transpose(1, 2)
        k, v = torch.split(self.linear_k_v(memory), self.h * self.d_k, dim=-1)
        k_h = torch.reshape(k, (b, -1, self.h, self.d_k)).transpose(1, 2)
        v_h = torch.reshape(v, (b, -1, self.h, self.d_k)).transpose(1, 2)
        out = attention(q_h, k_h, v_h, site=SITE, mask=memory_mask.bool().unsqueeze(1), scale=self.d_k**-0.5)
        return self.linear_out(out.transpose(1, 2).contiguous().view(b, -1, self.h * self.d_k))


class DecoderFeedForward(nn.Module):
    """``PositionwiseFeedForwardDecoderSANM``: w_2(norm(relu(w_1(x)))), w_2 without a bias."""

    def __init__(self, size: int, hidden: int, ops: Any = default_ops.default) -> None:
        super().__init__()
        self.w_1 = ops.Linear(size, hidden)
        self.w_2 = ops.Linear(hidden, size, bias=False)
        self.norm = layer_norm(hidden)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w_2(self.norm(torch.relu(self.w_1(x))))


class DecoderLayer(nn.Module):
    """``DecoderLayerSANM``: norm, feed-forward, then (with them) the memory block with a residual and cross-attention."""

    def __init__(self, size: int, heads: int, hidden: int, kernel_size: int, attend: bool, ops: Any = default_ops.default) -> None:
        super().__init__()
        self.feed_forward = DecoderFeedForward(size, hidden, ops)
        self.norm1 = layer_norm(size)
        self.self_attn: DecoderMemory | None = None
        self.src_attn: CrossAttention | None = None
        if attend:
            self.self_attn = DecoderMemory(size, kernel_size, ops)
            self.norm2 = layer_norm(size)
            self.src_attn = CrossAttention(heads, size, ops)
            self.norm3 = layer_norm(size)

    def forward(self, tgt: torch.Tensor, tgt_mask: torch.Tensor, memory: torch.Tensor, memory_mask: torch.Tensor) -> torch.Tensor:
        residual = tgt
        x = self.feed_forward(self.norm1(tgt))
        if self.self_attn is not None:
            x = residual + self.self_attn(self.norm2(x), tgt_mask)
        if self.src_attn is not None:
            x = x + self.src_attn(self.norm3(x), memory, memory_mask)
        return x
