"""The backbone's layers (upstream ``fish_omni/layers.py``): zero-centred RMSNorm, partial RoPE, gated softmax attention
and Gated DeltaNet, with the same operations in the same order.

Upstream keeps each layer's cache on the module, sized for the largest request so far; here each request has its own
(``AttentionState``, ``DeltaState``), as long as its prompt and frames. Attention reads only the cache's first ``kv_len``
positions either way, so the numbers are the same.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from mizuoto.runtime import ops as default_ops
from mizuoto.runtime.accel import Site, attention
from mizuoto_fishspeech.drama3.config import TextConfig

GDN_CHUNK = 64
"""The chunk of the gated delta rule's parallel form."""
BACKBONE = Site("fishspeech.drama3.backbone", approx_ok=False)  # autoregressive: quantised attention derails its sampling
"""The backbone's softmax attention: SDPA with a boolean mask (``grouped_attention``)."""


class ZeroCenteredRMSNorm(nn.Module):
    """RMSNorm scaling by ``1 + weight``, computed in float32."""

    def __init__(self, dim: int, eps: float) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.empty(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = x.float()
        out = out * torch.rsqrt(out.pow(2).mean(-1, keepdim=True) + self.eps)
        return (out * (1.0 + self.weight.float())).type_as(x)


class GatedRMSNorm(nn.Module):
    """Gated DeltaNet's output norm: ``weight · rmsnorm(x) · silu(gate)``."""

    def __init__(self, dim: int, eps: float) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.empty(dim))

    def forward(self, x: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        out = x.float()
        out = out * torch.rsqrt(out.pow(2).mean(-1, keepdim=True) + self.eps)
        out = self.weight * out.to(dtype)
        return (out * F.silu(gate.float())).to(dtype)


def apply_neox_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Rotate the first ``rotary_dim`` channels of ``x`` [B, T, heads, head_dim] (halves, not pairs)."""
    rotary_dim = cos.shape[-1]
    rotary, passthrough = x[..., :rotary_dim].float(), x[..., rotary_dim:]
    first, second = rotary.chunk(2, dim=-1)
    rotated = torch.cat((-second, first), dim=-1)
    cos, sin = cos[None, :, None, :], sin[None, :, None, :]
    out = rotary * cos + rotated * sin
    return torch.cat((out.to(x.dtype), passthrough), dim=-1)


def grouped_attention(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, mask: torch.Tensor | None, site: Site, scale: float | None = None) -> torch.Tensor:
    """Upstream's GQA attention: query [B, H, T, D], key / value [B, KVH, S, D], a boolean mask [T, S]. A single query
    row folds each key head's query heads into rows (the cache read once, the [1, S] mask broadcast over them); more
    rows repeat the key heads."""
    batch, heads, length, head_dim = query.shape
    kv_heads = key.shape[1]
    groups = heads // kv_heads
    if groups > 1 and length == 1:
        folded = query.reshape(batch, kv_heads, groups, head_dim)
        out = attention(folded, key, value, site=site, mask=mask, scale=scale)
        return out.reshape(batch, heads, 1, head_dim)
    if groups > 1:
        key = key.repeat_interleave(groups, dim=1)
        value = value.repeat_interleave(groups, dim=1)
    return attention(query, key, value, site=site, mask=mask, scale=scale)


@dataclass
class LayerContext:
    """What one forward of the backbone gives every layer."""

    input_pos: torch.Tensor
    """[T] contiguous positions of the rows."""
    cos: torch.Tensor
    sin: torch.Tensor
    kv_len: int
    """The cache prefix attended (last position + 1)."""
    continue_state: bool
    """The rows continue the cached sequence (not position 0)."""


class AttentionState:
    """A softmax attention layer's keys and values for one request: [1, KVH, capacity, D]."""

    def __init__(self, capacity: int, kv_heads: int, head_dim: int, dtype: torch.dtype, device: torch.device) -> None:
        self.k = torch.zeros((1, kv_heads, capacity, head_dim), dtype=dtype, device=device)
        self.v = torch.zeros((1, kv_heads, capacity, head_dim), dtype=dtype, device=device)


class DeltaState:
    """A Gated DeltaNet layer's convolution tail (the model's dtype) and recurrent state (float32) for one request."""

    def __init__(self, conv_dim: int, kernel: int, heads: int, key_dim: int, value_dim: int, dtype: torch.dtype, device: torch.device) -> None:
        self.conv = torch.zeros((1, conv_dim, kernel - 1), dtype=dtype, device=device)
        self.recurrent = torch.zeros((1, heads, key_dim, value_dim), dtype=torch.float32, device=device)


class FullAttention(nn.Module):
    """Gated softmax attention: GQA, q/k norm, partial RoPE, a sigmoid output gate."""

    def __init__(self, c: TextConfig, ops: Any = default_ops.default) -> None:
        super().__init__()
        self.num_heads = c.num_attention_heads
        self.num_kv_heads = c.num_key_value_heads
        self.head_dim = c.head_dim
        hidden = c.hidden_size
        # rows of q_proj: each head's [query, gate] pair
        self.q_proj = ops.Linear(hidden, self.num_heads * self.head_dim * 2, bias=False)
        self.k_proj = ops.Linear(hidden, self.num_kv_heads * self.head_dim, bias=False)
        self.v_proj = ops.Linear(hidden, self.num_kv_heads * self.head_dim, bias=False)
        self.o_proj = ops.Linear(self.num_heads * self.head_dim, hidden, bias=False)
        self.q_norm = ZeroCenteredRMSNorm(self.head_dim, c.rms_norm_eps)
        self.k_norm = ZeroCenteredRMSNorm(self.head_dim, c.rms_norm_eps)

    def new_state(self, capacity: int, dtype: torch.dtype, device: torch.device) -> AttentionState:
        return AttentionState(capacity, self.num_kv_heads, self.head_dim, dtype, device)

    def forward(self, x: torch.Tensor, context: LayerContext, state: AttentionState) -> torch.Tensor:
        batch, length, _ = x.shape
        query, gate = self.q_proj(x).view(batch, length, self.num_heads, 2 * self.head_dim).chunk(2, dim=-1)
        gate = gate.reshape(batch, length, self.num_heads * self.head_dim)
        query = self.q_norm(query)
        key = self.k_proj(x).view(batch, length, self.num_kv_heads, self.head_dim)
        key = self.k_norm(key)
        value = self.v_proj(x).view(batch, length, self.num_kv_heads, self.head_dim)
        query = apply_neox_rope(query, context.cos, context.sin).transpose(1, 2)
        key = apply_neox_rope(key, context.cos, context.sin).transpose(1, 2)
        value = value.transpose(1, 2)
        state.k[:batch].index_copy_(2, context.input_pos, key)
        state.v[:batch].index_copy_(2, context.input_pos, value)
        key = state.k[:batch, :, : context.kv_len]
        value = state.v[:batch, :, : context.kv_len]
        key_positions = torch.arange(context.kv_len, device=x.device)
        mask = key_positions[None, :] <= context.input_pos[:, None]
        attended = grouped_attention(query, key, value, mask, BACKBONE, scale=self.head_dim**-0.5)
        attended = attended.transpose(1, 2).reshape(batch, length, self.num_heads * self.head_dim)
        return self.o_proj(attended * torch.sigmoid(gate))


def l2norm(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    return x * torch.rsqrt((x * x).sum(dim=-1, keepdim=True) + eps)


def chunk_gated_delta_rule(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    decay: torch.Tensor,
    beta: torch.Tensor,
    initial_state: torch.Tensor | None = None,
    chunk_size: int = GDN_CHUNK,
) -> tuple[torch.Tensor, torch.Tensor]:
    """The gated delta rule over whole chunks, in float32. query / key [B, T, H, dk] (not yet l2-normed), value
    [B, T, H, dv], decay (log space, ≤ 0) and beta [B, T, H], initial_state [B, H, dk, dv]: the output [B, T, H, dv] in
    the query's dtype and the final float32 state."""
    output_dtype = query.dtype
    batch, length, heads, key_dim = key.shape
    value_dim = value.shape[-1]
    query, key, value, beta, decay = (t.transpose(1, 2).to(torch.float32).contiguous() for t in (query, key, value, beta, decay))
    query = l2norm(query) * key_dim**-0.5
    key = l2norm(key)

    pad = (chunk_size - length % chunk_size) % chunk_size
    query, key, value = (F.pad(t, (0, 0, 0, pad)) for t in (query, key, value))
    beta, decay = (F.pad(t, (0, pad)) for t in (beta, decay))
    chunk_count = (length + pad) // chunk_size

    value_beta = value * beta.unsqueeze(-1)
    key_beta = key * beta.unsqueeze(-1)
    query, key, key_beta, value_beta = (t.reshape(batch, heads, chunk_count, chunk_size, t.shape[-1]) for t in (query, key, key_beta, value_beta))
    decay = decay.reshape(batch, heads, chunk_count, chunk_size)
    upper = torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=query.device).triu(1)
    cumulative = decay.cumsum(dim=-1)
    pairwise = (cumulative.unsqueeze(-1) - cumulative.unsqueeze(-2)).masked_fill(upper, float("-inf")).exp()

    system = (key_beta @ key.transpose(-1, -2)) * pairwise
    intra = (query @ key.transpose(-1, -2)) * pairwise
    decayed_key_beta = key_beta * cumulative.exp().unsqueeze(-1)
    new_values = torch.linalg.solve_triangular(system, value_beta, upper=False, unitriangular=True)
    key_cumulative_decay = torch.linalg.solve_triangular(system, decayed_key_beta, upper=False, unitriangular=True)

    state = query.new_zeros(batch, heads, key_dim, value_dim) if initial_state is None else initial_state.to(torch.float32)
    output = torch.empty_like(new_values)
    query = query * cumulative.exp().unsqueeze(-1)
    key = key * (cumulative[..., -1:] - cumulative).exp().unsqueeze(-1)
    chunk_decay = cumulative[..., -1].exp()[..., None, None]
    for index in range(chunk_count):
        corrected = new_values[:, :, index] - key_cumulative_decay[:, :, index] @ state
        output[:, :, index] = query[:, :, index] @ state + intra[:, :, index] @ corrected
        state = state * chunk_decay[:, :, index] + key[:, :, index].transpose(-1, -2) @ corrected

    output = output.reshape(batch, heads, -1, value_dim)[:, :, :length]
    return output.transpose(1, 2).contiguous().to(output_dtype), state


def recurrent_gated_delta_rule(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    decay: torch.Tensor,
    beta: torch.Tensor,
    initial_state: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """The gated delta rule token by token; the same contract as ``chunk_gated_delta_rule``."""
    output_dtype = query.dtype
    batch, length, heads, key_dim = key.shape
    value_dim = value.shape[-1]
    query, key, value, beta, decay = (t.transpose(1, 2).to(torch.float32).contiguous() for t in (query, key, value, beta, decay))
    query = l2norm(query) * key_dim**-0.5
    key = l2norm(key)
    state = value.new_zeros(batch, heads, key_dim, value_dim) if initial_state is None else initial_state.to(torch.float32)
    output = torch.empty_like(value)
    for index in range(length):
        state = state * decay[:, :, index].exp()[..., None, None]
        k_t = key[:, :, index]
        remembered = (state * k_t.unsqueeze(-1)).sum(dim=-2)
        delta = (value[:, :, index] - remembered) * beta[:, :, index].unsqueeze(-1)
        state = state + k_t.unsqueeze(-1) * delta.unsqueeze(-2)
        output[:, :, index] = (state * query[:, :, index].unsqueeze(-1)).sum(dim=-2)
    return output.transpose(1, 2).contiguous().to(output_dtype), state


class GatedDeltaNet(nn.Module):
    """Linear attention by the gated delta rule, after a short causal depthwise convolution."""

    def __init__(self, c: TextConfig, ops: Any = default_ops.default) -> None:
        super().__init__()
        hidden = c.hidden_size
        self.num_k_heads = c.linear_num_key_heads
        self.num_v_heads = c.linear_num_value_heads
        self.head_k_dim = c.linear_key_head_dim
        self.head_v_dim = c.linear_value_head_dim
        self.key_dim = self.num_k_heads * self.head_k_dim
        self.value_dim = self.num_v_heads * self.head_v_dim
        self.conv_dim = 2 * self.key_dim + self.value_dim
        self.conv_kernel_size = c.linear_conv_kernel_dim
        # rows of in_proj_qkv: query, key, value
        self.in_proj_qkv = ops.Linear(hidden, self.conv_dim, bias=False)
        self.in_proj_z = ops.Linear(hidden, self.value_dim, bias=False)
        self.in_proj_b = ops.Linear(hidden, self.num_v_heads, bias=False)
        self.in_proj_a = ops.Linear(hidden, self.num_v_heads, bias=False)
        self.conv1d = ops.Conv1d(self.conv_dim, self.conv_dim, kernel_size=self.conv_kernel_size, groups=self.conv_dim, bias=False)
        self.dt_bias = nn.Parameter(torch.empty(self.num_v_heads))
        self.A_log = nn.Parameter(torch.empty(self.num_v_heads))
        self.norm = GatedRMSNorm(self.head_v_dim, c.rms_norm_eps)
        self.out_proj = ops.Linear(self.value_dim, hidden, bias=False)

    def new_state(self, capacity: int, dtype: torch.dtype, device: torch.device) -> DeltaState:
        return DeltaState(self.conv_dim, self.conv_kernel_size, self.num_v_heads, self.head_k_dim, self.head_v_dim, dtype, device)

    def forward(self, x: torch.Tensor, context: LayerContext, state: DeltaState) -> torch.Tensor:
        batch, length, _ = x.shape
        mixed = self.in_proj_qkv(x).transpose(1, 2)  # [B, C, T]
        history = self.conv_kernel_size - 1
        previous = state.conv[:batch].to(mixed.dtype) if context.continue_state else mixed.new_zeros(batch, self.conv_dim, history)
        padded = torch.cat((previous, mixed), dim=-1)
        state.conv[:batch].copy_(padded[..., -history:])
        # the convolution's weight read here, inside the layer's forward (a streamed unit's weights are there)
        mixed = F.silu(F.conv1d(padded, self.conv1d.weight, groups=self.conv_dim))
        query, key, value = mixed.transpose(1, 2).split([self.key_dim, self.key_dim, self.value_dim], dim=-1)
        query = query.reshape(batch, length, self.num_k_heads, self.head_k_dim)
        key = key.reshape(batch, length, self.num_k_heads, self.head_k_dim)
        value = value.reshape(batch, length, self.num_v_heads, self.head_v_dim)
        z = self.in_proj_z(x).reshape(batch, length, self.num_v_heads, self.head_v_dim)
        beta = torch.sigmoid(self.in_proj_b(x).float())
        decay = -self.A_log.float().exp() * F.softplus(self.in_proj_a(x).float() + self.dt_bias.float())
        groups = self.num_v_heads // self.num_k_heads
        if groups > 1:
            query = query.repeat_interleave(groups, dim=2)
            key = key.repeat_interleave(groups, dim=2)
        initial_state = state.recurrent[:batch] if context.continue_state else None
        rule = recurrent_gated_delta_rule if context.continue_state and length == 1 else chunk_gated_delta_rule
        core, final_state = rule(query, key, value, decay, beta, initial_state)
        state.recurrent[:batch].copy_(final_state)
        core = self.norm(core.reshape(-1, self.head_v_dim), z.reshape(-1, self.head_v_dim)).reshape(batch, length, self.value_dim)
        return self.out_proj(core)
