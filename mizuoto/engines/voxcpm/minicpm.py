"""MiniCPM-4, VoxCPM's decoder (upstream ``modules/minicpm4``): the text-semantic LM, the residual acoustic LM, the
local encoder and the local DiT are all this module at different sizes.

Kept as upstream computes it, for identical output:
- RMSNorm takes the variance in float32 and multiplies the weight after casting back.
- LongRoPE's cos/sin tables are built in float32 when the module is made and are buffers, so casting the model casts
  them (upstream's bf16 model rotates with bf16-rounded tables, in float32).
- Attention is SDPA with ``enable_gqa``, on contiguous q, k, v: causal for the prefill, bidirectional for the local
  modules, and one step against the static cache with an explicit ``(1, 1, 1, L)`` boolean mask (``StaticKVCache``:
  zeros past the current length). Upstream attends over the whole cache (8192 positions); on the CPU only over its
  first ``ATTEND_BLOCK`` multiple past the current length, mask kept: the masked positions add exact zeros, and SDPA
  gives the same bits for any cut at a multiple of 64 (checked at every position up to 3100, bf16 and fp32). Elsewhere
  over the whole cache (``attend_span``).
- muP scales the residual branches only when the config says ``use_mup``.
"""

from __future__ import annotations

import math

import torch
from torch import nn

from mizuoto.engines.voxcpm.config import LmConfig
from mizuoto.runtime import ops
from mizuoto.runtime.accel import MaskInfo, Site, attention

SITE = Site("voxcpm.minicpm")

ATTEND_BLOCK = 256
"""On the CPU a step attends over the cache up to the next multiple of this past the current length (``attend_span``)."""


def attend_span(length: int, max_length: int, device: torch.device) -> int:
    """How much of a ``max_length`` static cache holding ``length`` positions a step attends over: on the CPU the next
    ``ATTEND_BLOCK`` multiple, which gives the same bits as the whole cache there; elsewhere the whole cache, as upstream.
    On CUDA the step takes SDPA's math path, whose matrix-vector product and softmax split their sums by the key length,
    so no cut short of the whole cache gives the same bits (checked at every position up to 3100, bf16, fp16 and fp32)."""
    if device.type != "cpu":
        return max_length
    return min(max_length, -(-length // ATTEND_BLOCK) * ATTEND_BLOCK)


def rms_layernorm(hidden: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    old_dtype = hidden.dtype
    variance = hidden.to(torch.float32).pow(2).mean(dim=-1, keepdim=True)
    hidden = (hidden * torch.rsqrt(variance + eps)).to(old_dtype)
    return hidden * weight


class RMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return rms_layernorm(hidden_states, self.weight, self.variance_epsilon)


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    orig_dtype = q.dtype
    q = q.to(torch.float32)
    k = k.to(torch.float32)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed.to(orig_dtype), k_embed.to(orig_dtype)


class LongRoPE(nn.Module):
    """``MiniCPMLongRoPE``: the cos/sin tables of every position up to ``max_position_embeddings``."""

    cos_cached: torch.Tensor
    sin_cached: torch.Tensor

    def __init__(self, config: LmConfig) -> None:
        super().__init__()
        dim = config.head_dim
        scaling = config.rope_scaling
        scale = config.max_position_embeddings / scaling.original_max_position_embeddings
        self.scaling_factor = math.sqrt(1 + math.log(scale) / math.log(scaling.original_max_position_embeddings))
        inv_freq = 1.0 / (float(config.rope_theta) ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        seq_len = config.max_position_embeddings
        t = torch.arange(seq_len, dtype=inv_freq.dtype)
        factors = scaling.long_factor if seq_len > scaling.original_max_position_embeddings else scaling.short_factor
        ext_factors = torch.tensor(factors, dtype=torch.float32)
        freqs = torch.mul(torch.outer(t, 1.0 / ext_factors), inv_freq.to(torch.float32))
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos().to(torch.float32) * self.scaling_factor, persistent=False)
        self.register_buffer("sin_cached", emb.sin().to(torch.float32) * self.scaling_factor, persistent=False)

    def forward(self, position_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.cos_cached[position_ids], self.sin_cached[position_ids]


class Attention(nn.Module):
    def __init__(self, config: LmConfig) -> None:
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.head_dim = config.head_dim
        self.num_key_value_heads = config.num_key_value_heads
        self.q_proj = ops.default.Linear(config.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = ops.default.Linear(config.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.v_proj = ops.default.Linear(config.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.o_proj = ops.default.Linear(self.num_heads * self.head_dim, config.hidden_size, bias=False)

    def forward(
        self, hidden_states: torch.Tensor, position_emb: tuple[torch.Tensor, torch.Tensor] | None, is_causal: bool
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        bsz, q_len, _ = hidden_states.size()
        query_states = self.q_proj(hidden_states).view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        if position_emb is not None:
            cos, sin = position_emb
            query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
        query_states = query_states.contiguous()
        key_states = key_states.contiguous()
        value_states = value_states.contiguous()
        attn_output = attention(query_states, key_states, value_states, site=SITE, causal=is_causal, enable_gqa=True)
        attn_output = attn_output.transpose(1, 2).contiguous().reshape(bsz, q_len, self.num_heads * self.head_dim)
        return self.o_proj(attn_output), (key_states, value_states)

    def forward_step(
        self,
        hidden_states: torch.Tensor,
        position_emb: tuple[torch.Tensor, torch.Tensor] | None,
        position_id: torch.Tensor,
        kv_cache: tuple[torch.Tensor, torch.Tensor],
        span: int | None = None,
    ) -> torch.Tensor:
        """One position against ``kv_cache``; attention reads its first ``span`` positions (all by default)."""
        bsz, _ = hidden_states.size()
        query_states = self.q_proj(hidden_states).view(bsz, 1, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(bsz, 1, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(bsz, 1, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        if position_emb is not None:
            cos, sin = position_emb
            query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
        key_cache, value_cache = kv_cache
        key_cache[:, :, position_id, :] = key_states
        value_cache[:, :, position_id, :] = value_states
        if span is not None and span < key_cache.size(2):
            key_cache, value_cache = key_cache[:, :, :span], value_cache[:, :, :span]
        attn_mask = (torch.arange(key_cache.size(2), device=key_cache.device) <= position_id).view(1, 1, 1, -1)
        attn_output = attention(
            query_states.contiguous(),
            key_cache.contiguous(),
            value_cache.contiguous(),
            site=SITE,
            mask=attn_mask,
            enable_gqa=True,
            mask_info=MaskInfo(keys=attn_mask.view(1, -1), padding="right"),
        )
        attn_output = attn_output.transpose(1, 2).contiguous().reshape(bsz, self.num_heads * self.head_dim)
        return self.o_proj(attn_output)


class MLP(nn.Module):
    def __init__(self, config: LmConfig) -> None:
        super().__init__()
        self.gate_proj = ops.default.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj = ops.default.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = ops.default.Linear(config.intermediate_size, config.hidden_size, bias=False)
        self.act_fn = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


class DecoderLayer(nn.Module):
    def __init__(self, config: LmConfig) -> None:
        super().__init__()
        self.self_attn = Attention(config)
        self.mlp = MLP(config)
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.scale_depth = config.scale_depth
        self.num_hidden_layers = config.num_hidden_layers
        self.use_mup = config.use_mup

    def _residual(self, residual: torch.Tensor, hidden_states: torch.Tensor) -> torch.Tensor:
        if self.use_mup:
            return residual + hidden_states * (self.scale_depth / math.sqrt(self.num_hidden_layers))
        return residual + hidden_states

    def forward(
        self, hidden_states: torch.Tensor, position_emb: tuple[torch.Tensor, torch.Tensor] | None, is_causal: bool
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        residual = hidden_states
        hidden_states, present = self.self_attn(self.input_layernorm(hidden_states), position_emb, is_causal)
        hidden_states = self._residual(residual, hidden_states)
        residual = hidden_states
        hidden_states = self._residual(residual, self.mlp(self.post_attention_layernorm(hidden_states)))
        return hidden_states, present

    def forward_step(
        self,
        hidden_states: torch.Tensor,
        position_emb: tuple[torch.Tensor, torch.Tensor] | None,
        position_id: torch.Tensor,
        kv_cache: tuple[torch.Tensor, torch.Tensor],
        span: int | None = None,
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self._residual(residual, self.self_attn.forward_step(self.input_layernorm(hidden_states), position_emb, position_id, kv_cache, span))
        residual = hidden_states
        return self._residual(residual, self.mlp(self.post_attention_layernorm(hidden_states)))


class StaticKVCache:
    """Upstream's ``StaticKVCache``: one zeroed tensor for every layer's keys and values up to ``max_length``."""

    def __init__(self, num_layers: int, num_kv_heads: int, dim_kv_head: int, batch_size: int, device: torch.device | str, dtype: torch.dtype, max_length: int) -> None:
        self.max_length = max_length
        self.num_layers = num_layers
        self.kv_cache = torch.zeros(2, num_layers, batch_size, num_kv_heads, max_length, dim_kv_head, device=device, dtype=dtype)
        self.current_length = 0

    def get_layer_cache(self, layer_idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.kv_cache[0, layer_idx], self.kv_cache[1, layer_idx]

    def step(self) -> int:
        if self.current_length >= self.max_length:
            raise ValueError("KV cache is full")
        current = self.current_length
        self.current_length += 1
        return current

    def fill_caches(self, kv_caches: list[tuple[torch.Tensor, torch.Tensor]]) -> None:
        self.current_length = kv_caches[0][0].size(2)
        self.kv_cache.zero_()
        for i in range(self.num_layers):
            self.kv_cache[0, i, :, :, : self.current_length, :] = kv_caches[i][0]
            self.kv_cache[1, i, :, :, : self.current_length, :] = kv_caches[i][1]


class MiniCPMModel(nn.Module):
    """The decoder stack. ``forward`` runs a whole sequence (prefill, or the local modules); ``forward_step`` one position
    against ``kv_cache``, which the caller makes with ``new_cache``."""

    def __init__(self, config: LmConfig) -> None:
        super().__init__()
        self.config = config
        self.embed_tokens: nn.Module = ops.default.Embedding(config.vocab_size, config.hidden_size) if config.vocab_size > 0 else nn.Identity()
        self.layers = nn.ModuleList([DecoderLayer(config) for _ in range(config.num_hidden_layers)])
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rope_emb: LongRoPE | None = None if config.no_rope else LongRoPE(config)
        self.kv_cache: StaticKVCache | None = None

    def new_cache(self, max_length: int, device: torch.device | str, dtype: torch.dtype, batch_size: int = 1) -> StaticKVCache:
        c = self.config
        return StaticKVCache(c.num_hidden_layers, c.num_key_value_heads, c.head_dim, batch_size, device, dtype, max_length)

    def forward(self, inputs_embeds: torch.Tensor, is_causal: bool = True) -> tuple[torch.Tensor, list[tuple[torch.Tensor, torch.Tensor]]]:
        position_emb = None
        if self.rope_emb is not None:
            position_ids = torch.arange(0, inputs_embeds.size(1), dtype=torch.long, device=inputs_embeds.device)
            position_emb = self.rope_emb(position_ids)
        hidden_states = inputs_embeds
        caches = []
        for layer in self.layers:
            hidden_states, present = layer(hidden_states, position_emb, is_causal)
            caches.append(present)
        return self.norm(hidden_states), caches

    def forward_step(self, inputs_embeds: torch.Tensor, position_id: torch.Tensor) -> torch.Tensor:
        assert self.kv_cache is not None, "the KV cache is not set up"
        position_emb = self.rope_emb(position_id) if self.rope_emb is not None else None
        # the positions written so far (``step()`` counted this one): a whole block of them on the CPU
        span = attend_span(self.kv_cache.current_length, self.kv_cache.max_length, inputs_embeds.device)
        hidden_states = inputs_embeds
        for i, layer in enumerate(self.layers):
            hidden_states = layer.forward_step(hidden_states, position_emb, position_id, self.kv_cache.get_layer_cache(i), span)  # ty: ignore[call-non-callable]
        return self.norm(hidden_states)
