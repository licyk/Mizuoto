"""The Qwen2/Qwen3-family decoder, with transformers' numerics (4.57).

Module and parameter names follow transformers' (``layers.N.self_attn.q_proj``…), so checkpoints load
as they are.

Parity with transformers needs more than the same maths:
- **RMSNorm** computes in float32 and casts back *before* multiplying by the weight.
- **RoPE**: ``inv_freq`` is computed from int64 positions, as ``ROPE_INIT_FUNCTIONS["default"]`` does;
  cos and sin are computed in float32 and cast to the activations' dtype.
- **Attention** calls SDPA as ``sdpa_attention_forward`` does. Without a mask it passes
  ``enable_gqa=True`` and ``is_causal`` for multi-token queries. With a mask it expands the KV heads
  (``repeat_kv``) and passes the mask with ``is_causal=False``. ``mask_for`` decides between the two as
  transformers' ``create_causal_mask`` does: no mask unless padding or a sliding window needs one. The calls go
  through ``mizuoto.runtime.accel`` (site ``qwen``), whose reference is exactly that SDPA call; ``mask_for`` hands
  the mask down with its structure (``Masked``).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from mizuoto.runtime import ops as default_ops
from mizuoto.runtime.accel import Masked, MaskInfo, Site, attention
from mizuoto.runtime.accel.dispatch import active
from mizuoto.runtime.lm.cache import DynamicCache

SITE = Site("qwen", approx_ok=False)  # autoregressive: quantised attention derails its sampling


@dataclass
class QwenConfig:
    hidden_size: int
    intermediate_size: int
    num_layers: int
    num_heads: int
    num_kv_heads: int
    head_dim: int
    rms_norm_eps: float = 1e-6
    rope_theta: float = 1_000_000.0
    qk_norm: bool = True
    """Qwen3's per-head RMSNorm on q and k (Qwen2 has none)."""
    attention_bias: bool = False
    """Qwen2's bias on q, k and v (and o, when ``o_bias``)."""
    o_bias: bool = False
    sliding_window: int | None = None
    sdpa_gqa: bool = True
    """transformers ≥ 4.53 passes ``enable_gqa`` to SDPA when there is no mask; 4.51 always repeats the KV heads
    and makes q, k and v contiguous (CosyVoice 2 and 3 ran on 4.51)."""


class RMSNorm(nn.Module):
    def __init__(self, size: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.empty(size))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.weight * normalize(x, self.eps)


def normalize(x: torch.Tensor, eps: float) -> torch.Tensor:
    """RMSNorm without its weight, as transformers computes it: in float32, cast back before the weight multiplies.

    Without autograd on a GPU this is ``F.rms_norm`` where that gives the same bits (``_fused_norm_exact``): recent
    torch makes it one kernel instead of eight."""
    if x.device.type == "cuda" and not torch.is_grad_enabled() and _fused_norm_exact(x.device, x.dtype, x.shape[-1], eps):
        return F.rms_norm(x, (x.shape[-1],), eps=eps)
    dtype = x.dtype
    h = x.to(torch.float32)
    variance = h.pow(2).mean(-1, keepdim=True)
    h = h * torch.rsqrt(variance + eps)
    return h.to(dtype)


_FUSED_NORM: dict[tuple[Any, ...], bool] = {}


def _fused_norm_exact(device: torch.device, dtype: torch.dtype, size: int, eps: float) -> bool:
    """Whether ``F.rms_norm`` computes ``normalize``'s arithmetic bit for bit on this device, dtype and size: checked
    once, on rows of random values from 1e-3 to 1e3 (another order of summation, or another reciprocal square root,
    would differ on many of them). torch 2.10 on CUDA: identical in bf16 and fp16, not in fp32."""
    key = (str(device), dtype, size, eps)
    ok = _FUSED_NORM.get(key)
    if ok is None:
        if torch.cuda.is_current_stream_capturing():
            return False
        g = torch.Generator(device=device).manual_seed(0)
        x = (torch.randn(1024, size, generator=g, device=device) * torch.logspace(-3, 3, 1024, device=device)[:, None]).to(dtype)
        h = x.to(torch.float32)
        reference = (h * torch.rsqrt(h.pow(2).mean(-1, keepdim=True) + eps)).to(dtype)
        ok = _FUSED_NORM[key] = bool(torch.equal(reference, F.rms_norm(x, (size,), eps=eps)))
    return ok


def rope_inv_freq(theta: float, dim: int) -> torch.Tensor:
    """``ROPE_INIT_FUNCTIONS["default"]``: ``1 / theta ** (arange(0, dim, 2) / dim)`` in float32."""
    return 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.int64).to(dtype=torch.float) / dim))


class Rotary(nn.Module):
    def __init__(self, head_dim: int, theta: float) -> None:
        super().__init__()
        self.register_buffer("inv_freq", rope_inv_freq(theta, head_dim), persistent=False)

    @torch.no_grad()
    def forward(self, x: torch.Tensor, position_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """cos and sin for ``position_ids`` (batch, length), in ``x``'s dtype."""
        inv_freq = self.get_buffer("inv_freq")
        inv_freq_expanded = inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1).to(x.device)
        position_ids_expanded = position_ids[:, None, :].float()
        device_type = x.device.type if x.device.type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos()
            sin = emb.sin()
        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rope(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)
    return (q * cos) + (rotate_half(q) * sin), (k * cos) + (rotate_half(k) * sin)


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    batch, kv_heads, length, head_dim = x.shape
    if n_rep == 1:
        return x
    return x[:, :, None, :, :].expand(batch, kv_heads, n_rep, length, head_dim).reshape(batch, kv_heads * n_rep, length, head_dim)


def hf_sdpa(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: torch.Tensor | Masked | None, scale: float, n_rep: int, gqa: bool = True) -> torch.Tensor:
    """transformers' ``sdpa_attention_forward``; returns (batch, length, heads, head_dim). ``gqa=False``: 4.51's.
    Under an acceleration policy the key/value heads are not repeated (backends take grouped queries), but with a mask
    they are, as the reference does: SDPA computes grouped heads with a mask only in its math kernel (OmniVoice 0.71×
    and other bits under any policy ending in SDPA)."""
    info = None
    if isinstance(mask, Masked):
        mask, info = mask
    if active(SITE):
        m = mask[:, :, :, : k.shape[-2]] if mask is not None else None
        grouped = m is None or n_rep == 1
        if not grouped:
            k, v = repeat_kv(k, n_rep), repeat_kv(v, n_rep)
        out = attention(q, k, v, site=SITE, mask=m, scale=scale, causal=q.shape[2] > 1 and m is None, enable_gqa=grouped, mask_info=info)
    elif not gqa:
        k, v = repeat_kv(k, n_rep), repeat_kv(v, n_rep)
        m = mask[:, :, :, : k.shape[-2]] if mask is not None else None
        out = attention(q.contiguous(), k.contiguous(), v.contiguous(), site=SITE, mask=m, scale=scale, causal=q.shape[2] > 1 and m is None, mask_info=info)
    elif mask is None:
        out = attention(q, k, v, site=SITE, scale=scale, causal=q.shape[2] > 1, enable_gqa=True)
    else:
        k, v = repeat_kv(k, n_rep), repeat_kv(v, n_rep)
        out = attention(q, k, v, site=SITE, mask=mask[:, :, :, : k.shape[-2]], scale=scale, mask_info=info)
    return out.transpose(1, 2).contiguous()


class Attention(nn.Module):
    def __init__(self, c: QwenConfig, layer: int, ops: Any = default_ops.default) -> None:
        super().__init__()
        o = ops  # namespace of layer classes
        self.layer = layer
        self.head_dim = c.head_dim
        self.n_rep = c.num_heads // c.num_kv_heads
        self.scale = c.head_dim**-0.5
        self.gqa = c.sdpa_gqa
        self.q_proj = o.Linear(c.hidden_size, c.num_heads * c.head_dim, bias=c.attention_bias)
        self.k_proj = o.Linear(c.hidden_size, c.num_kv_heads * c.head_dim, bias=c.attention_bias)
        self.v_proj = o.Linear(c.hidden_size, c.num_kv_heads * c.head_dim, bias=c.attention_bias)
        self.o_proj = o.Linear(c.num_heads * c.head_dim, c.hidden_size, bias=c.o_bias)
        self.q_norm = RMSNorm(c.head_dim, c.rms_norm_eps) if c.qk_norm else nn.Identity()
        self.k_norm = RMSNorm(c.head_dim, c.rms_norm_eps) if c.qk_norm else nn.Identity()
        self.qk_norm = c.qk_norm
        self._ops = ops
        self.qkv_proj: nn.Module | None = None
        """q, k and v as one product, once ``fuse`` has run (q_proj, k_proj and v_proj are then gone)."""
        self._split: list[int] = []

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, mask: torch.Tensor | Masked | None, cache: DynamicCache | None) -> torch.Tensor:
        q, k, v = self.project(x, cos, sin)
        return self.o_proj(self.attend(q, k, v, mask, cache))

    def fuse(self) -> None:
        """Compute q, k and v with one product of the stacked weights (``qkv_proj``). Fewer, larger kernels; on a GPU the
        product picks other kernels than the three apart, so results change in the last bits (an opt-in)."""
        if self.qkv_proj is None:
            parts = [self.q_proj, self.k_proj, self.v_proj]
            self._split = [m.out_features for m in parts]
            self.qkv_proj = fuse_linears(parts, self._ops)
            del self.q_proj, self.k_proj, self.v_proj

    def _qkv(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """q, k and v, (batch, length, heads, head_dim)."""
        shape = (*x.shape[:-1], -1, self.head_dim)
        if self.qkv_proj is not None:
            q, k, v = self.qkv_proj(x).split(self._split, dim=-1)
            return q.view(shape), k.view(shape), v.view(shape)
        return self.q_proj(x).view(shape), self.k_proj(x).view(shape), self.v_proj(x).view(shape)

    def project(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """q, k (normed, rotated) and v, (batch, heads, length, head_dim).

        Without autograd, Qwen3's q and k are normed and rotated together, their heads side by side (identical: each
        head's norm and every element's rotation are computed alone; fewer, larger kernels)."""
        q, k, v = self._qkv(x)
        if self.qk_norm and not torch.is_grad_enabled():
            heads = q.shape[-2]
            qk = torch.cat((q, k), dim=-2)
            q_norm, k_norm = self.q_norm, self.k_norm
            assert isinstance(q_norm, RMSNorm) and isinstance(k_norm, RMSNorm)
            weight = torch.cat((q_norm.weight.expand(heads, -1), k_norm.weight.expand(k.shape[-2], -1)))
            qk = (weight * normalize(qk, q_norm.eps)).transpose(1, 2)
            qk = (qk * cos.unsqueeze(1)) + (rotate_half(qk) * sin.unsqueeze(1))
            return qk[:, :heads], qk[:, heads:], v.transpose(1, 2)
        q = self.q_norm(q).transpose(1, 2)
        k = self.k_norm(k).transpose(1, 2)
        v = v.transpose(1, 2)
        q, k = apply_rope(q, k, cos, sin)
        return q, k, v

    def attend(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: torch.Tensor | Masked | None, cache: DynamicCache | None) -> torch.Tensor:
        """The cache's update and attention: (batch, length, heads × head_dim), before ``o_proj``."""
        if cache is not None:
            k, v = cache.update(self.layer, k, v)
        out = hf_sdpa(q, k, v, mask, self.scale, self.n_rep, self.gqa)
        return out.reshape(*out.shape[:2], -1).contiguous()


class MLP(nn.Module):
    def __init__(self, hidden: int, intermediate: int, ops: Any = default_ops.default) -> None:
        super().__init__()
        o = ops
        self.gate_proj = o.Linear(hidden, intermediate, bias=False)
        self.up_proj = o.Linear(hidden, intermediate, bias=False)
        self.down_proj = o.Linear(intermediate, hidden, bias=False)
        self._ops = ops
        self.gate_up_proj: nn.Module | None = None
        """gate and up as one product, once ``fuse`` has run."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.gate_up_proj is not None:
            gate, up = self.gate_up_proj(x).chunk(2, dim=-1)
            return self.down_proj(F.silu(gate) * up)
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))

    def fuse(self) -> None:
        """gate and up with one product of the stacked weights (see ``Attention.fuse``)."""
        if self.gate_up_proj is None:
            self.gate_up_proj = fuse_linears([self.gate_proj, self.up_proj], self._ops)
            del self.gate_proj, self.up_proj


class DecoderLayer(nn.Module):
    def __init__(self, c: QwenConfig, layer: int, ops: Any = default_ops.default) -> None:
        super().__init__()
        self.self_attn = Attention(c, layer, ops)
        self.mlp = MLP(c.hidden_size, c.intermediate_size, ops)
        self.input_layernorm = RMSNorm(c.hidden_size, c.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(c.hidden_size, c.rms_norm_eps)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, mask: torch.Tensor | Masked | None, cache: DynamicCache | None) -> torch.Tensor:
        x = x + self.self_attn(self.input_layernorm(x), cos, sin, mask, cache)
        return x + self.mlp(self.post_attention_layernorm(x))

    # ``forward`` in two halves around ``self_attn.attend`` (the same operations in the same order), for steps that run
    # the parts of fixed shape as CUDA graphs (``mizuoto.runtime.graphs.DecoderSteps``).
    def attention_in(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.self_attn.project(self.input_layernorm(x), cos, sin)

    def attention_out(self, x: torch.Tensor, attended: torch.Tensor) -> torch.Tensor:
        x = x + self.self_attn.o_proj(attended)
        return x + self.mlp(self.post_attention_layernorm(x))


class QwenDecoder(nn.Module):
    """The layer stack and final norm, fed embeddings (no token embedding: models embed their own inputs)."""

    def __init__(self, c: QwenConfig, ops: Any = default_ops.default) -> None:
        super().__init__()
        self.config = c
        self.layers = nn.ModuleList([DecoderLayer(c, i, ops) for i in range(c.num_layers)])
        self.norm = RMSNorm(c.hidden_size, c.rms_norm_eps)
        self.rotary_emb = Rotary(c.head_dim, c.rope_theta)

    def forward(
        self,
        x: torch.Tensor,
        position_ids: torch.Tensor | None,
        mask: torch.Tensor | Masked | None = None,
        cache: DynamicCache | None = None,
        rope: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        """``rope``: cos and sin computed before for ``position_ids`` (which may then be None)."""
        assert rope is not None or position_ids is not None
        cos, sin = rope if rope is not None else self.rotary_emb(x, position_ids)
        for layer in self.layers:
            x = layer(x, cos, sin, mask, cache)
        return self.norm(x)

    def fuse(self) -> None:
        """Each layer's q/k/v and gate/up projections as one product each (qwen-tts-webui's fused layers): faster on a
        GPU, not bit-identical there. For inference only: the parameters' names change."""
        for layer in self.layers:
            assert isinstance(layer, DecoderLayer)
            layer.self_attn.fuse()
            layer.mlp.fuse()


def fuse_linears(parts: list[Any], ops: Any = default_ops.default) -> nn.Module:
    """One linear layer whose output is the outputs of ``parts`` side by side: their weights (and biases) stacked."""
    first = parts[0]
    biased = [m.bias is not None for m in parts]
    fused = ops.Linear(first.in_features, sum(m.out_features for m in parts), bias=any(biased), device=first.weight.device, dtype=first.weight.dtype)
    fused.weight = nn.Parameter(torch.cat([m.weight.detach() for m in parts]))
    if any(biased):
        fused.bias = nn.Parameter(torch.cat([m.bias.detach() if m.bias is not None else m.weight.new_zeros(m.out_features) for m in parts]))
    return fused


def mask_for(
    attention_mask: torch.Tensor | None,
    query_length: int,
    past_length: int,
    sliding_window: int | None = None,
    causal: bool = True,
    padding: str | None = None,
    padded: bool | None = None,
) -> Masked | None:
    """The 4-D boolean mask transformers passes to SDPA (True: attend), with its structure, or ``None`` when it skips
    it.

    ``attention_mask`` is the (batch, past + query) padding mask (1 for real tokens). As
    ``_ignore_causal_mask_sdpa`` decides, there is no mask when nothing is padded, the query is one
    token or covers every key, and no sliding window applies within the length. ``padded`` says whether
    ``attention_mask`` has a 0 in its first ``past + query`` columns, when the caller knows; otherwise it is read from
    the device (which waits for it).
    """
    kv_length = past_length + query_length
    if attention_mask is None:
        padded = False
    elif padded is None:
        padded = not bool(attention_mask[:, :kv_length].all())
    if not padded and (query_length == 1 or kv_length == query_length) and (sliding_window is None or kv_length < sliding_window):
        return None
    device = attention_mask.device if attention_mask is not None else torch.device("cpu")
    q_idx = torch.arange(past_length, kv_length, device=device)[:, None]
    kv_idx = torch.arange(kv_length, device=device)[None, :]
    allowed = kv_idx <= q_idx if causal else torch.ones(query_length, kv_length, dtype=torch.bool, device=device)
    if sliding_window is not None:
        allowed = allowed & (kv_idx > q_idx - sliding_window)
    batch = attention_mask.shape[0] if attention_mask is not None else 1
    mask = allowed[None, None].expand(batch, 1, query_length, kv_length)
    keys = None
    if attention_mask is not None:
        keys = attention_mask[:, :kv_length].bool()
        mask = mask & keys[:, None, None, :]
    return Masked(mask, MaskInfo(keys=keys, causal=causal, window=sliding_window, padding=padding if keys is not None else None))
