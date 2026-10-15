"""Fish Speech S2 Pro's dual-AR transformer (fish-speech ``models/text2semantic/llama.py``, Fish Audio Research License).

A slow transformer (Qwen3-4B-shaped: fused ``wqkv``, per-head RMSNorm on q and k) reads text tokens and,
on semantic positions, the sum of ten codebook embeddings scaled by 1/√11; it predicts the next token.
A fast transformer (4 layers) then predicts the frame's ten codes one after the other from the slow
model's normed hidden state.

The arithmetic follows upstream operation by operation:
- Fish's RMSNorm (``x * rsqrt(mean(x·x) + eps)`` in float32, then ``* weight``); q/k norms are ``nn.RMSNorm``;
- RoPE over adjacent pairs, with the frequency table rounded to bfloat16 as upstream stores it;
- the slow model's prefill calls SDPA with a boolean causal mask; its decoding steps run SDPA's math kernel
  (``torch.ops.aten._scaled_dot_product_attention_math``, the kernel upstream selects with
  ``sdpa_kernel(MATH)``, without changing any process-wide setting);
- the fast model's attention is upstream's hand-written softmax over a ten-position cache.

The slow model's cache grows by concatenation; after ``repeat_interleave`` the keys and values are the same
tensors upstream builds from its preallocated 32k cache.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from mizuoto.runtime import ops as default_ops
from mizuoto.runtime.accel import Site, attention
from mizuoto.runtime.accel import math_kernel as sdpa_math


@dataclass
class ModelArgs:
    vocab_size: int
    n_layer: int
    n_head: int
    n_local_heads: int
    head_dim: int
    dim: int
    intermediate_size: int
    rope_base: float
    norm_eps: float
    max_seq_len: int
    attention_qk_norm: bool
    attention_qkv_bias: bool
    attention_o_bias: bool
    tie_word_embeddings: bool
    codebook_size: int
    num_codebooks: int
    semantic_begin_id: int
    semantic_end_id: int
    n_fast_layer: int
    fast_dim: int
    fast_n_head: int
    fast_n_local_heads: int
    fast_head_dim: int
    fast_intermediate_size: int
    fast_attention_qk_norm: bool
    fast_attention_qkv_bias: bool
    fast_attention_o_bias: bool
    scale_codebook_embeddings: bool = True
    norm_fastlayer_input: bool = True
    use_gradient_checkpointing: bool = True
    """Upstream's training checkpoints each layer (recomputed in the backward pass)."""

    @classmethod
    def from_config(cls, data: dict[str, Any]) -> ModelArgs:
        """``BaseModelArgs.from_pretrained``: the S2 checkpoint's ``config.json`` (``fish_qwen3_omni``), or the flat
        ``dual_ar`` one upstream's ``save_pretrained`` writes (a fine-tuned S2 model, as ``merge_lora`` saves it)."""
        if data.get("model_type") == "dual_ar":
            return cls.from_dual_ar(data)
        if data.get("model_type") != "fish_qwen3_omni":
            raise ValueError(f"Unsupported Fish Speech model type {data.get('model_type')!r}")
        tc, adc = data["text_config"], data["audio_decoder_config"]

        def heads(c: dict[str, Any]) -> int:
            n = c.get("n_local_heads", -1)
            return c["n_head"] if n in (-1, None) else int(n)

        def inter(c: dict[str, Any]) -> int:
            if c.get("intermediate_size"):
                return int(c["intermediate_size"])
            hidden = int(2 * (4 * c["dim"]) / 3)
            return hidden if hidden % 256 == 0 else hidden + 256 - hidden % 256

        head_dim = tc.get("head_dim") or tc["dim"] // tc["n_head"]
        return cls(
            vocab_size=tc["vocab_size"],
            n_layer=tc["n_layer"],
            n_head=tc["n_head"],
            n_local_heads=heads(tc),
            head_dim=head_dim,
            dim=tc["dim"],
            intermediate_size=inter(tc),
            rope_base=tc.get("rope_base", 10000),
            norm_eps=tc.get("norm_eps", 1e-5),
            max_seq_len=tc.get("max_seq_len", 2048),
            attention_qk_norm=bool(tc.get("attention_qk_norm", False)),
            attention_qkv_bias=bool(tc.get("attention_qkv_bias", False)),
            attention_o_bias=bool(tc.get("attention_o_bias", False)),
            tie_word_embeddings=bool(tc.get("tie_word_embeddings", True)),
            codebook_size=adc["vocab_size"],
            num_codebooks=adc["num_codebooks"],
            semantic_begin_id=data.get("semantic_start_token_id", 0),
            semantic_end_id=data.get("semantic_end_token_id", 0),
            n_fast_layer=adc["n_layer"],
            fast_dim=adc.get("dim") or tc["dim"],
            fast_n_head=adc.get("n_head") or tc["n_head"],
            fast_n_local_heads=heads(adc) if adc.get("n_local_heads") else heads(tc),
            fast_head_dim=adc.get("head_dim") or head_dim,
            fast_intermediate_size=adc.get("intermediate_size") or inter(tc),
            fast_attention_qk_norm=bool(adc.get("attention_qk_norm", tc.get("attention_qk_norm", False))),
            fast_attention_qkv_bias=bool(adc.get("attention_qkv_bias", tc.get("attention_qkv_bias", False))),
            fast_attention_o_bias=bool(adc.get("attention_o_bias", tc.get("attention_o_bias", False))),
            use_gradient_checkpointing=bool(tc.get("use_gradient_checkpointing", True)),
        )

    @classmethod
    def from_dual_ar(cls, data: dict[str, Any]) -> ModelArgs:
        """``DualARModelArgs`` with its ``__post_init__`` defaults."""
        c = dual_ar_args(data)
        return cls(
            vocab_size=c["vocab_size"],
            n_layer=c["n_layer"],
            n_head=c["n_head"],
            n_local_heads=c["n_local_heads"],
            head_dim=c["head_dim"],
            dim=c["dim"],
            intermediate_size=c["intermediate_size"],
            rope_base=c["rope_base"],
            norm_eps=c["norm_eps"],
            max_seq_len=c["max_seq_len"],
            attention_qk_norm=bool(c["attention_qk_norm"]),
            attention_qkv_bias=bool(c["attention_qkv_bias"]),
            attention_o_bias=bool(c["attention_o_bias"]),
            tie_word_embeddings=bool(c["tie_word_embeddings"]),
            codebook_size=c["codebook_size"],
            num_codebooks=c["num_codebooks"],
            semantic_begin_id=c["semantic_begin_id"],
            semantic_end_id=c["semantic_end_id"],
            n_fast_layer=c["n_fast_layer"],
            fast_dim=c["fast_dim"],
            fast_n_head=c["fast_n_head"],
            fast_n_local_heads=c["fast_n_local_heads"],
            fast_head_dim=c["fast_head_dim"],
            fast_intermediate_size=c["fast_intermediate_size"],
            fast_attention_qk_norm=bool(c["fast_attention_qk_norm"]),
            fast_attention_qkv_bias=bool(c["fast_attention_qkv_bias"]),
            fast_attention_o_bias=bool(c["fast_attention_o_bias"]),
            scale_codebook_embeddings=bool(c["scale_codebook_embeddings"]),
            norm_fastlayer_input=bool(c["norm_fastlayer_input"]),
            use_gradient_checkpointing=bool(c["use_gradient_checkpointing"]),
        )

    def fast(self) -> ModelArgs:
        return replace(
            self,
            dim=self.fast_dim,
            n_head=self.fast_n_head,
            n_local_heads=self.fast_n_local_heads,
            head_dim=self.fast_head_dim,
            intermediate_size=self.fast_intermediate_size,
            attention_qkv_bias=self.fast_attention_qkv_bias,
            attention_qk_norm=self.fast_attention_qk_norm,
            attention_o_bias=self.fast_attention_o_bias,
        )


DUAL_AR_DEFAULTS: dict[str, Any] = {
    "model_type": "dual_ar",
    "vocab_size": 32000,
    "n_layer": 32,
    "n_head": 32,
    "dim": 4096,
    "intermediate_size": None,
    "n_local_heads": -1,
    "head_dim": 64,
    "rope_base": 10000,
    "norm_eps": 1e-5,
    "max_seq_len": 2048,
    "dropout": 0.0,
    "tie_word_embeddings": True,
    "attention_qkv_bias": False,
    "attention_o_bias": False,
    "attention_qk_norm": False,
    "codebook_size": 160,
    "num_codebooks": 4,
    "semantic_begin_id": 0,
    "semantic_end_id": 0,
    "use_gradient_checkpointing": True,
    "initializer_range": 0.02,
    "is_reward_model": False,
    "scale_codebook_embeddings": False,
    "audio_embed_dim": None,
    "n_fast_layer": 4,
    "fast_dim": None,
    "fast_n_head": None,
    "fast_n_local_heads": None,
    "fast_head_dim": None,
    "fast_intermediate_size": None,
    "fast_attention_qkv_bias": None,
    "fast_attention_qk_norm": None,
    "fast_attention_o_bias": None,
    "norm_fastlayer_input": False,
}
"""Upstream ``DualARModelArgs``' fields (``BaseModelArgs``' first) and their defaults."""


def dual_ar_args(data: dict[str, Any]) -> dict[str, Any]:
    """``DualARModelArgs(**data)`` as a dict, its ``__post_init__`` applied (unknown keys dropped): what upstream's
    ``config.save`` writes for a model loaded from ``data``."""
    c = {**DUAL_AR_DEFAULTS, **{k: v for k, v in data.items() if k in DUAL_AR_DEFAULTS}}
    if c["n_local_heads"] == -1:
        c["n_local_heads"] = c["n_head"]
    if c["intermediate_size"] is None:
        hidden = int(2 * (4 * c["dim"]) / 3)
        c["intermediate_size"] = hidden if hidden % 256 == 0 else hidden + 256 - hidden % 256
    if c["head_dim"] is None:
        c["head_dim"] = c["dim"] // c["n_head"]
    for name in ("dim", "n_head", "n_local_heads", "head_dim", "intermediate_size"):
        c[f"fast_{name}"] = c[f"fast_{name}"] or c[name]
    for name in ("attention_qkv_bias", "attention_qk_norm", "attention_o_bias"):
        if c[f"fast_{name}"] is None:
            c[f"fast_{name}"] = c[name]
    return c


def omni_to_dual_ar(data: dict[str, Any]) -> dict[str, Any]:
    """``BaseModelArgs._from_fish_qwen3_omni``: an S2 ``config.json`` as the flat ``dual_ar`` arguments."""
    tc, adc = data["text_config"], data["audio_decoder_config"]
    flat = {
        "model_type": "dual_ar",
        "vocab_size": tc["vocab_size"],
        "n_layer": tc["n_layer"],
        "n_head": tc["n_head"],
        "n_local_heads": tc.get("n_local_heads", -1),
        "head_dim": tc.get("head_dim"),
        "dim": tc["dim"],
        "intermediate_size": tc.get("intermediate_size"),
        "rope_base": tc.get("rope_base", 10000),
        "norm_eps": tc.get("norm_eps", 1e-5),
        "max_seq_len": tc.get("max_seq_len", 2048),
        "dropout": tc.get("dropout", 0.0),
        "tie_word_embeddings": tc.get("tie_word_embeddings", True),
        "attention_qkv_bias": tc.get("attention_qkv_bias", False),
        "attention_o_bias": tc.get("attention_o_bias", False),
        "attention_qk_norm": tc.get("attention_qk_norm", False),
        "use_gradient_checkpointing": tc.get("use_gradient_checkpointing", True),
        "initializer_range": tc.get("initializer_range", 0.02),
        "semantic_begin_id": data.get("semantic_start_token_id", 0),
        "semantic_end_id": data.get("semantic_end_token_id", 0),
        "scale_codebook_embeddings": True,
        "norm_fastlayer_input": True,
        "audio_embed_dim": adc.get("text_dim", tc["dim"]),
        "codebook_size": adc["vocab_size"],
        "num_codebooks": adc["num_codebooks"],
        "n_fast_layer": adc["n_layer"],
        "fast_dim": adc.get("dim"),
        "fast_n_head": adc.get("n_head"),
        "fast_n_local_heads": adc.get("n_local_heads"),
        "fast_head_dim": adc.get("head_dim"),
        "fast_intermediate_size": adc.get("intermediate_size"),
        "fast_attention_qkv_bias": adc.get("attention_qkv_bias"),
        "fast_attention_qk_norm": adc.get("attention_qk_norm"),
        "fast_attention_o_bias": adc.get("attention_o_bias"),
    }
    return dual_ar_args({k: v for k, v in flat.items() if v is not None})


def is_s2(data: dict[str, Any]) -> bool:
    """An S2 checkpoint: its own ``config.json``, or the ``dual_ar`` one upstream saves after fine-tuning it (S2's
    arguments scale the codebook embeddings and norm the fast transformer's input; older dual-AR models do neither)."""
    if data.get("model_type") == "fish_qwen3_omni":
        return True
    return data.get("model_type") == "dual_ar" and bool(data.get("scale_codebook_embeddings")) and bool(data.get("norm_fastlayer_input"))


class RMSNorm(nn.Module):
    """Fish's RMSNorm: normalised in float32, cast back, then times the weight."""

    def __init__(self, dim: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.empty(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = x.float()
        out = (h * torch.rsqrt(torch.mean(h * h, dim=-1, keepdim=True) + self.eps)).type_as(x)
        return out * self.weight


def precompute_freqs_cis(seq_len: int, n_elem: int, base: float = 10000) -> torch.Tensor:
    """(seq_len, n_elem/2, 2) cos and sin, in bfloat16 as upstream stores them."""
    freqs = 1.0 / (base ** (torch.arange(0, n_elem, 2)[: (n_elem // 2)].float() / n_elem))
    t = torch.arange(seq_len, device=freqs.device)
    freqs = torch.outer(t, freqs)
    cis = torch.polar(torch.ones_like(freqs), freqs)
    return torch.stack([cis.real, cis.imag], dim=-1).to(dtype=torch.bfloat16)


def apply_rotary_emb(x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
    xs = x.float().reshape(*x.shape[:-1], -1, 2)
    f = freqs_cis.view(1, xs.size(1), 1, xs.size(3), 2)
    out = torch.stack([xs[..., 0] * f[..., 0] - xs[..., 1] * f[..., 1], xs[..., 1] * f[..., 0] + xs[..., 0] * f[..., 1]], -1)
    return out.flatten(3).type_as(x)


class Attention(nn.Module):
    def __init__(self, c: ModelArgs, ops: Any, fast: bool) -> None:
        super().__init__()
        total = (c.n_head + 2 * c.n_local_heads) * c.head_dim
        self.wqkv = ops.Linear(c.dim, total, bias=c.attention_qkv_bias)
        self.wo = ops.Linear(c.n_head * c.head_dim, c.dim, bias=c.attention_o_bias)
        self.qk_norm = c.attention_qk_norm
        if c.attention_qk_norm:
            self.q_norm = ops.RMSNorm(c.head_dim, c.norm_eps)
            self.k_norm = ops.RMSNorm(c.head_dim, c.norm_eps)
        self.n_head, self.head_dim, self.n_local_heads = c.n_head, c.head_dim, c.n_local_heads
        self.fast = fast

    def forward(self, x: torch.Tensor, freqs_cis: torch.Tensor, mask: torch.Tensor, cache: Any, math_kernel: bool) -> torch.Tensor:
        q, k, v = self.project(x, freqs_cis)
        return self.wo(self.attend(q, k, v, mask, cache, math_kernel))

    def project(self, x: torch.Tensor, freqs_cis: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """q, k (normed, rotated) and v, (batch, heads, length, head_dim)."""
        bsz, seqlen, _ = x.shape
        q_size, kv_size = self.n_head * self.head_dim, self.n_local_heads * self.head_dim
        q, k, v = self.wqkv(x).split([q_size, kv_size, kv_size], dim=-1)
        q = q.view(bsz, seqlen, self.n_head, self.head_dim)
        k = k.view(bsz, seqlen, self.n_local_heads, self.head_dim)
        v = v.view(bsz, seqlen, self.n_local_heads, self.head_dim)
        if self.qk_norm:
            q, k = self.q_norm(q), self.k_norm(k)
        q, k = apply_rotary_emb(q, freqs_cis), apply_rotary_emb(k, freqs_cis)
        q, k, v = (t.transpose(1, 2) for t in (q, k, v))
        return q, k, v

    def attend(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: torch.Tensor, cache: Any, math_kernel: bool) -> torch.Tensor:
        """The cache's update and attention: (batch, length, heads × head_dim), before ``wo``."""
        bsz, _, seqlen, _ = q.shape
        k, v = cache.update(k, v, mask.shape[-1])
        rep = self.n_head // self.n_local_heads
        k, v = k.repeat_interleave(rep, dim=1), v.repeat_interleave(rep, dim=1)
        y = attention(q, k, v, site=FAST if self.fast else STEP if math_kernel else SLOW, mask=mask)
        return y.transpose(1, 2).contiguous().view(bsz, seqlen, self.n_head * self.head_dim)


def eq_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Upstream's ``eq_scaled_dot_product_attention`` (the fast transformer's)."""
    length, keys = q.size(-2), k.size(-2)
    bias = torch.zeros(1, 1, length, keys, dtype=q.dtype, device=q.device)
    bias = torch.where(mask.logical_not(), float("-inf"), bias)
    weight = q @ k.transpose(-2, -1) * (1 / math.sqrt(q.size(-1)))
    weight += bias
    return torch.softmax(weight, dim=-1) @ v


SLOW = Site("fishspeech.slow")
"""The slow transformer's prefill and training: SDPA with a boolean mask."""
STEP = Site("fishspeech.step", reference=sdpa_math)
"""Its decoding steps: SDPA's math kernel, as upstream pins it (``sdpa_kernel(MATH)``)."""
FAST = Site("fishspeech.fast", reference=lambda q, k, v, mask, causal, scale, gqa, dropout: eq_attention(q, k, v, mask))
"""The fast transformer: upstream's hand-written softmax over ten positions."""


class FeedForward(nn.Module):
    def __init__(self, c: ModelArgs, ops: Any) -> None:
        super().__init__()
        self.w1 = ops.Linear(c.dim, c.intermediate_size, bias=False)
        self.w3 = ops.Linear(c.dim, c.intermediate_size, bias=False)
        self.w2 = ops.Linear(c.intermediate_size, c.dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class TransformerBlock(nn.Module):
    def __init__(self, c: ModelArgs, ops: Any, fast: bool) -> None:
        super().__init__()
        self.attention = Attention(c, ops, fast)
        self.feed_forward = FeedForward(c, ops)
        self.ffn_norm = RMSNorm(c.dim, c.norm_eps)
        self.attention_norm = RMSNorm(c.dim, c.norm_eps)

    def forward(self, x: torch.Tensor, freqs_cis: torch.Tensor, mask: torch.Tensor, cache: Any, math_kernel: bool) -> torch.Tensor:
        h = x + self.attention(self.attention_norm(x), freqs_cis, mask, cache, math_kernel)
        return h + self.feed_forward(self.ffn_norm(h))

    # ``forward`` in two halves around ``attention.attend`` (the same operations in the same order), for steps run as
    # CUDA graphs (``mizuoto.runtime.graphs.LayerSteps``).
    def attention_in(self, x: torch.Tensor, freqs_cis: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.attention.project(self.attention_norm(x), freqs_cis)

    def attention_out(self, x: torch.Tensor, attended: torch.Tensor) -> torch.Tensor:
        h = x + self.attention.wo(attended)
        return h + self.feed_forward(self.ffn_norm(h))


class GrowingCache:
    """The slow model's per-layer keys and values, grown by concatenation."""

    def __init__(self) -> None:
        self.k: torch.Tensor | None = None
        self.v: torch.Tensor | None = None

    def update(self, k: torch.Tensor, v: torch.Tensor, active: int) -> tuple[torch.Tensor, torch.Tensor]:
        self.k = k if self.k is None else torch.cat([self.k, k], dim=2)
        self.v = v if self.v is None else torch.cat([self.v, v], dim=2)
        return self.k[:, :, :active], self.v[:, :, :active]


class FixedCache:
    """The fast model's cache: ``size`` positions written at their index, read whole (as upstream's ``KVCache``)."""

    def __init__(self, size: int, heads: int, head_dim: int, dtype: torch.dtype, device: torch.device) -> None:
        self.k = torch.zeros((1, heads, size, head_dim), dtype=dtype, device=device)
        self.v = torch.zeros((1, heads, size, head_dim), dtype=dtype, device=device)
        self.pos = 0

    def update(self, k: torch.Tensor, v: torch.Tensor, active: int) -> tuple[torch.Tensor, torch.Tensor]:
        self.k[:, :, self.pos] = k[:, :, 0]
        self.v[:, :, self.pos] = v[:, :, 0]
        return self.k[:, :, :active], self.v[:, :, :active]


@dataclass
class StepResult:
    logits: torch.Tensor
    hidden_states: torch.Tensor


class DualARTransformer(nn.Module):
    """Parameter names are upstream's after ``_remap_fish_qwen3_omni_keys``."""

    def __init__(self, config: ModelArgs, ops: Any = default_ops.default) -> None:
        super().__init__()
        self.config = c = config
        self.embeddings = ops.Embedding(c.vocab_size, c.dim)
        self.codebook_embeddings = ops.Embedding(c.codebook_size * c.num_codebooks, c.dim)
        self.layers = nn.ModuleList(TransformerBlock(c, ops, fast=False) for _ in range(c.n_layer))
        self.norm = RMSNorm(c.dim, c.norm_eps)
        if not c.tie_word_embeddings:
            self.output = ops.Linear(c.dim, c.vocab_size, bias=False)
        fc = c.fast()
        if c.fast_dim != c.dim:
            self.fast_project_in: nn.Module = ops.Linear(c.dim, c.fast_dim)
        else:
            self.fast_project_in = nn.Identity()
        self.fast_embeddings = ops.Embedding(c.codebook_size, c.fast_dim)
        self.fast_layers = nn.ModuleList(TransformerBlock(fc, ops, fast=True) for _ in range(c.n_fast_layer))
        self.fast_norm = RMSNorm(c.fast_dim, c.norm_eps)
        self.fast_output = ops.Linear(c.fast_dim, c.codebook_size, bias=False)
        self.register_buffer("freqs_cis", precompute_freqs_cis(c.max_seq_len, c.head_dim, c.rope_base), persistent=False)
        self.register_buffer("fast_freqs_cis", precompute_freqs_cis(c.num_codebooks, c.fast_head_dim, c.rope_base), persistent=False)

    def new_caches(self) -> list[GrowingCache]:
        return [GrowingCache() for _ in self.layers]

    def new_fast_caches(self) -> list[FixedCache]:
        fc = self.config.fast()
        p = self.fast_output.weight
        return [FixedCache(self.config.num_codebooks, fc.n_local_heads, fc.head_dim, p.dtype, p.device) for _ in self.fast_layers]

    def forward_generate(self, inp: torch.Tensor, input_pos: torch.Tensor, kv_len: int, caches: list[GrowingCache], math_kernel: bool, last: bool = False) -> StepResult:
        """``last``: ``inp`` is one frame at position ``kv_len - 1`` (a decoding step), whose row of the causal mask
        is all true: it is made as such, not cut from a ``kv_len`` square."""
        x = self.input_embeddings(inp)
        mask = self.step_mask(kv_len, x.device) if last else torch.ones(kv_len, kv_len, dtype=torch.bool, device=x.device).tril()[None, None, input_pos, :kv_len]
        freqs_cis = self.get_buffer("freqs_cis")[input_pos]
        for layer, cache in zip(self.layers, caches, strict=True):
            x = layer(x, freqs_cis, mask, cache, math_kernel)
        if x.size(1) > 1:
            x = x[:, -1:]
        return self.output_step(x)

    @staticmethod
    def step_mask(kv_len: int, device: torch.device) -> torch.Tensor:
        """A decoding step's mask: the last row of the causal square, (1, 1, 1, kv_len), all true."""
        return torch.ones(1, 1, 1, kv_len, dtype=torch.bool, device=device)

    def input_embeddings(self, inp: torch.Tensor) -> torch.Tensor:
        """The frames' embeddings: the main token's plus, for semantic tokens, the codebooks' (``where`` rather than
        upstream's masked assignment, which reads the mask back from the device; the same values)."""
        c = self.config
        embeds = [self.codebook_embeddings(inp[:, i + 1] + i * c.codebook_size) for i in range(c.num_codebooks)]
        vq_sum = torch.stack(embeds, dim=1).sum(dim=1)
        vq_masks = (inp[:, 0] >= c.semantic_begin_id) & (inp[:, 0] <= c.semantic_end_id)
        vq_sum = torch.where(vq_masks.unsqueeze(-1), vq_sum, 0)
        x = self.embeddings(inp[:, 0]) + vq_sum
        if c.scale_codebook_embeddings:
            x = torch.where(vq_masks.unsqueeze(-1).expand_as(x), x / math.sqrt(c.num_codebooks + 1), x)
        return x

    def output_step(self, x: torch.Tensor) -> StepResult:
        """The logits and the fast transformer's input from the last layer's output (one position)."""
        c = self.config
        slow_out = self.norm(x)
        logits = F.linear(slow_out, self.embeddings.weight) if c.tie_word_embeddings else self.output(slow_out)
        hidden = slow_out if c.norm_fastlayer_input else x
        return StepResult(logits, self.fast_project_in(hidden))

    def forward_generate_fast(self, x: torch.Tensor, pos: int, caches: list[FixedCache]) -> torch.Tensor:
        n = self.config.num_codebooks
        x = x.view(x.shape[0], 1, -1)
        # slices, not upstream's list indices (each a copy to the device that waits for it): the same values
        mask = torch.ones(n, n, dtype=torch.bool, device=x.device).tril()[None, None, pos : pos + 1, :n]
        freqs_cis = self.get_buffer("fast_freqs_cis")[pos : pos + 1]
        for layer, cache in zip(self.fast_layers, caches, strict=True):
            cache.pos = pos
            x = layer(x, freqs_cis, mask, cache, False)
        return self.fast_output(self.fast_norm(x))


def remap_keys(weights: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """``_remap_fish_qwen3_omni_keys``: ``text_model.model.*`` and ``audio_decoder.*`` to the module's names."""
    out: dict[str, torch.Tensor] = {}
    for k, v in weights.items():
        if k.startswith("text_model.model."):
            out[k[len("text_model.model.") :]] = v
        elif k.startswith("audio_decoder."):
            suffix = k[len("audio_decoder.") :]
            out[suffix if suffix.startswith("codebook_embeddings.") else "fast_" + suffix] = v
        else:
            out[k] = v
    return out
