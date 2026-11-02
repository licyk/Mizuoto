"""The codebook tower (upstream ``fish_omni/tower.py``): from the backbone's hidden state and the frame's main token it
predicts the frame's codes 1–9, one residual layer per code, each layer attending over the frames so far.

A frame is a row ``[main token, code 0, …, code 9]``; code 0 is the main token's place in the semantic band. Four
shared layers first mix in the main token's embedding (the first also the previous frame's codes the backbone did not
see); each residual layer then mixes in the code before it and its head draws the next.

Sampling is upstream's, draw for draw: every draw is ``argmax(p / E)`` with ``E ~ Exp(1)`` (``exponential_``) from the
request's generator, in upstream's order. The main token is drawn twice, at the request's settings and at the
repetition-aware ones, the second used when the first repeats one of the last 16 main tokens (default sampler: only an
audio token is redrawn). Fish Audio's reference sampler (``reference_argmax`` / ``reference_ras``) adds an epsilon
cutoff for the main token and draws the codes only from the codec's 1024 residual codes.
"""

from __future__ import annotations

import math
from typing import Any, Literal

import torch
from torch import nn
from torch.nn import functional as F

from mizuoto.runtime import ops as default_ops
from mizuoto.runtime.accel import Site
from mizuoto_fishspeech.drama3.config import Drama3Config, TowerConfig
from mizuoto_fishspeech.drama3.layers import ZeroCenteredRMSNorm, grouped_attention

RAS_WINDOW = 16
RAS_TEMPERATURE = 1.0
RAS_TOP_K = 30
RAS_TOP_P = 0.95
MIN_TEMPERATURE = 1e-5
REFERENCE_EPSILON_CUTOFF = 1e-3
"""The main token's epsilon cutoff under the reference sampler when a request gives none."""
RESIDUAL_CODEBOOK_SIZE = 1024
"""Codes in each of the codec's residual codebooks (the heads are as wide as the semantic codebook's 4096)."""
Fallback = Literal["argmax", "ras"]
TOWER = Site("fishspeech.drama3.tower", approx_ok=False)  # autoregressive
"""The tower's attention over the frames: SDPA with a boolean mask (``grouped_attention``)."""
CHECKPOINT_PREFIXES = {
    "audio_decoder.codebook_embeddings.": "codebook_embeddings.",
    "audio_decoder_mtp_shared.layers.": "shared_layers.",
    "audio_decoder_mtp.layers.": "residual_layers.",
    "audio_decoder_mtp.heads.": "codebook_heads.",
}


def sample_top_k_top_p(logits: torch.Tensor, temperature: float, top_k: int, top_p: float, generator: torch.Generator) -> torch.Tensor:
    """One id per row; drops the token crossing ``top_p`` (and beyond ``top_k``) but always keeps the first."""
    temperature = max(temperature, MIN_TEMPERATURE)
    values, ids = torch.sort(logits.float() / temperature, descending=True, dim=-1)
    ranks = torch.arange(values.shape[-1], device=values.device)
    dropped = (values.softmax(-1).cumsum(-1) > top_p) | (ranks >= top_k)
    dropped[..., 0].fill_(False)  # a fill on the device (a decoding step's CUDA graphs hold it)
    probabilities = values.masked_fill(dropped, -torch.inf).softmax(-1)
    noise = torch.empty_like(probabilities).exponential_(generator=generator)
    choice = (probabilities / noise).argmax(-1, keepdim=True)
    return ids.gather(-1, choice).squeeze(-1)


def reference_sample(
    logits: torch.Tensor, temperature: float, top_k: int, top_p: float, epsilon_cutoff: float, generator: torch.Generator, legal_column_count: int | None = None
) -> torch.Tensor:
    """One id per row under Fish Audio's reference sampler: rank r stays when r < top_k and (r = 0, or cum[r] ≤ top_p and
    p[r] ≥ epsilon_cutoff); top_k < 1, top_p ≥ 1 and epsilon_cutoff ≤ 0 turn their rule off. ``legal_column_count`` then
    drops later ids, keeping the most probable legal one."""
    temperature = max(temperature, MIN_TEMPERATURE)
    values, ids = torch.sort(logits.float() / temperature, descending=True, stable=True, dim=-1)
    probabilities = values.softmax(-1)
    ranks = torch.arange(values.shape[-1], device=values.device)
    kept = ((ranks < top_k) | (top_k < 1)).expand(values.shape).clone()
    kept &= (probabilities.cumsum(-1) <= top_p) | (top_p >= 1.0)
    kept &= probabilities >= epsilon_cutoff
    if legal_column_count is None:
        kept[..., 0].fill_(True)
    else:
        legal = ids < legal_column_count
        kept &= legal
        kept.scatter_(-1, legal.int().argmax(-1, keepdim=True), True)
    probabilities = values.masked_fill(~kept, -torch.inf).softmax(-1)
    noise = torch.empty_like(probabilities).exponential_(generator=generator)
    choice = (probabilities / noise).argmax(-1, keepdim=True)
    return ids.gather(-1, choice).squeeze(-1)


class TowerState:
    """One request's keys and values of every tower layer: [1, KVH, capacity, D] each."""

    def __init__(self, layers: int, capacity: int, kv_heads: int, head_dim: int, dtype: torch.dtype, device: torch.device) -> None:
        self.k = [torch.zeros((1, kv_heads, capacity, head_dim), dtype=dtype, device=device) for _ in range(layers)]
        self.v = [torch.zeros((1, kv_heads, capacity, head_dim), dtype=dtype, device=device) for _ in range(layers)]


class TowerAttention(nn.Module):
    """Gated attention over time; the rows of ``wqkv`` are [Q | gate | K | V]."""

    def __init__(self, c: TowerConfig, ops: Any = default_ops.default) -> None:
        super().__init__()
        self.config = c
        q_width, kv_width = c.n_head * c.head_dim, c.n_local_heads * c.head_dim
        self.wqkv = ops.Linear(c.dim, 2 * q_width + 2 * kv_width, bias=False)
        self.wo = ops.Linear(q_width, c.dim, bias=False)
        # always on, whatever audio_decoder_config.attention_qk_norm says: training norms q and k here, but the exporter
        # never writes the field, so released configs carry its default false (upstream's note); the weights are
        # zero-centred, not shifted by one
        self.q_norm = ZeroCenteredRMSNorm(c.head_dim, c.norm_eps)
        self.k_norm = ZeroCenteredRMSNorm(c.head_dim, c.norm_eps)

    def rotate(self, x: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        """Interleaved (pairwise) partial RoPE, in float32."""
        width = self.config.rotary_dim
        exponents = torch.arange(0, width, 2, device=x.device, dtype=torch.float32)
        angles = positions.float()[:, None] * self.config.rope_base ** (-exponents / width)
        cos, sin = angles.cos()[:, None], angles.sin()[:, None]
        even, odd = x[..., :width:2].float(), x[..., 1:width:2].float()
        rotated = torch.stack((even * cos - odd * sin, odd * cos + even * sin), -1)
        return torch.cat((rotated.flatten(-2).to(x.dtype), x[..., width:]), dim=-1)

    def forward(self, x: torch.Tensor, positions: torch.Tensor, k_cache: torch.Tensor, v_cache: torch.Tensor) -> torch.Tensor:
        q, k, v, gate = self.attention_in(x, positions)
        return self.attention_out(self.attend(q, k, v, positions, int(positions[-1]) + 1, k_cache, v_cache), gate)

    # the forward in three parts: a decoding step's CUDA graphs run the first and the last, the attention between them
    # eagerly (its keys grow by one each step)

    def attention_in(self, x: torch.Tensor, positions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Query, key and value [1, heads, T, D] (normed, rotated) and the output gate [T, heads · D]."""
        c = self.config
        length = len(positions)
        q_width, kv_width = c.n_head * c.head_dim, c.n_local_heads * c.head_dim
        q, gate, k, v = self.wqkv(x).split((q_width, q_width, kv_width, kv_width), -1)
        q = self.q_norm(q.reshape(length, c.n_head, c.head_dim))
        k = self.k_norm(k.reshape(length, c.n_local_heads, c.head_dim))
        q = self.rotate(q, positions).transpose(0, 1).unsqueeze(0)
        k = self.rotate(k, positions).transpose(0, 1).unsqueeze(0)
        v = v.reshape(length, c.n_local_heads, c.head_dim).transpose(0, 1).unsqueeze(0)
        return q, k, v, gate

    def attend(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, positions: torch.Tensor, end: int, k_cache: torch.Tensor, v_cache: torch.Tensor) -> torch.Tensor:
        """Cache the rows' keys and values at ``positions``, then attend over the first ``end`` (the last position + 1):
        [T, heads · D]."""
        c = self.config
        k_cache.index_copy_(2, positions, k)
        v_cache.index_copy_(2, positions, v)
        k, v = k_cache[:, :, :end], v_cache[:, :, :end]
        key_positions = torch.arange(end, device=q.device)
        mask = key_positions[None, :] <= positions[:, None]
        out = grouped_attention(q, k, v, mask, TOWER)
        return out.squeeze(0).transpose(0, 1).reshape(len(positions), c.n_head * c.head_dim)

    def attention_out(self, attended: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
        return self.wo(attended * gate.sigmoid())


class FP32OutputLinear(nn.Linear):
    """A float16 matmul with a float32 output (the down projections exceed float16's range); other dtypes as ``Linear``."""

    reset_parameters = default_ops.default.Linear.reset_parameters

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if input.dtype != torch.float16:
            return super().forward(input)
        if input.is_cuda:
            out = torch.mm(input.reshape(-1, self.in_features), self.weight.T, out_dtype=torch.float32)
            out = out.reshape(*input.shape[:-1], self.out_features)
        else:
            out = F.linear(input.float(), self.weight.float())
        return out if self.bias is None else out + self.bias.float()


class TowerMLP(nn.Module):
    def __init__(self, c: TowerConfig, ops: Any = default_ops.default) -> None:
        super().__init__()
        self.w1 = ops.Linear(c.dim, c.intermediate_size, bias=False)
        self.w3 = ops.Linear(c.dim, c.intermediate_size, bias=False)
        self.w2 = FP32OutputLinear(c.intermediate_size, c.dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class TowerTransformerLayer(nn.Module):
    def __init__(self, c: TowerConfig, ops: Any = default_ops.default) -> None:
        super().__init__()
        self.attention = TowerAttention(c, ops)
        self.attention_norm = ZeroCenteredRMSNorm(c.dim, c.norm_eps)
        self.ffn_norm = ZeroCenteredRMSNorm(c.dim, c.norm_eps)
        self.feed_forward = TowerMLP(c, ops)

    def forward(self, x: torch.Tensor, positions: torch.Tensor, k_cache: torch.Tensor, v_cache: torch.Tensor) -> torch.Tensor:
        dtype = self.attention.wqkv.weight.dtype
        residual = x.float() if dtype == torch.float16 else x
        residual = residual + self.attention(self.attention_norm(residual).to(dtype), positions, k_cache, v_cache)
        return residual + self.feed_forward(self.ffn_norm(residual).to(dtype))

    def attention_in(self, x: torch.Tensor, positions: torch.Tensor) -> tuple[torch.Tensor, tuple[torch.Tensor, ...]]:
        """The residual stream and the attention's ``attention_in``."""
        dtype = self.attention.wqkv.weight.dtype
        residual = x.float() if dtype == torch.float16 else x
        return residual, self.attention.attention_in(self.attention_norm(residual).to(dtype), positions)

    def attention_out(self, residual: torch.Tensor, attended: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
        dtype = self.attention.wqkv.weight.dtype
        residual = residual + self.attention.attention_out(attended, gate)
        return residual + self.feed_forward(self.ffn_norm(residual).to(dtype))


class TowerLayer(nn.Module):
    """Mix a code's embedding into the hidden state, then one pre-norm block."""

    def __init__(self, c: TowerConfig, shared: bool, ops: Any = default_ops.default) -> None:
        super().__init__()
        self.shared = shared
        # the checkpoint names the embedding differently in shared and residual layers
        self.add_module("semantic_embedding" if shared else "codebook_embedding", ops.Embedding(c.vocab_size, c.dim))
        self.enorm = ZeroCenteredRMSNorm(c.dim, c.norm_eps)
        self.hnorm = ZeroCenteredRMSNorm(c.dim, c.norm_eps)
        self.eh_proj = ops.Linear(2 * c.dim, c.dim, bias=False)
        self.transformer_layer = TowerTransformerLayer(c, ops)
        self.final_layernorm = ZeroCenteredRMSNorm(c.dim, c.norm_eps)

    def forward(
        self,
        hidden: torch.Tensor,
        codes: torch.Tensor,
        semantic: torch.Tensor,
        positions: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        shared_input: torch.Tensor | None = None,
    ) -> torch.Tensor:
        mixed = self.mix(hidden, codes, semantic, shared_input)
        return self.finish(mixed, self.transformer_layer(mixed, positions, k_cache, v_cache))

    def mix(self, hidden: torch.Tensor, codes: torch.Tensor, semantic: torch.Tensor, shared_input: torch.Tensor | None = None) -> torch.Tensor:
        """The block's input: the codes' embeddings (zero where the frame is not audio) mixed into the hidden state."""
        embedding: Any = self.semantic_embedding if self.shared else self.codebook_embedding
        codes = codes.clamp(0, embedding.num_embeddings - 1)
        e = torch.where(semantic[:, None], embedding(codes), 0.0)
        if shared_input is not None:
            e = e + shared_input
        return self.eh_proj(torch.cat((self.enorm(e), self.hnorm(hidden)), dim=-1))

    def finish(self, mixed: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
        """The layer's output from its block's ``out``."""
        return self.final_layernorm(out).to(mixed.dtype)


class CodebookTower(nn.Module):
    """Predicts codes 1..N-1 of each frame, attending over time (upstream ``FishOmniAudioDecoder``)."""

    def __init__(self, config: Drama3Config, ops: Any = default_ops.default) -> None:
        super().__init__()
        c = self.config = config.tower
        self.semantic_begin = config.semantic_start_token_id
        self.semantic_end = config.semantic_end_token_id
        self.eos_id = config.eos_token_id
        self.codebook_embeddings = ops.Embedding(c.vocab_size * c.num_codebooks, c.dim)
        self.shared_layers = nn.ModuleList(TowerLayer(c, True, ops) for _ in range(c.shared_enrichment_layers))
        self.residual_layers = nn.ModuleList(TowerLayer(c, False, ops) for _ in range(c.num_codebooks - 1))
        self.codebook_heads = nn.ModuleList(ops.Linear(c.dim, c.vocab_size, bias=False) for _ in range(c.num_codebooks - 1))

    @property
    def layers(self) -> list[Any]:
        return [*self.shared_layers, *self.residual_layers]

    def new_state(self, capacity: int, device: torch.device) -> TowerState:
        c = self.config
        return TowerState(len(self.layers), capacity, c.n_local_heads, c.head_dim, self.codebook_embeddings.weight.dtype, device)

    def is_semantic(self, tokens: torch.Tensor) -> torch.Tensor:
        return (tokens >= self.semantic_begin) & (tokens <= self.semantic_end)

    def embedding_sum(self, frames: torch.Tensor, first: int, end: int) -> torch.Tensor:
        """The summed embeddings of codes ``first``..``end - 1`` of each frame."""
        weight = self.codebook_embeddings.weight
        if first == end:
            return weight.new_zeros((len(frames), self.config.dim))
        offsets = torch.arange(first, end, device=frames.device) * self.config.vocab_size
        rows = frames[:, 1 + first : 1 + end] + offsets
        return F.embedding_bag(rows, weight, mode="sum")

    def embed_backbone(self, frames: torch.Tensor, text_embeddings: torch.Tensor) -> torch.Tensor:
        """The backbone's input: a semantic frame adds its codes 1..K to the token's embedding, scaled by 1/√(1+K)."""
        count = self.config.vq_backbone_codebooks
        extra = self.embedding_sum(frames, 1, 1 + count).to(text_embeddings.device)
        mixed = (text_embeddings + extra) / math.sqrt(1 + count)
        semantic = self.is_semantic(frames[:, 0]).to(text_embeddings.device)
        return torch.where(semantic[:, None], mixed, text_embeddings)

    def unseen(self, frames: torch.Tensor) -> torch.Tensor:
        """The summed embeddings of each frame's codes the backbone did not see (zero where it is not audio)."""
        unseen = self.embedding_sum(frames, 1 + self.config.vq_backbone_codebooks, self.config.num_codebooks)
        return torch.where(self.is_semantic(frames[:, 0])[:, None], unseen, 0.0)

    def run_shared(self, hidden: torch.Tensor, c0: torch.Tensor, semantic: torch.Tensor, positions: torch.Tensor, frames: torch.Tensor, state: TowerState) -> torch.Tensor:
        # the first shared layer also sees the codes the backbone did not
        unseen = self.unseen(frames)
        for index, layer in enumerate(self.shared_layers):
            hidden = layer(hidden, c0, semantic, positions, state.k[index], state.v[index], unseen if index == 0 else None)
        return hidden

    def prefill(self, hidden: torch.Tensor, positions: torch.Tensor, frames: torch.Tensor, next_frames: torch.Tensor, state: TowerState) -> None:
        """Teacher-force the caches: row p is labelled with frame p + 1."""
        semantic = self.is_semantic(next_frames[:, 0])
        hidden = self.run_shared(hidden, next_frames[:, 1], semantic, positions, frames, state)
        shared = len(self.shared_layers)
        for index, layer in enumerate(self.residual_layers):
            hidden = layer(hidden, next_frames[:, index + 1], semantic, positions, state.k[shared + index], state.v[shared + index])

    def main_candidates(self, device: torch.device) -> torch.Tensor:
        """The ids the main token is drawn from: eos and the semantic band, in order."""
        eos = torch.tensor([self.eos_id], device=device)
        band = torch.arange(self.semantic_begin, self.semantic_end + 1, device=device)
        return torch.cat((eos, band)).sort().values

    def decode_step(
        self,
        logits: torch.Tensor,
        hidden: torch.Tensor,
        positions: torch.Tensor,
        frames: torch.Tensor,
        history: torch.Tensor,
        state: TowerState,
        sampling: Any,
        generator: torch.Generator,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """The frame after the row (``[main, codes…]``), and whether every logits drawn from were finite (upstream raises
        at the first that is not; the caller checks this once per frame, where it waits for the device anyway)."""
        main, finite = self.draw_main(logits, history[-RAS_WINDOW:], self.main_candidates(logits.device), sampling, generator)
        c0, semantic = self.first_code(main)
        hidden = self.run_shared(hidden, c0, semantic, positions, frames, state)
        codes = [c0]
        shared = len(self.shared_layers)
        for index, (layer, head) in enumerate(zip(self.residual_layers, self.codebook_heads, strict=True)):
            hidden = layer(hidden, codes[-1], semantic, positions, state.k[shared + index], state.v[shared + index])
            code, finite = self.draw_code(head(hidden), finite, sampling, generator)
            codes.append(code)
        return torch.stack((main, *codes), dim=-1), finite

    def draw_main(self, logits: torch.Tensor, window: torch.Tensor, candidates: torch.Tensor, sampling: Any, generator: torch.Generator) -> tuple[torch.Tensor, torch.Tensor]:
        """The frame's main token from the backbone's logits, and whether the logits drawn from were finite. ``window``:
        the main tokens of the last ``RAS_WINDOW`` rows (fewer early on, or padded with ids no token has, like -1)."""
        candidate_logits = logits[:, candidates]
        finite = torch.isfinite(candidate_logits).all()
        fallback: Fallback | None = sampling.fallback
        if fallback is None:
            first = candidates[sample_top_k_top_p(candidate_logits, sampling.temperature, sampling.top_k, sampling.top_p, generator)]
            retry = candidates[sample_top_k_top_p(candidate_logits, RAS_TEMPERATURE, RAS_TOP_K, RAS_TOP_P, generator)]
            # only audio tokens are redrawn, as in S2: the window starts with prompt rows, which hold the user turn's eos
            repeated = (window == first[:, None]).any(-1)
            repeated &= self.is_semantic(first)
        else:
            first = candidates[reference_sample(candidate_logits, sampling.temperature, sampling.top_k, sampling.top_p, sampling.epsilon_cutoff, generator)]
            # neither fallback applies the epsilon cutoff
            if fallback == "argmax":
                retry = candidates[candidate_logits.float().argmax(-1)]
            else:
                retry = candidates[reference_sample(candidate_logits, RAS_TEMPERATURE, RAS_TOP_K, RAS_TOP_P, 0.0, generator)]
            repeated = (window == first[:, None]).any(-1)  # any repeated first draw is replaced, eos included
        return torch.where(repeated, retry, first), finite

    def first_code(self, main: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Code 0 (the main token's place in the semantic band) and whether the frame is audio."""
        return (main - self.semantic_begin).clamp(0, self.config.vocab_size - 1), self.is_semantic(main)

    def draw_code(self, head_logits: torch.Tensor, finite: torch.Tensor, sampling: Any, generator: torch.Generator) -> tuple[torch.Tensor, torch.Tensor]:
        """The next code from a residual layer's head, and ``finite`` and-ed with whether its logits were."""
        finite = finite & torch.isfinite(head_logits).all()
        if sampling.fallback is None:
            return sample_top_k_top_p(head_logits, sampling.temperature, sampling.top_k, sampling.top_p, generator), finite
        # no epsilon cutoff, and only codes the codec's residual codebooks hold
        return reference_sample(head_logits, sampling.temperature, sampling.top_k, sampling.top_p, 0.0, generator, legal_column_count=RESIDUAL_CODEBOOK_SIZE), finite


@torch.no_grad()
def load_tower(model: CodebookTower, weights: Any) -> None:
    """Copy the ``audio_decoder*`` tensors into ``model``, each parameter exactly once (values that the model's dtype
    cannot hold are refused, as upstream refuses them)."""
    targets = dict(model.named_parameters())
    loaded: set[str] = set()
    for key, tensor in weights:
        name = next((new + key[len(old) :] for old, new in CHECKPOINT_PREFIXES.items() if key.startswith(old)), None)
        if name is None or name not in targets:
            raise ValueError(f"unknown tower weight {key}")
        if name in loaded:
            raise ValueError(f"tower weight filled twice: {key}")
        target = targets[name]
        if tuple(target.shape) != tuple(tensor.shape):
            raise ValueError(f"{key}: shape {tuple(tensor.shape)} != {tuple(target.shape)}")
        value = tensor.to(dtype=target.dtype)
        if not bool(torch.isfinite(value).all()):
            raise FloatingPointError(f"{key} cannot be represented in {target.dtype}")
        target.copy_(value)
        loaded.add(name)
    missing = targets.keys() - loaded
    if missing:
        raise ValueError(f"tower weights not loaded: {sorted(missing)[:8]}")
