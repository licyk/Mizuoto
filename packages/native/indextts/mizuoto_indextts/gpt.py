"""IndexTTS's UnifiedVoice GPT (index-tts ``gpt/model_v2.py``; GPT-2 from transformers 4.52, beam sampling from the
generation utilities index-tts vendors).

GPT-2 is transformers' as IndexTTS runs it: ``Conv1D`` projections (``addmm``), the tanh GELU, SDPA attention —
causal without a mask on the prompt (the mask is all ones, so transformers drops it), unmasked on each new token
— and no position embeddings of its own (``wpe`` returns zeros; the mel and text tables are added outside).

The prompt is ``[speaker latents + emotion vector][speed embeddings][text]`` then the start token; generation is
transformers' beam sampling (``num_beams`` beams from one row, repeated): per step the log-softmax of the last
logits goes through repetition penalty (over every id so far, the prompt's placeholder ids included),
temperature, top-k and top-p (keeping two), the beams' scores are added, and ``2 × num_beams`` candidates are
drawn without replacement (``torch.multinomial``, from the request's generator) and sorted; ``BeamSearchScorer``
keeps the best beams and finished hypotheses.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from mizuoto.runtime import ops as default_ops
from mizuoto.runtime.accel import Site, attention
from mizuoto.runtime.sampling import multinomial_draws, repetition_penalty, top_k, top_p
from mizuoto_indextts.conditioning import ConformerEncoder, PerceiverResampler

SITE = Site("indextts.gpt", approx_ok=False)  # autoregressive: quantised attention derails its sampling

# -- GPT-2 ----------------------------------------------------------------------------------------------


class Conv1D(nn.Module):
    """transformers' ``Conv1D``: ``x @ weight + bias`` with the weight stored (in, out)."""

    STREAMABLE = True
    """Its weights are read in ``forward`` only: they may stream (``mizuoto.runtime.units``)."""

    def __init__(self, nf: int, nx: int) -> None:
        super().__init__()
        self.nf = nf
        self.weight = nn.Parameter(torch.empty(nx, nf))
        self.bias = nn.Parameter(torch.empty(nf))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        size_out = x.size()[:-1] + (self.nf,)
        return torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight).view(size_out)


def gelu_new(x: torch.Tensor) -> torch.Tensor:
    """transformers' ``NewGELUActivation``, ``0.5 · x · (1 + tanh(√(2/π) · (x + 0.044715 · x³)))``: the same operations
    in the same order, in place on one buffer (a product or sum with its operands swapped is the same number)."""
    inner = torch.pow(x, 3.0)
    inner.mul_(0.044715).add_(x).mul_(math.sqrt(2.0 / math.pi)).tanh_().add_(1.0)
    return (0.5 * x).mul_(inner)


class GPT2Attention(nn.Module):
    def __init__(self, dim: int, heads: int) -> None:
        super().__init__()
        self.c_attn = Conv1D(3 * dim, dim)
        self.c_proj = Conv1D(dim, dim)
        self.dim, self.heads, self.head_dim = dim, heads, dim // heads

    def forward(self, x: torch.Tensor, cache: list[torch.Tensor] | None) -> tuple[torch.Tensor, list[torch.Tensor]]:
        out, kv = self.attend(*self.project(x), cache)
        return self.c_proj(out), kv

    def project(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        q, k, v = self.c_attn(x).split(self.dim, dim=2)
        q = q.view(*q.shape[:-1], -1, self.head_dim).transpose(1, 2)
        k = k.view(*k.shape[:-1], -1, self.head_dim).transpose(1, 2)
        v = v.view(*v.shape[:-1], -1, self.head_dim).transpose(1, 2)
        return q, k, v

    def attend(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, cache: list[torch.Tensor] | None) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """The cache and attention, before ``c_proj``: (rows, length, dim), and the keys and values so far."""
        if isinstance(cache, _Grown):  # a BeamCaches layer: the last position is this step's
            cache[0][:, :, -1:].copy_(k)
            cache[1][:, :, -1:].copy_(v)
            k, v = cache
        elif cache is not None:
            k, v = torch.cat([cache[0], k], dim=-2), torch.cat([cache[1], v], dim=-2)
        is_causal = q.shape[-2] > 1
        out = attention(q.contiguous(), k.contiguous(), v.contiguous(), site=SITE, causal=is_causal)
        out = out.transpose(1, 2).contiguous()
        return out.reshape(*out.shape[:-2], -1).contiguous(), [k, v]


class _Grown(list):
    """One layer's ``[keys, values]`` from ``BeamCaches``, one position longer than the step's input: ``attend`` writes
    the new key and value there instead of concatenating."""


class BeamCaches(list):
    """The beams' keys and values for every layer, ``(layers, 2, beams, heads, length, head_dim)`` in one buffer.

    ``advance`` grows them by a position with the beams' rows reordered: one copy of the whole cache (one per beam, or
    one when no beam moved) instead of an ``index_select`` and a ``torch.cat`` for each layer's keys and values, each
    copying the whole tensor. Two buffers with room to grow take turns, so most steps allocate nothing. Each layer's
    keys and values stay contiguous, as ``torch.cat`` made them, and hold the same values: attention computes the same.
    """

    ROOM = 128
    """Positions a buffer grows by when the cache outgrows it (up to ``limit``)."""

    def __init__(self, caches: list[list[torch.Tensor]], limit: int) -> None:
        self.buffer = torch.stack([torch.stack(kv) for kv in caches])
        self.limit = limit
        """The longest the cache gets: the decoding's ``max_length``."""
        self._pool, self._spare = self.buffer.view(-1), self.buffer.new_empty(0)
        super().__init__([list(layer.unbind(0)) for layer in self.buffer.unbind(0)])

    def advance(self, kept: list[int]) -> None:
        """Make room for the next step's keys and values, beam ``i`` continuing beam ``kept[i]``."""
        old, length = self.buffer, self.buffer.shape[4]
        shape = (*old.shape[:4], length + 1, old.shape[5])
        need = math.prod(shape)
        if self._spare.numel() < need:
            room = max(min(self.ROOM, self.limit - length - 1), 0)
            self._spare = old.new_empty(need + math.prod(shape[:4]) * room * shape[5])
        new = self._spare[:need].view(shape)
        if kept == list(range(len(kept))):
            new[:, :, :, :, :length].copy_(old)
        else:
            for i, j in enumerate(kept):
                new[:, :, i, :, :length].copy_(old[:, :, j])
        self._pool, self._spare = self._spare, self._pool
        self.buffer = new
        self[:] = [_Grown(layer.unbind(0)) for layer in new.unbind(0)]


class GPT2MLP(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.c_fc = Conv1D(4 * dim, dim)
        self.c_proj = Conv1D(dim, 4 * dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.c_proj(gelu_new(self.c_fc(x)))


class GPT2Block(nn.Module):
    def __init__(self, dim: int, heads: int, ops: Any) -> None:
        super().__init__()
        self.ln_1 = ops.LayerNorm(dim, eps=1e-5)
        self.attn = GPT2Attention(dim, heads)
        self.ln_2 = ops.LayerNorm(dim, eps=1e-5)
        self.mlp = GPT2MLP(dim)

    def forward(self, x: torch.Tensor, cache: list[torch.Tensor] | None) -> tuple[torch.Tensor, list[torch.Tensor]]:
        attn, kv = self.attn(self.ln_1(x), cache)
        x = attn + x
        return x + self.mlp(self.ln_2(x)), kv

    # ``forward`` in two halves around ``attn.attend`` (the same operations in the same order), for steps run as CUDA
    # graphs (``mizuoto.runtime.graphs.LayerSteps``).
    def attention_in(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.attn.project(self.ln_1(x))

    def attention_out(self, x: torch.Tensor, attended: torch.Tensor) -> torch.Tensor:
        x = self.attn.c_proj(attended) + x
        return x + self.mlp(self.ln_2(x))


def wpe_zeros(embeds: torch.Tensor) -> torch.Tensor:
    """Upstream's position embeddings (``null_position_embeddings``): zeros, which transformers' GPT-2 adds to the input
    embeddings. They are made in torch's default dtype (float32), so a step's fp16 or bf16 embeddings come out float32
    and the residual stream runs in float32 from there (the prompt already is); adding them also turns -0.0 into 0.0."""
    return torch.zeros(embeds.shape, device=embeds.device)


class GPT2Model(nn.Module):
    def __init__(self, dim: int, heads: int, layers: int, ops: Any) -> None:
        super().__init__()
        self.h = nn.ModuleList(GPT2Block(dim, heads, ops) for _ in range(layers))
        self.ln_f = ops.LayerNorm(dim, eps=1e-5)

    def forward(self, embeds: torch.Tensor, caches: list[list[torch.Tensor]] | None = None) -> tuple[torch.Tensor, list[list[torch.Tensor]]]:
        x = embeds + wpe_zeros(embeds)
        new = []
        for i, block in enumerate(self.h):
            x, kv = block(x, caches[i] if caches is not None else None)
            new.append(kv)
        return self.ln_f(x), new


class LearnedPositionEmbeddings(nn.Module):
    def __init__(self, seq_len: int, dim: int, ops: Any) -> None:
        super().__init__()
        self.emb = ops.Embedding(seq_len, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.emb(torch.arange(0, x.shape[1], device=x.device))

    def fixed(self, index: int, device: torch.device) -> torch.Tensor:
        """Position ``index``'s embedding, (1, 1, dim): the table's row (upstream looks it up with
        ``torch.tensor([index], device=device)``, a copy to the device that waits for it; the same values)."""
        return self.emb.weight[index : index + 1].unsqueeze(0)


# -- UnifiedVoice -----------------------------------------------------------------------------------------


@dataclass
class GptConfig:
    model_dim: int = 1280
    layers: int = 24
    heads: int = 20
    max_mel_tokens: int = 1815
    max_text_tokens: int = 600
    number_text_tokens: int = 12000
    number_mel_codes: int = 8194
    start_mel_token: int = 8192
    stop_mel_token: int = 8193
    start_text_token: int = 0
    stop_text_token: int = 1
    condition_num_latent: int = 32
    condition_module: dict[str, Any] | None = None
    emo_condition_module: dict[str, Any] | None = None
    spk_cond_mode: str = "conformer"
    """``conformer`` (IndexTTS-2: w2v-BERT features through a Conformer and a perceiver, speed embeddings),
    ``campplus`` (2.5: the CAM++ embedding projected, a language embedding on the text) or ``mel`` (1.5: the
    reference's 100-band mel through the Conformer and perceiver; no emotion, no speed)."""
    num_languages: int = 107
    semantic_dim: int = 1024
    """w2v-BERT's width (the conditioning encoders' input)."""
    emo_dim: int = 1024
    """The emotion perceiver's width (upstream fixes it at 1024)."""

    @classmethod
    def from_yaml(cls, gpt: dict[str, Any], spk_cond_mode: str = "conformer", semantic_dim: int = 1024) -> GptConfig:
        if gpt.get("condition_type", "conformer_perceiver") != "conformer_perceiver" or not gpt.get("use_mel_codes_as_input", True):
            raise ValueError("Only IndexTTS-2's GPT configuration (conformer_perceiver, mel codes as input) is supported")
        names = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in gpt.items() if k in names}, spk_cond_mode=spk_cond_mode, semantic_dim=semantic_dim)


class UnifiedVoice(nn.Module):
    """Parameter names are upstream's (``gpt.h.N.*``, ``conditioning_encoder.*``…)."""

    def __init__(self, c: GptConfig, ops: Any = default_ops.default) -> None:
        super().__init__()
        self.c = c
        cm, em = c.condition_module or {}, c.emo_condition_module or {}
        if c.spk_cond_mode == "campplus":
            self.spk_emb_proj = ops.Linear(192, c.model_dim)
            self.lang_embedding = ops.Embedding(c.num_languages, c.model_dim)
        else:
            cond_in = 100 if c.spk_cond_mode == "mel" else c.semantic_dim
            self.conditioning_encoder = ConformerEncoder(cond_in, cm["output_size"], cm["attention_heads"], cm["linear_units"], cm["num_blocks"], cm["input_layer"], ops)
            self.perceiver_encoder = PerceiverResampler(c.model_dim, cm["output_size"], c.condition_num_latent, heads=cm["attention_heads"], ff_mult=cm["perceiver_mult"], ops=ops)
            if c.spk_cond_mode != "mel":
                self.speed_emb = ops.Embedding(2, c.model_dim)
        if c.spk_cond_mode != "mel":
            self.emo_conditioning_encoder = ConformerEncoder(c.semantic_dim, em["output_size"], em["attention_heads"], em["linear_units"], em["num_blocks"], em["input_layer"], ops)
            self.emo_perceiver_encoder = PerceiverResampler(c.emo_dim, em["output_size"], 1, heads=em["attention_heads"], ff_mult=em["perceiver_mult"], ops=ops)
            self.emo_layer = ops.Linear(c.model_dim, c.model_dim)
            self.emovec_layer = ops.Linear(c.emo_dim, c.model_dim)
        self.text_embedding = ops.Embedding(c.number_text_tokens + 1, c.model_dim)
        self.mel_embedding = ops.Embedding(c.number_mel_codes, c.model_dim)
        self.gpt = GPT2Model(c.model_dim, c.heads, c.layers, ops)
        self.mel_pos_embedding = LearnedPositionEmbeddings(c.max_mel_tokens + 2 + 1, c.model_dim, ops)
        self.text_pos_embedding = LearnedPositionEmbeddings(c.max_text_tokens + 2, c.model_dim, ops)
        self.final_norm = ops.LayerNorm(c.model_dim)
        self.text_head = ops.Linear(c.model_dim, c.number_text_tokens + 1)
        self.mel_head = ops.Linear(c.model_dim, c.number_mel_codes)
        self.graphs_for: Callable[[], Any] = lambda: None
        """CUDA graphs to decode with now (``compute.cuda_graphs``; the engine sets it), else None."""
        self._graphs: Any = None
        self._steps: Any = None
        self._step_ports: tuple[torch.Tensor, torch.Tensor] | None = None
        self.graphs_failed = False
        self.graphed = False

    def begin_decoding(self) -> None:
        """A decoding: the graphs to use now (new ones when the weights moved), else eager."""
        graphs = None if self.graphs_failed else self.graphs_for()
        if graphs is not self._graphs:
            self._graphs, self._steps, self._step_ports = graphs, None, None
        self.graphed = graphs is not None

    def step_logits(self, emb: torch.Tensor, caches: list[list[torch.Tensor]]) -> torch.Tensor:
        """A one-token step of every beam: the last position's mel logits in float32, (beams, codes); ``caches`` is
        updated. Eagerly, or as graphs around each layer's attention, which replay the same kernels."""
        if self._graphs is not None and self._steps is None:
            self._steps = self._capture_steps(emb)
        if self._steps is None:
            hidden, new = self.gpt(emb, caches)
            caches[:] = new
            return self.logits(hidden)[:, -1, :].clone().float()
        assert self._step_ports is not None
        x, logits = self._step_ports
        x.copy_(emb)
        with torch.cuda.device(self._graphs.device):
            self._steps(caches)
        return logits

    def _capture_steps(self, emb: torch.Tensor) -> Any:
        from mizuoto.runtime.device import is_oom
        from mizuoto.runtime.graphs import LayerSteps

        blocks: list[Any] = list(self.gpt.h)
        x = torch.zeros_like(emb)
        logits = torch.zeros((emb.shape[0], self.mel_head.out_features), dtype=torch.float32, device=emb.device)

        def prologue() -> torch.Tensor:
            return x + wpe_zeros(x)  # GPT2Model.forward's

        def attend(i: int, qkv: tuple[torch.Tensor, ...], caches: list[list[torch.Tensor]]) -> torch.Tensor:
            out, caches[i] = blocks[i].attn.attend(*qkv, caches[i])
            return out

        def epilogue(h: torch.Tensor) -> None:
            logits.copy_(self.logits(self.gpt.ln_f(h))[:, -1, :].float())

        try:
            with torch.cuda.device(self._graphs.device):
                steps = LayerSteps(
                    self._graphs,
                    len(blocks),
                    prologue,
                    lambda i, h: blocks[i].attention_in(h),
                    attend,
                    lambda i, h, attended: blocks[i].attention_out(h, attended),
                    epilogue,
                    lambda i: torch.zeros_like(emb),
                    [logits],
                )
        except Exception as e:  # any failure to capture means running eagerly (the same computation)
            if is_oom(e):
                raise
            import logging

            logging.getLogger(__name__).warning("IndexTTS: CUDA graphs failed (%s: %s); decoding eagerly", type(e).__name__, e)
            self.graphs_failed, self._graphs = True, None
            return None
        self._step_ports = (x, logits)
        return steps

    # -- conditioning --------------------------------------------------------------------------------

    def get_conditioning(self, features: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """(B, T, semantic_dim) w2v-BERT features (1.5: (B, T, 100) mels) → (B, latents, dim)."""
        x, mask = self.conditioning_encoder(features, lengths)
        conds_mask = F.pad(mask.squeeze(1), (self.c.condition_num_latent, 0), value=True)
        return self.perceiver_encoder(x, conds_mask)

    def get_emovec(self, features: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        x, mask = self.emo_conditioning_encoder(features, lengths)
        conds_mask = F.pad(mask.squeeze(1), (1, 0), value=True)
        vec = self.emo_perceiver_encoder(x, conds_mask).squeeze(1)
        return self.emo_layer(self.emovec_layer(vec))

    def merge_emovec(self, spk: torch.Tensor, emo: torch.Tensor, spk_lengths: torch.Tensor, emo_lengths: torch.Tensor, alpha: float = 1.0) -> torch.Tensor:
        emo_vec = self.get_emovec(emo, emo_lengths)
        base_vec = self.get_emovec(spk, spk_lengths)
        return base_vec + alpha * (emo_vec - base_vec)

    def campplus_latent(self, style: torch.Tensor) -> torch.Tensor:
        """2.5: the projected CAM++ embedding, (1, 1, dim)."""
        latent = self.spk_emb_proj(style)
        return latent.unsqueeze(0) if latent.ndim != 3 else latent

    def _conds(self, speech_latent: torch.Tensor, emo_vec: torch.Tensor | None, batch: int) -> torch.Tensor:
        device = speech_latent.device
        if self.c.spk_cond_mode == "mel" or emo_vec is None:
            return speech_latent
        if self.c.spk_cond_mode == "campplus":
            zeros = torch.zeros(speech_latent.size(0), 2, speech_latent.size(2)).to(device)
            return torch.cat((speech_latent + emo_vec.unsqueeze(1), zeros), 1)
        duration = self.speed_emb(torch.zeros(batch, device=device).long())
        duration_half = self.speed_emb(torch.ones(batch, device=device).long())
        return torch.cat((speech_latent + emo_vec.unsqueeze(1), duration_half.unsqueeze(1), duration.unsqueeze(1)), 1)

    # -- generation ----------------------------------------------------------------------------------

    def prompt(self, conds: torch.Tensor, text: torch.Tensor, lang: torch.Tensor | None = None) -> torch.Tensor:
        """``prepare_gpt_inputs`` for one text: ``[conds][start text… stop]`` embeddings, (1, s, dim).

        Every start and stop id is dropped from the text first, as upstream does (2.5's tiktoken ids 0 and 1 too)."""
        c = self.c
        tokens = text[0][(text[0] != c.stop_text_token) & (text[0] != c.start_text_token)]
        tokens = F.pad(F.pad(tokens, (1, 0), value=c.start_text_token), (0, 1), value=c.stop_text_token)
        pos = torch.arange(0, tokens.size(-1), device=text.device)
        text_emb = self.text_embedding(tokens) + self.text_pos_embedding.emb(pos)
        if lang is not None and c.spk_cond_mode == "campplus":
            text_emb += self.lang_embedding(lang[0])
        return torch.stack([torch.cat([conds.squeeze(0), text_emb])], dim=0)

    def logits(self, x: torch.Tensor) -> torch.Tensor:
        return self.mel_head(self.final_norm(x))

    def latents(self, speech_latent: torch.Tensor, text: torch.Tensor, codes: torch.Tensor, emo_vec: torch.Tensor | None) -> torch.Tensor:
        """``forward`` with ``return_latent``: the normed hidden states over the codes, (1, codes, dim).

        1.5 passes its codes' lengths as ``code_lens × 1024`` and pads past ``ceil(len) + 1``: nothing, for one row."""
        c = self.c
        conds = self._conds(speech_latent, emo_vec, 1)
        text_in = F.pad(F.pad(text, (0, 1), value=c.stop_text_token), (1, 0), value=c.start_text_token)
        text_emb = self.text_embedding(text_in) + self.text_pos_embedding(text_in)
        mel_in = F.pad(F.pad(codes, (0, 1), value=c.stop_mel_token), (1, 0), value=c.start_mel_token)
        mel_emb = self.mel_embedding(mel_in) + self.mel_pos_embedding(mel_in)
        emb = torch.cat([conds, text_emb, mel_emb], dim=1)
        hidden, _ = self.gpt(emb)
        enc = self.final_norm(hidden[:, conds.shape[1] :])
        return enc[:, -mel_emb.shape[1] :][:, :-2]


@dataclass
class BeamSampling:
    num_beams: int = 3
    temperature: float = 0.8
    top_k: int = 30
    top_p: float = 0.8
    repetition_penalty: float = 10.0
    length_penalty: float = 0.0
    max_new_tokens: int = 1500
    kv_cache: bool = True
    """IndexTTS-1.5 generates without one unless it runs in fp16: every step then reruns the whole sequence."""


class _Hypotheses:
    """``BeamHypotheses`` with ``early_stopping=False``."""

    def __init__(self, num_beams: int, length_penalty: float) -> None:
        self.num_beams, self.length_penalty = num_beams, length_penalty
        self.beams: list[tuple[float, torch.Tensor]] = []
        self.worst_score = 1e9

    def add(self, hyp: torch.Tensor, sum_logprobs: float, generated_len: int) -> None:
        score = sum_logprobs / (generated_len**self.length_penalty)
        if len(self.beams) < self.num_beams or score > self.worst_score:
            self.beams.append((score, hyp))
            if len(self.beams) > self.num_beams:
                ranked = sorted([(s, i) for i, (s, _) in enumerate(self.beams)])
                del self.beams[ranked[0][1]]
                self.worst_score = ranked[1][0]
            else:
                self.worst_score = min(score, self.worst_score)

    def is_done(self, best_sum_logprobs: float, cur_len: int, prompt_len: int) -> bool:
        if len(self.beams) < self.num_beams:
            return False
        return self.worst_score >= best_sum_logprobs / (cur_len - prompt_len) ** self.length_penalty


def beam_sample(
    model: UnifiedVoice,
    prompt: torch.Tensor,
    s: BeamSampling,
    generator: torch.Generator,
    check: Callable[[], None] | None = None,
) -> torch.Tensor:
    """``inference_model.generate(...)`` for one prompt: the generated codes (1, n), ending in the stop token when one was drawn."""
    c = model.c
    n, device = s.num_beams, prompt.device
    mel_len = prompt.shape[1]
    ids = torch.ones((n, mel_len + 1), dtype=torch.long, device=device)
    ids[:, -1] = c.start_mel_token
    prompt_len = ids.shape[1]
    max_length = prompt_len + s.max_new_tokens
    beam_scores = torch.zeros((1, n), dtype=torch.float, device=device)
    beam_scores[:, 1:] = -1e9
    beam_scores = beam_scores.view((n,))
    hyps = _Hypotheses(n, s.length_penalty)
    done = False
    caches: BeamCaches | None = None
    keep = 2 * n
    next_tokens = next_indices = None
    if s.kv_cache:
        model.begin_decoding()
    while True:
        if check is not None:
            check()
        if caches is None:
            start = model.mel_embedding(ids[:, mel_len:]) + model.mel_pos_embedding(ids[:, mel_len:])
            emb = torch.cat([prompt.repeat_interleave(n, 0), start], dim=1)
            hidden, new_caches = model.gpt(emb, caches)
            caches = BeamCaches(new_caches, max_length) if s.kv_cache else None
            logits = model.logits(hidden)[:, -1, :].clone().float()
        else:
            emb = model.mel_embedding(ids[:, -1:]) + model.mel_pos_embedding.fixed(ids.shape[1] - mel_len, device)
            logits = model.step_logits(emb, caches)
        scores = F.log_softmax(logits, dim=-1)
        scores = repetition_penalty(scores, ids, s.repetition_penalty)
        scores = scores / s.temperature
        scores = top_k(scores, s.top_k, min_tokens_to_keep=2)
        scores = top_p(scores, s.top_p, min_tokens_to_keep=2)
        scores = scores + beam_scores[:, None].expand_as(scores)
        vocab = scores.shape[-1]
        scores = scores.view(1, n * vocab)
        probs = F.softmax(scores, dim=-1)
        drawn = multinomial_draws(probs, keep, generator)
        top = torch.gather(scores, -1, drawn)
        top, order = torch.sort(top, descending=True, dim=1)
        drawn = torch.gather(drawn, -1, order)
        next_indices = torch.div(drawn, vocab, rounding_mode="floor")
        next_tokens = drawn % vocab
        # BeamSearchScorer.process, on the host: the candidates read back in one wait (upstream reads each token and
        # score with ``.item()``); the same values (float32 scores) and decisions
        cur_len = ids.shape[-1] + 1
        top_host, picks = top[0].to("cpu", non_blocking=True), torch.stack([next_tokens[0], next_indices[0]]).to("cpu", non_blocking=True)
        if device.type == "cuda":
            torch.cuda.current_stream(device).synchronize()
        tokens_host, indices_host = picks.tolist()
        new_scores = torch.zeros((1, n), dtype=top.dtype)
        new_tokens = torch.zeros((1, n), dtype=next_tokens.dtype)
        new_indices = torch.zeros((1, n), dtype=next_indices.dtype)
        if done:
            new_tokens[0, :] = c.stop_mel_token
        else:
            beam = 0
            for rank in range(keep):
                token, index = tokens_host[rank], indices_host[rank]
                if token == c.stop_mel_token:
                    if rank >= n:
                        continue
                    hyps.add(ids[index].clone(), float(top_host[rank]), cur_len - prompt_len)
                else:
                    new_scores[0, beam], new_tokens[0, beam], new_indices[0, beam] = top_host[rank], token, index
                    beam += 1
                if beam == n:
                    break
            done = done or hyps.is_done(float(top_host.max()), cur_len, prompt_len)
        kept = new_indices.view(-1).tolist()
        all_stopped = bool((new_tokens == c.stop_mel_token).all())
        beam_scores = new_scores.view(-1).to(device, non_blocking=True)
        moves = torch.stack([new_indices.view(-1), new_tokens.view(-1)]).to(device, non_blocking=True)
        beam_idx = moves[0]
        ids = torch.cat([ids[beam_idx, :], moves[1].unsqueeze(-1)], dim=-1)
        if done or ids.shape[-1] >= max_length or all_stopped:
            break
        if caches is not None:
            caches.advance(kept)
    # BeamSearchScorer.finalize
    if not done:
        for i in range(n):
            hyps.add(ids[i], beam_scores[i].item(), ids.shape[-1] - prompt_len)
    best = sorted(hyps.beams, key=lambda x: x[0])[-1][1]
    length = int(best.shape[-1])
    out_len = min(length + 1, max_length)
    out = ids.new(1, out_len)
    out[0, :length] = best
    if length < out_len:
        out[0, length] = c.stop_mel_token
    return out[:, prompt_len:]


def remove_long_silence(codes: torch.Tensor, stop_mel_token: int, silent_token: int = 52, max_consecutive: int = 30) -> tuple[torch.Tensor, torch.Tensor]:
    """IndexTTS-1.5's ``remove_long_silence``: codes cut at the stop token; runs of the silent code shortened to 10
    when a row holds more than ``max_consecutive`` of them."""
    from torch.nn.utils.rnn import pad_sequence

    code_lens: list[int] = []
    rows: list[torch.Tensor] = []
    fixed = False
    for i in range(codes.shape[0]):
        code = codes[i]
        if not torch.any(code == stop_mel_token).item():
            length = code.size(0)
        else:
            stops = (code == stop_mel_token).nonzero(as_tuple=False)
            length = int(stops[0].item()) if len(stops) > 0 else code.size(0)
        if torch.sum(code == silent_token).item() > max_consecutive:
            keep: list[int] = []
            n = 0
            for k in range(length):
                if code[k] != silent_token:
                    keep.append(k)
                    n = 0
                elif n < 10:
                    keep.append(k)
                    n += 1
            length = len(keep)
            rows.append(code[keep])
            fixed = True
        else:
            rows.append(code[:length])
        code_lens.append(length)
    if fixed:
        codes = pad_sequence(rows, batch_first=True, padding_value=stop_mel_token) if len(rows) > 1 else rows[0].unsqueeze(0)
    longest = max(code_lens)
    if longest < codes.shape[1]:
        codes = codes[:, :longest]
    return codes, torch.tensor(code_lens, dtype=torch.long, device=codes.device)
