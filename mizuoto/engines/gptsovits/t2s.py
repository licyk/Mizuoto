"""GPT-SoVITS's text-to-semantic model (``AR/models/t2s_model.py``, ``AR/models/utils.py``, ``AR/modules/embedding.py``;
MIT): phones and BERT features in, 50 Hz semantic tokens out, continuing the reference's tokens.

A post-norm transformer (fused ``in_proj``, ReLU MLP) read the way upstream's TorchScript ``T2SBlock`` reads it:
``F.linear`` on the fused weights, SDPA with the inverted boolean mask, LayerNorm after each residual. Two
decoding paths, as ``TTS.run`` picks them:
- ``infer_batch`` (``infer_panel_batch_infer``, the default): text rows left-padded to one length, one causal
  mask over [text][prompt + tokens], rows dropped as they finish;
- ``infer_naive`` (``infer_panel_naive``, streaming): one row, yielding chunks of tokens cut at a mute-like token.
Sampling (``sample``): repetition penalty over every token so far (the prompt included), top-p, temperature,
top-k, softmax, then ``argmax(p / q)`` with ``q`` exponential draws — from the request's generator here. EOS
cannot be drawn in the first 11 steps; a row stops at EOS sampled *or* argmax.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from typing import Any, cast

import torch
from torch import nn
from torch.nn import functional as F

from mizuoto.runtime import ops as default_ops
from mizuoto.runtime.accel import Site, attention

SITE = Site("gptsovits.t2s")

MAX_STEPS = 1500


@dataclass
class T2SConfig:
    hidden_dim: int = 512
    embedding_dim: int = 512
    num_head: int = 16
    num_layers: int = 24
    vocab_size: int = 1025
    phoneme_vocab_size: int = 732
    eos: int = 1024
    max_sec: int = 54
    bert_dim: int = 1024

    @classmethod
    def from_checkpoint(cls, config: dict[str, Any]) -> T2SConfig:
        m = config["model"]
        return cls(
            hidden_dim=m["hidden_dim"],
            embedding_dim=m["embedding_dim"],
            num_head=m["head"],
            num_layers=m["n_layer"],
            vocab_size=m["vocab_size"],
            phoneme_vocab_size=m["phoneme_vocab_size"],
            eos=m["EOS"],
            max_sec=config.get("data", {}).get("max_sec", 54),
        )


class TokenEmbedding(nn.Module):
    def __init__(self, dim: int, vocab: int, ops: Any) -> None:
        super().__init__()
        self.word_embeddings = ops.Embedding(vocab, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.word_embeddings(x)


class SinePositionalEmbedding(nn.Module):
    """``scale=False, alpha=True``: ``x + alpha · pe``, the table for 4000 positions in float32."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.x_scale = 1.0
        self.alpha = nn.Parameter(torch.empty(1))
        self.dim = dim
        self.pe = self._table(4000)

    def _table(self, length: int) -> torch.Tensor:
        pe = torch.zeros(length, self.dim)
        position = torch.arange(0, length, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, self.dim, 2, dtype=torch.float32) * -(math.log(10000.0) / self.dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe.unsqueeze(0)

    def table(self, length: int, like: torch.Tensor) -> torch.Tensor:
        if self.pe.size(1) < length:
            self.pe = self._table(length)
        if self.pe.dtype != like.dtype or self.pe.device != like.device:
            self.pe = self.pe.to(dtype=like.dtype, device=like.device)
        return self.pe

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pe = self.table(x.size(1), x)
        out = x.unsqueeze(-1) if x.ndim == 2 else x
        return out * self.x_scale + self.alpha * pe[:, : x.size(1)]


class MultiheadAttention(nn.Module):
    """Only the parameters: ``in_proj_weight``/``in_proj_bias`` and ``out_proj``."""

    def __init__(self, dim: int, ops: Any) -> None:
        super().__init__()
        self.in_proj_weight = nn.Parameter(torch.empty(3 * dim, dim))
        self.in_proj_bias = nn.Parameter(torch.empty(3 * dim))
        self.out_proj = ops.Linear(dim, dim)


class EncoderLayer(nn.Module):
    def __init__(self, dim: int, heads: int, ops: Any) -> None:
        super().__init__()
        self.self_attn = MultiheadAttention(dim, ops)
        self.linear1 = ops.Linear(dim, dim * 4)
        self.linear2 = ops.Linear(dim * 4, dim)
        self.norm1 = ops.LayerNorm(dim, eps=1e-5)
        self.norm2 = ops.LayerNorm(dim, eps=1e-5)
        self.heads, self.dim = heads, dim

    def _finish(self, x: torch.Tensor, attn: torch.Tensor) -> torch.Tensor:
        x = x + attn
        x = F.layer_norm(x, [self.dim], self.norm1.weight, self.norm1.bias, self.norm1.eps)
        x = x + F.linear(F.relu(F.linear(x, self.linear1.weight, self.linear1.bias)), self.linear2.weight, self.linear2.bias)
        return F.layer_norm(x, [self.dim], self.norm2.weight, self.norm2.bias, self.norm2.eps)

    def _attend(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
        return F.linear(self._attention(q, k, v, mask), self.self_attn.out_proj.weight, self.self_attn.out_proj.bias)

    def _attention(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
        b, q_len, kv_len = q.shape[0], q.shape[1], k.shape[1]
        q = q.view(b, q_len, self.heads, -1).transpose(1, 2)
        k = k.view(b, kv_len, self.heads, -1).transpose(1, 2)
        v = v.view(b, kv_len, self.heads, -1).transpose(1, 2)
        attn = attention(q, k, v, site=SITE, mask=(~mask) if mask is not None else None)
        return attn.transpose(1, 2).reshape(b, q_len, -1)

    def prompt(self, x: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        q, k, v = F.linear(x, self.self_attn.in_proj_weight, self.self_attn.in_proj_bias).chunk(3, dim=-1)
        return self._finish(x, self._attend(q, k, v, mask)), k, v

    def step(self, x: torch.Tensor, k_cache: torch.Tensor, v_cache: torch.Tensor, mask: torch.Tensor | None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        q, k, v = F.linear(x, self.self_attn.in_proj_weight, self.self_attn.in_proj_bias).chunk(3, dim=-1)
        k_cache = torch.cat([k_cache, k], dim=1)
        v_cache = torch.cat([v_cache, v], dim=1)
        return self._finish(x, self._attend(q, k_cache, v_cache, mask)), k_cache, v_cache

    # ``step`` in three parts (the same operations in the same order), for steps run as CUDA graphs around the eager
    # cache and attention (``mizuoto.runtime.graphs.LayerSteps``).
    def step_qkv(self, x: torch.Tensor) -> tuple[torch.Tensor, ...]:
        return F.linear(x, self.self_attn.in_proj_weight, self.self_attn.in_proj_bias).chunk(3, dim=-1)

    def step_attend(self, qkv: tuple[torch.Tensor, ...], ks: list[torch.Tensor], vs: list[torch.Tensor], i: int, mask: torch.Tensor | None) -> torch.Tensor:
        q, k, v = qkv
        ks[i] = torch.cat([ks[i], k], dim=1)
        vs[i] = torch.cat([vs[i], v], dim=1)
        return self._attention(q, ks[i], vs[i], mask)

    def step_finish(self, x: torch.Tensor, attended: torch.Tensor) -> torch.Tensor:
        return self._finish(x, F.linear(attended, self.self_attn.out_proj.weight, self.self_attn.out_proj.bias))


class Encoder(nn.Module):
    def __init__(self, c: T2SConfig, ops: Any) -> None:
        super().__init__()
        self.layers = nn.ModuleList(EncoderLayer(c.hidden_dim, c.num_head, ops) for _ in range(c.num_layers))


@dataclass
class Sampling:
    top_k: int = 15
    top_p: float = 1.0
    temperature: float = 1.0
    repetition_penalty: float = 1.35
    early_stop_num: int = -1


def make_pad_mask_left(lengths: torch.Tensor, max_len: int = 0) -> torch.Tensor:
    max_len = max(max_len, int(lengths.max()))
    n = lengths.size(0)
    seq = torch.arange(0, max_len, device=lengths.device).unsqueeze(0).repeat(n, 1)
    seq -= (max_len - lengths).unsqueeze(-1)
    return seq < 0


def sample(logits: torch.Tensor, previous: torch.Tensor, s: Sampling, generator: torch.Generator) -> torch.Tensor:
    """``sample`` → ``logits_to_probs`` + ``multinomial_sample_one_no_sync``: (batch, 1) int tokens."""
    if s.repetition_penalty != 1.0:
        previous = previous.long()
        score = torch.gather(logits, dim=1, index=previous)
        score = torch.where(score < 0, score * s.repetition_penalty, score / s.repetition_penalty)
        logits.scatter_(dim=1, index=previous, src=score)
    if s.top_p is not None and s.top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cum_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        remove = cum_probs > s.top_p
        remove[:, 0].fill_(False)  # upstream assigns a Python bool: a copy to the device that waits for it
        remove = remove.scatter(dim=1, index=sorted_indices, src=remove)
        logits = logits.masked_fill(remove, -float("Inf"))
    logits = logits / max(s.temperature, 1e-5)
    if s.top_k is not None:
        v, _ = torch.topk(logits, min(s.top_k, logits.size(-1)))
        pivot = v[:, -1].unsqueeze(-1)
        logits = torch.where(logits < pivot, -float("Inf"), logits)
    probs = F.softmax(logits, dim=-1)
    q = torch.empty_like(probs).exponential_(1, generator=generator)
    return torch.argmax(probs / q, dim=-1, keepdim=True).to(dtype=torch.int)


class Text2SemanticDecoder(nn.Module):
    """Parameter names are upstream's ``Text2SemanticLightningModule.model`` (checkpoint keys without ``model.``)."""

    def __init__(self, c: T2SConfig, ops: Any = default_ops.default) -> None:
        super().__init__()
        self.c = c
        self.bert_proj = ops.Linear(c.bert_dim, c.embedding_dim)
        self.ar_text_embedding = TokenEmbedding(c.embedding_dim, c.phoneme_vocab_size, ops)
        self.ar_text_position = SinePositionalEmbedding(c.embedding_dim)
        self.ar_audio_embedding = TokenEmbedding(c.embedding_dim, c.vocab_size, ops)
        self.ar_audio_position = SinePositionalEmbedding(c.embedding_dim)
        self.h = Encoder(c, ops)
        self.ar_predict_layer = ops.Linear(c.hidden_dim, c.vocab_size, bias=False)
        self.graphs_for: Callable[[], Any] = lambda: None
        """CUDA graphs to decode with now (``compute.cuda_graphs``; the engine sets it), else None."""
        self._graphs: Any = None
        self._steps: dict[int, Any] = {}
        self._step_ports: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
        self._graphs_failed = False
        self.graphed = False
        """The last decoding used graphs."""

    def memory_units(self) -> list[nn.Module]:
        """None (``runtime.units``): the layers run through ``prompt``/``step``, which read their weights directly
        (``F.linear`` on the fused projections), so the GPT stays whole on the device; it is small."""
        return []

    def _prompt(self, xy: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor], list[torch.Tensor]]:
        ks, vs = [], []
        for layer in cast("list[EncoderLayer]", self.h.layers):
            xy, k, v = layer.prompt(xy, mask)
            ks.append(k)
            vs.append(v)
        return xy, ks, vs

    def _step(self, xy: torch.Tensor, ks: list[torch.Tensor], vs: list[torch.Tensor], mask: torch.Tensor | None) -> torch.Tensor:
        for i, layer in enumerate(cast("list[EncoderLayer]", self.h.layers)):
            xy, ks[i], vs[i] = layer.step(xy, ks[i], vs[i], mask)
        return xy

    def _begin(self) -> None:
        """A decoding: the graphs to use now (new ones when the weights moved), else eager."""
        graphs = None if self._graphs_failed else self.graphs_for()
        if graphs is not self._graphs:
            self._graphs, self._steps, self._step_ports = graphs, {}, {}
        self.graphed = graphs is not None

    def _step_logits(self, xy: torch.Tensor, ks: list[torch.Tensor], vs: list[torch.Tensor], mask: torch.Tensor | None) -> torch.Tensor:
        """A one-token step's logits (rows, vocabulary): eagerly, or as graphs around the attention calls (one set per
        number of rows: rows drop as they finish), which replay the same kernels."""
        bsz = xy.shape[0]
        if self._graphs is None:
            return self.ar_predict_layer(self._step(xy, ks, vs, mask)[:, -1])
        steps = self._steps.get(bsz)
        if steps is None:
            steps = self._capture_steps(xy)
            if steps is None:
                return self.ar_predict_layer(self._step(xy, ks, vs, mask)[:, -1])
        x, logits = self._step_ports[bsz]
        x.copy_(xy)
        with torch.cuda.device(self._graphs.device):
            steps(mask, ks, vs)
        return logits

    def _capture_steps(self, xy: torch.Tensor) -> Any:
        from mizuoto.runtime.device import is_oom
        from mizuoto.runtime.graphs import LayerSteps

        bsz = xy.shape[0]
        layers = cast("list[EncoderLayer]", list(self.h.layers))
        x = torch.zeros_like(xy)
        logits = torch.zeros((bsz, self.c.vocab_size), dtype=xy.dtype, device=xy.device)

        def epilogue(h: torch.Tensor) -> None:
            logits.copy_(self.ar_predict_layer(h[:, -1]))

        try:
            with torch.cuda.device(self._graphs.device):
                steps = LayerSteps(
                    self._graphs,
                    len(layers),
                    lambda: x,
                    lambda i, h: layers[i].step_qkv(h),
                    lambda i, qkv, mask, ks, vs: layers[i].step_attend(qkv, ks, vs, i, mask),
                    lambda i, h, attended: layers[i].step_finish(h, attended),
                    epilogue,
                    lambda i: torch.zeros_like(xy),
                    [logits],
                )
        except Exception as e:  # any failure to capture means running eagerly (the same computation)
            if is_oom(e):
                raise
            import logging

            logging.getLogger(__name__).warning("GPT-SoVITS: CUDA graphs failed (%s: %s); decoding eagerly", type(e).__name__, e)
            self._graphs_failed, self._graphs, self._steps, self._step_ports = True, None, {}, {}
            return None
        while len(self._steps) >= 4:
            dropped = next(iter(self._steps))
            self._steps.pop(dropped)
            self._step_ports.pop(dropped)
        self._steps[bsz], self._step_ports[bsz] = steps, (x, logits)
        return steps

    def _next_input(self, y: torch.Tensor, position: int) -> torch.Tensor:
        y_emb = self.ar_audio_embedding(y[:, -1:])
        pe = self.ar_audio_position.table(position + 1, y_emb)
        return y_emb * self.ar_audio_position.x_scale + self.ar_audio_position.alpha * pe[:, position].to(dtype=y_emb.dtype, device=y_emb.device)

    def infer_batch(
        self,
        x: list[torch.Tensor],
        x_lens: torch.Tensor,
        prompts: torch.Tensor,
        bert_feature: list[torch.Tensor],
        s: Sampling,
        generator: torch.Generator,
        max_len: int | None = None,
        check: Callable[[], None] | None = None,
    ) -> tuple[list[torch.Tensor], list[int]]:
        """``infer_panel_batch_infer``: per row, the tokens (prompt included, EOS excluded) and the index it stopped at."""
        c = self.c
        max_len = int(x_lens.max()) if max_len is None else int(max_len)
        rows = []
        for x_item, bert_item in zip(x, bert_feature, strict=True):
            item = self.ar_text_embedding(x_item.unsqueeze(0))
            item = item + self.bert_proj(bert_item.transpose(0, 1).unsqueeze(0))
            item = self.ar_text_position(item).squeeze(0)
            item = F.pad(item, (0, 0, max_len - item.shape[0], 0), value=0) if item.shape[0] < max_len else item
            rows.append(item)
        xs = torch.stack(rows, dim=0)
        y = prompts
        x_len = xs.shape[1]
        y_emb = self.ar_audio_embedding(y)
        y_len = y_emb.shape[1]
        prefix_len = y.shape[1]
        y_lens = torch.LongTensor([y_emb.shape[1]] * y_emb.shape[0]).to(xs.device)
        xy_pos = torch.concat([xs, self.ar_audio_position(y_emb)], dim=1)
        bsz = xs.shape[0]
        src_len = x_len + y_len
        padding_mask = torch.concat([make_pad_mask_left(x_lens, max_len), make_pad_mask_left(y_lens, y_len)], dim=1)
        x_mask = F.pad(torch.zeros(x_len, x_len, dtype=torch.bool, device=xs.device), (0, y_len), value=True)
        y_mask = F.pad(torch.triu(torch.ones(y_len, y_len, dtype=torch.bool, device=xs.device), diagonal=1), (x_len, 0), value=False)
        causal = torch.concat([x_mask, y_mask], dim=0).view(1, src_len, src_len).repeat(bsz, 1, 1).to(xs.device)
        padding_mask = padding_mask.view(bsz, 1, src_len).repeat(1, src_len, 1)
        attn_mask = causal.logical_or(padding_mask).unsqueeze(1).expand(-1, c.num_head, -1, -1).bool()
        y_list: list[Any] = [None] * y.shape[0]
        batch_idx_map = list(range(y.shape[0]))
        idx_list: list[Any] = [None] * y.shape[0]
        ks: list[torch.Tensor] = []
        vs: list[torch.Tensor] = []
        samples = torch.zeros(0)
        self._begin()
        for idx in range(MAX_STEPS):
            if check is not None:
                check()
            if idx == 0:
                xy_dec, ks, vs = self._prompt(xy_pos, attn_mask)
                logits = self.ar_predict_layer(xy_dec[:, -1])
            else:
                logits = self._step_logits(xy_pos, ks, vs, attn_mask)
            attn_mask = F.pad(attn_mask[:, :, -1].unsqueeze(-2), (0, 1), value=False) if idx == 0 else F.pad(attn_mask, (0, 1), value=False)
            if idx < 11:
                logits = logits[:, :-1]
            samples = sample(logits, y, s, generator)
            y = torch.concat([y, samples], dim=1)
            tokens = torch.argmax(logits, dim=-1)
            reserved = None
            done = (samples[:, 0] == c.eos).logical_or(tokens == c.eos)
            if bool(done.any()):  # upstream's two ``in`` checks, read back in one wait
                removed = torch.where(done)[0].tolist()
                reserved = torch.where(~done)[0]
                for i in removed:
                    batch_index = batch_idx_map[i]
                    idx_list[batch_index] = idx
                    y_list[batch_index] = y[i, :-1]
                batch_idx_map = [batch_idx_map[i] for i in reserved.tolist()]
            if reserved is not None:
                y = torch.index_select(y, dim=0, index=reserved)
                attn_mask = torch.index_select(attn_mask, dim=0, index=reserved)
                for i in range(len(ks)):
                    ks[i] = torch.index_select(ks[i], dim=0, index=reserved)
                    vs[i] = torch.index_select(vs[i], dim=0, index=reserved)
            stop = False
            if (s.early_stop_num != -1 and (y.shape[1] - prefix_len) > s.early_stop_num) or idx == MAX_STEPS - 1:
                stop = True
                for i, batch_index in enumerate(batch_idx_map):
                    idx_list[batch_index] = idx
                    y_list[batch_index] = y[i, :-1]
            if None not in idx_list:
                stop = True
            if stop:
                if y.shape[1] == 0:
                    y = torch.concat([y, torch.zeros_like(samples)], dim=1)
                break
            xy_pos = self._next_input(y, y_len + idx)
        for i in range(len(idx_list)):
            if idx_list[i] is None:
                idx_list[i] = MAX_STEPS - 1
        return y_list, idx_list

    def infer_naive(
        self,
        x: torch.Tensor,
        prompts: torch.Tensor | None,
        bert_feature: torch.Tensor,
        s: Sampling,
        generator: torch.Generator,
        streaming: bool = False,
        chunk_length: int = 24,
        mute_emb_sim_matrix: torch.Tensor | None = None,
        chunk_split_threshold: float = 0.3,
        check: Callable[[], None] | None = None,
    ) -> Iterator[tuple[torch.Tensor | None, Any]]:
        """``infer_panel_naive``: one row. Streaming yields (chunk, is_final); otherwise (tokens, stop index) once."""
        c = self.c
        check_token_num = 2
        xs = self.ar_text_embedding(x)
        xs = xs + self.bert_proj(bert_feature.transpose(1, 2))
        xs = self.ar_text_position(xs)
        x_len = xs.shape[1]
        if prompts is not None:
            y = prompts
            y_emb = self.ar_audio_embedding(y)
            y_len = y_emb.shape[1]
            prefix_len = y.shape[1]
            xy_pos = torch.concat([xs, self.ar_audio_position(y_emb)], dim=1)
            ref_free = False
        else:
            y = torch.zeros(xs.shape[0], 0, dtype=torch.int, device=xs.device)
            y_len = prefix_len = 0
            xy_pos = xs
            ref_free = True
        bsz = xs.shape[0]
        src_len = x_len + y_len
        x_attn = F.pad(torch.zeros((x_len, x_len), dtype=torch.bool), (0, y_len), value=True)
        y_attn = F.pad(torch.triu(torch.ones(y_len, y_len, dtype=torch.bool), diagonal=1), (x_len, 0), value=False)
        mask: torch.Tensor | None = (
            torch.concat([x_attn, y_attn], dim=0).unsqueeze(0).expand(bsz * c.num_head, -1, -1).view(bsz, c.num_head, src_len, src_len).to(device=xs.device, dtype=torch.bool)
        )
        ks: list[torch.Tensor] = []
        vs: list[torch.Tensor] = []
        token_counter = 0
        curr_ptr = prefix_len
        idx = 0
        samples = torch.zeros(0)
        self._begin()
        for idx in range(MAX_STEPS):
            if check is not None:
                check()
            token_counter += 1
            if mask is not None:
                xy_dec, ks, vs = self._prompt(xy_pos, mask)
                logits = self.ar_predict_layer(xy_dec[:, -1])
            else:
                logits = self._step_logits(xy_pos, ks, vs, None)
            if idx == 0:
                mask = None
            if idx < 11:
                logits = logits[:, :-1]
            samples = sample(logits, y, s, generator)
            y = torch.concat([y, samples], dim=1)
            stop = False
            if s.early_stop_num != -1 and (y.shape[1] - prefix_len) > s.early_stop_num:
                stop = True
            if bool((torch.argmax(logits, dim=-1)[0] == c.eos) | (samples[0, 0] == c.eos)):  # one wait for both
                stop = True
                y = y[:, :-1]
                token_counter -= 1
            if idx == MAX_STEPS - 1:
                stop = True
            if stop:
                if y.shape[1] == 0:
                    y = torch.concat([y, torch.zeros_like(samples)], dim=1)
                if streaming:
                    yield (y[:, curr_ptr:] if curr_ptr < y.shape[1] else None), True
                break
            if streaming and mute_emb_sim_matrix is not None and token_counter >= chunk_length + check_token_num:
                score = mute_emb_sim_matrix[y[0, curr_ptr:]] - chunk_split_threshold
                score[score < 0] = -1
                score[:-1] = score[:-1] + score[1:]
                argmax_idx = score.argmax()
                if score[argmax_idx] >= 0 and argmax_idx + 1 >= chunk_length:
                    yield y[:, curr_ptr:], False
                    token_counter -= int(argmax_idx) + 1
                    curr_ptr += int(argmax_idx) + 1
            elif streaming and mute_emb_sim_matrix is None and token_counter >= chunk_length:
                yield y[:, -token_counter:], False
                curr_ptr += token_counter
                token_counter = 0
            xy_pos = self._next_input(y, y_len + idx)
        if not streaming:
            if ref_free:
                yield y, 0
            yield y, idx


def t2s_state(state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """A ``.ckpt``'s ``weight`` without the ``model.`` prefix."""
    return {k[len("model.") :] if k.startswith("model.") else k: v for k, v in state.items()}
