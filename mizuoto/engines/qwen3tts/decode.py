"""The talker's decoding loop, one frame at a time, over fixed tensors: eagerly, or as CUDA graphs on an NVIDIA GPU
(``compute.cuda_graphs``, ``mizuoto.runtime.graphs``).

A frame is three steps, each reading and writing the tensors of this object (its "ports"):

- ``sample``: the frame's first codebook from the talker's logits (transformers' processors and draw, the repetition
  penalty over a mask of the ids drawn so far, finished rows fed EOS);
- ``predict``: the other 15 codebooks from the code predictor, one step each, and the talker's next input;
- ``talk``: the talker's step (its logits and last hidden state).

The steps are the same code eagerly and captured. Captured, ``sample`` and ``predict`` are one graph each and
``talk`` one per stretch between attention calls (``DecoderSteps``), so a frame costs about thirty graph launches and
one wait (for whether every row has finished) instead of some six thousand kernel launches and thirty-odd waits; the
kernels and their order are the eager ones, so the codes are identical.
"""

from __future__ import annotations

import logging
from collections.abc import Generator
from contextlib import contextmanager
from typing import Any

import torch

from mizuoto.engines.qwen3tts.config import TalkerConfig
from mizuoto.runtime.device import is_oom
from mizuoto.runtime.graphs import DecoderSteps, Graphs
from mizuoto.runtime.lm.cache import DynamicCache, new_cache
from mizuoto.runtime.sampling import HfSampling, hf_sample

log = logging.getLogger(__name__)


class FrameDecoder:
    def __init__(self, talker: Any, c: TalkerConfig, batch: int, talker_sampling: HfSampling, sub_sampling: HfSampling, kv_cache: str, graphs: Graphs | None) -> None:
        weight = talker.codec_head.weight
        dev, dtype = weight.device, weight.dtype
        hidden = c.decoder.hidden_size
        self.talker = talker
        self.groups = c.num_code_groups
        self.eos = c.codec_eos_id
        self.talker_sampling = talker_sampling
        self.sub_sampling = sub_sampling
        self.kv_cache = kv_cache
        self.graphs = graphs
        self.generator: torch.Generator | None = None
        # ports
        self.logits = torch.zeros(batch, c.vocab_size, dtype=dtype, device=dev)
        self.hidden = torch.zeros(batch, 1, hidden, dtype=dtype, device=dev)
        self.token = torch.zeros(batch, dtype=torch.long, device=dev)
        self.seen = torch.zeros(batch, c.vocab_size, dtype=torch.bool, device=dev)
        self.unfinished = torch.ones(batch, dtype=torch.long, device=dev)
        self.status = torch.zeros(2, dtype=torch.long, device=dev)
        """Whether a row is unfinished, and whether the talker's probabilities held NaN (read once per frame)."""
        self.trail = torch.zeros(batch, 1, hidden, dtype=dtype, device=dev)
        self.frame = torch.zeros(batch, self.groups, dtype=torch.long, device=dev)
        self.embeds = torch.zeros(batch, 1, hidden, dtype=dtype, device=dev)
        self.positions = torch.zeros(batch, 1, dtype=torch.float32, device=dev)
        self._sample_graph: Any = None
        self._predict_graph: Any = None
        self._steps: DecoderSteps | None = None

    # -- a request ----------------------------------------------------------------------

    @contextmanager
    def run(self, generator: torch.Generator, logits: torch.Tensor, hidden: torch.Tensor) -> Generator[FrameDecoder, None, None]:
        """Decode from the prompt's ``logits`` (batch, vocab) and last ``hidden`` state, drawing from ``generator``."""
        self.logits.copy_(logits)
        self.hidden.copy_(hidden)
        self.seen.zero_()
        self.unfinished.fill_(1)
        self.status.zero_()
        if self.graphs is None:
            self.generator = generator
            yield self
            return
        with torch.cuda.device(self.graphs.device), self.graphs.drawing(generator) as own:
            self.generator = own
            yield self

    def read_status(self) -> tuple[bool, bool]:
        """(whether a row is unfinished, whether the probabilities held NaN): waits for the device."""
        unfinished, invalid = self.status.tolist()
        return bool(unfinished), bool(invalid)

    def sample(self, blocked: bool) -> None:
        """The frame's first codebook into ``token`` (``blocked``: EOS not allowed yet, min_new_tokens)."""
        if blocked or self.graphs is None:
            self._sample(blocked)
            return
        if self._sample_graph is None:
            self._sample_graph = self._capture(lambda: self._sample(False), [self.token, self.seen, self.unfinished, self.status])
            if self._sample_graph is None:
                self._sample(False)
                return
        self._sample_graph.replay()

    def predict(self) -> None:
        """The frame's codes into ``frame`` and the talker's next input into ``embeds`` (``trail`` filled first)."""
        if self.graphs is not None and self._predict_graph is None:
            self._predict_graph = self._capture(self._predict, [self.frame, self.embeds])
        if self._predict_graph is not None:
            self._predict_graph.replay()
        else:
            self._predict()

    def talk(self, mask: Any, cache: DynamicCache) -> None:
        """The talker's step on ``embeds`` at ``positions``: its logits and last hidden state into the ports."""
        if self.graphs is not None and self._steps is None:
            try:
                self._steps = DecoderSteps(self.graphs, self.talker.model, self.embeds, self.positions, self._epilogue, [self.logits, self.hidden])
            except Exception as e:  # any failure to capture means running eagerly
                self._give_up(e)
        if self._steps is not None:
            self._steps(mask, cache)
        else:
            self._epilogue(self.talker.model(self.embeds, self.positions, mask, cache))

    # -- the steps ----------------------------------------------------------------------

    def _sample(self, blocked: bool) -> None:
        s = self.talker_sampling
        token = hf_sample(self.logits, None, s, self.generator, seen=self.seen, length=0 if blocked else s.min_new_tokens, invalid=self.status[1])
        token = token * self.unfinished + self.eos * (1 - self.unfinished)
        self.token.copy_(token)
        self.seen.scatter_(1, token[:, None], True)
        self.unfinished.copy_(self.unfinished & (token != self.eos).long())
        self.status[0].copy_(self.unfinished.max())

    def _predict(self) -> None:
        cp = self.talker.code_predictor
        last = self.talker.codec(self.token[:, None])
        sub = self._codes(self.hidden, last)
        self.frame.copy_(torch.cat((self.token[:, None], sub), dim=-1))
        hiddens = torch.cat([last] + [cp.model.codec_embedding[i](sub[..., i : i + 1]) for i in range(self.groups - 1)], dim=1)
        self.embeds.copy_(hiddens.sum(1, keepdim=True) + self.trail)

    def _codes(self, past_hidden: torch.Tensor, last_hidden: torch.Tensor) -> torch.Tensor:
        """The code predictor's ``generate``: codebooks 1–15 of one frame."""
        cp = self.talker.code_predictor
        n = self.groups - 1
        cache = new_cache(self.kv_cache)
        logits = cp.step(torch.cat((past_hidden, last_hidden), dim=1), 0, cache)
        generated = torch.empty((past_hidden.shape[0], 0), dtype=torch.long, device=past_hidden.device)
        for i in range(n):
            token = hf_sample(logits[:, -1, :], generated, self.sub_sampling, self.generator)
            generated = torch.cat([generated, token[:, None]], dim=-1)
            if i == n - 1:
                break
            logits = cp.step(cp.model.codec_embedding[i](token[:, None]), i + 1, cache)
        return generated

    def _epilogue(self, hidden: torch.Tensor) -> None:
        self.hidden.copy_(hidden[:, -1:, :])
        self.logits.copy_(self.talker.codec_head(hidden)[:, -1, :])

    # -- capturing ----------------------------------------------------------------------

    def _capture(self, fn: Any, state: list[torch.Tensor]) -> Any:
        assert self.graphs is not None
        try:
            return self.graphs.capture(fn, state)
        except Exception as e:
            self._give_up(e)
            return None

    def _give_up(self, error: Exception) -> None:
        """Run eagerly from now on (the eager steps compute the same)."""
        if is_oom(error):
            raise error
        log.warning("Qwen3-TTS: CUDA graphs failed (%s: %s); decoding eagerly", type(error).__name__, error)
        if self.graphs is not None:
            self.generator = self.graphs.generator  # a fresh one, in the same state, after a failed capture
        self.graphs = None
        self._sample_graph = self._predict_graph = self._steps = None
