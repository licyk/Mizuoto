"""Qwen3-TTS's talker and code predictor (upstream ``modeling_qwen3_tts.py``), on the shared Qwen decoder.

The talker predicts each frame's first codebook (c0) from the prompt and the previous frames. The
code predictor (the "sub-talker") then predicts the other 15 codebooks of that frame, one at a time,
from the talker's last hidden state and c0. Parameter names are upstream's (``talker.model.layers.N…``,
``talker.code_predictor…``).
"""

from __future__ import annotations

from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from mizuoto.engines.qwen3tts.config import TalkerConfig
from mizuoto.runtime import ops as default_ops
from mizuoto.runtime.lm.cache import DynamicCache
from mizuoto.runtime.lm.qwen import QwenDecoder


class TalkerModel(QwenDecoder):
    """``talker.model``: the decoder plus the codec and text embeddings."""

    def __init__(self, c: TalkerConfig, ops: Any = default_ops.default) -> None:
        super().__init__(c.decoder, ops)
        self.codec_embedding = ops.Embedding(c.vocab_size, c.decoder.hidden_size)
        self.text_embedding = ops.Embedding(c.text_vocab_size, c.text_hidden_size)


class ResizeMLP(nn.Module):
    """``text_projection``: text embeddings into the talker's width (SiLU MLP with biases)."""

    def __init__(self, size_in: int, hidden: int, size_out: int, ops: Any = default_ops.default) -> None:
        super().__init__()
        self.linear_fc1 = ops.Linear(size_in, hidden, bias=True)
        self.linear_fc2 = ops.Linear(hidden, size_out, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear_fc2(F.silu(self.linear_fc1(x)))


class CodePredictorModel(QwenDecoder):
    def __init__(self, c: TalkerConfig, ops: Any = default_ops.default) -> None:
        super().__init__(c.predictor, ops)
        self.codec_embedding = nn.ModuleList([ops.Embedding(c.predictor_vocab_size, c.decoder.hidden_size) for _ in range(c.num_code_groups - 1)])


class CodePredictor(nn.Module):
    def __init__(self, c: TalkerConfig, ops: Any = default_ops.default) -> None:
        super().__init__()
        self.model = CodePredictorModel(c, ops)
        self.lm_head = nn.ModuleList([ops.Linear(c.predictor.hidden_size, c.predictor_vocab_size, bias=False) for _ in range(c.num_code_groups - 1)])
        if c.predictor.hidden_size != c.decoder.hidden_size:
            self.small_to_mtp_projection: nn.Module = ops.Linear(c.decoder.hidden_size, c.predictor.hidden_size, bias=True)
        else:
            self.small_to_mtp_projection = nn.Identity()
        self._ropes: dict[tuple[Any, ...], tuple[torch.Tensor, torch.Tensor]] = {}

    def step(self, inputs_embeds: torch.Tensor, step: int, cache: DynamicCache) -> torch.Tensor:
        """Logits of codebook ``step + 1`` for every position of ``inputs_embeds`` (talker width)."""
        x = self.small_to_mtp_projection(inputs_embeds)
        past = cache.length
        positions = torch.arange(past, past + x.shape[1], device=x.device).unsqueeze(0).expand(x.shape[0], -1)
        hidden = self.model(x, positions, None, cache, self._rope(x, past, positions))
        return self.lm_head[step](hidden)

    def _rope(self, x: torch.Tensor, past: int, positions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor] | None:
        """cos and sin for the step's positions, which are the same every frame (0 to 15): computed once (without
        autograd) per batch size, device and dtype."""
        if torch.is_grad_enabled():
            return None
        key = (past, *x.shape[:2], x.device, x.dtype)
        rope = self._ropes.get(key)
        if rope is None:
            rope = self._ropes[key] = self.model.rotary_emb(x, positions)
        return rope


class Talker(nn.Module):
    def __init__(self, c: TalkerConfig, ops: Any = default_ops.default) -> None:
        super().__init__()
        self.config = c
        self.model = TalkerModel(c, ops)
        self.text_projection = ResizeMLP(c.text_hidden_size, c.text_hidden_size, c.decoder.hidden_size, ops)
        self.codec_head = ops.Linear(c.decoder.hidden_size, c.vocab_size, bias=False)
        self.code_predictor = CodePredictor(c, ops)

    def text(self, ids: torch.Tensor) -> torch.Tensor:
        """Text ids projected into the talker's width."""
        return self.text_projection(self.model.text_embedding(ids))

    def codec(self, ids: torch.Tensor) -> torch.Tensor:
        return self.model.codec_embedding(ids)

    def forward(self, inputs_embeds: torch.Tensor, position_ids: torch.Tensor, mask: torch.Tensor | None, cache: DynamicCache) -> tuple[torch.Tensor, torch.Tensor]:
        """(logits over the codec vocabulary for every position, the last hidden states)."""
        hidden = self.model(inputs_embeds, position_ids, mask, cache)
        return self.codec_head(hidden), hidden
