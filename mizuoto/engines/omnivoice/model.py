"""OmniVoice's network (upstream ``models/omnivoice.py``, Apache-2.0, Xiaomi Corp.): a bidirectional Qwen3 over text and audio codes.

Each position's input is either its text token's embedding or the sum of its eight codebooks'
embeddings (``audio_embeddings``, offset per codebook). The decoder is transformers' Qwen3 run with a
full 4-D boolean mask (no causality), and ``audio_heads`` predicts every codebook at every position.
Parameter names are the checkpoint's (``llm.layers.N…``, ``audio_embeddings``, ``audio_heads``).
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
from torch import nn

from mizuoto.runtime import ops as default_ops
from mizuoto.runtime.lm.qwen import QwenConfig, QwenDecoder


def qwen_config(llm: Mapping[str, Any]) -> QwenConfig:
    if llm.get("model_type") != "qwen3":
        raise ValueError(f"OmniVoice over {llm.get('model_type')!r} is not supported")
    rope = llm.get("rope_parameters") or {}
    if rope.get("rope_type", "default") != "default":
        raise ValueError(f"RoPE {rope.get('rope_type')!r} is not supported")
    return QwenConfig(
        hidden_size=llm["hidden_size"],
        intermediate_size=llm["intermediate_size"],
        num_layers=llm["num_hidden_layers"],
        num_heads=llm["num_attention_heads"],
        num_kv_heads=llm["num_key_value_heads"],
        head_dim=llm.get("head_dim") or llm["hidden_size"] // llm["num_attention_heads"],
        rms_norm_eps=llm.get("rms_norm_eps", 1e-6),
        rope_theta=float(rope.get("rope_theta", llm.get("rope_theta", 1_000_000.0))),
        qk_norm=True,
        attention_bias=bool(llm.get("attention_bias", False)),
        o_bias=bool(llm.get("attention_bias", False)),
    )


class OmniVoiceModel(nn.Module):
    def __init__(self, config: Mapping[str, Any], ops: Any = default_ops.default) -> None:
        super().__init__()
        llm = config["llm_config"]
        self.num_codebooks = int(config.get("num_audio_codebook", 8))
        self.vocab = int(config.get("audio_vocab_size", 1025))
        self.mask_id = int(config.get("audio_mask_id", 1024))
        self.llm = QwenDecoder(qwen_config(llm), ops)
        self.llm.add_module("embed_tokens", ops.Embedding(llm["vocab_size"], llm["hidden_size"]))
        self.audio_embeddings = ops.Embedding(self.num_codebooks * self.vocab, llm["hidden_size"])
        self.register_buffer("codebook_layer_offsets", torch.arange(self.num_codebooks) * self.vocab)
        self.audio_heads = ops.Linear(llm["hidden_size"], self.num_codebooks * self.vocab, bias=False)

    def forward(self, input_ids: torch.Tensor, audio_mask: torch.Tensor, attention_mask: torch.Tensor, positions: torch.Tensor | None = None) -> torch.Tensor:
        """``input_ids`` (batch, codebooks, length), ``audio_mask`` (batch, length) → logits (batch, codebooks, length, vocab).
        ``positions`` (batch, length): each position's index for RoPE, by default ``0…length−1`` in every row."""
        text = self.llm.get_submodule("embed_tokens")(input_ids[:, 0, :])
        shifted = (input_ids * audio_mask.unsqueeze(1)) + self.get_buffer("codebook_layer_offsets").view(1, -1, 1)
        audio = self.audio_embeddings(shifted).sum(dim=1)
        x = torch.where(audio_mask.unsqueeze(-1), audio, text)
        if positions is None:
            positions = torch.arange(0, x.shape[1], device=x.device).unsqueeze(0)
        hidden = self.llm(x, positions, attention_mask)
        batch, length, _ = hidden.shape
        return self.audio_heads(hidden).view(batch, length, self.num_codebooks, self.vocab).permute(0, 2, 1, 3)
