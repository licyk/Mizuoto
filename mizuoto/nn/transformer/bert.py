"""BERT encoder with transformers' parameter names and numerics (``BertModel``, transformers 4.x).

Users: GPT-SoVITS's chinese-roberta-wwm-ext-large features (``hidden_states[-3]``) and G2PW's BERT-base (from its ONNX
graph, ``mizuoto.engines.gptsovits.text.g2pw``).

Parameter names are ``BertModel``'s (``embeddings.word_embeddings.weight``, ``encoder.layer.N.attention.self.query.weight``,
…), so a Hugging Face checkpoint loads once its ``bert.`` prefix is stripped (``load_bert``). The forward pass keeps
transformers' operation order:

- embeddings: ``word + token_type``, then ``+= position``, then LayerNorm (eps from the config);
- attention ``"eager"``: ``matmul(q, kᵀ) / sqrt(d)`` plus the additive mask ``(1 - mask) * finfo(dtype).min`` (always
  added, as transformers makes an all-ones mask when none is given), softmax, ``matmul(p, v)``;
- attention ``"sdpa"`` (transformers' default for BERT since 4.41): ``scaled_dot_product_attention`` with no mask when
  every position is attended (``_prepare_4d_attention_mask_for_sdpa`` returns ``None`` then), else a 4-D mask of 0 and
  ``finfo(dtype).min``;
- GELU with ``erf``; residual LayerNorms after attention and after the feed-forward block. No dropout.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, Literal

import torch
from torch import nn
from torch.nn import functional as F

from mizuoto.runtime import accel
from mizuoto.runtime import ops as default_ops

SITE = accel.Site("bert")

Attention = Literal["eager", "sdpa"]


@dataclass(frozen=True)
class BertConfig:
    vocab_size: int = 21128
    hidden_size: int = 768
    num_hidden_layers: int = 12
    num_attention_heads: int = 12
    intermediate_size: int = 3072
    max_position_embeddings: int = 512
    type_vocab_size: int = 2
    layer_norm_eps: float = 1e-12
    hidden_act: str = "gelu"
    pad_token_id: int = 0

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> BertConfig:
        known = {f.name for f in fields(cls)}
        config = cls(**{k: v for k, v in data.items() if k in known})
        if config.hidden_act != "gelu":
            raise ValueError(f"Unsupported BERT activation {config.hidden_act!r}")
        if data.get("position_embedding_type", "absolute") != "absolute":
            raise ValueError(f"Unsupported BERT position embedding {data['position_embedding_type']!r}")
        return config

    @classmethod
    def from_file(cls, path: str | Path) -> BertConfig:
        """A Hugging Face ``config.json``."""
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


class BertEmbeddings(nn.Module):
    def __init__(self, config: BertConfig, ops: Any) -> None:
        super().__init__()
        self.word_embeddings = ops.Embedding(config.vocab_size, config.hidden_size)
        self.position_embeddings = ops.Embedding(config.max_position_embeddings, config.hidden_size)
        self.token_type_embeddings = ops.Embedding(config.type_vocab_size, config.hidden_size)
        self.LayerNorm = ops.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(self, input_ids: torch.Tensor, token_type_ids: torch.Tensor, position_ids: torch.Tensor) -> torch.Tensor:
        embeddings = self.word_embeddings(input_ids) + self.token_type_embeddings(token_type_ids)
        embeddings += self.position_embeddings(position_ids)
        return self.LayerNorm(embeddings)


class BertSelfAttention(nn.Module):
    def __init__(self, config: BertConfig, ops: Any) -> None:
        super().__init__()
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = config.hidden_size // config.num_attention_heads
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.query = ops.Linear(config.hidden_size, self.all_head_size)
        self.key = ops.Linear(config.hidden_size, self.all_head_size)
        self.value = ops.Linear(config.hidden_size, self.all_head_size)

    def _heads(self, x: torch.Tensor) -> torch.Tensor:
        return x.view(x.shape[0], -1, self.num_attention_heads, self.attention_head_size).transpose(1, 2)

    def forward(self, hidden_states: torch.Tensor, mask: torch.Tensor | None, attention: Attention) -> torch.Tensor:
        batch, length, _ = hidden_states.shape
        query = self._heads(self.query(hidden_states))
        key = self._heads(self.key(hidden_states))
        value = self._heads(self.value(hidden_states))
        if attention == "sdpa":
            out = accel.attention(query, key, value, site=SITE, mask=mask)
            return out.transpose(1, 2).reshape(batch, length, self.all_head_size)
        scores = torch.matmul(query, key.transpose(-1, -2))
        scores = scores / math.sqrt(self.attention_head_size)
        if mask is not None:
            scores = scores + mask
        probs = F.softmax(scores, dim=-1)
        context = torch.matmul(probs, value).permute(0, 2, 1, 3).contiguous()
        return context.view(*context.shape[:-2], self.all_head_size)


class BertSelfOutput(nn.Module):
    def __init__(self, config: BertConfig, ops: Any) -> None:
        super().__init__()
        self.dense = ops.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = ops.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        return self.LayerNorm(self.dense(hidden_states) + input_tensor)


class BertAttention(nn.Module):
    def __init__(self, config: BertConfig, ops: Any) -> None:
        super().__init__()
        self.self = BertSelfAttention(config, ops)
        self.output = BertSelfOutput(config, ops)

    def forward(self, hidden_states: torch.Tensor, mask: torch.Tensor | None, attention: Attention) -> torch.Tensor:
        return self.output(self.self(hidden_states, mask, attention), hidden_states)


class BertIntermediate(nn.Module):
    def __init__(self, config: BertConfig, ops: Any) -> None:
        super().__init__()
        self.dense = ops.Linear(config.hidden_size, config.intermediate_size)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return F.gelu(self.dense(hidden_states))


class BertOutput(nn.Module):
    def __init__(self, config: BertConfig, ops: Any) -> None:
        super().__init__()
        self.dense = ops.Linear(config.intermediate_size, config.hidden_size)
        self.LayerNorm = ops.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        return self.LayerNorm(self.dense(hidden_states) + input_tensor)


class BertLayer(nn.Module):
    def __init__(self, config: BertConfig, ops: Any) -> None:
        super().__init__()
        self.attention = BertAttention(config, ops)
        self.intermediate = BertIntermediate(config, ops)
        self.output = BertOutput(config, ops)

    def forward(self, hidden_states: torch.Tensor, mask: torch.Tensor | None, attention: Attention) -> torch.Tensor:
        attention_output = self.attention(hidden_states, mask, attention)
        return self.output(self.intermediate(attention_output), attention_output)


class BertEncoder(nn.Module):
    def __init__(self, config: BertConfig, ops: Any) -> None:
        super().__init__()
        self.layer = nn.ModuleList([BertLayer(config, ops) for _ in range(config.num_hidden_layers)])


class BertModel(nn.Module):
    """``transformers.BertModel`` without the pooler: ``forward`` gives the last hidden state, ``hidden_states`` all of them."""

    def __init__(self, config: BertConfig, attention: Attention = "eager", ops: Any = default_ops.default) -> None:
        super().__init__()
        if attention not in ("eager", "sdpa"):
            raise ValueError(f"Unknown attention {attention!r}")
        self.config = config
        self.attention = attention
        self.embeddings = BertEmbeddings(config, ops)
        self.encoder = BertEncoder(config, ops)

    def _mask(self, attention_mask: torch.Tensor | None, batch: int, length: int, dtype: torch.dtype, device: torch.device) -> torch.Tensor | None:
        if self.attention == "sdpa":
            if attention_mask is None or bool(torch.all(attention_mask == 1)):
                return None
            expanded = attention_mask[:, None, None, :].expand(batch, 1, length, attention_mask.shape[-1]).to(dtype)
            inverted = 1.0 - expanded
            return inverted.masked_fill(inverted.to(torch.bool), torch.finfo(dtype).min)
        if attention_mask is None:
            attention_mask = torch.ones((batch, length), device=device)
        extended = attention_mask[:, None, None, :].to(dtype=dtype)
        return (1.0 - extended) * torch.finfo(dtype).min

    def hidden_states(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        token_type_ids: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
    ) -> list[torch.Tensor]:
        """The embeddings' output and every layer's, as transformers' ``output_hidden_states=True`` (layers + 1 tensors)."""
        batch, length = input_ids.shape
        device = input_ids.device
        if token_type_ids is None:
            token_type_ids = torch.zeros((batch, length), dtype=torch.long, device=device)
        if position_ids is None:
            position_ids = torch.arange(length, device=device).unsqueeze(0)
        hidden = self.embeddings(input_ids, token_type_ids, position_ids)
        mask = self._mask(attention_mask, batch, length, hidden.dtype, device)
        states = [hidden]
        for layer in self.encoder.layer:
            hidden = layer(hidden, mask, self.attention)
            states.append(hidden)
        return states

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        token_type_ids: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.hidden_states(input_ids, attention_mask, token_type_ids, position_ids)[-1]


BERT_IGNORED = ("cls.*", "pooler.*", "embeddings.position_ids", "embeddings.token_type_ids")
"""Checkpoint keys ``BertModel`` has no use for: the masked-LM head, the pooler, transformers' buffers."""


def load_bert(
    folder: str | Path,
    *,
    attention: Attention = "eager",
    device: str | torch.device = "cpu",
    dtype: torch.dtype = torch.float32,
    ops: Any = default_ops.default,
) -> BertModel:
    """A Hugging Face BERT folder (``config.json`` and ``model.safetensors`` or ``pytorch_model.bin``), ``bert.`` prefix or not."""
    from mizuoto.runtime.loading import load_into, load_state_dict, strip_prefix

    folder = Path(folder)
    config = BertConfig.from_file(folder / "config.json")
    weights = folder / "model.safetensors"
    if not weights.is_file():
        weights = folder / "pytorch_model.bin"
    sd = strip_prefix(load_state_dict(weights), "bert.")
    model = BertModel(config, attention=attention, ops=ops)
    load_into(model, sd, ignore=BERT_IGNORED, what=f"BERT ({folder.name})")
    return model.to(device=device, dtype=dtype).eval()
