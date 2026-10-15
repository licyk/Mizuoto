"""w2v-BERT 2.0 (``Wav2Vec2BertModel``, transformers 4.52, Apache-2.0) for its hidden states.

A Conformer: feature projection (LayerNorm + Linear over 160 stacked fbank features), then layers of
half-step FFN, self-attention with ``relative_key`` distance embeddings (clamped to [-64, 8]), a causal
depthwise convolution module (GLU, kernel 31, left padding), a second half-step FFN and a final LayerNorm.
Padded steps are zeroed and masked with the dtype's minimum, as upstream does.

``hidden_states(features, mask, n)`` returns upstream's ``output_hidden_states[n]``: the input of layer ``n``,
so only the first ``n`` layers run (and only they need to be built: ``layers``). Upstream draws one
``torch.rand([])`` per layer for layer drop even in eval mode; the draws never change the output and are not
made here.
"""

from __future__ import annotations

import math
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from mizuoto.runtime import ops as default_ops


class FeedForward(nn.Module):
    def __init__(self, hidden: int, intermediate: int, ops: Any) -> None:
        super().__init__()
        self.intermediate_dense = ops.Linear(hidden, intermediate)
        self.output_dense = ops.Linear(intermediate, hidden)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.output_dense(F.silu(self.intermediate_dense(x)))


class ConvolutionModule(nn.Module):
    def __init__(self, hidden: int, kernel_size: int, eps: float, ops: Any) -> None:
        super().__init__()
        self.layer_norm = ops.LayerNorm(hidden, eps=eps)
        self.pointwise_conv1 = ops.Conv1d(hidden, 2 * hidden, kernel_size=1, bias=False)
        self.depthwise_conv = ops.Conv1d(hidden, hidden, kernel_size, groups=hidden, bias=False)
        self.depthwise_layer_norm = ops.LayerNorm(hidden, eps=eps)
        self.pointwise_conv2 = ops.Conv1d(hidden, hidden, kernel_size=1, bias=False)
        self.kernel_size = kernel_size

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
        x = self.layer_norm(x)
        if mask is not None:
            x = x.masked_fill(~mask.bool().unsqueeze(-1), 0.0)
        x = F.glu(self.pointwise_conv1(x.transpose(1, 2)), dim=1)
        x = self.depthwise_conv(F.pad(x, (self.kernel_size - 1, 0)))
        x = F.silu(self.depthwise_layer_norm(x.transpose(1, 2)).transpose(1, 2))
        return self.pointwise_conv2(x).transpose(1, 2)


class SelfAttention(nn.Module):
    def __init__(self, hidden: int, heads: int, left: int, right: int, ops: Any) -> None:
        super().__init__()
        self.head_size = hidden // heads
        self.num_heads = heads
        self.linear_q = ops.Linear(hidden, hidden)
        self.linear_k = ops.Linear(hidden, hidden)
        self.linear_v = ops.Linear(hidden, hidden)
        self.linear_out = ops.Linear(hidden, hidden)
        self.left, self.right = left, right
        self.distance_embedding = ops.Embedding(left + right + 1, self.head_size)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
        b = x.size(0)
        q = self.linear_q(x).view(b, -1, self.num_heads, self.head_size).transpose(1, 2)
        k = self.linear_k(x).view(b, -1, self.num_heads, self.head_size).transpose(1, 2)
        v = self.linear_v(x).view(b, -1, self.num_heads, self.head_size).transpose(1, 2)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_size)
        lq, lk = q.shape[2], k.shape[2]
        left_ids = torch.arange(lq, dtype=torch.long, device=x.device).view(-1, 1)
        right_ids = torch.arange(lk, dtype=torch.long, device=x.device).view(1, -1)
        distance = torch.clamp(right_ids - left_ids, -self.left, self.right)
        positional = self.distance_embedding(distance + self.left).to(dtype=q.dtype)
        scores = scores + (torch.einsum("bhld,lrd->bhlr", q, positional) / math.sqrt(self.head_size))
        if mask is not None:
            scores = scores + mask
        probs = torch.softmax(scores, dim=-1)
        out = torch.matmul(probs, v).transpose(1, 2).reshape(b, -1, self.num_heads * self.head_size)
        return self.linear_out(out)


class EncoderLayer(nn.Module):
    def __init__(self, c: dict[str, Any], ops: Any) -> None:
        super().__init__()
        hidden, eps = c["hidden_size"], c["layer_norm_eps"]
        self.ffn1_layer_norm = ops.LayerNorm(hidden, eps=eps)
        self.ffn1 = FeedForward(hidden, c["intermediate_size"], ops)
        self.self_attn_layer_norm = ops.LayerNorm(hidden, eps=eps)
        self.self_attn = SelfAttention(hidden, c["num_attention_heads"], c["left_max_position_embeddings"], c["right_max_position_embeddings"], ops)
        self.conv_module = ConvolutionModule(hidden, c["conv_depthwise_kernel_size"], eps, ops)
        self.ffn2_layer_norm = ops.LayerNorm(hidden, eps=eps)
        self.ffn2 = FeedForward(hidden, c["intermediate_size"], ops)
        self.final_layer_norm = ops.LayerNorm(hidden, eps=eps)

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor | None, conv_mask: torch.Tensor | None) -> torch.Tensor:
        x = self.ffn1(self.ffn1_layer_norm(x)) * 0.5 + x
        x = self.self_attn(self.self_attn_layer_norm(x), attn_mask) + x
        x = x + self.conv_module(x, conv_mask)
        x = self.ffn2(self.ffn2_layer_norm(x)) * 0.5 + x
        return self.final_layer_norm(x)


class W2VBert(nn.Module):
    """Parameter names are ``Wav2Vec2BertModel``'s (``feature_projection.*``, ``encoder.layers.N.*``)."""

    def __init__(self, config: dict[str, Any], layers: int | None = None, ops: Any = default_ops.default) -> None:
        super().__init__()
        if config.get("position_embeddings_type") != "relative_key" or config.get("add_adapter") or config.get("use_intermediate_ffn_before_adapter"):
            raise ValueError("Only w2v-BERT 2.0's configuration is supported (relative_key positions, no adapter)")
        self.feature_projection = nn.Module()
        self.feature_projection.add_module("layer_norm", ops.LayerNorm(config["feature_projection_input_dim"], eps=config["layer_norm_eps"]))
        self.feature_projection.add_module("projection", ops.Linear(config["feature_projection_input_dim"], config["hidden_size"]))
        self.encoder = nn.Module()
        n = config["num_hidden_layers"] if layers is None else layers
        self.encoder.add_module("layers", nn.ModuleList(EncoderLayer(config, ops) for _ in range(n)))

    @staticmethod
    def ignored(config: dict[str, Any], layers: int | None) -> list[str]:
        """Checkpoint keys a model of ``layers`` layers does not use."""
        out = ["masked_spec_embed"]
        if layers is not None:
            out += [f"encoder.layers.{i}.*" for i in range(layers, config["num_hidden_layers"])]
        return out

    def hidden_states(self, features: torch.Tensor, attention_mask: torch.Tensor | None, n: int) -> torch.Tensor:
        """``model(features, attention_mask, output_hidden_states=True).hidden_states[n]``."""
        fp: Any = self.feature_projection
        x = fp.projection(fp.layer_norm(features))
        conv_mask = attention_mask
        attn_mask = None
        if attention_mask is not None:
            x = x.masked_fill(~attention_mask.bool().unsqueeze(-1), 0.0)
            m = 1.0 - attention_mask[:, None, None, :].to(dtype=x.dtype)
            m = m * torch.finfo(x.dtype).min
            attn_mask = m.expand(m.shape[0], 1, m.shape[-1], m.shape[-1])
        layers: Any = self.encoder.get_submodule("layers")
        if n > len(layers):
            raise ValueError(f"Hidden state {n} needs {n} layers; {len(layers)} are built")
        for layer in list(layers)[:n]:
            x = layer(x, attn_mask, conv_mask)
        return x
