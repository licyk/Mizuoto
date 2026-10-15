"""VITS' text-encoder transformer (``attentions.py``, ``modules.py``, ``commons.py`` of VITS and its descendants:
GPT-SoVITS, MIT).

- ``LayerNorm`` over the channel axis of (batch, channels, time), with upstream's ``gamma``/``beta`` names.
- ``MultiHeadAttention``: 1×1 convolutions for q, k, v and the output; relative-position logits and values
  within ``window_size`` (heads sharing one table), the mask filled with -1e4.
- ``FFN``: two convolutions with "same" (or causal) padding, ReLU or the sigmoid GELU.
- ``Encoder``: post-norm layers of attention and FFN.

Every operation keeps upstream's shapes and order, so float32 results are identical. Dropout is inactive in
inference and left out. The relative tables are allocated empty (upstream draws them at init).
"""

from __future__ import annotations

import math
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from mizuoto.runtime import ops as default_ops


def sequence_mask(length: torch.Tensor, max_length: int | None = None) -> torch.Tensor:
    if max_length is None:
        max_length = int(length.max())
    x = torch.arange(max_length, dtype=length.dtype, device=length.device)
    return x.unsqueeze(0) < length.unsqueeze(1)


def convert_pad_shape(pad_shape: list[list[int]]) -> list[int]:
    return [item for sublist in pad_shape[::-1] for item in sublist]


class LayerNorm(nn.Module):
    """Layer norm over dim 1 of (batch, channels, time)."""

    def __init__(self, channels: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.channels = channels
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(channels))
        self.beta = nn.Parameter(torch.zeros(channels))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(1, -1)
        x = F.layer_norm(x, (self.channels,), self.gamma, self.beta, self.eps)
        return x.transpose(1, -1)


class MultiHeadAttention(nn.Module):
    def __init__(
        self,
        channels: int,
        out_channels: int,
        n_heads: int,
        window_size: int | None = None,
        heads_share: bool = True,
        block_length: int | None = None,
        proximal_bias: bool = False,
        ops: Any = default_ops.default,
    ) -> None:
        super().__init__()
        assert channels % n_heads == 0
        self.channels = channels
        self.out_channels = out_channels
        self.n_heads = n_heads
        self.window_size = window_size
        self.block_length = block_length
        self.proximal_bias = proximal_bias
        self.k_channels = channels // n_heads
        self.conv_q = ops.Conv1d(channels, channels, 1)
        self.conv_k = ops.Conv1d(channels, channels, 1)
        self.conv_v = ops.Conv1d(channels, channels, 1)
        self.conv_o = ops.Conv1d(channels, out_channels, 1)
        if window_size is not None:
            n_heads_rel = 1 if heads_share else n_heads
            self.emb_rel_k = nn.Parameter(torch.empty(n_heads_rel, window_size * 2 + 1, self.k_channels))
            self.emb_rel_v = nn.Parameter(torch.empty(n_heads_rel, window_size * 2 + 1, self.k_channels))

    def forward(self, x: torch.Tensor, c: torch.Tensor, attn_mask: torch.Tensor | None = None) -> torch.Tensor:
        q = self.conv_q(x)
        k = self.conv_k(c)
        v = self.conv_v(c)
        x = self.attention(q, k, v, mask=attn_mask)
        return self.conv_o(x)

    def attention(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        b, d, t_s, t_t = (*key.size(), query.size(2))
        query = query.view(b, self.n_heads, self.k_channels, t_t).transpose(2, 3)
        key = key.view(b, self.n_heads, self.k_channels, t_s).transpose(2, 3)
        value = value.view(b, self.n_heads, self.k_channels, t_s).transpose(2, 3)

        scores = torch.matmul(query / math.sqrt(self.k_channels), key.transpose(-2, -1))
        if self.window_size is not None:
            assert t_s == t_t, "Relative attention is only available for self-attention."
            key_relative_embeddings = self._get_relative_embeddings(self.emb_rel_k, t_s)
            rel_logits = torch.matmul(query / math.sqrt(self.k_channels), key_relative_embeddings.unsqueeze(0).transpose(-2, -1))
            scores = scores + self._relative_position_to_absolute_position(rel_logits)
        if self.proximal_bias:
            assert t_s == t_t, "Proximal bias is only available for self-attention."
            scores = scores + self._attention_bias_proximal(t_s).to(device=scores.device, dtype=scores.dtype)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e4)
            if self.block_length is not None:
                assert t_s == t_t, "Local attention is only available for self-attention."
                block_mask = torch.ones_like(scores).triu(-self.block_length).tril(self.block_length)
                scores = scores.masked_fill(block_mask == 0, -1e4)
        p_attn = F.softmax(scores, dim=-1)
        output = torch.matmul(p_attn, value)
        if self.window_size is not None:
            relative_weights = self._absolute_position_to_relative_position(p_attn)
            value_relative_embeddings = self._get_relative_embeddings(self.emb_rel_v, t_s)
            output = output + torch.matmul(relative_weights, value_relative_embeddings.unsqueeze(0))
        return output.transpose(2, 3).contiguous().view(b, d, t_t)

    def _get_relative_embeddings(self, relative_embeddings: torch.Tensor, length: int) -> torch.Tensor:
        assert self.window_size is not None
        pad_length = max(length - (self.window_size + 1), 0)
        slice_start_position = max((self.window_size + 1) - length, 0)
        slice_end_position = slice_start_position + 2 * length - 1
        if pad_length > 0:
            padded = F.pad(relative_embeddings, convert_pad_shape([[0, 0], [pad_length, pad_length], [0, 0]]))
        else:
            padded = relative_embeddings
        return padded[:, slice_start_position:slice_end_position]

    @staticmethod
    def _relative_position_to_absolute_position(x: torch.Tensor) -> torch.Tensor:
        batch, heads, length, _ = x.size()
        x = F.pad(x, convert_pad_shape([[0, 0], [0, 0], [0, 0], [0, 1]]))
        x_flat = x.view([batch, heads, length * 2 * length])
        x_flat = F.pad(x_flat, convert_pad_shape([[0, 0], [0, 0], [0, length - 1]]))
        return x_flat.view([batch, heads, length + 1, 2 * length - 1])[:, :, :length, length - 1 :]

    @staticmethod
    def _absolute_position_to_relative_position(x: torch.Tensor) -> torch.Tensor:
        batch, heads, length, _ = x.size()
        x = F.pad(x, convert_pad_shape([[0, 0], [0, 0], [0, 0], [0, length - 1]]))
        x_flat = x.view([batch, heads, length**2 + length * (length - 1)])
        x_flat = F.pad(x_flat, convert_pad_shape([[0, 0], [0, 0], [length, 0]]))
        return x_flat.view([batch, heads, length, 2 * length])[:, :, :, 1:]

    @staticmethod
    def _attention_bias_proximal(length: int) -> torch.Tensor:
        r = torch.arange(length, dtype=torch.float32)
        diff = torch.unsqueeze(r, 0) - torch.unsqueeze(r, 1)
        return torch.unsqueeze(torch.unsqueeze(-torch.log1p(torch.abs(diff)), 0), 0)


class FFN(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        filter_channels: int,
        kernel_size: int,
        activation: str | None = None,
        causal: bool = False,
        ops: Any = default_ops.default,
    ) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.activation = activation
        self.causal = causal
        self.conv_1 = ops.Conv1d(in_channels, filter_channels, kernel_size)
        self.conv_2 = ops.Conv1d(filter_channels, out_channels, kernel_size)

    def padding(self, x: torch.Tensor) -> torch.Tensor:
        if self.kernel_size == 1:
            return x
        if self.causal:
            pad_l, pad_r = self.kernel_size - 1, 0
        else:
            pad_l, pad_r = (self.kernel_size - 1) // 2, self.kernel_size // 2
        return F.pad(x, convert_pad_shape([[0, 0], [0, 0], [pad_l, pad_r]]))

    def forward(self, x: torch.Tensor, x_mask: torch.Tensor) -> torch.Tensor:
        x = self.conv_1(self.padding(x * x_mask))
        if self.activation == "gelu":
            x = x * torch.sigmoid(1.702 * x)
        else:
            x = torch.relu(x)
        x = self.conv_2(self.padding(x * x_mask))
        return x * x_mask


class Encoder(nn.Module):
    def __init__(
        self,
        hidden_channels: int,
        filter_channels: int,
        n_heads: int,
        n_layers: int,
        kernel_size: int = 1,
        window_size: int | None = 4,
        ops: Any = default_ops.default,
    ) -> None:
        super().__init__()
        self.hidden_channels = hidden_channels
        self.n_layers = n_layers
        self.attn_layers = nn.ModuleList(MultiHeadAttention(hidden_channels, hidden_channels, n_heads, window_size=window_size, ops=ops) for _ in range(n_layers))
        self.norm_layers_1 = nn.ModuleList(LayerNorm(hidden_channels) for _ in range(n_layers))
        self.ffn_layers = nn.ModuleList(FFN(hidden_channels, hidden_channels, filter_channels, kernel_size, ops=ops) for _ in range(n_layers))
        self.norm_layers_2 = nn.ModuleList(LayerNorm(hidden_channels) for _ in range(n_layers))

    def forward(self, x: torch.Tensor, x_mask: torch.Tensor) -> torch.Tensor:
        attn_mask = x_mask.unsqueeze(2) * x_mask.unsqueeze(-1)
        x = x * x_mask
        for i in range(self.n_layers):
            y = self.attn_layers[i](x, x, attn_mask)
            x = self.norm_layers_1[i](x + y)
            y = self.ffn_layers[i](x, x_mask)
            x = self.norm_layers_2[i](x + y)
        return x * x_mask
