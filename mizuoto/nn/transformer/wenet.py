"""WeNet's encoders as CosyVoice carries them (``cosyvoice/transformer``, Apache-2.0): Conformer and Transformer
encoders with ESPnet's relative positional encoding, and CosyVoice 2's upsampling Conformer.

Parameter names are upstream's (``embed.out.0``, ``encoders.N.self_attn.linear_q``, ``pos_bias_u``…).
Attention is computed as upstream writes it (matmul, masked softmax, matmul), not with SDPA, so results match
operation for operation. Dropout is left out (inference only). ``forward_chunk`` keeps the key/value cache
as upstream does, for CosyVoice 1's language model.
"""

from __future__ import annotations

import math
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from mizuoto.runtime import ops as default_ops

ACTIVATIONS: dict[str, type[nn.Module]] = {"relu": nn.ReLU, "swish": nn.SiLU, "gelu": nn.GELU, "tanh": nn.Tanh}


def make_pad_mask(lengths: torch.Tensor, max_len: int = 0) -> torch.Tensor:
    """True where padded."""
    batch = lengths.size(0)
    max_len = max_len if max_len > 0 else int(lengths.max().item())
    seq = torch.arange(0, max_len, dtype=torch.int64, device=lengths.device).unsqueeze(0).expand(batch, max_len)
    return seq >= lengths.unsqueeze(-1)


def subsequent_chunk_mask(size: int, chunk_size: int, device: torch.device | str = "cpu") -> torch.Tensor:
    pos = torch.arange(size, device=device)
    block = (torch.div(pos, chunk_size, rounding_mode="trunc") + 1) * chunk_size
    return pos.unsqueeze(0) < block.unsqueeze(1)


def add_optional_chunk_mask(xs: torch.Tensor, masks: torch.Tensor, static_chunk_size: int) -> torch.Tensor:
    """``add_optional_chunk_mask`` without dynamic chunks (inference): a static chunk mask, or the padding mask."""
    if static_chunk_size > 0:
        chunk_masks = masks & subsequent_chunk_mask(xs.size(1), static_chunk_size, xs.device).unsqueeze(0)
    else:
        chunk_masks = masks
    if torch.compiler.is_compiling():  # the same mask without the data-dependent branch, which would break the graph
        return chunk_masks | (chunk_masks.sum(dim=-1, keepdim=True) == 0)
    if (chunk_masks.sum(dim=-1) == 0).sum().item() != 0:
        chunk_masks[chunk_masks.sum(dim=-1) == 0] = True
    return chunk_masks


class EspnetRelPositionalEncoding(nn.Module):
    """Positions from ``-(n-1)`` to ``n-1``, extended (and kept) as longer inputs come."""

    def __init__(self, d_model: int, max_len: int = 5000) -> None:
        super().__init__()
        self.d_model = d_model
        self.xscale = math.sqrt(d_model)
        self.pe: torch.Tensor | None = None
        self.extend_pe(torch.tensor(0.0).expand(1, max_len))

    def extend_pe(self, x: torch.Tensor) -> None:
        if self.pe is not None and self.pe.size(1) >= x.size(1) * 2 - 1:
            if self.pe.dtype != x.dtype or self.pe.device != x.device:
                self.pe = self.pe.to(dtype=x.dtype, device=x.device)
            return
        pos = torch.zeros(x.size(1), self.d_model)
        neg = torch.zeros(x.size(1), self.d_model)
        position = torch.arange(0, x.size(1), dtype=torch.float32).unsqueeze(1)
        div = torch.exp(torch.arange(0, self.d_model, 2, dtype=torch.float32) * -(math.log(10000.0) / self.d_model))
        pos[:, 0::2] = torch.sin(position * div)
        pos[:, 1::2] = torch.cos(position * div)
        neg[:, 0::2] = torch.sin(-1 * position * div)
        neg[:, 1::2] = torch.cos(-1 * position * div)
        pe = torch.cat([torch.flip(pos, [0]).unsqueeze(0), neg[1:].unsqueeze(0)], dim=1)
        self.pe = pe.to(device=x.device, dtype=x.dtype)

    def forward(self, x: torch.Tensor, offset: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
        self.extend_pe(x)
        return x * self.xscale, self.position_encoding(offset, x.size(1))

    def position_encoding(self, offset: int, size: int) -> torch.Tensor:
        assert self.pe is not None
        half = self.pe.size(1) // 2
        return self.pe[:, half - size - offset + 1 : half + size + offset]


class LinearNoSubsampling(nn.Module):
    """``linear``: Linear + LayerNorm (+ ReLU for ``linear_legacy``), then the positional encoding."""

    def __init__(self, idim: int, odim: int, legacy: bool, ops: Any) -> None:
        super().__init__()
        layers: list[nn.Module] = [ops.Linear(idim, odim), ops.LayerNorm(odim, eps=1e-5), nn.Dropout(0.0)]
        if legacy:
            layers.append(nn.ReLU())
        self.out = nn.Sequential(*layers)
        self.pos_enc = EspnetRelPositionalEncoding(odim)

    def forward(self, x: torch.Tensor, offset: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
        return self.pos_enc(self.out(x), offset)

    def position_encoding(self, offset: int, size: int) -> torch.Tensor:
        return self.pos_enc.position_encoding(offset, size)


class RelPositionMultiHeadedAttention(nn.Module):
    def __init__(self, n_head: int, n_feat: int, ops: Any, key_bias: bool = True, relative: bool = True) -> None:
        super().__init__()
        self.d_k = n_feat // n_head
        self.h = n_head
        self.relative = relative
        self.linear_q = ops.Linear(n_feat, n_feat)
        self.linear_k = ops.Linear(n_feat, n_feat, bias=key_bias)
        self.linear_v = ops.Linear(n_feat, n_feat)
        self.linear_out = ops.Linear(n_feat, n_feat)
        if relative:
            self.linear_pos = ops.Linear(n_feat, n_feat, bias=False)
            self.pos_bias_u = nn.Parameter(torch.empty(self.h, self.d_k))
            self.pos_bias_v = nn.Parameter(torch.empty(self.h, self.d_k))

    @staticmethod
    def rel_shift(x: torch.Tensor) -> torch.Tensor:
        zero_pad = torch.zeros((x.size()[0], x.size()[1], x.size()[2], 1), device=x.device, dtype=x.dtype)
        padded = torch.cat([zero_pad, x], dim=-1).view(x.size()[0], x.size()[1], x.size(3) + 1, x.size(2))
        return padded[:, :, 1:].view_as(x)[:, :, :, : x.size(-1) // 2 + 1]

    def _attend(self, v: torch.Tensor, scores: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        n_batch = v.size(0)
        if mask.size(2) > 0:
            m = mask.unsqueeze(1).eq(0)[:, :, :, : scores.size(-1)]
            scores = scores.masked_fill(m, -float("inf"))
            attn = torch.softmax(scores, dim=-1).masked_fill(m, 0.0)
        else:
            attn = torch.softmax(scores, dim=-1)
        x = torch.matmul(attn, v)
        return self.linear_out(x.transpose(1, 2).contiguous().view(n_batch, -1, self.h * self.d_k))

    def forward(self, x: torch.Tensor, mask: torch.Tensor, pos_emb: torch.Tensor, cache: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        n_batch = x.size(0)
        q = self.linear_q(x).view(n_batch, -1, self.h, self.d_k).transpose(1, 2)
        k = self.linear_k(x).view(n_batch, -1, self.h, self.d_k).transpose(1, 2)
        v = self.linear_v(x).view(n_batch, -1, self.h, self.d_k).transpose(1, 2)
        if cache is not None and cache.size(0) > 0:
            key_cache, value_cache = torch.split(cache, cache.size(-1) // 2, dim=-1)
            k = torch.cat([key_cache, k], dim=2)
            v = torch.cat([value_cache, v], dim=2)
        new_cache = torch.cat((k, v), dim=-1)
        if not self.relative:
            scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
            return self._attend(v, scores, mask), new_cache
        q = q.transpose(1, 2)
        p = self.linear_pos(pos_emb).view(pos_emb.size(0), -1, self.h, self.d_k).transpose(1, 2)
        q_u = (q + self.pos_bias_u).transpose(1, 2)
        q_v = (q + self.pos_bias_v).transpose(1, 2)
        ac = torch.matmul(q_u, k.transpose(-2, -1))
        bd = torch.matmul(q_v, p.transpose(-2, -1))
        if ac.shape != bd.shape:
            bd = self.rel_shift(bd)
        scores = (ac + bd) / math.sqrt(self.d_k)
        return self._attend(v, scores, mask), new_cache


class PositionwiseFeedForward(nn.Module):
    def __init__(self, idim: int, hidden: int, activation: nn.Module, ops: Any) -> None:
        super().__init__()
        self.w_1 = ops.Linear(idim, hidden)
        self.activation = activation
        self.w_2 = ops.Linear(hidden, idim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w_2(self.activation(self.w_1(x)))


class TransformerEncoderLayer(nn.Module):
    def __init__(self, size: int, self_attn: nn.Module, feed_forward: nn.Module, normalize_before: bool, ops: Any) -> None:
        super().__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.norm1 = ops.LayerNorm(size, eps=1e-12)
        self.norm2 = ops.LayerNorm(size, eps=1e-12)
        self.normalize_before = normalize_before

    def forward(self, x: torch.Tensor, mask: torch.Tensor, pos_emb: torch.Tensor, cache: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        residual = x
        if self.normalize_before:
            x = self.norm1(x)
        x_att, new_cache = self.self_attn(x, mask, pos_emb, cache)
        x = residual + x_att
        if not self.normalize_before:
            x = self.norm1(x)
        residual = x
        if self.normalize_before:
            x = self.norm2(x)
        x = residual + self.feed_forward(x)
        if not self.normalize_before:
            x = self.norm2(x)
        return x, new_cache


class ConformerEncoderLayer(nn.Module):
    """Without macaron feed-forward or convolution module (CosyVoice uses neither)."""

    def __init__(self, size: int, self_attn: nn.Module, feed_forward: nn.Module, normalize_before: bool, ops: Any) -> None:
        super().__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.norm_ff = ops.LayerNorm(size, eps=1e-12)
        self.norm_mha = ops.LayerNorm(size, eps=1e-12)
        self.normalize_before = normalize_before

    def forward(self, x: torch.Tensor, mask: torch.Tensor, pos_emb: torch.Tensor, cache: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        residual = x
        if self.normalize_before:
            x = self.norm_mha(x)
        x_att, new_cache = self.self_attn(x, mask, pos_emb, cache)
        x = residual + x_att
        if not self.normalize_before:
            x = self.norm_mha(x)
        residual = x
        if self.normalize_before:
            x = self.norm_ff(x)
        x = residual + 1.0 * self.feed_forward(x)
        if not self.normalize_before:
            x = self.norm_ff(x)
        return x, new_cache


def _layers(kind: str, n: int, size: int, heads: int, units: int, activation: str, normalize_before: bool, ops: Any, key_bias: bool = True) -> nn.ModuleList:
    layer = ConformerEncoderLayer if kind == "conformer" else TransformerEncoderLayer
    return nn.ModuleList(
        layer(size, RelPositionMultiHeadedAttention(heads, size, ops, key_bias), PositionwiseFeedForward(size, units, ACTIVATIONS[activation](), ops), normalize_before, ops)
        for _ in range(n)
    )


class Encoder(nn.Module):
    """``ConformerEncoder`` (``kind="conformer"``, Swish) or ``TransformerEncoder`` (``kind="transformer"``, ReLU), relative positions."""

    def __init__(
        self,
        input_size: int,
        output_size: int,
        attention_heads: int,
        linear_units: int,
        num_blocks: int,
        kind: str = "conformer",
        input_layer: str = "linear",
        normalize_before: bool = True,
        static_chunk_size: int = 0,
        activation: str | None = None,
        ops: Any = default_ops.default,
    ) -> None:
        super().__init__()
        self._output_size = output_size
        self.embed = LinearNoSubsampling(input_size, output_size, input_layer == "linear_legacy", ops)
        self.normalize_before = normalize_before
        self.after_norm = ops.LayerNorm(output_size, eps=1e-5)
        self.static_chunk_size = static_chunk_size
        act = activation or ("swish" if kind == "conformer" else "relu")
        self.encoders = _layers(kind, num_blocks, output_size, attention_heads, linear_units, act, normalize_before, ops)

    def output_size(self) -> int:
        return self._output_size

    def forward(self, xs: torch.Tensor, xs_lens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        masks = ~make_pad_mask(xs_lens, xs.size(1)).unsqueeze(1)
        xs, pos_emb = self.embed(xs)
        chunk_masks = add_optional_chunk_mask(xs, masks, self.static_chunk_size)
        for layer in self.encoders:
            xs, _ = layer(xs, chunk_masks, pos_emb)
        if self.normalize_before:
            xs = self.after_norm(xs)
        return xs, masks

    def forward_chunk(self, xs: torch.Tensor, offset: int, att_cache: torch.Tensor, att_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """WeNet's ``forward_chunk`` with ``required_cache_size=-1`` (keep everything) and no convolution cache."""
        xs, _ = self.embed(xs, offset)
        elayers, cache_t1 = att_cache.size(0), att_cache.size(2)
        pos_emb = self.embed.position_encoding(offset - cache_t1, cache_t1 + xs.size(1))
        caches = []
        for i, layer in enumerate(self.encoders):
            xs, new_cache = layer(xs, att_mask, pos_emb, att_cache[i : i + 1] if elayers > 0 else att_cache)
            caches.append(new_cache)
        if self.normalize_before:
            xs = self.after_norm(xs)
        return xs, torch.cat(caches, dim=0)


class Upsample1D(nn.Module):
    def __init__(self, channels: int, out_channels: int, stride: int, ops: Any) -> None:
        super().__init__()
        self.stride = stride
        self.conv = ops.Conv1d(channels, out_channels, stride * 2 + 1, stride=1, padding=0)

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = F.interpolate(x, scale_factor=float(self.stride), mode="nearest")
        return self.conv(F.pad(x, (self.stride * 2, 0), value=0.0)), lengths * self.stride


class PreLookaheadLayer(nn.Module):
    """Two causal convolutions that let each token see ``pre_lookahead_len`` tokens ahead (padded, or ``context``)."""

    def __init__(self, in_channels: int, channels: int, pre_lookahead_len: int, ops: Any) -> None:
        super().__init__()
        self.pre_lookahead_len = pre_lookahead_len
        self.conv1 = ops.Conv1d(in_channels, channels, kernel_size=pre_lookahead_len + 1, stride=1, padding=0)
        self.conv2 = ops.Conv1d(channels, in_channels, kernel_size=3, stride=1, padding=0)

    def forward(self, inputs: torch.Tensor, context: torch.Tensor | None = None) -> torch.Tensor:
        out = inputs.transpose(1, 2).contiguous()
        if context is None or context.size(1) == 0:
            out = F.pad(out, (0, self.pre_lookahead_len), mode="constant", value=0.0)
        else:
            ctx = context.transpose(1, 2).contiguous()
            out = F.pad(torch.concat([out, ctx], dim=2), (0, self.pre_lookahead_len - ctx.size(2)), mode="constant", value=0.0)
        out = F.leaky_relu(self.conv1(out))
        out = self.conv2(F.pad(out, (self.conv2.kernel_size[0] - 1, 0), mode="constant", value=0.0))
        return out.transpose(1, 2).contiguous() + inputs


class UpsampleConformerEncoder(nn.Module):
    """CosyVoice 2's flow encoder: pre-lookahead, 6 Conformer layers, 2× upsampling, 4 more layers."""

    def __init__(
        self,
        input_size: int = 512,
        output_size: int = 512,
        attention_heads: int = 8,
        linear_units: int = 2048,
        num_blocks: int = 6,
        static_chunk_size: int = 25,
        ops: Any = default_ops.default,
    ) -> None:
        super().__init__()
        self._output_size = output_size
        self.embed = LinearNoSubsampling(input_size, output_size, False, ops)
        self.after_norm = ops.LayerNorm(output_size, eps=1e-5)
        self.static_chunk_size = static_chunk_size
        self.pre_lookahead_layer = PreLookaheadLayer(output_size, output_size, 3, ops)  # upstream writes 512, its only size
        self.encoders = _layers("conformer", num_blocks, output_size, attention_heads, linear_units, "swish", True, ops)
        self.up_layer = Upsample1D(output_size, output_size, 2, ops)
        self.up_embed = LinearNoSubsampling(input_size, output_size, False, ops)
        self.up_encoders = _layers("conformer", 4, output_size, attention_heads, linear_units, "swish", True, ops)

    def output_size(self) -> int:
        return self._output_size

    def forward(self, xs: torch.Tensor, xs_lens: torch.Tensor, context: torch.Tensor | None = None, streaming: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
        masks = ~make_pad_mask(xs_lens, xs.size(1)).unsqueeze(1)
        xs, pos_emb = self.embed(xs)
        if context is not None and context.size(1) != 0:
            context, _ = self.embed(context, offset=xs.size(1))
        chunk_masks = add_optional_chunk_mask(xs, masks, self.static_chunk_size if streaming else 0)
        xs = self.pre_lookahead_layer(xs, context=context)
        for layer in self.encoders:
            xs, _ = layer(xs, chunk_masks, pos_emb)
        xs, xs_lens = self.up_layer(xs.transpose(1, 2).contiguous(), xs_lens)
        xs = xs.transpose(1, 2).contiguous()
        masks = ~make_pad_mask(xs_lens, xs.size(1)).unsqueeze(1)
        xs, pos_emb = self.up_embed(xs)
        chunk_masks = add_optional_chunk_mask(xs, masks, self.static_chunk_size * self.up_layer.stride if streaming else 0)
        for layer in self.up_encoders:
            xs, _ = layer(xs, chunk_masks, pos_emb)
        return self.after_norm(xs), masks
