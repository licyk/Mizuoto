"""Matcha-TTS's 1-D U-Net vector-field estimator, as CosyVoice 1 and 2 use it (MIT; CosyVoice's causal variant, Apache-2.0).

Down, middle and up blocks of a ResNet block followed by transformer blocks (diffusers'
``BasicTransformerBlock`` with GELU feed-forward), conditioned on a sinusoidal time embedding; the input is
[x | mu | speaker | condition] stacked on channels. ``causal=True`` is CosyVoice 2's
``CausalConditionalDecoder``: left-padded convolutions, LayerNorm blocks and, when streaming, chunked attention.

The attention is diffusers' ``AttnProcessor2_0``: SDPA with an additive mask (0, or -1e10 where masked)
repeated per head. Parameter names are upstream's.
"""

from __future__ import annotations

import math
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from mizuoto.nn.transformer.wenet import add_optional_chunk_mask
from mizuoto.runtime import ops as default_ops
from mizuoto.runtime.accel import Site, attention

SITE = Site("unet1d")


def mask_to_bias(mask: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    return (1.0 - mask.to(dtype)) * -1.0e10


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor, scale: float = 1000) -> torch.Tensor:
        if x.ndim < 1:
            x = x.unsqueeze(0)
        half = self.dim // 2
        emb = math.log(10000) / (half - 1)
        emb = torch.exp(torch.arange(half, device=x.device).float() * -emb)
        emb = scale * x.unsqueeze(1) * emb.unsqueeze(0)
        return torch.cat((emb.sin(), emb.cos()), dim=-1)


class TimestepEmbedding(nn.Module):
    def __init__(self, in_channels: int, time_embed_dim: int, ops: Any) -> None:
        super().__init__()
        self.linear_1 = ops.Linear(in_channels, time_embed_dim)
        self.linear_2 = ops.Linear(time_embed_dim, time_embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear_2(F.silu(self.linear_1(x)))


class Transpose(nn.Module):
    def __init__(self, dim0: int, dim1: int) -> None:
        super().__init__()
        self.dim0, self.dim1 = dim0, dim1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.transpose(x, self.dim0, self.dim1)


class CausalConv1d(nn.Conv1d):
    """Left-padded by ``kernel_size - 1`` (CosyVoice's ``flow.decoder.CausalConv1d``)."""

    reset_parameters = default_ops.Conv1d.reset_parameters

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int) -> None:
        super().__init__(in_channels, out_channels, kernel_size, 1, padding=0)
        self.causal_padding = kernel_size - 1

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # ty: ignore[invalid-method-override]
        return super().forward(F.pad(x, (self.causal_padding, 0), value=0.0))


class Downsample1D(nn.Module):
    def __init__(self, dim: int, ops: Any) -> None:
        super().__init__()
        self.conv = ops.Conv1d(dim, dim, 3, 2, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class Upsample1D(nn.Module):
    def __init__(self, dim: int, ops: Any) -> None:
        super().__init__()
        self.conv = ops.ConvTranspose1d(dim, dim, 4, 2, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class Block1D(nn.Module):
    def __init__(self, dim: int, dim_out: int, causal: bool, ops: Any, groups: int = 8) -> None:
        super().__init__()
        if causal:
            self.block = nn.Sequential(CausalConv1d(dim, dim_out, 3), Transpose(1, 2), ops.LayerNorm(dim_out), Transpose(1, 2), nn.Mish())
        else:
            self.block = nn.Sequential(ops.Conv1d(dim, dim_out, 3, padding=1), ops.GroupNorm(groups, dim_out), nn.Mish())

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return self.block(x * mask) * mask


class ResnetBlock1D(nn.Module):
    def __init__(self, dim: int, dim_out: int, time_emb_dim: int, causal: bool, ops: Any) -> None:
        super().__init__()
        self.mlp = nn.Sequential(nn.Mish(), ops.Linear(time_emb_dim, dim_out))
        self.block1 = Block1D(dim, dim_out, causal, ops)
        self.block2 = Block1D(dim_out, dim_out, causal, ops)
        self.res_conv = ops.Conv1d(dim, dim_out, 1)

    def forward(self, x: torch.Tensor, mask: torch.Tensor, time_emb: torch.Tensor) -> torch.Tensor:
        h = self.block1(x, mask)
        h += self.mlp(time_emb).unsqueeze(-1)
        h = self.block2(h, mask)
        return h + self.res_conv(x * mask)


class Attention(nn.Module):
    """diffusers' ``Attention`` (no bias on q, k, v) with ``AttnProcessor2_0``."""

    def __init__(self, dim: int, heads: int, dim_head: int, ops: Any) -> None:
        super().__init__()
        self.heads = heads
        inner = heads * dim_head
        self.to_q = ops.Linear(dim, inner, bias=False)
        self.to_k = ops.Linear(dim, inner, bias=False)
        self.to_v = ops.Linear(dim, inner, bias=False)
        self.to_out = nn.ModuleList([ops.Linear(inner, dim), nn.Dropout(0.0)])

    def head_mask(self, mask: torch.Tensor, b: int, n: int) -> torch.Tensor:
        """The additive mask as diffusers' processor prepares it: one per head, (b, heads, n, n)."""
        if mask.shape[-1] != n:
            mask = F.pad(mask, (0, n), value=0.0)
        if mask.shape[0] < b * self.heads:
            mask = mask.repeat_interleave(self.heads, dim=0)
        return mask.view(b, self.heads, -1, mask.shape[-1])

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """``mask``: (b, n, n), or already ``head_mask``'s (the blocks of one resolution share it)."""
        b, n, _ = x.shape
        if mask.ndim != 4:
            mask = self.head_mask(mask, b, n)
        q, k, v = self.to_q(x), self.to_k(x), self.to_v(x)
        head_dim = k.shape[-1] // self.heads
        q = q.view(b, -1, self.heads, head_dim).transpose(1, 2)
        k = k.view(b, -1, self.heads, head_dim).transpose(1, 2)
        v = v.view(b, -1, self.heads, head_dim).transpose(1, 2)
        out = attention(q, k, v, site=SITE, mask=mask)
        out = out.transpose(1, 2).reshape(b, -1, self.heads * head_dim).to(q.dtype)
        return self.to_out[0](out) / 1.0


class GELU(nn.Module):
    def __init__(self, dim_in: int, dim_out: int, ops: Any) -> None:
        super().__init__()
        self.proj = ops.Linear(dim_in, dim_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.gelu(self.proj(x), approximate="none")


class FeedForward(nn.Module):
    def __init__(self, dim: int, ops: Any, mult: int = 4) -> None:
        super().__init__()
        inner = int(dim * mult)
        self.net = nn.ModuleList([GELU(dim, inner, ops), nn.Dropout(0.0), ops.Linear(inner, dim)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for module in self.net:
            x = module(x)
        return x


class BasicTransformerBlock(nn.Module):
    def __init__(self, dim: int, heads: int, dim_head: int, ops: Any) -> None:
        super().__init__()
        self.norm1 = ops.LayerNorm(dim, elementwise_affine=True)
        self.attn1 = Attention(dim, heads, dim_head, ops)
        self.norm3 = ops.LayerNorm(dim, elementwise_affine=True)
        self.ff = FeedForward(dim, ops)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        x = self.attn1(self.norm1(x), mask) + x
        return self.ff(self.norm3(x)) + x


class ConditionalDecoder(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        channels: tuple[int, ...] = (256, 256),
        attention_head_dim: int = 64,
        n_blocks: int = 4,
        num_mid_blocks: int = 12,
        num_heads: int = 8,
        causal: bool = False,
        static_chunk_size: int = 50,
        ops: Any = default_ops.default,
    ) -> None:
        super().__init__()
        channels = tuple(channels)
        self.causal = causal
        self.static_chunk_size = static_chunk_size
        self.time_embeddings = SinusoidalPosEmb(in_channels)
        time_dim = channels[0] * 4
        self.time_mlp = TimestepEmbedding(in_channels, time_dim, ops)

        def transformers(dim: int) -> nn.ModuleList:
            return nn.ModuleList(BasicTransformerBlock(dim, num_heads, attention_head_dim, ops) for _ in range(n_blocks))

        def conv3(dim: int) -> nn.Module:
            return CausalConv1d(dim, dim, 3) if causal else ops.Conv1d(dim, dim, 3, padding=1)

        self.down_blocks = nn.ModuleList()
        out_ch = in_channels
        for i, ch in enumerate(channels):
            in_ch, out_ch = out_ch, ch
            last = i == len(channels) - 1
            down = conv3(out_ch) if last else Downsample1D(out_ch, ops)
            self.down_blocks.append(nn.ModuleList([ResnetBlock1D(in_ch, out_ch, time_dim, causal, ops), transformers(out_ch), down]))
        self.mid_blocks = nn.ModuleList(nn.ModuleList([ResnetBlock1D(channels[-1], out_ch, time_dim, causal, ops), transformers(out_ch)]) for _ in range(num_mid_blocks))
        rev = channels[::-1] + (channels[0],)
        self.up_blocks = nn.ModuleList()
        for i in range(len(rev) - 1):
            in_ch, out_ch = rev[i] * 2, rev[i + 1]
            last = i == len(rev) - 2
            up = conv3(out_ch) if last else Upsample1D(out_ch, ops)
            self.up_blocks.append(nn.ModuleList([ResnetBlock1D(in_ch, out_ch, time_dim, causal, ops), transformers(out_ch), up]))
        self.final_block = Block1D(rev[-1], rev[-1], causal, ops)
        self.final_proj = ops.Conv1d(rev[-1], out_channels, 1)

    def _attn_mask(self, x: torch.Tensor, mask: torch.Tensor, streaming: bool) -> torch.Tensor:
        if self.causal and streaming:
            m = add_optional_chunk_mask(x, mask.bool(), self.static_chunk_size)
        else:
            m = add_optional_chunk_mask(x, mask.bool(), 0).repeat(1, x.size(1), 1)
        return mask_to_bias(m, x.dtype)

    def _transformers(self, blocks: nn.ModuleList, x: torch.Tensor, mask: torch.Tensor, streaming: bool, biases: dict[tuple[int, int], torch.Tensor]) -> torch.Tensor:
        """``blocks`` on ``x``. Every block of a step attends under the same mask at a resolution: it is made once per
        forward (``biases``, by the mask's identity and length), as each block would make it."""
        x = x.transpose(1, 2).contiguous()  # einops' rearrange "b c t -> b t c", then .contiguous()
        key = (id(mask), x.shape[1])
        bias = biases.get(key)
        if bias is None:
            first: Attention = blocks[0].attn1  # ty: ignore[invalid-assignment]
            bias = biases[key] = first.head_mask(self._attn_mask(x, mask, streaming), x.shape[0], x.shape[1])
        for block in blocks:
            x = block(x, bias)
        return x.transpose(1, 2).contiguous()

    def forward(
        self, x: torch.Tensor, mask: torch.Tensor, mu: torch.Tensor, t: torch.Tensor, spks: torch.Tensor | None = None, cond: torch.Tensor | None = None, streaming: bool = False
    ) -> torch.Tensor:
        t = self.time_mlp(self.time_embeddings(t).to(t.dtype))
        x = torch.cat([x, mu], dim=1)
        if spks is not None:
            x = torch.cat([x, spks.unsqueeze(-1).expand(*spks.shape, x.shape[-1])], dim=1)
        if cond is not None:
            x = torch.cat([x, cond], dim=1)
        hiddens = []
        biases: dict[tuple[int, int], torch.Tensor] = {}
        masks = [mask]
        for resnet, blocks, downsample in self.down_blocks:  # ty: ignore[not-iterable]
            mask_down = masks[-1]
            x = self._transformers(blocks, resnet(x, mask_down, t), mask_down, streaming, biases)
            hiddens.append(x)
            x = downsample(x * mask_down)
            masks.append(mask_down[:, :, ::2])
        masks = masks[:-1]
        mask_mid = masks[-1]
        for resnet, blocks in self.mid_blocks:  # ty: ignore[not-iterable]
            x = self._transformers(blocks, resnet(x, mask_mid, t), mask_mid, streaming, biases)
        mask_up = mask
        for resnet, blocks, upsample in self.up_blocks:  # ty: ignore[not-iterable]
            mask_up = masks.pop()
            skip = hiddens.pop()
            x = torch.cat([x[:, :, : skip.shape[-1]], skip], dim=1)
            x = self._transformers(blocks, resnet(x, mask_up, t), mask_up, streaming, biases)
            x = upsample(x * mask_up)
        x = self.final_block(x, mask_up)
        return self.final_proj(x * mask_up) * mask
