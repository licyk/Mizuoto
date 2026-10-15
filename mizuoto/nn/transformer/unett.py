"""E2-TTS's UNet-style transformer (``f5_tts.model.backbones.unett``, MIT), with the same interface as ``DiT``.

Flat transformer layers with U-Net skip connections (concatenated and projected in the later half).
The time embedding is packed in front of the sequence, and the norms are x-transformers' RMSNorm.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from mizuoto.nn.transformer.dit import Attention, ConvPositionEmbedding, FeedForward, RotaryEmbedding, TimestepEmbedding, rope_tables
from mizuoto.runtime import ops as default_ops


class XRMSNorm(nn.Module):
    """x-transformers' ``RMSNorm``: ``normalize(x) * sqrt(dim) * g``."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.scale = dim**0.5
        self.g = nn.Parameter(torch.empty(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gamma = self.g + float(False)
        return F.normalize(x, dim=-1) * self.scale * gamma


class TextEmbedding(nn.Module):
    def __init__(self, text_num_embeds: int, text_dim: int, mask_padding: bool = True, ops: Any = default_ops.default) -> None:
        super().__init__()
        self.text_embed = ops.Embedding(text_num_embeds + 1, text_dim)
        self.mask_padding = mask_padding

    def forward(self, text: torch.Tensor, seq_len: int, drop_text: bool = False) -> torch.Tensor:
        text = text + 1
        text = text[:, :seq_len]
        text = F.pad(text, (0, seq_len - text.shape[1]), value=0)
        if drop_text:
            text = torch.zeros_like(text)
        return self.text_embed(text)


class InputEmbedding(nn.Module):
    def __init__(self, mel_dim: int, text_dim: int, out_dim: int, ops: Any = default_ops.default) -> None:
        super().__init__()
        self.proj = ops.Linear(mel_dim * 2 + text_dim, out_dim)
        self.conv_pos_embed = ConvPositionEmbedding(out_dim, ops=ops)

    def forward(self, x: torch.Tensor, cond: torch.Tensor, text_embed: torch.Tensor, drop_audio_cond: bool = False) -> torch.Tensor:
        if drop_audio_cond:
            cond = torch.zeros_like(cond)
        x = self.proj(torch.cat((x, cond, text_embed), dim=-1))
        return self.conv_pos_embed(x) + x


class UNetT(nn.Module):
    def __init__(
        self,
        *,
        dim: int,
        depth: int = 8,
        heads: int = 8,
        dim_head: int = 64,
        ff_mult: float = 4,
        mel_dim: int = 100,
        text_num_embeds: int = 256,
        text_dim: int | None = None,
        text_mask_padding: bool = True,
        pe_attn_head: int | None = None,
        ops: Any = default_ops.default,
        **_unused: Any,
    ) -> None:
        super().__init__()
        if depth % 2:
            raise ValueError("UNet-Transformer's depth should be even.")
        text_dim = text_dim if text_dim is not None else mel_dim
        self.dim = dim
        self.depth = depth
        self.time_embed = TimestepEmbedding(dim, ops=ops)
        self.text_embed = TextEmbedding(text_num_embeds, text_dim, mask_padding=text_mask_padding, ops=ops)
        self.input_embed = InputEmbedding(mel_dim, text_dim, dim, ops)
        self.rotary_embed = RotaryEmbedding(dim_head)
        self.layers = nn.ModuleList()
        for idx in range(depth):
            later = idx >= depth // 2
            skip_proj = ops.Linear(dim * 2, dim, bias=False) if later else None
            layer = nn.ModuleList()
            layer.add_module("0", skip_proj)  # upstream keeps a None in the first half, so the indices stay put
            for i, m in enumerate((XRMSNorm(dim), Attention(dim, heads, dim_head, pe_attn_head, ops), XRMSNorm(dim), FeedForward(dim, ff_mult, ops)), start=1):
                layer.add_module(str(i), m)
            self.layers.append(layer)
        self.norm_out = XRMSNorm(dim)
        self.proj_out = ops.Linear(dim, mel_dim)

    def text_condition(self, text: torch.Tensor, seq_len: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.text_embed(text, seq_len, drop_text=False), self.text_embed(text, seq_len, drop_text=True)

    def forward(
        self, x: torch.Tensor, cond: torch.Tensor, text_embeds: tuple[torch.Tensor, torch.Tensor], time: torch.Tensor, mask: torch.Tensor | None = None, cfg_infer: bool = True
    ) -> torch.Tensor:
        batch, seq_len = x.shape[0], x.shape[1]
        if time.ndim == 0:
            time = time.repeat(batch)
        t = self.time_embed(time)
        text_cond, text_uncond = text_embeds
        if cfg_infer:
            x = torch.cat((self.input_embed(x, cond, text_cond, drop_audio_cond=False), self.input_embed(x, cond, text_uncond, drop_audio_cond=True)), dim=0)
            t = torch.cat((t, t), dim=0)
            mask = torch.cat((mask, mask), dim=0) if mask is not None else None
        else:
            x = self.input_embed(x, cond, text_cond, drop_audio_cond=False)
        x = torch.cat([t.unsqueeze(1), x], dim=1)
        if mask is not None:
            mask = F.pad(mask, (1, 0), value=1)
        rope = rope_tables(self.rotary_embed.forward_from_seq_len(seq_len + 1))
        skips = []
        for idx, layer in enumerate(self.layers):
            parts = dict(layer.named_children())
            skip_proj, attn_norm, attn, ff_norm, ff = parts.get("0"), parts["1"], parts["2"], parts["3"], parts["4"]
            if idx + 1 <= self.depth // 2:
                skips.append(x)
            else:
                x = torch.cat((x, skips.pop()), dim=-1)
                assert skip_proj is not None
                x = skip_proj(x)
            x = attn(attn_norm(x), mask, rope) + x
            x = ff(ff_norm(x)) + x
        x = self.norm_out(x)[:, 1:, :]
        return self.proj_out(x)
