"""HuBERT, as transformers' ``HubertModel`` computes it in eval mode (Apache-2.0).

Parameter names are transformers' (``feature_extractor.conv_layers.N.conv``, ``encoder.layers.N.attention.q_proj``…).
The positional convolution's weight norm (``parametrizations.weight.original0/1``, over dim 2) is folded
when loading (``fold_pos_conv``). Both layouts are supported: post-norm (``do_stable_layer_norm=False``,
HuBERT base) and pre-norm (the large models).

Attention is SDPA without a mask, as transformers' ``sdpa`` implementation calls it for one unpadded clip.
transformers draws one ``torch.rand([])`` per layer for layer drop even in eval mode; pass ``layerdrop``
(a CPU generator) to make the same draws, so a request's later random numbers stay where they were.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from mizuoto.runtime import ops as default_ops
from mizuoto.runtime.accel import Site, attention

SITE = Site("hubert")


@dataclass
class HubertConfig:
    hidden_size: int = 768
    num_hidden_layers: int = 12
    num_attention_heads: int = 12
    intermediate_size: int = 3072
    conv_dim: list[int] = field(default_factory=lambda: [512] * 7)
    conv_kernel: list[int] = field(default_factory=lambda: [10, 3, 3, 3, 3, 2, 2])
    conv_stride: list[int] = field(default_factory=lambda: [5, 2, 2, 2, 2, 2, 2])
    conv_bias: bool = False
    feat_extract_norm: str = "group"
    feat_proj_layer_norm: bool = True
    num_conv_pos_embeddings: int = 128
    num_conv_pos_embedding_groups: int = 16
    do_stable_layer_norm: bool = False
    layer_norm_eps: float = 1e-5

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> HubertConfig:
        if d.get("hidden_act", "gelu") != "gelu" or d.get("feat_extract_activation", "gelu") != "gelu":
            raise ValueError("Only GELU HuBERT models are supported")
        if d.get("conv_pos_batch_norm"):
            raise ValueError("HuBERT with a batch-normed positional convolution is not supported")
        names = cls.__dataclass_fields__
        return cls(**{k: v for k, v in d.items() if k in names})


class ConvLayer(nn.Module):
    def __init__(self, c: HubertConfig, i: int, norm: str | None, ops: Any) -> None:
        super().__init__()
        in_dim = c.conv_dim[i - 1] if i > 0 else 1
        out_dim = c.conv_dim[i]
        self.conv = ops.Conv1d(in_dim, out_dim, kernel_size=c.conv_kernel[i], stride=c.conv_stride[i], bias=c.conv_bias)
        self.norm = norm
        if norm == "group":
            self.layer_norm = ops.GroupNorm(num_groups=out_dim, num_channels=out_dim, affine=True)
        elif norm == "layer":
            self.layer_norm = ops.LayerNorm(out_dim, elementwise_affine=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        if self.norm == "group":
            x = self.layer_norm(x)
        elif self.norm == "layer":
            x = self.layer_norm(x.transpose(-2, -1)).transpose(-2, -1)
        return F.gelu(x)


class FeatureEncoder(nn.Module):
    def __init__(self, c: HubertConfig, ops: Any) -> None:
        super().__init__()
        n = len(c.conv_dim)
        if c.feat_extract_norm == "group":
            layers = [ConvLayer(c, 0, "group", ops)] + [ConvLayer(c, i, None, ops) for i in range(1, n)]
        elif c.feat_extract_norm == "layer":
            layers = [ConvLayer(c, i, "layer", ops) for i in range(n)]
        else:
            raise ValueError(f"feat_extract_norm {c.feat_extract_norm!r}")
        self.conv_layers = nn.ModuleList(layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x[:, None]
        for layer in self.conv_layers:
            x = layer(x)
        return x


class FeatureProjection(nn.Module):
    def __init__(self, c: HubertConfig, ops: Any) -> None:
        super().__init__()
        self.feat_proj_layer_norm = c.feat_proj_layer_norm
        if c.feat_proj_layer_norm:
            self.layer_norm = ops.LayerNorm(c.conv_dim[-1], eps=c.layer_norm_eps)
        self.projection = ops.Linear(c.conv_dim[-1], c.hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.feat_proj_layer_norm:
            x = self.layer_norm(x)
        return self.projection(x)


class PositionalConvEmbedding(nn.Module):
    def __init__(self, c: HubertConfig, ops: Any) -> None:
        super().__init__()
        k = c.num_conv_pos_embeddings
        self.conv = ops.Conv1d(c.hidden_size, c.hidden_size, kernel_size=k, padding=k // 2, groups=c.num_conv_pos_embedding_groups)
        self.remove = 1 if k % 2 == 0 else 0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x.transpose(1, 2))
        if self.remove:
            x = x[:, :, : -self.remove]
        return F.gelu(x).transpose(1, 2)


class Attention(nn.Module):
    def __init__(self, dim: int, heads: int, ops: Any) -> None:
        super().__init__()
        self.head_dim = dim // heads
        self.scaling = self.head_dim**-0.5
        self.k_proj = ops.Linear(dim, dim)
        self.v_proj = ops.Linear(dim, dim)
        self.q_proj = ops.Linear(dim, dim)
        self.out_proj = ops.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz, length = x.shape[:-1]
        shape = (bsz, length, -1, self.head_dim)
        q = self.q_proj(x).view(*shape).transpose(1, 2)
        k = self.k_proj(x).view(*shape).transpose(1, 2)
        v = self.v_proj(x).view(*shape).transpose(1, 2)
        out = attention(q, k, v, site=SITE, scale=self.scaling)
        out = out.transpose(1, 2).contiguous().reshape(bsz, length, -1).contiguous()
        return self.out_proj(out)


class FeedForward(nn.Module):
    def __init__(self, c: HubertConfig, ops: Any) -> None:
        super().__init__()
        self.intermediate_dense = ops.Linear(c.hidden_size, c.intermediate_size)
        self.output_dense = ops.Linear(c.intermediate_size, c.hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.output_dense(F.gelu(self.intermediate_dense(x)))


class EncoderLayer(nn.Module):
    def __init__(self, c: HubertConfig, ops: Any) -> None:
        super().__init__()
        self.stable = c.do_stable_layer_norm
        self.attention = Attention(c.hidden_size, c.num_attention_heads, ops)
        self.layer_norm = ops.LayerNorm(c.hidden_size, eps=c.layer_norm_eps)
        self.feed_forward = FeedForward(c, ops)
        self.final_layer_norm = ops.LayerNorm(c.hidden_size, eps=c.layer_norm_eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.stable:
            x = x + self.attention(self.layer_norm(x))
            return x + self.feed_forward(self.final_layer_norm(x))
        x = self.layer_norm(x + self.attention(x))
        return self.final_layer_norm(x + self.feed_forward(x))


class Encoder(nn.Module):
    def __init__(self, c: HubertConfig, ops: Any) -> None:
        super().__init__()
        self.stable = c.do_stable_layer_norm
        self.pos_conv_embed = PositionalConvEmbedding(c, ops)
        self.layer_norm = ops.LayerNorm(c.hidden_size, eps=c.layer_norm_eps)
        self.layers = nn.ModuleList(EncoderLayer(c, ops) for _ in range(c.num_hidden_layers))

    def forward(self, x: torch.Tensor, layerdrop: torch.Generator | None = None) -> list[torch.Tensor]:
        """Every hidden state: the input of each layer, then the last output (normalised when pre-norm)."""
        x = x + self.pos_conv_embed(x)
        if not self.stable:
            x = self.layer_norm(x)
        states = []
        for layer in self.layers:
            states.append(x)
            if layerdrop is not None:
                torch.rand([], generator=layerdrop)
            x = layer(x)
        if self.stable:
            x = self.layer_norm(x)
        states.append(x)
        return states


class HubertModel(nn.Module):
    """(batch, samples) at 16 kHz → every hidden state, each (batch, frames, hidden)."""

    def __init__(self, c: HubertConfig, ops: Any = default_ops.default) -> None:
        super().__init__()
        self.config = c
        self.feature_extractor = FeatureEncoder(c, ops)
        self.feature_projection = FeatureProjection(c, ops)
        self.encoder = Encoder(c, ops)

    def forward(self, samples: torch.Tensor, layerdrop: torch.Generator | None = None) -> list[torch.Tensor]:
        features = self.feature_extractor(samples).transpose(1, 2)
        return self.encoder(self.feature_projection(features), layerdrop)

    @staticmethod
    def ignorable() -> list[str]:
        return ["masked_spec_embed"]


def fold_pos_conv(sd: Mapping[str, torch.Tensor], prefix: str = "", dtype: torch.dtype | None = torch.float32) -> dict[str, torch.Tensor]:
    """The positional convolution's weight norm (dim 2) folded into ``…pos_conv_embed.conv.weight``.

    In ``dtype`` (float32: transformers loads a float16 checkpoint into float32 parameters and computes the
    weight norm from them).
    """
    from mizuoto.runtime.loading import fold_weight_norm

    base = f"{prefix}encoder.pos_conv_embed.conv."
    norm = {k: v for k, v in sd.items() if k.startswith(base) and ("parametrizations" in k or k.endswith(("weight_g", "weight_v")))}
    out = {k: v for k, v in sd.items() if k not in norm}
    out.update(fold_weight_norm(norm, dim=2, dtype=dtype))
    return out


def load_hubert(folder: str | Path) -> HubertModel:
    """A transformers ``HubertModel`` folder (``config.json`` + ``model.safetensors`` or ``pytorch_model.bin``) in float32 and
    eval mode, as ``HubertModel.from_pretrained`` loads it (a float16 file is copied into float32 parameters)."""
    import json

    from mizuoto.runtime.loading import load_into, load_state_dict

    folder = Path(folder)
    weights = next((folder / n for n in ("model.safetensors", "pytorch_model.bin") if (folder / n).is_file()), None)
    if weights is None:
        raise FileNotFoundError(f"{folder} has no HuBERT weights")
    model = HubertModel(HubertConfig.from_dict(json.loads((folder / "config.json").read_text(encoding="utf-8"))))
    state = {k: v.float() if v.is_floating_point() else v for k, v in fold_pos_conv(load_state_dict(weights)).items()}
    load_into(model, state, ignore=model.ignorable(), what="HuBERT")
    return model.eval()
