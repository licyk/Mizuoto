"""Whisper's encoder-decoder as transformers 4.57.3's ``WhisperForConditionalGeneration`` computes it (Apache-2.0).

Built from the model's ``config.json`` at any size, under transformers' parameter names (``model.encoder.*``,
``model.decoder.*``; the output projection is the token embedding, as transformers ties it). What exactness
with transformers' SDPA path takes:
- the query is projected then scaled by ``head_dim ** -0.5`` and SDPA runs with ``scale=1``; queries, keys and
  values are made contiguous in (batch, heads, positions, head_dim) before the call, the output after it;
- the encoder adds its stored positions through the embedding (an ``arange`` lookup) to the permuted
  convolution output, and clamps each layer's output in float16 as transformers does;
- the decoder's self-attention is causal for a multi-token prompt (no mask), unmasked for each new token, and its
  cache grows by concatenation; the cross-attention's keys and values are computed once per generation call from
  the encoder output repeated per beam;
- logits are computed for every position given and copied to float32.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from mizuoto.runtime import ops as default_ops
from mizuoto.runtime.accel import Site, attention

SITE = Site("asr.whisper", approx_ok=False)  # autoregressive: quantised attention derails its sampling


@dataclass(frozen=True)
class WhisperConfig:
    d_model: int
    encoder_layers: int
    encoder_attention_heads: int
    encoder_ffn_dim: int
    decoder_layers: int
    decoder_attention_heads: int
    decoder_ffn_dim: int
    num_mel_bins: int
    max_source_positions: int
    max_target_positions: int
    vocab_size: int
    pad_token_id: int | None = None
    activation_function: str = "gelu"

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> WhisperConfig:
        if raw.get("model_type") != "whisper":
            raise ValueError("not a Whisper model")
        if raw.get("activation_function", "gelu") != "gelu":
            raise ValueError(f"unsupported activation {raw.get('activation_function')}")
        return cls(
            d_model=int(raw["d_model"]),
            encoder_layers=int(raw["encoder_layers"]),
            encoder_attention_heads=int(raw["encoder_attention_heads"]),
            encoder_ffn_dim=int(raw["encoder_ffn_dim"]),
            decoder_layers=int(raw["decoder_layers"]),
            decoder_attention_heads=int(raw["decoder_attention_heads"]),
            decoder_ffn_dim=int(raw["decoder_ffn_dim"]),
            num_mel_bins=int(raw.get("num_mel_bins", 80)),
            max_source_positions=int(raw.get("max_source_positions", 1500)),
            max_target_positions=int(raw.get("max_target_positions", 448)),
            vocab_size=int(raw["vocab_size"]),
            pad_token_id=raw.get("pad_token_id"),
        )

    @classmethod
    def load(cls, folder: Path) -> WhisperConfig:
        return cls.from_dict(json.loads((folder / "config.json").read_text(encoding="utf-8")))

    @property
    def input_stride(self) -> int:
        """Mel frames per encoder position (the two convolutions' strides)."""
        return 2

    @property
    def num_segment_frames(self) -> int:
        """Mel frames in one window (3000 for 30 s)."""
        return self.input_stride * self.max_source_positions


class Attention(nn.Module):
    """``WhisperAttention``: projections and SDPA, without the cache bookkeeping (the decoder holds it)."""

    def __init__(self, dim: int, heads: int, ops: Any) -> None:
        super().__init__()
        self.heads = heads
        self.head_dim = dim // heads
        if self.head_dim * heads != dim:
            raise ValueError(f"d_model {dim} is not divisible by {heads} heads")
        self.scaling = self.head_dim**-0.5
        self.k_proj = ops.Linear(dim, dim, bias=False)
        self.v_proj = ops.Linear(dim, dim, bias=True)
        self.q_proj = ops.Linear(dim, dim, bias=True)
        self.out_proj = ops.Linear(dim, dim, bias=True)

    def query(self, hidden: torch.Tensor) -> torch.Tensor:
        bsz, tgt_len = hidden.shape[:-1]
        q = self.q_proj(hidden) * self.scaling
        return q.view(bsz, tgt_len, -1, self.head_dim).transpose(1, 2).contiguous()

    def keys_values(self, states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        bsz = states.shape[0]
        k = self.k_proj(states).view(bsz, -1, self.heads, self.head_dim).transpose(1, 2).contiguous()
        v = self.v_proj(states).view(bsz, -1, self.heads, self.head_dim).transpose(1, 2).contiguous()
        return k, v

    def attend(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, *, causal: bool = False, mask: torch.Tensor | None = None) -> torch.Tensor:
        bsz, _, tgt_len, _ = q.shape
        out = attention(q, k, v, site=SITE, mask=mask, scale=1.0, causal=causal)
        out = out.transpose(1, 2).contiguous()
        out = out.reshape(bsz, tgt_len, -1).contiguous()
        return self.out_proj(out)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        """Unmasked self-attention (the encoder's)."""
        q = self.query(hidden)
        k, v = self.keys_values(hidden)
        return self.attend(q, k, v, causal=False)


class EncoderLayer(nn.Module):
    def __init__(self, c: WhisperConfig, ops: Any) -> None:
        super().__init__()
        self.self_attn = Attention(c.d_model, c.encoder_attention_heads, ops)
        self.self_attn_layer_norm = ops.LayerNorm(c.d_model)
        self.fc1 = ops.Linear(c.d_model, c.encoder_ffn_dim)
        self.fc2 = ops.Linear(c.encoder_ffn_dim, c.d_model)
        self.final_layer_norm = ops.LayerNorm(c.d_model)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        residual = hidden
        hidden = residual + self.self_attn(self.self_attn_layer_norm(hidden))
        residual = hidden
        hidden = self.fc2(F.gelu(self.fc1(self.final_layer_norm(hidden))))
        hidden = residual + hidden
        if hidden.dtype == torch.float16:
            clamp = torch.finfo(hidden.dtype).max - 1000
            hidden = torch.clamp(hidden, min=-clamp, max=clamp)
        return hidden


class Encoder(nn.Module):
    def __init__(self, c: WhisperConfig, ops: Any) -> None:
        super().__init__()
        self.config = c
        self.conv1 = ops.Conv1d(c.num_mel_bins, c.d_model, kernel_size=3, padding=1)
        self.conv2 = ops.Conv1d(c.d_model, c.d_model, kernel_size=3, stride=2, padding=1)
        self.embed_positions = ops.Embedding(c.max_source_positions, c.d_model)
        self.embed_positions.requires_grad_(False)
        self.layers = nn.ModuleList([EncoderLayer(c, ops) for _ in range(c.encoder_layers)])
        self.layer_norm = ops.LayerNorm(c.d_model)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """(batch, mel bins, 2 × max_source_positions) → (batch, max_source_positions, d_model)."""
        expected = self.config.num_segment_frames
        if features.shape[-1] != expected:
            raise ValueError(f"Whisper takes {expected} mel frames, not {features.shape[-1]}")
        embeds = F.gelu(self.conv1(features))
        embeds = F.gelu(self.conv2(embeds))
        embeds = embeds.permute(0, 2, 1)
        positions = torch.arange(self.embed_positions.num_embeddings, device=embeds.device)
        hidden = embeds + self.embed_positions(positions)
        for layer in self.layers:
            hidden = layer(hidden)
        return self.layer_norm(hidden)


class DecoderLayer(nn.Module):
    def __init__(self, c: WhisperConfig, ops: Any) -> None:
        super().__init__()
        self.self_attn = Attention(c.d_model, c.decoder_attention_heads, ops)
        self.self_attn_layer_norm = ops.LayerNorm(c.d_model)
        self.encoder_attn = Attention(c.d_model, c.decoder_attention_heads, ops)
        self.encoder_attn_layer_norm = ops.LayerNorm(c.d_model)
        self.fc1 = ops.Linear(c.d_model, c.decoder_ffn_dim)
        self.fc2 = ops.Linear(c.decoder_ffn_dim, c.d_model)
        self.final_layer_norm = ops.LayerNorm(c.d_model)


class DecoderCache:
    """transformers' ``EncoderDecoderCache`` of two ``DynamicCache``s: self-attention keys and values grown by
    concatenation, cross-attention keys and values computed on the first call."""

    def __init__(self, layers: int) -> None:
        self.keys: list[torch.Tensor | None] = [None] * layers
        self.values: list[torch.Tensor | None] = [None] * layers
        self.cross: list[tuple[torch.Tensor, torch.Tensor] | None] = [None] * layers
        self.cross_shared = False
        """Whether every beam of an item has bit-identical cross-attention keys and values (then reordering
        beams cannot change them, and is skipped)."""

    @property
    def length(self) -> int:
        k = self.keys[0]
        return 0 if k is None else int(k.shape[-2])

    def reorder(self, index: torch.Tensor) -> None:
        """``EncoderDecoderCache.reorder_cache``: rows taken in ``index``'s order."""
        for i, (k, v) in enumerate(zip(self.keys, self.values)):
            if k is not None and v is not None:
                self.keys[i] = k.index_select(0, index.to(k.device))
                self.values[i] = v.index_select(0, index.to(v.device))
        if not self.cross_shared:
            for i, kv in enumerate(self.cross):
                if kv is not None:
                    self.cross[i] = (kv[0].index_select(0, index.to(kv[0].device)), kv[1].index_select(0, index.to(kv[1].device)))


class Decoder(nn.Module):
    def __init__(self, c: WhisperConfig, ops: Any) -> None:
        super().__init__()
        self.config = c
        self.embed_tokens = ops.Embedding(c.vocab_size, c.d_model, c.pad_token_id)
        self.embed_positions = ops.Embedding(c.max_target_positions, c.d_model)
        self.layers = nn.ModuleList([DecoderLayer(c, ops) for _ in range(c.decoder_layers)])
        self.layer_norm = ops.LayerNorm(c.d_model)

    def forward(self, input_ids: torch.Tensor, encoder_hidden: torch.Tensor, cache: DecoderCache | None, self_mask: torch.Tensor | None = None) -> torch.Tensor:
        """Logits (in the model's dtype, every position) for ``input_ids`` after what ``cache`` holds; ``cache`` None: no cache."""
        past = cache.length if cache is not None else 0
        bsz, length = input_ids.shape
        embeds = self.embed_tokens(input_ids)
        position_ids = torch.arange(past, past + length, device=embeds.device).unsqueeze(0).repeat(bsz, 1)
        hidden = embeds + self.embed_positions.weight[position_ids].to(embeds.device)
        for i, layer in enumerate(self.layers):
            assert isinstance(layer, DecoderLayer)
            residual = hidden
            h = layer.self_attn_layer_norm(hidden)
            q = layer.self_attn.query(h)
            k, v = layer.self_attn.keys_values(h)
            if cache is not None:
                pk, pv = cache.keys[i], cache.values[i]
                if pk is not None and pv is not None:
                    k = torch.cat([pk, k], dim=-2)
                    v = torch.cat([pv, v], dim=-2)
                cache.keys[i], cache.values[i] = k, v
            causal = length > 1 and self_mask is None
            hidden = residual + layer.self_attn.attend(q, k, v, causal=causal, mask=self_mask)

            residual = hidden
            h = layer.encoder_attn_layer_norm(hidden)
            q = layer.encoder_attn.query(h)
            kv = cache.cross[i] if cache is not None else None
            if kv is None:
                kv = layer.encoder_attn.keys_values(encoder_hidden)
                if cache is not None:
                    cache.cross[i] = kv
            hidden = residual + layer.encoder_attn.attend(q, kv[0], kv[1])

            residual = hidden
            hidden = residual + layer.fc2(F.gelu(layer.fc1(layer.final_layer_norm(hidden))))
        hidden = self.layer_norm(hidden)
        return F.linear(hidden, self.embed_tokens.weight)


class WhisperModel(nn.Module):
    """``WhisperForConditionalGeneration``'s computations; parameters under its names (``model.…``)."""

    def __init__(self, c: WhisperConfig, ops: Any = default_ops.default) -> None:
        super().__init__()
        self.config = c
        self.model = nn.Module()
        self.model.add_module("encoder", Encoder(c, ops))
        self.model.add_module("decoder", Decoder(c, ops))

    @property
    def encoder(self) -> Encoder:
        enc = self.model.encoder
        assert isinstance(enc, Encoder)
        return enc

    @property
    def decoder(self) -> Decoder:
        dec = self.model.decoder
        assert isinstance(dec, Decoder)
        return dec

    @property
    def dtype(self) -> torch.dtype:
        return self.encoder.conv1.weight.dtype

    @property
    def device(self) -> torch.device:
        return self.encoder.conv1.weight.device


def load_whisper(folder: Path, dtype: torch.dtype, device: torch.device | str = "cpu") -> WhisperModel:
    """The model in ``folder`` (``config.json`` and ``model.safetensors``, or ``pytorch_model.bin``)."""
    from mizuoto.runtime.loading import load_into, load_state_dict

    model = WhisperModel(WhisperConfig.load(folder))
    path = folder / "model.safetensors"
    if not path.is_file():
        path = folder / "pytorch_model.bin"
    state = load_state_dict(path)
    load_into(model, state, ignore=["proj_out.weight"], what="Whisper model")
    del state
    return model.to(device=device, dtype=dtype).eval()


def sinusoids(length: int, channels: int, max_timescale: float = 10000) -> torch.Tensor:
    """transformers' ``sinusoids``: the encoder positions of a model built from scratch (tests)."""
    log_timescale_increment = math.log(max_timescale) / (channels // 2 - 1)
    inv_timescales = torch.exp(-log_timescale_increment * torch.arange(channels // 2))
    scaled_time = torch.arange(length).view(-1, 1) * inv_timescales.view(1, -1)
    return torch.cat([scaled_time.sin(), scaled_time.cos()], dim=1)
