"""IndexTTS-2.5's semantic codec (index-tts ``codec/models.py``, ``EnhancedCodec``): MaskGCT's RepCodec with a 2×
down/up-sampling around it and a decoder, used to turn the GPT's codes back into w2v-BERT-like features.

``decode``: factorised-VQ vectors → the Vocos decoder → nearest ×2 upsampling → a 3-tap convolution. The
encoder, quantizer projections and ``down`` are loaded for strictness (upstream keeps random weights for any
key its checkpoint lacks; this loader refuses such a checkpoint).
"""

from __future__ import annotations

from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from mizuoto.nn.codecs.repcodec import ResidualVQ
from mizuoto.nn.vocoders.vocos import VocosBackbone
from mizuoto.runtime import ops as default_ops


class EnhancedCodec(nn.Module):
    def __init__(
        self,
        codebook_size: int = 8192,
        hidden_size: int = 1024,
        codebook_dim: int = 8,
        vocos_dim: int = 384,
        vocos_intermediate_dim: int = 2048,
        vocos_num_layers: int = 12,
        num_quantizers: int = 1,
        ops: Any = default_ops.default,
    ) -> None:
        super().__init__()
        self.down = ops.Conv1d(hidden_size, hidden_size, kernel_size=3, stride=2, padding=1)
        self.up = ops.Conv1d(hidden_size, hidden_size, kernel_size=3, stride=1, padding=1)
        self.encoder = nn.Sequential(VocosBackbone(hidden_size, vocos_dim, vocos_intermediate_dim, vocos_num_layers, ops), ops.Linear(vocos_dim, hidden_size))
        self.decoder = nn.Sequential(VocosBackbone(hidden_size, vocos_dim, vocos_intermediate_dim, vocos_num_layers, ops), ops.Linear(vocos_dim, hidden_size))
        self.quantizer = ResidualVQ(hidden_size, num_quantizers, codebook_size, codebook_dim, ops)

    @classmethod
    def from_config(cls, c: dict[str, Any], ops: Any = default_ops.default) -> EnhancedCodec:
        if c.get("downsample_scale", 2) != 2:
            raise ValueError("Only IndexTTS-2.5's codec (downsample_scale 2) is supported")
        keys = ("codebook_size", "hidden_size", "codebook_dim", "vocos_dim", "vocos_intermediate_dim", "vocos_num_layers", "num_quantizers")
        return cls(**{k: c[k] for k in keys if k in c}, ops=ops)

    def decode(self, codes: torch.Tensor) -> torch.Tensor:
        """Codes (B, T) → features (B, 2T, hidden)."""
        if codes.dim() == 2:
            codes = codes.unsqueeze(0)
        x = self.decoder(self.quantizer.vq2emb(codes))
        x = F.interpolate(x.transpose(1, 2), scale_factor=2, mode="nearest")
        return self.up(x).transpose(1, 2)
