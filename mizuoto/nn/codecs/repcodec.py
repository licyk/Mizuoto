"""MaskGCT's semantic codec (Amphion, MIT; ``models/codec/kmeans/repcodec_model.py``) as IndexTTS-2 uses it:
w2v-BERT features → a Vocos encoder → residual factorised VQ (L2-normalised codebooks).

``quantize`` gives the codes and their quantised vectors; ``vq2emb`` turns codes back into vectors. The
decoder (reconstruction, training only) is not built: its weights are ignored at load (``IGNORED``).
Weight norm on the projections is folded at load.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from mizuoto.nn.vocoders.vocos import VocosBackbone
from mizuoto.runtime import ops as default_ops

IGNORED = ("decoder.*",)


class FactorizedVectorQuantize(nn.Module):
    def __init__(self, input_dim: int, codebook_size: int, codebook_dim: int, ops: Any) -> None:
        super().__init__()
        self.in_project = ops.Conv1d(input_dim, codebook_dim, kernel_size=1)
        self.out_project = ops.Conv1d(codebook_dim, input_dim, kernel_size=1)
        self.codebook = ops.Embedding(codebook_size, codebook_dim)

    def forward(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """(B, D, T) → (quantised (B, D, T), indices (B, T))."""
        z_e = self.in_project(z)
        b = z_e.size(0)
        encodings = F.normalize(z_e.permute(0, 2, 1).reshape(-1, z_e.shape[1]))
        codebook = F.normalize(self.codebook.weight)
        dist = encodings.pow(2).sum(1, keepdim=True) - 2 * encodings @ codebook.t() + codebook.pow(2).sum(1, keepdim=True).t()
        indices = (-dist).max(1)[1].reshape(b, -1)
        z_q = self.decode_code(indices)
        z_q = z_e + (z_q - z_e).detach()
        return self.out_project(z_q), indices

    def decode_code(self, ids: torch.Tensor) -> torch.Tensor:
        return F.embedding(ids, self.codebook.weight).transpose(1, 2)

    def vq2emb(self, ids: torch.Tensor) -> torch.Tensor:
        return self.out_project(self.decode_code(ids))


class ResidualVQ(nn.Module):
    def __init__(self, input_dim: int, num_quantizers: int, codebook_size: int, codebook_dim: int, ops: Any) -> None:
        super().__init__()
        self.quantizers = nn.ModuleList(FactorizedVectorQuantize(input_dim, codebook_size, codebook_dim, ops) for _ in range(num_quantizers))

    def forward(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        quantized: Any = 0.0
        residual = z
        indices = []
        for q in self.quantizers:
            z_q, idx = q(residual)
            quantized = quantized + z_q
            residual = residual - z_q
            indices.append(idx)
        return quantized, torch.stack(indices)

    def vq2emb(self, vq: torch.Tensor, n_quantizers: int | None = None) -> torch.Tensor:
        """``vq`` indexed by quantizer first, as upstream indexes it (``vq[i]``)."""
        out: Any = 0.0
        quantizers: list[Any] = list(self.quantizers)
        for i, q in enumerate(quantizers):
            if n_quantizers is not None and i >= n_quantizers:
                break
            out = out + q.vq2emb(vq[i])
        return out


class RepCodec(nn.Module):
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
        self.encoder = nn.Sequential(VocosBackbone(hidden_size, vocos_dim, vocos_intermediate_dim, vocos_num_layers, ops), ops.Linear(vocos_dim, hidden_size))
        self.quantizer = ResidualVQ(hidden_size, num_quantizers, codebook_size, codebook_dim, ops)

    @classmethod
    def from_config(cls, c: dict[str, Any], ops: Any = default_ops.default) -> RepCodec:
        keys = ("codebook_size", "hidden_size", "codebook_dim", "vocos_dim", "vocos_intermediate_dim", "vocos_num_layers", "num_quantizers")
        return cls(**{k: c[k] for k in keys if k in c}, ops=ops)

    def quantize(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """(B, T, hidden) features → (codes (B, T) for one quantizer, else (N, B, T); vectors (B, T, hidden))."""
        x = self.encoder(x.transpose(1, 2)).transpose(1, 2)
        quantized, indices = self.quantizer(x)
        if indices.shape[0] == 1:
            return indices.squeeze(0), quantized.transpose(1, 2)
        return indices, quantized.transpose(1, 2)
