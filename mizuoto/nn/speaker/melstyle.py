"""Meta-StyleSpeech's mel style encoder (``modules.py``: ``MelStyleEncoder``), as GPT-SoVITS (MIT) has it for its
reference encoder (``ref_enc``): spectral MLP with Mish, two gated convolutions, multi-head self-attention, a
projection and a masked temporal mean. (batch, channels, frames) → (batch, style_vector_dim, 1).

Parameter names are upstream's (``spectral.0.fc``, ``temporal.0.conv1.conv``, ``slf_attn.w_qs``…). Dropout is
inactive in inference; ``nn.Identity`` keeps its places in the ``Sequential`` so the indices match.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from mizuoto.runtime import ops as default_ops


class LinearNorm(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, bias: bool = True, ops: Any = default_ops.default) -> None:
        super().__init__()
        self.fc = ops.Linear(in_channels, out_channels, bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)


class Mish(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.tanh(F.softplus(x))


class ConvNorm(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 1, ops: Any = default_ops.default) -> None:
        super().__init__()
        assert kernel_size % 2 == 1
        self.conv = ops.Conv1d(in_channels, out_channels, kernel_size=kernel_size, stride=1, padding=int((kernel_size - 1) / 2), dilation=1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class Conv1dGLU(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, ops: Any = default_ops.default) -> None:
        super().__init__()
        self.out_channels = out_channels
        self.conv1 = ConvNorm(in_channels, 2 * out_channels, kernel_size=kernel_size, ops=ops)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.conv1(x)
        x1, x2 = torch.split(x, split_size_or_sections=self.out_channels, dim=1)
        x = x1 * torch.sigmoid(x2)
        return residual + x


class StyleAttention(nn.Module):
    """``modules.MultiHeadAttention``: linear q, k, v, scaled by √d_model (not √d_k), with a residual."""

    def __init__(self, n_head: int, d_model: int, d_k: int, d_v: int, ops: Any = default_ops.default) -> None:
        super().__init__()
        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v
        self.w_qs = ops.Linear(d_model, n_head * d_k)
        self.w_ks = ops.Linear(d_model, n_head * d_k)
        self.w_vs = ops.Linear(d_model, n_head * d_v)
        self.temperature = float(d_model) ** 0.5  # numpy.power(d_model, 0.5) upstream: the same C pow
        self.fc = ops.Linear(n_head * d_v, d_model)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        sz_b, len_x, _ = x.size()
        residual = x
        q = self.w_qs(x).view(sz_b, len_x, n_head, d_k)
        k = self.w_ks(x).view(sz_b, len_x, n_head, d_k)
        v = self.w_vs(x).view(sz_b, len_x, n_head, d_v)
        q = q.permute(2, 0, 1, 3).contiguous().view(-1, len_x, d_k)
        k = k.permute(2, 0, 1, 3).contiguous().view(-1, len_x, d_k)
        v = v.permute(2, 0, 1, 3).contiguous().view(-1, len_x, d_v)
        slf_mask = mask.repeat(n_head, 1, 1) if mask is not None else None
        attn = torch.bmm(q, k.transpose(1, 2))
        attn = attn / self.temperature
        if slf_mask is not None:
            attn = attn.masked_fill(slf_mask, -float("inf"))
        attn = F.softmax(attn, dim=2)
        output = torch.bmm(attn, v)
        output = output.view(n_head, sz_b, len_x, d_v)
        output = output.permute(1, 2, 0, 3).contiguous().view(sz_b, len_x, -1)
        output = self.fc(output)
        return output + residual


class MelStyleEncoder(nn.Module):
    def __init__(
        self,
        n_mel_channels: int = 80,
        style_hidden: int = 128,
        style_vector_dim: int = 256,
        style_kernel_size: int = 5,
        style_head: int = 2,
        ops: Any = default_ops.default,
    ) -> None:
        super().__init__()
        hidden = style_hidden
        self.spectral = nn.Sequential(
            LinearNorm(n_mel_channels, hidden, ops=ops),
            Mish(),
            nn.Identity(),
            LinearNorm(hidden, hidden, ops=ops),
            Mish(),
            nn.Identity(),
        )
        self.temporal = nn.Sequential(Conv1dGLU(hidden, hidden, style_kernel_size, ops=ops), Conv1dGLU(hidden, hidden, style_kernel_size, ops=ops))
        self.slf_attn = StyleAttention(style_head, hidden, hidden // style_head, hidden // style_head, ops=ops)
        self.fc = LinearNorm(hidden, style_vector_dim, ops=ops)

    @staticmethod
    def temporal_avg_pool(x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        if mask is None:
            return torch.mean(x, dim=1)
        len_ = (~mask).sum(dim=1).unsqueeze(1)
        x = x.masked_fill(mask.unsqueeze(-1), 0)
        dtype = x.dtype
        x = x.float()
        x = torch.div(x, len_.unsqueeze(1))
        return x.sum(dim=1).to(dtype)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        """``x``: (batch, channels, frames); ``mask``: (batch, 1, frames), 1 where valid."""
        x = x.transpose(1, 2)
        if mask is not None:
            mask = (mask.int() == 0).squeeze(1)
        max_len = x.shape[1]
        slf_attn_mask = mask.unsqueeze(1).expand(-1, max_len, -1) if mask is not None else None
        x = self.spectral(x)
        x = x.transpose(1, 2)
        x = self.temporal(x)
        x = x.transpose(1, 2)
        if mask is not None:
            x = x.masked_fill(mask.unsqueeze(-1), 0)
        x = self.slf_attn(x, mask=slf_attn_mask)
        x = self.fc(x)
        w = self.temporal_avg_pool(x, mask=mask)
        return w.unsqueeze(-1)
