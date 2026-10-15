"""VITS' WaveNet and residual coupling flow (``modules.py``: ``WN``, ``ResidualCouplingLayer``, ``Flip``;
``models.py``: ``ResidualCouplingBlock``), as GPT-SoVITS (MIT) has them.

Weight norm (on ``cond_layer``, ``in_layers`` and ``res_skip_layers``) is folded at load
(``mizuoto.runtime.loading.fold_weight_norm``): upstream recomputes ``_weight_norm(v, g)`` before every forward,
which in float32 is the same weight. Dropout is inactive in inference and left out.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from mizuoto.runtime import ops as default_ops


def fused_add_tanh_sigmoid_multiply(input_a: torch.Tensor, input_b: torch.Tensor, n_channels: int) -> torch.Tensor:
    in_act = input_a + input_b
    t_act = torch.tanh(in_act[:, :n_channels, :])
    s_act = torch.sigmoid(in_act[:, n_channels:, :])
    return t_act * s_act


class WN(nn.Module):
    def __init__(self, hidden_channels: int, kernel_size: int, dilation_rate: int, n_layers: int, gin_channels: int = 0, ops: Any = default_ops.default) -> None:
        super().__init__()
        assert kernel_size % 2 == 1
        self.hidden_channels = hidden_channels
        self.n_layers = n_layers
        self.gin_channels = gin_channels
        self.in_layers = nn.ModuleList()
        self.res_skip_layers = nn.ModuleList()
        if gin_channels != 0:
            self.cond_layer = ops.Conv1d(gin_channels, 2 * hidden_channels * n_layers, 1)
        for i in range(n_layers):
            dilation = dilation_rate**i
            padding = int((kernel_size * dilation - dilation) / 2)
            self.in_layers.append(ops.Conv1d(hidden_channels, 2 * hidden_channels, kernel_size, dilation=dilation, padding=padding))
            res_skip_channels = 2 * hidden_channels if i < n_layers - 1 else hidden_channels
            self.res_skip_layers.append(ops.Conv1d(hidden_channels, res_skip_channels, 1))

    def forward(self, x: torch.Tensor, x_mask: torch.Tensor, g: torch.Tensor | None = None) -> torch.Tensor:
        output = torch.zeros_like(x)
        if g is not None:
            g = self.cond_layer(g)
        for i in range(self.n_layers):
            x_in = self.in_layers[i](x)
            if g is not None:
                cond_offset = i * 2 * self.hidden_channels
                g_l = g[:, cond_offset : cond_offset + 2 * self.hidden_channels, :]
            else:
                g_l = torch.zeros_like(x_in)
            acts = fused_add_tanh_sigmoid_multiply(x_in, g_l, self.hidden_channels)
            res_skip_acts = self.res_skip_layers[i](acts)
            if i < self.n_layers - 1:
                res_acts = res_skip_acts[:, : self.hidden_channels, :]
                x = (x + res_acts) * x_mask
                output = output + res_skip_acts[:, self.hidden_channels :, :]
            else:
                output = output + res_skip_acts
        return output * x_mask


class Flip(nn.Module):
    def forward(self, x: torch.Tensor, x_mask: torch.Tensor, g: torch.Tensor | None = None, reverse: bool = True) -> torch.Tensor:
        return torch.flip(x, [1])


class ResidualCouplingLayer(nn.Module):
    def __init__(
        self,
        channels: int,
        hidden_channels: int,
        kernel_size: int,
        dilation_rate: int,
        n_layers: int,
        gin_channels: int = 0,
        mean_only: bool = False,
        ops: Any = default_ops.default,
    ) -> None:
        assert channels % 2 == 0, "channels should be divisible by 2"
        super().__init__()
        self.half_channels = channels // 2
        self.mean_only = mean_only
        self.pre = ops.Conv1d(self.half_channels, hidden_channels, 1)
        self.enc = WN(hidden_channels, kernel_size, dilation_rate, n_layers, gin_channels=gin_channels, ops=ops)
        self.post = ops.Conv1d(hidden_channels, self.half_channels * (2 - mean_only), 1)

    def forward(self, x: torch.Tensor, x_mask: torch.Tensor, g: torch.Tensor | None = None, reverse: bool = True) -> torch.Tensor:
        """The inverse transform (``reverse=True``), the direction inference uses; the forward one without its log-determinant."""
        x0, x1 = torch.split(x, [self.half_channels] * 2, 1)
        h = self.pre(x0) * x_mask
        h = self.enc(h, x_mask, g=g)
        stats = self.post(h) * x_mask
        if not self.mean_only:
            m, logs = torch.split(stats, [self.half_channels] * 2, 1)
        else:
            m = stats
            logs = torch.zeros_like(m)
        if not reverse:
            x1 = m + x1 * torch.exp(logs) * x_mask
        else:
            x1 = (x1 - m) * torch.exp(-logs) * x_mask
        return torch.cat([x0, x1], 1)


class ResidualCouplingBlock(nn.Module):
    """``n_flows`` mean-only coupling layers, each followed by a channel flip (``flows.<2i>`` and ``flows.<2i+1>``)."""

    def __init__(
        self,
        channels: int,
        hidden_channels: int,
        kernel_size: int,
        dilation_rate: int,
        n_layers: int,
        n_flows: int = 4,
        gin_channels: int = 0,
        ops: Any = default_ops.default,
    ) -> None:
        super().__init__()
        self.flows = nn.ModuleList()
        for _ in range(n_flows):
            self.flows.append(ResidualCouplingLayer(channels, hidden_channels, kernel_size, dilation_rate, n_layers, gin_channels=gin_channels, mean_only=True, ops=ops))
            self.flows.append(Flip())

    def forward(self, x: torch.Tensor, x_mask: torch.Tensor, g: torch.Tensor | None = None, reverse: bool = True) -> torch.Tensor:
        flows = reversed(self.flows) if reverse else self.flows
        for flow in flows:
            x = flow(x, x_mask, g=g, reverse=reverse)
        return x
