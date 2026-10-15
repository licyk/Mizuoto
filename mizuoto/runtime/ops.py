"""Layers for native engines.

Every engine builds its modules from an ``ops`` namespace instead of ``torch.nn``, as ComfyUI does
with ``comfy.ops``. Two namespaces:

- **``default``:** no random initialisation. Weights are allocated and left for the checkpoint;
  constructing a large model takes no time and draws no random numbers. That also keeps loading
  from touching any RNG.
- **``cast``:** the same layers, which also cast their weights to the input's dtype and device on
  every forward. For a component stored in one dtype and computed in another, or kept off the device.

Use ``init_random(module, generator)`` to give a ``default`` module real weights in tests.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import nn
from torch.nn import functional as F


def _cast(t: torch.Tensor, like: torch.Tensor) -> torch.Tensor:
    if t.dtype != like.dtype or t.device != like.device:
        return t.to(device=like.device, dtype=like.dtype, non_blocking=True)
    return t


def _cast_opt(t: torch.Tensor | None, like: torch.Tensor) -> torch.Tensor | None:
    return None if t is None else _cast(t, like)


def _no_init(self: Any) -> None:
    return None


# -- no initialisation ------------------------------------------------------------------------


class Linear(nn.Linear):
    reset_parameters = _no_init


class Conv1d(nn.Conv1d):
    reset_parameters = _no_init


class Conv2d(nn.Conv2d):
    reset_parameters = _no_init


class ConvTranspose1d(nn.ConvTranspose1d):
    reset_parameters = _no_init


class ConvTranspose2d(nn.ConvTranspose2d):
    reset_parameters = _no_init


class Embedding(nn.Embedding):
    reset_parameters = _no_init


class LayerNorm(nn.LayerNorm):
    reset_parameters = _no_init


class GroupNorm(nn.GroupNorm):
    reset_parameters = _no_init


class RMSNorm(nn.RMSNorm):
    reset_parameters = _no_init


class BatchNorm1d(nn.BatchNorm1d):
    reset_parameters = _no_init


class BatchNorm2d(nn.BatchNorm2d):
    reset_parameters = _no_init


# -- casting on every forward -----------------------------------------------------------------


class CastLinear(Linear):
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return F.linear(input, _cast(self.weight, input), _cast_opt(self.bias, input))


class CastConv1d(Conv1d):
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return self._conv_forward(input, _cast(self.weight, input), _cast_opt(self.bias, input))


class CastConv2d(Conv2d):
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return self._conv_forward(input, _cast(self.weight, input), _cast_opt(self.bias, input))


class CastConvTranspose1d(ConvTranspose1d):
    def forward(self, input: torch.Tensor, output_size: list[int] | None = None) -> torch.Tensor:
        if output_size is not None:
            raise NotImplementedError("output_size is not supported by the casting ConvTranspose1d")
        padding: Any = self.padding  # a tuple: never "same" for a transposed conv
        return F.conv_transpose1d(
            input,
            _cast(self.weight, input),
            _cast_opt(self.bias, input),
            self.stride,
            padding,
            self.output_padding,
            self.groups,
            self.dilation,
        )


class CastEmbedding(Embedding):
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        weight = self.weight if self.weight.device == input.device else self.weight.to(input.device)
        return F.embedding(input, weight, self.padding_idx, self.max_norm, self.norm_type, self.scale_grad_by_freq, self.sparse)


class CastLayerNorm(LayerNorm):
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return F.layer_norm(input, self.normalized_shape, _cast_opt(self.weight, input), _cast_opt(self.bias, input), self.eps)


class CastGroupNorm(GroupNorm):
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return F.group_norm(input, self.num_groups, _cast_opt(self.weight, input), _cast_opt(self.bias, input), self.eps)


class CastRMSNorm(RMSNorm):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.rms_norm(x, self.normalized_shape, _cast_opt(self.weight, x), self.eps)


class _Namespace:
    def __init__(self, **layers: type[nn.Module]) -> None:
        self.__dict__.update(layers)


default: Any = _Namespace(
    Linear=Linear,
    Conv1d=Conv1d,
    Conv2d=Conv2d,
    ConvTranspose1d=ConvTranspose1d,
    ConvTranspose2d=ConvTranspose2d,
    Embedding=Embedding,
    LayerNorm=LayerNorm,
    GroupNorm=GroupNorm,
    RMSNorm=RMSNorm,
    BatchNorm1d=BatchNorm1d,
    BatchNorm2d=BatchNorm2d,
)
"""Layers without random initialisation, computing in their weights' dtype."""

cast: Any = _Namespace(
    Linear=CastLinear,
    Conv1d=CastConv1d,
    Conv2d=CastConv2d,
    ConvTranspose1d=CastConvTranspose1d,
    ConvTranspose2d=ConvTranspose2d,
    Embedding=CastEmbedding,
    LayerNorm=CastLayerNorm,
    GroupNorm=CastGroupNorm,
    RMSNorm=CastRMSNorm,
    BatchNorm1d=BatchNorm1d,
    BatchNorm2d=BatchNorm2d,
)
"""The same layers, casting their weights to each input's dtype and device."""


def init_random(module: nn.Module, generator: torch.Generator | None = None, scale: float = 0.02) -> nn.Module:
    """Fill every parameter with seeded noise (tests and tiny configs): norms' weights 1, biases 0."""
    with torch.no_grad():
        for name, p in module.named_parameters():
            leaf = name.rsplit(".", 1)[-1]
            if leaf == "bias":
                p.zero_()
            elif p.dim() == 1:
                p.fill_(1.0)
            else:
                p.copy_(torch.randn(p.shape, generator=generator, dtype=torch.float32) * scale)
        for b in module.buffers():
            if b.is_floating_point() and not torch.isfinite(b).all():
                b.zero_()
    return module
