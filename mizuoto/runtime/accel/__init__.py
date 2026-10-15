"""Acceleration: switchable attention backends with compatibility checks and fallbacks.

- ``attention``: the one entry point for attention in native code, ``attention(q, k, v, ..., site=SITE)``;
- ``Site``: a call site and its reference computation (what upstream computed there);
- ``MaskInfo`` / ``Masked``: what a dense mask is made of, for kernels with structured masks;
- ``registry``: the backends, their constraints and whether they run on a device;
- ``policy``: what the settings chose, bound to a request or a stage (``bind``), resolution and fallbacks.
"""

from mizuoto.runtime.accel.dispatch import SITES, Masked, MaskInfo, Site, attention, math_kernel, sdpa, softmax_attention

__all__ = ["SITES", "MaskInfo", "Masked", "Site", "attention", "math_kernel", "sdpa", "softmax_attention"]
