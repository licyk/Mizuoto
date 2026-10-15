"""Attention backends, what each can take, and whether it runs here.

After ``comfy_kitchen``'s registry: a backend declares its constraints (devices, compute capability, dtypes, head
dims, masks, causal alignment, grouped-query attention, query lengths) and traits (exact, backward, dropout); a call
is described by a ``Call`` (the facts the constraints read, hashable, so a decode loop resolves once), and
``Constraints.check(call)`` gives the reason a backend cannot take it, or None. Whether a backend can run on a
device at all (installed, built for this torch, kernels present, a self-test passed) is its ``probe``, cached per
device for the process.
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, NamedTuple

import torch
from torch import version as torch_version

from mizuoto.runtime.accel.dispatch import MaskInfo

log = logging.getLogger("mizuoto.runtime.accel")

AttnFn = Callable[..., torch.Tensor]
"""``fn(q, k, v, mask, causal, scale, enable_gqa, dropout_p, mask_info)`` in ``(batch, heads, length, dim)``."""

DEVICE_KINDS = ("cuda", "rocm", "xpu", "mps", "cpu")


class Call(NamedTuple):
    """What a backend's constraints read from one call. Lengths only as the facts constraints use."""

    device: str
    """``cuda``, ``rocm``, ``xpu``, ``mps`` or ``cpu``."""
    index: int
    dtype: torch.dtype
    head_dim: int
    v_head_dim: int
    gqa: bool
    """Fewer key/value heads than query heads."""
    single_query: bool
    equal_lengths: bool
    long_query: bool
    """More than 1024 queries (SageAttention 3 is worth it only then)."""
    mask: str
    """``none``, ``bool`` or ``additive``."""
    structured: str
    """What ``mask_info`` holds: ``""`` (none), else ``+``-joined, in this order: ``causal``; ``keys``, ``prefix`` (keys
    padded on the right) or ``suffix`` (on the left); ``window``."""
    scaled: bool
    """A softmax scale other than ``head_dim ** -0.5``."""
    causal: bool
    dropout: bool
    grad: bool


def describe(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: torch.Tensor | None, causal: bool, scale: float | None, dropout_p: float, mask_info: MaskInfo | None) -> Call:
    device = q.device.type
    if device == "cuda" and getattr(torch_version, "hip", None):
        device = "rocm"
    structured = ""
    if mask_info is not None:
        keys = None if mask_info.keys is None else {"right": "prefix", "left": "suffix"}.get(mask_info.padding or "", "keys")
        structured = "+".join(n for n, on in (("causal", mask_info.causal), (keys, keys is not None), ("window", mask_info.window is not None)) if on and n)
    n_q, n_k = q.shape[-2], k.shape[-2]
    return Call(
        device=device,
        index=q.device.index or 0,
        dtype=q.dtype,
        head_dim=q.shape[-1],
        v_head_dim=v.shape[-1],
        gqa=k.shape[-3] != q.shape[-3],
        single_query=n_q == 1,
        equal_lengths=n_q == n_k,
        long_query=n_q > 1024,
        mask="none" if mask is None else "bool" if mask.dtype == torch.bool else "additive",
        structured=structured,
        scaled=scale is not None and abs(scale - q.shape[-1] ** -0.5) > 1e-9,
        causal=causal,
        dropout=dropout_p > 0,
        grad=torch.is_grad_enabled() and (q.requires_grad or k.requires_grad or v.requires_grad),
    )


@dataclass(frozen=True)
class Constraints:
    devices: frozenset[str]
    min_capability: tuple[int, int] | None = None
    """On CUDA (NVIDIA): the lowest compute capability."""
    dtypes: frozenset[torch.dtype] | None = None
    head_dims: frozenset[int] | None = None
    max_head_dim: int | None = None
    head_dim_multiple: int = 1
    same_v_dim: bool = False
    masks: frozenset[str] = frozenset({"none", "bool", "additive"})
    """Dense masks taken as they are."""
    structured: frozenset[str] = frozenset()
    """``mask_info`` forms taken instead of a dense mask (each a ``Call.structured`` value)."""
    causal_unequal: bool = True
    """``causal`` (SDPA's, top-left) with more than one query and unequal lengths (kernels whose causal flag aligns
    bottom-right cannot)."""
    bottom_right: bool = False
    """Takes a ``mask_info`` that is causal (bottom-right, a KV cache's) with unequal lengths."""
    gqa: bool = True
    long_query_only: bool = False
    default_scale: bool = False
    """Only ``head_dim ** -0.5`` (no scale argument)."""

    def check(self, call: Call, capability: tuple[int, int] | None) -> str | None:
        if call.device not in self.devices:
            return f"not on {call.device}"
        if self.min_capability and call.device == "cuda" and (capability is None or capability < self.min_capability):
            return f"needs compute capability {self.min_capability[0]}.{self.min_capability[1]}"
        if self.dtypes is not None and call.dtype not in self.dtypes:
            return f"no {str(call.dtype).removeprefix('torch.')}"
        if self.head_dims is not None and call.head_dim not in self.head_dims:
            return f"head dim {call.head_dim}"
        if self.max_head_dim is not None and call.head_dim > self.max_head_dim:
            return f"head dim {call.head_dim}"
        if call.head_dim % self.head_dim_multiple:
            return f"head dim {call.head_dim}"
        if self.same_v_dim and call.v_head_dim != call.head_dim:
            return "value head dim"
        if call.mask != "none" and call.mask not in self.masks and call.structured not in self.structured:
            return f"{call.mask} mask"
        if call.causal and not call.equal_lengths and not self.causal_unequal:
            return "causal with unequal lengths"
        if call.mask != "none" and call.mask not in self.masks and "causal" in call.structured and not call.equal_lengths and not self.bottom_right:
            return "causal mask over a cache"
        if call.gqa and not self.gqa:
            return "grouped-query attention"
        if self.long_query_only and not call.long_query:
            return "short sequence"
        if self.default_scale and call.scaled:
            return "custom scale"
        return None


@dataclass(eq=False)
class Backend:
    name: str
    fn: AttnFn
    constraints: Constraints
    exact: bool = False
    """Identical to SDPA's result for the same call (only ``sdpa`` and ``math``: they are SDPA)."""
    backward: bool = True
    dropout: bool = False
    deterministic: bool = True
    """Its backward pass repeats bit for bit (forward passes always do)."""
    probe: Callable[[torch.device], str | None] = field(default=lambda device: None)
    """Why it cannot run on ``device`` (None: it can). Called once per device."""
    accepts: Callable[[Call, Any], str | None] | None = None
    """A last check with the call's tensors (``q, k, v, mask``), e.g. torch's own eligibility test; cached by ``Call``."""


class Registry:
    def __init__(self) -> None:
        self._backends: dict[str, Backend] = {}
        self._available: dict[tuple[str, str], str | None] = {}
        self._capability: dict[str, tuple[int, int] | None] = {}
        self._lock = threading.Lock()

    def register(self, backend: Backend) -> None:
        with self._lock:
            self._backends[backend.name] = backend
            for key in [key for key in self._available if key[0] == backend.name]:
                del self._available[key]

    def get(self, name: str) -> Backend | None:
        return self._backends.get(name)

    def names(self) -> list[str]:
        return list(self._backends)

    def unavailable(self, name: str, device: torch.device | str) -> str | None:
        """Why backend ``name`` cannot run on ``device`` (None: it can); probed once per device."""
        device = torch.device(device)
        key = (name, str(device))
        if key in self._available:
            return self._available[key]
        backend = self._backends.get(name)
        if backend is None:
            reason: str | None = "unknown backend"
        else:
            try:
                reason = backend.probe(device)
            except Exception as e:  # a broken install: say so, never fail the caller
                reason = f"probe failed: {type(e).__name__}: {e}"
        with self._lock:
            self._available[key] = reason
        if reason and backend is not None and reason != "not installed":
            log.info("Attention backend %s is not used on %s: %s", name, device, reason)
        return reason

    def capability(self, device: torch.device) -> tuple[int, int] | None:
        key = str(device)
        if key not in self._capability:
            cap: tuple[int, int] | None = None
            if device.type == "cuda" and not getattr(torch_version, "hip", None):
                try:
                    major, minor = torch.cuda.get_device_capability(device)
                    cap = (major, minor)
                except Exception:
                    cap = None
            self._capability[key] = cap
        return self._capability[key]

    def report(self, device: torch.device | str) -> dict[str, str | None]:
        """Every backend and why it cannot run on ``device`` (None: it can)."""
        return {name: self.unavailable(name, device) for name in self._backends}


REGISTRY = Registry()
