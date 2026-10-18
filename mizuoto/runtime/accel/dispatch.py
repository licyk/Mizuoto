"""The attention entry point of native code.

Every attention computation of an engine or trainer goes through ``attention(q, k, v, ..., site=SITE)``:

- ``q``, ``k``, ``v`` are ``(batch, heads, length, head_dim)``; ``mask`` is boolean (True attends) or additive,
  broadcastable to ``(batch, heads, q_length, k_length)``; ``causal`` is SDPA's ``is_causal``: aligned top-left (query
  ``i`` sees key ``j`` when ``j ≤ i``), which is the usual causal mask when the lengths are equal. Sites pass it as they
  passed ``is_causal`` (CosyVoice's streamed-text steps feed several tokens over a cache with it); a backend whose
  causal flag aligns bottom-right takes such calls only with equal lengths;
- ``mask_info`` (optional) says what a dense boolean mask is made of (valid keys, causal, a window), so that kernels
  with structured masks can stand in for it; the reference ignores it;
- ``site`` (a ``Site``, one per call site) names the computation and holds its **reference**: the exact computation
  upstream made there (by default SDPA as called with these arguments).

With no policy bound (``compute.attention = auto``) the reference runs and the result is what the code gave before
this module existed, bit for bit. A bound policy (``mizuoto.runtime.accel.policy``) may choose another backend.
"""

from __future__ import annotations

import contextvars
import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, NamedTuple

import torch
from torch.nn import functional as F

from mizuoto.runtime.context import accel_policy

if TYPE_CHECKING:
    from mizuoto.runtime.accel.policy import Run

Reference = Callable[[torch.Tensor, torch.Tensor, torch.Tensor, "torch.Tensor | None", bool, "float | None", bool, float], torch.Tensor]
"""``reference(q, k, v, mask, causal, scale, enable_gqa, dropout_p)``."""

POLICY: contextvars.ContextVar[Run | None] = accel_policy
"""The policy of the request or stage running in this context (``RequestContext.bind``, ``policy.bind``); None runs
every site's reference."""
TRACE_POLICY: list[Run | None] = [None]
"""The policy a ``Compiled`` module traces with (torch.compile cannot read a context variable): its policy's
``trace_run`` (one object per policy, so the graph's guard on it holds across requests), set by ``Compiled``."""
PROCESS_POLICY: list[Run | None] = [None]
"""A training stage's policy (``StageContext.use_attention``), for every thread of its process: autograd runs the
backward pass, and gradient checkpointing's recomputed forwards, on threads of its own."""
DEBUG: bool = False
"""Check every ``mask_info`` against its dense mask (tests)."""


@dataclass(frozen=True, eq=False)
class MaskInfo:
    """What a dense boolean mask means: key ``j`` of row ``b`` is attended by query ``i`` when ``keys[b, j]`` (or
    always, without ``keys``), and, with ``causal``, ``j ≤ i'`` where ``i' = i + n_k − n_q`` (aligned bottom-right, as a
    KV cache's mask is), and, with ``window``, ``j > i' − window``."""

    keys: torch.Tensor | None = None
    """(batch, k_length) boolean: the real keys (padding is False)."""
    causal: bool = False
    window: int | None = None
    padding: str | None = None
    """``right`` or ``left``: each row of ``keys`` is one run of real keys, padded on that side only (kernels that take
    sequence lengths can then stand in for the mask)."""

    def dense(self, n_q: int, n_k: int, device: torch.device) -> torch.Tensor:
        """The boolean mask this describes, (batch or 1, 1, n_q, n_k)."""
        q_pos = torch.arange(n_k - n_q, n_k, device=device)[:, None]
        k_pos = torch.arange(n_k, device=device)[None, :]
        allowed = torch.ones(n_q, n_k, dtype=torch.bool, device=device)
        if self.causal:
            allowed = allowed & (k_pos <= q_pos)
        if self.window is not None:
            allowed = allowed & (k_pos > q_pos - self.window)
        allowed = allowed[None, None]
        if self.keys is not None:
            allowed = allowed & self.keys[:, None, None, :n_k].to(device)
        return allowed


class Masked(NamedTuple):
    """A dense mask with its structure, for code that hands masks down through layers (the Qwen decoders)."""

    dense: torch.Tensor
    info: MaskInfo


def sdpa(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: torch.Tensor | None, causal: bool, scale: float | None, enable_gqa: bool, dropout_p: float) -> torch.Tensor:
    """``F.scaled_dot_product_attention`` with these arguments (the reference of most sites)."""
    return F.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=dropout_p, is_causal=causal, scale=scale, enable_gqa=enable_gqa)


def math_kernel(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: torch.Tensor | None, causal: bool, scale: float | None, enable_gqa: bool, dropout_p: float
) -> torch.Tensor:
    """SDPA's math kernel, called directly (``sdpa_kernel(MATH)`` would switch it for the whole process). A bool mask is
    made additive first, as ``F.scaled_dot_product_attention`` does before it reaches the kernel: given a bool mask, the
    kernel adds it as numbers (1 where true) and masks nothing."""
    if mask is not None and mask.dtype == torch.bool:
        mask = torch.zeros(mask.shape, dtype=q.dtype, device=mask.device).masked_fill_(mask.logical_not(), float("-inf"))
    if enable_gqa:
        return torch.ops.aten._scaled_dot_product_attention_math(q, k, v, attn_mask=mask, dropout_p=dropout_p, is_causal=causal, scale=scale, enable_gqa=True)[0]
    return torch.ops.aten._scaled_dot_product_attention_math(q, k, v, attn_mask=mask, dropout_p=dropout_p, is_causal=causal, scale=scale)[0]


def softmax_attention(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: torch.Tensor | None, causal: bool, scale: float | None, enable_gqa: bool, dropout_p: float
) -> torch.Tensor:
    """``softmax(q·kᵀ·scale + mask)·v`` in the inputs' dtype, as hand-written upstream code does (no dropout)."""
    if enable_gqa and k.shape[1] != q.shape[1]:
        rep = q.shape[1] // k.shape[1]
        k, v = k.repeat_interleave(rep, dim=1), v.repeat_interleave(rep, dim=1)
    if causal:
        mask = causal_mask(q, k, mask)
    scores = (q @ k.transpose(-2, -1)) * (scale if scale is not None else 1.0 / math.sqrt(q.shape[-1]))
    if mask is not None:
        scores = scores.masked_fill(~mask, float("-inf")) if mask.dtype == torch.bool else scores + mask
    return torch.softmax(scores, dim=-1) @ v


def causal_mask(q: torch.Tensor, k: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
    """``mask`` (or nothing) with SDPA's causal mask (top-left) applied."""
    tri = torch.ones(q.shape[-2], k.shape[-2], dtype=torch.bool, device=q.device).tril()
    if mask is None:
        return tri
    return mask & tri if mask.dtype == torch.bool else mask.masked_fill(~tri, float("-inf"))


@dataclass(frozen=True, eq=False)
class Site:
    """One attention computation in native code. ``swappable=False``: only the reference may compute it.
    ``approx_ok=False``: no backend that is not exact (quantised kernels) may."""

    name: str
    reference: Reference = sdpa
    swappable: bool = True
    approx_ok: bool = True

    def __post_init__(self) -> None:
        SITES[self.name] = self


SITES: dict[str, Site] = {}
"""Every site declared in this process, by name (facts, for reports and tests)."""


def attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    site: Site,
    mask: torch.Tensor | None = None,
    causal: bool = False,
    scale: float | None = None,
    enable_gqa: bool = False,
    dropout_p: float = 0.0,
    mask_info: MaskInfo | None = None,
) -> torch.Tensor:
    policy = TRACE_POLICY[0] if torch.compiler.is_compiling() else POLICY.get() or PROCESS_POLICY[0]
    if DEBUG and mask_info is not None and not (q.is_cuda and torch.cuda.is_current_stream_capturing()):
        _check_mask_info(q, k, mask, mask_info)  # (not while a CUDA graph captures: it reads back; the warm-up checked)
    if policy is None:
        return site.reference(q, k, v, mask, causal, scale, enable_gqa, dropout_p)
    return policy.attend(site, q, k, v, mask, causal, scale, enable_gqa, dropout_p, mask_info)


def active(site: Site) -> bool:
    """Whether a policy may compute ``site`` with another backend than its reference (sites may then skip work only
    the reference needs, such as repeating key/value heads)."""
    run = TRACE_POLICY[0] if torch.compiler.is_compiling() else POLICY.get() or PROCESS_POLICY[0]
    return site.swappable and run is not None and not run.transparent


def _check_mask_info(q: torch.Tensor, k: torch.Tensor, mask: Any, info: MaskInfo) -> None:
    if info.keys is not None and info.padding is not None:
        keys = info.keys.bool()
        ordered = keys if info.padding == "right" else keys.flip(-1)
        if not torch.equal(ordered, ordered.cumprod(-1).bool()):
            raise AssertionError(f"mask_info's keys are not padded on the {info.padding} only")
    expected = info.dense(q.shape[-2], k.shape[-2], q.device)
    if (
        mask is None
        or mask.dtype != torch.bool
        or not torch.equal(
            torch.broadcast_to(mask, torch.broadcast_shapes(mask.shape, expected.shape)), torch.broadcast_to(expected, torch.broadcast_shapes(mask.shape, expected.shape))
        )
    ):
        raise AssertionError(f"mask_info {info} does not describe the mask")
