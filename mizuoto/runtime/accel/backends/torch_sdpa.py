"""torch's attention: SDPA as torch chooses its kernel (``sdpa``), its math kernel (``math``) and its fused kernels
pinned (``sdpa_flash``, ``sdpa_efficient``, ``sdpa_cudnn``), called through their aten operators so that nothing
process-wide is switched (``sdpa_kernel`` sets flags for every thread)."""

from __future__ import annotations

from typing import Any

import torch

from mizuoto.runtime.accel.dispatch import MaskInfo, causal_mask, math_kernel, sdpa
from mizuoto.runtime.accel.registry import REGISTRY, Backend, Call, Constraints, Registry

ALL = frozenset({"cuda", "rocm", "xpu", "mps", "cpu"})
HALF = frozenset({torch.float16, torch.bfloat16})


def _sdpa(q: Any, k: Any, v: Any, mask: Any, causal: bool, scale: float | None, enable_gqa: bool, dropout_p: float, info: MaskInfo | None) -> torch.Tensor:
    return sdpa(q, k, v, mask, causal, scale, enable_gqa, dropout_p)


def _math(q: Any, k: Any, v: Any, mask: Any, causal: bool, scale: float | None, enable_gqa: bool, dropout_p: float, info: MaskInfo | None) -> torch.Tensor:
    return math_kernel(q, k, v, mask, causal, scale, enable_gqa, dropout_p)


def repeat_heads(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if k.shape[-3] == q.shape[-3]:
        return k, v
    rep = q.shape[-3] // k.shape[-3]
    return k.repeat_interleave(rep, dim=-3), v.repeat_interleave(rep, dim=-3)


def additive(mask: torch.Tensor | None, q: torch.Tensor, k: torch.Tensor, causal: bool) -> torch.Tensor | None:
    """An additive bias in ``q``'s dtype (SDPA's causal mask folded in unless the call is causal alone with equal
    lengths); its last dimension padded to a multiple of 16 and sliced back, as SDPA does for the memory-efficient
    kernel."""
    if causal and (mask is not None or q.shape[-2] != k.shape[-2]):
        mask = causal_mask(q, k, mask)
    if mask is None:
        return None
    if mask.dtype == torch.bool:
        mask = torch.zeros(mask.shape, dtype=q.dtype, device=q.device).masked_fill_(~mask, float("-inf"))
    else:
        mask = mask.to(q.dtype)
    n_k = k.shape[-2]
    pad = -n_k % 16
    if pad:
        mask = torch.nn.functional.pad(mask.expand(*mask.shape[:-1], n_k), (0, pad))[..., :n_k]
    return mask.expand(q.shape[0], q.shape[1], q.shape[-2], n_k)


def _unstructured(mask: Any, causal: bool, info: MaskInfo | None) -> tuple[Any, bool]:
    """A dense mask that is only causal becomes ``causal`` (flash takes no mask)."""
    if mask is not None and info is not None and info.causal and info.keys is None and info.window is None:
        return None, True
    return mask, causal


def _flash(q: Any, k: Any, v: Any, mask: Any, causal: bool, scale: float | None, enable_gqa: bool, dropout_p: float, info: MaskInfo | None) -> torch.Tensor:
    mask, causal = _unstructured(mask, causal, info)
    k, v = repeat_heads(q, k, v)
    return torch.ops.aten._scaled_dot_product_flash_attention(q, k, v, dropout_p, causal, False, scale=scale)[0]


def _efficient(q: Any, k: Any, v: Any, mask: Any, causal: bool, scale: float | None, enable_gqa: bool, dropout_p: float, info: MaskInfo | None) -> torch.Tensor:
    k, v = repeat_heads(q, k, v)
    bias = additive(mask, q, k, causal)
    grad = torch.is_grad_enabled() and (q.requires_grad or k.requires_grad or v.requires_grad)
    is_causal = causal and bias is None
    return torch.ops.aten._scaled_dot_product_efficient_attention(q, k, v, bias, grad, dropout_p, is_causal, scale=scale)[0]


def _cudnn(q: Any, k: Any, v: Any, mask: Any, causal: bool, scale: float | None, enable_gqa: bool, dropout_p: float, info: MaskInfo | None) -> torch.Tensor:
    k, v = repeat_heads(q, k, v)
    bias = additive(mask, q, k, causal)
    grad = torch.is_grad_enabled() and (q.requires_grad or k.requires_grad or v.requires_grad)
    is_causal = causal and bias is None
    return torch.ops.aten._scaled_dot_product_cudnn_attention(q, k, v, bias, grad, dropout_p, is_causal, False, scale=scale)[0]


def _eligible(check: str) -> Any:
    """torch's own test of whether its fused kernel takes these tensors (``can_use_*_attention``)."""

    def accepts(call: Call, tensors: Any) -> str | None:
        q, k, v, mask = tensors
        k, v = repeat_heads(q, k, v)
        if check != "flash" and mask is not None:
            mask = additive(mask, q, k, call.causal)
        elif check == "flash":
            mask = None
        try:
            params = torch.backends.cuda.SDPAParams(q, k, v, mask, 0.1 if call.dropout else 0.0, call.causal and call.equal_lengths, False)
            ok = getattr(torch.backends.cuda, f"can_use_{check}_attention")(params, False)
        except Exception as e:
            return f"torch's check failed: {e}"
        return None if ok else f"torch's {check} kernel does not take this call"

    return accepts


def _cuda_only(device: torch.device) -> str | None:
    return None if device.type == "cuda" else "CUDA and ROCm only"


def register(registry: Registry = REGISTRY) -> None:
    registry.register(Backend("sdpa", _sdpa, Constraints(devices=ALL), exact=True, dropout=True, deterministic=False))
    registry.register(Backend("math", _math, Constraints(devices=ALL), exact=True, dropout=True))
    gpu = frozenset({"cuda", "rocm"})
    registry.register(
        Backend(
            "sdpa_flash",
            _flash,
            Constraints(
                devices=gpu,
                min_capability=(8, 0),
                dtypes=HALF,
                max_head_dim=256,
                head_dim_multiple=8,
                same_v_dim=True,
                masks=frozenset(),
                structured=frozenset({"causal"}),
                causal_unequal=False,
            ),
            dropout=True,
            deterministic=False,
            probe=_cuda_only,
            accepts=_eligible("flash"),
        )
    )
    registry.register(
        Backend(
            "sdpa_efficient",
            _efficient,
            Constraints(devices=gpu, dtypes=frozenset({torch.float16, torch.bfloat16, torch.float32})),
            dropout=True,
            deterministic=False,
            probe=_cuda_only,
            accepts=_eligible("efficient"),
        )
    )
    if hasattr(torch.backends.cuda, "can_use_cudnn_attention"):
        registry.register(
            Backend(
                "sdpa_cudnn",
                _cudnn,
                Constraints(devices=frozenset({"cuda"}), min_capability=(8, 0), dtypes=HALF, max_head_dim=128, head_dim_multiple=8, same_v_dim=True),
                dropout=True,
                deterministic=False,
                probe=_cuda_only,
                accepts=_eligible("cudnn"),
            )
        )
