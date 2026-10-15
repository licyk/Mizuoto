"""The optional attention packages: FlashAttention (``flash``), xFormers (``xformers``), SageAttention (``sage``) and
SageAttention 3 (``sage3``). None is a dependency; each is imported the first time a policy asks for
it on a device, checked (its kernels present, its version, the GPU), and tried on a tiny input against SDPA's math
kernel before it computes anything (the self-test: a wheel built for another torch or GPU fails there).

Their kernels are called directly: flash-attn (2.6 or newer) and xFormers register their own operators and autograd
formulas, so a ``torch.compile``'d module traces through them and trainers can use them; SageAttention has no backward
pass and is never used under autograd.
"""

from __future__ import annotations

import importlib
import importlib.metadata
import inspect
from typing import Any

import torch
from torch import version as torch_version

from mizuoto.runtime.accel.dispatch import MaskInfo
from mizuoto.runtime.accel.registry import REGISTRY, Backend, Constraints, Registry

HALF = frozenset({torch.float16, torch.bfloat16})
FLASH_MASKS = frozenset(
    "+".join(part for part in (causal, keys, window) if part)
    for causal in ("", "causal")
    for keys in ("", "prefix", "suffix")
    for window in ("", "window")
    if not (causal and keys == "prefix")  # flash aligns causal to each row's last real key; the mask to the padded end
) - {""}
"""The structured masks FlashAttention takes: causal (bottom-right), keys padded on one side (``cache_seqlens``,
``cache_leftpad``), a window (``window_size``)."""
NVIDIA = frozenset({"cuda"})
GPU = frozenset({"cuda", "rocm"})


def installed(dist: str) -> str | None:
    try:
        return importlib.metadata.version(dist)
    except importlib.metadata.PackageNotFoundError:
        return None


def _version(text: str) -> tuple[int, ...]:
    out = []
    for part in text.split("+")[0].split("."):
        digits = "".join(c for c in part if c.isdigit())
        if not digits:
            break
        out.append(int(digits))
    return tuple(out)


def self_test(fn: Any, device: torch.device, head_dim: int, tolerance: float) -> str | None:
    """Run ``fn`` on a 1 × 2 × 64 × ``head_dim`` attention in fp16 and compare it with SDPA's math kernel in fp32."""
    g = torch.Generator().manual_seed(0)
    q, k, v = (torch.randn(1, 2, 64, head_dim, generator=g) for _ in range(3))
    expected = torch.ops.aten._scaled_dot_product_attention_math(q, k, v)[0]
    try:
        half = [t.to(device, torch.float16) for t in (q, k, v)]
        out = fn(*half, None, False, None, False, 0.0, None).float().cpu()
    except Exception as e:
        return f"self-test failed: {type(e).__name__}: {str(e).strip().splitlines()[0] if str(e).strip() else ''}"
    if not torch.isfinite(out).all():
        return "self-test failed: not finite"
    err = (out - expected).abs().max().item()
    if err > tolerance:
        return f"self-test failed: error {err:.2e}"
    return None


def _cuda(device: torch.device) -> str | None:
    if device.type != "cuda":
        return "CUDA and ROCm only"
    return None


def repeat_heads(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if k.shape[-3] == q.shape[-3]:
        return k, v
    rep = q.shape[-3] // k.shape[-3]
    return k.repeat_interleave(rep, dim=-3), v.repeat_interleave(rep, dim=-3)


def bool_to_bias(mask: torch.Tensor, q: torch.Tensor, k: torch.Tensor, multiple: int) -> torch.Tensor:
    """An additive bias in ``q``'s dtype, (batch, heads, n_q, n_k), its last dimension's storage padded to
    ``multiple`` (xFormers' alignment), as ComfyUI pads it."""
    n_k = k.shape[-2]
    bias = mask if mask.dtype != torch.bool else torch.zeros(mask.shape, dtype=q.dtype, device=q.device).masked_fill_(~mask, float("-inf"))
    bias = bias.to(q.dtype).expand(q.shape[0], q.shape[1], q.shape[-2], n_k)
    padded = torch.empty(*bias.shape[:-1], n_k + (-n_k % multiple), dtype=q.dtype, device=q.device)
    padded[..., :n_k] = bias
    return padded[..., :n_k]


def lengths(info: MaskInfo) -> tuple[torch.Tensor, torch.Tensor | None]:
    """``(cache_seqlens, cache_leftpad)`` for a key mask that pads on one side (``info.padding``)."""
    assert info.keys is not None
    keys = info.keys
    n_k = keys.shape[-1]
    if info.padding == "left":
        return torch.full((keys.shape[0],), n_k, dtype=torch.int32, device=keys.device), (n_k - keys.sum(-1)).to(torch.int32)
    return keys.sum(-1).to(torch.int32), None


# -- FlashAttention -----------------------------------------------------------------------------------------


def _flash_module() -> Any:
    try:
        return importlib.import_module("flash_attn_interface")  # FlashAttention 3 (Hopper)
    except ImportError:
        return importlib.import_module("flash_attn")


def flash(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: Any, causal: bool, scale: float | None, enable_gqa: bool, dropout_p: float, info: MaskInfo | None
) -> torch.Tensor:
    fa = _flash_module()
    window = (-1, -1)
    if info is not None and mask is not None:
        causal = info.causal
        if info.window is not None:
            window = (info.window - 1, 0 if causal else -1)
    qt, kt, vt = (t.transpose(1, 2) for t in (q, k, v))
    if mask is not None and info is not None and info.keys is not None:
        seqlens, leftpad = lengths(info)
        kwargs: dict[str, Any] = {"cache_seqlens": seqlens, "softmax_scale": scale, "causal": causal, "window_size": window}
        if leftpad is not None:
            kwargs["cache_leftpad"] = leftpad
        out = fa.flash_attn_with_kvcache(qt, kt.contiguous(), vt.contiguous(), **kwargs)
    else:
        out = fa.flash_attn_func(qt, kt, vt, dropout_p=dropout_p, softmax_scale=scale, causal=causal, window_size=window)
    if isinstance(out, tuple):  # FlashAttention 3 returns (out, lse)
        out = out[0]
    return out.transpose(1, 2)


def _flash_probe(device: torch.device) -> str | None:
    if (reason := _cuda(device)) is not None:
        return reason
    version = installed("flash-attn") or installed("flash_attn_3")
    if version is None:
        return "not installed"
    try:
        fa = _flash_module()
    except Exception as e:
        return f"does not import: {type(e).__name__}: {e}"
    if "cache_leftpad" not in inspect.signature(fa.flash_attn_with_kvcache).parameters:
        return f"flash-attn {version} is too old (needs 2.6)"
    return self_test(flash, device, 64, 2e-3)


# -- xFormers ------------------------------------------------------------------------------------------------


def xformers(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: Any, causal: bool, scale: float | None, enable_gqa: bool, dropout_p: float, info: MaskInfo | None
) -> torch.Tensor:
    import xformers.ops as xops  # ty: ignore[unresolved-import]
    from xformers.ops.fmha import attn_bias as ab  # ty: ignore[unresolved-import]

    k, v = repeat_heads(q, k, v)
    bias: Any = None
    if info is not None and mask is not None and info.keys is None:
        if info.window is not None:
            bias = ab.LowerTriangularFromBottomRightLocalAttentionMask(info.window)
        elif info.causal:
            bias = ab.LowerTriangularFromBottomRightMask()
    elif mask is not None:
        bias = bool_to_bias(mask, q, k, 8)
        if causal:
            bias = bias.masked_fill(~torch.ones(q.shape[-2], k.shape[-2], dtype=torch.bool, device=q.device).tril(), float("-inf"))
    elif causal:
        bias = ab.LowerTriangularMask()
    out = xops.memory_efficient_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), attn_bias=bias, p=dropout_p, scale=scale)
    return out.transpose(1, 2)


def _xformers_probe(device: torch.device) -> str | None:
    if (reason := _cuda(device)) is not None:
        return reason
    version = installed("xformers")
    if version is None:
        return "not installed"
    if (0, 0, 21) <= _version(version) < (0, 0, 27):
        return f"xformers {version} is broken for large batches (0.0.21–0.0.26)"
    try:
        xf = importlib.import_module("xformers")
        importlib.import_module("xformers.ops")
    except Exception as e:
        return f"does not import: {type(e).__name__}: {e}"
    if not getattr(xf, "_has_cpp_library", True):
        return f"xformers {version} has no kernels for this torch (built for another version)"
    return self_test(xformers, device, 64, 2e-3)


# -- SageAttention -------------------------------------------------------------------------------------------


def sage(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: Any, causal: bool, scale: float | None, enable_gqa: bool, dropout_p: float, info: MaskInfo | None
) -> torch.Tensor:
    from sageattention import sageattn  # ty: ignore[unresolved-import]

    if mask is not None:  # only a causal mask over equal lengths gets here (the constraints)
        causal = True
    k, v = repeat_heads(q, k, v)
    return sageattn(q.contiguous(), k.contiguous(), v.contiguous(), tensor_layout="HND", is_causal=causal, sm_scale=scale)


def _sage_probe(device: torch.device) -> str | None:
    if device.type != "cuda" or getattr(torch_version, "hip", None):
        return "NVIDIA GPUs only"
    version = installed("sageattention")
    if version is None:
        return "not installed"
    try:
        importlib.import_module("sageattention")
    except Exception as e:
        return f"does not import: {type(e).__name__}: {e}"
    return self_test(sage, device, 64, 2e-2)


def sage3(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: Any, causal: bool, scale: float | None, enable_gqa: bool, dropout_p: float, info: MaskInfo | None
) -> torch.Tensor:
    from sageattn3 import sageattn3_blackwell  # ty: ignore[unresolved-import]

    k, v = repeat_heads(q, k, v)
    return sageattn3_blackwell(q.contiguous(), k.contiguous(), v.contiguous(), is_causal=causal)


def _sage3_probe(device: torch.device) -> str | None:
    if device.type != "cuda" or getattr(torch_version, "hip", None):
        return "NVIDIA GPUs only"
    try:
        importlib.import_module("sageattn3")
    except ImportError:
        return "not installed"
    except Exception as e:
        return f"does not import: {type(e).__name__}: {e}"
    return None  # its kernels need sequences over 1024 (the constraints): no tiny self-test


def register(registry: Registry = REGISTRY) -> None:
    registry.register(
        Backend(
            "flash",
            flash,
            Constraints(
                devices=GPU,
                min_capability=(8, 0),
                dtypes=HALF,
                max_head_dim=256,
                head_dim_multiple=8,
                same_v_dim=True,
                masks=frozenset(),
                structured=FLASH_MASKS,
                causal_unequal=False,
                bottom_right=True,
            ),
            dropout=True,
            deterministic=False,
            probe=_flash_probe,
        )
    )
    registry.register(
        Backend(
            "xformers",
            xformers,
            Constraints(
                devices=GPU,
                min_capability=(7, 5),
                dtypes=frozenset({torch.float16, torch.bfloat16, torch.float32}),
                max_head_dim=256,
                causal_unequal=False,
                bottom_right=True,
                structured=frozenset({"causal", "causal+window"}),
            ),
            dropout=True,
            deterministic=False,
            probe=_xformers_probe,
        )
    )
    registry.register(
        Backend(
            "sage",
            sage,
            Constraints(
                devices=NVIDIA,
                min_capability=(8, 0),
                dtypes=HALF,
                head_dims=frozenset({64, 96, 128}),
                same_v_dim=True,
                masks=frozenset(),
                structured=frozenset({"causal"}),
                causal_unequal=False,
            ),
            backward=False,
            probe=_sage_probe,
        )
    )
    registry.register(
        Backend(
            "sage3",
            sage3,
            Constraints(
                devices=NVIDIA,
                min_capability=(10, 0),
                dtypes=HALF,
                max_head_dim=255,
                same_v_dim=True,
                masks=frozenset(),
                causal_unequal=False,
                long_query_only=True,
                default_scale=True,
            ),
            backward=False,
            probe=_sage3_probe,
        )
    )
