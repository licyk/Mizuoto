"""The optional backends' glue (layouts, structured masks, windows, padding, probes), checked on the CPU against fake
packages that compute what the real kernels compute (the kernels themselves need a GPU)."""

from __future__ import annotations

import sys
import types

import pytest

torch = pytest.importorskip("torch")

from torch.nn import functional as F  # noqa: E402

from mizuoto.runtime.accel import MaskInfo  # noqa: E402
from mizuoto.runtime.accel.backends import optional  # noqa: E402


def _bshd_attention(q, k, v, mask, scale):
    """Reference attention in flash's (batch, length, heads, dim) layout with a boolean (n_q, n_k) or (b, 1, n_q, n_k) mask."""
    rep = q.shape[2] // k.shape[2]
    qt, kt, vt = q.transpose(1, 2), k.transpose(1, 2).repeat_interleave(rep, 1), v.transpose(1, 2).repeat_interleave(rep, 1)
    return F.scaled_dot_product_attention(qt, kt, vt, attn_mask=mask, scale=scale).transpose(1, 2)


def _band(n_q, n_k, causal, window):
    q_pos = torch.arange(n_k - n_q, n_k)[:, None]
    k_pos = torch.arange(n_k)[None, :]
    allowed = torch.ones(n_q, n_k, dtype=torch.bool)
    if causal:
        allowed &= k_pos <= q_pos
    left, right = window
    if left >= 0:
        allowed &= k_pos >= q_pos - left
    if right >= 0:
        allowed &= k_pos <= q_pos + right
    return allowed


@pytest.fixture
def fake_flash(monkeypatch):
    calls = []

    def flash_attn_func(q, k, v, dropout_p=0.0, softmax_scale=None, causal=False, window_size=(-1, -1)):
        calls.append(("func", causal, window_size))
        return _bshd_attention(q, k, v, _band(q.shape[1], k.shape[1], causal, window_size), softmax_scale)

    def flash_attn_with_kvcache(q, k_cache, v_cache, cache_seqlens, softmax_scale=None, causal=False, window_size=(-1, -1), cache_leftpad=None):
        calls.append(("kvcache", causal, window_size, cache_seqlens.tolist(), None if cache_leftpad is None else cache_leftpad.tolist()))
        b, n_q, n_k = q.shape[0], q.shape[1], k_cache.shape[1]
        out = []
        for i in range(b):
            start = 0 if cache_leftpad is None else int(cache_leftpad[i])
            end = int(cache_seqlens[i])
            keys = torch.zeros(n_k, dtype=torch.bool)
            keys[start:end] = True
            # flash aligns causal and windows to the end of the valid keys
            band = _band(n_q, end, causal, window_size)
            mask = torch.zeros(n_q, n_k, dtype=torch.bool)
            mask[:, :end] = band
            out.append(_bshd_attention(q[i : i + 1], k_cache[i : i + 1], v_cache[i : i + 1], mask & keys, softmax_scale))
        return torch.cat(out)

    mod = types.SimpleNamespace(flash_attn_func=flash_attn_func, flash_attn_with_kvcache=flash_attn_with_kvcache)
    monkeypatch.setitem(sys.modules, "flash_attn", mod)
    monkeypatch.setitem(sys.modules, "flash_attn_interface", None)
    return calls


def qkv(b=2, h=4, kv=2, n_q=6, n_k=6, d=16, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(b, h, n_q, d, generator=g), torch.randn(b, kv, n_k, d, generator=g), torch.randn(b, kv, n_k, d, generator=g)


def expected(q, k, v, mask, scale=None):
    rep = q.shape[1] // k.shape[1]
    return F.scaled_dot_product_attention(q, k.repeat_interleave(rep, 1), v.repeat_interleave(rep, 1), attn_mask=mask, scale=scale)


def test_flash_takes_causal_windows_and_one_sided_padding(fake_flash):
    q, k, v = qkv()
    out = optional.flash(q, k, v, None, True, 0.2, True, 0.0, None)
    assert torch.allclose(out, expected(q, k, v, torch.ones(6, 6, dtype=torch.bool).tril(), 0.2), atol=1e-5)
    info = MaskInfo(causal=True, window=3)
    dense = info.dense(6, 6, torch.device("cpu"))
    assert torch.allclose(optional.flash(q, k, v, dense, False, None, True, 0.0, info), expected(q, k, v, dense), atol=1e-5)
    assert fake_flash[-1] == ("func", True, (2, 0))
    # one query over a cache padded on the right (VoxCPM's steps), then on the left (Qwen3-TTS's batches)
    q1, k1, v1 = qkv(n_q=1, n_k=9)
    keys = torch.tensor([[True] * 7 + [False] * 2, [True] * 5 + [False] * 4])
    info = MaskInfo(keys=keys, padding="right")
    dense = info.dense(1, 9, torch.device("cpu"))
    assert torch.allclose(optional.flash(q1, k1, v1, dense, False, None, True, 0.0, info), expected(q1, k1, v1, dense), atol=1e-5)
    assert fake_flash[-1][:4] == ("kvcache", False, (-1, -1), [7, 5])
    keys = torch.tensor([[False] * 2 + [True] * 7, [True] * 9])
    info = MaskInfo(keys=keys, causal=True, padding="left")
    q3, k3, v3 = qkv(n_q=3, n_k=9)
    dense = info.dense(3, 9, torch.device("cpu"))
    assert torch.allclose(optional.flash(q3, k3, v3, dense, False, None, True, 0.0, info), expected(q3, k3, v3, dense), atol=1e-5)
    assert fake_flash[-1] == ("kvcache", True, (-1, -1), [9, 9], [2, 0])


@pytest.fixture
def fake_xformers(monkeypatch):
    calls = []

    class LowerTriangularMask:
        pass

    class LowerTriangularFromBottomRightMask:
        pass

    class LowerTriangularFromBottomRightLocalAttentionMask:
        def __init__(self, window):
            self.window = window

    def memory_efficient_attention(q, k, v, attn_bias=None, p=0.0, scale=None):
        calls.append(type(attn_bias).__name__)
        n_q, n_k = q.shape[1], k.shape[1]
        if isinstance(attn_bias, torch.Tensor):
            assert attn_bias.stride(-2) % 8 == 0  # xFormers' alignment
            mask = attn_bias
        elif isinstance(attn_bias, LowerTriangularMask):
            mask = torch.ones(n_q, n_k, dtype=torch.bool).tril()
        elif isinstance(attn_bias, LowerTriangularFromBottomRightLocalAttentionMask):
            mask = _band(n_q, n_k, True, (attn_bias.window - 1, 0))
        elif isinstance(attn_bias, LowerTriangularFromBottomRightMask):
            mask = _band(n_q, n_k, True, (-1, -1))
        else:
            mask = None
        return _bshd_attention(q, k, v, mask, scale)

    ab = types.SimpleNamespace(
        LowerTriangularMask=LowerTriangularMask,
        LowerTriangularFromBottomRightMask=LowerTriangularFromBottomRightMask,
        LowerTriangularFromBottomRightLocalAttentionMask=LowerTriangularFromBottomRightLocalAttentionMask,
    )
    fmha = types.SimpleNamespace(attn_bias=ab)
    ops = types.SimpleNamespace(memory_efficient_attention=memory_efficient_attention, fmha=fmha)
    xf = types.SimpleNamespace(ops=ops)
    for name, module in (("xformers", xf), ("xformers.ops", ops), ("xformers.ops.fmha", fmha), ("xformers.ops.fmha.attn_bias", ab)):
        monkeypatch.setitem(sys.modules, name, module)
    return calls


def test_xformers_takes_dense_masks_as_aligned_biases_and_causal_forms(fake_xformers):
    q, k, v = qkv(n_k=9, n_q=9)
    mask = torch.rand(2, 1, 9, 9, generator=torch.Generator().manual_seed(1)) > 0.3
    mask[..., 0] = True
    assert torch.allclose(optional.xformers(q, k, v, mask, False, None, True, 0.0, None), expected(q, k, v, mask), atol=1e-5)
    assert torch.allclose(optional.xformers(q, k, v, None, True, None, True, 0.0, None), expected(q, k, v, torch.ones(9, 9, dtype=torch.bool).tril()), atol=1e-5)
    info = MaskInfo(causal=True, window=4)
    q1 = q[:, :, -2:]
    dense = info.dense(2, 9, torch.device("cpu"))
    assert torch.allclose(optional.xformers(q1, k, v, dense, False, None, True, 0.0, info), expected(q1, k, v, dense), atol=1e-5)
    assert fake_xformers == ["Tensor", "LowerTriangularMask", "LowerTriangularFromBottomRightLocalAttentionMask"]


def test_sage_gets_its_layout_scale_and_causal_flag(monkeypatch):
    seen = {}

    def sageattn(q, k, v, tensor_layout="HND", is_causal=False, sm_scale=None):
        seen.update(layout=tensor_layout, causal=is_causal, scale=sm_scale, heads=(q.shape[1], k.shape[1]))
        return F.scaled_dot_product_attention(q, k, v, is_causal=is_causal, scale=sm_scale)

    monkeypatch.setitem(sys.modules, "sageattention", types.SimpleNamespace(sageattn=sageattn))
    q, k, v = qkv()
    causal = MaskInfo(causal=True)
    dense = causal.dense(6, 6, torch.device("cpu"))
    assert torch.allclose(optional.sage(q, k, v, dense, False, 0.3, True, 0.0, causal), expected(q, k, v, dense, 0.3), atol=1e-6)
    assert seen == {"layout": "HND", "causal": True, "scale": 0.3, "heads": (4, 4)}


def test_probes_explain_why_a_backend_is_not_used(monkeypatch):
    cuda = torch.device("cuda", 0)
    assert optional._flash_probe(torch.device("cpu")) == "CUDA and ROCm only"
    monkeypatch.setattr(optional, "installed", lambda dist: None)
    assert optional._xformers_probe(cuda) == "not installed"
    monkeypatch.setattr(optional, "installed", lambda dist: "0.0.25")
    assert "broken" in (optional._xformers_probe(cuda) or "")
    monkeypatch.setattr(optional, "installed", lambda dist: "0.0.32.post2")
    monkeypatch.setitem(sys.modules, "xformers", types.SimpleNamespace(_has_cpp_library=False))
    monkeypatch.setitem(sys.modules, "xformers.ops", types.SimpleNamespace())
    assert "no kernels" in (optional._xformers_probe(cuda) or "")
    monkeypatch.setattr(optional, "self_test", lambda fn, device, d, tol: "self-test failed: error 3.1e-01")
    monkeypatch.setitem(sys.modules, "xformers", types.SimpleNamespace(_has_cpp_library=True))
    assert optional._xformers_probe(cuda) == "self-test failed: error 3.1e-01"


def test_self_test_catches_wrong_and_broken_kernels():
    def good(q, k, v, *rest):
        return F.scaled_dot_product_attention(q.float(), k.float(), v.float()).half()

    def wrong(q, k, v, *rest):
        return torch.zeros_like(q)

    def broken(q, k, v, *rest):
        raise RuntimeError("CUDA error: no kernel image is available for execution on the device")

    cpu = torch.device("cpu")
    assert optional.self_test(good, cpu, 64, 2e-3) is None
    assert (optional.self_test(wrong, cpu, 64, 2e-3) or "").startswith("self-test failed: error")
    assert "no kernel image" in (optional.self_test(broken, cpu, 64, 2e-3) or "")


def test_the_fast_chain_skips_what_this_machine_lacks():
    from mizuoto.runtime import accel
    from mizuoto.runtime.accel.policy import AccelPolicy, bind

    p = AccelPolicy("flash")
    q, k, v = (t.to(torch.bfloat16) for t in qkv())
    run = p.run()
    with bind(run):
        accel.attention(q, k, v, site=accel.Site("test.chain"), enable_gqa=True)
    report = run.report()
    assert report["used"] == {"sdpa": 1}
    assert report["fallbacks"]["flash→sdpa"] == {"CUDA and ROCm only": 1}


def test_the_math_kernel_masks_with_a_bool_mask_as_sdpa_does():
    """``math_kernel`` (the ``math`` backend, Fish Speech's step reference) gives what SDPA's math kernel gives through
    ``F.scaled_dot_product_attention``, bit for bit: called directly, the aten kernel would add a bool mask as numbers."""
    import torch.nn.functional as F
    from torch.nn.attention import SDPBackend, sdpa_kernel

    from mizuoto.runtime.accel.dispatch import math_kernel

    g = torch.Generator().manual_seed(0)
    q, k, v = (torch.randn(1, 4, n, 32, generator=g) for n in (3, 9, 9))
    partial = torch.ones(1, 1, 3, 9, dtype=torch.bool)
    partial[..., 6:] = False
    for mask in (partial, torch.ones(1, 1, 3, 9, dtype=torch.bool)):
        with sdpa_kernel(SDPBackend.MATH):
            want = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        assert torch.equal(math_kernel(q, k, v, mask, False, None, False, 0.0), want)
