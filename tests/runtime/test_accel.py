"""The attention entry point, its backends, policies and fallbacks."""

from __future__ import annotations

import logging
import threading

import pytest

torch = pytest.importorskip("torch")

from torch.nn import functional as F  # noqa: E402

from mizuoto.runtime import accel  # noqa: E402
from mizuoto.runtime.accel import dispatch as attn_mod  # noqa: E402
from mizuoto.runtime.accel.policy import FAST, AccelError, AccelPolicy, bind, tracing  # noqa: E402
from mizuoto.runtime.accel.registry import Backend, Constraints, Registry  # noqa: E402

ALL = frozenset({"cuda", "rocm", "xpu", "mps", "cpu"})


def qkv(heads=4, kv_heads=4, n_q=6, n_k=6, d=16, seed=0, dtype=torch.float32):
    g = torch.Generator().manual_seed(seed)
    q = torch.randn(2, heads, n_q, d, generator=g).to(dtype)
    k = torch.randn(2, kv_heads, n_k, d, generator=g).to(dtype)
    v = torch.randn(2, kv_heads, n_k, d, generator=g).to(dtype)
    return q, k, v


SITE = accel.Site("test.site")
MATH_SITE = accel.Site("test.math", reference=accel.math_kernel)


def test_unbound_calls_run_the_reference_bit_for_bit():
    q, k, v = qkv()
    mask = torch.rand(2, 1, 6, 6, generator=torch.Generator().manual_seed(1)) > 0.3
    mask[..., 0] = True
    assert torch.equal(accel.attention(q, k, v, site=SITE), F.scaled_dot_product_attention(q, k, v))
    assert torch.equal(accel.attention(q, k, v, site=SITE, causal=True), F.scaled_dot_product_attention(q, k, v, is_causal=True))
    assert torch.equal(accel.attention(q, k, v, site=SITE, mask=mask, scale=0.3), F.scaled_dot_product_attention(q, k, v, attn_mask=mask, scale=0.3))
    q, k, v = qkv(heads=8, kv_heads=2, n_q=1, n_k=9)
    assert torch.equal(accel.attention(q, k, v, site=SITE, enable_gqa=True), F.scaled_dot_product_attention(q, k, v, enable_gqa=True))
    expected = torch.ops.aten._scaled_dot_product_attention_math(q, k.repeat_interleave(4, 1), v.repeat_interleave(4, 1))[0]
    assert torch.equal(accel.attention(q, k.repeat_interleave(4, 1), v.repeat_interleave(4, 1), site=MATH_SITE), expected)


def test_causal_is_sdpas_top_left_flag():
    """Sites pass ``causal`` as they passed ``is_causal`` (CosyVoice's streamed-text steps rely on it)."""
    q, k, v = qkv(n_q=3, n_k=7)
    top_left = torch.ones(3, 7, dtype=torch.bool).tril()
    expected = F.scaled_dot_product_attention(q, k, v, attn_mask=top_left)
    assert torch.allclose(accel.attention(q, k, v, site=SITE, causal=True), expected, atol=1e-6)
    assert torch.allclose(accel.softmax_attention(q, k, v, None, True, None, False, 0.0), expected, atol=1e-6)
    assert torch.allclose(accel.math_kernel(q, k, v, None, True, None, False, 0.0), expected, atol=1e-6)


def test_softmax_attention_matches_sdpa():
    q, k, v = qkv(heads=8, kv_heads=2)
    expected = F.scaled_dot_product_attention(q, k.repeat_interleave(4, 1), v.repeat_interleave(4, 1), is_causal=True)
    assert torch.allclose(accel.softmax_attention(q, k, v, None, True, None, True, 0.0), expected, atol=1e-6)
    bias = torch.randn(1, 1, 6, 6, generator=torch.Generator().manual_seed(3))
    expected = F.scaled_dot_product_attention(q, k.repeat_interleave(4, 1), v.repeat_interleave(4, 1), attn_mask=bias)
    assert torch.allclose(accel.softmax_attention(q, k, v, bias, False, None, True, 0.0), expected, atol=1e-6)


def test_mask_info_describes_dense_masks():
    from mizuoto.runtime.lm.qwen import mask_for

    keys = torch.tensor([[0, 1, 1, 1, 1], [1, 1, 1, 1, 1]])
    for q_len, past, window in ((5, 0, None), (1, 4, None), (2, 3, None), (5, 0, 3)):
        m = mask_for(keys, q_len, past, sliding_window=window)
        assert m is not None
        assert torch.equal(m.info.dense(q_len, past + q_len, torch.device("cpu")).expand_as(m.dense), m.dense)
    info = accel.MaskInfo(causal=True, window=2)
    assert info.dense(4, 4, torch.device("cpu"))[0, 0].int().tolist() == [[1, 0, 0, 0], [1, 1, 0, 0], [0, 1, 1, 0], [0, 0, 1, 1]]
    assert accel.MaskInfo(keys=torch.tensor([[True, True, False]])).dense(2, 3, torch.device("cpu"))[0, 0].int().tolist() == [[1, 1, 0], [1, 1, 0]]


def test_debug_mode_catches_a_wrong_mask_info(monkeypatch):
    monkeypatch.setattr(attn_mod, "DEBUG", True)
    q, k, v = qkv()
    causal = torch.ones(6, 6, dtype=torch.bool).tril()[None, None]
    accel.attention(q, k, v, site=SITE, mask=causal, mask_info=accel.MaskInfo(causal=True))
    with pytest.raises(AssertionError):
        accel.attention(q, k, v, site=SITE, mask=causal, mask_info=accel.MaskInfo(causal=True, window=2))


# -- policies with fake backends ------------------------------------------------------------------------


class Fake:
    """A backend that computes SDPA and counts its calls; ``fail`` makes it raise."""

    def __init__(self, fail: BaseException | None = None) -> None:
        self.calls = 0
        self.fail = fail

    def __call__(self, q, k, v, mask, causal, scale, gqa, dropout, info):
        self.calls += 1
        if self.fail is not None:
            raise self.fail
        return F.scaled_dot_product_attention(q, k, v, attn_mask=mask, is_causal=causal, scale=scale, enable_gqa=gqa) + 1.0


def registry_with(*backends: Backend) -> Registry:
    from mizuoto.runtime.accel.backends import torch_sdpa

    reg = Registry()
    torch_sdpa.register(reg)
    for b in backends:
        reg.register(b)
    return reg


def policy(name, *backends, strict=False, chain=None):
    p = AccelPolicy(name, strict=strict, registry=registry_with(*backends))
    if chain is not None:
        p.chain = lambda call: chain  # ty: ignore[invalid-assignment]
    return p


def test_a_policy_runs_the_first_backend_that_takes_the_call():
    fake = Fake()
    p = policy("fake", Backend("fake", fake, Constraints(devices=ALL)), chain=("fake", "sdpa"))
    q, k, v = qkv()
    run = p.run()
    with bind(run):
        out = accel.attention(q, k, v, site=SITE)
        accel.attention(q, k, v, site=SITE)
    assert torch.equal(out, F.scaled_dot_product_attention(q, k, v) + 1.0)
    assert fake.calls == 2
    assert run.report() == {"policy": "fake", "used": {"fake": 2}, "fallbacks": {}}


def test_constraints_send_calls_down_the_chain_and_say_why(caplog):
    fake = Fake()
    p = policy("fake", Backend("fake", fake, Constraints(devices=ALL, head_dims=frozenset({64}), masks=frozenset())), chain=("fake", "sdpa"))
    q, k, v = qkv(d=16)
    run = p.run()
    with caplog.at_level(logging.WARNING, logger="mizuoto.runtime.accel"), bind(run):
        out = accel.attention(q, k, v, site=SITE)
        accel.attention(q, k, v, site=SITE)
    assert torch.equal(out, F.scaled_dot_product_attention(q, k, v))
    assert fake.calls == 0
    assert run.report()["used"] == {"sdpa": 2}
    assert run.report()["fallbacks"] == {"fake→sdpa": {"head dim 16": 2}}
    assert sum("head dim 16" in r.message for r in caplog.records) == 1  # once per site, backend and reason
    assert run.warnings() == ["attention fake→sdpa: head dim 16 (2 calls)"]


def test_masks_grad_and_dropout_are_checked():
    fake = Fake()
    p = policy("fake", Backend("fake", fake, Constraints(devices=ALL, masks=frozenset()), backward=False), chain=("fake",))
    q, k, v = qkv()
    run = p.run()
    with bind(run):
        accel.attention(q, k, v, site=SITE, mask=torch.ones(6, 6, dtype=torch.bool))
        accel.attention(q.requires_grad_(), k, v, site=SITE)
        accel.attention(q.detach(), k, v, site=SITE, dropout_p=0.1)
        with torch.no_grad():
            accel.attention(q, k, v, site=SITE)
    assert fake.calls == 1
    assert set(run.report()["fallbacks"]["fake→reference"]) == {"bool mask", "no backward", "dropout"}


def test_a_structured_mask_lets_a_maskless_kernel_take_the_call():
    fake = Fake()
    p = policy("fake", Backend("fake", fake, Constraints(devices=ALL, masks=frozenset(), structured=frozenset({"causal"}), causal_unequal=False)), chain=("fake",))
    q, k, v = qkv()
    causal = torch.ones(6, 6, dtype=torch.bool).tril()[None, None]
    with bind(p.run()):
        accel.attention(q, k, v, site=SITE, mask=causal, mask_info=accel.MaskInfo(causal=True))
        q1 = q[:, :, :1]
        accel.attention(q1, k, v, site=SITE, mask=causal[..., -1:, :], mask_info=accel.MaskInfo(causal=True))  # over a cache
    assert fake.calls == 1


def test_sites_that_refuse_quantised_attention_or_are_unswappable_keep_their_reference():
    fake = Fake()
    p = policy("fake", Backend("fake", fake, Constraints(devices=ALL), approximate=True), chain=("fake",))
    q, k, v = qkv()
    exact = accel.Site("test.exact", approx_ok=False)
    fixed = accel.Site("test.fixed", reference=accel.softmax_attention, swappable=False)
    run = p.run()
    with bind(run):
        assert torch.equal(accel.attention(q, k, v, site=exact), F.scaled_dot_product_attention(q, k, v))
        assert torch.equal(accel.attention(q, k, v, site=fixed), accel.softmax_attention(q, k, v, None, False, None, False, 0.0))
    assert fake.calls == 0
    assert run.report()["used"] == {"reference": 2}


def test_a_failing_kernel_falls_back_once_and_is_not_tried_again():
    fake = Fake(fail=RuntimeError("no kernel image"))
    p = policy("fake", Backend("fake", fake, Constraints(devices=ALL)), chain=("fake", "sdpa"))
    q, k, v = qkv()
    run = p.run()
    with bind(run):
        out = accel.attention(q, k, v, site=SITE)
        accel.attention(q, k, v, site=SITE)
    assert torch.equal(out, F.scaled_dot_product_attention(q, k, v))
    assert fake.calls == 1
    assert run.report()["used"] == {"sdpa": 2}


def test_out_of_memory_is_not_a_fallback():
    fake = Fake(fail=torch.OutOfMemoryError("CUDA out of memory"))
    p = policy("fake", Backend("fake", fake, Constraints(devices=ALL)), chain=("fake", "sdpa"))
    q, k, v = qkv()
    with bind(p.run()), pytest.raises(torch.OutOfMemoryError):
        accel.attention(q, k, v, site=SITE)


def test_strict_mode_refuses_instead_of_falling_back():
    p = policy("fake", Backend("fake", Fake(), Constraints(devices=frozenset({"cuda"}))), strict=True, chain=("fake", "sdpa"))
    q, k, v = qkv()
    with bind(p.run()), pytest.raises(AccelError) as e:
        accel.attention(q, k, v, site=SITE)
    assert e.value.code == "compute_unavailable"
    assert e.value.detail["reason"] == "not on cpu"
    with bind(p.run()), pytest.raises(AccelError):  # every time, not only when the call is first resolved
        accel.attention(q, k, v, site=SITE)


def test_unavailable_backends_are_probed_once_per_device():
    probes = []

    def probe(device):
        probes.append(device)
        return "not installed"

    fake = Fake()
    p = policy("fake", Backend("fake", fake, Constraints(devices=ALL), probe=probe), chain=("fake", "sdpa"))
    q, k, v = qkv()
    run = p.run()
    with bind(run):
        accel.attention(q, k, v, site=SITE)
        accel.attention(q, k, v[..., :8], site=SITE)
    assert len(probes) == 1 and fake.calls == 0
    assert run.report()["fallbacks"] == {"fake→sdpa": {"not installed": 2}}  # the backend asked for by name


def test_a_probe_that_raises_makes_the_backend_unavailable():
    def probe(device):
        raise ImportError("libcudart.so.12: cannot open shared object file")

    reg = registry_with(Backend("fake", Fake(), Constraints(devices=ALL), probe=probe))
    assert (reg.unavailable("fake", "cpu") or "").startswith("probe failed: ImportError")


def test_bindings_do_not_leak_across_threads():
    fake = Fake()
    p = policy("fake", Backend("fake", fake, Constraints(devices=ALL)), chain=("fake",))
    q, k, v = qkv()
    seen = []
    with bind(p.run()):
        thread = threading.Thread(target=lambda: seen.append(accel.attention(q, k, v, site=SITE)))
        thread.start()
        thread.join()
        inside = accel.attention(q, k, v, site=SITE)
    assert torch.equal(seen[0], F.scaled_dot_product_attention(q, k, v))
    assert torch.equal(inside, seen[0] + 1.0)
    assert fake.calls == 1


def test_suspended_generators_keep_their_own_policy():
    fake = Fake()
    p = policy("fake", Backend("fake", fake, Constraints(devices=ALL)), chain=("fake",))
    q, k, v = qkv()

    def stream(run):
        for _ in range(2):
            with bind(run):
                out = accel.attention(q, k, v, site=SITE)
            yield out

    accelerated, plain = stream(p.run()), stream(None)
    results = [next(accelerated), next(plain), next(accelerated), next(plain)]
    assert [bool(torch.equal(r, F.scaled_dot_product_attention(q, k, v))) for r in results] == [False, True, False, True]


def test_compiled_modules_see_the_tracing_policy():
    fake = Fake()
    p = policy("fake", Backend("fake", fake, Constraints(devices=ALL)), chain=("fake",))
    q, k, v = qkv()

    def fn(q, k, v):
        return accel.attention(q, k, v, site=SITE)

    compiled = torch.compile(fn, backend="eager")
    with tracing(p.run()):
        out = compiled(q, k, v)
    assert torch.equal(out, F.scaled_dot_product_attention(q, k, v) + 1.0)
    torch._dynamo.reset()
    assert torch.equal(torch.compile(fn, backend="eager", fullgraph=True)(q, k, v), F.scaled_dot_product_attention(q, k, v))
    torch._dynamo.reset()


def test_fast_chain_on_the_cpu_is_sdpa_and_identical():
    p = AccelPolicy(FAST)
    q, k, v = qkv()
    run = p.run()
    with bind(run):
        out = accel.attention(q, k, v, site=SITE, causal=True)
    assert torch.equal(out, F.scaled_dot_product_attention(q, k, v, is_causal=True))
    assert run.report()["used"] == {"sdpa": 1}


def test_torch_kernels_pinned_off_cuda_say_so():
    from mizuoto.runtime.accel.registry import REGISTRY

    AccelPolicy(FAST)  # registers the backends
    report = REGISTRY.report("cpu")
    assert report["sdpa"] is None and report["math"] is None
    assert report["sdpa_flash"] == "CUDA and ROCm only"


def test_a_site_refusing_quantised_attention_still_takes_other_kernels():
    """``approx_ok=False`` keeps quantised (approximate) backends away, not every non-exact one: FlashAttention or
    xFormers still compute an autoregressive model's attention (A6: SageAttention made them run on without end)."""
    lm = accel.Site("test.lm", approx_ok=False)
    q, k, v = qkv()
    for approximate, used in ((True, "reference"), (False, "fake")):
        fake = Fake()
        p = policy("fake", Backend("fake", fake, Constraints(devices=ALL), approximate=approximate), chain=("fake",))
        run = p.run()
        with bind(run):
            accel.attention(q, k, v, site=lm)
        assert run.report()["used"] == {used: 1}


def test_fast_on_nvidia_uses_flash_only_where_keys_are_padded():
    """A7: ``fast`` on CUDA sends fp16/bf16 calls with padded keys to FlashAttention (it skips them) and every other
    call to SDPA; SageAttention and xFormers only by name; fp32 SDPA."""
    from mizuoto.runtime.accel.registry import Call

    p = AccelPolicy(FAST)

    def call(structured, dtype=torch.float16):
        return Call("cuda", 0, dtype, 64, 64, False, True, False, False, "bool" if structured else "none", structured, False, False, False, False)

    assert p.chain(call("prefix")) == ("flash", "sdpa") == p.chain(call("causal+suffix"))
    assert p.chain(call("")) == ("sdpa",) == p.chain(call("causal")) == p.chain(call("prefix", torch.float32))
    assert AccelPolicy("xformers").chain(call("")) == ("xformers", "sdpa")
