"""The native runtime's parts: context, manager, ops, attention, loading."""

from __future__ import annotations

import random
import threading

import pytest

torch = pytest.importorskip("torch")

from mizuoto.runtime import loading, ops  # noqa: E402
from mizuoto.runtime.context import Cancelled, RequestContext, check_cancel, current  # noqa: E402
from mizuoto.runtime.device import DeviceService  # noqa: E402
from mizuoto.runtime.engine import EngineError, engine_ids, get_engine  # noqa: E402
from mizuoto.runtime.handle import ModelHandle  # noqa: E402
from mizuoto.runtime.snapshot import diff, snapshot  # noqa: E402

# -- the request context ----------------------------------------------------------------


def test_generators_reproduce_the_seeded_default_stream():
    ctx = RequestContext("r", seed=1234)
    torch.manual_seed(1234)
    expected = torch.randn(64), torch.multinomial(torch.ones(10), 3)
    g = ctx.generator_for("cpu")
    assert torch.equal(torch.randn(64, generator=g), expected[0])
    assert torch.equal(torch.multinomial(torch.ones(10), 3, generator=g), expected[1])
    random.seed(1234)
    assert ctx.rng.random() == random.random()


def test_generators_are_per_request_and_reseedable():
    a, b = RequestContext("a", seed=7), RequestContext("b", seed=7)
    first = torch.rand(8, generator=a.generator)
    assert torch.equal(first, torch.rand(8, generator=b.generator))
    a.reseed(7)
    assert torch.equal(torch.rand(8, generator=a.generator), first)
    assert RequestContext("c").seed is None


def test_cancel_is_a_base_exception_seen_through_the_bound_context():
    ctx = RequestContext("r")
    with ctx.bind():
        assert current() is ctx
        check_cancel()
        ctx.cancel()
        with pytest.raises(Cancelled):
            try:
                check_cancel()
            except Exception:  # model code's ``except Exception`` must not swallow a cancel
                pytest.fail("Cancelled was caught as an Exception")
    assert current() is None
    check_cancel()  # no request bound: nothing to cancel


def test_text_pieces_end_on_final_or_cancel():
    import queue

    q: queue.Queue[tuple[str, bool]] = queue.Queue()
    ctx = RequestContext("r", text_input=q)
    q.put(("Hello ", False))
    q.put(("world.", True))
    assert list(ctx.text_pieces()) == ["Hello ", "world."]
    ctx2 = RequestContext("r2", text_input=queue.Queue())
    threading.Timer(0.15, ctx2.cancel).start()
    assert list(ctx2.text_pieces()) == []


# -- devices, handles and the manager ---------------------------------------------------------


def test_real_handle_moves_a_module():
    module = torch.nn.Linear(4, 4)
    h = ModelHandle(module, name="lin", load_device="cpu")
    assert h.size_bytes == (16 + 4) * 4 and h.loaded
    h.free()
    assert h.device == "none"


def test_dtype_choice():
    d = DeviceService()
    assert d.pick_dtype("cpu") == "fp32"
    assert d.pick_dtype("cpu", supported=("bf16", "fp16"), cpu="bf16") == "bf16"
    assert d.pick_dtype("cpu", requested="bf16") == "bf16"
    with pytest.raises(ValueError):
        d.pick_dtype("cpu", requested="fp16", supported=("fp32",))


# -- ops, attention, loading ------------------------------------------------------------------


def test_default_ops_skip_init_and_draw_no_random_numbers():
    torch.manual_seed(0)
    before = torch.random.get_rng_state()
    ops.default.Linear(256, 256)
    ops.default.Conv1d(8, 8, 3)
    ops.default.Embedding(100, 16)
    assert torch.equal(before, torch.random.get_rng_state())


def test_cast_ops_compute_in_the_inputs_dtype():
    lin = ops.init_random(ops.cast.Linear(8, 4), torch.Generator().manual_seed(0)).to(torch.bfloat16)
    x = torch.randn(2, 8, generator=torch.Generator().manual_seed(1))
    y = lin(x)
    assert y.dtype == torch.float32
    assert torch.allclose(y, x @ lin.weight.float().T + lin.bias.float())


def test_weight_norm_folds_in_both_formats(tmp_path):
    conv = torch.nn.utils.parametrizations.weight_norm(torch.nn.Conv1d(3, 5, 3))
    old = torch.nn.utils.weight_norm(torch.nn.Conv1d(3, 5, 3))
    x = torch.randn(1, 3, 10)
    for module in (conv, old):
        folded = loading.fold_weight_norm(module.state_dict())
        plain = torch.nn.Conv1d(3, 5, 3)
        loading.load_into(plain, folded)
        assert torch.allclose(plain(x), module(x), atol=1e-6)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU")
def test_weight_norm_folded_on_a_gpu_is_what_a_live_one_computes_there():
    gen = torch.Generator().manual_seed(0)
    for make in (torch.nn.utils.weight_norm, torch.nn.utils.parametrizations.weight_norm):
        live = make(torch.nn.Conv1d(64, 96, 5))
        with torch.no_grad():
            for p in live.parameters():
                p.copy_(torch.randn(p.shape, generator=gen))
        sd = {k: v.clone() for k, v in live.state_dict().items()}
        live = live.to("cuda", torch.float16)
        live(torch.zeros(1, 64, 8, device="cuda", dtype=torch.float16))  # the old-style hook computes the weight at a call
        folded = loading.fold_weight_norm(sd, dtype=torch.float16, device="cuda")
        assert folded["weight"].device.type == "cpu"
        assert torch.equal(folded["weight"], live.weight.detach().cpu())


def test_loading_is_strict_and_reads_tagged_zips(tmp_path):
    module = torch.nn.Linear(3, 2)
    path = tmp_path / "w.pth"
    torch.save({"weight": module.state_dict(), "config": {"n": 1}}, path)
    data = bytearray(path.read_bytes())
    data[:2] = b"05"  # GPT-SoVITS writes its version tag over the zip magic
    path.write_bytes(bytes(data))
    sd = loading.unwrap(loading.load_state_dict(path), "weight")
    target = torch.nn.Linear(3, 2)
    loading.load_into(target, sd)
    assert torch.equal(target.weight, module.weight)
    with pytest.raises(EngineError):
        loading.load_into(torch.nn.Linear(3, 2), {"weight": module.weight})  # bias missing
    loading.load_into(torch.nn.Linear(3, 2), {"weight": module.weight, "extra": module.bias}, ignore=["bias", "extra"])
    one, two = torch.ones(1), torch.ones(2)
    assert loading.strip_prefix({"model.a": one, "b": two}, "model.", only=True) == {"a": one}


def test_safetensors_round_trip(tmp_path):
    from safetensors.torch import save_file

    save_file({"x": torch.arange(4.0)}, str(tmp_path / "m.safetensors"))
    assert torch.equal(loading.load_state_dict(tmp_path / "m.safetensors")["x"], torch.arange(4.0))


# -- the registry and the snapshot -----------------------------------------------------------


def test_registry_knows_the_dummy_engine():
    assert "dummy" in engine_ids()
    assert get_engine("dummy") is get_engine("dummy")
    with pytest.raises(EngineError) as e:
        get_engine("nonexistent")
    assert e.value.code == "capability_not_supported"


def test_native_engines_offer_their_manifests_options():
    from mizuoto.core.engines.registry import MANIFESTS_DIR, load_manifest

    for engine_id in engine_ids():
        path = MANIFESTS_DIR / f"{engine_id}.toml"
        if not path.is_file():
            continue
        options = set(get_engine(engine_id).OPTION_SCHEMA)
        assert options == set(load_manifest(path).options), f"{engine_id}: native OPTION_SCHEMA {sorted(options)} != the manifest's options"


def test_snapshot_names_what_changed():
    before = snapshot()
    torch.manual_seed(99)
    changed = diff(before, snapshot())
    assert any(c.startswith("rng.") for c in changed)
