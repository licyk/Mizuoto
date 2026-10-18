"""Device memory: units, handles placed whole or partly, streaming, the manager's budget and
eviction, the policy. On a simulated device (the CPU playing a GPU of a given size), so CI covers the logic; streamed
weights rest as meta tensors there, so a unit read outside its ``forward`` fails."""

from __future__ import annotations

from typing import Any

import pytest

torch = pytest.importorskip("torch")

from torch import nn  # noqa: E402

from mizuoto.runtime import ops  # noqa: E402
from mizuoto.runtime.device import DeviceService  # noqa: E402
from mizuoto.runtime.handle import ModelHandle  # noqa: E402
from mizuoto.runtime.manager import ModelManager  # noqa: E402
from mizuoto.runtime.policy import VramPolicy  # noqa: E402
from mizuoto.runtime.stream import Streamer  # noqa: E402
from mizuoto.runtime.units import split  # noqa: E402

MB = 2**20
WIDTH = 512  # a 512 × 512 float32 Linear is 1 MB (+ its bias)


class Blocks(nn.Module):
    def __init__(self, n: int, width: int = WIDTH, seed: int = 0) -> None:
        super().__init__()
        self.inp = ops.default.Linear(width, width)
        self.blocks: Any = nn.ModuleList(nn.Sequential(ops.default.Linear(width, width), nn.GELU()) for _ in range(n))
        self.norm = ops.default.LayerNorm(width)
        self.register_buffer("scale", torch.ones(width))
        ops.init_random(self, torch.Generator().manual_seed(seed))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.inp(x)
        for b in self.blocks:
            x = b(x)
        return self.norm(x) * self.scale


def unit_bytes() -> int:
    return (WIDTH * WIDTH + WIDTH) * 4


def manager(total_mb: float, mode: str = "normal", reserve_mb: float = 0) -> ModelManager:
    devices = DeviceService(simulate={"cpu": int(total_mb * MB)}, strict=True)
    policy = VramPolicy(mode=mode, reserve_bytes=int(reserve_mb * MB), host_copies="on")
    return ModelManager(devices, policy=lambda: policy)


# -- units ------------------------------------------------------------------------------------


def test_split_makes_list_children_units_and_keeps_the_rest_fixed():
    model = Blocks(4)
    parts = split(model)
    assert [u.name for u in parts.units] == ["blocks.0", "blocks.1", "blocks.2", "blocks.3"]
    assert all(u.bytes == unit_bytes() for u in parts.units)
    fixed = {f"{type(m).__name__}.{n}" for m, n in parts.fixed_params}
    assert fixed == {"Linear.weight", "Linear.bias", "LayerNorm.weight", "LayerNorm.bias"}  # inp and norm
    assert [n for _, n in parts.buffers] == ["scale"]


def test_split_respects_marks_tied_weights_and_memory_units():
    model = Blocks(3)
    model.blocks[2][0].weight = model.inp.weight  # tied with a module outside the unit: stays fixed
    parts = split(model, {id(model.blocks[0]): "fixed", id(model.blocks[1]): "hot"})
    assert [u.name for u in parts.units] == ["blocks.1"]  # blocks.2 is left with its bias, too small for a unit
    assert parts.units[0].priority == "hot"
    assert any(m is model.inp and n == "weight" for m, n in parts.fixed_params)

    class Named(Blocks):
        def memory_units(self) -> list[nn.Module]:
            return [self.blocks[0][0], self.norm]

    named = split(Named(2))
    assert [u.name for u in named.units] == ["blocks.0.0"]  # the norm is under 64 KB


# -- handles ----------------------------------------------------------------------------------


def run(model: Any, x: torch.Tensor) -> torch.Tensor:
    with torch.inference_mode():
        return model(x)


@pytest.mark.parametrize("keep", [0, 2, 4])
def test_partly_loaded_model_computes_exactly_what_the_resident_one_does(keep):
    model = Blocks(4)
    x = torch.randn(3, WIDTH, generator=torch.Generator().manual_seed(1))
    expected = run(model, x)
    h = ModelHandle(model, name="m", load_device="cpu", kind="sequence")
    h.simulated = h.strict = True
    h.keep_host = True
    streamer = Streamer("cpu")
    budget = h.fixed_bytes + keep * unit_bytes() + h.stream_reserve(h.units, streamer.copies)
    h.load(budget, streamer)
    assert sum(u.resident for u in h.units) == keep
    assert h.loaded_bytes == h.fixed_bytes + keep * unit_bytes()
    streamed = [u for u in h.units if not u.resident]
    # at rest a streamed unit's weights are not there
    assert all(m._parameters[n].device.type == "meta" for u in streamed for m, n in u.slots)
    assert torch.equal(run(model, x), expected)
    assert all(m._parameters[n].device.type == "meta" for u in streamed for m, n in u.slots)
    assert streamer.copied_bytes == (4 - keep) * unit_bytes()
    # sequence handles keep a prefix of their units, in order
    assert [u.resident for u in h.units] == [i < keep for i in range(4)]
    h.offload()
    assert not h.placed and h.loaded_bytes == 0
    with pytest.raises(RuntimeError):  # parked (strict): off the device nothing is there
        run(model, x)
    h.unpark()
    assert torch.equal(run(model, x), expected)


def test_a_unit_read_outside_its_forward_fails_in_strict_mode():
    model = Blocks(2)
    h = ModelHandle(model, name="m", load_device="cpu")
    h.simulated = h.strict = h.keep_host = True
    h.load(h.minimum_bytes(1), Streamer("cpu"))
    with pytest.raises((RuntimeError, NotImplementedError)):
        model.blocks[0][0].weight.sum().item()


def test_oneshot_handles_keep_their_largest_units_and_hot_units_first():
    class Mixed(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.inp = ops.default.Linear(8, 8)  # the first parameter stays fixed
            self.layers = nn.ModuleList([ops.default.Linear(512, 512), ops.default.Linear(1024, 1024), ops.default.Linear(512, 768)])

    model = Mixed()
    h = ModelHandle(model, name="m", load_device="cpu")
    one, four, one_and_half = (u.bytes for u in h.units)
    assert h.fill(h.fixed_bytes + four + one_and_half + one, copies=1) == {0, 1, 2}
    # less room: the largest stays, the others stream (the reserve holds a copy of the larger)
    assert h.fill(h.fixed_bytes + four + one_and_half, copies=1) == {1}
    h.mark(model.layers[0], "hot")
    assert h.fill(h.fixed_bytes + one + four, copies=1) == {0}


def test_partial_unload_streams_units_in_reverse_fill_order_then_offloads():
    model = Blocks(4)
    h = ModelHandle(model, name="m", load_device="cpu", kind="sequence")
    h.simulated = h.strict = h.keep_host = True
    streamer = Streamer("cpu")
    h.load(None, streamer)
    assert h.loaded and h.loaded_bytes == h.size_bytes
    h._streamer = streamer
    freed = h.partially_unload(unit_bytes() + 1)
    assert freed == 2 * unit_bytes()
    assert [u.resident for u in h.units] == [True, True, False, False]
    x = torch.randn(2, WIDTH)
    assert torch.equal(run(model, x), run(Blocks(4), x))
    freed = h.partially_unload(10 * h.size_bytes)
    assert not h.placed and freed == h.fixed_bytes + 2 * unit_bytes()


def test_free_drops_without_copying_back():
    model = Blocks(2)
    h = ModelHandle(model, name="m", load_device="cpu")
    h.simulated = True
    h.load()
    h.free()
    assert h.module is None and h.device == "none" and h.loaded_bytes == 0


def test_replace_resets_the_handle():
    h = ModelHandle(Blocks(2), name="m", load_device="cpu")
    h.simulated = True
    h.load()
    bigger = Blocks(4)
    h.replace(bigger)
    assert h.module is bigger and not h.placed and len(h.units) == 4
    h.load()
    assert h.loaded


# -- the manager ------------------------------------------------------------------------------


def test_manager_loads_whole_when_it_fits_and_evicts_the_least_recently_used():
    m = manager(total_mb=12, mode="high")
    a, b, c = (ModelHandle(Blocks(4, seed=i), name=n, load_device="cpu") for i, n in enumerate("abc"))
    for h in (a, b, c):
        m.register(h.name.upper(), [h])
    m.load([a])
    m.load([b])
    assert a.loaded and b.loaded
    m.load([c])  # a (least recently used) goes, whole
    assert not a.placed and b.loaded and c.loaded
    with m.use([b]):
        m.load([a], memory_required=MB)  # b is in use, so c goes
        assert b.loaded and a.loaded and not c.placed
    assert {r.name for r in m.resident() if r.loaded} == {"a", "b"}


def test_manager_shrinks_other_models_unit_by_unit_before_streaming():
    m = manager(total_mb=9)
    a, b = ModelHandle(Blocks(4), name="a", load_device="cpu"), ModelHandle(Blocks(4, seed=1), name="b", load_device="cpu")
    m.register("A", [a])
    m.register("B", [b])
    m.load([a])
    assert a.loaded
    m.load([b])  # b needs ~5 MB of the ~4 left: a gives up only what is missing (two units)
    assert b.loaded and a.partial
    assert 0 < a.loaded_bytes < a.size_bytes
    x = torch.randn(2, WIDTH)
    assert torch.equal(run(a.module, x), run(Blocks(4), x))


def test_manager_loads_partly_when_the_model_does_not_fit():
    m = manager(total_mb=4.5)
    h = ModelHandle(Blocks(6), name="big", load_device="cpu", kind="sequence")
    m.register("BIG", [h])
    x = torch.randn(2, WIDTH)
    expected = run(Blocks(6), x)
    with m.use([h]):
        assert h.partial
        assert torch.equal(run(h.module, x), expected)
    assert {r.name: r.streamed_bytes > 0 for r in m.resident()} == {"big": True}


def test_below_the_minimum_the_manager_tries_anyway_and_high_mode_never_streams():
    """No estimate refuses a load (ComfyUI's way): below its smallest working set a model is placed at that minimum,
    and only an allocation that really fails is an error (a simulated device never runs out)."""
    m = manager(total_mb=1.5)
    h = ModelHandle(Blocks(4), name="m", load_device="cpu")
    m.register("M", [h])
    m.load([h])
    assert h.partial and 0 < h.loaded_bytes < h.size_bytes
    high = manager(total_mb=4, mode="high")
    h2 = ModelHandle(Blocks(4), name="m", load_device="cpu")
    high.register("M", [h2])
    high.load([h2])  # the old behaviour: whole (a real GPU would run out of memory here)
    assert h2.loaded


def test_mode_none_streams_everything_but_the_fixed_parts():
    m = manager(total_mb=100, mode="none")
    h = ModelHandle(Blocks(4), name="m", load_device="cpu")
    m.register("M", [h])
    m.load([h])
    assert h.placed and not any(u.resident for u in h.units)


def test_engine_override_and_working_memory_shape_the_budget():
    m = manager(total_mb=8)
    h = ModelHandle(Blocks(4), name="m", load_device="cpu")
    m.register("M", [h], mode="high")
    m.load([h], memory_required=6 * MB)
    assert h.loaded  # high: whole regardless
    m2 = manager(total_mb=8)
    m2.devices.simulate_working = True  # count working memory against the simulated size too
    h2 = ModelHandle(Blocks(4), name="m", load_device="cpu")
    m2.register("M", [h2])
    m2.load([h2], memory_required=4 * MB)
    assert h2.partial and h2.loaded_bytes + 4 * MB <= 8 * MB


def test_a_partial_load_says_why():
    """A component loaded partly records why, with the numbers the manager weighed; outputs report it (``memory.partial``)."""
    m = manager(total_mb=8)
    m.devices.simulate_working = True
    h = ModelHandle(Blocks(4), name="tts/m:talker", load_device="cpu")
    m.register("M", [h])
    m.load([h], memory_required=4 * MB)
    assert h.partial
    [why] = m.owner_placements("M")
    assert why["component"] == "talker" and "exceed the free memory" in why["reason"]
    assert (why["free_mb"], why["working_mb"], why["reserve_mb"]) == (8, 4, 0)
    assert (why["loaded_mb"], why["size_mb"]) == (round(h.loaded_bytes / MB), round(h.size_bytes / MB))
    m.load([h])  # room enough now: whole, nothing to explain
    assert h.loaded and m.owner_placements("M") == []


def test_a_component_that_gave_up_memory_says_for_which():
    m = manager(total_mb=9)
    a, b = ModelHandle(Blocks(4), name="tts/m:talker", load_device="cpu"), ModelHandle(Blocks(4, seed=1), name="tts/m:decoder", load_device="cpu")
    m.register("M", [a, b])
    m.load([a])
    m.load([b])
    assert a.partial and b.loaded
    taker = {"gave_up_for": "decoder", "loaded_mb": round(a.loaded_bytes / MB), "size_mb": round(a.size_bytes / MB)}
    assert m.owner_placements("M") == [{"component": "talker", "reason": "gave up memory for decoder", **taker}]


def test_a_partly_loaded_component_that_gives_up_more_keeps_its_first_reason():
    m = manager(total_mb=8)
    m.devices.simulate_working = True
    a, b = ModelHandle(Blocks(4), name="tts/m:talker", load_device="cpu"), ModelHandle(Blocks(2, seed=1), name="tts/m:decoder", load_device="cpu")
    m.register("M", [a, b])
    m.load([a], memory_required=4 * MB)
    assert a.partial
    before = a.loaded_bytes
    m.load([b], memory_required=3 * MB)
    assert a.loaded_bytes < before
    [why] = m.owner_placements("M")
    assert "exceed the free memory" in why["reason"] and why["gave_up_for"] == "decoder" and why["loaded_mb"] == round(a.loaded_bytes / MB)


def test_a_cache_peaks_one_layer_above_its_size_not_at_twice_it():
    from mizuoto.runtime.lm.cache import peak_bytes

    assert peak_bytes(10, layers=4, positions=100) == 10 * 100 * 5
    assert peak_bytes(10, layers=4, positions=100, rows=3) == 3 * 10 * 100 * 5
    assert peak_bytes(10, layers=4, positions=100, kind="static") == 10 * 256 * 4 + 10 * 128  # buffers of 256 positions
    assert peak_bytes(10, layers=4, positions=300, kind="static") == 10 * 512 * 4 + 10 * 256


def test_unregister_frees_and_release_idle_offloads():
    m = manager(total_mb=100)
    a, b = ModelHandle(Blocks(2), name="a", load_device="cpu"), ModelHandle(Blocks(2), name="b", load_device="cpu")
    m.register("A", [a])
    m.register("B", [b])
    m.load([a, b])
    with m.use([a]):
        assert m.release_idle() == ["b"]
    assert a.loaded and not b.placed
    m.unregister("A")
    assert a.module is None and [r.name for r in m.resident()] == ["b"]


def test_real_cpu_devices_move_nothing():
    m = ModelManager(DeviceService())
    model = Blocks(2)
    weight = model.blocks[0][0].weight
    h = ModelHandle(model, name="m", load_device="cpu")
    m.register("M", [h])
    m.load([h])
    assert h.loaded and model.blocks[0][0].weight is weight and weight.data_ptr() == model.blocks[0][0].weight.data_ptr()


def test_policy():
    p = VramPolicy()
    assert p.mode_for("cuda") == "normal" and p.mode_for("mps") == "shared" and p.mode_for("cuda", "low") == "low"
    assert p.streams_for("cuda") == 2 and p.streams_for("xpu") == 0
    assert p.ram_reserve(64 * 2**30) == int(6.4 * 2**30) and p.ram_reserve(8 * 2**30) == 2 * 2**30
    # int(): pinned_budget truncates, and Windows' budget is exactly 40 %.
    assert int(0.4 * 64 * 2**30) <= p.pinned_budget(64 * 2**30) <= 0.9 * 64 * 2**30


def test_the_streamer_learns_the_order_and_prefetches():
    """With side streams (here simulated: synchronous copies) each unit is copied once per pass, ahead of its turn,
    and the result is the resident one's."""
    model = Blocks(6)
    x = torch.randn(2, WIDTH, generator=torch.Generator().manual_seed(3))
    expected = run(model, x)
    h = ModelHandle(model, name="m", load_device="cpu", kind="oneshot")
    h.simulated = h.strict = h.keep_host = True
    streamer = Streamer("cpu", streams=2)
    h.load(h.minimum_bytes(streamer.copies), streamer)
    assert not any(u.resident for u in h.units)
    copied, hits = [], []
    for _ in range(4):  # passes of one request, as an autoregressive loop makes them
        before, hit = streamer.copied_bytes, streamer.hits
        assert torch.equal(run(model, x), expected)
        copied.append((streamer.copied_bytes - before) // unit_bytes())
        hits.append(streamer.hits - hit)
    # once the order is learnt each unit is copied once a pass, ahead of its turn (the first ones during the pass before)
    assert copied[0] == 6 and copied[2:] == [6, 6]
    assert hits[0] == 0 and hits[2:] == [6, 6]
    assert streamer._next == {(id(h), i): (id(h), i + 1) for i in range(5)} | {(id(h), 5): (id(h), 0)}
    h.settle()
    assert streamer.cached_bytes == 0


def test_a_sequence_handle_streams_one_unit_at_a_time_without_prefetching():
    """An autoregressive model's layers are not prefetched (a token's work hides none of a copy) and hold one copy, so
    its placement sets aside one unit's room instead of ``streams + 1``; the result is the resident one's."""
    model = Blocks(6)
    x = torch.randn(2, WIDTH, generator=torch.Generator().manual_seed(3))
    expected = run(model, x)
    h = ModelHandle(model, name="m", load_device="cpu", kind="sequence")
    h.simulated = h.strict = h.keep_host = True
    streamer = Streamer("cpu", streams=2)
    assert streamer.copies_for(h) == 1 and h.minimum_bytes(streamer.copies_for(h)) == h.fixed_bytes + unit_bytes()
    h.load(h.minimum_bytes(1), streamer)
    assert not any(u.resident for u in h.units)
    for _ in range(3):
        assert torch.equal(run(model, x), expected)
        assert streamer.peak_bytes <= unit_bytes()  # never more than the running unit's copy
    assert streamer.hits == 0
    h.settle()
    assert streamer.cached_bytes == 0


# -- running out of memory --------------------------------------------------------------------


def test_a_phase_that_runs_out_of_memory_runs_again_with_the_same_draws(monkeypatch):
    """The request's generators are put back, so the retry gives exactly what an undisturbed
    run gives; it runs with more working memory set aside (here: more of the model streams)."""
    from mizuoto.runtime.context import RequestContext

    def phase(ctx, model, x):
        noise = torch.randn(x.shape, generator=ctx.generator_for("cpu"))
        extra = ctx.rng.random()
        return run(model, x + noise * extra)

    x = torch.randn(2, WIDTH, generator=torch.Generator().manual_seed(4))
    model = Blocks(6)
    expected = phase(RequestContext("a", seed=11), model, x)

    m = manager(total_mb=200)
    m.devices.simulate_working = True
    h = ModelHandle(model, name="m", load_device="cpu", kind="sequence")
    m.register("M", [h])
    fails = {"left": 1}
    block = model.blocks[3]
    original = block.forward

    def flaky(*args, **kwargs):
        if fails["left"]:
            fails["left"] -= 1
            raise torch.OutOfMemoryError("CUDA out of memory. Tried to allocate 2.00 GiB")
        return original(*args, **kwargs)

    monkeypatch.setattr(block, "forward", flaky)
    ctx = RequestContext("b", seed=11)
    ctx.manager = m
    got = ctx.run("blocks", [h], lambda: phase(ctx, model, x), memory=150 * MB)  # fits whole: the cap makes it partial
    assert torch.equal(got, expected)
    assert ctx.applied["oom_retries"] == {"blocks": 1} and ctx.warnings
    assert h.partial and h.cap_bytes is not None  # after running out, half its weights stream until it is offloaded

    fails["left"] = 5  # more failures than retries: the error comes through
    ctx = RequestContext("c", seed=11)
    ctx.manager = m
    with pytest.raises(torch.OutOfMemoryError):
        ctx.run("blocks", [h], lambda: phase(ctx, model, x))
    m.policy  # noqa: B018 - the policy is read live
    off = ModelManager(m.devices, policy=lambda: VramPolicy(oom_retry=False, reserve_bytes=0))
    off.register("M2", [ModelHandle(Blocks(1), name="n", load_device="cpu")])
    fails["left"] = 1
    ctx = RequestContext("d", seed=11)
    ctx.manager = off
    with pytest.raises(torch.OutOfMemoryError):
        ctx.run("blocks", [], lambda: run(model, x))


def test_a_fallback_runs_only_when_allowed_and_is_reported(monkeypatch):
    from mizuoto.runtime.context import RequestContext

    def failing():
        raise torch.OutOfMemoryError("CUDA out of memory")

    for allowed in (False, True):
        policy = VramPolicy(reserve_bytes=0, oom_fallback=allowed)
        m = ModelManager(DeviceService(simulate={"cpu": 100 * MB}, strict=True), policy=lambda policy=policy: policy)
        ctx = RequestContext("f", seed=1)
        ctx.manager = m
        if not allowed:
            with pytest.raises(torch.OutOfMemoryError):
                ctx.run("decoder", [], failing, fallback=("in tiles", lambda: "tiled"))
            continue
        assert ctx.run("decoder", [], failing, fallback=("in tiles", lambda: "tiled")) == "tiled"
        assert ctx.applied["fallbacks"] == {"decoder": "in tiles"} and ctx.applied["oom_retries"] == {"decoder": 2}
        assert any("differs slightly" in w for w in ctx.warnings)


def test_measured_working_memory_corrects_the_estimate(tmp_path):
    """A phase that used more than its estimate gets more set aside next time, remembered in a
    file; one that used less keeps its estimate."""

    class Measuring(DeviceService):
        def __init__(self) -> None:
            super().__init__(simulate={"cpu": 1 << 40}, strict=True)
            self.simulate_working = True
            self.used = 0

        def allocated(self, name: str) -> int | None:
            return 1000

        def start_peak(self, name: str) -> None:
            pass

        def peak(self, name: str) -> int | None:
            return 1000 + self.used

    devices = Measuring()
    m = ModelManager(devices, policy=lambda: VramPolicy(reserve_bytes=0, host_copies="on"))
    m.load_calibration(tmp_path / "calibration.json")
    h = ModelHandle(Blocks(2), name="tts/m:decoder", load_device="cpu")
    m.register("M", [h])
    asked: list[int] = []
    original = m.load
    m.load = lambda handles, memory_required=0: (asked.append(memory_required), original(handles, memory_required))[1]  # ty: ignore[invalid-assignment]
    devices.used = 300 * MB
    with m.use([h], 100 * MB):
        pass
    assert m.calibration == {"tts:decoder": pytest.approx(3.3)}
    with m.use([h], 100 * MB):
        pass
    assert asked == [100 * MB, int(100 * MB * 3.3)]
    devices.used = 10 * MB  # less than estimated: the correction stays
    with m.use([h], 100 * MB):
        pass
    assert m.calibration["tts:decoder"] == pytest.approx(3.3)
    again = ModelManager(devices)
    again.load_calibration(tmp_path / "calibration.json")
    assert again.calibration == m.calibration


def test_working_memory_leaves_out_streamed_copies_and_what_a_phase_keeps():
    """The calibration measures working memory only: the copies of streamed weights (counted with the weights) and what
    a phase leaves allocated (CUDA graphs' pools; there next time) do not make later requests set aside more."""

    class Measuring(DeviceService):
        def __init__(self) -> None:
            super().__init__(simulate={"cpu": 1 << 40}, strict=True)
            self.used = self.kept = 0

        def allocated(self, name: str) -> int | None:
            return 1000 + self.kept

        def start_peak(self, name: str) -> None:
            pass

        def peak(self, name: str) -> int | None:
            return 1000 + self.used

    devices = Measuring()
    m = ModelManager(devices, policy=lambda: VramPolicy(reserve_bytes=0, host_copies="on"))
    h = ModelHandle(Blocks(2), name="tts/m:talker", load_device="cpu")
    m.register("M", [h])
    streamer = m._streamer("cpu", m.policy)
    with m.use([h], 100 * MB):
        devices.used = 300 * MB
        streamer.peak_bytes = 250 * MB  # the streamed units' copies at once
    assert m.calibration == {}
    devices.kept = 0
    with m.use([h], 100 * MB):
        devices.used = devices.kept = 280 * MB  # kept after the phase
    assert m.calibration == {}
    devices.kept = 0
    with m.use([h], 100 * MB):
        devices.used = 300 * MB  # working memory itself: more than estimated
    assert m.calibration == {"tts:talker": pytest.approx(3.3)}


def test_a_calibration_saved_by_an_earlier_version_is_dropped(tmp_path):
    path = tmp_path / "calibration.json"
    path.write_text('{"qwen3tts:talker": 4.0}', encoding="utf-8")  # version 1: measured with streamed copies
    m = ModelManager(DeviceService())
    m.load_calibration(path)
    assert m.calibration == {}


# -- training ---------------------------------------------------------------------------------


def test_checkpointing_gives_the_same_gradients_with_private_generators():
    """Blocks that draw (dropout from a private generator, as LoRA's) run again in the
    backward with the same draws: the gradients equal those without checkpointing."""
    from mizuoto.training.memory import checkpoint_every, checkpoint_units

    class Drawing(torch.nn.Module):
        def __init__(self, gen: torch.Generator) -> None:
            super().__init__()
            self.lin = torch.nn.Linear(16, 16)
            self.gen = gen

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            mask = (torch.rand(x.shape, generator=self.gen) > 0.3).to(x.dtype)
            return torch.tanh(self.lin(x * mask)) + torch.nn.functional.dropout(x, 0.2)  # the global generator too

    def grads(every: int) -> tuple[list[torch.Tensor], torch.Tensor]:
        torch.manual_seed(0)
        gen = torch.Generator().manual_seed(1)
        blocks = torch.nn.ModuleList(Drawing(gen) for _ in range(5))
        if every:
            assert checkpoint_units(blocks, [gen], every) == len(range(0, 5, every))
        x = torch.randn(4, 16, generator=torch.Generator().manual_seed(2), requires_grad=True)
        h = x
        for b in blocks:
            h = b(h)
        h.square().sum().backward()
        return [p.grad.clone() for p in blocks.parameters()] + [x.grad.clone()], torch.rand(3, generator=gen)

    plain, after = grads(0)
    for every in (1, 2):
        got, got_after = grads(every)
        assert all(torch.equal(a, b) for a, b in zip(plain, got, strict=True))
        assert torch.equal(after, got_after)  # the generator ends where it would have
    assert checkpoint_every("off") == 0 and checkpoint_every("blocks") == 1 and checkpoint_every("every:3") == 3
    with pytest.raises(ValueError):
        checkpoint_every("sometimes")


def test_training_estimates():
    from mizuoto.training.memory import estimate

    full = estimate(1_000_000_000, 1_000_000_000, weight_bytes=4, optimizer_bytes=8, activation_bytes_per_token=10_000, tokens=8_000)
    assert full.as_mb()["total"] == pytest.approx((4e9 + 4e9 + 8e9 + 8e7) / MB, abs=0.1)
    checkpointed = estimate(1_000_000_000, 1_000_000_000, activation_bytes_per_token=10_000, tokens=8_000, checkpointing=1, blocks=20)
    assert checkpointed.activations < full.activations / 4
    lora = estimate(1_000_000_000, 10_000_000, weight_bytes=2, master_bytes=4)
    assert lora.optimizer == 10_000_000 * 12 and lora.gradients == 10_000_000 * 2


def test_frozen_weights_in_the_autocast_dtype_give_the_same_products():
    """Under bf16 autocast a linear layer's float32 weight is cast to bf16 at every product: storing it so is the same."""
    from mizuoto.training.memory import frozen_in

    gen = torch.Generator().manual_seed(0)
    layer = torch.nn.Linear(32, 16)
    with torch.no_grad():
        layer.weight.copy_(torch.randn(16, 32, generator=gen))
        layer.bias.copy_(torch.randn(16, generator=gen))
    layer.requires_grad_(False)
    x = torch.randn(4, 32, generator=gen, requires_grad=True)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        before = layer(x)
    saved = frozen_in(layer, torch.bfloat16)
    assert saved == (16 * 32 + 16) * 2 and layer.weight.dtype == torch.bfloat16
    with torch.autocast("cpu", dtype=torch.bfloat16):
        after = layer(x)
    assert torch.equal(before, after)


def test_a_streamed_frozen_linear_gives_the_resident_gradients():
    from mizuoto.training.memory import streamed_linear

    gen = torch.Generator().manual_seed(0)
    w, b = torch.randn(16, 32, generator=gen), torch.randn(16, generator=gen)
    x1 = torch.randn(2, 5, 32, generator=gen, requires_grad=True)
    x2 = x1.detach().clone().requires_grad_(True)
    y1 = torch.nn.functional.linear(x1, w, b)
    y2 = streamed_linear(x2, w.clone(), b.clone(), torch.device("cpu"))
    assert torch.equal(y1, y2)
    g = torch.randn(y1.shape, generator=gen)
    y1.backward(g)
    y2.backward(g)
    assert torch.equal(x1.grad, x2.grad)


def test_the_host_optimizer_steps_as_the_plain_one_on_the_cpu():
    from mizuoto.training.memory import host_optimizer

    def run(host: bool) -> list[torch.Tensor]:
        model = torch.nn.Linear(8, 4)
        torch.nn.init.constant_(model.weight, 0.1)
        torch.nn.init.zeros_(model.bias)
        make = lambda ps: torch.optim.AdamW(ps, lr=1e-2, weight_decay=0.01)  # noqa: E731
        opt = host_optimizer(list(model.parameters()), make) if host else make(list(model.parameters()))
        sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: 1.0 / (1 + s))
        gen = torch.Generator().manual_seed(5)
        for _ in range(3):
            loss = model(torch.randn(3, 8, generator=gen)).square().sum()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            opt.zero_grad()
        return [p.detach().clone() for p in model.parameters()]

    assert all(torch.equal(a, b) for a, b in zip(run(False), run(True), strict=True))


def test_a_request_that_does_not_fit_waits_for_another_to_let_go():
    """Two GPU jobs at once (compute.gpu_jobs): the second, short of its minimum while the first holds the device,
    waits instead of failing, and runs once the first lets go."""
    import threading

    m = manager(total_mb=7)
    a, b = ModelHandle(Blocks(4), name="a", load_device="cpu"), ModelHandle(Blocks(4, seed=1), name="b", load_device="cpu")
    m.register("A", [a])
    m.register("B", [b])
    m._policy = lambda: VramPolicy(mode="high", reserve_bytes=0, host_copies="on")  # whole only: b cannot load beside a
    done = threading.Event()
    order: list[str] = []

    def second() -> None:
        with m.use([b]):
            order.append("b")
        done.set()

    with m.use([a]):
        order.append("a")
        t = threading.Thread(target=second)
        t.start()
        assert not done.wait(0.5)  # b waits while a is in use
    assert done.wait(5) and order == ["a", "b"]
    t.join()


def test_high_mode_always_tries_and_only_a_real_allocation_failure_fails(monkeypatch):
    """ComfyUI's way: whole or not at all is tried whatever the estimate says; what fails is the allocation itself."""
    m = manager(total_mb=2, mode="high")
    h = ModelHandle(Blocks(4), name="m", load_device="cpu")  # ~6 MB on a 2 MB device
    m.register("M", [h])
    calls = []

    def load(*args, **kwargs):
        calls.append(1)
        raise torch.OutOfMemoryError("CUDA out of memory. Tried to allocate 4.00 MiB")

    monkeypatch.setattr(h, "load", load)
    with pytest.raises(torch.OutOfMemoryError):
        m.load([h])
    assert calls == [1]  # tried, not refused on the estimate


def test_every_mode_tries_and_only_a_real_allocation_failure_fails(monkeypatch):
    for mode in ("normal", "low", "none"):
        m = manager(total_mb=1, mode=mode)
        h = ModelHandle(Blocks(4), name="m", load_device="cpu")
        m.register("M", [h])

        def load(*args, **kwargs):
            raise torch.OutOfMemoryError("CUDA out of memory. Tried to allocate 1.00 MiB")

        monkeypatch.setattr(h, "load", load)
        with pytest.raises(torch.OutOfMemoryError):
            m.load([h])


def test_a_whole_load_that_waited_long_enough_is_tried(monkeypatch):
    """With two GPU jobs at once a whole-or-nothing request first waits for the other to let go; past the wait it
    tries anyway (a real failure then says what went wrong)."""
    import threading

    from mizuoto.runtime import manager as manager_module

    monkeypatch.setattr(manager_module, "WAIT_SECONDS", 0.3)
    m = manager(total_mb=7)
    a, b = ModelHandle(Blocks(4), name="a", load_device="cpu"), ModelHandle(Blocks(4, seed=1), name="b", load_device="cpu")
    m.register("A", [a])
    m.register("B", [b])
    m._policy = lambda: VramPolicy(mode="high", reserve_bytes=0, host_copies="on")
    done = threading.Event()

    def second() -> None:
        with m.use([b]):
            done.set()

    with m.use([a]):
        t = threading.Thread(target=second)
        t.start()
        assert done.wait(5)  # tried while a still holds the device
        assert b.loaded
    t.join()


def test_low_mode_keeps_weights_under_half_the_device():
    m = manager(total_mb=8, mode="low")
    h = ModelHandle(Blocks(6), name="m", load_device="cpu")  # ~7 MB: it would fit, but not in half
    m.register("M", [h])
    m.load([h])
    assert h.partial and h.loaded_bytes <= 4 * MB
    small = ModelHandle(Blocks(1), name="s", load_device="cpu")
    m.register("S", [small])
    m.load([small])
    assert small.loaded  # 2 MB fits in half
