"""Loading weights: checkpoint tensors left in their file's mapping while a model loads for a device."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from mizuoto.runtime import loading  # noqa: E402


class Net(torch.nn.Module):
    def __init__(self, dtype: torch.dtype = torch.float32) -> None:
        super().__init__()
        self.a = torch.nn.Linear(64, 32, dtype=dtype)
        self.b = torch.nn.Linear(32, 8, bias=False, dtype=dtype)
        self.register_buffer("scale", torch.ones(8, dtype=dtype))


def _weights(seed: int = 0) -> dict:
    g = torch.Generator().manual_seed(seed)
    return {k: torch.randn(v.shape, generator=g) for k, v in Net().state_dict().items()}


def _shares(param: torch.Tensor, src: torch.Tensor) -> bool:
    return param.data_ptr() == src.data_ptr()


@pytest.fixture(autouse=True)
def adopts(monkeypatch):
    """As on Linux and macOS; Windows never adopts (its own test below)."""
    monkeypatch.setattr(loading, "ADOPTS", True)


@pytest.fixture
def pickle(tmp_path):
    torch.save(_weights(), tmp_path / "w.pt")
    return tmp_path / "w.pt"


def test_loading_for_a_device_keeps_the_checkpoints_tensors(pickle):
    net = Net()
    weight = net.a.weight
    with loading.placing("cuda:0") as placement:
        sd = loading.load_state_dict(pickle)
        loading.load_into(net, sd)
        assert placement.settle([SimpleNamespace(load_device="cuda:0", module=net)]) == 0
    assert net.a.weight is weight and weight.requires_grad and not weight.is_inference()
    assert _shares(net.a.weight, sd["a.weight"]) and _shares(net.b.weight, sd["b.weight"])
    assert not _shares(net.scale, sd["scale"])  # buffers are copied
    for k, v in _weights().items():
        assert torch.equal(net.state_dict()[k], v)


def test_weights_the_cpu_computes_with_are_copied(pickle):
    for scope in (_no_scope(), loading.placing("cpu")):  # training, previews; a model loading for the CPU
        with scope:
            sd = loading.load_state_dict(pickle)
            net = Net()
            loading.load_into(net, sd)
        assert not _shares(net.a.weight, sd["a.weight"]) and torch.equal(net.a.weight, sd["a.weight"])


def _no_scope():
    import contextlib

    return contextlib.nullcontext()


def test_settle_copies_what_no_handle_on_the_device_holds(tmp_path):
    from safetensors.torch import save_file

    save_file({"a": torch.zeros(3), "w": torch.randn(32, 64)}, str(tmp_path / "w.safetensors"))
    placed, kept_on_cpu = torch.nn.Linear(64, 32, bias=False), torch.nn.Linear(64, 32, bias=False)
    with loading.placing("cuda:0") as placement:
        sd = loading.load_state_dict(tmp_path / "w.safetensors")
        loading.load_into(placed, {"weight": sd["w"]})
        sd2 = loading.load_state_dict(tmp_path / "w.safetensors")
        loading.load_into(kept_on_cpu, {"weight": sd2["w"]})
        assert _shares(placed.weight, sd["w"]) and _shares(kept_on_cpu.weight, sd2["w"])
        handles = [SimpleNamespace(load_device="cuda:0", module=placed), SimpleNamespace(load_device="cpu", module=kept_on_cpu)]
        assert placement.settle(handles) == 1
    assert _shares(placed.weight, sd["w"])  # computes on the device's copies
    assert not _shares(kept_on_cpu.weight, sd2["w"]) and kept_on_cpu.weight.data_ptr() % 64 == 0 and torch.equal(kept_on_cpu.weight, sd2["w"])


def test_only_tensors_read_from_a_checkpoint_are_taken_and_each_once(pickle):
    source, target = Net(), Net()
    with loading.placing("cuda:0"):
        loading.load_into(target, source.state_dict())  # a live module's tensors: copied, never shared
        sd = loading.load_state_dict(pickle)
        first, second = Net(), Net()
        loading.load_into(first, sd)
        loading.load_into(second, sd)  # the same checkpoint into a second module: copied, so the two stay apart
    assert not _shares(target.a.weight, source.a.weight)
    assert _shares(first.a.weight, sd["a.weight"]) and not _shares(second.a.weight, sd["a.weight"])
    assert torch.equal(first.a.weight, second.a.weight)


def test_cast_and_tied_weights_are_copied(pickle, tmp_path):
    class Tied(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.x = torch.nn.Linear(4, 4, bias=False)
            self.y = torch.nn.Linear(4, 4, bias=False)
            self.y.weight = self.x.weight

    torch.save({"x.weight": torch.ones(4, 4), "y.weight": torch.full((4, 4), 2.0)}, tmp_path / "t.pt")
    with loading.placing("cuda:0"):
        sd = loading.load_state_dict(pickle)
        half = Net(torch.bfloat16)
        loading.load_into(half, sd)
        tied = Tied()
        st = loading.load_state_dict(tmp_path / "t.pt")
        loading.load_into(tied, st)
    assert half.a.weight.dtype == torch.bfloat16 and torch.equal(half.a.weight, sd["a.weight"].to(torch.bfloat16))
    assert tied.x.weight is tied.y.weight and not _shares(tied.x.weight, st["x.weight"]) and not _shares(tied.x.weight, st["y.weight"])
    assert torch.equal(tied.x.weight, torch.full((4, 4), 2.0))  # the last key wins, as load_state_dict copies


def test_nothing_is_taken_on_windows(pickle, monkeypatch):
    monkeypatch.setattr(loading, "ADOPTS", False)  # what sys.platform == "win32" gives
    with loading.placing("cuda:0"):
        sd = loading.load_state_dict(pickle)
        net = Net()
        loading.load_into(net, sd)
    assert not _shares(net.a.weight, sd["a.weight"]) and torch.equal(net.a.weight, sd["a.weight"])


def test_strictness_is_unchanged_for_taken_keys(tmp_path):
    torch.save({"a.weight": torch.zeros(32, 64)}, tmp_path / "w.pt")
    with loading.placing("cuda:0"), pytest.raises(loading.EngineError) as e:
        loading.load_into(Net(), loading.load_state_dict(tmp_path / "w.pt"))
    assert "a.weight" not in e.value.detail["missing"] and "a.bias" in e.value.detail["missing"]


def test_trim_heap_runs_anywhere():
    loading.trim_heap()
    loading.trim_heap()  # the second call reuses the function looked up by the first


class Built(torch.nn.Module):
    """What engines build before loading: unwritten weights the checkpoint fills, values computed when built that it
    does not hold (a rotary table, a norm's ones), a tied pair and an integer buffer."""

    def __init__(self) -> None:
        super().__init__()
        self.proj = torch.nn.Linear(16, 16)
        self.norm = torch.nn.LayerNorm(16)
        self.embed = torch.nn.Embedding(10, 16)
        self.head = torch.nn.Linear(16, 10, bias=False)
        self.head.weight = self.embed.weight
        self.register_buffer("rope", torch.arange(32, dtype=torch.float32).div(7).sin(), persistent=False)
        self.register_buffer("positions", torch.arange(4))
        self.register_buffer("scale", torch.full((16,), 1 / 3))
        self.rotary = Rotary()


class Rotary(torch.nn.Module):
    """Keeps its table float32 whatever the cast, by its own ``_apply`` (as ``mizuoto.runtime.lm.qwen.Rotary``)."""

    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("inv_freq", torch.arange(8, dtype=torch.float32).div(3).exp(), persistent=False)

    def _apply(self, fn, recurse=True):
        super()._apply(fn, recurse)
        if self.inv_freq.dtype != torch.float32:
            self.inv_freq = torch.arange(8, dtype=torch.float32).div(3).exp()
        return self


def test_cast_before_loading_gives_what_module_to_gives():
    g = torch.Generator().manual_seed(1)
    sd = {
        "proj.weight": torch.randn(16, 16, generator=g),
        "proj.bias": torch.randn(16, generator=g),
        "embed.weight": torch.randn(10, 16, generator=g),
        "scale": torch.randn(16, generator=g),
    }
    expected, got = Built(), Built()
    expected.to(dtype=torch.bfloat16)
    loading.load_into(expected, sd, ignore=["norm.*", "head.weight", "positions"])
    weight = got.proj.weight
    assert loading.cast(got, torch.bfloat16, sd) is got
    loading.load_into(got, sd, ignore=["norm.*", "head.weight", "positions"])
    assert got.proj.weight is weight and weight.requires_grad and got.head.weight is got.embed.weight
    want, have = expected.state_dict(keep_vars=True), got.state_dict(keep_vars=True)
    assert want.keys() == have.keys()
    for k in want:
        assert want[k].dtype == have[k].dtype and torch.equal(want[k], have[k]), k
    assert got.rope.dtype == torch.bfloat16 and torch.equal(got.rope, expected.rope)  # computed when built: rounded as before
    assert got.positions.dtype == torch.int64
    assert got.rotary.inv_freq.dtype == torch.float32 and torch.equal(got.rotary.inv_freq, expected.rotary.inv_freq)  # its _apply decides


def test_cast_leaves_what_the_checkpoint_fills_unwritten(monkeypatch):
    calls: list[tuple] = []
    real = torch.Tensor.to

    def spy(self, *args, **kwargs):
        calls.append(tuple(self.shape))
        return real(self, *args, **kwargs)

    built = Built()
    monkeypatch.setattr(torch.Tensor, "to", spy)
    loading.cast(built, torch.bfloat16, {"proj.weight": torch.zeros(16, 16), "embed.weight": torch.zeros(10, 16), "proj.bias": torch.zeros(3)})
    monkeypatch.undo()
    assert (16, 16) not in calls and (10, 16) not in calls  # filled by the checkpoint: new memory, nothing copied
    assert (16,) in calls  # proj.bias's shape differs from the checkpoint's: cast as Module.to casts it
    assert built.proj.weight.dtype == built.embed.weight.dtype == torch.bfloat16
