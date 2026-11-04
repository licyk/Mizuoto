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
