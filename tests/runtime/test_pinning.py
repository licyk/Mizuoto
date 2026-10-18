"""Host copies page-locked in place (``mizuoto.runtime.pinning``): registered within the budget, released with their
storage, and loads and reloads from them give the same weights."""

from __future__ import annotations

import gc

import pytest

torch = pytest.importorskip("torch")

from mizuoto.runtime import pinning  # noqa: E402
from mizuoto.runtime.handle import ModelHandle  # noqa: E402
from mizuoto.runtime.stream import PinBudget, Streamer  # noqa: E402


class Stack(torch.nn.Module):
    """Four 1 MB layers: four units."""

    def __init__(self) -> None:
        super().__init__()
        self.layers = torch.nn.ModuleList(torch.nn.Linear(512, 512) for _ in range(4))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x


nvidia = pytest.mark.skipif(not torch.cuda.is_available() or torch.version.hip is not None, reason="needs an NVIDIA GPU")


def test_only_nvidia_gpus_register_host_memory():
    budget = PinBudget(1 << 30)
    assert not pinning.pin(torch.zeros(1 << 20), budget, "cpu")
    assert not pinning.pin(torch.zeros(1 << 20), budget, "mps")
    assert budget.used == 0


@nvidia
def test_a_storage_is_registered_once_within_the_budget_and_released_when_it_dies():
    before = pinning.pinned_bytes()
    budget = PinBudget(3 << 20)
    big = torch.randn(1 << 18)  # 1 MB
    assert pinning.pin(big, budget, "cuda") and big.is_pinned()
    assert pinning.pin(big[10:], budget, "cuda")  # a view: the storage is registered already, nothing more taken
    assert budget.used == 1 << 20 and pinning.pinned_bytes() == before + (1 << 20)

    small = torch.randn(1000)  # shares its pages with other allocations: left pageable
    assert not pinning.pin(small, budget, "cuda") and not small.is_pinned()
    over = torch.randn(1 << 20)  # 4 MB: beyond what is left of the budget
    assert not pinning.pin(over, budget, "cuda") and not over.is_pinned() and budget.used == 1 << 20

    view = big[5:]
    del big
    gc.collect()
    assert budget.used == 1 << 20  # the view keeps the storage, and its pages stay locked
    del view
    gc.collect()
    assert budget.used == 0 and pinning.pinned_bytes() == before


@nvidia
def test_kept_host_copies_are_pinned_as_they_load_and_reloads_give_the_same_weights():
    model = torch.nn.Sequential(*(torch.nn.Linear(512, 512) for _ in range(4)))
    want = [p.detach().clone() for p in model.parameters()]
    h = ModelHandle(model, name="m", load_device="cuda")
    h.keep_host = True
    budget = PinBudget(1 << 30)
    h.load(None, Streamer("cuda", pin=budget))
    assert h.loaded and budget.used == 4 * 512 * 512 * 4  # the weights (the biases are too small)
    assert all(t.is_pinned() for t in h._host.values() if t.numel() * t.element_size() >= pinning.MIN_BYTES)
    for _ in range(2):
        h.offload()
        h.load(None, Streamer("cuda", pin=budget))
        torch.cuda.synchronize()
        assert all(p.device.type == "cuda" and torch.equal(p.detach().cpu(), w) for p, w in zip(model.parameters(), want, strict=True))
    h.free()
    del h
    gc.collect()
    assert budget.used == 0


@nvidia
def test_streamed_units_are_pinned_in_place():
    model = Stack()
    x = torch.randn(2, 512)
    with torch.no_grad():
        want = model(x)
    h = ModelHandle(model, name="m", load_device="cuda", kind="sequence")
    budget = PinBudget(1 << 30)
    streamer = Streamer("cuda", pin=budget)
    h.load(h.minimum_bytes(1), streamer)
    assert not h.loaded and budget.used > 0
    streamed = [h._host[(id(m), n)] for u in h.units if not u.resident for m, n in u.slots]
    assert streamed and all(t.is_pinned() for t in streamed)
    with torch.no_grad():
        got = model(x.cuda())
    assert torch.allclose(got.cpu(), want, atol=1e-5)  # the GPU's own sums: close to the CPU's
    with torch.no_grad():
        again = model(x.cuda())
    assert torch.equal(again, got)
