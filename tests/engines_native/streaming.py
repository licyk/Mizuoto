"""Run a tiny engine on a simulated device with its weights placed phase by phase and partly streamed
, and check it gives exactly what it gives on the plain CPU.

The CPU plays a device of a given size (``DeviceService(simulate=…)``) in strict mode: a component off the device rests
as meta tensors and a streamed unit's weights are there only while the unit runs, so an engine that computes with a
component it did not place for the phase (``ctx.use``), or reads a unit's weights from outside the unit, fails here.
Unit size floors are lowered so the tiny models have units at all.
"""

from __future__ import annotations

from collections.abc import Callable, Generator
from contextlib import contextmanager
from typing import Any

import numpy as np
import torch

from mizuoto.runtime import units
from mizuoto.runtime.device import DeviceService
from mizuoto.runtime.manager import ModelManager
from mizuoto.runtime.policy import VramPolicy

FRACTIONS = (0.0, 0.5, 1.0)
"""Device sizes between the engine's minimum and the whole model; at 0 the mode is ``none`` (every unit streams in every
phase)."""


@contextmanager
def simulated(loaded: Any, fraction: float, monkeypatch: Any) -> Generator[ModelManager, None, None]:
    """``loaded``'s handles in a manager on a simulated device sized ``fraction`` of the way from the minimum to all."""
    monkeypatch.setattr(units, "UNIT_MIN_BYTES", 0)
    monkeypatch.setattr(units, "TINY_BYTES", 1)
    for h in loaded.handles:
        h._split = None
    devices = DeviceService(simulate={"cpu": 1 << 40}, strict=True)
    policy = VramPolicy(mode="none" if fraction == 0 else "normal", reserve_bytes=0, host_copies="on")
    manager = ModelManager(devices, policy=lambda: policy)
    manager.register("tiny", loaded.handles)
    copies = 1
    total = sum(h.size_bytes for h in loaded.handles)
    minimum = sum(h.minimum_bytes(copies) for h in loaded.handles)
    devices._simulate["cpu"] = int(minimum + fraction * (total - minimum)) + 1
    try:
        yield manager
    finally:
        manager.release("tiny")


def check_streamed(loaded: Any, run: Callable[[Any], Any], monkeypatch: Any, fractions: tuple[float, ...] = FRACTIONS) -> list[int]:
    """``run(ctx_manager)`` → audio (numpy) or a tuple of arrays, with and without the simulated device: identical.
    Returns the bytes streamed at each fraction (0 means nothing streamed)."""
    expected = run(None)
    streamed = []
    for fraction in fractions:
        with simulated(loaded, fraction, monkeypatch) as manager:
            before = manager.streamed_bytes("cpu")
            whole = None if type(loaded).PHASED else manager.use(loaded.handles)
            if whole is not None:
                with whole:
                    got = run(manager)
            else:
                got = run(manager)
            streamed.append(manager.streamed_bytes("cpu") - before)
        _equal(got, expected, fraction)
    return streamed


def _equal(got: Any, expected: Any, fraction: float) -> None:
    if isinstance(expected, tuple | list):
        assert len(got) == len(expected)
        for g, e in zip(got, expected, strict=True):
            _equal(g, e, fraction)
        return
    if isinstance(expected, torch.Tensor):
        assert torch.equal(got, expected), f"differs with {fraction:.0%} of the room"
        return
    assert np.array_equal(np.asarray(got), np.asarray(expected)), f"differs with {fraction:.0%} of the room"


def check_oom_retry(loaded: Any, run: Callable[..., Any], module: Any, monkeypatch: Any) -> dict[str, int]:
    """``module`` (a layer inside one of ``loaded``'s phases) runs out of memory the first time it is called: the phase
    runs again and the result is exactly the undisturbed one. Returns the request's ``oom_retries``."""
    expected = run(None)
    state = {"left": 1}
    original = module.forward

    def flaky(*args: Any, **kwargs: Any) -> Any:
        if state["left"]:
            state["left"] -= 1
            raise torch.OutOfMemoryError("CUDA out of memory. Tried to allocate 1.00 GiB")
        return original(*args, **kwargs)

    retries: dict[str, int] = {}
    with simulated(loaded, 1.0, monkeypatch) as manager:
        monkeypatch.setattr(module, "forward", flaky)
        got = run(manager, retries)
        monkeypatch.setattr(module, "forward", original)
    assert state["left"] == 0, "the layer never ran"
    _equal(got, expected, 1.0)
    return retries
