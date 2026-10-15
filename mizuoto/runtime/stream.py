"""Weights that stream to the device as their units run.

One ``Streamer`` per device, owned by the model manager. A partly loaded handle (``runtime.handle``) asks it for the
device copies of a unit's weights when the unit is entered and gives them back when it leaves. The streamer:

- **keeps a few copies** (``copies``: the running unit and those in flight) and reuses one when its unit runs again
  before it is dropped (CFG halves, beams, a unit called twice);
- **learns the order** in which units run (each unit's successor, as last seen) and, with side streams (CUDA and ROCm),
  **prefetches** the next ``streams`` units on them while the current one computes. The compute stream waits on a
  copy's event before it uses it, and each copy is recorded on the compute stream, so the caching allocator does not
  reuse its memory early. A unit entered out of the learnt order is copied there and then; correctness never depends
  on the order;
- **pins** the host copies of streamed weights (page-locked memory makes the copies asynchronous and about twice as
  fast) within the policy's budget; beyond it they stay pageable and copy synchronously.

Copies are made in the weights' own dtype, outside inference mode, as the resident weights are: a streamed unit
computes with the same tensors (same values, same kind) as a resident one.
"""

from __future__ import annotations

import logging
import threading
from collections import OrderedDict
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from mizuoto.runtime.handle import ModelHandle
    from mizuoto.runtime.units import Unit

logger = logging.getLogger(__name__)

Key = tuple[int, int]
"""(handle id, unit index)"""


class PinBudget:
    """Page-locked bytes, shared by every streamer of the process."""

    def __init__(self, limit: int) -> None:
        self.limit = limit
        self.used = 0
        self._lock = threading.Lock()

    def take(self, nbytes: int) -> bool:
        with self._lock:
            if self.used + nbytes > self.limit:
                return False
            self.used += nbytes
            return True

    def give(self, nbytes: int) -> None:
        with self._lock:
            self.used = max(0, self.used - nbytes)


class Streamer:
    def __init__(self, device: str, streams: int = 0, pin: PinBudget | None = None) -> None:
        self.device = device
        self.streams = streams
        self.copies = streams + 1
        """Copies held at once: the running unit and the ``streams`` prefetched."""
        self._pin = pin
        self._cache: OrderedDict[Key, tuple[list[Any], Any]] = OrderedDict()
        self._busy: set[Key] = set()
        self._next: dict[Key, Key] = {}
        self._last: Key | None = None
        self._units: dict[Key, tuple[ModelHandle, Unit]] = {}
        self._pinned: dict[int, list[tuple[tuple[int, str], int]]] = {}
        """Per handle: the host tensors this streamer pinned (key, bytes)."""
        self._side: list[Any] = []
        self._turn = 0
        self._lock = threading.RLock()
        self.copied_bytes = 0
        """Bytes copied to the device so far (reported per request)."""
        self.hits = 0
        """Units whose copy was there when they ran (prefetched, or kept from a call before)."""
        self._held = 0
        self.peak_bytes = 0
        """The most bytes the copies held at once since ``reset_peak`` (a phase's measured working memory leaves them out)."""
        if streams > 0 and device.split(":", 1)[0] == "cuda":
            import torch

            self._side = [torch.cuda.Stream(device=torch.device(device)) for _ in range(streams)]

    # -- handles ----------------------------------------------------------------------

    def attach(self, handle: ModelHandle) -> None:
        """``handle`` streams some of its units here: know them, and pin their host copies within the budget."""
        with self._lock:
            for u in handle.units:
                self._units[(id(handle), u.index)] = (handle, u)
            if self._pin is None or self.device == "cpu":
                return
            pinned = self._pinned.setdefault(id(handle), [])
            done = {k for k, _ in pinned}
            for u in handle.units:
                if u.resident:
                    continue
                for m, n in u.slots:
                    key = (id(m), n)
                    host = handle._host.get(key)
                    if host is None or key in done or host.is_pinned():
                        continue
                    size = host.numel() * host.element_size()
                    if not self._pin.take(size):
                        return
                    try:
                        pinned_host = host.pin_memory()
                    except RuntimeError as e:  # the OS refused (limits); the rest stays pageable
                        self._pin.give(size)
                        logger.info("Could not pin %s's weights: %s", handle.name, e)
                        return
                    handle._host[key] = pinned_host
                    if m._parameters[n].data is host:
                        m._parameters[n].data = pinned_host
                    pinned.append((key, size))

    def detach(self, handle: ModelHandle) -> None:
        with self._lock:
            self.release_all(handle)
            for key in [k for k in self._units if k[0] == id(handle)]:
                self._units.pop(key, None)
                self._next.pop(key, None)
            self._next = {k: v for k, v in self._next.items() if v[0] != id(handle)}
            if self._last is not None and self._last[0] == id(handle):
                self._last = None
            for _key, size in self._pinned.pop(id(handle), []):
                if self._pin is not None:
                    self._pin.give(size)

    # -- copies -----------------------------------------------------------------------

    def fetch(self, handle: ModelHandle, unit: Unit, hosts: list[Any]) -> list[Any]:
        """The device copies of ``unit``'s weights, ready for the compute stream."""
        key = (id(handle), unit.index)
        with self._lock:
            if self._last is not None and self._last != key:
                self._next[self._last] = key
            self._last = key
            entry = self._cache.pop(key, None)
            if entry is None:
                tensors = [self._copy(h) for h in hosts]
                self.copied_bytes += _nbytes(tensors)
                self._grow(_nbytes(tensors))
            else:
                self.hits += 1
                tensors, event = entry
                if event is not None:
                    import torch

                    current = torch.cuda.current_stream(torch.device(self.device))
                    current.wait_event(event)
                    for t in tensors:
                        t.record_stream(current)
            self._cache[key] = (tensors, None)
            self._busy.add(key)
            self._prefetch(key)
            self._trim()
            return tensors

    def done(self, handle: ModelHandle, unit: Unit) -> None:
        with self._lock:
            self._busy.discard((id(handle), unit.index))
            self._trim()

    def release_all(self, handle: ModelHandle) -> None:
        """Drop every copy of ``handle``'s units (the request ended, or memory is needed)."""
        with self._lock:
            for key in [k for k in self._cache if k[0] == id(handle)]:
                self._held -= _nbytes(self._cache.pop(key)[0])
                self._busy.discard(key)

    def clear(self) -> None:
        with self._lock:
            self._cache.clear()
            self._busy.clear()
            self._held = 0

    def reset_peak(self) -> None:
        with self._lock:
            self.peak_bytes = self._held

    def _grow(self, nbytes: int) -> None:
        self._held += nbytes
        self.peak_bytes = max(self.peak_bytes, self._held)

    @property
    def cached_bytes(self) -> int:
        with self._lock:
            return self._held

    def _copy(self, host: Any) -> Any:
        import torch

        with torch.inference_mode(False), torch.no_grad():
            return host.to(self.device, copy=True, non_blocking=self.device != "cpu" and host.is_pinned())

    def _prefetch(self, key: Key) -> None:
        """Copy the units that followed ``key`` last time, on the side streams (on a device without them, a test's
        simulated one, synchronously): by the time they run their weights are there."""
        if self.streams <= 0:
            return
        nxt = key
        for _ in range(self.streams):
            nxt = self._next.get(nxt)  # type: ignore[assignment]
            if nxt is None or nxt == key:
                return
            if nxt in self._cache:
                continue
            found = self._units.get(nxt)
            if found is None:
                return
            handle, unit = found
            if unit.resident:
                continue
            found_hosts = [handle._host.get((id(m), n)) for m, n in unit.slots]
            hosts: list[Any] = [h for h in found_hosts if h is not None]
            if len(hosts) != len(found_hosts):
                continue
            tensors, event = self._side_copy(hosts)
            self.copied_bytes += _nbytes(tensors)
            self._grow(_nbytes(tensors))
            self._cache[nxt] = (tensors, event)

    def _side_copy(self, hosts: list[Any]) -> tuple[list[Any], Any]:
        """``hosts`` copied on the next side stream, with the event the compute stream waits on before using them. No
        wait the other way: the copies go to fresh memory, and ``fetch`` records them on the compute stream, so the
        caching allocator reuses none of it early (ComfyUI fences reused cast buffers instead)."""
        import torch

        if not self._side:
            return [self._copy(h) for h in hosts], None
        side = self._side[self._turn % len(self._side)]
        self._turn += 1
        with torch.cuda.stream(side), torch.inference_mode(False), torch.no_grad():
            tensors = [h.to(self.device, copy=True, non_blocking=h.is_pinned()) for h in hosts]
            event = torch.cuda.Event()
            event.record(side)
        return tensors, event

    def _trim(self) -> None:
        while len(self._cache) > self.copies:
            victim = next((k for k in self._cache if k not in self._busy), None)
            if victim is None:
                return
            self._held -= _nbytes(self._cache.pop(victim)[0])


def _nbytes(tensors: list[Any]) -> int:
    return sum(t.numel() * t.element_size() for t in tensors)
