"""Which model components are on which device, and how much of each.

The equivalent of ComfyUI's ``load_models_gpu`` / ``free_memory`` and its ``current_loaded_models`` list. Every use of
a model's components begins with ``load(handles, memory_required)``:

1. **Room.** Free memory must cover the components' bytes not yet on the device, the request's working memory and the
   policy's reserve. When it does not, other components give up memory, unit by unit (ComfyUI's smart memory): partly
   loaded ones first (already slow), then the least recently used; components in use never, pinned models' last.
   ``mode=high`` gives up whole components instead.
2. **Whole or partly.** When the components fit, they load whole. Otherwise (modes ``normal``, ``low``, ``none``) the
   weights get what working memory and the reserve leave: smaller components whole first, the largest partly, its
   remaining units streaming from host memory through the device's ``Streamer``. Below even the components' minimum
   (fixed parts and the copies in flight) they are tried at that minimum anyway (ComfyUI's way): only an allocation
   that really fails is an error. In every mode no estimate refuses a load, except to wait while another request holds
   the device (``_load_waiting``).

- Offloading keeps the weights in RAM, so the next request reloads them cheaply; with host copies it costs nothing.
- ``unregister`` frees them, with no copy back, when a model is unloaded.
- On the CPU nothing is moved and nothing is counted, unless the device is simulated (``DeviceService.simulate``:
  tests and the parity harness), where the manager counts what it placed against the simulated size.
- ``empty_cache`` runs after components give up memory, never per request (ComfyUI's policy).
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable, Generator, Iterable
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any

from mizuoto.core import codes
from mizuoto.runtime.device import DeviceService, device_kind
from mizuoto.runtime.handle import ModelHandle
from mizuoto.runtime.policy import VramPolicy
from mizuoto.runtime.stream import PinBudget, Streamer

logger = logging.getLogger(__name__)

CALIBRATION_VERSION = 2
"""The calibration file's format; ratios saved by another (version 1: a flat mapping, measured with streamed copies
against the worst-case estimates) are dropped."""

WAIT_SECONDS = 600.0
"""How long a request whose minimum does not fit waits for another request on the device to let go."""


@dataclass
class _Entry:
    handle: ModelHandle
    owner: str
    last_used: float = field(default_factory=time.monotonic)
    in_use: int = 0
    mode: str | None = None
    pinned: bool = False


@dataclass
class ResidentInfo:
    name: str
    owner: str
    device: str
    load_device: str
    bytes: int
    loaded: bool
    in_use: bool
    idle_seconds: float
    loaded_bytes: int = 0
    streamed_bytes: int = 0
    host_bytes: int = 0
    pinned_bytes: int = 0


class ModelManager:
    def __init__(self, devices: DeviceService, reserve_bytes: Callable[[str], int] | None = None, policy: Callable[[], VramPolicy] | None = None) -> None:
        self.devices = devices
        self._reserve = reserve_bytes
        self._policy = policy or (lambda: VramPolicy())
        self._entries: list[_Entry] = []
        self._streamers: dict[str, Streamer] = {}
        self._pin: PinBudget | None = None
        self._lock = threading.RLock()
        self._waiting = threading.local()
        """``active``: this thread is in ``_load_waiting`` before its deadline (a ``high`` load may then wait)."""
        self._released = threading.Condition(self._lock)
        """Notified when a request lets go of components (another may be waiting for the memory)."""
        self.calibration: dict[str, float] = {}
        """Per phase (its components' names): the most working memory measured over the estimate, ≥ 1."""
        self.calibration_path: Any = None
        """Where the calibration is kept between runs (a JSON file the core names), or None."""

    @property
    def policy(self) -> VramPolicy:
        policy = self._policy()
        if self._reserve is not None:
            return policy.with_(reserve_bytes=self._reserve(""))
        return policy

    # -- registration -----------------------------------------------------------------

    def register(self, owner: str, handles: Iterable[ModelHandle], mode: str | None = None, pinned: bool = False) -> None:
        """Track ``owner``'s components. ``mode`` overrides the policy's for them (``engines.<id>.vram_mode``);
        ``pinned`` components give up memory last."""
        with self._lock:
            for h in handles:
                found = next((e for e in self._entries if e.handle is h), None)
                if found is None:
                    h.simulated = self.devices.simulated(h.load_device) is not None
                    h.strict = h.simulated and self.devices.strict
                    if h.strict:
                        h.keep_host = True
                        h.park()
                    self._entries.append(_Entry(h, owner, mode=mode, pinned=pinned))
                else:
                    found.mode, found.pinned = mode, pinned

    def release(self, owner: str) -> None:
        """Forget ``owner``'s components without freeing them: back on the host, as the engine built them (a test hands
        a model to another manager)."""
        with self._lock:
            mine = [e for e in self._entries if e.owner == owner]
            self._entries = [e for e in self._entries if e.owner != owner]
        for e in mine:
            e.handle.offload()
            e.handle.unpark()
            e.handle.simulated = e.handle.strict = False
            e.handle._split = None

    def unregister(self, owner: str) -> None:
        """Forget ``owner``'s components and free their weights (no copy back)."""
        with self._lock:
            mine = [e for e in self._entries if e.owner == owner]
            self._entries = [e for e in self._entries if e.owner != owner]
        devices = {e.handle.load_device for e in mine if e.handle.loaded_bytes}
        for e in mine:
            e.handle.free()
        for d in devices:
            if self._accelerated(d):
                self.devices.empty_cache(d)

    # -- placement --------------------------------------------------------------------

    def load(self, handles: Iterable[ModelHandle], memory_required: int = 0) -> None:
        """Put ``handles`` on their devices, whole or partly, making room first. Inside ``_load_waiting`` (before its
        deadline) a ``high`` load refuses while another request holds what is missing, so that it waits instead."""
        wait = getattr(self._waiting, "active", False)
        handles = list(handles)
        with self._lock:
            by_device: dict[str, list[ModelHandle]] = {}
            for h in handles:
                by_device.setdefault(h.load_device, []).append(h)
            for device, group in by_device.items():
                if self._accelerated(device):
                    self._load_on(device, group, memory_required, wait)
                else:
                    for h in group:
                        h.load()
            now = time.monotonic()
            for e in self._entries:
                if any(e.handle is h for h in handles):
                    e.last_used = now
            # most recently used first, as ComfyUI's current_loaded_models
            self._entries.sort(key=lambda e: -e.last_used)

    def _load_on(self, device: str, group: list[ModelHandle], memory: int, wait: bool = False) -> None:
        policy = self.policy
        if self.devices.simulated(device) is not None and not self.devices.simulate_working:
            memory = 0  # a simulated device holds weights only: activations live in the CPU's own memory
        kind = device_kind(device)
        mode = policy.mode_for(kind, self._mode(group))
        streamer = self._streamer(device, policy)
        reserve = policy.reserve_bytes
        need = sum(h.size_bytes - h.loaded_bytes for h in group) + memory + reserve
        free = self._free(device)
        if free is not None and free < need:
            self._evict(device, need - free, keep=group, whole=mode in ("high", "shared"))
            free = self._free(device)
        capped = any(h.cap_bytes is not None and h.cap_bytes < h.size_bytes for h in group) and mode not in ("high", "shared")
        if wait and mode in ("high", "shared") and free is not None and free < need and self._others_busy(group):
            # whole or not at all, and another request holds what is missing: it waits for it (``_load_waiting``);
            # otherwise whole loads are tried whatever the estimate says, failing only when an allocation does
            raise _out_of_memory(device, group, free - memory - reserve, sum(h.size_bytes - h.loaded_bytes for h in group), memory, reserve)
        whole = (free is None or free >= need or mode in ("high", "shared")) and not capped
        if whole and mode == "low":  # weights take at most ``low_ratio`` of the device, fitting or not
            _, total = self._memory(device)
            whole = total is None or sum(h.size_bytes for h in group) <= total * policy.low_ratio
        if whole and mode != "none":
            for h in sorted(group, key=lambda h: h.size_bytes):
                # (unified memory: a host copy would be a second copy in the same memory)
                h.keep_host = self._keep_host(policy) and mode != "shared"
                h.load(None, streamer)
                h.placement = None
            return
        if mode == "none":
            why = "vram_mode none: every unit streams"
        elif capped:
            why = "capped at half its weights after running out of memory"
        elif mode == "low" and (free is None or free >= need):
            why = f"vram_mode low: weights take at most {policy.low_ratio:.0%} of the device"
        else:
            why = "the weights, the working memory and the reserve exceed the free memory"
        placement = {
            "reason": why,
            "mode": mode,
            "free_mb": None if free is None else round((free + sum(h.loaded_bytes for h in group)) / 2**20),
            "working_mb": round(memory / 2**20),
            "reserve_mb": round(reserve / 2**20),
        }
        # partly: the weights get what the request's working memory and the reserve leave
        if free is None:  # a device that cannot tell (its components are capped after running out of memory)
            free = sum(h.size_bytes for h in group) + memory + reserve
        budget = free + sum(h.loaded_bytes for h in group) - memory - reserve
        if mode == "low":
            _, total = self._memory(device)
            budget = min(budget, int((total or budget) * policy.low_ratio))
        copies = streamer.copies
        minimums = [h.minimum_bytes(copies) for h in group]
        if mode == "none":
            budget = min(budget, sum(minimums))
        if budget < sum(minimums):
            if wait and self._others_busy(group):
                # another request holds what is missing: it waits for it (``_load_waiting``)
                raise _out_of_memory(device, group, budget, sum(minimums), memory, reserve)
            # ComfyUI's way: tried at the smallest working set whatever the estimate says; only an allocation fails
            logger.warning(
                "%s: about %d MB free for %s, below its smallest working set (%d MB); trying anyway",
                device,
                max(0, budget) >> 20,
                ", ".join(h.name for h in group),
                sum(minimums) >> 20,
            )
            budget = sum(minimums)
        # smaller components whole first, each leaving the others at least their minimum
        order = sorted(range(len(group)), key=lambda i: group[i].size_bytes)
        rest = sum(minimums)
        for i in order:
            h = group[i]
            rest -= minimums[i]
            share = budget - rest
            if h.cap_bytes is not None:
                share = min(share, max(h.cap_bytes, minimums[i]))
            h.keep_host = True  # a streamed unit rests on its host copy
            h.load(None if share >= h.size_bytes and mode != "none" else share, streamer)
            budget -= h.loaded_bytes + h.stream_reserve([u for u in h.units if not u.resident], copies)
            h.placement = {**placement, "loaded_mb": round(h.loaded_bytes / 2**20), "size_mb": round(h.size_bytes / 2**20)} if h.partial else None
            if h.partial:
                logger.info(
                    "%s: %d of %d MB on %s, the rest streams from RAM (%s: %s MB free, %d MB set aside for working memory, %d MB reserve)",
                    h.name,
                    h.loaded_bytes >> 20,
                    h.size_bytes >> 20,
                    device,
                    why,
                    placement["free_mb"],
                    placement["working_mb"],
                    placement["reserve_mb"],
                )

    def _evict(self, device: str, shortfall: int, keep: list[ModelHandle], whole: bool) -> int:
        """Free ``shortfall`` bytes on ``device`` from idle components; returns what was freed."""
        victims = [e for e in self._entries if e.handle.load_device == device and e.handle.loaded_bytes > 0 and e.in_use == 0 and not any(e.handle is k for k in keep)]
        # partly loaded first (already slow), then unpinned, then the least recently used
        victims.sort(key=lambda e: (not e.handle.partial, e.pinned, e.last_used))
        freed = 0
        for e in victims:
            if freed >= shortfall:
                break
            logger.info("Making room on %s: %s gives up memory", device, e.handle.name)
            if whole or not e.handle.units:
                freed += e.handle.loaded_bytes
                e.handle.offload()
            else:
                streamer = self._streamer(device, self.policy)
                if e.handle._streamer is None:
                    e.handle._streamer = streamer
                    streamer.attach(e.handle)
                freed += e.handle.partially_unload(shortfall - freed)
                if e.handle.partial:  # why it was partly loaded, if it was, and for what it gave up memory since
                    taker = ", ".join(k.name.split(":", 1)[-1] for k in keep)
                    prior = e.handle.placement or {"reason": f"gave up memory for {taker}"}
                    e.handle.placement = {**prior, "gave_up_for": taker, "loaded_mb": round(e.handle.loaded_bytes / 2**20), "size_mb": round(e.handle.size_bytes / 2**20)}
        if freed:
            self.devices.empty_cache(device)
        return freed

    @contextmanager
    def use(self, handles: Iterable[ModelHandle], memory_required: int = 0) -> Generator[None, None, None]:
        """Load ``handles`` and keep them from giving up memory for the block. The working memory is the estimate
        corrected by what this phase was measured to need before (``calibration``), and is measured again."""
        handles = list(handles)
        key = calibration_key(handles)
        estimate = int(memory_required)
        with self._lock:
            entries = [e for e in self._entries if any(e.handle is h for h in handles)]
            for e in entries:
                e.in_use += 1
            ratio = self.calibration.get(key, 1.0)
        measured: list[tuple[str, int]] = []
        try:
            self._load_waiting(handles, int(estimate * ratio) if estimate else estimate)
            if estimate and len({h.load_device for h in handles}) == 1:
                device = handles[0].load_device
                base = self.devices.allocated(device)
                if base is not None:
                    self.devices.start_peak(device)
                    streamer = self._streamers.get(device)
                    if streamer is not None:
                        streamer.reset_peak()
                    measured.append((device, base))
            yield
            for device, base in measured:
                peak, now = self.devices.peak(device), self.devices.allocated(device)
                if peak is not None and now is not None:
                    self._calibrate(key, self._working(device, peak - base, now - base), estimate)
        finally:
            with self._released:
                for e in entries:
                    e.in_use -= 1
                    e.last_used = time.monotonic()
                    if e.in_use == 0:
                        e.handle.settle()
                self._released.notify_all()

    def _load_waiting(self, handles: list[ModelHandle], memory: int) -> None:
        """``load``; when even the minimum does not fit while another request holds components on the device (two GPU
        jobs at once, ``compute.gpu_jobs``), wait for it to let go instead of failing, checking for a cancel."""
        from mizuoto.runtime.context import check_cancel
        from mizuoto.runtime.engine import EngineError

        deadline = time.monotonic() + WAIT_SECONDS
        while True:
            try:
                self._waiting.active = time.monotonic() < deadline
                self.load(handles, memory)
                return
            except EngineError as e:
                if e.code != codes.COMPUTE_UNAVAILABLE or time.monotonic() > deadline or not self._others_busy(handles):
                    raise
            finally:
                self._waiting.active = False
            logger.info("Waiting for another request to free %s", handles[0].load_device if handles else "the device")
            with self._released:
                self._released.wait(timeout=1.0)
            check_cancel()

    def _others_busy(self, handles: list[ModelHandle]) -> bool:
        devices = {h.load_device for h in handles}
        with self._lock:
            return any(e.in_use > 0 and e.handle.load_device in devices and not any(e.handle is h for h in handles) for e in self._entries)

    def _working(self, device: str, peak: int, kept: int) -> int:
        """A phase's working memory from its peak above the placed weights: without the copies of streamed weights (the
        budget counts those with the weights) nor what the phase keeps after it (CUDA graphs' pools, buffers kept with
        the model: there next time, in the free memory's count). Both are taken at their most, so this may come out
        under the truth, never over: a calibration only ever sets aside more."""
        streamer = self._streamers.get(device)
        copies, cached = (streamer.peak_bytes, streamer.cached_bytes) if streamer is not None else (0, 0)
        return max(0, peak - copies - max(0, kept - cached))

    def _calibrate(self, key: str, used: int, estimate: int) -> None:
        """Remember how far a phase's working memory went over its estimate (never under: estimates may be generous)."""
        ratio = used / estimate
        with self._lock:
            if ratio <= self.calibration.get(key, 1.0):
                return
            self.calibration[key] = min(ratio * 1.1, 16.0)
            path = self.calibration_path
        if ratio > 1.0:
            logger.info("%s needed %.1f× its working-memory estimate; later requests set aside more", key, ratio)
        if path is not None:
            import json
            from pathlib import Path

            try:
                Path(path).parent.mkdir(parents=True, exist_ok=True)
                Path(path).write_text(json.dumps({"version": CALIBRATION_VERSION, "phases": self.calibration}, indent=1, sort_keys=True), encoding="utf-8")
            except OSError:
                logger.warning("Could not save the memory calibration to %s", path)

    def load_calibration(self, path: Any) -> None:
        """Read a calibration kept by an earlier run (its file), and keep it there from now on."""
        import json
        from pathlib import Path

        self.calibration_path = path
        try:
            data = json.loads(Path(path).read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return
        if not isinstance(data, dict) or data.get("version") != CALIBRATION_VERSION or not isinstance(data.get("phases"), dict):
            return  # measured against other estimates (or with streamed copies counted): started again
        data = data["phases"]
        with self._lock:
            self.calibration.update({str(k): float(v) for k, v in data.items() if isinstance(v, (int, float)) and 1.0 <= float(v) <= 16.0})

    def release_idle(self, device: str | None = None) -> list[str]:
        """Offload every component not in use (on ``device``), e.g. after running out of memory."""
        with self._lock:
            victims = [e for e in self._entries if e.in_use == 0 and e.handle.loaded_bytes > 0 and (device is None or e.handle.load_device == device)]
            for e in victims:
                e.handle.offload()
            for d, s in self._streamers.items():
                if device is None or d == device:
                    s.clear()
        for d in {e.handle.load_device for e in victims}:
            if self._accelerated(d):
                self.devices.empty_cache(d)
        return [e.handle.name for e in victims]

    def after_oom(self, handles: Iterable[ModelHandle]) -> None:
        """A phase over ``handles`` ran out of memory: every other idle component on their devices leaves them and the
        cached copies of streamed units go, before it runs again (with more working memory set aside)."""
        handles = list(handles)
        with self._lock:
            devices = {h.load_device for h in handles}
            for e in self._entries:
                if e.in_use == 0 and e.handle.loaded_bytes > 0 and e.handle.load_device in devices and not any(e.handle is h for h in handles):
                    e.handle.offload()
            for h in handles:
                h.settle()
                if h.loaded_bytes and self.policy.mode_for(device_kind(h.load_device), self._mode([h])) not in ("high", "shared"):
                    # half of what it had on the device, until it is offloaded (the next requests would run out too)
                    streamer = self._streamer(h.load_device, self.policy)
                    h.cap_bytes = max(h.minimum_bytes(streamer.copies), h.loaded_bytes // 2)
            for d in devices:
                streamer = self._streamers.get(d)
                if streamer is not None:
                    streamer.clear()
        for d in devices:
            if self._accelerated(d):
                self.devices.empty_cache(d)

    def drop_host_copies(self) -> int:
        """Host memory is short: forget the host copies of fully loaded components (offloading them then copies back)."""
        with self._lock:
            return sum(e.handle.drop_host_copies() for e in self._entries if e.handle.loaded)

    # -- memory -----------------------------------------------------------------------

    def _accelerated(self, device: str) -> bool:
        return device_kind(device) != "cpu" or self.devices.simulated(device) is not None

    def _memory(self, device: str) -> tuple[int | None, int | None]:
        simulated = self.devices.simulated(device)
        if simulated is not None:
            used = sum(e.handle.loaded_bytes for e in self._entries if e.handle.load_device == device)
            streamer = self._streamers.get(device)
            used += streamer.cached_bytes if streamer is not None else 0
            return max(0, simulated - used), simulated
        return self.devices.memory(device)

    def _free(self, device: str) -> int | None:
        return self._memory(device)[0]

    def _mode(self, group: list[ModelHandle]) -> str | None:
        for e in self._entries:
            if any(e.handle is h for h in group) and e.mode:
                return e.mode
        return None

    def _streamer(self, device: str, policy: VramPolicy) -> Streamer:
        streamer = self._streamers.get(device)
        if streamer is None:
            kind = device_kind(device)
            if self._pin is None:
                _, ram = self.devices.memory("cpu")
                self._pin = PinBudget(policy.pinned_budget(ram) if kind == "cuda" else 0)
            streams = policy.streams_for(kind) if self.devices.simulated(device) is None else 0
            streamer = Streamer(device, streams, self._pin if kind == "cuda" else None)
            self._streamers[device] = streamer
        return streamer

    def _keep_host(self, policy: VramPolicy) -> bool:
        if policy.host_copies != "auto":
            return policy.host_copies == "on"
        free, total = self.devices.memory("cpu")
        if free is None:
            return False
        held = sum(e.handle.host_bytes for e in self._entries)
        return free - held > policy.ram_reserve(total)

    # -- reporting --------------------------------------------------------------------

    def resident(self) -> list[ResidentInfo]:
        now = time.monotonic()
        with self._lock:
            return [
                ResidentInfo(
                    name=e.handle.name,
                    owner=e.owner,
                    device=e.handle.device,
                    load_device=e.handle.load_device,
                    bytes=e.handle.size_bytes,
                    loaded=e.handle.loaded,
                    in_use=e.in_use > 0,
                    idle_seconds=now - e.last_used,
                    loaded_bytes=e.handle.loaded_bytes,
                    streamed_bytes=e.handle.streamed_bytes,
                    host_bytes=e.handle.host_bytes,
                    pinned_bytes=e.handle.pinned_bytes,
                )
                for e in self._entries
            ]

    def owner_bytes(self, owner: str) -> int:
        with self._lock:
            return sum(e.handle.size_bytes for e in self._entries if e.owner == owner)

    def owner_placements(self, owner: str) -> list[dict[str, Any]]:
        """Why ``owner``'s partly loaded components are partly loaded: each one's last placement (``ModelHandle.placement``)."""
        with self._lock:
            return [{"component": e.handle.name.split(":", 1)[-1], **e.handle.placement} for e in self._entries if e.owner == owner and e.handle.partial and e.handle.placement]

    def owner_usage(self, owner: str) -> dict[str, int]:
        """``loaded``, ``streamed`` and ``host`` bytes of ``owner``'s components."""
        with self._lock:
            mine = [e.handle for e in self._entries if e.owner == owner]
            return {"loaded": sum(h.loaded_bytes for h in mine), "streamed": sum(h.streamed_bytes for h in mine), "host": sum(h.host_bytes for h in mine)}

    def streamed_bytes(self, device: str) -> int:
        """Bytes copied to ``device`` by its streamer so far (a request reports the difference)."""
        streamer = self._streamers.get(device)
        return streamer.copied_bytes if streamer is not None else 0


def _out_of_memory(device: str, group: list[ModelHandle], budget: int, minimum: int, memory: int, reserve: int) -> Exception:
    from mizuoto.runtime.engine import EngineError

    names = ", ".join(h.name for h in group)
    short = (minimum - budget) / 2**20
    return EngineError(
        f"{device} lacks {short:.0f} MB to run {names}, even with most of its weights streamed from RAM",
        {
            "reason": codes.OUT_OF_MEMORY,
            "device": device,
            "budget_mb": round(budget / 2**20),
            "minimum_mb": round(minimum / 2**20),
            "working_mb": round(memory / 2**20),
            "reserve_mb": round(reserve / 2**20),
            "hint": "Free the GPU (other programs or Mizuoto's other models), lower compute.vram_reserve_mb, or run the engine on another device",
        },
        code=codes.COMPUTE_UNAVAILABLE,
    )


def calibration_key(handles: list[ModelHandle]) -> str:
    """A phase's identity for the calibration: its components' names without the model's (the same phase of every model
    of an engine variant shares it: ``qwen3tts:talker``)."""
    names = sorted(h.name.split(":", 1)[-1] for h in handles)
    engine = handles[0].name.split("/", 1)[0] if handles else ""
    return f"{engine}:{'+'.join(names)}"
