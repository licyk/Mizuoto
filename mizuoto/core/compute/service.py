"""Devices, and the memory held on them.

Each holder of accelerator memory (an engine's loaded models, a training job) registers a ``MemoryHolder``:
whether it is busy, what it has cached, and how to release it. "Free GPU memory" releases every
holder that is not busy; an out-of-memory error releases every idle holder on that device.
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field

from mizuoto.core.compute.models import ComputeInfo, ComputeUsage, DeviceInfo, HolderUsage
from mizuoto.core.compute.probe import TorchProbe, probe
from mizuoto.core.errors import BusyError, ComputeError, DeviceError
from mizuoto.core.events import EventBus
from mizuoto.core.events.models import ComputeChangedEvent
from mizuoto.core.settings import SettingsService

logger = logging.getLogger(__name__)

PROBE_MAX_AGE = 600.0
USAGE_EVENT_INTERVAL = 1.0


def runs_on(kind: str, supported: Iterable[str]) -> bool:
    """Whether a device of ``kind`` suits an engine that lists ``supported`` kinds. ``cuda`` is torch's device type,
    so it covers AMD GPUs under ROCm (kind ``rocm``, named ``cuda:N``) too; ``rocm`` alone names AMD only."""
    kinds = set(supported)
    return kind in kinds or (kind == "rocm" and "cuda" in kinds)


@dataclass(frozen=True)
class MemoryHolder:
    name: str
    busy: Callable[[], str | None]
    """What keeps it busy, or None."""
    cached: Callable[[], list[str]] = field(default=lambda: [])
    release: Callable[[], None] = field(default=lambda: None)
    device: Callable[[], str | None] = field(default=lambda: None)


class ComputeService:
    def __init__(self, settings: SettingsService, events: EventBus, prober: Callable[[], ComputeInfo] | None = None) -> None:
        self._settings = settings
        self._events = events
        self._prober: Callable[[], ComputeInfo] = prober or (lambda: probe(self.torch_probe))
        self.torch_probe: TorchProbe | None = None
        """What torch sees when it is loaded in this process (set by the runtime service; None: never ask torch here)."""
        self._info: ComputeInfo | None = None
        self._probed_at = 0.0
        self._holders: dict[str, MemoryHolder] = {}
        self._lock = threading.RLock()
        self._last_emit = 0.0

    # -- devices ----------------------------------------------------------------------

    def info(self, refresh: bool = False) -> ComputeInfo:
        with self._lock:
            if refresh or self._info is None or time.monotonic() - self._probed_at > PROBE_MAX_AGE:
                self._info = self._prober()
                self._probed_at = time.monotonic()
            return self._info

    def devices(self) -> list[DeviceInfo]:
        """The devices Mizuoto may use: the probed ones, narrowed by ``compute.devices``."""
        allowed = set(self._settings.settings.compute.devices)
        return [d for d in self.info().devices if not allowed or d.id in allowed or d.kind == "cpu"]

    def resolve_device(self, supported: list[str], requested: str = "auto") -> str:
        """The device an engine runs on: ``requested`` when given, else ``compute.default_device``, else the
        accelerator with the most free memory the engine supports, else the CPU."""
        devices = self.devices()
        if requested == "auto":
            requested = self._settings.settings.compute.default_device
        if requested != "auto":
            if requested == "cpu" and "cpu" in supported:
                return "cpu"
            match = next((d for d in devices if d.id == requested), None)
            if match is None:
                raise DeviceError(f"No device {requested} on this machine", {"reason": "missing", "devices": [d.id for d in devices]})
            if not runs_on(match.kind, supported):
                raise DeviceError(f"This engine cannot run on {requested}", {"reason": "unsupported", "supported": supported})
            return match.id
        accelerators = [d for d in devices if d.kind != "cpu" and runs_on(d.kind, supported)]
        if accelerators:
            best = max(accelerators, key=lambda d: (d.memory_total_mb or 0) - (d.memory_used_mb or 0))
            return best.id
        if "cpu" in supported:
            return "cpu"
        raise ComputeError("No device this engine can run on", {"reason": "no_device", "supported": supported})

    # -- holders ----------------------------------------------------------------------

    def add_holder(self, holder: MemoryHolder) -> None:
        with self._lock:
            self._holders[holder.name] = holder
        self.emit_usage(force=True)

    def remove_holder(self, name: str) -> None:
        with self._lock:
            self._holders.pop(name, None)
        self.emit_usage(force=True)

    def usage(self) -> ComputeUsage:
        with self._lock:
            holders = list(self._holders.values())
        rows = [HolderUsage(name=h.name, busy=h.busy(), cached=h.cached()) for h in holders]
        busy = [f"{r.name}: {r.busy}" for r in rows if r.busy]
        return ComputeUsage(devices=self.devices(), holders=rows, busy=busy, can_release=not busy and any(r.cached for r in rows))

    def release(self) -> ComputeUsage:
        """Unload everything; refused while anything is busy."""
        usage = self.usage()
        if usage.busy:
            raise BusyError("Memory is in use: " + "; ".join(usage.busy), {"busy": usage.busy})
        self._release_idle(None)
        return self.usage()

    def release_after_oom(self, source: str, device: str | None = None) -> None:
        """After a real allocation failure (``device``: where it happened): unload every model not in use from every GPU,
        as ComfyUI's ``unload_all_models`` does after an out-of-memory error, so the next try starts on empty devices.
        Models on the CPU stay."""
        logger.warning("Out of memory in %s%s: unloading every model not in use from the GPUs", source, f" on {device}" if device else "")
        self._release_idle(None, gpus_only=True)

    def _release_idle(self, device: str | None, gpus_only: bool = False) -> None:
        with self._lock:
            holders = list(self._holders.values())
        for h in holders:
            where = h.device()
            if gpus_only and where == "cpu":
                continue
            if h.busy() is None and (device is None or where in (None, device)):
                try:
                    h.release()
                except Exception:
                    logger.exception("Releasing %s failed", h.name)
        self.emit_usage(force=True)

    def emit_usage(self, force: bool = False) -> None:
        now = time.monotonic()
        if not force and now - self._last_emit < USAGE_EVENT_INTERVAL:
            return
        self._last_emit = now
        try:
            self._events.publish(ComputeChangedEvent(usage=self.usage()))
        except Exception:
            logger.exception("Could not publish compute usage")
