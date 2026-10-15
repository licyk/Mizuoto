"""How the runtime spends device and host memory.

``VramPolicy`` is built from the settings at each use (``InferenceService.runtime``), so a change reaches the next
load without a restart. Nothing here is a module global: the manager asks its runtime for the policy it has now.

- ``mode``:
  - ``high``: models are loaded whole or not at all, as before partial loading (other models are offloaded whole);
  - ``normal``: whole when it fits, else partly (the rest of the weights stream from RAM as layers need them);
  - ``low``: weights take at most ``low_ratio`` of the device's memory (its total, within ``vram_limit_mb``), leaving room for other models;
  - ``none``: everything but the always-resident parts streams (ComfyUI's ``--novram``);
  - ``shared``: unified memory (Apple silicon): offloading frees nothing, so evicting means unloading, and nothing is
    loaded partly. ``auto`` is ``shared`` on MPS and ``normal`` elsewhere.
- ``reserve_bytes``: device memory left free beyond a request's working memory (``compute.vram_reserve_mb``).
- ``ram_reserve_bytes``: host memory left free; below it host copies are dropped first and then idle models unloaded.
- ``host_copies``: keep a host copy of the weights of a model on a device, so offloading it costs no transfer.
- ``max_pinned_bytes``: page-locked host memory for weights that stream (0: the automatic budget).
- ``streams``: side streams for copies that overlap compute (-1: automatic, 2 on CUDA and ROCm, 0 elsewhere).
- ``oom_retry``: retry a phase that ran out of memory with more of its weights streamed (the same draws replayed).
- ``oom_fallback``: allow an engine's cheaper algorithm for a phase (tiling, split batches) after that; changes numbers.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, replace

MODES = ("auto", "high", "normal", "low", "none", "shared")

MB = 2**20
GB = 2**30


@dataclass(frozen=True)
class VramPolicy:
    mode: str = "auto"
    reserve_bytes: int = 1024 * MB
    ram_reserve_bytes: int | None = None
    host_copies: str = "auto"
    max_pinned_bytes: int = 0
    streams: int = -1
    oom_retry: bool = True
    oom_fallback: bool = False
    low_ratio: float = 0.5

    def mode_for(self, kind: str, override: str | None = None) -> str:
        """The effective mode on a device of ``kind`` (``cuda``, ``xpu``, ``mps``, ``cpu``), with an engine's override."""
        mode = override if override and override != "auto" else self.mode
        if kind == "mps":
            return "shared"
        if mode == "auto":
            return "normal"
        return "normal" if mode == "shared" else mode

    def streams_for(self, kind: str) -> int:
        """Side streams for copies on a device of ``kind``: non-blocking copies only on CUDA and ROCm (ComfyUI's choice)."""
        if kind != "cuda":
            return 0
        return 2 if self.streams < 0 else self.streams

    def ram_reserve(self, total: int | None) -> int:
        """Host memory to keep free: the setting, else ``max(2 GB, 10 % of RAM)``."""
        if self.ram_reserve_bytes is not None:
            return self.ram_reserve_bytes
        return max(2 * GB, int((total or 0) * 0.1))

    def pinned_budget(self, ram_total: int | None) -> int:
        """Page-locked bytes allowed (ComfyUI's ``MAX_PINNED_MEMORY``): Windows 40 % of RAM; elsewhere
        ``max(0.4·RAM, min(0.9·RAM, RAM − 4 GB))``."""
        if self.max_pinned_bytes > 0:
            return self.max_pinned_bytes
        if not ram_total:
            return 0
        if os.name == "nt":
            return int(ram_total * 0.4)
        return int(max(ram_total * 0.4, min(ram_total * 0.9, ram_total - 4 * GB)))

    def with_(self, **changes: object) -> VramPolicy:
        return replace(self, **changes)  # type: ignore[arg-type]
