"""Devices, memory and dtypes for native engines.

The equivalent of ComfyUI's ``comfy/model_management.py`` device half, as an object instead of
module globals.

- Device names are Mizuoto's: ``cpu``, ``cuda:N`` (NVIDIA, or AMD on ROCm), ``xpu:N``, ``mps``.
  Resolving ``auto`` is the core's job (``ComputeService.resolve_device``).
- Engines get a real ``torch.device``; nothing narrows ``CUDA_VISIBLE_DEVICES``.
- Storage dtype and compute dtype are separate: an engine names the dtypes a component supports,
  and ``pick_dtype`` chooses one for the device and the user's setting.
- A module that computes in float64 runs where float64 exists: ``float64_device`` is the CPU on MPS (and on XPU
  devices without fp64, such as Intel Arc), and ``pin_to_cpu`` keeps such a module there when its parent moves.
- ``probe_devices`` is what torch sees (``python -m mizuoto.runtime.device`` prints it as JSON for the core's
  device probe, which never imports torch itself).
"""

from __future__ import annotations

import gc
import os
import sys
import types
from typing import Any

DTYPE_NAMES = {"fp32": "float32", "fp16": "float16", "bf16": "bfloat16", "fp64": "float64"}


def device_kind(name: str) -> str:
    """``cpu``, ``cuda``, ``mps`` or ``xpu``: the part of a device name before the index."""
    return name.split(":", 1)[0]


def is_oom(error: BaseException) -> bool:
    """Whether ``error`` means a device ran out of memory (CUDA, ROCm, MPS, XPU). Loads no torch: one that is not
    loaded raised nothing."""
    import sys

    torch = sys.modules.get("torch")
    if torch is not None:
        oom = getattr(torch, "OutOfMemoryError", None) or getattr(getattr(torch, "cuda", None), "OutOfMemoryError", None)
        if oom is not None and isinstance(error, oom):
            return True
    text = str(error).lower()
    return "out of memory" in text and any(k in text for k in ("cuda", "hip", "mps", "xpu", "gpu", "device", "allocate"))


def torch_build() -> dict[str, Any]:
    """What the installed torch was built for, and which of its accelerators work here (``mizuoto doctor``)."""
    import torch
    import torch.version

    hip = getattr(torch.version, "hip", None)
    xpu = hasattr(torch, "xpu") and torch.xpu.is_available()
    mps = getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available()
    return {
        "version": torch.__version__,
        "cuda": None if hip else torch.version.cuda,
        "hip": hip,
        "cuda_available": bool(torch.cuda.is_available()),
        "device_count": int(torch.cuda.device_count()) if torch.cuda.is_available() else 0,
        "xpu_available": bool(xpu),
        "mps_available": bool(mps),
    }


def probe_devices(initialise: bool = True) -> dict[str, Any] | None:
    """The accelerators torch sees: ``{version, cuda, hip, xpu, devices: [{id, kind, name, total_mb, free_mb}]}``.

    ``kind`` is ``cuda`` (NVIDIA), ``rocm`` (AMD, whose devices torch names ``cuda:N``), ``xpu`` or ``mps``. Free memory
    is the driver's (other processes' use counted); MPS reports none (unified memory). ``initialise=False`` (this
    process, which may never use the GPU) returns None rather than initialise CUDA or XPU: that would leave a context
    of a few hundred MB on every GPU it asks about.
    """
    import torch
    import torch.version

    hip = getattr(torch.version, "hip", None)
    devices: list[dict[str, Any]] = []
    report: dict[str, Any] = {"version": torch.__version__, "cuda": None if hip else torch.version.cuda, "hip": hip, "xpu": getattr(torch.version, "xpu", None), "devices": devices}
    if torch.cuda.is_available():
        if not initialise and not torch.cuda.is_initialized():
            return None
        for i in range(torch.cuda.device_count()):
            free, total = torch.cuda.mem_get_info(i)
            devices.append({"id": f"cuda:{i}", "kind": "rocm" if hip else "cuda", "name": torch.cuda.get_device_name(i), "total_mb": total / 2**20, "free_mb": free / 2**20})
    xpu = getattr(torch, "xpu", None)
    if xpu is not None and xpu.is_available():
        if not initialise and not xpu.is_initialized():
            return None
        for i in range(xpu.device_count()):
            props = xpu.get_device_properties(i)
            total, free = float(props.total_memory), None
            try:
                free, total = (float(v) for v in xpu.mem_get_info(i))
            except (AttributeError, RuntimeError):
                pass  # an older torch: the total only
            devices.append({"id": f"xpu:{i}", "kind": "xpu", "name": props.name, "total_mb": total / 2**20, "free_mb": None if free is None else free / 2**20})
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        recommended = getattr(torch.mps, "recommended_max_memory", None)
        total = float(recommended()) / 2**20 if recommended is not None else None
        devices.append({"id": "mps", "kind": "mps", "name": "Apple GPU", "total_mb": total, "free_mb": None})
    return report


def main() -> None:
    """``python -m mizuoto.runtime.device``: ``probe_devices()`` as one line of JSON on stdout."""
    import json
    import sys

    sys.stdout.write(json.dumps(probe_devices()) + "\n")


def to_dtype(name: str) -> Any:
    """``fp32`` / ``fp16`` / ``bf16`` / ``fp64`` (or a torch dtype name) as a ``torch.dtype``."""
    import torch

    return getattr(torch, DTYPE_NAMES.get(name, name))


def dtype_name(dtype: Any) -> str:
    import torch

    return {torch.float32: "fp32", torch.float16: "fp16", torch.bfloat16: "bf16", torch.float64: "fp64"}.get(dtype, str(dtype).replace("torch.", ""))


class DeviceService:
    """What a device can do and how much memory it has.

    - ``simulate``: device name → bytes. The device (the CPU, in tests and the parity harness) plays an accelerator of
      that size: the model manager places and streams weights on it as on a GPU, counting what it placed.
    - ``strict``: with ``simulate``, streamed weights rest as meta tensors (code reading them outside their unit fails).
    - ``limit_bytes``: free memory on an accelerator is reported as at most this minus what this process holds (a
      smaller GPU played by a big one; ``MIZUOTO_VRAM_LIMIT_MB``).
    """

    _simulate: dict[str, int] = {}
    strict = False
    simulate_working = False
    """(Tests) Count a request's working memory against a simulated device's size too; by default a simulated device
    holds weights only (activations live in the CPU's memory)."""
    limit_bytes: int | None = None

    def __init__(self, simulate: dict[str, int] | None = None, strict: bool = False, limit_bytes: int | None = None) -> None:
        self._simulate = dict(simulate or {})
        self.strict = strict
        self.limit_bytes = limit_bytes

    def simulated(self, name: str) -> int | None:
        """The simulated size of ``name``, or None for a real device."""
        return self._simulate.get(name)

    def torch_device(self, name: str) -> Any:
        import torch

        return torch.device(name)

    # -- memory -----------------------------------------------------------------------

    def memory(self, name: str) -> tuple[int | None, int | None]:
        """(free, total) bytes on the device; ``None`` where it cannot be known. The CPU's needs no torch."""
        kind = device_kind(name)
        if kind == "cpu":
            return _system_memory()
        import torch

        if kind == "cuda" and torch.cuda.is_available():
            index = torch.device(name).index or 0
            free, total = torch.cuda.mem_get_info(index)
            # memory torch holds but does not use is free for our purposes (ComfyUI's get_free_memory)
            stats = torch.cuda.memory_stats(index)
            inactive = stats.get("reserved_bytes.all.current", 0) - stats.get("active_bytes.all.current", 0)
            free = int(free + max(0, inactive))
            if self.limit_bytes is not None:
                free = min(free, max(0, self.limit_bytes - int(stats.get("active_bytes.all.current", 0))))
                total = min(int(total), self.limit_bytes)
            return free, int(total)
        if kind == "xpu" and hasattr(torch, "xpu") and torch.xpu.is_available():
            index = torch.device(name).index or 0
            try:
                free, total = torch.xpu.mem_get_info(index)
                inactive = int(torch.xpu.memory_reserved(index)) - int(torch.xpu.memory_allocated(index))
                return int(free + max(0, inactive)), int(total)
            except (AttributeError, RuntimeError):  # an older torch: what this process has not reserved
                total = int(torch.xpu.get_device_properties(index).total_memory)
                return total - int(torch.xpu.memory_reserved(index)), total
        if kind == "mps" and getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            # unified memory: the system's free memory, capped by what Metal recommends for one process
            free, total = _system_memory()
            recommended = getattr(torch.mps, "recommended_max_memory", None)
            if recommended is not None:
                cap = int(recommended())
                total = cap if total is None else min(total, cap)
                free = None if free is None else min(free, max(0, cap - int(torch.mps.driver_allocated_memory())))
            return free, total
        return _system_memory()

    def allocated(self, name: str) -> int | None:
        """Bytes torch has allocated on the device now (None where it cannot tell: the CPU, MPS)."""
        kind = device_kind(name)
        if kind not in ("cuda", "xpu"):
            return None
        import torch

        backend = torch.cuda if kind == "cuda" else getattr(torch, "xpu", None)
        if backend is None or not backend.is_available():
            return None
        return int(backend.memory_allocated(torch.device(name)))

    def start_peak(self, name: str) -> None:
        """Start measuring the peak of what torch allocates on the device (``peak``). Allocator statistics only: no
        computation sees them."""
        kind = device_kind(name)
        if kind in ("cuda", "xpu"):
            import torch

            backend = torch.cuda if kind == "cuda" else getattr(torch, "xpu", None)
            if backend is not None and backend.is_available():
                backend.reset_peak_memory_stats(torch.device(name))

    def peak(self, name: str) -> int | None:
        """The most torch allocated on the device since ``start_peak`` (None where it cannot tell)."""
        kind = device_kind(name)
        if kind not in ("cuda", "xpu"):
            return None
        import torch

        backend = torch.cuda if kind == "cuda" else getattr(torch, "xpu", None)
        if backend is None or not backend.is_available():
            return None
        return int(backend.max_memory_allocated(torch.device(name)))

    def empty_cache(self, name: str | None = None) -> None:
        """Collect garbage and hand cached device memory back to the driver."""
        import torch

        gc.collect()
        kind = device_kind(name) if name else None
        if kind in (None, "cuda") and torch.cuda.is_available():
            torch.cuda.empty_cache()
        if kind in (None, "mps") and getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            torch.mps.empty_cache()
        if kind in (None, "xpu") and hasattr(torch, "xpu") and torch.xpu.is_available():
            torch.xpu.empty_cache()

    def synchronize(self, name: str) -> None:
        import torch

        kind = device_kind(name)
        if kind == "cuda":
            torch.cuda.synchronize(torch.device(name))
        elif kind == "mps":
            torch.mps.synchronize()
        elif kind == "xpu":
            torch.xpu.synchronize(torch.device(name))

    # -- dtypes -----------------------------------------------------------------------

    def supports(self, name: str, dtype: str) -> bool:
        """Whether ``dtype`` (fp32, fp16, bf16, fp64) can be stored and computed on the device."""
        import torch

        kind = device_kind(name)
        if dtype == "fp32":
            return True
        if kind == "cpu":
            return dtype in ("fp64", "bf16", "fp16")
        if kind == "mps":
            return dtype == "fp16" or (dtype == "bf16" and _mps_bf16())
        if kind == "cuda":
            if dtype == "fp64":
                return True
            index = torch.device(name).index or 0
            if dtype == "bf16":
                return bool(torch.cuda.is_bf16_supported())
            major, _ = torch.cuda.get_device_capability(index)
            return major >= 6 and not _fp16_blacklisted(torch.cuda.get_device_name(index))
        if kind == "xpu":
            if dtype == "fp64":  # Intel Arc has no fp64; the data-centre GPUs have it
                index = torch.device(name).index or 0
                return bool(getattr(torch.xpu.get_device_properties(index), "has_fp64", False))
            return dtype in ("fp16", "bf16")
        return False

    def float64_device(self, name: str) -> str:
        """Where a module that computes in float64 runs when its engine runs on ``name``: there, or the CPU when the
        device has no float64 (MPS, XPU without fp64)."""
        return float64_home(name, device_kind(name) == "cpu" or self.supports(name, "fp64"))

    def pick_dtype(self, name: str, requested: str = "auto", supported: tuple[str, ...] = ("bf16", "fp16", "fp32"), cpu: str = "fp32") -> str:
        """The dtype a component runs in.

        ``requested`` is the user's ``engines.<id>.dtype``. ``auto`` takes the first of ``supported``
        (the engine's preference order) the device handles well, and ``cpu`` on the CPU, where fp32
        is the fast choice for most models.
        """
        if requested != "auto":
            if requested not in supported:
                raise ValueError(f"{requested} is not supported by this model (it takes {', '.join(supported)})")
            if not self.supports(name, requested):
                raise ValueError(f"{name} cannot run {requested}")
            return requested
        if device_kind(name) == "cpu":
            return cpu if cpu in supported else supported[-1]
        for dtype in supported:
            if self.supports(name, dtype):
                return dtype
        return "fp32"


def float64_home(name: str, has_fp64: bool) -> str:
    """``name`` when the device has float64, else ``cpu`` (the decision ``DeviceService.float64_device`` makes)."""
    return name if has_fp64 or device_kind(name) == "cpu" else "cpu"


def pin_to_cpu(module: Any) -> Any:
    """Move ``module`` (an ``nn.Module``) to the CPU and keep it there: moving or casting a parent (``.to(device)``,
    ``.half()``, the model manager's loads and offloads) leaves it as it is. For a float64 part of a model on a device
    without float64; its caller moves inputs to it and results back (``module_device``)."""
    module.to("cpu")

    def _apply(self: Any, fn: Any, recurse: bool = True) -> Any:  # nn.Module._apply's signature; nothing moves
        return self

    # an instance attribute: nn.Module._apply calls ``child._apply(fn)`` on each child, and finds this one first
    module._apply = types.MethodType(_apply, module)
    return module


def module_device(module: Any) -> Any:
    """The device of ``module``'s first parameter or buffer (the CPU for a module without any)."""
    import itertools

    import torch

    for tensor in itertools.chain(module.parameters(), module.buffers()):
        return tensor.device
    return torch.device("cpu")


def _fp16_blacklisted(device_name: str) -> bool:
    """GTX 16xx cards report fp16 support but produce NaNs or run slowly (ComfyUI's table)."""
    return any(f"GTX {n}" in device_name for n in ("1650", "1660", "1630")) or "MX550" in device_name or "MX450" in device_name


def _mps_bf16() -> bool:
    import platform

    try:
        return int(platform.mac_ver()[0].split(".")[0]) >= 14
    except (ValueError, IndexError):
        return False


def _system_memory() -> tuple[int | None, int | None]:
    """(available, total) bytes of RAM: Linux's ``MemAvailable`` (page cache counts as available), within a cgroup's
    limit when there is one (ComfyUI's ``system_memory``); elsewhere what ``sysconf`` knows."""
    available, total = _meminfo()
    if available is None:
        if sys.platform == "win32":  # no sysconf
            return None, None
        try:
            page = os.sysconf("SC_PAGE_SIZE")
            available, total = int(os.sysconf("SC_AVPHYS_PAGES") * page), int(os.sysconf("SC_PHYS_PAGES") * page)
        except (ValueError, OSError, AttributeError):
            return None, None
    limit, used = _cgroup_memory()
    if limit is not None and total is not None and limit < total:
        total = limit
        if used is not None:
            available = min(available, max(0, limit - used))
    return available, total


def _meminfo() -> tuple[int | None, int | None]:
    try:
        with open("/proc/meminfo", encoding="ascii") as f:
            fields = {line.split(":")[0]: int(line.split()[1]) * 1024 for line in f if line.split()[1:2] and line.split()[1].isdigit()}
    except (OSError, ValueError, IndexError):
        return None, None
    if "MemAvailable" not in fields or "MemTotal" not in fields:
        return None, None
    return fields["MemAvailable"], fields["MemTotal"]


def _cgroup_memory() -> tuple[int | None, int | None]:
    """(limit, used) of this process's cgroup v2, without the reclaimable page cache; (None, None) without one."""
    root = "/sys/fs/cgroup"
    try:
        with open(f"{root}/memory.max", encoding="ascii") as f:
            raw = f.read().strip()
        if raw == "max":
            return None, None
        limit = int(raw)
        with open(f"{root}/memory.current", encoding="ascii") as f:
            used = int(f.read().strip())
        try:
            with open(f"{root}/memory.stat", encoding="ascii") as f:
                stat = {k: int(v) for k, v in (line.split() for line in f if len(line.split()) == 2)}
            used -= stat.get("inactive_file", 0)
        except (OSError, ValueError):
            pass
        return limit, max(0, used)
    except (OSError, ValueError):
        return None, None


if __name__ == "__main__":
    main()
