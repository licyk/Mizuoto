"""Finding the machine's accelerators without importing torch into the host.

In order, the first that finds a GPU wins:

1. ``nvidia-smi`` (NVIDIA): names, memory and the highest CUDA version the driver runs.
2. ``rocm-smi``, else ``amd-smi`` (AMD). torch's ROCm build exposes these GPUs as CUDA devices, so they are named
   ``cuda:N`` (kind ``rocm``), numbered in the tool's order, which is HIP's for discrete GPUs.
3. torch, when the runtime has already imported it in this process (``torch_probe``, given by the core's inference
   layer): once torch is loaded its own numbers are used, CUDA (NVIDIA, and AMD by HIP), XPU, MPS, and a
   torch that sees no GPU means the CPU.
4. Apple silicon: MPS, from the platform (the chip's name, the unified memory as its total).
5. torch in a child process (``python -m mizuoto.runtime.device``). It costs one torch import (1–2 s, more on a cold
   disk), so it runs only when the installed torch's build can drive a GPU the tools above did not see — an XPU build
   (Intel GPUs have no such tool), a ROCm build without rocm-smi where ``/dev/kfd`` exists, a CUDA build where the
   NVIDIA driver's device files exist but nvidia-smi is not on the path (some containers) — which is read from
   ``torch/version.py`` without importing torch. A CPU build, or the CUDA build PyPI installs on a machine without an
   NVIDIA driver, never pays it.
6. The CPU alone.
"""

from __future__ import annotations

import importlib.util
import json
import logging
import platform
import re
import shutil
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

from mizuoto.core.compute.models import ComputeInfo, DeviceInfo

logger = logging.getLogger(__name__)

TIMEOUT = 10.0
TORCH_TIMEOUT = 90.0
"""The child process that imports torch: a cold import on Windows can take tens of seconds."""
MIB = 2**20

TorchProbe = Callable[[], dict[str, Any] | None]
"""Returns ``mizuoto.runtime.device.probe_devices()``'s report, or None when torch is not loaded in this process."""


def _run(cmd: list[str], timeout: float = TIMEOUT, env: dict[str, str] | None = None) -> str | None:
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, check=True, env=env).stdout
    except (OSError, subprocess.SubprocessError):
        return None


def _float(text: Any) -> float | None:
    try:
        return float(text)
    except (TypeError, ValueError):
        return None


def cpu_device() -> DeviceInfo:
    return DeviceInfo(id="cpu", kind="cpu", name=platform.processor() or platform.machine() or "CPU")


# -- NVIDIA ---------------------------------------------------------------------------------------


def find_nvidia_smi() -> str | None:
    """nvidia-smi on the path, or where WSL keeps it (``/usr/lib/wsl/lib``, not always on the path)."""
    found = shutil.which("nvidia-smi")
    if found:
        return found
    wsl = Path("/usr/lib/wsl/lib/nvidia-smi")
    return str(wsl) if sys.platform.startswith("linux") and wsl.is_file() else None


def visible_gpus(rows: list[tuple[str, str]], value: str | None) -> list[int]:
    """Which of ``rows`` (``nvidia-smi``'s index and UUID per GPU) torch will see, in its order, under
    ``CUDA_VISIBLE_DEVICES=value``: every one when it is unset; else its entries in turn, each an index or a ``GPU-``
    UUID prefix, up to the first that names no GPU (as CUDA reads it; empty: none). Indices are ``nvidia-smi``'s, in
    PCI bus order, which is CUDA's on machines whose GPUs are alike (``CUDA_DEVICE_ORDER=PCI_BUS_ID`` makes it so)."""
    if value is None:
        return list(range(len(rows)))
    out: list[int] = []
    for entry in (e.strip() for e in value.split(",")):
        found = next((i for i, (index, uuid) in enumerate(rows) if (entry.isdigit() and entry == index) or (entry.startswith("GPU-") and uuid.startswith(entry))), None)
        if found is None or found in out:
            break
        out.append(found)
    return out


def probe_nvidia(nvidia_smi: str) -> ComputeInfo | None:
    """The NVIDIA GPUs torch will see (``CUDA_VISIBLE_DEVICES`` applied, numbered as torch numbers them)."""
    import os

    out = _run([nvidia_smi, "--query-gpu=index,uuid,name,memory.total,memory.used,driver_version", "--format=csv,noheader,nounits"])
    if not out:
        return None
    rows: list[tuple[str, str]] = []
    found: list[tuple[str, float | None, float | None]] = []
    driver = None
    for line in out.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 6:
            continue
        index, uuid, name, total, used, driver = parts[:6]
        rows.append((index, uuid))
        found.append((name, _float(total), _float(used)))
    devices = [
        DeviceInfo(id=f"cuda:{n}", kind="cuda", name=found[i][0], memory_total_mb=found[i][1], memory_used_mb=found[i][2])
        for n, i in enumerate(visible_gpus(rows, os.environ.get("CUDA_VISIBLE_DEVICES")))
    ]
    if not devices:
        return None
    header = _run([nvidia_smi]) or ""
    match = re.search(r"CUDA Version:\s*([0-9.]+)", header)
    backend = f"CUDA {match.group(1)}" if match else "CUDA"
    return ComputeInfo(backend=backend, driver=driver, cuda_version=match.group(1) if match else None, devices=[*devices, cpu_device()], probed_with="nvidia-smi")


# -- AMD ------------------------------------------------------------------------------------------


def _json_in(text: str | None) -> Any:
    """The JSON document in a tool's output (rocm-smi may print warnings before it); None when there is none."""
    if not text:
        return None
    starts = [i for i in (text.find("{"), text.find("[")) if i >= 0]
    if not starts:
        return None
    try:
        return json.JSONDecoder().raw_decode(text[min(starts) :])[0]
    except ValueError:
        return None


def rocm_version() -> str | None:
    """The installed ROCm's version (``6.2``) from ``<rocm>/.info/version``; None when it cannot be read."""
    import os

    for root in (os.environ.get("ROCM_PATH"), "/opt/rocm"):
        if not root:
            continue
        try:
            text = (Path(root) / ".info" / "version").read_text(encoding="utf-8").strip()
        except OSError:
            continue
        match = re.match(r"(\d+\.\d+)", text)
        if match:
            return match.group(1)
    return None


def _first(row: dict[str, Any], *keys: str) -> Any:
    lowered = {k.lower(): v for k, v in row.items()}
    for key in keys:
        value = lowered.get(key.lower())
        if value not in (None, "", "N/A"):
            return value
    return None


def parse_rocm_smi(data: Any) -> tuple[list[DeviceInfo], str | None]:
    """``rocm-smi --showproductname --showmeminfo vram --showdriverversion --json``: (devices, driver)."""
    if not isinstance(data, dict):
        return [], None
    cards = sorted((k for k in data if re.fullmatch(r"card\d+", k) and isinstance(data[k], dict)), key=lambda k: int(k[4:]))
    devices: list[DeviceInfo] = []
    for i, key in enumerate(cards):
        row = data[key]
        name = _first(row, "Card Series", "Card series", "Device Name", "Card SKU", "Card model") or "AMD GPU"
        total = _float(_first(row, "VRAM Total Memory (B)"))
        used = _float(_first(row, "VRAM Total Used Memory (B)"))
        devices.append(
            DeviceInfo(id=f"cuda:{i}", kind="rocm", name=str(name), memory_total_mb=None if total is None else total / MIB, memory_used_mb=None if used is None else used / MIB)
        )
    system = data.get("system")
    driver = _first(system, "Driver version") if isinstance(system, dict) else None
    return devices, None if driver is None else str(driver)


def _mib(value: Any) -> float | None:
    """A memory size from amd-smi: ``{"value": 16368, "unit": "MB"}``, ``"16368 MB"`` or a bare number (MB)."""
    unit = "MB"
    if isinstance(value, dict):
        value, unit = value.get("value"), str(value.get("unit") or "MB")
    elif isinstance(value, str):
        match = re.match(r"\s*([0-9.]+)\s*([A-Za-z]*)", value)
        if not match:
            return None
        value, unit = match.group(1), match.group(2) or "MB"
    number = _float(value)
    if number is None:
        return None
    scale = {"B": 1 / MIB, "KB": 1 / 1024, "KIB": 1 / 1024, "MB": 1.0, "MIB": 1.0, "GB": 1024.0, "GIB": 1024.0}.get(unit.upper(), 1.0)
    return number * scale


def _gpu_rows(data: Any) -> list[dict[str, Any]]:
    """amd-smi's per-GPU rows: a list, or ``{"gpu_data": [...]}`` in newer releases."""
    if isinstance(data, dict):
        data = data.get("gpu_data", [data])
    return [row for row in data if isinstance(row, dict)] if isinstance(data, list) else []


def parse_amd_smi(static: Any, metric: Any) -> list[DeviceInfo]:
    """``amd-smi static --asic --vram --json`` and ``amd-smi metric --mem-usage --json``: the devices."""
    used_by_gpu: dict[int, tuple[float | None, float | None]] = {}
    for row in _gpu_rows(metric):
        usage = row.get("mem_usage") or {}
        if isinstance(usage, dict):
            used_by_gpu[int(row.get("gpu", len(used_by_gpu)))] = (_mib(usage.get("total_vram")), _mib(usage.get("used_vram")))
    devices: list[DeviceInfo] = []
    for i, row in enumerate(_gpu_rows(static)):
        index = int(row.get("gpu", i))
        asic = row.get("asic") if isinstance(row.get("asic"), dict) else {}
        vram = row.get("vram") if isinstance(row.get("vram"), dict) else {}
        name = _first(asic or {}, "market_name", "asic_name") or "AMD GPU"
        total, used = used_by_gpu.get(index, (None, None))
        total = total if total is not None else _mib((vram or {}).get("size"))
        devices.append(DeviceInfo(id=f"cuda:{i}", kind="rocm", name=str(name), memory_total_mb=total, memory_used_mb=used))
    return devices


def probe_rocm(rocm_smi: str | None, amd_smi: str | None) -> ComputeInfo | None:
    """AMD GPUs through rocm-smi, else amd-smi."""
    devices: list[DeviceInfo] = []
    driver: str | None = None
    tool = None
    if rocm_smi:
        devices, driver = parse_rocm_smi(_json_in(_run([rocm_smi, "--showproductname", "--showmeminfo", "vram", "--showdriverversion", "--json"])))
        if not devices:  # a release that refuses one of the flags: the essentials alone
            devices, driver = parse_rocm_smi(_json_in(_run([rocm_smi, "--showproductname", "--showmeminfo", "vram", "--json"])))
        tool = "rocm-smi"
    if not devices and amd_smi:
        static = _json_in(_run([amd_smi, "static", "--asic", "--vram", "--json"]))
        metric = _json_in(_run([amd_smi, "metric", "--mem-usage", "--json"]))
        devices, tool = parse_amd_smi(static, metric), "amd-smi"
    if not devices or tool is None:
        return None
    version = rocm_version()
    return ComputeInfo(backend=f"ROCm {version}" if version else "ROCm", driver=driver, devices=[*devices, cpu_device()], probed_with=tool)


# -- torch ----------------------------------------------------------------------------------------


def installed_torch_build() -> dict[str, str | None] | None:
    """What the installed torch was built for (``version``, ``cuda``, ``hip``, ``xpu``), read from its
    ``version.py`` without importing it; None when torch is not installed or the file cannot be read."""
    try:
        spec = importlib.util.find_spec("torch")
    except (ImportError, ValueError):
        return None
    if spec is None or not spec.origin:
        return None
    try:
        text = (Path(spec.origin).parent / "version.py").read_text(encoding="utf-8")
    except OSError:
        return None
    out: dict[str, str | None] = {}
    for key, name in (("version", "__version__"), ("cuda", "cuda"), ("hip", "hip"), ("xpu", "xpu")):
        match = re.search(rf"^{re.escape(name)}\s*(?::[^=]*)?=\s*(.+)$", text, re.MULTILINE)
        value = match.group(1).strip() if match else "None"
        out[key] = None if value == "None" else value.strip("'\"")
    return out


def torch_may_find_gpu(build: dict[str, str | None] | None, system: str | None = None) -> bool:
    """Whether asking torch could find a GPU the command-line tools did not: the build can drive one and, where
    that can be told cheaply, its driver is present."""
    if not build:
        return False
    system = system or sys.platform
    version = build.get("version") or ""
    linux = system.startswith("linux")
    if build.get("xpu") or "+xpu" in version:
        return True
    if build.get("hip") or "+rocm" in version:
        return not linux or Path("/dev/kfd").exists()
    if build.get("cuda"):
        # Windows' and WSL's nvidia-smi come with the driver (found above); in a Linux container the driver's
        # device files can be there without the tool.
        return linux and (Path("/proc/driver/nvidia/version").exists() or Path("/dev/nvidia0").exists() or Path("/dev/dxg").exists())
    return False


def info_from_torch(report: dict[str, Any] | None, source: str = "torch") -> ComputeInfo | None:
    """``mizuoto.runtime.device.probe_devices()``'s report as a ``ComputeInfo``; None when it lists no accelerator."""
    if not report:
        return None
    devices: list[DeviceInfo] = []
    for d in report.get("devices") or []:
        total, free = _float(d.get("total_mb")), _float(d.get("free_mb"))
        used = None if total is None or free is None else max(0.0, total - free)
        devices.append(DeviceInfo(id=str(d["id"]), kind=d["kind"], name=str(d.get("name") or d["id"]), memory_total_mb=total, memory_used_mb=used))
    if not devices:
        return None
    kinds = {d.kind for d in devices}
    cuda, hip = report.get("cuda"), report.get("hip")
    if "cuda" in kinds:
        backend = f"CUDA {cuda}" if cuda else "CUDA"
    elif "rocm" in kinds:
        backend = "ROCm " + ".".join(str(hip).split(".")[:2]) if hip else "ROCm"
    elif "xpu" in kinds:
        backend = "XPU"
    else:
        backend = "MPS"
    # torch's CUDA version is the toolkit it was built with, which the driver evidently runs
    return ComputeInfo(backend=backend, cuda_version=str(cuda) if cuda and "cuda" in kinds else None, devices=[*devices, cpu_device()], probed_with=source)


def probe_torch_subprocess(python: str | None = None) -> dict[str, Any] | None:
    """``probe_devices()`` in a child process, so this one never imports torch; None when it fails. The child finds
    this Mizuoto first on its path, wherever this one was imported from."""
    import os

    here = str(Path(__file__).resolve().parents[3])
    env = {**os.environ, "PYTHONPATH": os.pathsep.join(p for p in (here, os.environ.get("PYTHONPATH", "")) if p)}
    out = _run([python or sys.executable, "-m", "mizuoto.runtime.device"], timeout=TORCH_TIMEOUT, env=env)
    data = _json_in(out)
    if not isinstance(data, dict):
        logger.debug("The torch device probe gave nothing usable: %r", out)
        return None
    return data


def probe_accel_subprocess(devices: list[str], python: str | None = None) -> dict[str, dict[str, str | None]] | None:
    """Whether each attention backend runs on each of ``devices`` (``python -m mizuoto.runtime.accel``), in a child
    process; None when it fails."""
    import os

    here = str(Path(__file__).resolve().parents[3])
    env = {**os.environ, "PYTHONPATH": os.pathsep.join(p for p in (here, os.environ.get("PYTHONPATH", "")) if p)}
    out = _run([python or sys.executable, "-m", "mizuoto.runtime.accel", *devices], timeout=TORCH_TIMEOUT, env=env)
    data = _json_in(out)
    return data if isinstance(data, dict) else None


# -- Apple ----------------------------------------------------------------------------------------


def probe_apple() -> ComputeInfo:
    """Apple silicon's GPU (MPS): the chip's name and the unified memory, from sysctl."""
    name = (_run(["sysctl", "-n", "machdep.cpu.brand_string"]) or "").strip() or "Apple GPU"
    memory = _float((_run(["sysctl", "-n", "hw.memsize"]) or "").strip())
    mps = DeviceInfo(id="mps", kind="mps", name=name, memory_total_mb=None if memory is None else memory / MIB)
    return ComputeInfo(backend="MPS", devices=[mps, cpu_device()], probed_with="platform")


def is_apple_silicon() -> bool:
    return sys.platform == "darwin" and platform.machine() == "arm64"


# -------------------------------------------------------------------------------------------------


def probe(torch_probe: TorchProbe | None = None) -> ComputeInfo:
    """The machine's devices (see the module's docstring for the order)."""
    smi = find_nvidia_smi()
    if smi:
        info = probe_nvidia(smi)
        if info is not None:
            return info
    rocm_smi, amd_smi = shutil.which("rocm-smi"), shutil.which("amd-smi")
    if rocm_smi or amd_smi:
        info = probe_rocm(rocm_smi, amd_smi)
        if info is not None:
            return info
    report = None
    if torch_probe is not None:
        try:
            report = torch_probe()
        except Exception:
            logger.exception("Probing devices through torch failed")
    if report is not None:  # torch is loaded here: what it sees is what engines get
        return info_from_torch(report) or ComputeInfo(backend="CPU", devices=[cpu_device()], probed_with="torch")
    if is_apple_silicon():
        return probe_apple()
    if torch_may_find_gpu(installed_torch_build()):
        info = info_from_torch(probe_torch_subprocess())
        if info is not None:
            return info
    return ComputeInfo(backend="CPU", devices=[cpu_device()], probed_with="none")
