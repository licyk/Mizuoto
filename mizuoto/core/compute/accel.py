"""The optional attention packages: which are installed, read from their metadata
without importing them (or torch), and the command that installs a build matching this torch without replacing it.

None of them is a dependency: each ties itself to a torch version, a CUDA or ROCm runtime or a GPU generation, and some
would pull in a torch of their own. Whether one really runs on a device (kernels present, the GPU's generation, a
self-test) is the runtime's ``mizuoto.runtime.accel`` probe, reported per device when a GPU is there.
"""

from __future__ import annotations

import importlib.metadata
import importlib.util
import sys
from dataclasses import dataclass

XFORMERS_FOR_TORCH = {"2.8.0": "0.0.32.post2", "2.9.0": "0.0.33.post1", "2.9.1": "0.0.33.post2", "2.10.0": "0.0.34"}
"""The xFormers release built for each torch release (each pins it exactly); from 0.0.35 one build covers torch ≥ 2.10
(PyPI, checked October 2026)."""
XFORMERS_STABLE = ("2.10", "0.0.35")
"""From this torch on, this xFormers (built against torch's stable ABI) and later ones."""


@dataclass(frozen=True)
class AccelPackage:
    backend: str
    """The ``compute.attention`` value that uses it."""
    title: str
    dists: tuple[str, ...]
    """Distribution names, any of which provides it."""
    module: str
    hardware: str
    """Where it runs (for the doctor)."""


PACKAGES = (
    AccelPackage("flash", "FlashAttention", ("flash-attn", "flash_attn_3"), "flash_attn", "NVIDIA GPUs from Ampere on, AMD Instinct; fp16/bf16"),
    AccelPackage("xformers", "xFormers", ("xformers",), "xformers", "NVIDIA GPUs from Turing on, ROCm builds"),
    AccelPackage("sage", "SageAttention", ("sageattention",), "sageattention", "NVIDIA GPUs from Ampere on; inference only"),
    AccelPackage("sage3", "SageAttention 3", ("sageattn3",), "sageattn3", "NVIDIA Blackwell GPUs; inference only"),
)


def installed_version(package: AccelPackage) -> str | None:
    """The installed version (from the distribution's metadata), "installed" when the module is there without
    metadata, or None."""
    for dist in package.dists:
        try:
            return importlib.metadata.version(dist)
        except importlib.metadata.PackageNotFoundError:
            continue
    try:
        found = importlib.util.find_spec(package.module) is not None
    except (ImportError, ValueError):
        found = False
    return "installed" if found else None


def _python(python: str | None) -> str:
    exe = python or sys.executable
    return f'"{exe}"' if " " in exe else exe


def _release(version: str) -> tuple[int, ...]:
    out = []
    for part in version.split("+")[0].split(".")[:3]:
        digits = "".join(c for c in part if c.isdigit())
        out.append(int(digits or 0))
    return tuple(out)


def install_command(backend: str, torch_build: dict[str, str | None] | None, python: str | None = None) -> str | None:
    """How to install ``backend``'s package for this torch (``torch_build``: ``version``, ``cuda``, ``hip``), with
    ``--no-deps`` so torch stays as it is; None when this torch cannot use it (a CPU, XPU or MPS build)."""
    exe = _python(python)
    build = torch_build or {}
    version = (build.get("version") or "").split("+")[0]
    gpu = bool(build.get("cuda") or build.get("hip"))
    if not gpu:
        return None
    if backend == "xformers":
        pin = XFORMERS_FOR_TORCH.get(version)
        if pin is None and _release(version) >= _release(XFORMERS_STABLE[0]):
            pin = f">={XFORMERS_STABLE[1]}"
        elif pin is not None:
            pin = f"=={pin}"
        if pin is None:
            return None
        local = (build.get("version") or "").partition("+")[2]
        index = f" --index-url https://download.pytorch.org/whl/{local}" if local and local != "cu128" else ""
        return f'{exe} -m pip install "xformers{pin}" --no-deps{index}'
    if build.get("hip") and backend != "flash":
        return None
    if backend == "flash":
        return f"{exe} -m pip install flash-attn --no-build-isolation --no-deps  # compiles for a long time (needs nvcc), unless a wheel for this torch exists"
    if backend == "sage":
        return f"{exe} -m pip install sageattention==2.2.0 --no-build-isolation --no-deps  # needs nvcc; or sageattention==1.0.6 (Triton) without a compiler"
    if backend == "sage3":
        return "build it from SageAttention's sageattention3_blackwell folder (Blackwell GPUs)"
    return None
