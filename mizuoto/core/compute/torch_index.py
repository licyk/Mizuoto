"""Which torch build suits this machine, and the command that installs it (``mizuoto doctor``).

torch is a dependency of Mizuoto, but its build (CUDA, ROCm, XPU or CPU) is the user's choice, as in ComfyUI's
README. This module knows the builds PyTorch publishes for the torch Mizuoto needs and
picks the one that runs on the machine's driver. Standard library only.
"""

from __future__ import annotations

import re
import sys

from mizuoto.core.compute.models import ComputeInfo

TORCH_REQUIREMENT = "torch>=2.8"
PYTORCH_INDEX = "https://download.pytorch.org/whl/{flavour}"
CUDA_FLAVOURS = ("cu126", "cu128", "cu129", "cu130", "cu132")
"""CUDA builds of torch 2.8 or newer on download.pytorch.org, oldest first (October 2026: cu126 and cu130 carry every
release from 2.8 or 2.9 to 2.14; cu128 stops at 2.11, cu129 at 2.13; cu132 starts at 2.12). The newest one the driver
runs is chosen, and pip takes the newest torch that index has."""
ROCM_FLAVOURS = ("rocm6.4", "rocm7.1", "rocm7.2")
"""ROCm builds, oldest first (rocm6.4: 2.8–2.9; rocm7.1: 2.10–2.13; rocm7.2: 2.11–2.14, Linux only). The wheels carry
their own ROCm libraries, so the newest is chosen."""


def cuda_tuple(flavour: str) -> tuple[int, int] | None:
    """``cu128`` → (12, 8); anything else → None."""
    m = re.fullmatch(r"cu(\d+)(\d)", flavour)
    return (int(m.group(1)), int(m.group(2))) if m else None


def driver_tuple(version: str | None) -> tuple[int, int] | None:
    m = re.match(r"(\d+)\.(\d+)", version or "")
    return (int(m.group(1)), int(m.group(2))) if m else None


def choose_flavour(compute: ComputeInfo, cuda: tuple[str, ...] = CUDA_FLAVOURS, rocm: tuple[str, ...] = ROCM_FLAVOURS) -> str:
    """The build for this machine: the newest CUDA build the driver runs, else ROCm, else Apple's (``mps``: PyPI's
    default build), else the CPU build."""
    if any(d.kind == "cuda" for d in compute.devices) and cuda:
        driver = driver_tuple(compute.cuda_version)
        usable = [f for f in cuda if driver is None or (cuda_tuple(f) or (99, 0)) <= driver]
        if usable:
            return usable[-1]
    if any(d.kind == "rocm" for d in compute.devices) and rocm:
        return rocm[-1]
    if any(d.kind == "xpu" for d in compute.devices):
        return "xpu"
    if any(d.kind == "mps" for d in compute.devices):
        return "mps"
    return "cpu"


def index_url(flavour: str, template: str = "auto") -> str:
    return (PYTORCH_INDEX if template in ("", "auto") else template).replace("{flavour}", flavour)


def install_command(flavour: str, python: str | None = None) -> str:
    """The command that installs the ``flavour`` build of torch into this interpreter (macOS takes PyPI's own)."""
    exe = python or sys.executable
    exe = f'"{exe}"' if " " in exe else exe
    base = f'{exe} -m pip install --upgrade --force-reinstall "{TORCH_REQUIREMENT}"'
    if flavour == "mps":
        return base
    return f"{base} --index-url {index_url(flavour)}"
