"""Compute records."""

from typing import Literal

from pydantic import Field

from mizuoto.core.record import Record


class DeviceInfo(Record):
    id: str
    """``cpu``, ``cuda:0``, ``mps``…"""
    kind: Literal["cpu", "cuda", "rocm", "xpu", "mps"]
    """``cuda`` (NVIDIA), ``rocm`` (AMD: torch names its GPUs ``cuda:N``), ``xpu`` (Intel), ``mps`` (Apple) or ``cpu``."""
    name: str
    memory_total_mb: float | None = None
    memory_used_mb: float | None = None


class ComputeInfo(Record):
    backend: str
    """``CUDA 13.2``, ``ROCm 6.4``, ``XPU``, ``MPS`` or ``CPU``."""
    driver: str | None = None
    cuda_version: str | None = None
    """The highest CUDA version the NVIDIA driver supports (nvidia-smi's header)."""
    devices: list[DeviceInfo]
    probed_with: str
    """nvidia-smi, rocm-smi, amd-smi, torch, platform (Apple silicon) or none."""


class HolderUsage(Record):
    name: str
    busy: str | None = None
    cached: list[str] = Field(default_factory=list)


class ComputeUsage(Record):
    devices: list[DeviceInfo]
    holders: list[HolderUsage]
    busy: list[str]
    can_release: bool
