"""What an engine gets when it loads: the devices, the model manager and the runtime options."""

from __future__ import annotations

from dataclasses import dataclass, field

from mizuoto.runtime.device import DeviceService
from mizuoto.runtime.manager import ModelManager
from mizuoto.runtime.policy import VramPolicy


@dataclass
class RuntimeOptions:
    compile: bool = False
    """``compute.compile``: torch.compile where an engine offers it."""
    kv_cache: str = "dynamic"
    """``compute.kv_cache``: ``dynamic`` (grown by concatenation, transformers' way) or ``static`` (preallocated
    buffers, ``runtime.lm.cache.BufferCache``), for the language models that use ``runtime.lm.cache``."""
    cuda_graphs: bool = True
    """``compute.cuda_graphs``: decoding steps as CUDA graphs on NVIDIA GPUs (``runtime.graphs``; identical results)."""


@dataclass
class Runtime:
    devices: DeviceService
    manager: ModelManager
    options: RuntimeOptions = field(default_factory=RuntimeOptions)
    policy: VramPolicy = field(default_factory=VramPolicy)
    """Read by the manager at each load (the core refreshes it from the settings)."""
