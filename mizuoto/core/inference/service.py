"""The native runtime for this process: one device service, one model manager, every engine in-process.

Built by ``build_services``. The runtime is created on first use, from the ``compute`` settings, so a
command that never loads a model never imports anything of it. This is the core's only door to
``mizuoto.runtime`` and ``mizuoto.engines``: every import of them is inside a function.
"""

from __future__ import annotations

import functools
import importlib.util
import logging
import re
import threading
from typing import TYPE_CHECKING, Any

from mizuoto.core.settings import SettingsService

if TYPE_CHECKING:
    from mizuoto.runtime.runtime import Runtime


logger = logging.getLogger(__name__)

DISTRIBUTIONS: dict[str, str] = {"fishspeech": "mizuoto-fishspeech", "indextts": "mizuoto-indextts"}
"""Engines implemented in distributions of their own (their licences cover the ports), by engine id."""
TORCH_FREE = frozenset({"dummy"})
"""Engines that run without torch (the dummy engine: numpy only)."""

_REQUIREMENT = re.compile(r"^\s*([A-Za-z0-9._-]+)\s*(?:\[[^\]]*\])?\s*\(?([^;)]*)\)?\s*(?:;.*)?$")
_SPECIFIER = re.compile(r"^\s*(~=|==|!=|<=|>=|<|>)\s*([0-9][0-9.]*)")


def distribution_version(engine_id: str) -> str | None:
    """The installed version of the distribution a licence-gated engine comes in (None: built in, or not installed).
    The gated distributions are released on their own, so their version is not Mizuoto's."""
    from importlib import metadata

    name = DISTRIBUTIONS.get(engine_id)
    if name is None:
        return None
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def mizuoto_range(engine_id: str) -> str | None:
    """The Mizuoto versions an installed gated distribution says it works with (its ``mizuoto`` requirement:
    ``<0.3,>=0.2.0``); None when it is not installed or names no range."""
    from importlib import metadata

    name = DISTRIBUTIONS.get(engine_id)
    if name is None:
        return None
    try:
        requires = metadata.requires(name) or []
    except metadata.PackageNotFoundError:
        return None
    for requirement in requires:
        found = _REQUIREMENT.match(requirement)
        if found and found.group(1).lower() == "mizuoto" and found.group(2).strip():
            return found.group(2).strip()
    return None


def _release(version: str) -> tuple[int, ...]:
    """A version's release numbers (``0.2.0rc1`` → ``(0, 2, 0)``), without trailing zeros so ``0.3`` equals ``0.3.0``."""
    found = re.match(r"\d+(?:\.\d+)*", version.strip())
    numbers = [int(n) for n in found.group(0).split(".")] if found else [0]
    while len(numbers) > 1 and numbers[-1] == 0:
        numbers.pop()
    return tuple(numbers)


def satisfies(version: str, spec: str) -> bool:
    """Whether ``version`` is in ``spec`` (``>=0.2.0,<0.3``), by release numbers (the ranges Mizuoto's distributions
    declare; a clause this does not read is taken as met)."""
    have = _release(version)
    for clause in spec.split(","):
        found = _SPECIFIER.match(clause)
        if not found:
            continue
        op, bound = found.group(1), _release(found.group(2))
        if op == "~=":
            prefix = bound[:-1] if len(bound) > 1 else bound
            ok = have >= bound and have[: len(prefix)] == prefix
        else:
            ok = {"==": have == bound, "!=": have != bound, "<=": have <= bound, ">=": have >= bound, "<": have < bound, ">": have > bound}[op]
        if not ok:
            return False
    return True


@functools.cache
def torch_installed() -> bool:
    """Whether torch can be imported here (looked up once: installing it means restarting Mizuoto)."""
    return importlib.util.find_spec("torch") is not None


def torch_version() -> str | None:
    """The installed torch's version, from its metadata (torch is not imported); None when it is missing."""
    from importlib import metadata

    try:
        return metadata.version("torch")
    except metadata.PackageNotFoundError:
        return None


ALLOCATORS = {"async": "backend:cudaMallocAsync", "expandable": "expandable_segments:True"}
"""``compute.cuda_allocator`` → ``PYTORCH_CUDA_ALLOC_CONF``."""


def set_allocator(choice: str) -> bool:
    """Set torch's CUDA allocator for this process (and the training stages it starts) before CUDA is first used: the
    host's decision, made once, never by engine code. Leaves an allocator the environment names alone. Returns whether
    it set one."""
    import os
    import sys

    conf = ALLOCATORS.get(choice)
    if conf is None or "PYTORCH_CUDA_ALLOC_CONF" in os.environ:
        return False
    torch = sys.modules.get("torch")
    if torch is not None and torch.cuda.is_initialized():
        logger.warning("compute.cuda_allocator = %s applies at the next start: CUDA is already in use", choice)
        return False
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = conf
    return True


def policy_from(c: Any) -> Any:
    """The runtime's ``VramPolicy`` from the ``compute`` settings."""
    from mizuoto.runtime.policy import VramPolicy

    return VramPolicy(
        mode=c.vram_mode,
        reserve_bytes=c.vram_reserve_mb * 2**20,
        ram_reserve_bytes=None if c.ram_reserve_mb is None else c.ram_reserve_mb * 2**20,
        host_copies=c.host_copies,
        max_pinned_bytes=c.max_pinned_mb * 2**20,
        streams=c.async_offload,
        oom_retry=c.oom_retry,
        oom_fallback=c.oom_fallback,
    )


class InferenceService:
    def __init__(self, settings: SettingsService) -> None:
        self._settings = settings
        self.fast: list[str] = []
        """``compute.fast`` as applied when the runtime was built (process-wide; outputs list it)."""
        self._runtime: Runtime | None = None
        self._lock = threading.Lock()

    @property
    def runtime(self) -> Runtime:
        with self._lock:
            if self._runtime is None:
                from mizuoto.core.compute import cpu

                c = self._settings.settings.compute
                cpu.apply(c.cpu_allocator, c.cpu_threads)  # before torch is imported: OpenMP and torch read them then
                set_allocator(c.cuda_allocator)

                from mizuoto.runtime.accel import features
                from mizuoto.runtime.device import DeviceService
                from mizuoto.runtime.manager import ModelManager
                from mizuoto.runtime.runtime import Runtime, RuntimeOptions

                devices = DeviceService()
                runtime = Runtime(devices, ModelManager(devices), RuntimeOptions())
                runtime.manager._policy = lambda: runtime.policy  # the runtime's, refreshed below
                runtime.manager.load_calibration(self._settings.cache_dir / "memory" / "calibration.json")
                self.fast = features.apply(list(c.fast))
                self._runtime = runtime
            # Read at every use, so a changed setting reaches the next load (loaded models keep theirs).
            c = self._settings.settings.compute
            self._runtime.options.compile = c.compile
            self._runtime.options.kv_cache = c.kv_cache
            self._runtime.options.cuda_graphs = c.cuda_graphs
            self._runtime.policy = policy_from(c)
            devices = self._runtime.devices
            devices._simulate = {"cpu": c.simulate_vram_mb * 2**20} if c.simulate_vram_mb else {}
            devices.strict = bool(c.simulate_vram_mb)
            devices.limit_bytes = c.vram_limit_mb * 2**20 if c.vram_limit_mb else None
            return self._runtime

    @property
    def started(self) -> bool:
        """Whether the runtime was built in this process (a model was loaded at some point)."""
        return self._runtime is not None

    # -- what is installed ---------------------------------------------------------------

    def implemented(self, engine_id: str) -> bool:
        """Whether ``engine_id``'s implementation is installed (built in, or a distribution registered under ``mizuoto.engines``)."""
        from mizuoto.runtime.engine import engine_ids

        return engine_id in engine_ids()

    def missing(self, engine_id: str) -> list[str]:
        """The packages ``engine_id`` needs that are not installed here: ``torch``, its own distribution (a licence-gated
        engine's). Empty: the engine is available."""
        out: list[str] = []
        if engine_id not in TORCH_FREE and not torch_installed():
            out.append("torch")
        if not self.implemented(engine_id):
            out.append(DISTRIBUTIONS.get(engine_id, f"mizuoto-{engine_id}"))
        return out

    def available(self, engine_id: str) -> bool:
        return not self.missing(engine_id)

    def incompatible(self, engine_id: str, options: set[str]) -> str | None:
        """Why an installed licence-gated engine does not go with this Mizuoto, or None. It is released on its own: its
        ``mizuoto`` range must hold this version (pip enforces it, unless told not to), and its ``OPTION_SCHEMA`` must
        name the options of the manifest Mizuoto ships for it (``options``). Imports the engine's module (not torch)."""
        from mizuoto.version import VERSION

        name = DISTRIBUTIONS.get(engine_id)
        if name is None or not self.implemented(engine_id):
            return None
        version = distribution_version(engine_id) or "?"
        spec = mizuoto_range(engine_id)
        if spec and not satisfies(VERSION, spec):
            return f"{name} {version} works with Mizuoto {spec}, not {VERSION}"
        from mizuoto.runtime.engine import get_engine

        schema = set(get_engine(engine_id).OPTION_SCHEMA)
        if schema != options:
            return f"{name} {version} and Mizuoto {VERSION} disagree on its options ({', '.join(sorted(schema ^ options))})"
        return None

    def extras_missing(self, engine_id: str) -> dict[str, list[str]]:
        """The engine's optional features whose packages do not import: install spec → the missing modules
        (``{"mizuoto[gptsovits-ja]": ["pyopenjtalk"]}``). Imports the engine's module (not torch)."""
        if not self.implemented(engine_id):
            return {}
        from mizuoto.runtime.engine import get_engine

        extras: dict[str, tuple[str, ...]] = type(get_engine(engine_id)).EXTRAS
        out = {spec: [m for m in modules if importlib.util.find_spec(m) is None] for spec, modules in extras.items()}
        return {spec: modules for spec, modules in out.items() if modules}

    def trainer(self, engine_id: str) -> str | None:
        """``module:Class`` of the engine's native trainer, or None (its module imports no torch)."""
        from mizuoto.training.trainer import trainer_for

        return trainer_for(engine_id)

    def trainer_needs(self, trainer: str) -> tuple[tuple[str, ...] | None, list[str]]:
        """A trainer's own device kinds (``Trainer.DEVICES``; None: its manifest method's) and the modules it needs that
        do not import here (``Trainer.REQUIRES``), without importing its heavy modules."""
        from mizuoto.training.trainer import missing_requirements, trainer_class

        return trainer_class(trainer).DEVICES, missing_requirements(trainer)

    def probe_devices(self) -> dict[str, Any] | None:
        """What torch sees (``mizuoto.runtime.device.probe_devices``), asked in this process only once the runtime is
        built (torch is loaded then) and without initialising CUDA or XPU here; None otherwise, so the core's device
        probe asks a child process instead."""
        if self._runtime is None:
            return None
        from mizuoto.runtime.device import probe_devices

        return probe_devices(initialise=False)

    def accel_report(self, devices: list[str]) -> dict[str, dict[str, str | None]] | None:
        """Whether each attention backend runs on each device: in this process once the runtime is loaded, else in a
        child process (``python -m mizuoto.runtime.accel``)."""
        if self.started:
            from mizuoto.runtime.accel.__main__ import report

            return report(devices)
        from mizuoto.core.compute.probe import probe_accel_subprocess

        return probe_accel_subprocess(devices)

    def torch_build(self) -> dict[str, Any]:
        """What the installed torch was built for and sees (imports torch): ``version``, ``cuda``, ``hip``, ``xpu``,
        ``mps`` and whether each is usable."""
        from mizuoto.runtime.device import torch_build

        return torch_build()
