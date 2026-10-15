"""The engines of this process and the models they hold.

``loaded`` is the core's way to a model: it gives the model's ``EngineHandle``, loading it first unless it is
loaded. Behind it, one engine per (engine, device), made on demand: a ``LocalEngine`` in this process, or a
``RemoteEngine`` session for a device named ``<host>/<device>``. The service registers each local one as a memory
holder with the compute service, publishes ``runtime_changed``, keeps at most ``compute.max_resident_models``
models loaded (the least recently used idle one goes first) and, in the server, runs a reaper that unloads
models idle for longer than ``compute.unload_after_minutes`` (or ``engines.<id>.idle_minutes``).

``doctor`` reports what stands between this machine and the engines: the torch build against the hardware,
each engine's missing packages and optional extras, and the engine environments v1 left behind.
"""

from __future__ import annotations

import logging
import shutil
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from mizuoto.core.compute.models import ComputeInfo, DeviceInfo
from mizuoto.core.compute.service import ComputeService, MemoryHolder
from mizuoto.core.compute.torch_index import ROCM_FLAVOURS, choose_flavour, install_command
from mizuoto.core.engines.registry import EngineRegistry
from mizuoto.core.errors import NotFoundError
from mizuoto.core.events import EventBus
from mizuoto.core.events.models import RuntimeChangedEvent
from mizuoto.core.inference.engine import EngineHandle, EngineRequest, ModelRef
from mizuoto.core.inference.models import DoctorFinding, DoctorReport, EngineRuntime, LoadedModelInfo, RuntimeStatus
from mizuoto.core.inference.service import DISTRIBUTIONS, InferenceService, torch_version
from mizuoto.core.settings import SettingsService

if TYPE_CHECKING:
    from mizuoto.core.inference.local import LocalEngine
    from mizuoto.core.remote.client import RemoteEngine
    from mizuoto.core.remote.service import RemoteService

    Engine = LocalEngine | RemoteEngine

logger = logging.getLogger(__name__)

REAP_INTERVAL = 10.0
EVENT_INTERVAL = 0.5


class RuntimeService:
    def __init__(self, settings: SettingsService, events: EventBus, compute: ComputeService, registry: EngineRegistry, inference: InferenceService) -> None:
        self._settings = settings
        self._events = events
        self._compute = compute
        self._registry = registry
        self.inference = inference
        compute.torch_probe = inference.probe_devices
        self.model_options: Callable[[str], dict[str, Any]] | None = None
        """Set by ``build_services``: what every load of a model gets besides the engine's load options (its assets)."""
        self.remote: RemoteService | None = None
        """Set by ``build_services``: devices named ``<host>/<device>`` are a remote host's."""
        self._engines: dict[tuple[str, str], Engine] = {}
        self._lock = threading.RLock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._last_emit = 0.0
        self._emit_pending = False

    @property
    def compute(self) -> ComputeService:
        return self._compute

    # -- models ---------------------------------------------------------------------------

    def loaded(self, model: ModelRef, device: str, dtype: str = "auto", options: dict[str, Any] | None = None, request: EngineRequest | None = None) -> EngineHandle:
        """The handle of ``model`` on ``device``, loading it first unless it is loaded (here, or on the remote host
        ``device`` names). ``request`` carries the caller's cancel and progress for the load."""
        return self.engine(model.engine, device).load(model.id, model.path, model.variant, dtype, options, request)

    def description(self, model_id: str) -> dict[str, Any] | None:
        """What a loaded model offers (``tasks.Description`` as a dict), wherever it is loaded; None when it is not."""
        engine_id = model_id.split("/", 1)[0]
        for e in self._all():
            if e.engine == engine_id:
                found = e.description(model_id)
                if found:
                    return found
        return None

    # -- engines --------------------------------------------------------------------------

    def device_for(self, engine_id: str) -> str:
        """The device ``engine_id`` runs on, from its settings and the machine (or a remote host's ``<host>/<device>``)."""
        manifest = self._registry.manifest(engine_id)
        es = self._settings.settings.engines.get(engine_id)
        if es is not None and self.remote is not None and self.remote.split(es.device) is not None:
            return self.remote.check_device(es.device)
        return self._compute.resolve_device(list(manifest.devices), es.device if es else "auto")

    def _local(self) -> list[LocalEngine]:
        from mizuoto.core.inference.local import LocalEngine

        with self._lock:
            return [e for e in self._engines.values() if isinstance(e, LocalEngine)]

    def holds(self, device: str) -> bool:
        """Whether this process has a model loaded on ``device`` (its memory is ours to reclaim)."""
        return any(e.device == device and e.loaded_models() for e in self._local())

    def engine(self, engine_id: str, device: str) -> Engine:
        """``engine_id`` on ``device``: in this process, or a session with the remote host ``device`` names."""
        from mizuoto.core.inference.local import LocalEngine

        key = (engine_id, device)
        with self._lock:
            found = self._engines.get(key)
            if found is not None:
                return found
            es = self._settings.settings.engines.get(engine_id)
            options = dict(es.load_options) if es is not None else {}
            if self.remote is not None and self.remote.split(device) is not None:
                remote = self.remote.engine(engine_id, device, options, on_change=self._changed)
                self._engines[key] = remote
                return remote
            self._registry.manifest(engine_id)
            local = LocalEngine(engine_id, device, self._settings.cache_dir / "scratch" / engine_id, options, self.inference, on_change=self._changed)
            local.model_options = lambda model_id: self.model_options(model_id) if self.model_options is not None else {}
            local.make_room = self.make_room
            local.vram_mode = lambda e=engine_id: self._engine_vram_mode(e)
            local.attention = lambda e=engine_id: self.attention(e)
            local.pinned = lambda model_id, e=engine_id: self._pinned(e, model_id)
            self._engines[key] = local
            self._compute.add_holder(
                MemoryHolder(
                    f"engine:{engine_id}:{device}",
                    busy=lambda e=local: f"{e.engine} is speaking" if e.busy else None,
                    cached=lambda e=local: e.loaded_models(),
                    release=lambda e=local: e.release_memory(),
                    device=lambda e=local: e.device,
                )
            )
            return local

    def find(self, engine_id: str, device: str | None = None) -> list[Engine]:
        with self._lock:
            found = [e for (eid, d), e in self._engines.items() if eid == engine_id and device in (None, d)]
        if not found:
            raise NotFoundError(f"{engine_id} is not running{' on ' + device if device else ''}", {"kind": "engine", "id": engine_id})
        return found

    # -- what is loaded -------------------------------------------------------------------

    def status(self) -> RuntimeStatus:
        engines: list[EngineRuntime] = []
        models: list[LoadedModelInfo] = []
        for e in self._all():
            info = e.info()
            if info.state != "stopped" or info.last_error is not None or e.loaded_models():
                engines.append(info)
            models += e.loaded_info()
        models.sort(key=lambda m: -(m.last_used or 0))
        return RuntimeStatus(torch=torch_version(), models=models, engines=engines)

    def unload(self, model_id: str | None = None) -> RuntimeStatus:
        """Unload ``model_id`` from every engine that holds it, or every model not in use."""
        if model_id is not None:
            if not any(model_id in s.loaded_models() for s in self._all()):
                raise NotFoundError(f"{model_id} is not loaded", {"kind": "model", "id": model_id})
            self.unload_model(model_id)
        else:
            for s in self._all():
                if not s.busy:
                    for m in s.loaded_models():
                        self._unload(s, m)
        return self.status()

    def _all(self) -> list[Engine]:
        with self._lock:
            return list(self._engines.values())

    def _unload(self, e: Engine, model_id: str) -> None:
        try:
            e.unload(model_id)
        except Exception:
            logger.exception("Unloading %s failed; stopping %s", model_id, e.engine)
            e.shutdown()

    def unload_model(self, model_id: str) -> None:
        """Unload a model wherever it is loaded (its files are about to change)."""
        engine_id = model_id.split("/", 1)[0]
        for e in self._all():
            if e.engine == engine_id and model_id in e.loaded_models():
                self._unload(e, model_id)

    def unload_engine(self, engine_id: str) -> None:
        """Unload every model of an engine and forget it (its settings changed)."""
        with self._lock:
            found = [e for (eid, _), e in self._engines.items() if eid == engine_id]
            for e in found:
                self._engines.pop((e.engine, e.device), None)
                self._compute.remove_holder(f"engine:{e.engine}:{e.device}")
        for e in found:
            e.shutdown()
        if found:
            self._changed()

    def release_device(self, device: str) -> list[str]:
        """Move every idle model's weights on ``device`` off it (to RAM; models stay loaded): another process needs it."""
        if not self.inference.started or "/" in device:
            return []
        released = self.inference.runtime.manager.release_idle(device)
        if released:
            logger.info("Moved %s off %s", ", ".join(released), device)
            self._changed()
        return released

    def relieve_ram(self, keep: str | None = None) -> None:
        """Below ``compute.ram_reserve_mb`` of free RAM: drop host copies of weights on a device, then unload idle
        models, the least recently used first (never ``keep``, never a pinned or busy one)."""
        if not self.inference.started:
            return
        runtime = self.inference.runtime
        free, total = runtime.devices.memory("cpu")
        if free is None:
            return
        reserve = runtime.policy.ram_reserve(total)
        if free >= reserve:
            return
        dropped = runtime.manager.drop_host_copies()
        if dropped:
            logger.info("RAM is short: dropped %d MB of host copies", dropped >> 20)
            free, _ = runtime.devices.memory("cpu")
            if free is None or free >= reserve:
                return
        idle = sorted(
            ((s, m) for s in self._local() if not s.busy for m in s.loaded_models() if m != keep and not self._pinned(s.engine, m)),
            key=lambda x: x[0].last_used_at(x[1]) or 0,
        )
        for s, m in idle:
            logger.info("RAM is short (%d MB free, %d MB reserved): unloading %s", (free or 0) >> 20, reserve >> 20, m)
            self._unload(s, m)
            free, _ = runtime.devices.memory("cpu")
            if free is None or free >= reserve:
                return

    def _memory_finding(self) -> DoctorFinding:
        """How models use device and host memory, from the settings."""
        c = self._settings.settings.compute
        parts = [
            f"vram_mode {c.vram_mode}",
            f"reserve {c.vram_reserve_mb} MB",
            f"limit {c.vram_limit_mb} MB" if c.vram_limit_mb else "no limit",
            f"RAM reserve {c.ram_reserve_mb} MB" if c.ram_reserve_mb is not None else "RAM reserve auto",
            f"host copies {c.host_copies}",
            f"pinned {c.max_pinned_mb} MB" if c.max_pinned_mb else "pinned auto",
            "side streams auto" if c.async_offload < 0 else f"{c.async_offload} side streams",
            "retry when out of memory" if c.oom_retry else "no retry when out of memory",
            "fallbacks allowed" if c.oom_fallback else "no fallbacks",
        ]
        overrides = sorted(f"{e}={s.vram_mode}" for e, s in self._settings.settings.engines.items() if s.vram_mode != "auto")
        if overrides:
            parts.append("engines: " + ", ".join(overrides))
        return DoctorFinding(level="ok", check="memory", message="memory: " + "; ".join(parts))

    def _accel_finding(self, compute: ComputeInfo) -> DoctorFinding:
        """The attention backends: the settings, each optional package installed or not,
        whether it runs on each GPU (asked of torch only when a package and a GPU are there) and how to install it."""
        from mizuoto.core.compute.accel import PACKAGES, install_command, installed_version
        from mizuoto.core.compute.probe import installed_torch_build

        c = self._settings.settings.compute
        training = self._settings.settings.training.attention
        overrides = sorted(f"{e}={s.attention}" for e, s in self._settings.settings.engines.items() if s.attention != "inherit")
        parts = [f"attention {c.attention}" + (" (strict)" if c.attention_strict else ""), f"training {training}"]
        if overrides:
            parts.append("engines: " + ", ".join(overrides))
        versions = {p.backend: installed_version(p) for p in PACKAGES}
        gpus = [d.id for d in compute.devices if d.kind in ("cuda", "rocm")]
        runs: dict[str, dict[str, str | None]] = {}
        if gpus and any(versions.values()):
            runs = self.inference.accel_report(gpus) or {}
        packages = []
        listed: list[dict[str, Any]] = []
        for p in PACKAGES:
            version = versions[p.backend]
            listed.append(
                {"backend": p.backend, "title": p.title, "version": version, "runs": {device: reasons.get(p.backend) for device, reasons in runs.items()} if version else {}}
            )
            if version is None:
                packages.append(f"{p.title} not installed")
                continue
            where = [f"{device}: {reasons.get(p.backend) or 'runs'}" for device, reasons in runs.items()]
            packages.append(f"{p.title} {version}" + (f" ({'; '.join(where)})" if where else ""))
        parts.append(", ".join(packages))
        chosen = {c.attention, training, *(s.attention for s in self._settings.settings.engines.values())}
        level: Literal["ok", "warning"] = "ok"
        fix = None
        unusable: list[str] = []
        for p in PACKAGES:
            if p.backend not in chosen:
                continue
            usable = versions[p.backend] is not None and (not runs or any(reasons.get(p.backend) is None for reasons in runs.values()))
            if not usable or not gpus:
                level = "warning"
                command = install_command(p.backend, installed_torch_build())
                fix = command if versions[p.backend] is None and command else f"choose another backend than {p.backend} (it needs {p.hardware})"
                parts.append(f"{p.backend} is chosen but cannot run here: calls fall back to SDPA")
                unusable.append(p.backend)
        data = {
            "attention": c.attention,
            "strict": c.attention_strict,
            "training": training,
            "engines": dict(o.split("=", 1) for o in overrides),
            "packages": listed,
            "unusable": unusable,
        }
        return DoctorFinding(level=level, check="acceleration", message="; ".join(parts), fix=fix, data=data)

    def attention(self, engine_id: str) -> tuple[str, bool]:
        """``(backend, strict)`` for ``engine_id``'s models: ``engines.<id>.attention`` unless ``inherit``, else
        ``compute.attention``."""
        c = self._settings.settings.compute
        es = self._settings.settings.engines.get(engine_id)
        name = c.attention
        if es is not None and es.attention != "inherit":
            name = es.attention
        return name, c.attention_strict

    def _engine_vram_mode(self, engine_id: str) -> str | None:
        es = self._settings.settings.engines.get(engine_id)
        return es.vram_mode if es is not None and es.vram_mode != "auto" else None

    def _pinned(self, engine_id: str, model_id: str) -> bool:
        es = self._settings.settings.engines.get(engine_id)
        return bool(es and model_id in es.pinned_models)

    def make_room(self, model_id: str) -> None:
        """Before ``model_id`` loads here: unload the least recently used idle models beyond ``compute.max_resident_models``."""
        limit = self._settings.settings.compute.max_resident_models
        loaded = [(s, m) for s in self._local() for m in s.loaded_models() if m != model_id]
        excess = len(loaded) + 1 - limit
        if excess > 0:
            idle = sorted(((s, m) for s, m in loaded if not s.busy and not self._pinned(s.engine, m)), key=lambda x: x[0].last_used_at(x[1]) or 0)
            for s, m in idle[:excess]:
                logger.info("Unloading %s to make room for %s (compute.max_resident_models = %d)", m, model_id, limit)
                self._unload(s, m)
        self.relieve_ram(keep=model_id)

    # -- events ---------------------------------------------------------------------------

    def _changed(self) -> None:
        now = time.monotonic()
        if now - self._last_emit < EVENT_INTERVAL:
            self._emit_pending = True
            return
        self._emit()

    def _emit(self) -> None:
        self._last_emit = time.monotonic()
        self._emit_pending = False
        try:
            self._events.publish(RuntimeChangedEvent(runtime=self.status()))
        except Exception:
            logger.exception("Could not publish the runtime's state")
        self._compute.emit_usage()

    # -- background -----------------------------------------------------------------------

    def start(self) -> None:
        if self._thread is None:
            self._stop.clear()
            self._thread = threading.Thread(target=self._reap_loop, name="runtime", daemon=True)
            self._thread.start()

    def _reap_loop(self) -> None:
        while not self._stop.wait(min(REAP_INTERVAL, EVENT_INTERVAL * 2)):
            if self._emit_pending:
                self._emit()
            try:
                self.reap()
            except Exception:
                logger.exception("Unloading idle models failed")

    def reap(self, now: float | None = None) -> list[str]:
        """Unload models idle for longer than their engine's idle time and, beyond ``compute.max_resident_models``, the
        least recently used idle ones; end remote sessions idle as long. Returns what was unloaded (model ids)."""
        now = time.time() if now is None else now
        settings = self._settings.settings

        def minutes(engine_id: str) -> int:
            es = settings.engines.get(engine_id)
            return es.idle_minutes if es and es.idle_minutes is not None else settings.compute.unload_after_minutes

        unloaded: list[str] = []
        local = self._local()
        for s in local:
            limit = minutes(s.engine)
            for m in s.loaded_models():
                used = s.last_used_at(m)
                if limit and not s.busy and not self._pinned(s.engine, m) and used is not None and now - used > limit * 60:
                    self._unload(s, m)
                    unloaded.append(m)
        loaded = [(s, m) for s in local for m in s.loaded_models()]
        excess = len(loaded) - settings.compute.max_resident_models
        idle = sorted(((s, m) for s, m in loaded if not s.busy and not self._pinned(s.engine, m)), key=lambda x: x[0].last_used_at(x[1]) or 0)
        for s, m in idle[: max(0, excess)]:
            self._unload(s, m)
            unloaded.append(m)
        mono = time.monotonic()
        for s in self._all():
            if s in local or s.state not in ("ready", "busy", "starting"):
                continue
            limit = minutes(s.engine)
            if limit and not s.busy and mono - s.last_used > limit * 60:
                unloaded += s.loaded_models()
                s.shutdown()
        if unloaded:
            logger.info("Unloaded idle models: %s", ", ".join(unloaded))
        self.relieve_ram()
        return unloaded

    def close(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5)
            self._thread = None
        for s in self._all():
            try:
                s.shutdown()
            except Exception:
                logger.exception("Stopping %s failed", s.engine)

    # -- doctor ---------------------------------------------------------------------------

    def doctor(self, engine_id: str | None = None) -> DoctorReport:
        """The torch build against the hardware (with the command that fixes a mismatch), and each engine's missing
        packages and optional extras."""
        compute = self._compute.info()
        recommended = choose_flavour(compute)
        if recommended == "cpu" and _has_rocm():
            recommended = ROCM_FLAVOURS[-1]
        command = install_command(recommended)
        version = torch_version()
        findings: list[DoctorFinding] = [_devices_finding(compute), self._memory_finding()]
        findings.append(self._accel_finding(compute))
        build: str | None = None
        if version is None:
            findings.append(DoctorFinding(level="error", check="torch", message="torch is not installed: no engine can run", fix=command))
        else:
            try:
                info = self.inference.torch_build()
            except Exception as e:
                findings.append(DoctorFinding(level="error", check="torch", message=f"torch {version} does not import: {type(e).__name__}: {e}", fix=command))
            else:
                build = _build_of(info)
                findings.append(_torch_finding(info, build, recommended, compute.cuda_version, command))
        for eid in [engine_id] if engine_id else self._registry.ids():
            findings += self._engine_findings(eid, strict=engine_id is not None, torch_command=command)
        return DoctorReport(
            compute=compute,
            torch=version,
            torch_build=build,
            recommended_build=recommended,
            install_command=command,
            findings=findings,
        )

    def _engine_findings(self, engine_id: str, strict: bool, torch_command: str) -> list[DoctorFinding]:
        m = self._registry.manifest(engine_id)
        missing = self.inference.missing(engine_id)
        if missing:
            fixes = [torch_command if p == "torch" else f"pip install {p}" for p in missing]
            gated = [p for p in missing if p in DISTRIBUTIONS.values()]
            note = f" ({m.licence.name}: a distribution of its own)" if gated else ""
            return [
                DoctorFinding(
                    level="error" if strict else "warning",
                    engine=engine_id,
                    check="engine",
                    message=f"{m.title} is not available: {', '.join(missing)} {'is' if len(missing) == 1 else 'are'} not installed{note}",
                    fix="; ".join(fixes),
                )
            ]
        findings = [DoctorFinding(level="ok", engine=engine_id, check="engine", message=f"{m.title}: available")]
        try:
            extras = self.inference.extras_missing(engine_id)
        except Exception as e:
            extras = {}
            findings.append(DoctorFinding(level="error", engine=engine_id, check="engine", message=f"{m.title} does not import: {type(e).__name__}: {e}"))
        for spec, modules in extras.items():
            findings.append(
                DoctorFinding(
                    level="warning",
                    engine=engine_id,
                    check="extra",
                    message=f"{m.title}: the optional {spec} is not installed (missing {', '.join(modules)})",
                    fix=f"pip install '{spec}'",
                )
            )
        if any(t.enabled for t in m.training.values()):
            trainer = self.inference.trainer(engine_id)
            lacking = self.inference.trainer_needs(trainer)[1] if trainer else []
            if lacking:
                findings.append(
                    DoctorFinding(
                        level="warning", engine=engine_id, check="training", message=f"{m.title}: training needs {', '.join(lacking)}", fix=f"pip install {' '.join(lacking)}"
                    )
                )
        return findings


def _devices_finding(compute: ComputeInfo) -> DoctorFinding:
    """The accelerators the probe found, and with what (``mizuoto.core.compute.probe``)."""
    gpus = [d for d in compute.devices if d.kind != "cpu"]
    if not gpus:
        return DoctorFinding(level="ok", check="devices", message="No GPU found (nvidia-smi, rocm-smi, amd-smi, torch): engines run on the CPU")

    def label(d: DeviceInfo) -> str:
        memory = f", {d.memory_total_mb / 1024:.0f} GB" if d.memory_total_mb else ""
        return f"{d.id} ({d.name}{memory})"

    return DoctorFinding(level="ok", check="devices", message=f"{compute.backend}: {', '.join(label(d) for d in gpus)}; found by {compute.probed_with}")


def _gpus_hidden() -> bool:
    import os

    return any(os.environ.get(name) in ("", "-1") for name in ("CUDA_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES") if name in os.environ)


def _has_rocm() -> bool:
    return shutil.which("rocm-smi") is not None or Path("/opt/rocm").is_dir()


def _build_of(info: dict[str, Any]) -> str:
    """The build a torch is (``cu128``, ``rocm6.4``, ``xpu``, ``mps``, ``cpu``), from its version suffix or what it was built with."""
    import sys

    version = str(info.get("version") or "")
    if "+" in version:
        suffix = version.split("+", 1)[1]
        if suffix.startswith(("cu", "rocm", "xpu", "cpu")):
            return suffix
    if info.get("hip"):
        return "rocm" + ".".join(str(info["hip"]).split(".")[:2])
    if info.get("cuda"):
        return "cu" + str(info["cuda"]).replace(".", "")
    if info.get("xpu_available"):
        return "xpu"
    return "mps" if sys.platform == "darwin" else "cpu"


def _torch_finding(info: dict[str, Any], build: str, recommended: str, driver_cuda: str | None, command: str) -> DoctorFinding:
    version = info.get("version")
    if recommended.startswith("cu"):
        if not build.startswith("cu"):
            return DoctorFinding(level="warning", check="torch", message=f"torch {version} is the {build} build, and this machine has an NVIDIA GPU", fix=command)
        if not info.get("cuda_available") and _gpus_hidden():
            return DoctorFinding(level="warning", check="torch", message=f"torch {version} (CUDA {info.get('cuda')}) sees no GPU: CUDA_VISIBLE_DEVICES hides them")
        if not info.get("cuda_available"):
            return DoctorFinding(
                level="error",
                check="torch",
                message=f"torch {version} (CUDA {info.get('cuda')}) cannot use the GPU: the driver runs CUDA {driver_cuda or 'unknown'}",
                fix=command,
            )
        return DoctorFinding(level="ok", check="torch", message=f"torch {version} (CUDA {info.get('cuda')}) sees {info.get('device_count')} GPU(s)")
    if recommended.startswith("rocm"):
        if not build.startswith("rocm") or not info.get("cuda_available"):
            return DoctorFinding(level="warning", check="torch", message=f"torch {version} ({build}) cannot use this machine's AMD GPU", fix=command)
        return DoctorFinding(level="ok", check="torch", message=f"torch {version} (ROCm {info.get('hip')}) sees {info.get('device_count')} GPU(s)")
    if recommended == "xpu" and not info.get("xpu_available"):
        return DoctorFinding(level="warning", check="torch", message=f"torch {version} ({build}) cannot use this machine's Intel GPU", fix=command)
    if recommended == "mps" and not info.get("mps_available"):
        return DoctorFinding(level="warning", check="torch", message=f"torch {version} cannot use the Apple GPU (MPS)", fix=command)
    return DoctorFinding(level="ok", check="torch", message=f"torch {version} ({build}) on {'the CPU' if recommended == 'cpu' else recommended.upper()}")
