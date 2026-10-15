"""What the runtime has loaded, and what ``mizuoto doctor`` finds."""

from typing import Any, Literal

from pydantic import Field

from mizuoto.core.compute.models import ComputeInfo
from mizuoto.core.jobs.models import ErrorInfo
from mizuoto.core.record import Record

EngineState = Literal["starting", "ready", "busy", "stopping", "stopped", "crashed"]
"""``starting``/``stopping`` and ``crashed`` only for an engine on a remote host (its session); a local engine is
``ready``, ``busy`` or ``stopped`` (nothing loaded), and ``crashed`` when its implementation failed to import."""


class LoadedModelInfo(Record):
    """A model loaded in this process (or on a remote host's agent)."""

    model_id: str
    engine: str
    device: str
    bytes: int | None = None
    """Bytes of weights the model manager holds for it (on its device, or offloaded to RAM); None on a remote host."""
    loaded_bytes: int | None = None
    """Of those, the bytes on its device now (the rest in RAM; a model that does not fit streams them as it runs)."""
    streamed_bytes: int | None = None
    """Bytes of its weights left in RAM to stream to the device as its layers run (0: it fits whole)."""
    host_bytes: int | None = None
    """Bytes of host copies kept of its weights (RAM), so moving it off the device costs nothing."""
    last_used: float | None = None
    """Unix time of its last request."""
    attention: str | None = None
    """How its attention is computed (``compute.attention`` when it loaded: ``auto``, ``fast`` or a backend)."""
    busy: bool = False
    remote: str | None = None
    """The remote host it is loaded on."""


class EngineRuntime(Record):
    """An engine on one device: in this process, or a session with a remote host's agent."""

    engine: str
    device: str
    state: EngineState
    current_request: str | None = None
    started_at: str | None = None
    last_error: ErrorInfo | None = None
    restarts: int = 0
    """Sessions with a remote host opened again after it was lost."""
    remote: str | None = None


class RuntimeStatus(Record):
    torch: str | None = None
    """The installed torch's version (read from its metadata, not imported); None when torch is missing."""
    models: list[LoadedModelInfo] = Field(default_factory=list)
    engines: list[EngineRuntime] = Field(default_factory=list)


class UnloadRequest(Record):
    model: str | None = None
    """The model to unload; None: every model not in use."""


class DoctorFinding(Record):
    level: Literal["ok", "warning", "error"]
    engine: str | None = None
    check: str
    """``torch``, ``engine``, ``extra``, ``training``, ``envs``…"""
    message: str
    fix: str | None = None
    """What to run, or do, about it."""
    data: dict[str, Any] | None = None
    """The finding as fields, for an interface that words it in its own language (``acceleration``: the settings,
    each optional package's version and where it runs, the chosen backends that cannot run here)."""


class DoctorReport(Record):
    compute: ComputeInfo | None = None
    """The devices found (``mizuoto.core.compute.probe``): the backend, each accelerator, and the probe that found them."""
    torch: str | None = None
    """The installed torch's version; None when missing."""
    torch_build: str | None = None
    """``cu128``, ``rocm6.4``, ``xpu``, ``cpu`` or ``mps`` (macOS): what the installed torch was built for."""
    recommended_build: str
    """The build for this machine's hardware."""
    install_command: str
    """The command that installs the recommended build into this interpreter."""
    findings: list[DoctorFinding] = Field(default_factory=list)
