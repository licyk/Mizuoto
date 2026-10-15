"""Remote hosts: what an agent reports, and a host's state as the UI shows it."""

from __future__ import annotations

from typing import Literal

from pydantic import Field

from mizuoto.core.compute.models import DeviceInfo
from mizuoto.core.record import Record


class AgentEngine(Record):
    id: str
    ready: bool
    """The engine is available on the agent (its packages import there)."""
    missing: list[str] = Field(default_factory=list)
    """What the agent lacks for it (``EngineStatus.missing`` there)."""


class AgentInfo(Record):
    """What a ``mizuoto agent`` has: its version, devices, engines and installed models."""

    name: str
    version: str
    protocol: int
    devices: list[DeviceInfo] = Field(default_factory=list)
    engines: list[AgentEngine] = Field(default_factory=list)
    models: list[str] = Field(default_factory=list)
    """Installed models (catalogue and user models), by id."""


class RemoteHostStatus(Record):
    name: str
    address: str
    tls: bool
    state: Literal["online", "offline", "refused"]
    """``refused``: reachable, but the token or the protocol version is wrong."""
    error: str | None = None
    info: AgentInfo | None = None
    engines: list[str] = Field(default_factory=list)
    """The engines whose device setting names this host."""
