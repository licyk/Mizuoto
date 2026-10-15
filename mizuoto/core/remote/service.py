"""Remote hosts (``remote.hosts``): other machines running ``mizuoto agent``.

An engine runs remotely when its device is ``<host>/<device>`` (``engines.qwen3tts.device =
"gpu-box/cuda:0"``). Its environment and models are the agent's: the host plans requests against
its own catalogue and the agent's list of installed models, and sends the work through a
``RemoteEngine`` session (its models are ``RemoteEngineHandle``s). Training stays local.
"""

from __future__ import annotations

import hashlib
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Literal

from mizuoto.core.errors import DeviceError, RemoteError, ValidationError
from mizuoto.core.remote.client import RemoteConfig, RemoteEngine
from mizuoto.core.remote.frames import ConnectionClosed, FrameConnection, FrameError
from mizuoto.core.remote.models import AgentInfo, RemoteHostStatus
from mizuoto.core.remote.wire import AGENT_PROTOCOL, UPGRADE_HINT, client_socket
from mizuoto.core.settings import SettingsService
from mizuoto.core.settings.models import RemoteHost

if TYPE_CHECKING:
    from mizuoto.core.engines.registry import EngineRegistry

INFO_TTL = 15.0
"""Seconds an agent's report is reused (model states are read per model)."""
FAILED_TTL = 5.0


class RemoteService:
    def __init__(self, settings: SettingsService) -> None:
        self._settings = settings
        self._lock = threading.Lock()
        self._info: dict[str, tuple[float, AgentInfo | None, RemoteError | None]] = {}
        self._sha: dict[tuple[str, int, int], str] = {}
        self.registry: EngineRegistry | None = None
        """Set by ``build_services``: engine ids are checked against it."""

    # -- hosts ------------------------------------------------------------------------------

    def hosts(self) -> dict[str, RemoteHost]:
        return dict(self._settings.settings.remote.hosts)

    def host(self, name: str) -> RemoteHost:
        found = self.hosts().get(name)
        if found is None:
            raise DeviceError(f"No remote host {name!r}", {"host": name, "hint": "mizuoto remote add <name> <address> --token …"})
        return found

    @property
    def heartbeat_timeout(self) -> float:
        return self._settings.settings.remote.heartbeat_timeout

    @staticmethod
    def split(device: str) -> tuple[str, str] | None:
        """``gpu-box/cuda:0`` → (``gpu-box``, ``cuda:0``); a local device → None."""
        if "/" not in device:
            return None
        host, _, inner = device.partition("/")
        return host, inner or "auto"

    def host_for(self, engine_id: str) -> str | None:
        """The remote host ``engine_id`` runs on, from its device setting."""
        es = self._settings.settings.engines.get(engine_id)
        parts = self.split(es.device) if es else None
        return parts[0] if parts else None

    def engines_on(self, name: str) -> list[str]:
        return sorted(e for e, es in self._settings.settings.engines.items() if (self.split(es.device) or ("",))[0] == name)

    # -- the agent ------------------------------------------------------------------------------

    def connect(self, name: str) -> FrameConnection:
        """A session with ``name``'s agent, past the handshake."""
        conn, _ = self._handshake(name)
        return conn

    def _handshake(self, name: str) -> tuple[FrameConnection, AgentInfo]:
        host = self.host(name)
        timeout = self._settings.settings.remote.connect_timeout
        try:
            sock = client_socket(host.address, host.tls, host.ca_file, timeout)
        except OSError as e:
            raise RemoteError(f"Cannot reach {name} at {host.address}: {e}", {"host": name, "address": host.address, "hint": "is `mizuoto agent` running there?"}) from e
        conn = FrameConnection(sock)
        try:
            sock.settimeout(timeout)
            conn.send({"type": "hello", "token": host.token, "protocol": AGENT_PROTOCOL})
            reply, _ = conn.recv()
            sock.settimeout(None)
        except (ConnectionClosed, FrameError, OSError, ValueError) as e:
            conn.close()
            raise RemoteError(f"{name} closed the connection: {e}", {"host": name, "hint": "with tls on, the agent must serve TLS (and the other way round)"}) from e
        if reply.get("type") != "info":
            conn.close()
            message = str(reply.get("message") or reply.get("type"))
            detail = reply.get("detail")
            detail = detail if isinstance(detail, dict) else {}
            protocol = "protocol" in message or "agent_protocol" in detail
            hint = UPGRADE_HINT if protocol else f"mizuoto remote add {name} {host.address} --token <the agent's token>"
            raise RemoteError(f"{name} refused: {message}", {"host": name, "refused": True, "hint": hint})
        info = AgentInfo.model_validate(reply.get("result") or {})
        if info.protocol != AGENT_PROTOCOL:
            conn.close()
            raise RemoteError(
                f"{name} refused: its agent (Mizuoto {info.version}) speaks protocol {info.protocol}, this host {AGENT_PROTOCOL}",
                {"host": name, "refused": True, "hint": UPGRADE_HINT},
            )
        with self._lock:
            self._info[name] = (time.monotonic(), info, None)
        return conn, info

    def info(self, name: str, refresh: bool = False) -> AgentInfo:
        """What ``name``'s agent has; reused for a few seconds unless ``refresh``."""
        with self._lock:
            cached = self._info.get(name)
        if cached is not None and not refresh:
            at, info, error = cached
            ttl = INFO_TTL if info is not None else FAILED_TTL
            if time.monotonic() - at < ttl:
                if error is not None:
                    raise error
                assert info is not None
                return info
        try:
            conn, info = self._handshake(name)
        except RemoteError as e:
            with self._lock:
                self._info[name] = (time.monotonic(), None, e)
            raise
        conn.close()
        return info

    def status(self, refresh: bool = False) -> list[RemoteHostStatus]:
        out: list[RemoteHostStatus] = []
        for name, host in sorted(self.hosts().items()):
            try:
                info = self.info(name, refresh=refresh)
                out.append(RemoteHostStatus(name=name, address=host.address, tls=host.tls, state="online", info=info, engines=self.engines_on(name)))
            except RemoteError as e:
                state = "refused" if e.detail.get("refused") else "offline"
                out.append(RemoteHostStatus(name=name, address=host.address, tls=host.tls, state=state, error=e.message, engines=self.engines_on(name)))
        return out

    def forget(self, name: str) -> None:
        with self._lock:
            self._info.pop(name, None)

    # -- editing --------------------------------------------------------------------------------

    def add_host(self, name: str, address: str, token: str | None = None, tls: bool = False, ca_file: str = "") -> RemoteHostStatus:
        """Add or change a host (``token`` None keeps the saved one) and report whether it answers."""
        name = name.strip()
        if not name or "/" in name or "." in name:
            raise ValidationError("A host's name cannot be empty or contain / or .")
        if not address.strip():
            raise ValidationError("Give the agent's address: host or host:port")
        old = self.hosts().get(name)
        token = token if token is not None else (old.token if old else "")
        if not token:
            raise ValidationError(f"{name} needs the agent's token", {"hint": "`mizuoto agent` prints it when it makes one (remote.agent.token there)"})
        self._settings.update({"remote": {"hosts": {name: {"address": address.strip(), "token": token, "tls": tls, "ca_file": ca_file}}}})
        self.forget(name)
        return next(st for st in self.status(refresh=True) if st.name == name)

    def remove_host(self, name: str) -> list[str]:
        """Remove a host; the engines that ran there run here again. Returns them."""
        self.host(name)
        engines = self.engines_on(name)
        self._settings.update({"remote": {"hosts": {name: None}}, "engines": {e: {"device": "auto"} for e in engines}})
        self.forget(name)
        return engines

    def use(self, engine_id: str, name: str | None, device: str = "auto") -> str:
        """Run ``engine_id`` on ``name`` (its ``device`` there), or here again (``name`` None). Returns the device setting."""
        if self.registry is not None:
            self.registry.manifest(engine_id)
        if name is None:
            value = "auto"
        else:
            self.host(name)
            if "/" in device:
                raise ValidationError("The device on the host is cuda:0, cpu or auto")
            value = f"{name}/{device or 'auto'}"
        self._settings.update({"engines": {engine_id: {"device": value}}})
        return value

    # -- what the host plans against -------------------------------------------------------

    def model_state(self, name: str, model_id: str) -> Literal["installed", "missing"]:
        """``installed`` when the agent has the model, ``missing`` when not or unreachable."""
        try:
            return "installed" if model_id in self.info(name).models else "missing"
        except RemoteError:
            return "missing"

    def engine_missing(self, name: str, engine_id: str) -> list[str]:
        """What the agent lacks for ``engine_id`` (empty: available there); ``["remote host"]`` when it cannot be reached."""
        try:
            info = self.info(name)
        except RemoteError:
            return [f"remote host {name}"]
        found = next((e for e in info.engines if e.id == engine_id), None)
        if found is None:
            return [f"{engine_id} on {name}"]
        return [] if found.ready else list(found.missing) or [f"{engine_id} on {name}"]

    def check_device(self, device: str) -> str:
        parts = self.split(device)
        if parts is None:
            return device
        self.host(parts[0])
        return device

    # -- engines ------------------------------------------------------------------------------

    def engine(self, engine_id: str, device: str, options: dict, on_change: Callable[[], None] | None = None) -> RemoteEngine:
        """A session with the agent for ``engine_id`` on ``device`` (``<host>/<device>``), opened on first use."""
        parts = self.split(device)
        if parts is None:
            raise DeviceError(f"{device} is not a remote device")
        self.host(parts[0])
        config = RemoteConfig(
            engine=engine_id, device=device, host=parts[0], inner_device=parts[1], options=options, max_models=self._settings.settings.compute.max_resident_models
        )
        return RemoteEngine(self, config, on_change)

    def sha256(self, path: Path) -> str:
        st = path.stat()
        key = (str(path), st.st_size, st.st_mtime_ns)
        with self._lock:
            found = self._sha.get(key)
        if found is not None:
            return found
        digest = hashlib.sha256()
        with path.open("rb") as f:
            for block in iter(lambda: f.read(1 << 20), b""):
                digest.update(block)
        value = digest.hexdigest()
        with self._lock:
            self._sha[key] = value
        return value
