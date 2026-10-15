"""``mizuoto agent``: this machine's engines, driven by another Mizuoto.

The agent owns its own data folder (models, settings) and serves its engines (in its own process, as the
host runs its own) to hosts holding its token. ``AGENT_PROTOCOL`` 2: after ``hello``
(token and protocol: a host of another protocol is refused with an upgrade hint), a connection is a session for one
engine on one device: ``open``, then ``load``, ``prepare_voice``, ``synthesize``, ``convert``, ``edit``, ``call``,
``unload`` and ``status``, run one after another; ``have``/``put`` send files by SHA-256; ``cancel`` and ``text``
(streamed text) arrive at any time. The agent answers each operation with its ``progress``, ``chunk`` (binary:
float32 little-endian mono) and ``file`` frames (what a ``prepare_voice`` wrote) and the ``result`` or ``error``,
and beats every few seconds while it works.
"""

from __future__ import annotations

import contextlib
import hashlib
import hmac
import logging
import os
import queue
import shutil
import socket
import ssl
import threading
import time
from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING, Any

from mizuoto.core import codes
from mizuoto.core.clock import new_id
from mizuoto.core.errors import EnvMissingError, MizuotoError, ValidationError
from mizuoto.core.inference.engine import EngineRequest
from mizuoto.core.inference.tasks import conversion_task_from_dict, edit_task_from_dict, synthesis_task_from_dict, voice_from_dict
from mizuoto.core.remote.frames import ConnectionClosed, FrameConnection, FrameError
from mizuoto.core.remote.models import AgentEngine, AgentInfo
from mizuoto.core.remote.wire import AGENT_PROTOCOL, FILE_REF, OUT_REF, UPGRADE_HINT
from mizuoto.version import VERSION

if TYPE_CHECKING:
    from mizuoto.core.context import Services
    from mizuoto.core.inference.local import LocalEngine, LocalHandle

logger = logging.getLogger(__name__)

HEARTBEAT = 5.0
KEEP_FILES_DAYS = 30


class AgentServer:
    def __init__(self, services: Services, host: str, port: int, token: str, ssl_context: ssl.SSLContext | None = None, heartbeat: float = HEARTBEAT) -> None:
        if not token:
            raise ValidationError("The agent needs a token", {"hint": "mizuoto config set remote.agent.token <a long random string>"})
        self.services = services
        self._host = host
        self._port = port
        self._token = token
        self._ssl = ssl_context
        self.heartbeat = heartbeat
        self._listener: socket.socket | None = None
        self._stop = threading.Event()
        self._sessions: set[_Session] = set()
        self._lock = threading.Lock()
        self.files_dir = services.settings.cache_dir / "remote" / "files"
        self.out_dir = services.settings.cache_dir / "remote" / "out"

    @property
    def address(self) -> tuple[str, int]:
        if self._listener is None:
            return self._host, self._port
        host, port = self._listener.getsockname()[:2]
        return str(host), int(port)

    def start(self) -> None:
        self._prune()
        family = socket.AF_INET6 if ":" in self._host else socket.AF_INET
        listener = socket.socket(family, socket.SOCK_STREAM)
        listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            listener.bind((self._host, self._port))
        except OSError as e:
            listener.close()
            raise ValidationError(f"Cannot listen on {self._host}:{self._port}: {e.strerror or e}", {"hint": "another agent may run there; choose another --port"}) from e
        listener.listen(16)
        self._listener = listener
        threading.Thread(target=self._accept_loop, name="agent-accept", daemon=True).start()

    def serve_forever(self) -> None:
        if self._listener is None:
            self.start()
        try:
            while not self._stop.wait(1.0):
                pass
        finally:
            self.close()

    def close(self) -> None:
        self._stop.set()
        if self._listener is not None:
            try:
                self._listener.close()
            except OSError:
                pass
        with self._lock:
            sessions = list(self._sessions)
        for s in sessions:
            s.close()

    def _prune(self) -> None:
        """Files no host has asked for in a month go; unfinished uploads and outputs always."""
        cutoff = time.time() - KEEP_FILES_DAYS * 86400
        if self.files_dir.is_dir():
            for folder in self.files_dir.iterdir():
                try:
                    if folder.stat().st_mtime < cutoff or not any(f for f in folder.iterdir() if not f.name.endswith(".part")):
                        shutil.rmtree(folder, ignore_errors=True)
                except OSError:
                    pass
        shutil.rmtree(self.out_dir, ignore_errors=True)

    def _accept_loop(self) -> None:
        assert self._listener is not None
        while not self._stop.is_set():
            try:
                sock, peer = self._listener.accept()
            except OSError:
                return
            threading.Thread(target=self._open_session, args=(sock, peer), name=f"agent-{peer[0]}", daemon=True).start()

    def _open_session(self, sock: socket.socket, peer: Any) -> None:
        try:
            sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            if self._ssl is not None:
                sock.settimeout(30)
                sock = self._ssl.wrap_socket(sock, server_side=True)
                sock.settimeout(None)
        except (OSError, ssl.SSLError) as e:
            logger.warning("TLS with %s failed: %s", peer[0], e)
            sock.close()
            return
        session = _Session(self, FrameConnection(sock), f"{peer[0]}:{peer[1]}")
        with self._lock:
            self._sessions.add(session)
        try:
            session.run()
        finally:
            with self._lock:
                self._sessions.discard(session)

    def check_token(self, token: str) -> bool:
        return hmac.compare_digest(token.encode("utf-8"), self._token.encode("utf-8"))

    def info(self) -> AgentInfo:
        s = self.services
        engines = []
        for engine_id in s.engines.ids():
            m = s.engines.manifest(engine_id)
            status = s.engines.status(engine_id)
            engines.append(AgentEngine(id=m.id, ready=status.available, missing=status.missing))
        models = [m.id for m in s.models.list_models(installed=True)]
        return AgentInfo(name=socket.gethostname(), version=VERSION, protocol=AGENT_PROTOCOL, devices=s.compute.info().devices, engines=engines, models=models)

    # -- files ---------------------------------------------------------------------------

    def file_path(self, sha256: str, name: str) -> Path:
        safe = Path(name).name or "file"
        return self.files_dir / sha256 / safe

    def resolve_file(self, sha256: str, name: str) -> Path | None:
        """The stored file with this content, under ``name`` (a link to the copy it arrived as)."""
        wanted = self.file_path(sha256, name)
        if wanted.is_file():
            return wanted
        folder = self.files_dir / sha256
        found = next((f for f in sorted(folder.iterdir()) if f.is_file() and not f.name.endswith(".part")), None) if folder.is_dir() else None
        if found is None:
            return None
        try:
            os.link(found, wanted)
        except OSError:
            shutil.copy2(found, wanted)
        return wanted

    def has_file(self, sha256: str) -> bool:
        folder = self.files_dir / sha256
        if not folder.is_dir():
            return False
        found = [f for f in folder.iterdir() if not f.name.endswith(".part")]
        if found:
            os.utime(folder)
        return bool(found)


class _Session:
    def __init__(self, agent: AgentServer, conn: FrameConnection, peer: str) -> None:
        self.agent = agent
        self.conn = conn
        self.peer = peer
        self.ops: queue.Queue[tuple[dict[str, Any], bytes | None] | None] = queue.Queue()
        self.cancels: dict[str, threading.Event] = {}
        self.texts: dict[str, queue.Queue[tuple[str, bool]]] = {}
        self.lock = threading.Lock()
        self.closed = threading.Event()
        self.authed = False
        self.engine: str | None = None
        self.device: str | None = None
        self.upload: tuple[str, Path, Any] | None = None
        """(sha256, .part path, open file) of the upload in progress."""

    def close(self) -> None:
        self.closed.set()
        self.conn.close()
        self.ops.put(None)

    def send(self, message: dict[str, Any], payload: bytes | None = None) -> None:
        try:
            self.conn.send(message, payload)
        except ConnectionClosed:
            self.close()

    def run(self) -> None:
        threading.Thread(target=self._read, name=f"agent-read-{self.peer}", daemon=True).start()
        threading.Thread(target=self._beat, name=f"agent-beat-{self.peer}", daemon=True).start()
        try:
            while not self.closed.is_set():
                item = self.ops.get()
                if item is None:
                    break
                message, payload = item
                op_id = str(message.get("id") or "")
                try:
                    result = self._handle(message, payload)
                    if result is not None:
                        self.send({"type": "result", "id": op_id, "result": result})
                except MizuotoError as e:
                    self.send({"type": "error", "id": op_id, "code": e.code, "message": e.message, "detail": _plain(e.detail)})
                except Exception as e:
                    logger.exception("Agent request %s failed", message.get("type"))
                    self.send({"type": "error", "id": op_id, "code": codes.INTERNAL_ERROR, "message": f"{type(e).__name__}: {e}", "detail": {}})
                finally:
                    with self.lock:
                        self.cancels.pop(op_id, None)
                        self.texts.pop(op_id, None)
        finally:
            self.close()
            if self.upload is not None:
                self.upload[2].close()
                self.upload[1].unlink(missing_ok=True)

    def _read(self) -> None:
        while not self.closed.is_set():
            try:
                message, payload = self.conn.recv()
            except (ConnectionClosed, FrameError, OSError, ValueError):
                self.close()
                return
            kind = message.get("type")
            if not self.authed:
                if kind != "hello" or not isinstance(message.get("token"), str) or not self.agent.check_token(message["token"]):
                    logger.warning("Refused %s: wrong token", self.peer)
                    self.send({"type": "error", "id": "", "code": codes.REMOTE_UNAVAILABLE, "message": "The agent refused the token", "detail": {}})
                    self.close()
                    return
                if message.get("protocol") != AGENT_PROTOCOL:
                    logger.warning("Refused %s: it speaks protocol %s, this agent %s", self.peer, message.get("protocol"), AGENT_PROTOCOL)
                    self.send(
                        {
                            "type": "error",
                            "id": "",
                            "code": codes.REMOTE_UNAVAILABLE,
                            "message": f"The agent speaks protocol {AGENT_PROTOCOL} (Mizuoto {VERSION}), the host {message.get('protocol')}",
                            "detail": {"agent": VERSION, "agent_protocol": AGENT_PROTOCOL, "host_protocol": message.get("protocol"), "hint": UPGRADE_HINT},
                        }
                    )
                    self.close()
                    return
                self.authed = True
                self.send({"type": "info", "id": "", "result": self.agent.info().model_dump(mode="json")})
                continue
            op_id = str(message.get("id") or "")
            if kind == "cancel":
                with self.lock:
                    event = self.cancels.get(op_id)
                if event is not None:
                    event.set()
            elif kind == "text":
                with self.lock:
                    texts = self.texts.setdefault(op_id, queue.Queue()) if op_id in self.cancels else None
                if texts is not None:
                    texts.put((str(message.get("text") or ""), bool(message.get("final"))))
            elif kind == "ping":
                self.send({"type": "pong", "id": op_id})
            else:
                with self.lock:
                    self.cancels[op_id] = threading.Event()
                self.ops.put((message, payload))

    def _beat(self) -> None:
        while not self.closed.wait(self.agent.heartbeat):
            if self.authed:
                self.send({"type": "heartbeat", "id": ""})

    # -- operations ----------------------------------------------------------------------------

    def _engine(self) -> LocalEngine:
        if self.engine is None or self.device is None:
            raise ValidationError("Open an engine first")
        from mizuoto.core.inference.local import LocalEngine

        engine = self.agent.services.runtime.engine(self.engine, self.device)
        if not isinstance(engine, LocalEngine):
            raise ValidationError("An agent runs its engines itself, not on another remote host")
        return engine

    def _status(self) -> dict[str, Any]:
        """The engine's state on this machine and the models it holds, as the host shows them."""
        engine = self._engine()
        return {"engine": engine.info().model_dump(mode="json"), "models": [m.model_dump(mode="json") for m in engine.loaded_info()]}

    def _handle_of(self, model_id: str) -> LocalHandle:
        handle = self._engine().handle(model_id)
        if handle is None:
            raise ValidationError(f"{model_id} is not loaded on {socket.gethostname()}", {"model": model_id})
        return handle

    def _request(self, op_id: str, message: dict[str, Any]) -> EngineRequest:
        """The engine request for an operation: the host's cancel and streamed text reach it, progress goes back."""
        with self.lock:
            cancel = self.cancels.get(op_id)
            texts = self.texts.setdefault(op_id, queue.Queue())

        def progress(stage: str, value: float | None, text: str | None) -> None:
            self.send({"type": "progress", "id": op_id, "stage": stage, "value": value, "message": text})

        return EngineRequest(id=op_id, cancel=cancel, text_input=texts if message.get("type") == "synthesize" else None, progress=progress)

    def _handle(self, message: dict[str, Any], payload: bytes | None) -> Any:
        kind = message.get("type")
        op_id = str(message.get("id") or "")
        s = self.agent.services
        if kind == "info":
            return self.agent.info().model_dump(mode="json")
        if kind == "have":
            return {"missing": [h for h in message.get("files") or [] if not self.agent.has_file(str(h))]}
        if kind == "put":
            return self._put(message, payload or b"")
        if kind == "open":
            engine = str(message["engine"])
            manifest = s.engines.manifest(engine)
            status = s.engines.status(engine)
            if not status.available:
                raise EnvMissingError(
                    f"{manifest.title} needs {', '.join(status.missing)} on {socket.gethostname()}",
                    {"engine": engine, "missing": status.missing, "hint": f"on the agent: mizuoto doctor {engine}"},
                )
            device = s.compute.resolve_device(list(manifest.devices), str(message.get("device") or "auto"))
            self.engine, self.device = engine, device
            return {"device": device, "status": self._status()}
        if kind == "load":
            model_id = str(message["model_id"])
            info = s.models.require(model_id)
            if info.engine != self.engine:
                raise ValidationError(f"{model_id} is not a {self.engine} model")
            engine = self._engine()
            handle = engine.load(
                model_id,
                info.path,
                str(message.get("variant") or info.variant),
                str(message.get("dtype") or "auto"),
                dict(message.get("options") or {}),
                self._request(op_id, message),
            )
            return {"description": handle.description, "status": self._status()}
        if kind in ("prepare_voice", "synthesize", "convert", "edit", "call"):
            return self._run(op_id, str(kind), self._local(message))
        if kind == "unload":
            model_id = message.get("model_id")
            if model_id is None:
                if self.engine is not None and self.device is not None:
                    for engine in s.runtime.find(self.engine, self.device):
                        engine.shutdown()
            else:
                self._engine().unload(str(model_id))
            return {}
        if kind == "status":
            return {"status": self._status()}
        raise ValidationError(f"Unknown agent request: {kind}")

    def _run(self, op_id: str, kind: str, message: dict[str, Any]) -> dict[str, Any]:
        """``prepare_voice``, ``synthesize``, ``convert``, ``edit`` or ``call`` on a loaded model."""
        handle = self._handle_of(str(message["model_id"]))
        request = self._request(op_id, message)
        result: dict[str, Any] = {}
        if kind == "prepare_voice":
            folder = self.agent.out_dir / new_id("o")
            folder.mkdir(parents=True, exist_ok=True)
            try:
                cache = handle.prepare_voice(voice_from_dict(message.get("voice") or {}), folder, request)
                # The files written go back to the host, named relative to the folder.
                for path in sorted(p for p in folder.rglob("*") if p.is_file()):
                    self.send({"type": "file", "id": op_id, "path": path.relative_to(folder).as_posix()}, path.read_bytes())
                result["cache"] = _relative(asdict(cache), folder) if cache is not None else None
            finally:
                shutil.rmtree(folder, ignore_errors=True)
        elif kind == "call":
            result["result"] = handle.call(str(message["op"]), dict(message.get("payload") or {}), request)
        else:
            task = message.get("task") or {}
            if kind == "synthesize":
                stream = handle.synthesize(synthesis_task_from_dict(task), request)
            elif kind == "convert":
                stream = handle.convert(conversion_task_from_dict(task), request)
            else:
                stream = handle.edit(edit_task_from_dict(task), request)
            with contextlib.closing(stream):
                for chunk in stream:
                    if self.closed.is_set():
                        break  # the host is gone: closing the stream stops the engine
                    header = {"type": "chunk", "id": op_id, "sample_rate": chunk.sample_rate, "segment": chunk.segment, "final": chunk.final}
                    self.send(header, chunk.samples.astype("<f4", copy=False).tobytes())
        return {**result, "applied_params": request.applied, "warnings": request.warnings, "timings": request.timings, "status": self._status()}

    def _put(self, message: dict[str, Any], payload: bytes) -> dict[str, Any] | None:
        sha = str(message["sha256"])
        if len(sha) != 64 or any(c not in "0123456789abcdef" for c in sha):
            raise ValidationError("Not a SHA-256")
        if int(message.get("offset") or 0) == 0:
            if self.upload is not None:
                self.upload[2].close()
                self.upload[1].unlink(missing_ok=True)
            final = self.agent.file_path(sha, str(message.get("name") or "file"))
            final.parent.mkdir(parents=True, exist_ok=True)
            part = final.with_name(final.name + ".part")
            self.upload = (sha, part, part.open("wb"))
        if self.upload is None or self.upload[0] != sha:
            raise ValidationError("An upload's first part must come first")
        self.upload[2].write(payload)
        if not message.get("final"):
            return {}
        _, part, handle = self.upload
        handle.close()
        self.upload = None
        digest = hashlib.sha256()
        with part.open("rb") as f:
            for block in iter(lambda: f.read(1 << 20), b""):
                digest.update(block)
        if digest.hexdigest() != sha:
            part.unlink(missing_ok=True)
            raise ValidationError("The file arrived damaged (SHA-256 differs)")
        part.replace(part.with_name(part.name.removesuffix(".part")))
        return {}

    def _local(self, value: Any) -> Any:
        """The request with its file references turned into this machine's paths."""
        if isinstance(value, str):
            if value.startswith(FILE_REF):
                sha, _, name = value[len(FILE_REF) :].partition("/")
                path = self.agent.resolve_file(sha, name)
                if path is None:
                    raise ValidationError(f"The agent lacks a file the request names ({name})", {"sha256": sha})
                return str(path)
            return value
        if isinstance(value, dict):
            return {k: self._local(v) for k, v in value.items()}
        if isinstance(value, list):
            return [self._local(v) for v in value]
        return value


def _relative(value: Any, folder: Path) -> Any:
    """Paths inside the output folder written as ``mizuoto-out:out/<relative path>``."""
    if isinstance(value, str):
        if value == str(folder) or value.startswith(str(folder) + os.sep):
            rel = Path(value).relative_to(folder).as_posix()
            return f"{OUT_REF}out" + (f"/{rel}" if rel != "." else "")
        return value
    if isinstance(value, dict):
        return {k: _relative(v, folder) for k, v in value.items()}
    if isinstance(value, list):
        return [_relative(v, folder) for v in value]
    return value


def _plain(detail: dict[str, Any]) -> dict[str, Any]:
    """An error's detail as JSON can carry it."""
    import json

    try:
        json.dumps(detail)
        return detail
    except (TypeError, ValueError):
        return {k: str(v) for k, v in detail.items()}
