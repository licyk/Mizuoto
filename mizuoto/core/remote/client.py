"""An engine on another machine: ``RemoteEngine`` is the session with the agent for
one engine on one device, and ``RemoteEngineHandle`` the ``EngineHandle`` of a model loaded there.

The session speaks ``AGENT_PROTOCOL`` 2 at the level of a loaded engine: ``load``, ``prepare_voice``,
``synthesize``, ``convert``, ``edit``, ``call``, ``unload`` and ``status``, with ``text`` (streamed text) and
``cancel`` sent while one runs. The agent answers each with ``progress``, ``chunk`` (binary frames: float32
little-endian mono), ``file`` frames and the ``result`` or ``error``. The files a request names are sent once by
content (the agent keeps them by SHA-256) and the folder a ``prepare_voice`` writes into comes back file by file
into the host's own folder.
"""

from __future__ import annotations

import logging
import queue
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from mizuoto.core import codes
from mizuoto.core.clock import new_id, now_iso
from mizuoto.core.errors import CancelledError, OperationTimeoutError, RemoteError, WorkerError, error_from_code
from mizuoto.core.inference.engine import AudioStream, EngineHandle, EngineRequest
from mizuoto.core.inference.models import EngineRuntime, EngineState, LoadedModelInfo
from mizuoto.core.inference.tasks import AudioChunk, SpeechEditTask, SynthesisTask, VoiceCache, VoiceConversionTask, VoiceSpec, to_dict
from mizuoto.core.jobs.models import ErrorInfo
from mizuoto.core.remote.frames import ConnectionClosed, FrameConnection, FrameError
from mizuoto.core.remote.wire import AGENT_PROTOCOL, CHUNK, FILE_REF, OUT_REF

if TYPE_CHECKING:
    from mizuoto.core.remote.service import RemoteService

logger = logging.getLogger(__name__)


@dataclass
class RemoteConfig:
    engine: str
    device: str
    """As the host names it: ``<host>/<device>``."""
    host: str
    inner_device: str
    """As the agent names it."""
    options: dict[str, Any]
    max_models: int = 3
    """Models of this engine kept loaded on the agent (the host's ``compute.max_resident_models``)."""


class _Pending:
    def __init__(self) -> None:
        self.inbox: queue.Queue[tuple[str, dict[str, Any], bytes | None]] = queue.Queue()


class _Op:
    """One operation on the agent: what it sends, and the outcome as frames arrive."""

    def __init__(self, session: RemoteEngine, kind: str, fields: dict[str, Any], payload: bytes | None, request: EngineRequest | None, timeout: float | None) -> None:
        self.session = session
        self.kind = kind
        self.id = new_id("r")
        self.fields = fields
        self.payload = payload
        self.request = request
        self.timeout = timeout if timeout is not None else (request.timeout if request is not None else None)
        self.pending = _Pending()
        self.cancel_sent = False
        self.started = time.monotonic()

    def frames(self) -> Any:
        """Yield ``(kind, message, payload)`` for each ``progress``, ``chunk`` and ``file`` frame; return the result."""
        s = self.session
        with s._lock:
            s._pending[self.id] = self.pending
        cancel = self.request.cancel if self.request is not None else None
        texts = self.request.text_input if self.request is not None else None
        try:
            s._send({"type": self.kind, "id": self.id, **self.fields}, self.payload)
            while True:
                now = time.monotonic()
                if cancel is not None and cancel.is_set() and not self.cancel_sent:
                    self.cancel_sent = True
                    s._send({"type": "cancel", "id": self.id})
                if self.timeout is not None and now - self.started > self.timeout and not self.cancel_sent:
                    self.cancel_sent = True
                    s._send({"type": "cancel", "id": self.id})
                    raise OperationTimeoutError(f"{self.kind} took longer than {self.timeout:.0f} s on {s.config.host}", {"stage": self.kind})
                if now - s._last_frame > s.remote.heartbeat_timeout:
                    s._close()
                    raise WorkerError(f"{s.config.host} stopped answering", {"engine": s.config.engine, "host": s.config.host, "during": "heartbeat"})
                if texts is not None and not self.cancel_sent:
                    while True:
                        try:
                            text, final = texts.get_nowait()
                        except queue.Empty:
                            break
                        s._send({"type": "text", "id": self.id, "text": text, "final": final})
                try:
                    got, message, data = self.pending.inbox.get(timeout=0.2 if texts is None else 0.02)
                except queue.Empty:
                    continue
                if got == "progress":
                    if self.request is not None and self.request.progress is not None:
                        self.request.progress(str(message.get("stage") or ""), message.get("value"), message.get("message"))
                elif got in ("chunk", "file"):
                    if not (got == "chunk" and self.cancel_sent):
                        yield got, message, data or b""
                elif got == "result":
                    if self.cancel_sent and cancel is not None and cancel.is_set():
                        raise CancelledError("Cancelled")
                    result = dict(message.get("result") or {})
                    if self.request is not None:
                        self.request.applied.update(result.get("applied_params") or {})
                        self.request.warnings.extend(w for w in result.get("warnings") or [] if w not in self.request.warnings)
                        self.request.timings.update(result.get("timings") or {})
                    if result.get("status"):
                        s._status = dict(result["status"])
                    return result
                elif got == "error":
                    code = str(message.get("code") or codes.INTERNAL_ERROR)
                    if code == codes.CANCELLED:
                        raise CancelledError("Cancelled")
                    detail = {**dict(message.get("detail") or {}), "engine": s.config.engine, "device": s.config.device, "host": s.config.host}
                    raise error_from_code(code, str(message.get("message") or "The agent failed"), detail)
                elif got == "closed":
                    raise RemoteError(f"The connection to {s.config.host} was lost", {"engine": s.config.engine, "host": s.config.host})
        finally:
            with s._lock:
                s._pending.pop(self.id, None)

    def run(self, on_file: Callable[[dict[str, Any], bytes], None] | None = None) -> dict[str, Any]:
        """Run to the result, handing ``file`` frames to ``on_file``."""
        frames = self.frames()
        while True:
            try:
                got, message, data = next(frames)
            except StopIteration as stop:
                return dict(stop.value or {})
            if got == "file" and on_file is not None:
                on_file(message, data)


class RemoteEngine:
    """The session with an agent for one engine on one device, and the models it has loaded there."""

    def __init__(self, remote: RemoteService, config: RemoteConfig, on_change: Callable[[], None] | None = None) -> None:
        self.remote = remote
        self.config = config
        self.engine = config.engine
        self.device = config.device
        self.options = config.options
        self._on_change = on_change or (lambda: None)
        self._lock = threading.RLock()
        self._start_lock = threading.Lock()
        self._conn: FrameConnection | None = None
        self._pending: dict[str, _Pending] = {}
        self._loaded: dict[str, float] = {}
        self._descriptions: dict[str, dict[str, Any]] = {}
        self._state: EngineState = "stopped"
        self._current: str | None = None
        self._started_at: str | None = None
        self._last_error: ErrorInfo | None = None
        self._last_frame = time.monotonic()
        self._known: set[str] = set()
        """Files the agent has (SHA-256)."""
        self._status: dict[str, Any] = {}
        """The agent's last report of this engine: ``engine`` (its state there) and ``models``."""
        self._restarts = 0
        self.last_used = time.monotonic()

    @property
    def remote_host(self) -> str:
        return self.config.host

    # -- state ---------------------------------------------------------------------------

    @property
    def state(self) -> EngineState:
        return self._state

    def _set_state(self, state: EngineState) -> None:
        with self._lock:
            changed = self._state != state
            self._state = state
        if changed:
            self._on_change()

    @property
    def busy(self) -> bool:
        return self._current is not None

    def loaded_models(self) -> list[str]:
        with self._lock:
            return list(self._loaded)

    def description(self, model_id: str) -> dict[str, Any] | None:
        return self._descriptions.get(model_id)

    def last_used_at(self, model_id: str) -> float | None:
        return self._loaded.get(model_id)

    def info(self) -> EngineRuntime:
        with self._lock:
            return EngineRuntime(
                engine=self.config.engine,
                device=self.config.device,
                state=self._state,
                current_request=self._current,
                started_at=self._started_at,
                last_error=self._last_error,
                restarts=self._restarts,
                remote=self.config.host,
            )

    def loaded_info(self) -> list[LoadedModelInfo]:
        with self._lock:
            reported = {str(m.get("model_id")): m for m in self._status.get("models") or [] if isinstance(m, dict)}
            return [
                LoadedModelInfo(
                    model_id=m,
                    engine=self.config.engine,
                    device=self.config.device,
                    bytes=reported.get(m, {}).get("bytes"),
                    last_used=t,
                    busy=self._current is not None,
                    remote=self.config.host,
                )
                for m, t in self._loaded.items()
            ]

    # -- connection -----------------------------------------------------------------------

    def start(self) -> None:
        """Open the session (a connection past the handshake, then ``open``)."""
        with self._start_lock:
            if self._conn is not None:
                return
            self._set_state("starting")
            try:
                conn = self.remote.connect(self.config.host)
            except RemoteError as e:
                self._fail(e)
                raise
            self._conn = conn
            self._last_frame = time.monotonic()
            if self._started_at is not None:
                self._restarts += 1
            self._started_at = now_iso()
            threading.Thread(target=self._read, args=(conn,), name=f"remote-{self.config.host}-{self.config.engine}", daemon=True).start()
            try:
                opened = self._op("open", {"engine": self.config.engine, "device": self.config.inner_device}).run()
            except Exception as e:
                self._close()
                if isinstance(e, WorkerError | RemoteError):
                    self._fail(e)
                raise
            self._status = dict(opened.get("status") or {})
            self._set_state("ready")

    def _fail(self, e: Exception) -> None:
        code = getattr(e, "code", codes.WORKER_FAILED)
        self._last_error = ErrorInfo(code=str(code), message=str(getattr(e, "message", e)), detail={})
        self._set_state("crashed")

    def _read(self, conn: FrameConnection) -> None:
        while True:
            try:
                message, payload = conn.recv()
            except (ConnectionClosed, FrameError, OSError, ValueError):
                break
            self._last_frame = time.monotonic()
            kind = str(message.get("type"))
            if kind in ("heartbeat", "pong"):
                continue
            with self._lock:
                pending = self._pending.get(str(message.get("id") or ""))
            if pending is not None:
                pending.inbox.put((kind, message, payload))
        with self._lock:
            lost = self._conn is conn
            if lost:
                self._conn = None
                self._loaded.clear()
                self._descriptions.clear()
            pendings = list(self._pending.values())
        for p in pendings:
            p.inbox.put(("closed", {}, None))
        if lost and self._state not in ("stopped", "stopping"):
            self._last_error = ErrorInfo(code=codes.REMOTE_UNAVAILABLE, message=f"The connection to {self.config.host} was lost", detail={})
            self._set_state("crashed")

    def _close(self) -> None:
        with self._lock:
            conn, self._conn = self._conn, None
            self._loaded.clear()
            self._descriptions.clear()
        if conn is not None:
            conn.close()

    def _send(self, message: dict[str, Any], payload: bytes | None = None) -> None:
        conn = self._conn
        if conn is None:
            raise RemoteError(f"Not connected to {self.config.host}", {"host": self.config.host})
        try:
            conn.send(message, payload)
        except ConnectionClosed as e:
            raise RemoteError(f"The connection to {self.config.host} was lost", {"host": self.config.host}) from e

    def _op(self, kind: str, fields: dict[str, Any], payload: bytes | None = None, request: EngineRequest | None = None, timeout: float | None = None) -> _Op:
        return _Op(self, kind, fields, payload, request, timeout)

    # -- files ------------------------------------------------------------------------------

    def _refs(self, value: Any, uploads: dict[str, Path]) -> Any:
        """``value`` with each existing file path turned into a reference the agent resolves (collected in ``uploads``)."""
        if isinstance(value, str):
            if len(value) < 4096 and Path(value).is_absolute():
                path = Path(value)
                if path.is_file():
                    sha = self.remote.sha256(path)
                    uploads[sha] = path
                    return f"{FILE_REF}{sha}/{path.name}"
            return value
        if isinstance(value, dict):
            return {k: self._refs(v, uploads) for k, v in value.items()}
        if isinstance(value, list):
            return [self._refs(v, uploads) for v in value]
        return value

    def _upload(self, uploads: dict[str, Path]) -> None:
        wanted = {sha: path for sha, path in uploads.items() if sha not in self._known}
        if not wanted:
            return
        missing = set(self._op("have", {"files": sorted(wanted)}).run().get("missing") or [])
        for sha in sorted(missing):
            path = wanted[sha]
            size = path.stat().st_size
            with path.open("rb") as f:
                offset = 0
                while True:
                    block = f.read(CHUNK)
                    final = offset + len(block) >= size
                    self._op("put", {"sha256": sha, "name": path.name, "offset": offset, "final": final}, block, timeout=600).run()
                    offset += len(block)
                    if final:
                        break
        self._known.update(wanted)

    def with_files(self, value: Any) -> Any:
        """``value`` with the files it names on the agent (sent unless it has them)."""
        uploads: dict[str, Path] = {}
        out = self._refs(value, uploads)
        self._upload(uploads)
        return out

    # -- running ----------------------------------------------------------------------------

    def _begin(self, request_id: str) -> None:
        self.start()
        with self._lock:
            self._current = request_id
        self._set_state("busy")
        self.last_used = time.monotonic()

    def _end(self, request_id: str) -> None:
        with self._lock:
            if self._current == request_id:
                self._current = None
        self.last_used = time.monotonic()
        if self._state == "busy":
            self._set_state("ready")

    def touch(self, model_id: str) -> None:
        with self._lock:
            if model_id in self._loaded:
                self._loaded[model_id] = time.time()

    def load(
        self, model_id: str, model_dir: str | None, variant: str, dtype: str = "auto", options: dict[str, Any] | None = None, request: EngineRequest | None = None
    ) -> RemoteEngineHandle:
        """Load ``model_id`` on the agent from its own models folder (``model_dir`` is this machine's)."""
        del model_dir
        self.start()
        with self._lock:
            if model_id in self._loaded:
                self._loaded[model_id] = time.time()
                return RemoteEngineHandle(self, model_id, self._descriptions.get(model_id, {}))
            excess = len(self._loaded) - max(1, self.config.max_models) + 1
            victims = sorted(self._loaded, key=lambda m: self._loaded[m])[: max(0, excess)]
        for victim in victims:
            self.unload(victim)
        request = request or EngineRequest()
        self._begin(request.id)  # busy while it loads: the reaper leaves it alone
        try:
            result = self._op("load", {"model_id": model_id, "variant": variant, "dtype": dtype, "options": {**self.config.options, **(options or {})}}, request=request).run()
        finally:
            self._end(request.id)
        description = dict(result.get("description") or {})
        with self._lock:
            self._loaded[model_id] = time.time()
            self._descriptions[model_id] = description
        self._on_change()
        return RemoteEngineHandle(self, model_id, description)

    def handle(self, model_id: str) -> RemoteEngineHandle | None:
        with self._lock:
            if model_id not in self._loaded:
                return None
            return RemoteEngineHandle(self, model_id, self._descriptions.get(model_id, {}))

    def run(self, kind: str, fields: dict[str, Any], request: EngineRequest | None, out: Path | None = None) -> dict[str, Any]:
        """One operation on a loaded model; ``out``: the folder ``fields["cache_dir"]`` names, which the agent's
        files come back into."""
        request = request or EngineRequest()
        self._begin(request.id)
        try:
            data = self.with_files(fields)
            if out is not None:
                data["cache_dir"] = f"{OUT_REF}out"

            def on_file(info: dict[str, Any], payload: bytes) -> None:
                rel = Path(str(info.get("path") or ""))
                if out is None or rel.is_absolute() or ".." in rel.parts:
                    return
                target = out / rel
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(payload)

            result = self._op(kind, data, request=request).run(on_file)
            return dict(_local(result, out))
        finally:
            self._end(request.id)

    def stream(self, kind: str, fields: dict[str, Any], request: EngineRequest | None) -> AudioStream:
        """The chunks of a ``synthesize``, ``convert`` or ``edit`` on the agent as they arrive."""
        import numpy as np

        request = request or EngineRequest()
        self._begin(request.id)
        op: _Op | None = None
        done = False
        try:
            op = self._op(kind, self.with_files(fields), request=request)
            frames = op.frames()
            first: float | None = None
            started = time.monotonic()
            while True:
                try:
                    got, message, data = next(frames)
                except StopIteration:
                    done = True
                    break
                if got != "chunk":
                    continue
                if first is None:
                    first = time.monotonic() - started
                samples = np.frombuffer(data, dtype="<f4").astype(np.float32)
                yield AudioChunk(samples=samples, sample_rate=int(message["sample_rate"]), segment=int(message.get("segment") or 0), final=bool(message.get("final")))
            if first is not None:
                request.timings.setdefault("first_chunk", first)
        finally:
            if op is not None and not done:
                # Closed before the end (or failed): the agent stops; its answer goes to nobody.
                if not op.cancel_sent and self._conn is not None:
                    try:
                        self._send({"type": "cancel", "id": op.id})
                    except RemoteError:
                        pass
                with self._lock:
                    self._pending.pop(op.id, None)
            self._end(request.id)

    def unload(self, model_id: str) -> None:
        if self._conn is None:
            return
        self._op("unload", {"model_id": model_id}).run()
        with self._lock:
            self._loaded.pop(model_id, None)
            self._descriptions.pop(model_id, None)
        self._on_change()

    def status(self) -> dict[str, Any]:
        """The agent's report of this engine (and refresh it)."""
        self.start()
        self._status = dict(self._op("status", {}).run().get("status") or {})
        return self._status

    def release_memory(self) -> None:
        """Nothing: the agent manages its own memory."""

    def shutdown(self, timeout: float = 5.0) -> None:
        """Unload the engine's models on the agent and close the session."""
        if self._conn is not None:
            self._set_state("stopping")
            try:
                self._op("unload", {"model_id": None}, timeout=timeout).run()
            except Exception as e:
                logger.debug("Stopping %s on %s: %s", self.config.engine, self.config.host, e)
        self._close()
        self._set_state("stopped")


class RemoteEngineHandle(EngineHandle):
    """A model loaded on a remote host's agent."""

    def __init__(self, session: RemoteEngine, model_id: str, description: dict[str, Any]) -> None:
        super().__init__(session.engine, session.device, model_id, description)
        self.remote = session.config.host
        self._session = session

    def prepare_voice(self, voice: VoiceSpec, cache_dir: Path, request: EngineRequest | None = None) -> VoiceCache | None:
        result = self._session.run("prepare_voice", {"model_id": self.model_id, "voice": to_dict(voice)}, request, out=Path(cache_dir))
        cache = result.get("cache")
        return VoiceCache(files=dict(cache.get("files") or {}), meta=dict(cache.get("meta") or {})) if cache else None

    def synthesize(self, task: SynthesisTask, request: EngineRequest | None = None) -> AudioStream:
        return self._session.stream("synthesize", {"model_id": self.model_id, "task": to_dict(task)}, request)

    def convert(self, task: VoiceConversionTask, request: EngineRequest | None = None) -> AudioStream:
        return self._session.stream("convert", {"model_id": self.model_id, "task": to_dict(task)}, request)

    def edit(self, task: SpeechEditTask, request: EngineRequest | None = None) -> AudioStream:
        return self._session.stream("edit", {"model_id": self.model_id, "task": to_dict(task)}, request)

    def call(self, op: str, payload: dict[str, Any] | None = None, request: EngineRequest | None = None) -> dict[str, Any]:
        return dict(self._session.run("call", {"model_id": self.model_id, "op": op, "payload": dict(payload or {})}, request).get("result") or {})


def _local(value: Any, out: Path | None) -> Any:
    """The agent's ``mizuoto-out:out/<path>`` as paths in the host's own folder."""
    if isinstance(value, str) and value.startswith(OUT_REF):
        _, _, rel = value[len(OUT_REF) :].partition("/")
        if out is not None:
            return str(out / rel) if rel else str(out)
        return value
    if isinstance(value, dict):
        return {k: _local(v, out) for k, v in value.items()}
    if isinstance(value, list):
        return [_local(v, out) for v in value]
    return value


__all__ = ["AGENT_PROTOCOL", "RemoteConfig", "RemoteEngine", "RemoteEngineHandle"]
