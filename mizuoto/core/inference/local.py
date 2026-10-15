"""Engines in this process: ``LocalEngine`` holds one engine's models on one
device, and ``LocalHandle`` is the ``EngineHandle`` of one of them.

Every call runs in the calling thread (a job's). Around it the engine gets its ``RequestContext`` (the request's
generators, progress, streamed text, a scratch folder), bound to the thread so layers can check for cancellation,
and ``torch.inference_mode()`` (loading runs outside it: weights made inside it are inference tensors, which take
other CPU kernels than ordinary parameters on strided inputs, and upstream loads outside it). A small watcher
thread passes the caller's cancel on and notices the timeout. Errors become the core's: an engine's code is kept,
running out of memory frees what is idle and fails with ``compute_unavailable``, a cancel is ``cancelled``.
"""

from __future__ import annotations

import contextlib
import logging
import sys
import threading
import time
import traceback
from collections.abc import Callable, Generator, Iterator
from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING, Any, TypeVar

from mizuoto.core import codes
from mizuoto.core.clock import now_iso
from mizuoto.core.errors import CancelledError, MizuotoError, OperationTimeoutError, error_from_code
from mizuoto.core.inference.engine import SNAPSHOT_OP, AudioStream, EngineHandle, EngineRequest, seed_of
from mizuoto.core.inference.models import EngineRuntime, EngineState, LoadedModelInfo
from mizuoto.core.inference.tasks import AudioChunk, SpeechEditTask, SynthesisTask, VoiceCache, VoiceConversionTask, VoiceSpec
from mizuoto.core.jobs.models import ErrorInfo

if TYPE_CHECKING:
    from mizuoto.core.inference.service import InferenceService
    from mizuoto.runtime.context import RequestContext
    from mizuoto.runtime.engine import Engine, LoadedEngine

logger = logging.getLogger(__name__)

T = TypeVar("T")
WATCH_INTERVAL = 0.02


class _Loaded:
    def __init__(self, engine: LoadedEngine, description: dict[str, Any]) -> None:
        self.engine = engine
        self.description = description
        self.last_used = time.time()
        self.attention = "auto"
        """``compute.attention`` (or the engine's) when it loaded."""
        self.accel: Any = None
        """Its attention policy (``mizuoto.runtime.accel.policy.AccelPolicy``), None for ``auto``."""


class LocalEngine:
    """One engine on one device in this process, and the models it has loaded."""

    remote: str | None = None

    def __init__(self, engine: str, device: str, scratch_dir: Path, options: dict[str, Any], inference: InferenceService, on_change: Callable[[], None] | None = None) -> None:
        self.engine = engine
        self.device = device
        self.scratch_dir = scratch_dir
        self.options = options
        """The engine's load options from its settings."""
        self._inference = inference
        self._on_change = on_change or (lambda: None)
        self._state: EngineState = "stopped"
        self._loaded: dict[str, _Loaded] = {}
        self._current: str | None = None
        self._current_model: str | None = None
        self._started_at: str | None = None
        self._last_error: ErrorInfo | None = None
        self._lock = threading.RLock()
        self._impl: Engine | None = None
        self.last_used = time.monotonic()
        self.model_options: Callable[[str], dict[str, Any]] | None = None
        """Options every load of a model gets besides the given ones (its shared assets' folders)."""
        self.make_room: Callable[[str], None] | None = None
        """Called before a model loads (with its id): the runtime service unloads others beyond ``compute.max_resident_models``."""
        self.vram_mode: Callable[[], str | None] = lambda: None
        """The engine's ``vram_mode`` override (its settings)."""
        self.pinned: Callable[[str], bool] = lambda _model_id: False
        self.attention: Callable[[], tuple[str, bool]] = lambda: ("auto", False)
        """``(backend, strict)``: ``compute.attention`` or the engine's own, and ``compute.attention_strict``."""
        """Whether a model is in the engine's ``pinned_models`` (its components give up memory last)."""

    # -- state ------------------------------------------------------------------------

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
        loaded = self._loaded.get(model_id)
        return loaded.description if loaded else None

    def last_used_at(self, model_id: str) -> float | None:
        loaded = self._loaded.get(model_id)
        return loaded.last_used if loaded else None

    def info(self) -> EngineRuntime:
        with self._lock:
            return EngineRuntime(engine=self.engine, device=self.device, state=self._state, current_request=self._current, started_at=self._started_at, last_error=self._last_error)

    def loaded_info(self) -> list[LoadedModelInfo]:
        manager = self._inference.runtime.manager if self._inference.started else None
        with self._lock:
            return [
                LoadedModelInfo(
                    model_id=m,
                    engine=self.engine,
                    device=self.device,
                    bytes=manager.owner_bytes(self._owner(m)) if manager is not None else None,
                    loaded_bytes=(usage := manager.owner_usage(self._owner(m)) if manager is not None else {}).get("loaded"),
                    streamed_bytes=usage.get("streamed"),
                    host_bytes=usage.get("host"),
                    last_used=x.last_used,
                    attention=x.attention,
                    busy=self._current is not None and self._current_model in (None, m),
                )
                for m, x in self._loaded.items()
            ]

    def _owner(self, model_id: str) -> str:
        return f"{model_id}@{self.device}"

    # -- loading ----------------------------------------------------------------------

    def start(self) -> None:
        """Find the engine's implementation (imports its module, not torch)."""
        if self._impl is not None and self._state != "stopped":
            return
        from mizuoto.runtime.engine import get_engine

        try:
            self._impl = get_engine(self.engine)
        except Exception as e:
            err = self._error(e)
            self._last_error = ErrorInfo(code=err.code, message=err.message, detail=err.detail)
            self._set_state("crashed")
            raise err from e
        self._started_at = now_iso()
        self._set_state("ready")

    def load(
        self, model_id: str, model_dir: str | None, variant: str, dtype: str = "auto", options: dict[str, Any] | None = None, request: EngineRequest | None = None
    ) -> LocalHandle:
        """The handle of ``model_id``, loading it unless it is loaded (the runtime service first unloads others beyond
        its limit: ``make_room``)."""
        from mizuoto.runtime.engine import LoadSpec

        self.start()
        with self._lock:
            loaded = self._loaded.get(model_id)
            if loaded is not None:
                loaded.last_used = time.time()
                return LocalHandle(self, model_id, loaded)
        if self.make_room is not None:
            self.make_room(model_id)
        merged = {**(options or {}), **(self.model_options(model_id) if self.model_options is not None else {})}
        spec = LoadSpec(model_id, Path(model_dir) if model_dir else None, variant, self.device, dtype, merged)
        runtime = self._inference.runtime
        impl = self._impl
        assert impl is not None
        # Outside inference mode: see the module's docstring.
        with self._request(request or EngineRequest(), None, model_id) as ctx, ctx.bind():
            engine = impl.load(spec, runtime, ctx)
            runtime.manager.register(self._owner(model_id), engine.handles, mode=self.vram_mode(), pinned=self.pinned(model_id))
            loaded = _Loaded(engine, asdict(engine.describe()))
        loaded.attention, strict = self.attention()
        if loaded.attention != "auto":
            from mizuoto.runtime.accel.policy import policy_for

            loaded.accel = policy_for(loaded.attention, strict)
        with self._lock:
            self._loaded[model_id] = loaded
        self._on_change()
        return LocalHandle(self, model_id, loaded)

    def handle(self, model_id: str) -> LocalHandle | None:
        """The handle of a loaded model, or None."""
        with self._lock:
            loaded = self._loaded.get(model_id)
        return LocalHandle(self, model_id, loaded) if loaded is not None else None

    def touch(self, model_id: str) -> None:
        with self._lock:
            if model_id in self._loaded:
                self._loaded[model_id].last_used = time.time()

    # -- running ----------------------------------------------------------------------

    def _require(self, model_id: str, loaded: _Loaded) -> LoadedEngine:
        with self._lock:
            if self._loaded.get(model_id) is not loaded:
                raise error_from_code(codes.INVALID_INPUT, f"Model {model_id} is not loaded", {"engine": self.engine})
        loaded.last_used = time.time()
        return loaded.engine

    @contextlib.contextmanager
    def _request(self, request: EngineRequest, seed: int | None, model_id: str | None, loaded: _Loaded | None = None) -> Generator[RequestContext, None, None]:
        """The engine's context for one call: the engine is busy, the caller's cancel and the timeout reach it, and
        what it raises becomes the core's error. Nothing here is bound to the thread, so a stream may hold it
        across its chunks. ``loaded``: the model's attention policy runs the request (and is reported)."""
        from mizuoto.runtime.context import Cancelled, RequestContext

        stop = threading.Event()
        cancel_event = threading.Event()
        timed_out = threading.Event()
        ctx = RequestContext(
            request.id,
            device=self.device,
            seed=seed,
            log=logging.getLogger(f"mizuoto.engines.{self.engine}"),
            progress=request.progress,
            cancel=cancel_event,
            scratch_root=self.scratch_dir,
            text_input=request.text_input,
        )
        if loaded is not None and loaded.accel is not None:
            ctx.accel = loaded.accel.run()
        watcher = threading.Thread(target=_watch, args=(stop, request.cancel, cancel_event, request.timeout, timed_out), name=f"watch-{request.id}", daemon=True)
        with self._lock:
            self._current = request.id
            self._current_model = model_id
        self._set_state("busy")
        self.last_used = time.monotonic()
        started = time.perf_counter()
        streamed = self._inference.runtime.manager.streamed_bytes(self.device) if self._inference.started else 0
        watcher.start()
        try:
            yield ctx
            if cancel_event.is_set():
                raise Cancelled()
        except Cancelled:
            if timed_out.is_set():
                raise OperationTimeoutError(f"The request took longer than {request.timeout:.0f} s", {"engine": self.engine, "stage": "request"}) from None
            raise CancelledError("Cancelled") from None
        except MizuotoError:
            raise
        except Exception as e:
            raise self._error(e) from e
        finally:
            stop.set()
            watcher.join(timeout=1)
            ctx.close()
            if model_id is not None and self._inference.started:
                manager = self._inference.runtime.manager
                usage = manager.owner_usage(self._owner(model_id))
                request.memory.update(
                    {
                        "streamed_mb": round((manager.streamed_bytes(self.device) - streamed) / 2**20, 1),
                        "loaded_mb": round(usage["loaded"] / 2**20, 1),
                        "size_mb": round(manager.owner_bytes(self._owner(model_id)) / 2**20, 1),
                    }
                )
                partial = manager.owner_placements(self._owner(model_id))
                if partial:
                    request.memory["partial"] = partial
            request.applied.update(ctx.applied)
            request.warnings.extend(w for w in ctx.warnings if w not in request.warnings)
            if self._inference.fast:
                request.applied["fast"] = list(self._inference.fast)
            if ctx.accel is not None:
                request.applied["attention"] = ctx.accel.report()
                request.warnings.extend(w for w in ctx.accel.warnings() if w not in request.warnings)
            request.timings["total"] = time.perf_counter() - started
            with self._lock:
                self._current = None
                self._current_model = None
            self.last_used = time.monotonic()
            if self._state == "busy":
                self._set_state("ready")

    def run(self, model_id: str, loaded: _Loaded, fn: Callable[[LoadedEngine, RequestContext], T], request: EngineRequest | None, seed: int | None = None, task: Any = None) -> T:
        """``fn(engine, ctx)`` on the model, its components on their device."""
        engine = self._require(model_id, loaded)
        with self._request(request or EngineRequest(), seed, model_id, loaded) as ctx, ctx.bind(), _inference_mode(), self._place(engine, task, ctx):
            return fn(engine, ctx)

    def stream(
        self, model_id: str, loaded: _Loaded, start: Callable[[LoadedEngine, RequestContext], Iterator[Any]], request: EngineRequest | None, seed: int | None, task: Any = None
    ) -> AudioStream:
        """The engine's chunks as float32 numpy arrays, one at a time. The engine runs only while the caller asks for
        the next chunk; bound to the thread and in inference mode then, and only then."""
        import numpy as np

        from mizuoto.runtime.engine import EngineError

        engine = self._require(model_id, loaded)
        request = request or EngineRequest()
        with self._request(request, seed, model_id, loaded) as ctx, contextlib.ExitStack() as stack:
            started = time.perf_counter()
            first: float | None = None
            with ctx.bind(), _inference_mode():
                stack.enter_context(self._place(engine, task, ctx))
                chunks = iter(start(engine, ctx))
            try:
                while True:
                    with ctx.bind(), _inference_mode():
                        try:
                            chunk = next(chunks)
                        except StopIteration:
                            break
                        ctx.check_cancel()
                    if not isinstance(chunk, AudioChunk):
                        raise EngineError(f"The engine yielded {type(chunk).__name__}, not AudioChunk")
                    if first is None:
                        first = time.perf_counter() - started
                        request.timings["first_chunk"] = first
                    samples = np.asarray(chunk.samples).reshape(-1).astype(np.float32)
                    yield AudioChunk(samples=samples, sample_rate=chunk.sample_rate, segment=chunk.segment, final=chunk.final)
            finally:
                close = getattr(chunks, "close", None)
                if close is not None:
                    with ctx.bind(), _inference_mode():
                        close()

    def _place(self, engine: LoadedEngine, task: Any, ctx: RequestContext) -> contextlib.AbstractContextManager[Any]:
        """The model's components for a request: the engine places them per phase (``PHASED``), else all of them now."""
        manager = self._inference.runtime.manager
        ctx.manager = manager
        if type(engine).PHASED:
            return contextlib.nullcontext()
        return manager.use(engine.handles, _memory(engine, task))

    def snapshot(self) -> dict[str, Any]:
        from mizuoto.runtime.snapshot import snapshot

        return {"snapshot": snapshot()}

    def _error(self, e: BaseException) -> MizuotoError:
        from mizuoto.runtime.device import is_oom
        from mizuoto.runtime.engine import EngineError

        where = {"engine": self.engine, "device": self.device}
        if isinstance(e, EngineError):
            return error_from_code(e.code, e.message, {**e.detail, **where})
        if is_oom(e):
            self._inference.runtime.manager.release_idle(self.device)
            return error_from_code(codes.COMPUTE_UNAVAILABLE, f"Out of memory: {e}", {"reason": codes.OUT_OF_MEMORY, **where})
        trace = traceback.format_exc()
        logger.error("%s", trace)
        return error_from_code(codes.ENGINE_ERROR, str(e) or type(e).__name__, {"exception": type(e).__name__, "traceback": trace, **where})

    # -- unloading --------------------------------------------------------------------

    def unload(self, model_id: str) -> None:
        with self._lock:
            loaded = self._loaded.pop(model_id, None)
        if loaded is None:
            return
        self._inference.runtime.manager.unregister(self._owner(model_id))
        try:
            loaded.engine.close()
        finally:
            self._on_change()

    def release_memory(self) -> None:
        if not self.busy:
            for model_id in self.loaded_models():
                self.unload(model_id)
            self._inference.runtime.manager.release_idle(self.device)

    def shutdown(self) -> None:
        """Unload every model."""
        for model_id in self.loaded_models():
            try:
                self.unload(model_id)
            except Exception:
                logger.exception("Unloading %s failed", model_id)
        self._set_state("stopped")


class LocalHandle(EngineHandle):
    """A model loaded in this process."""

    def __init__(self, owner: LocalEngine, model_id: str, loaded: _Loaded) -> None:
        super().__init__(owner.engine, owner.device, model_id, loaded.description)
        self._owner = owner
        self._loaded = loaded

    def prepare_voice(self, voice: VoiceSpec, cache_dir: Path, request: EngineRequest | None = None) -> VoiceCache | None:
        return self._owner.run(self.model_id, self._loaded, lambda engine, ctx: engine.prepare_voice(voice, cache_dir, ctx), request, task=voice)

    def synthesize(self, task: SynthesisTask, request: EngineRequest | None = None) -> AudioStream:
        return self._owner.stream(self.model_id, self._loaded, lambda engine, ctx: engine.synthesize(task, ctx), request, seed_of(task.params), task)

    def convert(self, task: VoiceConversionTask, request: EngineRequest | None = None) -> AudioStream:
        return self._owner.stream(self.model_id, self._loaded, lambda engine, ctx: engine.convert(task, ctx), request, seed_of(task.params), task)

    def edit(self, task: SpeechEditTask, request: EngineRequest | None = None) -> AudioStream:
        return self._owner.stream(self.model_id, self._loaded, lambda engine, ctx: engine.edit(task, ctx), request, seed_of(task.params), task)

    def call(self, op: str, payload: dict[str, Any] | None = None, request: EngineRequest | None = None) -> dict[str, Any]:
        if op == SNAPSHOT_OP:
            return self._owner.snapshot()
        return self._owner.run(self.model_id, self._loaded, lambda engine, ctx: engine.call(op, dict(payload or {}), ctx), request)


def _watch(stop: threading.Event, cancel: threading.Event | None, cancel_event: threading.Event, timeout: float | None, timed_out: threading.Event) -> None:
    started = time.monotonic()
    while not stop.wait(WATCH_INTERVAL):
        if cancel is not None and cancel.is_set():
            cancel_event.set()
        if timeout is not None and time.monotonic() - started > timeout and not cancel_event.is_set():
            timed_out.set()
            cancel_event.set()


def _memory(engine: LoadedEngine, task: Any) -> int:
    """The working memory ``engine`` estimates for ``task`` (a synthesis, conversion, edit or voice), 0 without one."""
    return engine.memory_required(task) if task is not None else 0


def _inference_mode() -> contextlib.AbstractContextManager[Any]:
    """``torch.inference_mode()`` when an engine has loaded torch; nothing otherwise (the dummy engine)."""
    torch = sys.modules.get("torch")
    return torch.inference_mode() if torch is not None else contextlib.nullcontext()
