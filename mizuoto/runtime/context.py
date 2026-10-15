"""The request context native engines receive.

One ``RequestContext`` per request. It carries:

- **Cancellation:** a cooperative cancel event. ``check_cancel()`` raises ``Cancelled``, a
  ``BaseException``, so model code's ``except Exception`` cannot swallow it.
- **Randomness:** generators seeded from the request's seed. ``generator_for(device)`` gives a
  ``torch.Generator``, and ``rng`` a ``random.Random``. Engines never touch the global RNGs. A fresh
  generator seeded with *s* yields the same stream as torch's default generator after
  ``torch.manual_seed(s)``, on the CPU and on CUDA (``scripts/parity/check_rng.py``). So a native engine
  making upstream's draws in upstream's order reproduces a seeded upstream run.
- **Reporting:** progress, the parameters the engine applied and its warnings.
- **Streamed text:** text pieces as the client sends them (``SynthesisTask.text_stream``).
- **Scratch:** a scratch folder removed when the request ends.

The context is also bound to the running thread while an engine works (``bind``), so layers can check
for cancellation without being handed it (``check_cancel()`` at module level).
"""

from __future__ import annotations

import contextvars
import logging
import queue
import random
import shutil
import tempfile
import threading
from collections.abc import Callable, Generator, Iterable
from contextlib import contextmanager
from pathlib import Path
from typing import Any, TypeVar

T = TypeVar("T")

ProgressFn = Callable[[str, "float | None", "str | None"], None]

_current: contextvars.ContextVar[RequestContext | None] = contextvars.ContextVar("mizuoto_request", default=None)
accel_policy: contextvars.ContextVar[Any] = contextvars.ContextVar("mizuoto_accel_policy", default=None)
"""The attention policy of the running request or stage (a ``mizuoto.runtime.accel.policy.Run``; None: every site's
reference). Here, not in ``mizuoto.runtime.accel``, so that binding a request loads no torch."""


OOM_RETRIES = 2
"""Times ``RequestContext.run`` runs a phase again after it runs out of device memory."""


class Cancelled(BaseException):
    """The request was cancelled."""


def current() -> RequestContext | None:
    """The request the running thread works for, if any."""
    return _current.get()


def check_cancel() -> None:
    """Raise ``Cancelled`` when the running thread's request was cancelled."""
    ctx = _current.get()
    if ctx is not None and ctx.cancel_event.is_set():
        raise Cancelled()


class RequestContext:
    def __init__(
        self,
        request_id: str,
        *,
        device: str = "cpu",
        seed: int | None = None,
        log: logging.Logger | None = None,
        progress: ProgressFn | None = None,
        cancel: threading.Event | None = None,
        scratch_root: Path | None = None,
        text_input: queue.Queue[tuple[str, bool]] | None = None,
    ) -> None:
        self.request_id = request_id
        self.device = device
        self.log = log or logging.getLogger("mizuoto.engine")
        self.cancel_event = cancel or threading.Event()
        self.applied: dict[str, Any] = {}
        """Parameter values the engine actually used, reported with the result."""
        self.warnings: list[str] = []
        self._progress = progress
        self._scratch_root = scratch_root
        self._tmpdir: Path | None = None
        self._text_input = text_input
        self._seed: int | None = None
        self._generators: dict[str, Any] = {}
        self.rng = random.Random()
        self.reseed(seed)
        self.manager: Any = None
        """The model manager that places the engine's components for ``use`` and ``run`` (None: they are where they are)."""
        self.accel: Any = None
        """The attention policy of the request (``mizuoto.runtime.accel.policy.Run``), bound with the request."""

    # -- randomness -------------------------------------------------------------------

    @property
    def seed(self) -> int | None:
        return self._seed

    def reseed(self, seed: int | None) -> None:
        """Start the request's random streams again from ``seed`` (``None``: fresh entropy)."""
        self._seed = None if seed is None else int(seed)
        self._generators.clear()
        self.rng = random.Random(self._seed)

    def generator_for(self, device: Any = None) -> Any:
        """The request's ``torch.Generator`` on ``device`` (default: the request's device).

        One per device type and index, created on first use and seeded with the request's seed.
        Without a seed, each is seeded from fresh entropy.
        """
        import torch

        dev = torch.device(device if device is not None else self.device)
        if dev.type == "cuda" and dev.index is None:
            dev = torch.device("cuda", torch.cuda.current_device())
        elif dev.type == "xpu" and dev.index is None:
            dev = torch.device("xpu", torch.xpu.current_device())
        key = str(dev)
        gen = self._generators.get(key)
        if gen is None:
            gen = torch.Generator(device=dev)
            if self._seed is None:
                gen.seed()
            else:
                gen.manual_seed(self._seed)
            self._generators[key] = gen
        return gen

    @property
    def generator(self) -> Any:
        """``generator_for(self.device)``."""
        return self.generator_for(None)

    # -- memory -----------------------------------------------------------------------

    @contextmanager
    def use(self, handles: Iterable[Any], memory: int = 0) -> Generator[None, None, None]:
        """One phase of the request: ``handles`` on their device (whole, or partly with the rest
        streaming) with room for ``memory`` bytes of working memory, kept there for the block. The engine's other
        components may give up their memory meanwhile."""
        if self.manager is None:
            yield
            return
        with self.manager.use(list(handles), int(memory)):
            yield

    def run(
        self, name: str, handles: Iterable[Any], fn: Callable[[], T], memory: int = 0, generators: Iterable[Any] = (), fallback: tuple[str, Callable[[], T]] | None = None
    ) -> T:
        """``fn()`` as a phase (``use``), for a block whose only effect is its result.

        When it runs out of device memory, it runs again, at most ``OOM_RETRIES`` times, with
        others' idle components off the device and half the weights it had there left in RAM to stream; the request's
        generators (and ``generators``, the engine's own) are put back as they were before the
        first try, so the same draws give the same result, only slower. The recovery happens once the ``except``
        block is left: its traceback holds the failed attempt's tensors (ComfyUI's lesson).

        ``fallback`` is (what it does, a cheaper way to the same result: smaller batches, tiles), run after the retries
        when ``compute.oom_fallback`` allows it; its output differs slightly, so it is reported (``applied`` and a
        warning)."""
        handles = list(handles)
        extra = list(generators)
        tries = 0
        while True:
            saved = self._save_random(extra)
            try:
                with self.use(handles, memory):
                    return fn()
            except Exception as error:
                if not self._retry_oom(error, tries):
                    if fallback is None or not self._fallback_oom(error):
                        raise
                    use_fallback = True
                else:
                    use_fallback = False
            if use_fallback and fallback is not None:
                self.manager.after_oom(handles)
                self._restore_random(saved, extra)
                what, cheaper = fallback
                self.applied.setdefault("fallbacks", {})[name] = what
                self.warn(f"{name} ran out of device memory even with its weights in RAM, so it ran again {what}; the output differs slightly")
                with self.use(handles, memory):
                    return cheaper()
            tries += 1
            self.manager.after_oom(handles)
            self._restore_random(saved, extra)
            retries = self.applied.setdefault("oom_retries", {})
            retries[name] = tries
            self.warn(f"{name} ran out of device memory and was run again with more of its weights in RAM")
            self.log.warning("%s ran out of device memory: try %d with more of its weights in RAM", name, tries + 1)

    def _fallback_oom(self, error: BaseException) -> bool:
        from mizuoto.runtime.device import is_oom

        return self.manager is not None and is_oom(error) and bool(self.manager.policy.oom_fallback)

    def _retry_oom(self, error: BaseException, tries: int) -> bool:
        from mizuoto.runtime.device import is_oom

        if self.manager is None or tries >= OOM_RETRIES or not is_oom(error):
            return False
        return bool(self.manager.policy.oom_retry)

    def _save_random(self, extra: list[Any]) -> tuple[dict[str, Any], Any, list[Any]]:
        return {k: g.get_state() for k, g in self._generators.items()}, self.rng.getstate(), [g.get_state() for g in extra]

    def _restore_random(self, saved: tuple[dict[str, Any], Any, list[Any]], extra: list[Any]) -> None:
        states, rng, others = saved
        for key in list(self._generators):
            if key in states:
                self._generators[key].set_state(states[key])
            else:  # made during the failed try: made again, seeded afresh, on the next
                del self._generators[key]
        self.rng.setstate(rng)
        for g, state in zip(extra, others, strict=True):
            g.set_state(state)

    # -- cancellation -----------------------------------------------------------------

    def cancelled(self) -> bool:
        return self.cancel_event.is_set()

    def check_cancel(self) -> None:
        if self.cancel_event.is_set():
            raise Cancelled()

    def cancel(self) -> None:
        self.cancel_event.set()

    @contextmanager
    def bind(self) -> Generator[RequestContext, None, None]:
        """Make this the running thread's request for the block (``current()``, ``check_cancel()``), and its attention
        policy the context's."""
        token = _current.set(self)
        policy = accel_policy.set(self.accel)
        try:
            yield self
        finally:
            accel_policy.reset(policy)
            _current.reset(token)

    # -- reporting --------------------------------------------------------------------

    def progress(self, stage: str, value: float | None = None, message: str | None = None) -> None:
        if self._progress is not None:
            self._progress(stage, value, message)

    def warn(self, message: str) -> None:
        if message not in self.warnings:
            self.warnings.append(message)

    # -- streamed text ----------------------------------------------------------------

    def text_pieces(self) -> Generator[str, None, None]:
        """The request's text as the client sends it (``SynthesisTask.text_stream``), up to its last piece.

        A real generator (CosyVoice tests ``isinstance(text, Generator)``), safe to consume on another
        thread. It ends early when the request is cancelled.
        """
        if self._text_input is None:
            return
        while True:
            try:
                text, final = self._text_input.get(timeout=0.1)
            except queue.Empty:
                if self.cancelled():
                    return
                continue
            if text:
                yield text
            if final:
                return

    # -- scratch ----------------------------------------------------------------------

    @property
    def tmpdir(self) -> Path:
        """A scratch folder removed when the request ends."""
        if self._tmpdir is None:
            if self._scratch_root is not None:
                self._scratch_root.mkdir(parents=True, exist_ok=True)
            self._tmpdir = Path(tempfile.mkdtemp(prefix=f"{self.request_id}-", dir=self._scratch_root))
        return self._tmpdir

    def close(self) -> None:
        if self._tmpdir is not None:
            shutil.rmtree(self._tmpdir, ignore_errors=True)
            self._tmpdir = None
