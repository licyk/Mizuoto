"""The child process of a training stage, in the host's own interpreter:

    python -m mizuoto.training.stage <request.json>

It reads the request ``TrainingService`` wrote, runs the stage (``stage_main``) and reports as a job worker: JSON
events on stdout (``mizuoto.core.jobs.jsonl``), a heartbeat every ``HEARTBEAT_SECONDS``, and an exit code of
``mizuoto.core.codes`` (cancelled, failed). Anything else libraries write to stdout goes to stderr, which the host keeps
as log text. ``--entry module:function`` runs another job function the same way (``function(request, emitter)``).

The stage is its own process, so it may seed the process-wide generators (dropout and data shuffling draw from
them): ``training.seed`` from the request, 1234 by default. The process also gives back the memory after a run and
contains crashes.
"""

from __future__ import annotations

import faulthandler
import importlib
import json
import logging
import os
import signal
import sys
import threading
import traceback
from collections.abc import Callable
from pathlib import Path
from typing import Any

from mizuoto.core import codes
from mizuoto.core.jobs.events import ErrorEvent, HeartbeatEvent, LogEvent, ResultEvent
from mizuoto.core.jobs.jsonl import Emitter

JobMain = Callable[[dict[str, Any], Emitter], None]
DEFAULT_SEED = 1234
HEARTBEAT_SECONDS = 10.0
LOGGERS = ("mizuoto",)
"""Loggers whose records become log events."""


def seed_everything(seed: int) -> None:
    """The process-wide generators a training loop draws from: Python's, NumPy's and torch's (every device)."""
    import random

    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed % 2**32)
    torch.manual_seed(seed)


def stage_main(request: dict[str, Any], emitter: Emitter) -> None:
    """Run one stage of a native trainer and report its result."""
    from mizuoto.training.context import StageContext
    from mizuoto.training.trainer import load_trainer

    seed_everything(int(request.get("seed") or DEFAULT_SEED))
    ctx = StageContext(emitter, request)
    if request.get("fast"):
        from mizuoto.runtime.accel import features

        done = features.apply(list(request["fast"]))
        if "fp16_accumulation" in done:
            ctx.log("compute.fast fp16_accumulation: training may be numerically unstable (NaN losses)", "warning")
    if ctx.attention != "auto":
        ctx.use_attention(ctx.attention)
    trainer = load_trainer(str(request["trainer"]))
    result = trainer.run_stage(str(request["stage"]), ctx)
    if ctx.accel is not None:
        ctx.log(f"attention: {json.dumps(ctx.accel.report(), ensure_ascii=False)}")
        for warning in ctx.accel.warnings():
            ctx.log(warning, "warning")
    json.dumps(result)  # the result must travel as JSON
    emitter.emit(ResultEvent(data=result))


class _LogHandler(logging.Handler):
    def __init__(self, emitter: Emitter) -> None:
        super().__init__()
        self.emitter = emitter

    def emit(self, record: logging.LogRecord) -> None:
        try:
            self.emitter.emit(LogEvent(level=record.levelname.lower(), message=self.format(record)))
        except Exception:
            pass


def run_job(main: JobMain, request_path: str) -> int:
    """Run ``main`` on the request file and map its failures to events and exit codes."""
    from mizuoto.runtime.context import Cancelled
    from mizuoto.runtime.device import is_oom
    from mizuoto.runtime.engine import EngineError

    real_stdout = os.fdopen(os.dup(sys.stdout.fileno()), "w", encoding="utf-8", buffering=1)
    os.dup2(sys.stderr.fileno(), sys.stdout.fileno())
    faulthandler.enable()
    emitter = Emitter(real_stdout)
    handler = _LogHandler(emitter)
    handler.setFormatter(logging.Formatter("%(message)s"))
    for name in LOGGERS:
        log = logging.getLogger(name)
        log.addHandler(handler)
        log.setLevel(logging.INFO)
    if hasattr(signal, "SIGTERM"):
        signal.signal(signal.SIGTERM, lambda *_: sys.exit(codes.EXIT_CANCELLED))
    stop = threading.Event()

    def beat() -> None:
        while not stop.wait(HEARTBEAT_SECONDS):
            emitter.emit(HeartbeatEvent())

    threading.Thread(target=beat, daemon=True).start()
    try:
        request = json.loads(Path(request_path).read_text(encoding="utf-8"))
        main(request, emitter)
        return codes.EXIT_OK
    except (Cancelled, KeyboardInterrupt):
        return codes.EXIT_CANCELLED
    except EngineError as e:
        emitter.emit(ErrorEvent(code=e.code, message=e.message, detail=e.detail))
        return codes.EXIT_FAILED
    except Exception as e:
        emitter.emit(LogEvent(level="error", message=traceback.format_exc()))
        if is_oom(e):
            emitter.emit(ErrorEvent(code=codes.COMPUTE_UNAVAILABLE, message=f"Out of memory: {e}", detail={"reason": codes.OUT_OF_MEMORY}))
        else:
            emitter.emit(ErrorEvent(code=codes.ENGINE_ERROR, message=str(e) or type(e).__name__, detail={"exception": type(e).__name__}))
        return codes.EXIT_FAILED
    finally:
        stop.set()
        try:
            real_stdout.flush()
        except Exception:
            pass


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    if len(args) == 1:
        return run_job(stage_main, args[0])
    if len(args) == 3 and args[0] == "--entry":
        module_name, _, func_name = args[1].partition(":")
        return run_job(getattr(importlib.import_module(module_name), func_name), args[2])
    print("usage: python -m mizuoto.training.stage [--entry module:function] <request.json>", file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main())
