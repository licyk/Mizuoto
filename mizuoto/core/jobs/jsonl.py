"""Reading and writing job-worker events as JSON lines. Standard library only."""

from __future__ import annotations

import json
import sys
import threading
from typing import IO, Any

from mizuoto.core.jobs.events import LogEvent, WorkerEvent, from_dict, to_dict

PREFIX = "@mizuoto "
"""Marks an event line, so stray prints from libraries are kept as log text, not parsed."""


class Emitter:
    """Write events to a stream (stdout by default), one per line, safe across threads."""

    def __init__(self, stream: IO[str] | None = None) -> None:
        self._stream = stream or sys.stdout
        self._lock = threading.Lock()

    def emit(self, event: WorkerEvent) -> None:
        line = PREFIX + json.dumps(to_dict(event), ensure_ascii=False, default=str)
        with self._lock:
            self._stream.write(line + "\n")
            self._stream.flush()


def parse_line(line: str) -> WorkerEvent:
    """An event line becomes its event; any other line becomes an info log line."""
    text = line.rstrip("\r\n")
    if text.startswith(PREFIX):
        try:
            data: Any = json.loads(text[len(PREFIX) :])
        except ValueError:
            data = None
        if isinstance(data, dict):
            event = from_dict(data)
            if event is not None:
                return event
    return LogEvent(level="info", message=text)


def parse_lines(line: str) -> list[WorkerEvent]:
    """A line of a worker's output as its events: stdout (events) and stderr (libraries) share one pipe,
    so an event can follow another program's unfinished line (a progress bar redrawn with ``\r``)."""
    text = line.rstrip("\r\n")
    at = text.find(PREFIX)
    if at <= 0:
        return [parse_line(text)]
    before = text[:at]
    events: list[WorkerEvent] = [LogEvent(level="info", message=before)] if before.strip() else []
    return [*events, parse_line(text[at:])]
