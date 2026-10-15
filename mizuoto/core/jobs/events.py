"""Events of job workers (training stages), one JSON object per line on stdout. Standard library only.

A job worker receives one JSON request file (its path is the argument) and reports with these
events, written by ``jsonl.Emitter``; the host reads them with ``jsonl.parse_lines`` (``JobContext.run_worker``).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
from typing import Any


@dataclass
class ProgressEvent:
    """Overall progress 0..1, or None while indeterminate; ``step`` names the current step."""

    value: float | None
    step: str | None = None
    message: str | None = None
    type: str = "progress"


@dataclass
class StepEvent:
    """A step changed state (pending, running, done, skipped, failed, cancelled)."""

    id: str
    state: str
    progress: float | None = None
    title: str | None = None
    type: str = "step"


@dataclass
class LogEvent:
    level: str
    """debug, info, warning or error."""
    message: str
    type: str = "log"


@dataclass
class MetricEvent:
    """A training measurement at an epoch and step."""

    epoch: int
    step: int
    values: dict[str, float]
    lr: float | None = None
    type: str = "metric"


@dataclass
class OutputEvent:
    """A file the worker finished writing (a preview sample, a checkpoint)."""

    path: str
    label: str
    kind: str = "file"
    source: str | None = None
    type: str = "output"


@dataclass
class HeartbeatEvent:
    """Written at least every few seconds, so the host can tell a long silent stage from a hang."""

    type: str = "heartbeat"


@dataclass
class ResultEvent:
    data: dict[str, Any] = field(default_factory=dict)
    type: str = "result"


@dataclass
class ErrorEvent:
    """``code`` is one of ``codes``; the host maps it to its error class."""

    code: str
    message: str
    detail: dict[str, Any] = field(default_factory=dict)
    type: str = "error"


WorkerEvent = ProgressEvent | StepEvent | LogEvent | MetricEvent | OutputEvent | HeartbeatEvent | ResultEvent | ErrorEvent

_TYPES: dict[str, Any] = {
    "progress": ProgressEvent,
    "step": StepEvent,
    "log": LogEvent,
    "metric": MetricEvent,
    "output": OutputEvent,
    "heartbeat": HeartbeatEvent,
    "result": ResultEvent,
    "error": ErrorEvent,
}


def to_dict(event: WorkerEvent) -> dict[str, Any]:
    return asdict(event)


def from_dict(data: dict[str, Any]) -> WorkerEvent | None:
    """Rebuild an event, ignoring unknown keys; None for an unknown type or missing fields."""
    cls = _TYPES.get(str(data.get("type")))
    if cls is None:
        return None
    known = {f.name for f in fields(cls)}
    try:
        return cls(**{k: v for k, v in data.items() if k in known})
    except TypeError:
        return None
