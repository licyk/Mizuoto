"""Job records."""

from typing import Any, Literal

from pydantic import Field

from mizuoto.core.record import Record

JobKind = Literal[
    "speak",
    "design",
    "convert",
    "edit",
    "transcribe",
    "voice_prepare",
    "engine_op",
    "model_download",
    "model_import",
    "env_install",
    "env_remove",
    "asset_download",
    "dataset_prepare",
    "train",
    "export",
    "project_render",
    "project_export",
]
JobState = Literal["queued", "waiting", "running", "cancelling", "cancelled", "failed", "completed", "interrupted"]
StepState = Literal["pending", "running", "done", "skipped", "failed", "cancelled"]

ACTIVE_STATES: frozenset[str] = frozenset({"queued", "waiting", "running", "cancelling"})
FINISHED_STATES: frozenset[str] = frozenset({"cancelled", "failed", "completed", "interrupted"})


class ErrorInfo(Record):
    code: str
    message: str
    detail: dict[str, Any] = Field(default_factory=dict)


class JobStep(Record):
    id: str
    title: str
    state: StepState = "pending"
    progress: float | None = None


class Transfer(Record):
    """Bytes moved by a download, with the rate over the last few seconds."""

    done_bytes: int
    total_bytes: int
    bytes_per_second: float | None = None
    eta_seconds: float | None = None


class Job(Record):
    id: str
    kind: JobKind
    title: str
    state: JobState = "queued"
    waiting_for: str | None = None
    """The resource the job waits for: ``gpu:cuda:0``, ``worker:cosyvoice:cuda:0``, ``io``…"""
    progress: float | None = None
    """0..1; None while indeterminate."""
    step: str | None = None
    message: str | None = None
    """What is happening now, in words ("Loading the model: flow")."""
    steps: list[JobStep] = Field(default_factory=list)
    error: ErrorInfo | None = None
    result: dict[str, Any] | None = None
    request: dict[str, Any] = Field(default_factory=dict)
    origin: str | None = None
    """Who started it: the UI screen, ``cli`` or ``api``."""
    created_at: str
    started_at: str | None = None
    finished_at: str | None = None
    can_retry: bool = False
    can_run_anyway: bool = False
    transfer: Transfer | None = None


class JobPage(Record):
    items: list[Job]
    next_cursor: str | None = None


class JobLog(Record):
    job_id: str
    offset: int
    next_offset: int
    lines: list[str]
