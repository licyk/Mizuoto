"""Output records."""

from typing import Any, Literal

from pydantic import Field

from mizuoto.core.record import Record

OutputKind = Literal["speech", "design", "conversion", "edit", "sample", "preview", "line", "project"]
"""``line``: a take of a project's line (left out of lists unless asked for); ``project``: an exported project or chapter."""


class Output(Record):
    id: str
    kind: OutputKind
    name: str
    path: str
    format: str
    sample_rate: int
    duration: float
    size: int
    job_id: str | None = None
    engine: str | None = None
    model: str | None = None
    voice_id: str | None = None
    text: str | None = None
    language: str | None = None
    params: dict[str, Any] = Field(default_factory=dict)
    """The parameter values the engine used (``applied_params``)."""
    meta: dict[str, Any] = Field(default_factory=dict)
    """Timings, warnings and the request summary."""
    licence_tags: list[str] = Field(default_factory=list)
    favourite: bool = False
    created_at: str


class OutputPage(Record):
    items: list[Output]
    next_cursor: str | None = None


class OutputQuery(Record):
    voice_id: str | None = None
    model: str | None = None
    engine: str | None = None
    kind: OutputKind | None = None
    job_id: str | None = None
    text: str | None = None
    favourite: bool | None = None
    cursor: str | None = None
    limit: int = Field(default=50, ge=1, le=500)
