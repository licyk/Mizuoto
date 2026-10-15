"""Long-form projects: a script of chapters and lines, a cast of speakers, and a take per line."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import Field

from mizuoto.core.record import Record
from mizuoto.core.tts.models import CommonParams, OutputFormatName

LineState = Literal["empty", "missing", "stale", "rendered", "failed", "rendering"]


class Speaker(Record):
    """A member of the cast: who speaks a line, with which model and voice.

    ``voice`` is a library voice (id or slug), ``preset`` a preset speaker of the model; neither: the
    model's own voice (models with ``auto_voice``).
    """

    id: str
    """Short and stable: lines refer to it."""
    name: str
    model: str | None = None
    """None: the project's ``model``."""
    voice: str | None = None
    preset: str | None = None
    language: str | None = None
    """None: the project's ``language``."""
    instruct: str | None = None
    """A style instruction for every line of this speaker (models with ``instruct``)."""
    params: CommonParams = Field(default_factory=CommonParams)
    engine_options: dict[str, Any] = Field(default_factory=dict)


class Take(Record):
    """One rendering of a line: an output of kind ``line`` and what it was made from."""

    output_id: str
    fingerprint: str
    duration: float
    created_at: str


class Line(Record):
    id: str
    text: str = ""
    speaker: str | None = None
    """A speaker id; None: the project's narrator (its first speaker)."""
    instruct: str | None = None
    """This line's style instruction, instead of its speaker's."""
    params: CommonParams = Field(default_factory=CommonParams)
    """Over its speaker's parameters (``None`` keeps the speaker's)."""
    pause_ms: int | None = Field(default=None, ge=0, le=60000)
    """Silence after this line. None: the project's ``paragraph_pause_ms`` when it ends a paragraph, else ``line_pause_ms``."""
    ends_paragraph: bool = False
    takes: list[Take] = Field(default_factory=list)
    """Newest last; at most ``MAX_TAKES`` are kept."""
    take: str | None = None
    """The output id of the take in use (one of ``takes``)."""
    error: str | None = None
    """Why the last rendering failed."""
    state: LineState = "missing"
    """Computed when the project is read: ``stale`` when the text, speaker or settings changed since the take."""


class Chapter(Record):
    id: str
    title: str = ""
    lines: list[Line] = Field(default_factory=list)


class ProjectStats(Record):
    chapters: int = 0
    lines: int = 0
    rendered: int = 0
    """Lines whose take is current."""
    stale: int = 0
    missing: int = 0
    failed: int = 0
    seconds: float = 0.0
    """The duration of the takes in use."""


class Project(Record):
    id: str
    slug: str
    name: str
    description: str = ""
    model: str | None = None
    """The model of speakers that name none."""
    language: str | None = None
    speakers: list[Speaker] = Field(default_factory=list)
    chapters: list[Chapter] = Field(default_factory=list)
    line_pause_ms: int = Field(default=350, ge=0, le=60000)
    paragraph_pause_ms: int = Field(default=800, ge=0, le=60000)
    """After a line that ends a paragraph (a blank line in an imported script)."""
    chapter_pause_ms: int = Field(default=2000, ge=0, le=60000)
    """Between chapters when the whole project is exported as one file."""
    format: OutputFormatName = "wav"
    sample_rate: int = Field(default=24000, ge=8000, le=192000)
    """Every take is resampled to it when exported (speakers may use models of different rates)."""
    normalize: Literal["none", "peak", "loudness"] = "loudness"
    stats: ProjectStats = Field(default_factory=ProjectStats)
    created_at: str
    updated_at: str


class ProjectSummary(Record):
    """A project without its script, for lists."""

    id: str
    slug: str
    name: str
    description: str
    model: str | None
    speakers: int
    stats: ProjectStats
    updated_at: str


class ProjectInput(Record):
    """A new project, or the whole editable part of one (``PUT``): what the editor sends.

    Takes are not editable: a line keeps the takes it has under its id, and only ``take`` (which one
    is used) may change to one of them.
    """

    name: str
    description: str = ""
    model: str | None = None
    language: str | None = None
    speakers: list[Speaker] = Field(default_factory=list)
    chapters: list[Chapter] = Field(default_factory=list)
    line_pause_ms: int = Field(default=350, ge=0, le=60000)
    paragraph_pause_ms: int = Field(default=800, ge=0, le=60000)
    chapter_pause_ms: int = Field(default=2000, ge=0, le=60000)
    format: OutputFormatName = "wav"
    sample_rate: int = Field(default=24000, ge=8000, le=192000)
    normalize: Literal["none", "peak", "loudness"] = "loudness"


class ScriptImport(Record):
    """A plain-text script.

    ``# Title`` starts a chapter; ``Name: text`` (or ``Name：``) is a line of that speaker when the name
    is in the cast or begins two lines or more; other lines are the narrator's; a blank line ends a
    paragraph. ``split = sentence`` makes a line of each sentence.
    """

    text: str
    split: Literal["line", "sentence"] = "line"
    replace: bool = False
    """Replace the chapters (else append them)."""
    chapter: str | None = None
    """Append the lines to this chapter instead of making chapters."""


class RenderRequest(Record):
    chapters: list[str] | None = None
    """None: every chapter."""
    lines: list[str] | None = None
    """Only these lines (ids)."""
    force: bool = False
    """Render lines whose take is current too (a new take each)."""


class ExportRequest(Record):
    chapters: list[str] | None = None
    """None: the whole project as one file."""
    format: OutputFormatName | None = None
    """None: the project's."""
    allow_missing: bool = False
    """Export what is rendered instead of refusing: a line changed since its take keeps that take; a line
    never rendered is left out."""
