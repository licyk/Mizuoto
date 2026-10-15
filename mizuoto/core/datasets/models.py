"""Datasets: clips with their transcripts, the one format every engine's trainer reads."""

from __future__ import annotations

from typing import Literal

from pydantic import Field

from mizuoto.core.record import Record

ImportLayout = Literal["auto", "plain", "gptsovits_list", "ljspeech", "csv"]
RowIssue = Literal["too_short", "too_long", "empty_text", "clipping"]
"""The lists of ``DatasetReport`` a row can be on."""


class DatasetRow(Record):
    """One clip: ``audio/<id>.flac`` in the dataset's folder (mono, at the clip's own rate)."""

    id: str
    audio: str
    text: str = ""
    speaker: str = ""
    language: str | None = None
    duration: float
    sample_rate: int
    excluded: bool = False
    """Kept in the dataset but left out of training."""
    source: str = ""
    """Where it came from: a file name, ``voice:<id>``, ``output:<id>``."""
    licence_tags: list[str] = Field(default_factory=list)
    """Restrictive licences of the audio (an engine's output): some forbid training other engines on it."""


class DatasetStats(Record):
    rows: int = 0
    seconds: float = 0.0
    excluded: int = 0
    missing_text: int = 0
    speakers: list[str] = Field(default_factory=list)


class Dataset(Record):
    id: str
    slug: str
    name: str
    language: str | None = None
    """The main language; each row may say its own."""
    consent: str = ""
    """Whose voices these are, and why they may be used."""
    description: str = ""
    stats: DatasetStats = Field(default_factory=DatasetStats)
    fingerprint: str = ""
    """Changes with every edit of the rows or their audio; experiments compare it."""
    created_at: str
    updated_at: str


class DatasetPage(Record):
    items: list[DatasetRow]
    total: int


class RowPatch(Record):
    text: str | None = None
    speaker: str | None = None
    language: str | None = None
    excluded: bool | None = None


class PrepareRequest(Record):
    """What ``dataset_prepare`` does, in this order."""

    slice: bool = True
    """Cut clips longer than ``max_seconds`` at their silences (GPT-SoVITS's slicer)."""
    max_seconds: float = Field(default=12.0, ge=2.0, le=60.0)
    min_seconds: float = Field(default=1.0, ge=0.2, le=10.0)
    """Clips shorter than this after cutting are excluded."""
    threshold_db: float = Field(default=-34.0, ge=-80.0, le=0.0)
    transcribe: bool = True
    """Transcribe clips without text (the speech recognition engine)."""
    normalize: bool = True
    """Level every clip's loudness."""
    language: str | None = None


class DurationBucket(Record):
    seconds: float
    """The bucket's lower edge."""
    count: int


class DatasetReport(Record):
    dataset: str
    rows: int
    usable: int
    seconds: float
    histogram: list[DurationBucket]
    too_short: list[str] = Field(default_factory=list)
    too_long: list[str] = Field(default_factory=list)
    empty_text: list[str] = Field(default_factory=list)
    clipping: list[str] = Field(default_factory=list)
    sample_rates: dict[str, int] = Field(default_factory=dict)
    speakers: dict[str, int] = Field(default_factory=dict)
    languages: dict[str, int] = Field(default_factory=dict)
    licence_tags: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
