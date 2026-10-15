"""Experiments: a fine-tune of a base model on a dataset, run as stages."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import Field

from mizuoto.core.engines.models import TrainingMethod
from mizuoto.core.record import Record

StageStatus = Literal["pending", "running", "done", "stale", "failed"]


class StageState(Record):
    id: str
    title: str
    status: StageStatus = "pending"
    fingerprint: str | None = None
    """Of what the stage read when it last finished; a different one now makes it stale."""
    finished_at: str | None = None
    job_id: str | None = None
    error: str | None = None
    result: dict[str, Any] = Field(default_factory=dict)


class Artifact(Record):
    """What an export registered."""

    kind: Literal["voice", "model"]
    id: str
    title: str = ""
    at: str


class Experiment(Record):
    name: str
    engine: str
    base_model: str
    dataset: str
    """The dataset's id."""
    method: str
    hyperparams: dict[str, Any] = Field(default_factory=dict)
    voice: str | None = None
    """Exports of voice weights go to this library voice (None: a new voice from the dataset's clips)."""
    language: str | None = None
    stages: list[StageState] = Field(default_factory=list)
    artifacts: list[Artifact] = Field(default_factory=list)
    created_at: str
    updated_at: str

    @property
    def key(self) -> str:
        return f"{self.engine}/{self.name}"


class ExperimentInput(Record):
    name: str
    base_model: str
    dataset: str
    method: str | None = None
    """None: the first method that can train the base model."""
    hyperparams: dict[str, Any] = Field(default_factory=dict)
    voice: str | None = None
    language: str | None = None


class MethodInfo(Record):
    """A training method as a base model sees it."""

    id: str
    engine: str
    method: TrainingMethod
    available: bool
    reason: str | None = None
    """Why it cannot train this model."""
    memory_mb: dict[str, float] | None = None
    """Roughly what a run holds on its GPU, by ``training.memory`` preset (``standard``, ``low``, ``lowest``)."""


class ExperimentPatch(Record):
    hyperparams: dict[str, Any] | None = None
    voice: str | None = None
    language: str | None = None


class RunRequest(Record):
    stages: list[str] | None = None
    """These stages, in order (they need the earlier ones done). None: every stage not done, from the first."""
    restart: bool = False
    """Run every stage again from the first."""


class MetricPoint(Record):
    stage: str
    epoch: int
    step: int
    values: dict[str, float]
    lr: float | None = None
