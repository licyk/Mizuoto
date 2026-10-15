"""Event models. Each has an event name and is published on the EventBus.

Every event carries a snapshot, never a live object: the socket bridge serialises it later, on
the event loop, while services keep writing. Events of later milestones join this module as their
services arrive.
"""

from typing import ClassVar

from mizuoto.core.compute.models import ComputeUsage
from mizuoto.core.inference.models import RuntimeStatus
from mizuoto.core.jobs.models import Job
from mizuoto.core.record import Record
from mizuoto.core.training.models import MetricPoint


class EventBase(Record):
    __event_name__: ClassVar[str] = ""

    @classmethod
    def get_events(cls) -> list[type["EventBase"]]:
        """Every concrete event class, for the OpenAPI schema."""
        out: list[type[EventBase]] = []
        stack = list(cls.__subclasses__())
        while stack:
            sub = stack.pop()
            stack.extend(sub.__subclasses__())
            if sub.__event_name__:
                out.append(sub)
        return sorted(out, key=lambda c: c.__event_name__)


class SettingsChangedEvent(EventBase):
    __event_name__ = "settings_changed"
    keys: list[str]


class JobUpdatedEvent(EventBase):
    __event_name__ = "job_updated"
    job: Job


class JobLogEvent(EventBase):
    __event_name__ = "job_log"
    job_id: str
    offset: int
    lines: list[str]
    """At most 50; the client fetches the rest over REST."""


class JobsClearedEvent(EventBase):
    __event_name__ = "jobs_cleared"
    ids: list[str]


class OutputsAddedEvent(EventBase):
    __event_name__ = "outputs_added"
    job_id: str | None
    ids: list[str]


class OutputsRemovedEvent(EventBase):
    __event_name__ = "outputs_removed"
    ids: list[str]


class RuntimeChangedEvent(EventBase):
    """A model was loaded or unloaded, or an engine became busy or idle."""

    __event_name__ = "runtime_changed"
    runtime: RuntimeStatus


class ComputeChangedEvent(EventBase):
    __event_name__ = "compute_changed"
    usage: ComputeUsage


class EnginesChangedEvent(EventBase):
    __event_name__ = "engines_changed"
    engines: list[str]


class VoicesChangedEvent(EventBase):
    __event_name__ = "voices_changed"
    ids: list[str]
    """The voices added, changed or removed."""


class DatasetsChangedEvent(EventBase):
    __event_name__ = "datasets_changed"
    ids: list[str]


class ProjectsChangedEvent(EventBase):
    __event_name__ = "projects_changed"
    ids: list[str]
    """The projects added, changed (script, cast, takes, render state) or removed."""


class TrainChangedEvent(EventBase):
    __event_name__ = "experiments_changed"
    names: list[str]
    """``<engine>/<name>`` of the experiments that changed."""


class TrainMetricsEvent(EventBase):
    __event_name__ = "train_metrics"
    experiment: str
    """``<engine>/<name>``."""
    stage: str
    points: list[MetricPoint]


class PresetsChangedEvent(EventBase):
    __event_name__ = "presets_changed"
    ids: list[str]


class ModelsChangedEvent(EventBase):
    __event_name__ = "models_changed"
    ids: list[str]
    """The models downloaded, imported, verified or deleted."""
