"""Build every service once. The API, the command line and Python callers all start here."""

from __future__ import annotations

from collections.abc import Generator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from mizuoto.core.asr.service import AsrService
from mizuoto.core.audio.service import AudioService
from mizuoto.core.compute.service import ComputeService
from mizuoto.core.datasets.service import DatasetService
from mizuoto.core.db import Database
from mizuoto.core.engines.registry import EngineRegistry
from mizuoto.core.events import EventBus, LocalEventBus
from mizuoto.core.events.models import EnginesChangedEvent, SettingsChangedEvent
from mizuoto.core.inference.runtime import RuntimeService
from mizuoto.core.inference.service import InferenceService
from mizuoto.core.jobs.service import JobService
from mizuoto.core.models.service import ModelService
from mizuoto.core.outputs.service import OutputService
from mizuoto.core.presets.service import PresetService
from mizuoto.core.projects.service import ProjectService
from mizuoto.core.remote.service import RemoteService
from mizuoto.core.settings import SettingsService
from mizuoto.core.storage import StorageService
from mizuoto.core.training.service import TrainingService
from mizuoto.core.tts.service import TtsService
from mizuoto.core.voices.service import VoiceService

DB_FILE_NAME = "mizuoto.db"


@dataclass
class Services:
    """Every service, built in dependency order. Later milestones add their services here."""

    settings: SettingsService
    events: EventBus
    db: Database
    storage: StorageService
    compute: ComputeService
    jobs: JobService
    engines: EngineRegistry
    runtime: RuntimeService
    """The engines of this process (and of remote hosts) and the models they hold."""
    models: ModelService
    audio: AudioService
    outputs: OutputService
    presets: PresetService
    datasets: DatasetService
    training: TrainingService
    tts: TtsService
    asr: AsrService
    voices: VoiceService
    projects: ProjectService
    remote: RemoteService
    inference: InferenceService
    """The native runtime: devices, the model manager, the engines' implementations."""

    def close(self) -> None:
        """Stop the jobs, unload the models, then close the database."""
        self.jobs.close()
        self.runtime.close()
        self.db.close()


def build_services(
    data_dir: Path | None = None,
    settings_path: Path | None = None,
    environ: Mapping[str, str] | None = None,
    settings_overrides: dict[str, Any] | None = None,
    start_background: bool = False,
    config_dir: Path | None = None,
) -> Services:
    """Create all services.

    ``start_background`` is false for the command line, which runs a job in the foreground, and
    true for the server, whose scheduler and idle unloading then run. ``environ`` replaces
    ``os.environ`` in tests. ``config_dir`` and ``settings_path`` put the settings file outside the
    data directory, and ``settings_overrides`` pin values; both come from a host application.
    """
    settings = SettingsService(data_dir=data_dir, settings_path=settings_path, environ=environ, overrides=settings_overrides, config_dir=config_dir)
    events = LocalEventBus()
    db = Database(settings.data_dir / DB_FILE_NAME)
    storage = StorageService(settings, db)
    storage.events = events
    # Records of files inside the data folder follow it: rewritten here when it is opened somewhere new.
    storage.reconcile()
    compute = ComputeService(settings, events)
    jobs = JobService(settings, db, events, compute)
    engines = EngineRegistry(settings)
    inference = InferenceService(settings)
    runtime = RuntimeService(settings, events, compute, engines, inference)
    engines.inference = inference
    remote = RemoteService(settings)
    remote.registry = engines
    runtime.remote = remote
    engines.remote = remote
    models = ModelService(settings, engines, jobs)
    models.remote = remote
    models.before_remove = runtime.unload_model
    runtime.model_options = models.load_options
    models.events = events
    audio = AudioService(settings, db, storage)
    outputs = OutputService(settings, db, events, storage)
    presets = PresetService(db, events, engines, models)
    asr = AsrService(settings, db, engines, models, runtime, audio, jobs)
    tts = TtsService(settings, engines, models, jobs, runtime, audio, outputs, asr)
    voices = VoiceService(settings, engines, models, audio, outputs, jobs, asr)
    voices.tts = tts
    datasets = DatasetService(settings, audio, outputs, jobs, asr)
    datasets.voices = voices
    datasets.events = events
    training = TrainingService(settings, engines, models, datasets, jobs, runtime, outputs)
    training.voices = voices
    training.events = events
    voices.events = events
    tts.voices = voices
    projects = ProjectService(settings, tts, outputs, jobs)
    projects.voices = voices
    projects.events = events
    services = Services(
        settings=settings,
        events=events,
        db=db,
        storage=storage,
        compute=compute,
        jobs=jobs,
        engines=engines,
        runtime=runtime,
        models=models,
        audio=audio,
        outputs=outputs,
        presets=presets,
        datasets=datasets,
        training=training,
        tts=tts,
        asr=asr,
        voices=voices,
        projects=projects,
        remote=remote,
        inference=inference,
    )

    def on_settings_keys(keys: list[str]) -> None:
        # Every change (the API, the command line, or settings.toml edited by hand) tells the clients.
        events.publish(SettingsChangedEvent(keys=keys))
        # A changed engine setting (device, dtype, load options) takes effect when its models load again.
        changed = sorted({k.split(".")[1] for k in keys if k.startswith("engines.")})
        for engine_id in changed:
            if engine_id in engines.ids():
                runtime.unload_engine(engine_id)
        # Another remote host, address or token: sessions with the old one end.
        hosts = sorted({k.split(".")[2] for k in keys if k.startswith("remote.hosts.") and k.count(".") >= 2})
        if any(k == "remote.hosts" or k.startswith("remote.hosts") for k in keys):
            for name in hosts or list(remote.hosts()):
                remote.forget(name)
            for engine_id in sorted({e for name in hosts or list(remote.hosts()) for e in remote.engines_on(name)}):
                runtime.unload_engine(engine_id)
                changed = sorted({*changed, engine_id})
        if changed:
            events.publish(EnginesChangedEvent(engines=changed))

    settings.on_keys_change(on_settings_keys)
    if start_background:
        jobs.start()
        runtime.start()
    return services


@contextmanager
def open_services(**kwargs: Any) -> Generator[Services, None, None]:
    """Build the services for one piece of work and close them afterwards.

    This is the entry point for using Mizuoto from Python without the server::

        with open_services() as s:
            print(s.settings.view())
    """
    services = build_services(**kwargs)
    try:
        yield services
    finally:
        services.close()
