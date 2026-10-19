"""Training: experiments, their stages, and what they export.

An experiment is ``<experiments>/<engine>/<name>/experiment.json``: a base model, a dataset, a method
(``[training.<method>]`` in the engine's manifest) and its hyperparameters. The method's stages run in
order, each in a child process of this interpreter (``python -m mizuoto.training.stage``: the engine's native
trainer), inside one ``train`` job whose steps are the stages; the trainer's work folder is ``work/`` beside it.

Each finished stage keeps a fingerprint of what it read: the previous stage's fingerprint, the
dataset's, the base model's revision and the hyperparameters it ``uses``. When one of them changes the
stage is ``stale``, and so is everything after it. ``export`` registers the result: a voice's own
weights, or a new user model.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import shutil
import threading
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

from mizuoto.core.clock import now_iso
from mizuoto.core.compute import cpu
from mizuoto.core.engines.models import EngineManifest, TrainingMethod
from mizuoto.core.engines.params import check_options
from mizuoto.core.engines.registry import EngineRegistry
from mizuoto.core.errors import AssetMissingError, CapabilityError, ConflictError, NotFoundError, ValidationError
from mizuoto.core.files import trash
from mizuoto.core.inference.service import allocator_env
from mizuoto.core.jobs.models import Job, JobStep
from mizuoto.core.jobs.service import JobContext, JobService, JobSpec
from mizuoto.core.process import host_target
from mizuoto.core.settings import SettingsService
from mizuoto.core.training.estimate import BATCH, PRESETS, estimate_mb
from mizuoto.core.training.models import Artifact, Experiment, ExperimentInput, ExperimentPatch, MethodInfo, MetricPoint, RunRequest, StageState

if TYPE_CHECKING:
    from mizuoto.core.datasets.service import DatasetService
    from mizuoto.core.inference.runtime import RuntimeService
    from mizuoto.core.models.service import ModelService
    from mizuoto.core.outputs.service import OutputService
    from mizuoto.core.voices.service import VoiceService

logger = logging.getLogger(__name__)

EXPERIMENT_FILE = "experiment.json"
METRICS_FILE = "metrics.jsonl"
NAME = re.compile(r"^[\w][\w.-]{0,63}$", re.UNICODE)
STAGE_MODULE = "mizuoto.training.stage"
METRIC_FLUSH_SECONDS = 1.0
STAGE_HEARTBEAT_TIMEOUT = 1800.0
"""A stage can be silent for long (a first epoch, a model load); stages beat every 10 s anyway."""


class TrainingService:
    def __init__(
        self,
        settings: SettingsService,
        registry: EngineRegistry,
        models: ModelService,
        datasets: DatasetService,
        jobs: JobService,
        runtime: RuntimeService,
        outputs: OutputService,
    ) -> None:
        self._settings = settings
        self._registry = registry
        self._models = models
        self._datasets = datasets
        self._jobs = jobs
        self._runtime = runtime
        self._outputs = outputs
        self.voices: VoiceService | None = None
        """Set by ``build_services``."""
        self.events: Any = None
        """Set by ``build_services``: ``experiments_changed`` and ``train_metrics``."""
        self._lock = threading.RLock()
        jobs.register("train", self._train_job)

    def _changed(self, exp: Experiment) -> None:
        if self.events is not None:
            from mizuoto.core.events.models import TrainChangedEvent

            self.events.publish(TrainChangedEvent(names=[exp.key]))

    # -- folders ------------------------------------------------------------------------------

    @property
    def root(self) -> Path:
        return self._settings.experiments_dir

    def folder(self, exp: Experiment) -> Path:
        return self.root / exp.engine / exp.name

    def _write(self, exp: Experiment) -> Experiment:
        exp.updated_at = now_iso()
        folder = self.folder(exp)
        folder.mkdir(parents=True, exist_ok=True)
        with self._lock:
            tmp = folder / (EXPERIMENT_FILE + ".tmp")
            tmp.write_text(exp.model_dump_json(indent=2), encoding="utf-8")
            tmp.replace(folder / EXPERIMENT_FILE)
        self._changed(exp)
        return exp

    def _read(self, path: Path) -> Experiment | None:
        try:
            return Experiment.model_validate_json(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None

    # -- experiments --------------------------------------------------------------------------

    def list_experiments(self) -> list[Experiment]:
        if not self.root.is_dir():
            return []
        found = [self._read(p) for p in sorted(self.root.glob(f"*/*/{EXPERIMENT_FILE}"))]
        return sorted((self._with_status(e) for e in found if e is not None), key=lambda e: e.updated_at, reverse=True)

    def get(self, key: str) -> Experiment:
        """``<engine>/<name>``, or a name only one engine has."""
        if "/" in key:
            exp = self._read(self.root / key / EXPERIMENT_FILE)
            if exp is not None:
                return self._with_status(exp)
        else:
            hits = [e for e in self.list_experiments() if e.name == key]
            if len(hits) == 1:
                return hits[0]
            if hits:
                raise ConflictError(f"Several engines have an experiment {key!r}; say which: <engine>/{key}", {"experiments": [e.key for e in hits]})
        raise NotFoundError(f"No experiment {key!r}", {"kind": "experiment", "id": key, "hint": "mizuoto train list"})

    def methods(self, model_id: str) -> list[MethodInfo]:
        """The engine's training methods, and whether each can start from this model."""
        info = self._models.get(model_id)
        manifest = self._registry.manifest(info.engine)
        out: list[MethodInfo] = []
        for method_id, method in manifest.training.items():
            reason = None
            if not method.enabled:
                reason = method.note or "Not available yet"
            elif method.variants is not None and info.variant not in method.variants:
                reason = f"Starts from {', '.join(method.variants)} models, not {info.variant}"
            elif info.remote is not None:
                reason = f"{info.engine} runs on {info.remote}, and training runs on this machine: mizuoto remote use {info.engine} --local"
            elif info.state not in ("builtin", "installed"):
                reason = f"{info.title} is {info.state}"
            else:
                reason = self._missing(info.engine, method, info.variant) or self._refused(info.engine, method_id, info.path)
            # As the model sees it: the hyperparameters of other variants left out.
            own = {k: v for k, v in method.hyperparams.items() if v.variants is None or info.variant in v.variants}
            memory = estimate_mb(info.engine, method_id, info.size_mb, "lora" in method_id or "lora_rank" in own) if info.size_mb else None
            out.append(
                MethodInfo(id=method_id, engine=info.engine, method=method.model_copy(update={"hyperparams": own}), available=reason is None, reason=reason, memory_mb=memory)
            )
        return out

    def _preflight(self, exp: Experiment, device: str, ctx: JobContext) -> None:
        """Say so when the run is likely to need more of the GPU than is free: a rough estimate, so a warning
        in the job's log, with the preset that would fit, rather than a refusal."""
        try:
            info = self._models.get(exp.base_model)
            devices = self._runtime.compute.info(refresh=True).devices
            free = next((d.memory_total_mb - (d.memory_used_mb or 0) for d in devices if d.id == device and d.memory_total_mb), None)
            hp = self._effective(exp, self.method(exp, exp.engine)[1])
        except Exception:  # a check, never a reason to fail the run
            return
        if free is None or not info.size_mb:
            return
        estimate = estimate_mb(exp.engine, exp.method, info.size_mb, "lora" in exp.method, hp)
        preset = self._settings.settings.training.memory
        if estimate[preset] <= free:
            return
        better = next((p for p in PRESETS if PRESETS.index(p) > PRESETS.index(preset) and estimate[p] <= free), None)
        hint = f"set Settings › Training › Memory to {better} (mizuoto config set training.memory {better})" if better else "use a bigger GPU or the CPU"
        if better is None and exp.engine in BATCH:  # a smaller batch may fit where no preset does
            name, default, _ = BATCH[exp.engine]
            current = hp.get(name) or default  # 0: the recipe's own (VoxCPM's 16)
            if isinstance(current, (int, float)) and current > 1:
                fits = next(
                    (b for b in range(int(current) - 1, 0, -1) if estimate_mb(exp.engine, exp.method, info.size_mb, "lora" in exp.method, {**hp, name: b})[preset] <= free), None
                )
                if fits is not None:
                    hint = f"lower the {name} hyperparameter to {fits}"
        ctx.log(f"Warning: this run may need about {estimate[preset] / 1024:.1f} GB of {device}, which has {free / 1024:.1f} GB free: {hint}")

    def trainer(self, engine: str) -> str | None:
        """``module:Class`` of the engine's native trainer (``Engine.TRAINER``), or None."""
        if not self._runtime.inference.implemented(engine):
            return None
        return self._runtime.inference.trainer(engine)

    def _refused(self, engine: str, method: str, model_dir: str | None) -> str | None:
        """What the engine's trainer says about this model (``Trainer.refuses``), when the model is on disk."""
        trainer = self.trainer(engine)
        if trainer is None or not model_dir:
            return None
        try:
            return self._runtime.inference.trainer_refuses(trainer, method, model_dir)
        except Exception:  # a check: the stage itself says what is wrong
            logger.debug("Asking %s about %s failed", trainer, model_dir, exc_info=True)
            return None

    def _missing(self, engine: str, method: TrainingMethod, variant: str) -> str | None:
        """What training still needs: the engine and its trainer, a device it runs on, the trainer's packages, the
        training assets."""
        status = self._registry.status(engine)
        if not status.available:
            return f"{engine} needs {', '.join(status.missing)}: mizuoto doctor {engine}"
        trainer = self.trainer(engine)
        if trainer is None:
            return f"{engine} has no trainer"
        own, lacking = self._runtime.inference.trainer_needs(trainer)
        device = self._device(engine)
        devices = own or method.devices
        if devices is not None and device.split(":")[0] not in devices:
            kinds = " or ".join(d.upper() if d == "cuda" else d for d in devices)
            return f"{method.title} runs on a {kinds} device, and training would run on {device}: mizuoto config set training.devices '[\"cuda:0\"]'"
        if lacking:
            return f"Training needs {', '.join(lacking)}: pip install {' '.join(lacking)}"
        missing = [a for a in self.training_assets(method, variant) if self._models.asset(a).state != "installed"]
        if missing:
            return f"Training needs {', '.join(missing)}: mizuoto asset download {' '.join(missing)}"
        return None

    @staticmethod
    def training_assets(method: TrainingMethod, variant: str) -> list[str]:
        return [*method.assets.get("*", []), *method.assets.get(variant, [])]

    def method(self, exp: Experiment | ExperimentInput, engine: str) -> tuple[EngineManifest, TrainingMethod]:
        manifest = self._registry.manifest(engine)
        method = manifest.training.get(exp.method or "")
        if method is None:
            raise CapabilityError(f"{manifest.title} has no training method {exp.method!r}", {"methods": sorted(manifest.training)})
        return manifest, method

    def create(self, body: ExperimentInput) -> Experiment:
        if not NAME.match(body.name):
            raise ValidationError("An experiment's name is letters, digits, '-', '_' and '.'", {"name": body.name})
        info = self._models.get(body.base_model)
        options = self.methods(info.id)
        if body.method is None:
            fitting = [m for m in options if m.available]
            if not fitting:
                reasons = {m.id: m.reason for m in options}
                raise CapabilityError(f"{info.title} cannot be trained in Mizuoto", {"model": info.id, "methods": reasons})
            body = body.model_copy(update={"method": fitting[0].id})
        manifest, method = self.method(body, info.engine)
        chosen = next(m for m in options if m.id == body.method)
        if not chosen.available:
            raise CapabilityError(f"{manifest.title}'s {method.title} cannot start from {info.title}: {chosen.reason}", {"method": body.method, "model": info.id})
        if (self.root / info.engine / body.name).exists():
            raise ConflictError(f"There is already an experiment {info.engine}/{body.name}", {"experiment": f"{info.engine}/{body.name}"})
        dataset = self._datasets.get(body.dataset)
        rows = self._datasets.training_rows(dataset.id, info.engine, self._registry)
        seconds = sum(r.duration for r in rows)
        if seconds < method.min_seconds:
            raise ValidationError(f"{dataset.name} has {seconds:.0f} s of usable audio; {method.title} needs {method.min_seconds:.0f} s", {"seconds": seconds})
        hyperparams = check_options(manifest, info.variant, body.hyperparams, schema=method.hyperparams)
        if body.voice and self.voices is not None:
            self.voices.get(body.voice)
        stamp = now_iso()
        exp = Experiment(
            name=body.name,
            engine=info.engine,
            base_model=info.id,
            dataset=dataset.id,
            method=body.method or "",
            hyperparams=hyperparams,
            voice=body.voice,
            language=body.language or dataset.language,
            stages=[StageState(id=s.id, title=s.title) for s in method.stages],
            created_at=stamp,
            updated_at=stamp,
        )
        return self._with_status(self._write(exp))

    def update(self, key: str, patch: ExperimentPatch) -> Experiment:
        exp = self.get(key)
        manifest, method = self.method(exp, exp.engine)
        if patch.hyperparams is not None:
            variant = self._models.get(exp.base_model).variant
            exp.hyperparams = check_options(manifest, variant, patch.hyperparams, schema=method.hyperparams)
        if patch.voice is not None:
            if patch.voice and self.voices is not None:
                self.voices.get(patch.voice)
            exp.voice = patch.voice or None
        if patch.language is not None:
            exp.language = patch.language or None
        return self._with_status(self._write(exp))

    def delete(self, key: str) -> None:
        exp = self.get(key)
        if any(s.status == "running" for s in exp.stages):
            raise ConflictError(f"{exp.key} is running; cancel its job first")
        trash(self.folder(exp), self._settings.data_dir / "trash")
        self._changed(exp)

    # -- stages and fingerprints --------------------------------------------------------------

    def _effective(self, exp: Experiment, method: TrainingMethod) -> dict[str, Any]:
        return {**{k: s.default for k, s in method.hyperparams.items() if s.default is not None}, **exp.hyperparams}

    def fingerprints(self, exp: Experiment) -> dict[str, str]:
        """What each stage would read now, as a hash chained from the first stage."""
        _, method = self.method(exp, exp.engine)
        try:
            dataset_fp = self._datasets.get(exp.dataset).fingerprint
        except NotFoundError:
            dataset_fp = "missing"
        try:
            revision = self._models.revision(exp.base_model)
        except NotFoundError:
            revision = "missing"
        hp = self._effective(exp, method)
        out: dict[str, str] = {}
        previous = ""
        for stage in method.stages:
            used = hp if stage.uses is None else {k: hp.get(k) for k in stage.uses}
            source = json.dumps(
                {"previous": previous, "stage": stage.id, "dataset": dataset_fp, "model": [exp.base_model, revision], "method": exp.method, "hp": used, "language": exp.language},
                sort_keys=True,
            )
            previous = out[stage.id] = hashlib.sha256(source.encode()).hexdigest()[:20]
        return out

    def _with_status(self, exp: Experiment) -> Experiment:
        """Stage states with ``stale`` worked out: a finished stage whose inputs changed, and what follows it."""
        try:
            current = self.fingerprints(exp)
        except (CapabilityError, NotFoundError):
            return exp
        broken = False
        for stage in exp.stages:
            if stage.status == "running":
                continue
            if broken and stage.status == "done":
                stage.status = "stale"
            elif stage.status in ("done", "stale"):
                stage.status = "done" if stage.fingerprint == current.get(stage.id) else "stale"
            broken = broken or stage.status != "done"
        return exp

    # -- running ------------------------------------------------------------------------------

    def run(self, key: str, req: RunRequest | None = None, *, wait: bool = False, origin: str | None = None) -> Job:
        """Run stages in a ``train`` job: the given ones, or every stage from the first not done."""
        exp = self.get(key)
        req = req or RunRequest()
        _, method = self.method(exp, exp.engine)
        problem = self._missing(exp.engine, method, self._models.get(exp.base_model).variant)
        if problem:
            raise CapabilityError(f"{exp.key} cannot train yet. {problem}", {"hint": problem.split(": ", 1)[-1]})
        stages = self._selected(exp, req)
        if not stages:
            raise ValidationError(f"Every stage of {exp.key} is done; restart it to train again", {"hint": f"mizuoto train run {exp.key} --restart"})
        request = {"experiment": exp.key, "stages": stages}
        return self._jobs.run_now("train", request, origin) if wait else self._jobs.submit("train", request, origin)

    def _selected(self, exp: Experiment, req: RunRequest) -> list[str]:
        ids = [s.id for s in exp.stages]
        if req.restart:
            return ids
        if req.stages:
            unknown = [s for s in req.stages if s not in ids]
            if unknown:
                raise ValidationError(f"Unknown stages: {', '.join(unknown)}", {"stages": ids})
            first = min(ids.index(s) for s in req.stages)
            missing = [s.id for s in exp.stages[:first] if s.status != "done"]
            if missing:
                raise ValidationError(f"Run {', '.join(missing)} first", {"stages": missing})
            return [s for s in ids if s in req.stages]
        first = next((i for i, s in enumerate(exp.stages) if s.status != "done"), None)
        return [] if first is None else ids[first:]

    def _device(self, engine: str) -> str:
        devices = self._settings.settings.training.devices
        return devices[0] if devices else self._runtime.device_for(engine)

    def _train_job(self, request: dict[str, Any]) -> JobSpec:
        exp = self.get(str(request["experiment"]))
        stages: list[str] = list(request["stages"])
        device = self._device(exp.engine)
        # A fine-tune takes the device: it waits for the engine's own inference and for other jobs on the GPU.
        resources = {f"engine:{exp.engine}:{device}"} | ({f"gpu:{device}"} if device != "cpu" else {"cpu"})
        titles = {s.id: s.title for s in exp.stages}
        return JobSpec(
            title=f"Train {exp.key}",
            run=lambda ctx: self._run(exp.key, stages, device, ctx),
            resources=frozenset(resources),
            exclusive=frozenset({f"gpu:{device}"} if device != "cpu" else set()),
            steps=[JobStep(id=s, title=titles.get(s, s)) for s in stages],
        )

    def _run(self, key: str, stages: list[str], device: str, ctx: JobContext) -> dict[str, Any]:
        exp = self.get(key)
        manifest, method = self.method(exp, exp.engine)
        trainer = self.trainer(exp.engine)
        if not trainer:
            raise CapabilityError(f"{manifest.title} has no trainer")
        if device != "cpu":
            # The stage's process needs the device: the engine's own models go, and every other model there moves to
            # RAM. The job holds the whole device, so nothing comes back while it runs.
            self._runtime.unload_engine(exp.engine)
            self._runtime.release_device(device)
            self._preflight(exp, device, ctx)
        info = self._models.require(exp.base_model)
        problem = self._missing(exp.engine, method, info.variant)
        if problem:
            raise CapabilityError(f"{manifest.title} cannot train yet. {problem}", {"hint": problem.split(": ", 1)[-1]})
        refused = self._refused(exp.engine, exp.method, info.path)
        if refused:
            raise CapabilityError(f"{manifest.title}'s {method.title} cannot start from {info.title}: {refused}", {"method": exp.method, "model": info.id})
        assets = self._models.asset_paths(info.id)
        for asset_id in self.training_assets(method, info.variant):
            asset = self._models.asset(asset_id)
            if asset.state != "installed" or asset.path is None:
                raise AssetMissingError(f"Training {info.title} needs {asset.title}", [asset_id], {"hint": f"mizuoto asset download {asset_id}"})
            assets[asset_id] = str(asset.path)
        rows = self._datasets.training_rows(exp.dataset, exp.engine, self._registry)
        dataset_dir = self._datasets.folder(self._datasets.get(exp.dataset))
        folder = self.folder(exp)
        workdir = folder / "work"
        # A child of this interpreter: the native modules, every device visible under its own name.
        compute = self._settings.settings.compute
        # the lowest preset's gigabytes of optimizer state in RAM stall on huge pages (cpu.py); other stages gain from them
        huge_pages = self._settings.settings.training.memory != "lowest"
        env = {**cpu.child_env(compute.cpu_allocator, compute.cpu_threads, huge_pages=huge_pages), **allocator_env(compute.cuda_allocator)}
        deterministic = self._settings.settings.training.deterministic
        if deterministic:  # cuBLAS's deterministic workspaces, read when CUDA starts in the stage
            env["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        target = host_target(STAGE_MODULE, cwd=workdir, env=env)
        current = self.fingerprints(exp)
        hyperparams = self._effective(exp, method)
        workdir.mkdir(parents=True, exist_ok=True)
        results: dict[str, Any] = {}
        for stage_id in stages:
            exp = self.get(key)
            state = next(s for s in exp.stages if s.id == stage_id)
            ctx.step(stage_id, "running", 0.0)
            ctx.message(state.title)
            state.status, state.job_id, state.error = "running", ctx.job_id, None
            self._write(exp)
            request = {
                "trainer": trainer,
                "stage": stage_id,
                "engine": exp.engine,
                "method": exp.method,
                "experiment": exp.name,
                "workdir": str(workdir),
                "dataset_dir": str(dataset_dir),
                "rows": [
                    {
                        "id": r.id,
                        "audio": str(dataset_dir / r.audio),
                        "text": r.text,
                        "speaker": r.speaker,
                        "language": r.language or exp.language,
                        "duration": r.duration,
                        "sample_rate": r.sample_rate,
                    }
                    for r in rows
                ],
                "base_model": {"id": info.id, "dir": info.path, "variant": info.variant},
                "assets": assets,
                "previews": self._settings.settings.training.previews,
                "keep_checkpoints": self._settings.settings.training.keep_checkpoints,
                "device": device,
                "precision": self._settings.settings.training.precision,
                "memory": self._settings.settings.training.memory,
                "attention": self._settings.settings.training.attention,
                "attention_strict": self._settings.settings.compute.attention_strict,
                "fast": list(self._settings.settings.compute.fast),
                "deterministic": deterministic,
                "hyperparams": hyperparams,
                "language": exp.language,
                "results": {s.id: s.result for s in exp.stages if s.status == "done"},
            }
            self._drop_metrics(exp, stage_id)
            on_event, flush = self._events_for(exp, stage_id, ctx)
            try:
                result = ctx.run_worker(target, request, on_event=on_event, heartbeat_timeout=STAGE_HEARTBEAT_TIMEOUT)
            except BaseException as e:
                flush()
                exp = self.get(key)
                state = next(s for s in exp.stages if s.id == stage_id)
                state.status = "failed"
                state.error = getattr(e, "message", None) or str(e) or type(e).__name__
                self._write(exp)
                raise
            flush()
            exp = self.get(key)
            state = next(s for s in exp.stages if s.id == stage_id)
            if stage_id == "export":
                made = self._register(exp, method, result.get("artifacts") or [], rows)
                # The same voice or model exported again (its id reused) is listed once, at its latest export.
                exp.artifacts = [a for a in exp.artifacts if (a.kind, a.id) not in {(m.kind, m.id) for m in made}] + made
                result = {k: v for k, v in result.items() if k != "artifacts"}
            state.status, state.fingerprint, state.finished_at, state.result = "done", current[stage_id], now_iso(), result
            # A stage run again made new files: the finished stages after it read the old ones, so they are
            # stale even when the inputs are the same (their fingerprints would be).
            ids = [s.id for s in exp.stages]
            for later in exp.stages[ids.index(stage_id) + 1 :]:
                if later.status in ("done", "stale") and later.id not in stages:
                    later.fingerprint = None
            self._write(exp)
            ctx.step(stage_id, "done")
            results[stage_id] = result
        exp = self.get(key)
        return {"experiment": exp.key, "stages": results, "artifacts": [a.model_dump() for a in exp.artifacts]}

    def _drop_metrics(self, exp: Experiment, stage: str) -> None:
        """A stage run again starts its curve afresh: its earlier points are removed."""
        path = self.folder(exp) / METRICS_FILE
        if not path.is_file():
            return
        lines = path.read_text(encoding="utf-8").splitlines()
        kept = [line for line in lines if line.strip() and json.loads(line).get("stage") != stage]
        if len(kept) != len(lines):
            path.write_text("".join(line + "\n" for line in kept), encoding="utf-8")

    def _events_for(self, exp: Experiment, stage: str, ctx: JobContext) -> tuple[Any, Any]:
        """The stage's event handler (metrics batched to ``metrics.jsonl`` and ``train_metrics``), and its flush."""
        from mizuoto.core.jobs.events import MetricEvent, OutputEvent, ProgressEvent, StepEvent

        metrics_path = self.folder(exp) / METRICS_FILE
        pending: list[dict[str, Any]] = []
        last = [time.monotonic()]

        def flush() -> None:
            if not pending:
                return
            with open(metrics_path, "a", encoding="utf-8") as f:
                for p in pending:
                    f.write(json.dumps(p) + "\n")
            if self.events is not None:
                from mizuoto.core.events.models import TrainMetricsEvent

                self.events.publish(TrainMetricsEvent(experiment=exp.key, stage=stage, points=[MetricPoint(**p) for p in pending]))
            pending.clear()
            last[0] = time.monotonic()

        def on_event(event: Any) -> None:
            if isinstance(event, MetricEvent):
                pending.append({"stage": stage, "epoch": event.epoch, "step": event.step, "values": event.values, "lr": event.lr})
                if time.monotonic() - last[0] >= METRIC_FLUSH_SECONDS:
                    flush()
            elif isinstance(event, ProgressEvent):
                flush()
                ctx.step(stage, "running", event.value)
                if event.message:
                    ctx.message(event.message)
            elif isinstance(event, StepEvent):
                ctx.message(event.title or event.id)
            elif isinstance(event, OutputEvent) and event.kind == "sample":
                self._sample(exp, stage, Path(event.path), event.label, ctx.job_id)

        return on_event, flush

    def _sample(self, exp: Experiment, stage: str, path: Path, label: str, job_id: str) -> None:
        """A preview the trainer made: kept as an output of kind ``sample``."""
        from mizuoto.core.audio.io import probe
        from mizuoto.core.outputs.models import Output

        if not path.is_file():
            return
        target = self._outputs.new_path({"voice": exp.name, "model": "sample", "engine": exp.engine}, path.suffix.lstrip(".") or "wav")
        shutil.copyfile(path, target)
        info = probe(target)
        output = Output(
            id=self._outputs.new_id(),
            kind="sample",
            name=f"{exp.name}: {label}",
            path=str(target),
            format=target.suffix.lstrip("."),
            sample_rate=info.sample_rate,
            duration=info.duration or 0.0,
            size=target.stat().st_size,
            job_id=job_id,
            engine=exp.engine,
            model=exp.base_model,
            text=label,
            params={},
            meta={"experiment": exp.key, "stage": stage},
            licence_tags=[],
            created_at=now_iso(),
        )
        self._outputs.add(output, job_id)

    def _register(self, exp: Experiment, method: TrainingMethod, artifacts: list[dict[str, Any]], rows: list[Any]) -> list[Artifact]:
        out: list[Artifact] = []
        for a in artifacts:
            if a.get("kind") == "voice_weights":
                if self.voices is None:
                    raise ValidationError("The voice library is not available")
                voice = self.voices.get(exp.voice) if exp.voice else self._new_voice(exp, rows)
                files = {role: Path(p) for role, p in dict(a["files"]).items()}
                voice = self.voices.add_weights(voice.id, exp.engine, files, move=True)
                out.append(Artifact(kind="voice", id=voice.id, title=voice.name, at=now_iso()))
            elif a.get("kind") == "model":
                model = self._models.register_user_model(
                    exp.engine,
                    str(a.get("variant") or self._models.get(exp.base_model).variant),
                    Path(a["folder"]),
                    exp.name,
                    str(a.get("title") or exp.name),
                    extra={"trained_from": exp.base_model, "experiment": exp.key, **({"speakers": list(a["speakers"])} if a.get("speakers") else {})},
                )
                out.append(Artifact(kind="model", id=model.id, title=model.title, at=now_iso()))
        return out

    def _new_voice(self, exp: Experiment, rows: list[Any]) -> Any:
        """A voice for exported weights: up to three of the dataset's clips (3–10 s, with text) as its references."""
        assert self.voices is not None
        dataset = self._datasets.get(exp.dataset)
        picks = sorted((r for r in rows if 3.0 <= r.duration <= 10.0 and r.text.strip()), key=lambda r: abs(r.duration - 6.0))[:3] or rows[:1]
        folder = self._datasets.folder(dataset)
        voice, _ = self.voices.create_from_audio(
            exp.name,
            [folder / r.audio for r in picks],
            [r.text for r in picks],
            language=exp.language,
            consent=dataset.consent or None,
            description=f"Trained from {dataset.name} ({exp.key})",
            transcribe=False,
        )
        return voice

    def metrics(self, key: str, stage: str | None = None) -> list[MetricPoint]:
        exp = self.get(key)
        path = self.folder(exp) / METRICS_FILE
        if not path.is_file():
            return []
        points = [MetricPoint.model_validate_json(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        return [p for p in points if stage is None or p.stage == stage]
