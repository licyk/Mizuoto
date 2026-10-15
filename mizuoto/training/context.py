"""What a native training stage sees: its request, and the ways it reports (as v1's trainer stage context)."""

from __future__ import annotations

import contextlib
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from mizuoto.core.jobs.events import LogEvent, MetricEvent, OutputEvent, ProgressEvent

if TYPE_CHECKING:
    from mizuoto.core.jobs.jsonl import Emitter

log = logging.getLogger("mizuoto.training")


@dataclass
class Row:
    """A dataset row as trainers see it: an absolute audio path and its text."""

    id: str
    audio: str
    text: str
    speaker: str = ""
    language: str | None = None
    duration: float = 0.0
    sample_rate: int = 0


@dataclass
class StageContext:
    emitter: Emitter
    request: dict[str, Any]
    workdir: Path = field(init=False)
    rows: list[Row] = field(init=False)
    hyperparams: dict[str, Any] = field(init=False)
    base_model: dict[str, Any] = field(init=False)
    """``{id, dir, variant}``."""
    assets: dict[str, str] = field(init=False)
    device: str = field(init=False)
    """``cuda:N`` or ``cpu``: the device as the host names it (the stage sees every device)."""
    precision: str = field(init=False)
    """``training.precision``: auto, fp32, fp16 or bf16."""
    results: dict[str, dict[str, Any]] = field(init=False)
    """What the earlier stages returned, by stage id."""
    accel: Any = field(init=False, default=None)
    """The stage's attention policy (``mizuoto.runtime.accel.policy.Run``), None for ``auto``."""

    def __post_init__(self) -> None:
        r = self.request
        self.workdir = Path(r["workdir"])
        self.workdir.mkdir(parents=True, exist_ok=True)
        self.rows = [Row(**row) for row in r.get("rows") or []]
        self.hyperparams = dict(r.get("hyperparams") or {})
        self.base_model = dict(r.get("base_model") or {})
        self.assets = dict(r.get("assets") or {})
        self.device = str(r.get("device") or "cpu")
        self.precision = str(r.get("precision") or "auto")
        self.results = dict(r.get("results") or {})

    @property
    def name(self) -> str:
        return str(self.request.get("experiment") or "experiment")

    @property
    def engine(self) -> str:
        return str(self.request.get("engine") or "")

    def hp(self, name: str, default: Any = None) -> Any:
        value = self.hyperparams.get(name)
        return default if value is None else value

    # -- reporting --------------------------------------------------------------------

    def progress(self, value: float | None, message: str | None = None) -> None:
        self.emitter.emit(ProgressEvent(value=value, message=message))

    def metric(self, epoch: int, step: int, values: dict[str, float], lr: float | None = None) -> None:
        """A step's values; on a GPU also ``memory/peak_mb``, the most allocated since the last metric."""
        values = {k: float(v) for k, v in values.items()}
        if self.device != "cpu":
            from mizuoto.training.memory import peak_mb, start_peak

            peak = peak_mb(self.device)
            if peak is not None:
                values["memory/peak_mb"] = peak
                start_peak(self.device)
        self.emitter.emit(MetricEvent(epoch=epoch, step=step, values=values, lr=lr))

    @property
    def attention(self) -> str:
        """``training.attention``: ``auto`` (each recipe's own attention) or a faster backend."""
        return str(self.request.get("attention") or "auto")

    def use_attention(self, name: str) -> None:
        """Compute the rest of the stage's attention with ``name`` (``mizuoto.runtime.accel``)."""
        from mizuoto.runtime.accel import dispatch
        from mizuoto.runtime.accel.policy import policy_for

        policy = policy_for(name, bool(self.request.get("attention_strict")))
        self.accel = policy.run() if policy is not None else None
        dispatch.PROCESS_POLICY[0] = self.accel
        self.log(f"attention: {name}")

    @property
    def memory(self) -> str:
        """``training.memory``: ``standard`` (the upstream recipe), ``low`` or ``lowest`` (``mizuoto.training.memory``)."""
        return str(self.request.get("memory") or "standard")

    def output(self, path: Path | str, label: str, kind: str = "file") -> None:
        """A file the stage made: ``kind="sample"`` for a preview clip, ``checkpoint`` for weights."""
        self.emitter.emit(OutputEvent(path=str(path), label=label, kind=kind))

    def log(self, message: str, level: str = "info") -> None:
        self.emitter.emit(LogEvent(level=level, message=message))

    # -- previews ---------------------------------------------------------------------

    def preview(
        self,
        text: str,
        reference: Row | None,
        label: str,
        *,
        weights: dict[str, Any] | None = None,
        model_dir: Path | None = None,
        variant: str | None = None,
        preset: str | None = None,
        language: str | None = None,
        options: dict[str, Any] | None = None,
    ) -> Path | None:
        """Speak ``text`` with what was trained, through the engine's native implementation in this process, and keep it
        as a sample: a voice's new ``weights`` on the base model, or a new model in ``model_dir`` (with its ``preset``
        speaker, or cloning ``reference``). Never fails the stage."""
        if not self.request.get("previews", True):
            return None
        from mizuoto.nn.dsp.wav import write_wav
        from mizuoto.runtime.context import RequestContext
        from mizuoto.runtime.device import DeviceService
        from mizuoto.runtime.engine import LoadSpec, Reference, SynthesisTask, VoiceSpec, get_engine
        from mizuoto.runtime.manager import ModelManager
        from mizuoto.runtime.runtime import Runtime

        folder = self.workdir / "preview"
        folder.mkdir(parents=True, exist_ok=True)
        target = folder / f"{label.lower().replace(' ', '-')}.wav"
        loaded = None
        manager = None
        try:
            import numpy as np
            import torch

            self.progress(None, "Speaking a preview")
            devices = DeviceService()
            manager = ModelManager(devices)
            runtime = Runtime(devices, manager)
            spec = LoadSpec(
                f"{self.base_model.get('id', 'model')}+preview",
                model_dir or Path(self.base_model["dir"]),
                variant or str(self.base_model.get("variant", "")),
                self.device,
                options={"assets": self.assets, **(options or {})},
            )
            ctx = RequestContext("preview", device=self.device, seed=1234, log=log, scratch_root=folder)
            loaded = get_engine(self.engine).load(spec, runtime, ctx)
            manager.register("preview", loaded.handles)
            voice = VoiceSpec(weights=weights, preset=preset)
            if reference is not None:
                # Engines read the 16-bit WAV the host gives them; dataset clips are FLAC (16-bit: the conversion is exact).
                from mizuoto.nn.dsp.flac import read_audio

                clip, rate = read_audio(reference.audio)
                wav = folder / f"reference-{reference.id}.wav"
                write_wav(wav, clip, rate)
                voice.references = [Reference(str(wav), reference.text or None, reference.language, rate, reference.duration or None)]
            language = language or (reference.language if reference else None) or self.request.get("language")
            task = SynthesisTask(text=text, language=language, voice=voice, params={"seed": 1234})
            ctx.manager = manager
            whole = contextlib.nullcontext() if type(loaded).PHASED else manager.use(loaded.handles)
            with whole, torch.inference_mode():
                chunks = list(loaded.synthesize(task, ctx))
            if not chunks:
                return None
            samples = np.concatenate([np.asarray(c.samples, dtype=np.float32).reshape(-1) for c in chunks])
            write_wav(target, np.clip(samples, -1.0, 1.0), int(chunks[0].sample_rate))
        except Exception as e:  # a preview is a courtesy; the trained weights are what matters
            self.log(f"No preview: {type(e).__name__}: {e}", "warning")
            return None
        finally:
            if manager is not None:
                manager.unregister("preview")
            if loaded is not None:
                loaded.close()
        self.output(target, label, kind="sample")
        return target
