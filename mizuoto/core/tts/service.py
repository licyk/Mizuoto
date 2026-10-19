"""Synthesis: validate, plan, dispatch, assemble, store.

Everything that can be checked is checked here, in the host, from the engine's manifest, before a
job is queued: an invalid request fails at once with a precise error and never loads a model.
The job then holds the engine on its device, gets the loaded model's handle (``RuntimeService.loaded``),
asks it for each segment's audio, joins the audio, levels and encodes it, and records the output.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import logging
import queue
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from mizuoto.core.asr.service import AsrService
from mizuoto.core.audio import dsp
from mizuoto.core.audio.io import encode, resample
from mizuoto.core.audio.service import AudioService
from mizuoto.core.clock import now_iso
from mizuoto.core.engines import languages
from mizuoto.core.engines.models import EngineManifest, ModelParamSpecs, VariantInfo, VoiceMode
from mizuoto.core.engines.params import check_options, check_params
from mizuoto.core.engines.registry import EngineRegistry
from mizuoto.core.errors import CapabilityError, LanguageError, NotFoundError, ReferenceInvalidError, ValidationError
from mizuoto.core.inference.engine import AudioStream, EngineHandle, EngineRequest, ModelRef
from mizuoto.core.inference.runtime import RuntimeService
from mizuoto.core.inference.service import distribution_version
from mizuoto.core.inference.tasks import (
    AudioChunk,
    EditSpan,
    EmotionSpec,
    Reference,
    SpeechEditTask,
    StyleSpec,
    SynthesisTask,
    VoiceConversionTask,
    VoiceSpec,
    conversion_task_from_dict,
    edit_task_from_dict,
    to_dict,
)
from mizuoto.core.jobs.models import Job
from mizuoto.core.jobs.service import INFERENCE, JobContext, JobService, JobSpec
from mizuoto.core.models.service import ModelService
from mizuoto.core.outputs.models import Output
from mizuoto.core.outputs.service import OutputService
from mizuoto.core.settings import SettingsService
from mizuoto.core.text.segment import segment
from mizuoto.core.tts.models import (
    OPENAI_FORMATS,
    OPENAI_PCM_RATE,
    OPENAI_VOICES,
    CommonParams,
    ConvertRequest,
    EditRequest,
    OpenAiSpeechRequest,
    OutputSpec,
    PlannedReference,
    ReferenceInput,
    SegmentPreview,
    SourceInput,
    SpeakRequest,
    StyleInput,
    SynthesisPlan,
    VoiceInput,
)
from mizuoto.version import VERSION

if TYPE_CHECKING:
    import numpy as np

    from mizuoto.core.voices.service import VoiceService

logger = logging.getLogger(__name__)

MIZUOTO_SEGMENT_PAUSE_MS = 120
"""The pause Mizuoto puts between the segments it split, when the request names none."""


class TextFeed:
    """Text that arrives while a synthesis runs (``SpeakRequest.text_stream``): ``put`` pieces, then ``close``."""

    def __init__(self, max_chars: int | None = None) -> None:
        self._queue: queue.Queue[tuple[str, bool]] = queue.Queue()
        self._parts: list[str] = []
        self._chars = 0
        self.max_chars = max_chars
        self.closed = False

    def put(self, text: str) -> None:
        if self.closed:
            raise ValidationError("The text has already ended")
        self._chars += len(text)
        if self.max_chars is not None and self._chars > self.max_chars:
            raise ValidationError(f"The text is longer than {self.max_chars} characters", {"max_chars": self.max_chars})
        if text:
            self._parts.append(text)
            self._queue.put((text, False))

    def close(self) -> None:
        if not self.closed:
            self.closed = True
            self._queue.put(("", True))

    @property
    def text_input(self) -> queue.Queue[tuple[str, bool]]:
        """The pieces as they come, ``(text, last)``: the engine request's ``text_input``."""
        return self._queue

    @property
    def text(self) -> str:
        return "".join(self._parts)


@dataclass
class StreamSink:
    """Receives a synthesis as it happens: ``start`` once, ``chunk`` per piece, then ``end`` or ``error``."""

    start: Callable[[int, str | None], None] = lambda sample_rate, granularity: None
    chunk: Callable[[np.ndarray, int, int], None] = lambda samples, sample_rate, segment: None
    end: Callable[[list[str]], None] = lambda output_ids: None
    error: Callable[[Exception], None] = lambda error: None
    text: TextFeed | None = None
    """The rest of the text, for a request with ``text_stream``."""


@dataclass
class _Transform:
    """A conversion or an edit, checked: everything its job needs."""

    op: str
    kind: str
    title: str
    model: str
    engine: str
    variant: str
    device: str
    manifest: EngineManifest
    model_path: str | None
    payload: dict[str, Any]
    output: OutputSpec
    model_sample_rate: int
    text: str | None = None
    voice_id: str | None = None
    warnings: list[str] = field(default_factory=list)
    dtype: str = "auto"
    load_options: dict[str, Any] = field(default_factory=dict)


@dataclass
class _Prepared:
    """A plan plus what the job needs that is not part of the plan's record."""

    plan: SynthesisPlan
    request: SpeakRequest
    manifest: EngineManifest
    variant: VariantInfo
    model_path: str | None
    voice: VoiceSpec
    style: StyleSpec
    dtype: str = "auto"
    load_options: dict[str, Any] = field(default_factory=dict)
    voice_name: str | None = None
    """The library voice's slug, for file names."""


class TtsService:
    def __init__(
        self,
        settings: SettingsService,
        registry: EngineRegistry,
        models: ModelService,
        jobs: JobService,
        runtime: RuntimeService,
        audio: AudioService,
        outputs: OutputService,
        asr: AsrService | None = None,
    ) -> None:
        self._settings = settings
        self._registry = registry
        self._models = models
        self._jobs = jobs
        self._runtime = runtime
        self._audio = audio
        self._outputs = outputs
        self._asr = asr
        self.voices: VoiceService | None = None
        """The voice library; set by ``build_services``."""
        self._sink_lock = threading.Lock()
        self._next_sink: StreamSink | None = None
        jobs.register("speak", self._speak_job)
        jobs.register("voice_prepare", self._prepare_job)
        jobs.register("convert", self._convert_job)
        jobs.register("edit", self._edit_job)
        jobs.register("engine_op", self._op_job)

    # -- what a model offers ------------------------------------------------------------

    def model_spec(self, model_id: str) -> ModelParamSpecs:
        info = self._models.get(model_id)
        described = self._runtime.description(model_id)
        speakers = list(described["speakers"]) if described and described.get("speakers") else None
        return self._registry.param_specs(model_id, info.engine, info.variant, speakers or self._models.speakers(model_id) or None)

    def segment_preview(self, text: str, model: str | None = None, segmentation: str = "engine") -> SegmentPreview:
        """How ``text`` will be split for ``model`` (any model: Mizuoto's split)."""
        if model:
            info = self._models.get(model)
            caps = set(self._registry.variant(info.engine, info.variant).capabilities)
            if segmentation != "mizuoto" and "long_text" in caps:
                return SegmentPreview(segments=[text.strip()] if text.strip() else [], segmented_by="engine")
        return SegmentPreview(segments=segment(text), segmented_by="mizuoto")

    # -- the OpenAI-compatible endpoint ---------------------------------------------------

    def model_for_name(self, name: str) -> str:
        """A Mizuoto model id for a name an OpenAI client sends: an id, an alias, or ``tts.default_model``."""
        settings = self._settings.settings.tts
        name = settings.model_aliases.get(name, name)
        try:
            self._models.get(name)
            return name
        except NotFoundError:
            if settings.default_model:
                return settings.default_model
            raise NotFoundError(
                f"No model {name!r}: give a Mizuoto model id, or set tts.model_aliases or tts.default_model",
                {"model": name, "hint": "mizuoto config set tts.default_model <engine>/<model>"},
            ) from None

    def from_openai(self, body: OpenAiSpeechRequest) -> SpeakRequest:
        """The ``SpeakRequest`` for an OpenAI speech request. Parameters a model does not take are dropped."""
        model = self.model_for_name(body.model)
        info = self._models.get(model)
        variant = self._registry.variant(info.engine, info.variant)
        caps = set(variant.capabilities)
        voice_name = body.voice.id if body.voice is not None and not isinstance(body.voice, str) else (body.voice or "").strip()
        voice: str | None = None
        spec: VoiceInput | None = None
        if voice_name:
            library = None
            if self.voices is not None:
                try:
                    library = self.voices.get(voice_name)
                except NotFoundError:
                    library = None
            if library is not None:
                voice = library.id
            elif "preset_speaker" in caps and voice_name in self.model_spec(model).speakers:
                spec = VoiceInput(preset=voice_name)
            elif voice_name.lower() not in OPENAI_VOICES:
                raise NotFoundError(f"No voice {voice_name!r}: give a library voice or a speaker of {info.title}", {"voice": voice_name, "model": model})
        if voice is None and spec is None and "auto_voice" not in caps:
            speakers = self.model_spec(model).speakers
            if speakers:
                spec = VoiceInput(preset=speakers[0])
        instruct = (body.instructions or "").strip()
        return SpeakRequest(
            model=model,
            text=body.input,
            voice=voice,
            voice_spec=spec,
            style=StyleInput(instruct=instruct) if instruct and "instruct" in caps else None,
            params=CommonParams(speed=body.speed),
            output=OutputSpec(format=OPENAI_FORMATS[body.response_format], sample_rate=OPENAI_PCM_RATE if body.response_format == "pcm" else None),
            strict=False,
        )

    # -- planning -------------------------------------------------------------------------

    def plan(self, req: SpeakRequest, stream: bool = False) -> SynthesisPlan:
        """Validate ``req`` against the model and say what will happen, without running anything."""
        return self._prepare(req, stream).plan

    def _prepare(self, req: SpeakRequest, stream: bool = False) -> _Prepared:
        settings = self._settings.settings
        info = self._models.require(req.model)
        engine = info.engine
        self._registry.check_enabled(engine)
        self._registry.check_licence(engine)
        self._registry.check_available(engine)
        manifest = self._registry.manifest(engine)
        if manifest.internal:
            raise CapabilityError(f"{info.title} is not a speech synthesis model", {"model": req.model, "engine": engine})
        variant = self._registry.variant(engine, info.variant)
        caps = set(variant.capabilities)
        warnings: list[str] = []
        if req.ignored_fields:
            warnings.append(f"Ignored fields the request does not have: {', '.join(req.ignored_fields)}")

        # text
        texts = [t for t in (req.texts or []) if t.strip()]
        if req.texts is not None:
            if "batch" not in caps:
                raise CapabilityError(f"{info.title} does not take a batch of texts", {"capability": "batch", "model": req.model})
            if not texts:
                raise ValidationError("The batch is empty")
            if variant.max_batch and len(texts) > variant.max_batch:
                raise ValidationError(f"{info.title} takes at most {variant.max_batch} texts at once", {"max_batch": variant.max_batch})
        elif req.text_stream:
            if "stream_text_input" not in caps:
                raise CapabilityError(f"{info.title} cannot take its text as it is written", {"capability": "stream_text_input", "model": req.model})
            if not stream:
                raise ValidationError("Streamed text needs a stream: the /tts/stream WebSocket")
        elif not req.text.strip():
            raise ValidationError("There is no text to speak")
        total = sum(len(t) for t in texts) if texts else len(req.text)
        if total > settings.tts.max_text_chars:
            raise ValidationError(f"The text is longer than {settings.tts.max_text_chars} characters", {"max_chars": settings.tts.max_text_chars, "chars": total})

        # language
        language = languages.normalize(req.language)
        langs = list(variant.languages or manifest.languages)
        if not languages.supports(langs, language):
            raise LanguageError(f"{info.title} does not speak {language}", {"language": language, "supported": langs})

        # voice
        voice, mode, references, voice_id = self._resolve_voice(req, variant, info.title, warnings)
        style = self._resolve_style(req, variant, mode, info.title)

        # streaming
        granularity = variant.stream_granularity or ("emulated" if stream else None)
        if stream and "stream_audio" not in caps:
            warnings.append("The model does not stream; Mizuoto sends each segment as it is ready")

        # parameters
        strict = settings.tts.strict_params if req.strict is None else req.strict
        params, param_warnings = check_params(manifest, variant.id, req.params.model_dump(), strict=strict, streaming=stream)
        warnings += param_warnings
        options = check_options(manifest, variant.id, req.engine_options)

        # segmentation
        if texts:
            segments, segmented_by = texts, "batch"
        elif req.text_stream:
            segments, segmented_by = [req.text], "engine"
        elif req.segmentation == "mizuoto" or "long_text" not in caps:
            segments, segmented_by = segment(req.text) or [req.text.strip()], "mizuoto"
        else:
            segments, segmented_by = [req.text.strip()], "engine"

        # device and output
        device = self._runtime.device_for(engine)
        out = req.output
        fmt = out.format or settings.tts.output_format
        out_rate = out.sample_rate or settings.tts.output_sample_rate or variant.sample_rate
        normalize = out.normalize or settings.tts.normalize
        licence_tags = [manifest.licence.id] if manifest.licence.gated or not manifest.licence.commercial else []
        es = settings.engines.get(engine)
        plan = SynthesisPlan(
            model=req.model,
            engine=engine,
            variant=variant.id,
            device=device,
            voice_mode=mode,
            voice_id=voice_id,
            language=language,
            segments=segments,
            segmented_by=segmented_by,
            params=params,
            options=options,
            references=references,
            output_format=fmt,
            model_sample_rate=variant.sample_rate,
            output_sample_rate=out_rate,
            normalize=normalize,
            stream_granularity=granularity,
            text_stream=req.text_stream,
            licence_tags=licence_tags,
            warnings=warnings,
        )
        return _Prepared(
            plan=plan,
            request=req,
            manifest=manifest,
            variant=variant,
            model_path=info.path,
            voice=voice,
            style=style,
            dtype=es.dtype if es else "auto",
            load_options=self._load_options(req.model, es),
            voice_name=self.voices.get(voice_id).slug if voice_id and self.voices is not None else None,
        )

    def _load_options(self, model_id: str, es: Any) -> dict[str, Any]:
        """The engine's load options (the worker layer adds the model's assets), checked here so a missing asset fails at plan time."""
        self._models.asset_paths(model_id)
        return dict(es.load_options) if es else {}

    def _reference(self, ref: ReferenceInput, variant: VariantInfo, warnings: list[str]) -> tuple[Reference, PlannedReference]:
        if ref.audio_id:
            audio = self._audio.get(ref.audio_id)
        elif ref.path:
            audio = self._audio.register_path(Path(ref.path))
        else:
            raise ValidationError("A reference needs an audio file (audio_id) or a path")
        limits = variant.reference
        transcript = ref.transcript.strip() if ref.transcript and ref.transcript.strip() else None
        reason = None
        if limits.min_seconds is not None and audio.duration < limits.min_seconds:
            reason = "too_short"
        elif limits.max_seconds is not None and audio.duration > limits.max_seconds:
            if self._settings.settings.tts.reference_policy == "trim":
                original = audio
                audio = self._audio.trimmed(audio, limits.max_seconds)
                transcript = None
                warnings.append(f"{original.name} was cut from {original.duration:.1f} s to {audio.duration:.1f} s for this model")
            else:
                reason = "too_long"
        elif limits.min_sample_rate is not None and audio.sample_rate < limits.min_sample_rate:
            reason = "sample_rate"
        if reason is not None:
            raise ReferenceInvalidError(
                f"{audio.name} cannot be a reference for this model ({reason.replace('_', ' ')}: {audio.duration:.1f} s at {audio.sample_rate} Hz)",
                {
                    "reason": reason,
                    "min_s": limits.min_seconds,
                    "max_s": limits.max_seconds,
                    "min_sample_rate": limits.min_sample_rate,
                    "actual_s": audio.duration,
                    "file": audio.name,
                    "hint": "mizuoto config set tts.reference_policy trim" if reason == "too_long" else None,
                },
            )
        wav = self._audio.canonical_wav(audio)
        language = languages.normalize(ref.language)
        reference = Reference(audio_path=str(wav), transcript=transcript, language=language, sample_rate=audio.sample_rate, duration=audio.duration, sha256=audio.sha256)
        planned = PlannedReference(audio_id=audio.id, name=audio.name, duration=audio.duration, sample_rate=audio.sample_rate, transcript=transcript, language=language)
        return reference, planned

    def _resolve_voice(self, req: SpeakRequest, variant: VariantInfo, title: str, warnings: list[str]) -> tuple[VoiceSpec, VoiceMode, list[PlannedReference], str | None]:
        caps = set(variant.capabilities)
        if req.voice:
            if req.voice_spec is not None and (req.voice_spec.references or req.voice_spec.preset or req.voice_spec.description or req.voice_spec.attributes):
                raise ValidationError("Give a library voice or a voice with the request, not both")
            if self.voices is None:
                raise ValidationError("The voice library is not available")
            return self._library_voice(req, variant, title, warnings)
        spec = req.voice_spec
        if spec is None or not (spec.references or spec.preset or spec.description or spec.attributes):
            if "auto_voice" not in caps:
                raise CapabilityError(
                    f"{title} needs a voice: a reference clip, a preset speaker or a description", {"capability": "auto_voice", "voice_modes": self._registry.voice_modes(variant)}
                )
            return VoiceSpec(), "auto", [], None
        given = [name for name, value in (("references", spec.references), ("preset", spec.preset), ("description", spec.description or spec.attributes)) if value]
        if len(given) > 1:
            raise ValidationError(f"Give one kind of voice, not {' and '.join(given)}", {"given": given})
        if spec.preset:
            if "preset_speaker" not in caps:
                raise CapabilityError(f"{title} has no preset speakers", {"capability": "preset_speaker"})
            speakers = self._models.speakers(req.model) if req.model else variant.speakers
            if speakers and spec.preset not in speakers:
                raise ValidationError(f"{title} has no speaker {spec.preset!r}", {"speakers": speakers})
            return VoiceSpec(preset=spec.preset), "preset", [], None
        if spec.description or spec.attributes:
            if spec.description and "voice_design_freeform" not in caps:
                raise CapabilityError(f"{title} cannot design a voice from a description", {"capability": "voice_design_freeform", "attributes": variant.design_attributes})
            if spec.attributes:
                if "voice_design" not in caps or not variant.design_attributes:
                    raise CapabilityError(f"{title} cannot design a voice from attributes", {"capability": "voice_design"})
                bad = {k: v for k, v in spec.attributes.items() if v not in variant.design_attributes.get(k, [])}
                if bad:
                    raise ValidationError(f"Unknown voice attributes: {', '.join(f'{k}={v}' for k, v in bad.items())}", {"attributes": variant.design_attributes})
            return VoiceSpec(description=spec.description, attributes=spec.attributes), "design", [], None
        if len(spec.references) > 1 and ("multi_reference" not in caps or len(spec.references) > variant.reference.max_references):
            raise CapabilityError(f"{title} takes {variant.reference.max_references if 'multi_reference' in caps else 1} reference clip(s)", {"capability": "multi_reference"})
        voice, mode, planned = self._clone(spec.references, variant, title, warnings)
        return voice, mode, planned, None

    def _library_voice(self, req: SpeakRequest, variant: VariantInfo, title: str, warnings: list[str]) -> tuple[VoiceSpec, VoiceMode, list[PlannedReference], str | None]:
        voices = self.voices
        assert voices is not None and req.voice is not None
        voice = voices.get(req.voice)
        fit = voices.fit_voice(voice, req.model, variant)
        if not fit.usable or fit.mode is None:
            raise CapabilityError(f"{title} cannot use the voice {voice.name}: {fit.reason}", {"voice": voice.id, "model": req.model, "reason": fit.reason})
        warnings += [w for w in fit.warnings if w not in warnings]
        if fit.mode == "preset":
            assert voice.preset is not None
            return VoiceSpec(preset=voice.preset.speaker, voice_id=voice.id), "preset", [], voice.id
        if fit.mode == "design":
            assert voice.design is not None
            return VoiceSpec(description=voice.design.description, attributes=voice.design.attributes, voice_id=voice.id), "design", [], voice.id
        limit = variant.reference.max_references if "multi_reference" in set(variant.capabilities) else 1
        inputs = [
            ReferenceInput(path=str(voices.folder(voice) / r.file), transcript=r.transcript, language=r.language or voice.language)
            for r in voices.ordered_references(voice)[:limit]
        ]
        spec, mode, planned = self._clone(inputs, variant, title, warnings)
        spec.voice_id = voice.id
        if fit.weights:
            spec.weights = voices.weight_paths(voice, req.model.split("/", 1)[0])
        return spec, mode, planned, voice.id

    def _clone(self, inputs: list[ReferenceInput], variant: VariantInfo, title: str, warnings: list[str]) -> tuple[VoiceSpec, VoiceMode, list[PlannedReference]]:
        """References for cloning: the voice mode they allow, and the transcripts still to make."""
        caps = set(variant.capabilities)
        if "clone" not in caps and "clone_no_transcript" not in caps:
            raise CapabilityError(f"{title} cannot clone a voice from a clip", {"capability": "clone"})
        refs = [self._reference(r, variant, warnings) for r in inputs]
        missing = [(ref, planned) for ref, planned in refs if planned.transcript is None]
        mode: VoiceMode = "clone" if "clone" in caps else "clone_no_transcript"
        if missing and "clone" in caps:
            asr = self._asr
            if self._settings.settings.tts.auto_transcribe and asr is not None and asr.available():
                for ref, planned in missing:
                    hit = asr.cached(ref.sha256 or "")
                    if hit is not None:
                        ref.transcript = planned.transcript = hit.text
                    else:
                        planned.transcribe = True
            elif "clone_no_transcript" in caps:
                mode = "clone_no_transcript"
                warnings.append("A reference has no transcript, so the model follows the voice less closely (add it, or install speech recognition)")
            else:
                reason = asr.unavailable_reason() if asr is not None else "speech recognition is not set up"
                raise CapabilityError(
                    f"{title} needs the transcript of each reference clip, and {reason}",
                    {
                        "capability": "clone_no_transcript",
                        "missing": [p.name for _, p in missing],
                        "hint": "Add the transcript (--ref-text), or download speech recognition: mizuoto model download asr/whisper-large-v3-turbo",
                    },
                )
        if mode == "clone_no_transcript":
            for ref, planned in refs:
                ref.transcript = planned.transcript = None
        return VoiceSpec(references=[r for r, _ in refs]), mode, [p for _, p in refs]

    def _resolve_style(self, req: SpeakRequest, variant: VariantInfo, mode: VoiceMode, title: str) -> StyleSpec:
        caps = set(variant.capabilities)
        style = req.style
        if style is None:
            return StyleSpec()
        out = StyleSpec()
        if style.instruct and style.instruct.strip():
            if "instruct" not in caps:
                raise CapabilityError(f"{title} does not take instructions", {"capability": "instruct"})
            if variant.instruct_requires and mode not in variant.instruct_requires:
                raise CapabilityError(
                    f"{title} takes an instruction only with a {' or '.join(variant.instruct_requires)} voice", {"capability": "instruct", "requires": variant.instruct_requires}
                )
            out.instruct = style.instruct.strip()
        e = style.emotion
        if e is not None:
            emotion = EmotionSpec(strength=e.strength)
            if e.vector:
                if "emotion_vector" not in caps:
                    raise CapabilityError(f"{title} does not take an emotion vector", {"capability": "emotion_vector"})
                unknown = sorted(set(e.vector) - set(variant.emotions))
                if unknown:
                    raise ValidationError(f"Unknown emotions: {', '.join(unknown)}", {"emotions": variant.emotions})
                emotion.vector = {k: float(v) for k, v in e.vector.items()}
            if e.audio_id or e.path:
                if "emotion_audio" not in caps:
                    raise CapabilityError(f"{title} does not take an emotion clip", {"capability": "emotion_audio"})
                audio = self._audio.get(e.audio_id) if e.audio_id else self._audio.register_path(Path(e.path or ""))
                emotion.audio_path = str(self._audio.canonical_wav(audio))
            if e.text:
                if "emotion_text" not in caps:
                    raise CapabilityError(f"{title} does not take an emotion description", {"capability": "emotion_text"})
                emotion.text = e.text
            out.emotion = emotion
        return out

    # -- running --------------------------------------------------------------------------

    def speak(self, req: SpeakRequest, *, wait: bool = False, origin: str | None = None, stream: StreamSink | None = None) -> Job:
        """Queue a synthesis (``wait``: run it on this thread and return it finished)."""
        if req.text_stream and (stream is None or stream.text is None):
            raise ValidationError("Streamed text needs a stream: the /tts/stream WebSocket")
        request = req.model_dump(mode="json")
        if stream is not None:
            request["_stream"] = True
        with self._sink_lock:
            self._next_sink = stream
            try:
                return self._jobs.run_now("speak", request, origin) if wait else self._jobs.submit("speak", request, origin)
            finally:
                self._next_sink = None

    def prepare(self, voice: str, model: str, *, wait: bool = False, origin: str | None = None) -> Job:
        """Compute a library voice's derived state for a model ahead of its first use."""
        request = {"voice": voice, "model": model}
        self._prepare(self._prepare_request(request))
        return self._jobs.run_now("voice_prepare", request, origin) if wait else self._jobs.submit("voice_prepare", request, origin)

    @staticmethod
    def _prepare_request(request: dict[str, Any]) -> SpeakRequest:
        return SpeakRequest(model=str(request["model"]), text=".", voice=str(request["voice"]))

    def _prepare_job(self, request: dict[str, Any]) -> JobSpec:
        prepared = self._prepare(self._prepare_request(request))
        plan = prepared.plan
        resources = {f"engine:{plan.engine}:{plan.device}"} | ({f"gpu:{plan.device}"} if plan.device != "cpu" else set())

        def run(ctx: JobContext) -> dict[str, Any]:
            self._transcribe_references(prepared, ctx)
            ctx.message("Loading the model")
            eng = self._load(prepared, ctx)
            self._prepare_voice(prepared, eng, ctx)
            return {"voice_mode": plan.voice_mode, "files": sorted((prepared.voice.cache or {}).keys())}

        return JobSpec(title=f"Prepare {request['voice']} for {plan.model}", run=run, resources=frozenset(resources))

    # -- engine-specific operations ------------------

    def engine_op(self, model: str, op: str, payload: dict[str, Any] | None = None, *, wait: bool = False, origin: str | None = None) -> Job:
        """Run an operation the engine's manifest declares (``switch_weights``, ``codec_encode``…) with ``model`` loaded."""
        self._op_target(model, op)
        request = {"model": model, "op": op, "payload": dict(payload or {})}
        return self._jobs.run_now("engine_op", request, origin) if wait else self._jobs.submit("engine_op", request, origin)

    def _op_target(self, model: str, op: str) -> tuple[Any, EngineManifest, str]:
        info = self._models.require(model)
        self._registry.check_enabled(info.engine)
        self._registry.check_licence(info.engine)
        self._registry.check_available(info.engine)
        manifest = self._registry.manifest(info.engine)
        if op not in manifest.ops:
            raise CapabilityError(f"{manifest.title} has no operation {op!r}", {"ops": manifest.ops})
        self._models.asset_paths(model)
        return info, manifest, self._runtime.device_for(info.engine)

    def _op_job(self, request: dict[str, Any]) -> JobSpec:
        info, manifest, device = self._op_target(str(request["model"]), str(request["op"]))
        resources = {f"engine:{info.engine}:{device}"} | ({f"gpu:{device}"} if device != "cpu" else set())
        es = self._settings.settings.engines.get(info.engine)

        def run(ctx: JobContext) -> dict[str, Any]:
            ctx.message(f"Starting {manifest.title}")
            ctx.message("Loading the model")
            model = ModelRef(info.id, info.engine, info.variant, info.path)
            eng = self._runtime.loaded(model, device, es.dtype if es else "auto", dict(es.load_options) if es else {}, self._request(ctx, "load"))
            ctx.message(str(request["op"]))
            call = self._request(ctx, "op")
            result = dict(eng.call(str(request["op"]), dict(request["payload"]), call))
            result.setdefault("timings", {})["total"] = call.timings.get("total")
            return result

        return JobSpec(title=f"{request['op']} on {info.id}", run=run, resources=frozenset(resources))

    # -- conversion and editing ----------------------------------------------------------

    def _model_for(self, model_id: str, capability: str, what: str) -> tuple[Any, EngineManifest, VariantInfo]:
        info = self._models.require(model_id)
        self._registry.check_enabled(info.engine)
        self._registry.check_licence(info.engine)
        self._registry.check_available(info.engine)
        manifest = self._registry.manifest(info.engine)
        variant = self._registry.variant(info.engine, info.variant)
        if capability not in variant.capabilities:
            raise CapabilityError(f"{info.title} cannot {what}", {"capability": capability, "model": model_id})
        return info, manifest, variant

    def _source(self, source: SourceInput) -> Path:
        if source.audio_id:
            audio = self._audio.get(source.audio_id)
        elif source.path:
            audio = self._audio.register_path(Path(source.path))
        else:
            raise ValidationError("Give the recording: an audio file (audio_id) or a path")
        return self._audio.canonical_wav(audio)

    def _transform(
        self,
        op: str,
        kind: str,
        title: str,
        info: Any,
        manifest: EngineManifest,
        variant: VariantInfo,
        params: dict[str, Any],
        options: dict[str, Any],
        strict: bool | None,
        output: OutputSpec,
        payload: dict[str, Any],
        warnings: list[str],
    ) -> _Transform:
        settings = self._settings.settings
        checked, param_warnings = check_params(manifest, variant.id, params, strict=settings.tts.strict_params if strict is None else strict)
        payload = {**payload, "params": checked, "options": check_options(manifest, variant.id, options)}
        device = self._runtime.device_for(manifest.id)
        es = settings.engines.get(manifest.id)
        return _Transform(
            op=op,
            kind=kind,
            title=title,
            model=info.id,
            engine=manifest.id,
            variant=variant.id,
            device=device,
            manifest=manifest,
            model_path=info.path,
            payload=payload,
            output=output,
            model_sample_rate=variant.sample_rate,
            warnings=warnings + param_warnings,
            dtype=es.dtype if es else "auto",
            load_options=self._load_options(info.id, es),
        )

    def _prepare_convert(self, req: ConvertRequest) -> _Transform:
        info, manifest, variant = self._model_for(req.model, "voice_conversion", "convert a recording to another voice")
        warnings: list[str] = []
        voice_id = None
        target = "a clip"
        if req.voice:
            if self.voices is None:
                raise ValidationError("The voice library is not available")
            voice = self.voices.get(req.voice)
            if not voice.references:
                raise CapabilityError(f"{voice.name} has no reference clip to convert to", {"voice": voice.id})
            ref = self.voices.ordered_references(voice)[0]
            inputs = [ReferenceInput(path=str(self.voices.folder(voice) / ref.file), transcript=ref.transcript)]
            voice_id, target = voice.id, voice.name
        elif req.voice_spec and req.voice_spec.references:
            inputs = req.voice_spec.references[:1]
        else:
            raise ValidationError("Give the target voice: a library voice or a reference clip")
        reference, _ = self._reference(inputs[0], variant, warnings)
        task = VoiceConversionTask(source_path=str(self._source(req.source)), voice=VoiceSpec(references=[reference], voice_id=voice_id))
        t = self._transform(
            "convert",
            "conversion",
            f"Convert to {target}",
            info,
            manifest,
            variant,
            req.params.model_dump(),
            req.engine_options,
            req.strict,
            req.output,
            to_dict(task),
            warnings,
        )
        t.voice_id = voice_id
        return t

    def _prepare_edit(self, req: EditRequest) -> _Transform:
        info, manifest, variant = self._model_for(req.model, "speech_edit", "edit speech")
        if not req.transcript.strip():
            raise ValidationError("Give the recording's whole text after the edit")
        for span in req.edits:
            if span.end <= span.start:
                raise ValidationError(f"An edit ends before it starts ({span.start}–{span.end} s)")
        task = SpeechEditTask(
            audio_path=str(self._source(req.source)),
            transcript=req.transcript.strip(),
            edits=[EditSpan(start=e.start, end=e.end, text=e.text, duration=e.duration) for e in req.edits],
        )
        t = self._transform("edit", "edit", "Edit speech", info, manifest, variant, req.params.model_dump(), req.engine_options, req.strict, req.output, to_dict(task), [])
        t.text = req.transcript.strip()
        return t

    def convert(self, req: ConvertRequest, *, wait: bool = False, origin: str | None = None) -> Job:
        """Queue a voice conversion (``wait``: run it here and return it finished)."""
        if req.ignored_fields:
            logger.warning("Voice conversion: ignored fields the request does not have: %s", ", ".join(req.ignored_fields))
        self._prepare_convert(req)
        request = req.model_dump(mode="json")
        return self._jobs.run_now("convert", request, origin) if wait else self._jobs.submit("convert", request, origin)

    def edit(self, req: EditRequest, *, wait: bool = False, origin: str | None = None) -> Job:
        """Queue a speech edit (``wait``: run it here and return it finished)."""
        if req.ignored_fields:
            logger.warning("Speech edit: ignored fields the request does not have: %s", ", ".join(req.ignored_fields))
        self._prepare_edit(req)
        request = req.model_dump(mode="json")
        return self._jobs.run_now("edit", request, origin) if wait else self._jobs.submit("edit", request, origin)

    def _transform_spec(self, t: _Transform) -> JobSpec:
        resources = {INFERENCE, f"engine:{t.engine}:{t.device}"} | ({f"gpu:{t.device}"} if t.device != "cpu" else set())
        return JobSpec(title=t.title, run=lambda ctx: self._run_transform(t, ctx), resources=frozenset(resources))

    def _convert_job(self, request: dict[str, Any]) -> JobSpec:
        return self._transform_spec(self._prepare_convert(ConvertRequest.model_validate(request)))

    def _edit_job(self, request: dict[str, Any]) -> JobSpec:
        return self._transform_spec(self._prepare_edit(EditRequest.model_validate(request)))

    def _run_transform(self, t: _Transform, ctx: JobContext) -> dict[str, Any]:
        import numpy as np

        started = time.perf_counter()
        ctx.message(f"Starting {t.manifest.title}")
        ctx.message("Loading the model")
        eng = self._runtime.loaded(ModelRef(t.model, t.engine, t.variant, t.model_path), t.device, t.dtype, t.load_options, self._request(ctx, "load"))
        load_seconds = time.perf_counter() - started
        pieces: list[np.ndarray] = []
        rates: set[int] = set()

        def on_chunk(chunk: AudioChunk) -> None:
            pieces.append(chunk.samples)
            rates.add(chunk.sample_rate)

        ctx.message("Converting" if t.op == "convert" else "Editing")
        request = self._request(ctx, t.op)
        stream = eng.convert(conversion_task_from_dict(t.payload), request) if t.op == "convert" else eng.edit(edit_task_from_dict(t.payload), request)
        _drain(stream, on_chunk)
        ctx.check_cancel()
        rate = rates.pop() if rates else t.model_sample_rate
        settings = self._settings.settings
        audio = dsp.normalize(np.concatenate(pieces) if pieces else np.zeros(0, np.float32), t.output.normalize or settings.tts.normalize)
        out_rate = t.output.sample_rate or settings.tts.output_sample_rate or rate
        if out_rate != rate:
            audio = resample(audio, rate, out_rate)
        fmt = t.output.format or settings.tts.output_format
        path = self._outputs.new_path({"voice": t.op, "model": t.model.split("/", 1)[1], "engine": t.engine}, fmt)
        encode(audio, out_rate, path, fmt)
        duration = len(audio) / out_rate if out_rate else 0.0
        total = time.perf_counter() - started
        warnings = t.warnings + [w for w in request.warnings if w not in t.warnings]
        output = Output(
            id=self._outputs.new_id(),
            kind=t.kind,  # ty: ignore[invalid-argument-type]
            name=t.output.name or path.stem,
            path=str(path),
            format=fmt,
            sample_rate=out_rate,
            duration=duration,
            size=path.stat().st_size,
            job_id=ctx.job_id,
            engine=t.engine,
            model=t.model,
            voice_id=t.voice_id,
            text=t.text,
            params=dict(request.applied),
            meta={
                "warnings": warnings,
                "timings": {"load": load_seconds, "total": total, "rtf": total / duration if duration else None},
                "engine_options": t.payload.get("options", {}),
            },
            licence_tags=[t.manifest.licence.id] if t.manifest.licence.gated or not t.manifest.licence.commercial else [],
            created_at=now_iso(),
        )
        self._outputs.add(output, ctx.job_id)
        return {"output_ids": [output.id], "output_id": output.id, "warnings": warnings, "applied_params": output.params}

    def _speak_job(self, request: dict[str, Any]) -> JobSpec:
        stream = bool(request.get("_stream"))
        req = SpeakRequest.model_validate({k: v for k, v in request.items() if not k.startswith("_")})
        prepared = self._prepare(req, stream)
        sink = self._next_sink
        plan = prepared.plan
        resources = {INFERENCE, f"engine:{plan.engine}:{plan.device}"}
        if plan.device != "cpu":
            resources.add(f"gpu:{plan.device}")
        first = (req.texts or [req.text])[0].strip().replace("\n", " ")
        title = f"Speak: {first[:48]}{'…' if len(first) > 48 else ''}"
        group = self._batch_group(prepared) if sink is None and not stream else None
        return JobSpec(title=title, run=lambda ctx: self._run(prepared, ctx, sink), resources=frozenset(resources), priority=0, group=group, payload=prepared)

    def _batch_group(self, p: _Prepared) -> str | None:
        """Requests that may run as one batch (micro-batching): one text each, no seed (a batch draws its random
        numbers differently), on a variant that takes a batch; the same model, device, voice, style, language,
        parameters and options."""
        plan = p.plan
        if not self._settings.settings.tts.micro_batch or "batch" not in p.variant.capabilities or (p.variant.max_batch or 0) < 2:
            return None
        if plan.segmented_by == "batch" or len(plan.segments) != 1 or plan.text_stream or plan.params.get("seed") is not None:
            return None
        key = {
            "model": plan.model,
            "device": plan.device,
            "voice": to_dict(p.voice),
            "style": to_dict(p.style),
            "language": plan.language,
            "params": plan.params,
            "options": plan.options,
            "references": [r.model_dump(mode="json") for r in plan.references],
        }
        return "speak:" + hashlib.sha256(json.dumps(key, sort_keys=True, default=str).encode("utf-8")).hexdigest()[:24]

    def _run(self, p: _Prepared, ctx: JobContext, sink: StreamSink | None) -> dict[str, Any]:

        plan = p.plan
        started = time.perf_counter()
        ctx.message(f"Starting {p.manifest.title}")
        try:
            self._transcribe_references(p, ctx)
            ctx.check_cancel()
            ctx.message("Loading the model")
            load_started = time.perf_counter()
            eng = self._load(p, ctx)
            load_seconds = time.perf_counter() - load_started
            ctx.check_cancel()
            self._prepare_voice(p, eng, ctx)
            pieces: dict[int, list[np.ndarray]] = {}
            rates: set[int] = set()
            first_chunk: list[float] = []
            applied: dict[str, Any] = {}
            memory: dict[str, Any] = {}
            warnings = list(plan.warnings)
            if sink is not None:
                sink.start(plan.model_sample_rate, plan.stream_granularity)

            def on_chunk(chunk: AudioChunk) -> None:
                pieces.setdefault(chunk.segment, []).append(chunk.samples)
                rates.add(chunk.sample_rate)
                if not first_chunk:
                    first_chunk.append(time.perf_counter() - started)
                if sink is not None:
                    sink.chunk(chunk.samples, chunk.sample_rate, chunk.segment)

            batch = plan.segmented_by == "batch"
            calls = [plan.segments] if batch else [[s] for s in plan.segments]
            # Micro-batching: queued requests that differ only in their text go in this one's call.
            adopted = ctx.adopt((p.variant.max_batch or 1) - 1) if sink is None and ctx.payload is p and not batch else []
            if adopted:
                return self._run_batch(p, ctx, eng, adopted, started, load_seconds)
            feed = sink.text if sink is not None and plan.text_stream else None
            if plan.text_stream and feed is None:
                raise ValidationError("Streamed text needs a stream: the /tts/stream WebSocket")
            ctx.message("Speaking")
            for index, texts in enumerate(calls):
                ctx.check_cancel()
                task = SynthesisTask(
                    text=texts[0],
                    texts=texts if batch else None,
                    language=plan.language,
                    voice=p.voice,
                    style=p.style,
                    params=plan.params,
                    options=plan.options,
                    segment=index,
                    segments=len(calls),
                    stream=sink is not None,
                    text_stream=feed is not None,
                )
                request = self._request(ctx, str(index), text_input=feed.text_input if feed is not None else None)
                _drain(eng.synthesize(task, request), on_chunk)
                memory = _merge_memory(memory, request.memory)
                applied.update({**request.applied, **merge_attention(applied.get("attention"), request.applied.get("attention"))})
                warnings += [w for w in request.warnings if w not in warnings]
                ctx.progress((index + 1) / len(calls))
            ctx.check_cancel()
            if len(rates) > 1:
                raise ValidationError(f"The engine returned audio at several sample rates: {sorted(rates)}")
            rate = rates.pop() if rates else plan.model_sample_rate
            if feed is not None:
                p.request = p.request.model_copy(update={"text": p.request.text + feed.text})
            output_ids = self._store(p, ctx, pieces, rate, applied, warnings, started, load_seconds, first_chunk[0] if first_chunk else None, memory)
        except Exception as e:
            if sink is not None:
                sink.error(e)
            raise
        if sink is not None:
            sink.end(output_ids)
        return {"output_ids": output_ids, "output_id": output_ids[0] if output_ids else None, "warnings": warnings, "applied_params": applied, "plan": plan.model_dump(mode="json")}

    def _run_batch(self, p: _Prepared, ctx: JobContext, eng: EngineHandle, adopted: list[JobContext], started: float, load_seconds: float) -> dict[str, Any]:
        """This request and the adopted ones in one batched call; each job gets its own output."""

        members: list[tuple[_Prepared, JobContext]] = [(p, ctx), *((a.payload, a) for a in adopted)]
        plan = p.plan
        pieces: dict[int, list[np.ndarray]] = {}
        rates: set[int] = set()
        first_chunk: list[float] = []

        def on_chunk(chunk: AudioChunk) -> None:
            pieces.setdefault(chunk.segment, []).append(chunk.samples)
            rates.add(chunk.sample_rate)
            if not first_chunk:
                first_chunk.append(time.perf_counter() - started)

        texts = [m.plan.segments[0] for m, _ in members]
        ctx.message(f"Speaking (a batch of {len(texts)})")
        ctx.log(f"Micro-batch with {', '.join(a.job_id for a in adopted)}")
        try:
            task = SynthesisTask(text=texts[0], texts=texts, language=plan.language, voice=p.voice, style=p.style, params=plan.params, options=plan.options)
            request = self._request(ctx, "batch")
            _drain(eng.synthesize(task, request), on_chunk)
            if len(rates) > 1:
                raise ValidationError(f"The engine returned audio at several sample rates: {sorted(rates)}")
        except BaseException:
            for a in adopted:
                a.release()  # each runs on its own
            raise
        rate = rates.pop() if rates else plan.model_sample_rate
        applied = dict(request.applied)
        warnings = list(dict.fromkeys([*plan.warnings, *request.warnings]))
        own: dict[str, Any] = {}
        for index, (member, job) in enumerate(members):
            try:
                output_ids = self._store(member, job, {0: pieces.get(index, [])}, rate, applied, warnings, started, load_seconds, first_chunk[0] if first_chunk else None)
                done = {
                    "output_ids": output_ids,
                    "output_id": output_ids[0] if output_ids else None,
                    "warnings": warnings,
                    "applied_params": applied,
                    "plan": member.plan.model_dump(mode="json"),
                    "batch": {"leader": ctx.job_id, "size": len(members)},
                }
            except Exception as e:
                if job is ctx:
                    raise
                job.fail(e)
                continue
            if job is ctx:
                own = done
            else:
                job.complete(done)
        return own

    def _load(self, p: _Prepared, ctx: JobContext) -> EngineHandle:
        """The handle of the plan's model, loaded on its device (the job's cancel and progress reach the load)."""
        plan = p.plan
        model = ModelRef(plan.model, plan.engine, plan.variant, p.model_path)
        return self._runtime.loaded(model, plan.device, p.dtype, p.load_options, self._request(ctx, "load"))

    @staticmethod
    def _request(ctx: JobContext, name: str, text_input: queue.Queue[tuple[str, bool]] | None = None) -> EngineRequest:
        """An engine request of the job: its cancel; loading progress named in the job's message."""

        def progress(stage: str, value: float | None, message: str | None) -> None:
            if stage.startswith("loading:"):
                ctx.message(f"Loading the model: {stage.split(':', 1)[1]}")

        return EngineRequest(id=f"{ctx.job_id}-{name}", cancel=ctx.cancel_event, text_input=text_input, progress=progress)

    def _transcribe_references(self, p: _Prepared, ctx: JobContext) -> None:
        """Make the reference transcripts the plan left to the speech recognition engine."""
        todo = [(i, pr) for i, pr in enumerate(p.plan.references) if pr.transcribe and not pr.transcript]
        if not todo:
            return
        if self._asr is None:
            raise ValidationError("Speech recognition is not set up")
        for i, pr in todo:
            ctx.message(f"Transcribing {pr.name}")
            t = self._asr.transcribe(self._audio.get(pr.audio_id), pr.language, ctx=ctx)
            ctx.log(f"Transcript of {pr.name} ({t.language}): {t.text}")
            pr.transcript = t.text
            p.voice.references[i].transcript = t.text
            if p.plan.voice_id and self.voices is not None:
                try:
                    self.voices.set_transcript(p.plan.voice_id, p.voice.references[i].sha256 or "", t.text, "asr")
                except NotFoundError:
                    pass  # a trimmed copy, not one of the voice's own clips

    def _prepare_voice(self, p: _Prepared, eng: EngineHandle, ctx: JobContext) -> None:
        """Derived voice state (a reference prompt, a speaker embedding), made once per voice and model."""
        plan = p.plan
        if plan.voice_mode not in ("clone", "clone_no_transcript") or not p.voice.references:
            return
        key_source = {
            "model": plan.model,
            "revision": self._models.revision(plan.model),
            "references": [(r.sha256, r.transcript) for r in p.voice.references],
            "mode": plan.voice_mode,
            "mizuoto": VERSION,
        }
        # A licence-gated engine is released on its own: its version keys the cache too (built-in engines are Mizuoto's).
        engine_version = distribution_version(plan.engine)
        if engine_version is not None:
            key_source["engine_version"] = engine_version
        key = hashlib.sha256(json.dumps(key_source, sort_keys=True).encode("utf-8")).hexdigest()[:20]
        if plan.voice_id and self.voices is not None:
            base = self.voices.cache_dir(self.voices.get(plan.voice_id), plan.engine, plan.model)
        else:
            base = self._settings.cache_dir / "voices" / plan.engine / plan.model.split("/", 1)[-1]
        folder = base / key
        marker = folder / "cache.json"
        files: dict[str, Any] | None = None
        try:
            files = json.loads(marker.read_text(encoding="utf-8")).get("files")
            if files and not all(Path(f).exists() for f in files.values()):
                files = None
        except (OSError, ValueError):
            files = None
        if files is None:
            ctx.message("Preparing the voice")
            cache = eng.prepare_voice(p.voice, folder, self._request(ctx, "voice"))
            files = dict(cache.files) if cache is not None else {}
            folder.mkdir(parents=True, exist_ok=True)
            meta = cache.meta if cache is not None else {}
            marker.write_text(json.dumps({"files": files, "meta": meta, "key": key_source}, indent=2, ensure_ascii=False), encoding="utf-8")
        p.voice.cache = files or None

    def _store(
        self,
        p: _Prepared,
        ctx: JobContext,
        pieces: dict[int, list[np.ndarray]],
        rate: int,
        applied: dict[str, Any],
        warnings: list[str],
        started: float,
        load_seconds: float,
        first_chunk: float | None,
        memory: dict[str, Any] | None = None,
    ) -> list[str]:
        import numpy as np

        plan = p.plan
        if plan.segmented_by == "batch":
            groups = [(plan.segments[i], [np.concatenate(pieces.get(i, [np.zeros(0, np.float32)]))]) for i in range(len(plan.segments))]
        else:
            segments = [np.concatenate(pieces[i]) if pieces.get(i) else np.zeros(0, np.float32) for i in range(len(plan.segments))]
            groups = [(p.request.text, segments)]
        output_ids: list[str] = []
        for text, segs in groups:
            silence = plan.params.get("segment_silence_ms") if plan.segmented_by == "mizuoto" else 0
            if plan.segmented_by == "mizuoto" and silence is None:
                silence = MIZUOTO_SEGMENT_PAUSE_MS
            audio = dsp.assemble(segs, rate, silence_ms=int(silence or 0))
            audio = dsp.normalize(audio, plan.normalize)
            out_rate = plan.output_sample_rate
            if out_rate != rate:
                audio = resample(audio, rate, out_rate)
            name_voice = p.voice_name or p.voice.preset or plan.voice_mode
            path = self._outputs.new_path({"voice": name_voice, "model": plan.model.split("/", 1)[1], "engine": plan.engine}, plan.output_format)
            encode(audio, out_rate, path, plan.output_format)
            duration = len(audio) / out_rate if out_rate else 0.0
            total = time.perf_counter() - started
            output = Output(
                id=self._outputs.new_id(),
                kind="line" if p.request.output.project else "speech",
                name=p.request.output.name or path.stem,
                path=str(path),
                format=plan.output_format,
                sample_rate=out_rate,
                duration=duration,
                size=path.stat().st_size,
                job_id=ctx.job_id,
                engine=plan.engine,
                model=plan.model,
                voice_id=plan.voice_id,
                text=text,
                language=plan.language,
                params=applied,
                meta={
                    "voice_mode": plan.voice_mode,
                    "warnings": warnings,
                    "timings": {"load": load_seconds, "first_chunk": first_chunk, "total": total, "rtf": total / duration if duration else None},
                    "segments": len(plan.segments) if plan.segmented_by != "batch" else 1,
                    "engine_options": plan.options,
                    **({"memory": memory} if memory else {}),
                    **({"project": p.request.output.project} if p.request.output.project else {}),
                },
                licence_tags=plan.licence_tags,
                created_at=now_iso(),
            )
            self._outputs.add(output, ctx.job_id)
            output_ids.append(output.id)
        return output_ids


def _drain(stream: AudioStream, on_chunk: Callable[[AudioChunk], None]) -> None:
    """Hand on each chunk as the engine makes it; the stream is closed whatever happens."""
    with contextlib.closing(stream):
        for chunk in stream:
            on_chunk(chunk)


def merge_attention(total: dict[str, Any] | None, one: dict[str, Any] | None) -> dict[str, Any]:
    """``{"attention": …}``: a request's attention report over its calls (the counts add up), or nothing."""
    if not one:
        return {"attention": total} if total else {}
    if not total:
        return {"attention": one}
    used = dict(total.get("used", {}))
    for name, n in one.get("used", {}).items():
        used[name] = used.get(name, 0) + n
    fallbacks = {k: dict(v) for k, v in total.get("fallbacks", {}).items()}
    for key, reasons in one.get("fallbacks", {}).items():
        mine = fallbacks.setdefault(key, {})
        for reason, n in reasons.items():
            mine[reason] = mine.get(reason, 0) + n
    return {"attention": {"policy": one.get("policy"), "used": used, "fallbacks": fallbacks}}


def _merge_memory(total: dict[str, Any], one: dict[str, Any]) -> dict[str, Any]:
    """A request's memory report over its calls: streamed bytes add up, the residency is the last call's."""
    if not one:
        return total
    return {**one, "streamed_mb": round(float(total.get("streamed_mb", 0.0)) + float(one.get("streamed_mb", 0.0)), 1)}
