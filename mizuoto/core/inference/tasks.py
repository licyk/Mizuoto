"""What an engine is asked to do and what it gives back: plain standard-library dataclasses.

The one definition of the task types. The core builds the requests (``TtsService``); engines take them through
``mizuoto.runtime.engine``, which re-exports them, and the remote agent passes them on as JSON.

Every path is absolute and every audio file already exists on the engine's machine; the core has
validated the request against the model's manifest, so an engine maps rather than checks.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any, TypeVar

T = TypeVar("T")


@dataclass
class Reference:
    """A reference clip: the audio, and what is said in it."""

    audio_path: str
    transcript: str | None = None
    language: str | None = None
    sample_rate: int | None = None
    duration: float | None = None
    sha256: str | None = None


@dataclass
class VoiceSpec:
    """Who speaks. The host sets exactly the fields the voice mode needs."""

    references: list[Reference] = field(default_factory=list)
    preset: str | None = None
    """A preset speaker of the model."""
    description: str | None = None
    """A free-text voice description (voice design)."""
    attributes: dict[str, str] | None = None
    """A structured voice description: gender, age, pitch, accent…"""
    weights: dict[str, Any] | None = None
    """Fine-tuned weights of a trained voice: engine-specific file paths."""
    cache: dict[str, Any] | None = None
    """Paths of the derived voice state ``prepare_voice`` produced earlier, when cached."""
    voice_id: str | None = None


@dataclass
class EmotionSpec:
    audio_path: str | None = None
    vector: dict[str, float] | None = None
    text: str | None = None
    strength: float | None = None


@dataclass
class StyleSpec:
    instruct: str | None = None
    emotion: EmotionSpec | None = None


@dataclass
class SynthesisTask:
    """One synthesis call. ``params`` are the common parameters that were set (validated by the host,
    in their common names); ``options`` are the engine's own options."""

    text: str
    language: str | None = None
    voice: VoiceSpec = field(default_factory=VoiceSpec)
    style: StyleSpec = field(default_factory=StyleSpec)
    params: dict[str, Any] = field(default_factory=dict)
    options: dict[str, Any] = field(default_factory=dict)
    texts: list[str] | None = None
    """A batch: one output per text (models with the ``batch`` capability)."""
    segment: int = 0
    segments: int = 1
    stream: bool = False
    """The caller plays the audio as it comes: an engine that can stream finer (tokens) does."""
    text_stream: bool = False
    """``text`` is only the beginning: the rest arrives while the engine speaks (``RequestContext.text_pieces``)."""


@dataclass
class VoiceConversionTask:
    source_path: str
    voice: VoiceSpec = field(default_factory=VoiceSpec)
    params: dict[str, Any] = field(default_factory=dict)
    options: dict[str, Any] = field(default_factory=dict)


@dataclass
class EditSpan:
    """Seconds ``start``–``end`` of the recording are replaced by speech of ``text``."""

    start: float
    end: float
    text: str
    duration: float | None = None
    """Seconds the new speech lasts; None: as long as the span."""


@dataclass
class SpeechEditTask:
    audio_path: str
    transcript: str
    edits: list[EditSpan] = field(default_factory=list)
    params: dict[str, Any] = field(default_factory=dict)
    options: dict[str, Any] = field(default_factory=dict)


# -- what loading takes and what an engine gives back ------------------------------------------


@dataclass(frozen=True)
class OptionSpec:
    """One engine option, mirrored by the engine's manifest (a test keeps them equal)."""

    type: str
    """float, int, bool, str or enum."""
    default: Any = None
    min: float | None = None
    max: float | None = None
    choices: tuple[str, ...] | None = None


@dataclass
class LoadSpec:
    model_id: str
    model_dir: Path | None
    variant: str
    device: str
    dtype: str = "auto"
    options: dict[str, Any] = field(default_factory=dict)


@dataclass
class Description:
    """What a loaded model really offers, which may refine its manifest."""

    sample_rate: int
    capabilities: list[str] = field(default_factory=list)
    speakers: list[str] = field(default_factory=list)
    languages: list[str] = field(default_factory=list)
    vram_mb: float | None = None
    ignored_params: list[str] = field(default_factory=list)
    """Parameters this instance accepts but does not apply (e.g. sampling under an accelerated path)."""


@dataclass
class AudioChunk:
    """A piece of audio: float32 mono samples (a numpy array or anything with ``tobytes``)."""

    samples: Any
    sample_rate: int
    segment: int = 0
    final: bool = False


@dataclass
class VoiceCache:
    """Derived voice state written under the ``cache_dir`` the core gave, keyed by the core."""

    files: dict[str, str]
    meta: dict[str, Any] = field(default_factory=dict)


def to_dict(task: Any) -> dict[str, Any]:
    return asdict(task)


def _build(cls: type[T], data: dict[str, Any]) -> T:
    known = {f.name for f in fields(cls)}  # ty: ignore[invalid-argument-type]
    return cls(**{k: v for k, v in data.items() if k in known})


def voice_from_dict(data: dict[str, Any]) -> VoiceSpec:
    voice = _build(VoiceSpec, data)
    voice.references = [_build(Reference, r) if isinstance(r, dict) else r for r in voice.references]
    return voice


def style_from_dict(data: dict[str, Any]) -> StyleSpec:
    style = _build(StyleSpec, data)
    if isinstance(style.emotion, dict):
        style.emotion = _build(EmotionSpec, style.emotion)
    return style


def synthesis_task_from_dict(data: dict[str, Any]) -> SynthesisTask:
    task = _build(SynthesisTask, data)
    task.voice = voice_from_dict(data.get("voice") or {})
    task.style = style_from_dict(data.get("style") or {})
    return task


def conversion_task_from_dict(data: dict[str, Any]) -> VoiceConversionTask:
    task = _build(VoiceConversionTask, data)
    task.voice = voice_from_dict(data.get("voice") or {})
    return task


def edit_task_from_dict(data: dict[str, Any]) -> SpeechEditTask:
    task = _build(SpeechEditTask, data)
    task.edits = [_build(EditSpan, e) if isinstance(e, dict) else e for e in task.edits]
    return task
