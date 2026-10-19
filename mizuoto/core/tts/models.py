"""Synthesis requests and plans."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import Field, PrivateAttr, model_validator

from mizuoto.core.engines.models import VoiceMode
from mizuoto.core.record import Record


class CommonParams(Record):
    """Parameters many engines share. ``None``: the engine's own default (Mizuoto imposes none)."""

    temperature: float | None = None
    top_p: float | None = None
    top_k: int | None = None
    repetition_penalty: float | None = None
    max_new_tokens: int | None = None
    """In the engine's own units (tokens or frames)."""
    num_steps: int | None = None
    guidance_scale: float | None = None
    speed: float | None = None
    """1.0 is natural."""
    duration: float | None = None
    """Seconds of generated speech, without the reference."""
    seed: int | None = None
    text_normalization: bool | None = None
    segment_silence_ms: int | None = None
    """The pause between segments."""


class ReferenceInput(Record):
    """A reference clip: an audio file id (an upload) or a path on the server, and its transcript."""

    audio_id: str | None = None
    path: str | None = None
    transcript: str | None = None
    language: str | None = None


class VoiceInput(Record):
    """A voice given with the request instead of a library voice. Set the fields of one mode."""

    references: list[ReferenceInput] = Field(default_factory=list)
    preset: str | None = None
    description: str | None = None
    attributes: dict[str, str] | None = None


class EmotionInput(Record):
    audio_id: str | None = None
    path: str | None = None
    vector: dict[str, float] | None = None
    text: str | None = None
    strength: float | None = Field(default=None, ge=0.0, le=2.0)


class StyleInput(Record):
    instruct: str | None = None
    emotion: EmotionInput | None = None


OutputFormatName = Literal["wav", "flac", "mp3", "ogg", "m4a"]


class OutputSpec(Record):
    format: OutputFormatName | None = None
    """None: ``tts.output_format``."""
    sample_rate: int | None = Field(default=None, ge=8000, le=192000)
    """None: ``tts.output_sample_rate``, else the model's own rate."""
    normalize: Literal["none", "peak", "loudness"] | None = None
    name: str | None = None
    project: str | None = None
    """Set by projects: the output is a take of one of this project's lines (kind ``line``)."""


class _Request(Record):
    """A request body. A field it does not have is ignored, as every record ignores one (a client sending a field of a
    newer version keeps working), and remembered, so that the request says so (``ignored_fields``): a misspelt field,
    ``options`` for ``engine_options`` say, would otherwise silently give the defaults."""

    _ignored: list[str] = PrivateAttr(default_factory=list)

    @model_validator(mode="wrap")
    @classmethod
    def _remember_ignored(cls, data: Any, handler: Any) -> Any:
        record = handler(data)
        if isinstance(data, dict):
            known = set(cls.model_fields) | {f.alias for f in cls.model_fields.values() if f.alias}
            record._ignored = sorted(str(k) for k in data if k not in known)
        return record

    @property
    def ignored_fields(self) -> list[str]:
        """The body's top-level fields this request does not have (and ignored)."""
        return list(self._ignored)


class SpeakRequest(_Request):
    model: str
    """``<engine>/<model>``, e.g. ``cosyvoice/cosyvoice3-0.5b``."""
    text: str = ""
    texts: list[str] | None = None
    """A batch: one output per text (models with the ``batch`` capability)."""
    language: str | None = None
    """A canonical code (``zh``, ``en``…) or ``auto``."""
    voice: str | None = None
    """A library voice's id or slug."""
    voice_spec: VoiceInput | None = None
    style: StyleInput | None = None
    params: CommonParams = Field(default_factory=CommonParams)
    engine_options: dict[str, Any] = Field(default_factory=dict)
    output: OutputSpec = Field(default_factory=OutputSpec)
    segmentation: Literal["engine", "mizuoto"] = "engine"
    """Who splits long text: the engine, when it can, or Mizuoto."""
    strict: bool | None = None
    """Refuse parameters the model does not take. None: ``tts.strict_params``."""
    text_stream: bool = False
    """``text`` is only the beginning: the rest arrives while the model speaks (the stream WebSocket's
    ``text`` messages; models with ``stream_text_input``)."""


class PlannedReference(Record):
    audio_id: str
    name: str
    duration: float
    sample_rate: int
    transcript: str | None = None
    language: str | None = None
    transcribe: bool = False
    """The transcript is made by the speech recognition engine when the job runs."""


class SynthesisPlan(Record):
    """What a request will do, checked before anything runs."""

    model: str
    engine: str
    variant: str
    device: str
    voice_mode: VoiceMode
    voice_id: str | None = None
    language: str | None = None
    segments: list[str]
    segmented_by: Literal["engine", "mizuoto", "batch"]
    params: dict[str, Any]
    options: dict[str, Any]
    references: list[PlannedReference] = Field(default_factory=list)
    output_format: OutputFormatName
    model_sample_rate: int
    output_sample_rate: int
    normalize: Literal["none", "peak", "loudness"]
    stream_granularity: Literal["token", "segment", "emulated"] | None = None
    text_stream: bool = False
    """The text arrives while the model speaks."""
    licence_tags: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class SegmentPreview(Record):
    """Where long text will be cut: by Mizuoto, or not at all when the engine splits it itself."""

    segments: list[str]
    segmented_by: Literal["engine", "mizuoto"]


class SourceInput(Record):
    """A recording to work on: an audio file id (an upload) or a path on the server."""

    audio_id: str | None = None
    path: str | None = None


class ConvertRequest(_Request):
    """Say what a recording says in another voice (models with ``voice_conversion``)."""

    model: str
    source: SourceInput
    voice: str | None = None
    """A library voice (its reference clip is the target voice)."""
    voice_spec: VoiceInput | None = None
    """Or a reference clip given with the request."""
    params: CommonParams = Field(default_factory=CommonParams)
    engine_options: dict[str, Any] = Field(default_factory=dict)
    output: OutputSpec = Field(default_factory=OutputSpec)
    strict: bool | None = None


class EditSpanInput(Record):
    start: float = Field(ge=0)
    """Seconds into the recording where the replaced speech begins."""
    end: float = Field(ge=0)
    text: str
    """What the new speech says (for the record; the model reads ``transcript``)."""
    duration: float | None = Field(default=None, gt=0)
    """Seconds the new speech lasts; None: as long as the span."""


class EditRequest(_Request):
    """Replace spans of a recording with new speech in the same voice (models with ``speech_edit``)."""

    model: str
    source: SourceInput
    transcript: str
    """The whole text of the recording after the edit."""
    edits: list[EditSpanInput] = Field(min_length=1)
    params: CommonParams = Field(default_factory=CommonParams)
    engine_options: dict[str, Any] = Field(default_factory=dict)
    output: OutputSpec = Field(default_factory=OutputSpec)
    strict: bool | None = None


OpenAiFormat = Literal["mp3", "opus", "aac", "flac", "wav", "pcm"]

OPENAI_FORMATS: dict[str, OutputFormatName] = {"mp3": "mp3", "opus": "ogg", "aac": "m4a", "flac": "flac", "wav": "wav", "pcm": "wav"}
"""OpenAI's response formats → Mizuoto's (``pcm`` is made as WAV and sent bare)."""

OPENAI_PCM_RATE = 24000
"""OpenAI's ``pcm`` is 24 kHz mono 16-bit little-endian."""

OPENAI_VOICES = frozenset({"alloy", "ash", "ballad", "cedar", "coral", "echo", "fable", "marin", "nova", "onyx", "sage", "shimmer", "verse"})
"""OpenAI's own voice names: clients send them by default, so they fall back to the model's own voice."""


class OpenAiVoiceRef(Record):
    id: str


class OpenAiSpeechRequest(Record):
    """``POST /v1/audio/speech`` as OpenAI defines it."""

    model: str
    """A Mizuoto model id, an alias from ``tts.model_aliases``, or anything else for ``tts.default_model``."""
    input: str
    voice: str | OpenAiVoiceRef | None = None
    """A library voice (id or slug), a preset speaker of the model, or one of OpenAI's names."""
    response_format: OpenAiFormat = "mp3"
    speed: float | None = Field(default=None, ge=0.25, le=4.0)
    instructions: str | None = None
    """How to say it, for models with ``instruct``; ignored otherwise."""
    stream_format: Literal["audio", "sse"] | None = None
