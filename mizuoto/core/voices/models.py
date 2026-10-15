"""Voice records."""

from __future__ import annotations

from typing import Literal

from pydantic import Field

from mizuoto.core.engines.models import VoiceMode
from mizuoto.core.record import Record

VoiceKind = Literal["reference", "designed", "preset", "trained"]


class VoiceReference(Record):
    """A reference clip kept with the voice: ``references/<sha256>.flac``, mono at its own rate."""

    sha256: str
    file: str
    """Relative to the voice's folder."""
    transcript: str | None = None
    transcript_source: Literal["user", "asr", "design"] | None = None
    language: str | None = None
    duration: float
    sample_rate: int
    primary: bool = False


class VoicePreset(Record):
    engine: str
    model: str
    speaker: str


class VoiceDesign(Record):
    """How a designed voice was made; the clip it produced is its primary reference."""

    model: str
    description: str | None = None
    attributes: dict[str, str] | None = None
    sample_text: str
    language: str | None = None
    seed: int | None = None


class VoiceWeights(Record):
    """A voice's own fine-tuned weights for one engine, in ``weights/<engine>/`` of its folder."""

    version: str | None = None
    """The engine's version of them (GPT-SoVITS: ``v2Pro``), which decides the models they fit."""
    files: dict[str, str]
    """Role → path in the voice's folder."""
    names: dict[str, str] = Field(default_factory=dict)
    """Role → the file's original name."""
    added_at: str


class WeightsUpload(Record):
    """Weights uploaded file by file: the roles in so far; the voice once every role is in."""

    engine: str
    staged: list[str]
    missing: list[str]
    voice: Voice | None = None


class Consent(Record):
    source: str
    """Whose voice it is and how it was obtained: "my own voice", "licensed from …"."""
    note: str = ""
    confirmed_at: str


class Voice(Record):
    id: str
    slug: str
    name: str
    kind: VoiceKind
    language: str | None = None
    description: str = ""
    tags: list[str] = Field(default_factory=list)
    consent: Consent | None = None
    references: list[VoiceReference] = Field(default_factory=list)
    preset: VoicePreset | None = None
    design: VoiceDesign | None = None
    weights: dict[str, VoiceWeights] = Field(default_factory=dict)
    """Fine-tuned weights per engine: imported, or made by training (M6)."""
    created_at: str
    updated_at: str


class VoiceSummary(Record):
    id: str
    slug: str
    name: str
    kind: VoiceKind
    language: str | None = None
    tags: list[str] = Field(default_factory=list)
    references: int = 0
    duration: float = 0.0
    """Seconds of reference audio."""
    preset_model: str | None = None
    primary: str | None = None
    """The SHA-256 of the main reference clip, to play it."""
    description: str = ""
    updated_at: str


class VoiceFit(Record):
    """Whether and how a model can use a voice."""

    voice_id: str
    model: str
    usable: bool
    mode: VoiceMode | None = None
    needs: list[str] = Field(default_factory=list)
    """What happens first: ``transcribe`` (a reference lacks its transcript), ``trim`` (a clip is cut to the model's window)."""
    warnings: list[str] = Field(default_factory=list)
    reason: str | None = None
    """Why it is not usable."""
    weights: bool = False
    """The voice's own fine-tuned weights for the model's engine are used."""
