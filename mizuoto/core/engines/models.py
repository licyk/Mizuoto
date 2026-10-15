"""What the host knows about an engine without loading it: its manifest."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import Field

from mizuoto.core.record import Record

Capability = Literal[
    # where the voice comes from
    "clone",
    "clone_no_transcript",
    "multi_reference",
    "preset_speaker",
    "voice_design",
    "voice_design_freeform",
    "auto_voice",
    "finetuned_voice",
    # controls
    "instruct",
    "emotion_audio",
    "emotion_vector",
    "emotion_text",
    "inline_tags",
    "pronunciation",
    "speed",
    "duration",
    "dialogue",
    "seed",
    # execution
    "stream_audio",
    "stream_text_input",
    "batch",
    "long_text",
    # other tasks
    "voice_conversion",
    "speech_edit",
    # training
    "train_full",
    "train_lora",
    "train_speaker",
]

CAPABILITIES: tuple[str, ...] = Capability.__args__

CommonParam = Literal[
    "temperature",
    "top_p",
    "top_k",
    "repetition_penalty",
    "max_new_tokens",
    "num_steps",
    "guidance_scale",
    "speed",
    "duration",
    "seed",
    "text_normalization",
    "segment_silence_ms",
]

COMMON_PARAMS: tuple[str, ...] = CommonParam.__args__

DeviceKind = Literal["cpu", "cuda", "rocm", "xpu", "mps"]

VoiceMode = Literal["clone", "clone_no_transcript", "preset", "design", "auto", "trained"]


class LicenceInfo(Record):
    id: str
    name: str
    url: str = ""
    gated: bool = False
    """Must be accepted (``engines.<id>.accepted_licence``) before the engine is used."""
    version: str = ""
    commercial: bool = True
    notice: str = ""
    """A line shown with results, when the licence asks for one."""
    file: str = ""
    """``licences/<file>`` beside the manifests: the full text, shown before it is accepted."""
    trains_others: bool = True
    """Its outputs may be training data for other engines' models (False: the datasets refuse them)."""


class Upstream(Record):
    url: str = ""
    commit: str = ""


class ReferenceLimits(Record):
    """What a reference clip must be for this model."""

    min_seconds: float | None = None
    max_seconds: float | None = None
    min_sample_rate: int | None = None
    max_references: int = 1


class ParamSpec(Record):
    """A common parameter as one engine takes it."""

    target: str = ""
    """The engine's own name for it, for documentation and ``applied_params``."""
    type: Literal["float", "int", "bool"] = "float"
    min: float | None = None
    max: float | None = None
    step: float | None = None
    default: Any = None
    """The engine's own default (shown in the UI); Mizuoto never imposes it."""
    help: str = ""
    approximate: bool = False
    """The engine's parameter only resembles the common one; a result carries a warning."""
    variants: list[str] | None = None
    """Only these variants take it; None: all."""
    streaming: bool = True
    """False: not available while streaming."""


class OptionSpec(Record):
    """An engine-specific option (``engine_options``)."""

    type: Literal["float", "int", "bool", "str", "enum"]
    default: Any = None
    min: float | None = None
    max: float | None = None
    step: float | None = None
    choices: list[str] | None = None
    group: Literal["basic", "advanced", "hidden"] = "advanced"
    help: str = ""
    variants: list[str] | None = None


class VariantInfo(Record):
    """A family of models of one engine that share their capabilities."""

    id: str
    title: str = ""
    sample_rate: int
    capabilities: list[Capability] = Field(default_factory=list)
    languages: list[str] | None = None
    """None: the engine's languages."""
    stream_granularity: Literal["token", "segment"] | None = None
    reference: ReferenceLimits = Field(default_factory=ReferenceLimits)
    instruct_requires: list[VoiceMode] = Field(default_factory=list)
    """Voice modes an instruction needs (CosyVoice v2/v3: a reference clip)."""
    design_attributes: dict[str, list[str]] | None = None
    """The closed vocabulary of a structured design (OmniVoice): attribute → allowed values."""
    inline_tags: list[str] = Field(default_factory=list)
    emotions: list[str] = Field(default_factory=list)
    """The named axes of an emotion vector, in order."""
    speakers: list[str] = Field(default_factory=list)
    """Preset speakers known without loading (refined by the worker's description)."""
    max_batch: int | None = None
    weights_version: str | None = None
    """The version a voice's own fine-tuned weights must have to be used with this variant (GPT-SoVITS: ``v2Pro``)."""


class TrainingStage(Record):
    id: str
    title: str
    help: str = ""
    uses: list[str] | None = None
    """The hyperparameters whose change makes this stage stale (None: all of them)."""


class TrainingMethod(Record):
    """A way an engine fine-tunes: its stages, its settings and what it makes."""

    title: str
    description: str = ""
    variants: list[str] | None = None
    """Base models of these variants only (None: every variant)."""
    export: Literal["voice_weights", "model"]
    """What the run registers: a voice's own weights, or a new model."""
    stages: list[TrainingStage]
    hyperparams: dict[str, OptionSpec] = Field(default_factory=dict)
    min_seconds: float = 30.0
    """Less audio than this is refused."""
    assets: dict[str, list[str]] = Field(default_factory=dict)
    """Shared assets only training needs, by variant (``"*"``: every variant), e.g. pretrained discriminators."""
    devices: list[str] | None = None
    """The kinds of device the trainer runs on (``cuda``…); None: the engine's. A native trainer's ``DEVICES`` win."""
    enabled: bool = True
    note: str = ""
    """Why a method is shipped disabled, or what to expect."""


class WeightsSpec(Record):
    """The files a voice's fine-tuned weights for an engine are made of."""

    roles: dict[str, str]
    """Role → file suffix, e.g. ``gpt = ".ckpt"``, ``sovits = ".pth"``."""
    help: str = ""


class ModelFiles(Record):
    """Where a model's files come from."""

    hf: str | None = None
    """The Hugging Face repository (also served by hf-mirror and custom endpoints)."""
    modelscope: str | None = None
    revision: str = "main"
    """The Hugging Face revision; ``scripts/dev.py model-manifests`` pins it to a commit in the files manifest."""
    modelscope_revision: str = "master"
    subdir: str = ""
    """Only this folder of the repository."""
    into: str = ""
    """Put the files in this folder of the model's folder (an extra repository: a codec, a vocoder)."""
    include: list[str] = Field(default_factory=list)
    """Glob patterns of the files to fetch; empty: every file but ``exclude``."""
    exclude: list[str] = Field(default_factory=lambda: [".gitattributes", "*.md", "*.png", "*.jpg", "*.gif", "*.mp4"])
    manifest: str | None = None
    """``files/<name>.files.json`` beside the engine manifest: the pinned revision, paths, sizes and SHA-256."""
    unpack: bool = False
    """Every file of this part is an archive (zip, tar.gz) unpacked into ``into`` after it is verified."""


class ModelEntry(Record):
    id: str
    """``<engine>/<name>``."""
    variant: str
    title: str
    description: str = ""
    builtin: bool = False
    """Needs no files (the dummy engine)."""
    files: ModelFiles | None = None
    extra: list[ModelFiles] = Field(default_factory=list)
    size_mb: float | None = None
    vram_mb: float | None = None
    languages: list[str] | None = None
    recommended: bool = False
    assets: list[str] = Field(default_factory=list)
    """Shared assets the model needs (``<engine>/<name>``), downloaded with it."""


class AssetEntry(Record):
    """Files several models of an engine share (encoders, text front-end data), downloaded once."""

    id: str
    """``<engine>/<name>``."""
    title: str
    description: str = ""
    files: ModelFiles
    extra: list[ModelFiles] = Field(default_factory=list)


class EngineManifest(Record):
    """What the core knows of an engine without importing it. The implementation is ``mizuoto.engines.<id>`` (or a
    distribution registered under the ``mizuoto.engines`` entry points); its ``OPTION_SCHEMA`` mirrors ``options``."""

    id: str
    title: str
    description: str = ""
    internal: bool = False
    """A helper engine (ASR) that Mizuoto uses itself; not offered for synthesis."""
    licence: LicenceInfo
    upstream: Upstream = Field(default_factory=Upstream)
    devices: list[DeviceKind] = Field(default_factory=lambda: list[DeviceKind](["cpu"]))
    languages: list[str] = Field(default_factory=lambda: ["*"])
    """Canonical codes; ``*``: any."""
    variants: dict[str, VariantInfo]
    params: dict[CommonParam, ParamSpec] = Field(default_factory=dict)
    options: dict[str, OptionSpec] = Field(default_factory=dict)
    load_options: dict[str, OptionSpec] = Field(default_factory=dict)
    models: list[ModelEntry] = Field(default_factory=list)
    assets: list[AssetEntry] = Field(default_factory=list)
    ops: list[str] = Field(default_factory=list)
    """Engine-specific operations (``call``)."""
    weights: WeightsSpec | None = None
    """What a voice's own weights for this engine are (engines with ``finetuned_voice``)."""
    training: dict[str, TrainingMethod] = Field(default_factory=dict)
    """Its training methods; the engine's native trainer (``Engine.TRAINER``) runs their stages."""


# -- what the API and the CLI show ------------------------------------------------------


class EngineStatus(Record):
    id: str
    title: str
    description: str
    licence: LicenceInfo
    licence_accepted: bool
    enabled: bool
    available: bool
    """Its packages import here (or on its remote host): torch, and its own distribution for a licence-gated engine."""
    missing: list[str] = []
    """What it lacks when it is not available: ``torch``, ``mizuoto-fishspeech``…"""
    internal: bool = False
    """A helper engine Mizuoto uses itself (speech recognition)."""
    variants: list[str]
    models: list[str]
    devices: list[str]
    weights: WeightsSpec | None = None
    """The files of a voice's own fine-tuned weights, for engines that take them."""
    training: dict[str, list[str] | None] = {}
    """The engine's training methods (enabled ones), each with the variants it starts from (None: any)."""
    load_options: dict[str, OptionSpec] = {}
    """What ``engines.<id>.load_options`` takes."""
    remote: str | None = None
    """The remote host it runs on (``engines.<id>.device = "<host>/<device>"``); ``available`` and ``missing`` are the agent's."""


class ParamField(Record):
    """One parameter of the UI's parameter panel for one model."""

    name: str
    kind: Literal["common", "option"]
    type: Literal["float", "int", "bool", "str", "enum"]
    default: Any = None
    min: float | None = None
    max: float | None = None
    step: float | None = None
    choices: list[str] | None = None
    group: Literal["basic", "advanced", "hidden"] = "basic"
    help: str = ""
    approximate: bool = False
    streaming: bool = True


class ModelParamSpecs(Record):
    """Everything the UI needs to build a form for one model."""

    model: str
    engine: str
    variant: str
    sample_rate: int
    capabilities: list[Capability]
    voice_modes: list[VoiceMode]
    languages: list[str]
    params: list[ParamField]
    reference: ReferenceLimits
    design_attributes: dict[str, list[str]] | None = None
    inline_tags: list[str] = Field(default_factory=list)
    emotions: list[str] = Field(default_factory=list)
    speakers: list[str] = Field(default_factory=list)
    stream_granularity: Literal["token", "segment", "emulated"] | None = None


class LicenceText(Record):
    """An engine's licence as shown before it is accepted."""

    engine: str
    key: str
    """``id@version``: what accepting records."""
    name: str
    url: str = ""
    text: str = ""
    """The full text (empty when Mizuoto does not ship it; read it at ``url``)."""
    accepted: bool
