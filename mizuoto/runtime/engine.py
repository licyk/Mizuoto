"""The native engine contract and the engine registry.

An ``Engine`` is one engine implementation; ``load`` gives a ``LoadedEngine`` per model. The
LoadedEngine owns its ``ModelHandle``s, which the model manager places on devices.

Requests and results are the task types of ``mizuoto.core.inference.tasks`` (``SynthesisTask``,
``VoiceSpec``, ``AudioChunk``…), standard-library dataclasses the core builds; this module re-exports them.
Errors an engine raises are ``EngineError`` (with a code of ``mizuoto.core.codes``) and ``Unsupported``.

Engines are found by id:
- the built-in ones are listed in ``BUILTIN``;
- other distributions register an ``Engine`` subclass under the ``mizuoto.engines`` entry-point group
  (the licence-gated engines, distributed apart because their licences cover the ports).
"""

from __future__ import annotations

import functools
import importlib
import threading
from abc import ABC, abstractmethod
from collections.abc import Iterator
from importlib import metadata
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar

from mizuoto.core import codes
from mizuoto.core.inference.tasks import (
    AudioChunk,
    Description,
    EditSpan,
    EmotionSpec,
    LoadSpec,
    OptionSpec,
    Reference,
    SpeechEditTask,
    StyleSpec,
    SynthesisTask,
    VoiceCache,
    VoiceConversionTask,
    VoiceSpec,
)

if TYPE_CHECKING:
    from mizuoto.runtime.context import RequestContext
    from mizuoto.runtime.handle import ModelHandle
    from mizuoto.runtime.runtime import Runtime

__all__ = [
    "AudioChunk",
    "Description",
    "EditSpan",
    "EmotionSpec",
    "Engine",
    "EngineError",
    "LoadSpec",
    "LoadedEngine",
    "OptionSpec",
    "Reference",
    "SpeechEditTask",
    "StyleSpec",
    "SynthesisTask",
    "Unsupported",
    "VoiceCache",
    "VoiceConversionTask",
    "VoiceSpec",
    "DEVICE_KINDS",
    "engine_ids",
    "get_engine",
]

ENTRY_POINT_GROUP = "mizuoto.engines"
DEVICE_KINDS = ("cuda", "rocm", "xpu", "mps", "cpu")
"""Every kind of device the runtime drives: NVIDIA (``cuda``), AMD under ROCm (``rocm``: torch's ``cuda:N``), Intel
(``xpu``), Apple (``mps``) and the CPU."""


class EngineError(Exception):
    """An error with a domain code (``mizuoto.core.codes``), which the core turns into its error class as it is."""

    code = codes.ENGINE_ERROR

    def __init__(self, message: str, detail: dict[str, Any] | None = None, code: str | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.detail = detail or {}
        if code is not None:
            self.code = code


class Unsupported(EngineError):
    """The engine does not implement this operation."""

    code = codes.CAPABILITY_NOT_SUPPORTED


BUILTIN: dict[str, str] = {
    "dummy": "mizuoto.engines.dummy:DummyEngine",
    "qwen3tts": "mizuoto.engines.qwen3tts:Qwen3TtsEngine",
    "f5tts": "mizuoto.engines.f5tts:F5TtsEngine",
    "omnivoice": "mizuoto.engines.omnivoice:OmniVoiceEngine",
    "cosyvoice": "mizuoto.engines.cosyvoice:CosyVoiceEngine",
    "gptsovits": "mizuoto.engines.gptsovits:GptSovitsEngine",
    "asr": "mizuoto.engines.asr:AsrEngine",
    "voxcpm": "mizuoto.engines.voxcpm:VoxCpmEngine",
}
"""Engine id → ``module:Class`` of the engines in this distribution."""


class LoadedEngine(ABC):
    """One loaded model. Not re-entrant: the job scheduler gives it one request at a time."""

    handles: list[ModelHandle]
    """Every component with weights, for the model manager (empty for an engine without any)."""
    PHASED: ClassVar[bool] = False
    """The engine places its components itself, phase by phase (``ctx.use``/``ctx.run``).
    Otherwise every component is placed for the whole request, with ``memory_required(task)``."""

    def __init__(self, spec: LoadSpec) -> None:
        self.spec = spec
        self.handles = []

    @property
    def model_id(self) -> str:
        return self.spec.model_id

    @abstractmethod
    def describe(self) -> Description: ...

    def memory_required(self, task: Any) -> int:
        """Bytes of device memory a request needs beyond the weights (activations, caches)."""
        return 0

    def prepare_voice(self, voice: VoiceSpec, cache_dir: Path, ctx: RequestContext) -> VoiceCache | None:
        """Derive reusable voice state (prompt codes, embeddings) into ``cache_dir``; ``None`` when the engine has none."""
        return None

    @abstractmethod
    def synthesize(self, task: SynthesisTask, ctx: RequestContext) -> Iterator[AudioChunk]: ...

    def convert(self, task: VoiceConversionTask, ctx: RequestContext) -> Iterator[AudioChunk]:
        raise Unsupported("This engine does not convert voices")

    def edit(self, task: SpeechEditTask, ctx: RequestContext) -> Iterator[AudioChunk]:
        raise Unsupported("This engine does not edit speech")

    def call(self, op: str, payload: dict[str, Any], ctx: RequestContext) -> dict[str, Any]:
        raise Unsupported(f"Unknown operation: {op}", {"op": op})

    def close(self) -> None:
        """Drop the weights. The manager has already taken the handles off their devices."""
        self.handles = []


class Engine(ABC):
    """One engine implementation."""

    id: ClassVar[str]
    VERSION: ClassVar[str] = "1"
    OPTION_SCHEMA: ClassVar[dict[str, OptionSpec]] = {}
    """The engine's options, mirrored by its manifest (a test keeps them equal)."""
    DEVICES: ClassVar[tuple[str, ...]] = DEVICE_KINDS
    """The kinds of device the implementation runs on, mirrored by its manifest's ``devices`` (a test keeps them
    equal). A module that needs float64 runs on the CPU on a device without it (``DeviceService.float64_device``)."""
    TRAINER: ClassVar[str | None] = None
    """``module:Class`` of the engine's native trainer (``mizuoto.training.Trainer``), when it has one."""
    EXTRAS: ClassVar[dict[str, tuple[str, ...]]] = {}
    """Optional features: the install spec of an extra (``mizuoto[gptsovits-ja]``) → the modules it provides
    (``mizuoto doctor`` lists the ones that do not import)."""

    def detect_variant(self, model_dir: Path) -> str | None:
        """The variant of the model in ``model_dir``, from its files only (no torch), or ``None``."""
        return None

    @abstractmethod
    def load(self, spec: LoadSpec, runtime: Runtime, ctx: RequestContext) -> LoadedEngine: ...

    def call(self, op: str, payload: dict[str, Any], ctx: RequestContext) -> dict[str, Any]:
        """An operation that needs no loaded model."""
        raise Unsupported(f"Unknown operation: {op}", {"op": op})


_instances: dict[str, Engine] = {}
_lock = threading.Lock()


@functools.cache
def _entry_points() -> dict[str, metadata.EntryPoint]:
    """The engines other distributions register (read once: installing one means restarting Mizuoto)."""
    return {ep.name: ep for ep in metadata.entry_points(group=ENTRY_POINT_GROUP)}


def engine_ids() -> list[str]:
    """Every engine with a native implementation: built in, or registered by another distribution."""
    return sorted({*BUILTIN, *_entry_points()})


def get_engine(engine_id: str) -> Engine:
    """The engine's implementation (one instance per process)."""
    with _lock:
        engine = _instances.get(engine_id)
        if engine is not None:
            return engine
        if engine_id in BUILTIN:
            module, _, name = BUILTIN[engine_id].partition(":")
            cls = getattr(importlib.import_module(module), name)
        else:
            ep = _entry_points().get(engine_id)
            if ep is None:
                raise EngineError(f"{engine_id} has no native implementation yet", {"engine": engine_id}, code=codes.CAPABILITY_NOT_SUPPORTED)
            cls = ep.load()
        engine = cls()
        _instances[engine_id] = engine
        return engine
