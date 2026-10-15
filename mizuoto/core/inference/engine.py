"""What the core holds to use a loaded model: an ``EngineHandle``, and the
``EngineRequest`` each of its calls takes.

``RuntimeService.loaded`` gives the handle: a ``LocalHandle`` (``core/inference/local.py``) for a model in this
process, a ``RemoteEngineHandle`` (``core/remote/client.py``) for one on a remote host's agent. Both answer the
same calls, so ``TtsService`` and ``AsrService`` do not know where the model runs:

- ``prepare_voice(voice, cache_dir, request)``: derived voice state written into ``cache_dir``;
- ``synthesize(task, request)``, ``convert(task, request)``, ``edit(task, request)``: the audio as it comes, one
  ``AudioChunk`` (float32 mono numpy samples) at a time; close the stream (``closing``) to stop early;
- ``call(op, payload, request)``: an engine operation (ASR's ``transcribe``, Fish's ``codec_*``,
  ``mizuoto.snapshot``).

The request carries the caller's cancel event (a job's; cancellation is cooperative), an optional timeout, the
streamed text (``text_input``: what ``TextFeed`` puts) and a progress callback; the handle fills in what the engine
reports: the parameters it applied, its warnings and the timings. Errors are the core's (``mizuoto.core.errors``).

Standard library only at module level: the core imports this freely.
"""

from __future__ import annotations

import queue
import threading
from abc import ABC, abstractmethod
from collections.abc import Callable, Generator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from mizuoto.core.clock import new_id
from mizuoto.core.inference.tasks import AudioChunk, SpeechEditTask, SynthesisTask, VoiceCache, VoiceConversionTask, VoiceSpec

SNAPSHOT_OP = "mizuoto.snapshot"
"""``call`` op: digests of the process-wide state requests must leave alone (the contract suite's no-residue check)."""

ProgressFn = Callable[[str, "float | None", "str | None"], None]
AudioStream = Generator[AudioChunk, None, None]
"""Chunks as the engine makes them; ``close()`` it (``contextlib.closing``) to stop before the end."""


@dataclass
class ModelRef:
    """The model to load: its id, engine, variant and folder on this machine (a remote host uses its own)."""

    id: str
    engine: str
    variant: str
    path: str | None = None


@dataclass
class EngineRequest:
    """One call to a loaded model, as the core makes it, and what the engine reported back."""

    id: str = field(default_factory=lambda: new_id("r"))
    cancel: threading.Event | None = None
    """The caller's cancel (a job's ``cancel_event``): the engine stops at its next check."""
    timeout: float | None = None
    """Seconds the call may take before it is cancelled and fails with ``timeout``."""
    text_input: queue.Queue[tuple[str, bool]] | None = None
    """Text arriving while it runs (``SynthesisTask.text_stream``): ``(piece, last)``, as ``TextFeed`` puts them."""
    progress: ProgressFn | None = None
    """``(stage, value, message)``: ``loading:<part>``, ``segment``, …"""
    applied: dict[str, Any] = field(default_factory=dict)
    """Filled in: the parameter values the engine used."""
    warnings: list[str] = field(default_factory=list)
    """Filled in: the engine's warnings."""
    timings: dict[str, float | None] = field(default_factory=dict)
    """Filled in: ``total`` seconds and, for audio, ``first_chunk``."""
    memory: dict[str, Any] = field(default_factory=dict)
    """Filled in (in this process): ``streamed_mb`` copied to the device for weights left in RAM, and the model's
    ``loaded_mb`` on the device and ``size_mb`` at the end; ``partial``, when components are
    partly loaded, lists each with why (the reason, the free memory, working memory and reserve the manager weighed)."""


class EngineHandle(ABC):
    """A model loaded on one device, here or on a remote host. Not re-entrant: one call at a time (the job
    scheduler's ``engine:<engine>:<device>`` resource sees to it)."""

    remote: str | None = None
    """The remote host it runs on; None in this process."""

    def __init__(self, engine: str, device: str, model_id: str, description: dict[str, Any]) -> None:
        self.engine = engine
        self.device = device
        self.model_id = model_id
        self.description = description
        """What the loaded model offers (``tasks.Description`` as a dict): sample rate, speakers, …"""

    @abstractmethod
    def prepare_voice(self, voice: VoiceSpec, cache_dir: Path, request: EngineRequest | None = None) -> VoiceCache | None:
        """Derive reusable voice state into ``cache_dir``; ``None`` when the engine keeps none."""

    @abstractmethod
    def synthesize(self, task: SynthesisTask, request: EngineRequest | None = None) -> AudioStream: ...

    @abstractmethod
    def convert(self, task: VoiceConversionTask, request: EngineRequest | None = None) -> AudioStream: ...

    @abstractmethod
    def edit(self, task: SpeechEditTask, request: EngineRequest | None = None) -> AudioStream: ...

    @abstractmethod
    def call(self, op: str, payload: dict[str, Any] | None = None, request: EngineRequest | None = None) -> dict[str, Any]: ...


def seed_of(params: dict[str, Any] | None) -> int | None:
    """A request's seed, from its common parameters."""
    seed = (params or {}).get("seed")
    return None if seed is None else int(seed)
