"""The dummy engine: tones per character, pitch from the voice (see the package docstring)."""

from __future__ import annotations

import hashlib
import math
import random
import time
import wave
from collections.abc import Iterator
from pathlib import Path
from typing import TYPE_CHECKING, Any

from mizuoto.core import codes
from mizuoto.runtime.engine import AudioChunk, Description, Engine, EngineError, LoadedEngine, LoadSpec, OptionSpec, VoiceCache

if TYPE_CHECKING:
    from mizuoto.runtime.context import RequestContext
    from mizuoto.runtime.engine import SpeechEditTask, SynthesisTask, VoiceConversionTask, VoiceSpec
    from mizuoto.runtime.runtime import Runtime

SAMPLE_RATE = 24000
PRESET_SPEAKERS = ("alto", "bass", "soprano", "tenor")
CHAR_SECONDS = 0.07
PAUSE_SECONDS = {" ": 0.05, ",": 0.15, "，": 0.15, "、": 0.12, ".": 0.25, "。": 0.25, "!": 0.25, "！": 0.25, "?": 0.25, "？": 0.25, "\n": 0.3}


def _hash_unit(text: str) -> float:
    """A stable number in [0, 1) from a string."""
    return int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:8], 16) / 2**32


def voice_pitch(voice: VoiceSpec) -> float:
    """The base pitch (Hz) a voice speaks at: from its preset, description, references or weights."""
    if voice.preset:
        key = f"preset:{voice.preset}"
    elif voice.description or voice.attributes:
        key = f"design:{voice.description}:{sorted((voice.attributes or {}).items())}"
    elif voice.references:
        key = "ref:" + ",".join(r.sha256 or r.audio_path for r in voice.references)
    elif voice.weights:
        key = f"weights:{sorted(voice.weights.items())}"
    else:
        return 180.0
    return 110.0 + 220.0 * _hash_unit(key)


class DummyEngine(Engine):
    id = "dummy"
    DEVICES = ("cpu",)  # numpy only
    VERSION = "1"
    TRAINER = "mizuoto.engines.dummy.train:DummyTrainer"
    OPTION_SCHEMA = {
        "delay_ms": OptionSpec("int", 0, 0, 10000),
        "chunk_ms": OptionSpec("int", 0, 0, 10000),
        "load_delay_ms": OptionSpec("int", 0, 0, 60000),
        "fail": OptionSpec("enum", "none", choices=("none", "error", "oom", "crash", "exception")),
        "fail_at_segment": OptionSpec("int", 0, 0, 1000),
        "brightness": OptionSpec("float", 0.3, 0.0, 1.0),
        "attend": OptionSpec("int", 0, 0, 1000),
    }

    def detect_variant(self, model_dir: Path) -> str | None:
        return "default"

    def load(self, spec: LoadSpec, runtime: Runtime, ctx: RequestContext) -> LoadedEngine:
        delay = int(spec.options.get("load_delay_ms", 0)) / 1000
        for part in ("weights", "vocoder"):
            ctx.check_cancel()
            ctx.progress(f"loading:{part}", None)
            _sleep_cancellable(delay / 2, ctx)
        return LoadedDummy(spec)

    def call(self, op: str, payload: dict[str, Any], ctx: RequestContext) -> dict[str, Any]:
        return _call(op, payload, ctx)


class LoadedDummy(LoadedEngine):
    def describe(self) -> Description:
        return Description(sample_rate=SAMPLE_RATE, speakers=list(PRESET_SPEAKERS), languages=["*"], vram_mb=0.0)

    def prepare_voice(self, voice: VoiceSpec, cache_dir: Path, ctx: RequestContext) -> VoiceCache | None:
        cache_dir.mkdir(parents=True, exist_ok=True)
        path = cache_dir / "pitch.txt"
        path.write_text(f"{voice_pitch(voice):.3f}", encoding="utf-8")
        return VoiceCache(files={"pitch": str(path)}, meta={"pitch": voice_pitch(voice)})

    def synthesize(self, task: SynthesisTask, ctx: RequestContext) -> Iterator[AudioChunk]:
        options = task.options
        fail = options.get("fail", "none")
        params = task.params
        speed = float(params.get("speed") or 1.0)
        temperature = float(params.get("temperature") or 0.0)
        pitch = voice_pitch(task.voice)
        if task.voice.cache and task.voice.cache.get("pitch"):
            pitch = float(Path(task.voice.cache["pitch"]).read_text(encoding="utf-8"))
        if task.voice.weights and task.voice.weights.get("model"):
            pitch = float(Path(task.voice.weights["model"]).read_text(encoding="utf-8"))
            ctx.applied["voice_weights"] = True
        brightness = float(options.get("brightness", 0.3))
        if task.style.emotion and task.style.emotion.vector:
            brightness = min(1.0, brightness + 0.5 * float(task.style.emotion.vector.get("happy", 0.0)))
        ctx.applied.update({"speed": speed, "temperature": temperature, "pitch_hz": round(pitch, 3)})
        if "seed" in params:
            ctx.applied["seed"] = params["seed"]
        if task.style.instruct:
            ctx.applied["instruct"] = task.style.instruct
        if int(options.get("attend", 0)):
            _attend(int(options["attend"]))
        rng = ctx.rng
        if task.text_stream:
            # Each piece of text is spoken as soon as the next one arrives (or the text ends).
            previous: str | None = task.text if task.text.strip() else None
            for text in ctx.text_pieces():
                ctx.check_cancel()
                if previous is not None:
                    yield AudioChunk(samples=render(previous, pitch, speed, temperature, brightness, None, rng), sample_rate=SAMPLE_RATE, segment=task.segment)
                previous = text
            ctx.check_cancel()
            yield AudioChunk(samples=render(previous or "", pitch, speed, temperature, brightness, None, rng), sample_rate=SAMPLE_RATE, segment=task.segment, final=True)
            return
        texts = task.texts if task.texts else [task.text]
        chunk_ms = int(options.get("chunk_ms", 0))
        delay = int(options.get("delay_ms", 0)) / 1000
        for index, text in enumerate(texts):
            segment = task.segment + index if task.texts else task.segment
            if fail != "none" and segment == int(options.get("fail_at_segment", 0)):
                _fail(fail)
            duration = float(params["duration"]) if params.get("duration") else None
            samples = render(text, pitch, speed, temperature, brightness, duration, rng)
            step = max(1, int(SAMPLE_RATE * chunk_ms / 1000)) if chunk_ms else len(samples) or 1
            pieces = [samples[i : i + step] for i in range(0, len(samples), step)] or [samples]
            for n, piece in enumerate(pieces):
                ctx.check_cancel()
                if delay:
                    _sleep_cancellable(delay, ctx)
                ctx.progress("segment", (n + 1) / len(pieces), None)
                yield AudioChunk(samples=piece, sample_rate=SAMPLE_RATE, segment=segment, final=n == len(pieces) - 1)

    def convert(self, task: VoiceConversionTask, ctx: RequestContext) -> Iterator[AudioChunk]:
        seconds = wav_duration(Path(task.source_path)) or 1.0
        text = "a" * max(1, int(seconds / CHAR_SECONDS))
        yield AudioChunk(samples=render(text, voice_pitch(task.voice), 1.0, 0.0, 0.3, seconds, ctx.rng), sample_rate=SAMPLE_RATE, final=True)

    def edit(self, task: SpeechEditTask, ctx: RequestContext) -> Iterator[AudioChunk]:
        new_text = task.transcript
        for span in task.edits:
            new_text += " " + span.text
        yield AudioChunk(samples=render(new_text, 180.0, 1.0, 0.0, 0.3, None, ctx.rng), sample_rate=SAMPLE_RATE, final=True)

    def call(self, op: str, payload: dict[str, Any], ctx: RequestContext) -> dict[str, Any]:
        return _call(op, payload, ctx)


def _call(op: str, payload: dict[str, Any], ctx: RequestContext) -> dict[str, Any]:
    if op == "echo":
        return {"echo": payload}
    if op == "sleep":
        _sleep_cancellable(float(payload.get("seconds", 1.0)), ctx)
        return {}
    if op == "transcribe":
        # Stands in for the speech recognition engine in tests.
        seconds = wav_duration(Path(payload["path"])) or 0.0
        return {"text": f"dummy transcript of {seconds:.1f} seconds", "language": payload.get("language") or "en", "segments": [], "duration": seconds}
    raise EngineError(f"Unknown operation: {op}", {"op": op}, code=codes.CAPABILITY_NOT_SUPPORTED)


def _attend(calls: int) -> None:
    """``calls`` attention computations through ``mizuoto.runtime.accel`` (tests of the acceleration plumbing; the only
    use of torch in this engine)."""
    import torch

    from mizuoto.runtime import accel

    site = accel.SITES.get("dummy") or accel.Site("dummy")
    q = torch.zeros(1, 2, 4, 8)
    for _ in range(calls):
        accel.attention(q, q, q, site=site, causal=True)


def _fail(kind: str) -> None:
    if kind == "error":
        raise EngineError("The dummy engine failed on purpose", {"reason": "requested"})
    if kind == "oom":
        raise RuntimeError("CUDA out of memory. Tried to allocate 20.00 GiB (dummy)")
    if kind == "crash":
        raise EngineError("The dummy engine crashed on purpose", {"reason": "requested"}, code=codes.WORKER_FAILED)
    raise ValueError("an unexpected exception (dummy)")


def _sleep_cancellable(seconds: float, ctx: RequestContext) -> None:
    end = time.monotonic() + seconds
    while time.monotonic() < end:
        ctx.check_cancel()
        time.sleep(min(0.02, max(0.0, end - time.monotonic())))


def render(text: str, pitch: float, speed: float, temperature: float, brightness: float, duration: float | None, rng: random.Random) -> Any:
    """The tones of ``text`` as a float32 numpy array."""
    import numpy as np

    sr = SAMPLE_RATE
    parts: list[Any] = []
    for ch in text:
        if ch in PAUSE_SECONDS or ch.isspace():
            parts.append(np.zeros(int(sr * PAUSE_SECONDS.get(ch, 0.05) / speed), dtype=np.float32))
            continue
        n = max(1, int(sr * CHAR_SECONDS / speed))
        t = np.arange(n, dtype=np.float64) / sr
        jitter = 1.0 + temperature * 0.05 * (rng.random() - 0.5)
        freq = pitch * 2 ** ((ord(ch) % 12) / 24) * jitter
        tone = np.sin(2 * math.pi * freq * t) + brightness * 0.5 * np.sin(4 * math.pi * freq * t)
        envelope = np.minimum(1.0, np.minimum(t, t[-1] - t) / 0.008)
        parts.append((0.3 * tone * envelope).astype(np.float32))
    samples = np.concatenate(parts) if parts else np.zeros(0, dtype=np.float32)
    if duration is not None:
        target = max(1, int(sr * duration))
        if len(samples) == 0:
            samples = np.zeros(target, dtype=np.float32)
        else:
            idx = np.linspace(0, len(samples) - 1, target)
            samples = np.interp(idx, np.arange(len(samples)), samples).astype(np.float32)
    return samples


def wav_duration(path: Path) -> float | None:
    try:
        with wave.open(str(path), "rb") as w:
            return w.getnframes() / float(w.getframerate())
    except (OSError, wave.Error, EOFError):
        return None
