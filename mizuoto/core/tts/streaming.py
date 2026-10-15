"""Synthesis as it happens, as mono 16-bit PCM: for the stream WebSocket, ``?stream=true`` and ``mizuoto speak --stream``.

``pcm_sink`` turns a ``StreamSink`` into events: ``start`` (the rate), ``segment`` (a new segment
begins), ``audio`` (mono 16-bit little-endian PCM at that rate), then ``end`` (the output ids) or
``error``. The sink is called on the job's thread; ``emit`` must be safe there.
"""

from __future__ import annotations

import struct
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from mizuoto.core.audio.io import resample
from mizuoto.core.errors import MizuotoError
from mizuoto.core.tts.service import StreamSink, TextFeed

if TYPE_CHECKING:
    import numpy as np

Emit = Callable[[str, Any], None]


def pcm16(samples: np.ndarray) -> bytes:
    import numpy as np

    return (np.clip(samples, -1.0, 1.0) * 32767.0).astype("<i2").tobytes()


def wav_header(sample_rate: int) -> bytes:
    """A WAV header for mono 16-bit PCM of unknown length (both sizes at their maximum, as streaming servers write)."""
    fmt = struct.pack("<IHHIIHH", 16, 1, 1, sample_rate, sample_rate * 2, 2, 16)
    return b"RIFF" + struct.pack("<I", 0xFFFFFFFF) + b"WAVE" + b"fmt " + fmt + b"data" + struct.pack("<I", 0xFFFFFFFF)


def error_body(error: BaseException) -> dict[str, Any]:
    if isinstance(error, MizuotoError):
        return {"code": error.code, "message": error.message, "detail": error.detail}
    return {"code": "internal_error", "message": str(error) or type(error).__name__, "detail": {}}


def pcm_sink(emit: Emit, sample_rate: int | None = None, text: TextFeed | None = None) -> StreamSink:
    """A sink whose audio reaches ``emit`` as PCM16 at ``sample_rate`` (None: the model's rate)."""
    state: dict[str, Any] = {"rate": sample_rate, "segment": -1}

    def start(rate: int, granularity: str | None) -> None:
        state["rate"] = state["rate"] or rate
        emit("start", {"sample_rate": state["rate"], "granularity": granularity})

    def chunk(samples: np.ndarray, rate: int, segment: int) -> None:
        if segment != state["segment"]:
            state["segment"] = segment
            emit("segment", segment)
        target = state["rate"] or rate
        if rate != target:
            samples = resample(samples, rate, target)
        emit("audio", pcm16(samples))

    return StreamSink(start=start, chunk=chunk, end=lambda ids: emit("end", ids), error=lambda e: emit("error", error_body(e)), text=text)
