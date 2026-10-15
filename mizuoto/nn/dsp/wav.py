"""Reading the WAV files the host hands engines (references, sources), as soundfile reads them; writing 16-bit WAV.

The host gives engines ``cache/refs/<sha>.wav``: PCM, mono, at the clip's own rate. ``read_wav`` returns
float32 samples exactly as ``soundfile.read(path, dtype="float32")``: 16-bit PCM over 32768, 24-bit over
2**23, 32-bit over 2**31. Channels are averaged as the v1 adapters did (``data.mean(axis=1)``).
"""

from __future__ import annotations

import wave
from pathlib import Path

import numpy as np


def read_wav(path: str | Path) -> tuple[np.ndarray, int]:
    path = Path(path)
    with path.open("rb") as f:
        head = f.read(12)
    if head[:4] != b"RIFF" or head[8:12] != b"WAVE":
        raise ValueError(f"{path.name} is not a WAV file")
    with wave.open(str(path), "rb") as w:
        channels, width, rate, frames = w.getnchannels(), w.getsampwidth(), w.getframerate(), w.getnframes()
        raw = w.readframes(frames)
    if width == 2:
        data = np.frombuffer(raw, dtype="<i2").astype(np.float32) / np.float32(32768.0)
    elif width == 3:
        b = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 3)
        ints = (b[:, 0].astype(np.int32) | (b[:, 1].astype(np.int32) << 8) | (b[:, 2].astype(np.int32) << 16)) << 8 >> 8
        data = ints.astype(np.float32) / np.float32(2**23)
    elif width == 4:
        data = (np.frombuffer(raw, dtype="<i4").astype(np.float64) / 2**31).astype(np.float32)
    elif width == 1:
        data = (np.frombuffer(raw, dtype=np.uint8).astype(np.float32) - 128.0) / np.float32(128.0)
    else:
        raise ValueError(f"{path.name}: {width * 8}-bit samples are not supported")
    data = data.reshape(-1, channels)
    return np.ascontiguousarray(data.mean(axis=1), dtype=np.float32), int(rate)


def write_wav(path: str | Path, samples: np.ndarray, rate: int) -> None:
    """Mono float samples as 16-bit PCM, as ``soundfile.write(path, samples, rate)`` stores them.

    What libsndfile 1.2 does, found by comparison: ``lrint(x × 2³¹)``, clipped to 32 bits, shifted right by 16. That
    is ``floor(x × 32768)`` except just below a multiple of 2⁻¹⁵ (tiny negative values become 0, not -1).
    """
    x = np.asarray(samples, dtype=np.float32).reshape(-1)
    wide = np.clip(np.rint(x.astype(np.float64) * 2.0**31), -(2.0**31), 2.0**31 - 1).astype(np.int64)
    data = (wide >> 16).astype("<i2")
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(int(rate))
        w.writeframes(data.tobytes())
