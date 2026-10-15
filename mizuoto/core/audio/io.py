"""Decode, encode and resample with PyAV. Samples are float32 mono numpy arrays in [-1, 1]."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    import numpy as np

from mizuoto.core.errors import ValidationError

OutputFormat = Literal["wav", "flac", "mp3", "ogg", "m4a"]

# container format, codec, fixed rate (None: the audio's own)
FORMATS: dict[str, tuple[str, str, int | None]] = {
    "wav": ("wav", "pcm_s16le", None),
    "flac": ("flac", "flac", None),
    "mp3": ("mp3", "libmp3lame", None),
    "ogg": ("ogg", "libopus", 48000),
    "m4a": ("ipod", "aac", None),
}
MIME = {"wav": "audio/wav", "flac": "audio/flac", "mp3": "audio/mpeg", "ogg": "audio/ogg", "m4a": "audio/mp4"}
MP3_RATES = (8000, 11025, 12000, 16000, 22050, 24000, 32000, 44100, 48000)


@dataclass
class AudioProbe:
    sample_rate: int
    channels: int
    duration: float | None
    codec: str
    container: str


def probe(path: Path) -> AudioProbe:
    """Rate, channels, duration and codec of an audio file; ``ValidationError`` when it is not audio."""
    import av

    try:
        with av.open(str(path)) as c:
            if not c.streams.audio:
                raise ValidationError(f"{path.name} has no audio", {"path": str(path)})
            s = c.streams.audio[0]
            duration = float(s.duration * s.time_base) if s.duration is not None and s.time_base is not None else (c.duration / 1_000_000 if c.duration else None)
            return AudioProbe(sample_rate=int(s.rate or 0), channels=int(s.channels or 1), duration=duration, codec=s.codec_context.name, container=c.format.name)
    except (av.error.FFmpegError, OSError) as e:
        raise ValidationError(f"{path.name} is not a readable audio file: {e}", {"path": str(path)}) from e


def decode(path: Path, sample_rate: int | None = None) -> tuple[np.ndarray, int]:
    """The whole file as float32 mono at ``sample_rate`` (default: the file's own)."""
    import av
    import numpy as np

    try:
        with av.open(str(path)) as c:
            if not c.streams.audio:
                raise ValidationError(f"{path.name} has no audio", {"path": str(path)})
            stream = c.streams.audio[0]
            rate = sample_rate or int(stream.rate or 0) or 24000
            resampler = av.AudioResampler(format="flt", layout="mono", rate=rate)
            parts = [f.to_ndarray().reshape(-1) for frame in c.decode(stream) for f in resampler.resample(frame)]
            parts += [f.to_ndarray().reshape(-1) for f in resampler.resample(None)]
    except (av.error.FFmpegError, OSError) as e:
        raise ValidationError(f"{path.name} is not a readable audio file: {e}", {"path": str(path)}) from e
    samples = np.concatenate(parts).astype(np.float32, copy=False) if parts else np.zeros(0, dtype=np.float32)
    return samples, rate


def _frames(samples: np.ndarray, sample_rate: int, size: int = 1 << 16) -> Any:
    import av
    import numpy as np

    data = np.ascontiguousarray(samples, dtype=np.float32).reshape(-1)
    for start in range(0, len(data), size):
        frame = av.AudioFrame.from_ndarray(data[start : start + size].reshape(1, -1), format="flt", layout="mono")
        frame.sample_rate = sample_rate
        yield frame


def resample(samples: np.ndarray, sample_rate: int, target: int) -> np.ndarray:
    """Resample with libswresample."""
    import av
    import numpy as np

    if sample_rate == target or len(samples) == 0:
        return np.asarray(samples, dtype=np.float32)
    rs = av.AudioResampler(format="flt", layout="mono", rate=target)
    parts = [f.to_ndarray().reshape(-1) for frame in _frames(samples, sample_rate) for f in rs.resample(frame)]
    parts += [f.to_ndarray().reshape(-1) for f in rs.resample(None)]
    return np.concatenate(parts).astype(np.float32, copy=False) if parts else np.zeros(0, dtype=np.float32)


def encode(samples: np.ndarray, sample_rate: int, path: Path, fmt: OutputFormat | str) -> Path:
    """Write ``samples`` to ``path`` as ``fmt``. Opus is always 48 kHz; MP3 snaps to a rate it supports."""
    import av

    if fmt not in FORMATS:
        raise ValidationError(f"Unknown audio format: {fmt}", {"formats": sorted(FORMATS)})
    container_fmt, codec, fixed = FORMATS[fmt]
    rate = fixed or sample_rate
    if fmt == "mp3" and rate not in MP3_RATES:
        rate = min(MP3_RATES, key=lambda r: abs(r - sample_rate))
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".part")
    with av.open(str(tmp), "w", format=container_fmt) as c:
        stream = c.add_stream(codec, rate=rate)
        assert isinstance(stream, av.AudioStream)  # FORMATS names audio codecs only
        stream.layout = "mono"
        target_format = stream.codec_context.format.name if stream.codec_context.format else "s16"
        rs = av.AudioResampler(format=target_format, layout="mono", rate=rate)
        for frame in _frames(samples, sample_rate):
            for out in rs.resample(frame):
                for packet in stream.encode(out):
                    c.mux(packet)
        for out in rs.resample(None):
            for packet in stream.encode(out):
                c.mux(packet)
        for packet in stream.encode(None):
            c.mux(packet)
    tmp.replace(path)
    return path


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()
