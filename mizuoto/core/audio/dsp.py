"""Joining, levelling and summarising audio with numpy."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    import numpy as np

LOUDNESS_TARGET_DB = -16.0
PEAK_TARGET = 0.95


def assemble(segments: list[np.ndarray], sample_rate: int, silence_ms: int = 0, crossfade_ms: int = 0) -> np.ndarray:
    """Join segments with a pause between them, or a short cross-fade when there is no pause."""
    import numpy as np

    parts = [np.asarray(s, dtype=np.float32).reshape(-1) for s in segments if len(s)]
    if not parts:
        return np.zeros(0, dtype=np.float32)
    gap = np.zeros(int(sample_rate * silence_ms / 1000), dtype=np.float32)
    fade = int(sample_rate * crossfade_ms / 1000)
    out = parts[0]
    for part in parts[1:]:
        if len(gap):
            out = np.concatenate([out, gap, part])
        elif fade and len(out) > fade and len(part) > fade:
            ramp = np.linspace(0.0, 1.0, fade, dtype=np.float32)
            mixed = out[-fade:] * (1 - ramp) + part[:fade] * ramp
            out = np.concatenate([out[:-fade], mixed, part[fade:]])
        else:
            out = np.concatenate([out, part])
    return out


def sanitize(samples: np.ndarray) -> np.ndarray:
    """No NaN or infinity, nothing beyond [-1, 1]."""
    import numpy as np

    out = np.nan_to_num(np.asarray(samples, dtype=np.float32), nan=0.0, posinf=1.0, neginf=-1.0)
    return np.clip(out, -1.0, 1.0)


def normalize(samples: np.ndarray, mode: Literal["none", "peak", "loudness"]) -> np.ndarray:
    """Peak normalisation to -0.45 dBFS, or an RMS level near -16 LUFS (an approximation without
    K-weighting), never pushing peaks above the same ceiling."""
    import numpy as np

    x = sanitize(samples)
    if mode == "none" or not len(x):
        return x
    peak = float(np.max(np.abs(x)))
    if peak < 1e-6:
        return x
    if mode == "peak":
        return (x * (PEAK_TARGET / peak)).astype(np.float32)
    rms = float(np.sqrt(np.mean(x.astype(np.float64) ** 2)))
    gain = 10 ** (LOUDNESS_TARGET_DB / 20) / max(rms, 1e-9)
    gain = min(gain, PEAK_TARGET / peak)
    return (x * gain).astype(np.float32)


def peaks(samples: np.ndarray, points: int) -> list[float]:
    """Interleaved min and max of ``points`` windows, for a waveform drawing."""
    import numpy as np

    x = np.asarray(samples, dtype=np.float32).reshape(-1)
    points = max(1, int(points))
    if not len(x):
        return [0.0, 0.0] * points
    edges = np.linspace(0, len(x), points + 1).astype(int)
    out: list[float] = []
    for a, b in zip(edges[:-1], edges[1:]):
        window = x[a : max(b, a + 1)]
        out += [round(float(window.min()), 4), round(float(window.max()), 4)]
    return out


def trim_to(samples: np.ndarray, sample_rate: int, max_seconds: float, search_seconds: float = 2.0) -> np.ndarray:
    """At most ``max_seconds``, cut at the quietest 20 ms in the last ``search_seconds`` before the limit."""
    import numpy as np

    limit = int(max_seconds * sample_rate)
    if len(samples) <= limit:
        return samples
    frame = max(1, int(0.02 * sample_rate))
    start = max(0, limit - int(search_seconds * sample_rate))
    window = np.asarray(samples[start:limit], dtype=np.float32)
    count = len(window) // frame
    if count == 0:
        return samples[:limit]
    energy = (window[: count * frame].reshape(count, frame) ** 2).mean(axis=1)
    cut = start + int(np.argmin(energy)) * frame + frame // 2
    return samples[:cut]
