"""OmniVoice's audio post-processing (upstream ``omnivoice/utils/audio.py``, Apache-2.0, Xiaomi Corp.), without pydub.

Upstream removes silences with pydub, on 16-bit samples, in milliseconds. ``Segment`` reproduces the
parts it uses exactly: lengths are ``round(1000 * frames / rate)`` ms, positions ``int(ms * rate / 1000)``
frames (so a slice to the end can drop a last partial millisecond), a slice past the end is padded with
up to 2 ms of silence, ``rms`` is ``audioop.rms`` (the floor of the root of the mean square) and
``dBFS`` is ``20 * log10(rms / 32768)``. Arrays are float32, shape (channels, samples), mono only.
"""

from __future__ import annotations

import math

import numpy as np


class Segment:
    """A mono 16-bit pydub ``AudioSegment``: just what ``split_on_silence`` and edge trimming use."""

    def __init__(self, samples: np.ndarray, rate: int) -> None:
        self.samples = samples
        self.rate = rate

    @classmethod
    def from_float(cls, audio: np.ndarray, rate: int) -> Segment:
        """``numpy_to_audiosegment``: ``(audio * 32768).clip(...)`` truncated to int16."""
        if audio.shape[0] != 1:
            raise ValueError("Only mono audio is supported")
        return cls((audio * 32768.0).clip(-32768, 32767).astype(np.int16)[0], rate)

    def to_float(self) -> np.ndarray:
        """``audiosegment_to_numpy``."""
        return (self.samples.astype(np.float32) / 32768.0)[np.newaxis, :]

    def __len__(self) -> int:
        return round(1000 * (float(len(self.samples)) / self.rate))

    def _frames(self, ms: float) -> int:
        if ms < 0:
            ms = len(self) - abs(ms)
        return int(ms * (self.rate / 1000.0))

    def slice(self, start: float | None = None, end: float | None = None) -> Segment:
        length = len(self)
        start = min(start if start is not None else 0, length)
        end = min(end if end is not None else length, length)
        a, b = self._frames(start), self._frames(end)
        data = self.samples[a:b] if a >= 0 else self.samples[0:0]
        missing = (b - a) - len(data)
        if missing > 0 and len(data):
            if missing > self.rate * 2 / 1000.0:
                raise ValueError(f"{missing} frames missing from a slice")
            data = np.concatenate([data, np.zeros(missing, dtype=np.int16)])
        return Segment(data, self.rate)

    def reverse(self) -> Segment:
        return Segment(self.samples[::-1].copy(), self.rate)

    @property
    def rms(self) -> int:
        if not len(self.samples):
            return 0
        x = self.samples.astype(np.int64)
        return int(math.sqrt(float(np.dot(x, x)) / len(x)))

    @property
    def dbfs(self) -> float:
        rms = self.rms
        if not rms:
            return -math.inf
        return 20 * math.log(rms / 32768.0, 10)


def detect_silence(seg: Segment, min_silence_len: int, silence_thresh: float, seek_step: int) -> list[list[int]]:
    seg_len = len(seg)
    if seg_len < min_silence_len:
        return []
    threshold = 10 ** (float(silence_thresh) / 20) * 32768.0
    last_start = seg_len - min_silence_len
    starts = list(range(0, last_start + 1, seek_step))
    if last_start % seek_step:
        starts.append(last_start)
    silent = [i for i in starts if seg.slice(i, i + min_silence_len).rms <= threshold]
    if not silent:
        return []
    ranges = []
    prev = silent.pop(0)
    current = prev
    for i in silent:
        if i != prev + seek_step and i > prev + min_silence_len:
            ranges.append([current, prev + min_silence_len])
            current = i
        prev = i
    ranges.append([current, prev + min_silence_len])
    return ranges


def detect_nonsilent(seg: Segment, min_silence_len: int, silence_thresh: float, seek_step: int) -> list[list[int]]:
    silent = detect_silence(seg, min_silence_len, silence_thresh, seek_step)
    seg_len = len(seg)
    if not silent:
        return [[0, seg_len]]
    if silent[0][0] == 0 and silent[0][1] == seg_len:
        return []
    ranges = []
    prev_end = 0
    end = 0
    for start, end in silent:
        ranges.append([prev_end, start])
        prev_end = end
    if end != seg_len:
        ranges.append([prev_end, seg_len])
    if ranges[0] == [0, 0]:
        ranges.pop(0)
    return ranges


def split_on_silence(seg: Segment, min_silence_len: int, silence_thresh: float, keep_silence: int, seek_step: int) -> list[Segment]:
    ranges = [[start - keep_silence, end + keep_silence] for start, end in detect_nonsilent(seg, min_silence_len, silence_thresh, seek_step)]
    for a, b in zip(ranges, ranges[1:], strict=False):
        if b[0] < a[1]:
            a[1] = (a[1] + b[0]) // 2
            b[0] = a[1]
    return [seg.slice(max(start, 0), min(end, len(seg))) for start, end in ranges]


def leading_silence(seg: Segment, threshold: float = -50.0, chunk: int = 10) -> int:
    trim = 0
    while seg.slice(trim, trim + chunk).dbfs < threshold and trim < len(seg):
        trim += chunk
    return min(trim, len(seg))


def remove_silence_edges(seg: Segment, lead_sil: int = 100, trail_sil: int = 300, threshold: float = -50) -> Segment:
    seg = seg.slice(max(0, leading_silence(seg, threshold) - lead_sil))
    seg = seg.reverse()
    seg = seg.slice(max(0, leading_silence(seg, threshold) - trail_sil))
    return seg.reverse()


def remove_silence(audio: np.ndarray, rate: int, mid_sil: int = 300, lead_sil: int = 100, trail_sil: int = 300) -> np.ndarray:
    """Middle silences longer than ``mid_sil`` ms shortened to it, the edges trimmed to the given margins."""
    seg = Segment.from_float(audio, rate)
    if mid_sil > 0:
        pieces = split_on_silence(seg, min_silence_len=mid_sil, silence_thresh=-50, keep_silence=mid_sil, seek_step=10)
        # pydub joins them onto an empty 11025 Hz segment, which takes the pieces' (higher) rate
        if rate < 11025:
            raise ValueError("Rates below 11025 Hz are not supported")
        seg = Segment(np.concatenate([p.samples for p in pieces]) if pieces else np.zeros(0, dtype=np.int16), rate)
    return remove_silence_edges(seg, lead_sil, trail_sil, -50).to_float()


def fade_and_pad_audio(audio: np.ndarray, pad_duration: float = 0.1, fade_duration: float = 0.1, sample_rate: int = 24000) -> np.ndarray:
    if audio.shape[-1] == 0:
        return audio
    fade_samples = int(fade_duration * sample_rate)
    pad_samples = int(pad_duration * sample_rate)
    processed = audio.copy()
    if fade_samples > 0:
        k = min(fade_samples, processed.shape[-1] // 2)
        if k > 0:
            processed[..., :k] *= np.linspace(0, 1, k, dtype=np.float32)[np.newaxis, :]
            processed[..., -k:] *= np.linspace(1, 0, k, dtype=np.float32)[np.newaxis, :]
    if pad_samples > 0:
        silence = np.zeros((processed.shape[0], pad_samples), dtype=processed.dtype)
        processed = np.concatenate([silence, processed, silence], axis=-1)
    return processed


def cross_fade_chunks(chunks: list[np.ndarray], sample_rate: int, silence_duration: float = 0.3) -> np.ndarray:
    """Chunks joined with a third of ``silence_duration`` of silence, faded out and in over another third each."""
    if len(chunks) == 1:
        return chunks[0]
    fade_n = int(silence_duration * sample_rate) // 3
    merged = chunks[0].copy()
    for chunk in chunks[1:]:
        fout_n = min(fade_n, merged.shape[-1])
        if fout_n > 0:
            merged[..., -fout_n:] *= np.linspace(1, 0, fout_n, dtype=np.float32)[np.newaxis, :]
        faded = chunk.copy()
        fin_n = min(fade_n, faded.shape[-1])
        if fin_n > 0:
            faded[..., :fin_n] *= np.linspace(0, 1, fin_n, dtype=np.float32)[np.newaxis, :]
        merged = np.concatenate([merged, np.zeros((chunks[0].shape[0], fade_n), dtype=np.float32), faded], axis=-1)
    return merged
