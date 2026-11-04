"""What a piece of audio looks like, for people judging a result: a mel spectrogram for display and its pitch curve.

Both are numpy only (the core loads no torch), and neither feeds a model:

- ``spectrogram``: a 128-band mel spectrogram (Hann-windowed STFT, centred frames, Slaney's mel scale and area
  normalisation as librosa's default filters), in decibels below the loudest bin, 80 dB of range, scaled to bytes. The
  hop is chosen so the picture has about ``columns`` frames whatever the length. It shows a voice's high-frequency
  cut-off (a 22 kHz model's 11 kHz edge), noise, clicks and breaths.
- ``pitch_curve``: YIN (de Cheveigné and Kawahara, 2002) on 16 kHz audio every 10 ms: the cumulative mean normalised
  difference, its first dip under 0.15 between 50 and 1,000 Hz, refined by a parabola; frames quieter than 45 dB
  under the loudest are unvoiced. It shows octave jumps, monotone stretches and how high a voice sits.
"""

from __future__ import annotations

import numpy as np

SPEC_ROWS = 128
SPEC_FLOOR_DB = -80.0
PITCH_RATE = 16000
PITCH_HOP = 0.01
"""Seconds between pitch frames."""
PITCH_FMIN = 50.0
PITCH_FMAX = 1000.0
YIN_WINDOW = 512
"""Samples (32 ms at 16 kHz) each difference sums over."""
YIN_THRESHOLD = 0.15
SILENCE_DB = -45.0
"""Frames this far under the loudest frame's level are unvoiced."""


def _hz_to_mel(hz: np.ndarray) -> np.ndarray:
    """Slaney's mel scale (librosa's default): linear to 1 kHz, logarithmic above."""
    f_sp, min_log_hz = 200.0 / 3, 1000.0
    min_log_mel, logstep = min_log_hz / f_sp, np.log(6.4) / 27.0
    hz = np.asarray(hz, dtype=np.float64)
    return np.where(hz >= min_log_hz, min_log_mel + np.log(np.maximum(hz, 1e-10) / min_log_hz) / logstep, hz / f_sp)


def _mel_to_hz(mel: np.ndarray) -> np.ndarray:
    f_sp, min_log_hz = 200.0 / 3, 1000.0
    min_log_mel, logstep = min_log_hz / f_sp, np.log(6.4) / 27.0
    mel = np.asarray(mel, dtype=np.float64)
    return np.where(mel >= min_log_mel, min_log_hz * np.exp(logstep * (mel - min_log_mel)), f_sp * mel)


def mel_filters(sample_rate: int, n_fft: int, n_mels: int, fmin: float = 0.0, fmax: float | None = None) -> np.ndarray:
    """``[n_mels, n_fft // 2 + 1]`` triangular filters on Slaney's scale, each normalised by its width (librosa's
    ``filters.mel`` defaults)."""
    fmax = sample_rate / 2 if fmax is None else fmax
    bins = np.linspace(0, sample_rate / 2, n_fft // 2 + 1)
    edges = _mel_to_hz(np.linspace(_hz_to_mel(np.array(fmin)), _hz_to_mel(np.array(fmax)), n_mels + 2))
    widths = np.diff(edges)
    ramps = edges[:, None] - bins[None, :]
    lower = -ramps[:-2] / widths[:-1, None]
    upper = ramps[2:] / widths[1:, None]
    weights = np.maximum(0, np.minimum(lower, upper))
    weights *= (2.0 / (edges[2 : n_mels + 2] - edges[:n_mels]))[:, None]
    return weights.astype(np.float32)


def _frames(samples: np.ndarray, size: int, hop: int, count: int) -> np.ndarray:
    """``count`` frames of ``size`` samples every ``hop``, centred on ``i * hop`` (zero-padded at the ends)."""
    pad = size // 2
    need = (count - 1) * hop + size
    padded = np.zeros(max(need, samples.shape[0] + 2 * pad), dtype=np.float32)
    padded[pad : pad + samples.shape[0]] = samples
    return np.lib.stride_tricks.as_strided(padded, shape=(count, size), strides=(padded.strides[0] * hop, padded.strides[0]), writeable=False)


def spectrogram(samples: np.ndarray, sample_rate: int, columns: int = 800, rows: int = SPEC_ROWS) -> tuple[np.ndarray, float]:
    """A mel spectrogram of mono ``samples`` with about ``columns`` frames: ``uint8 [rows, frames]``, row 0 the lowest
    band, 0 at ``SPEC_FLOOR_DB`` under the loudest bin and 255 the loudest; and the top band's frequency."""
    n = int(samples.shape[0])
    n_fft = 2048 if sample_rate > 24000 else 1024
    hop = max(64, -(-n // max(16, int(columns))))
    count = max(1, n // hop + 1)
    window = (0.5 - 0.5 * np.cos(2 * np.pi * np.arange(n_fft) / n_fft)).astype(np.float32)  # periodic Hann
    filters = mel_filters(sample_rate, n_fft, rows)
    out = np.empty((rows, count), dtype=np.float32)
    frames = _frames(samples.astype(np.float32, copy=False), n_fft, hop, count)  # views: nothing copied yet
    for start in range(0, count, 512):  # a block of frames at a time: bounded memory for long audio
        block = frames[start : start + 512] * window
        power = np.abs(np.fft.rfft(block, axis=1)) ** 2
        out[:, start : start + block.shape[0]] = filters @ power.T
    db = 10.0 * np.log10(np.maximum(out, 1e-10))
    db -= db.max() if db.size else 0.0
    scaled = np.clip((db - SPEC_FLOOR_DB) / -SPEC_FLOOR_DB * 255.0, 0, 255).round().astype(np.uint8)
    return scaled, sample_rate / 2


def pitch_curve(samples_16k: np.ndarray) -> np.ndarray:
    """Hz every 10 ms of 16 kHz mono audio, 0 where there is no pitch (YIN)."""
    n = int(samples_16k.shape[0])
    hop = int(PITCH_RATE * PITCH_HOP)
    if n < hop:
        return np.zeros(0, dtype=np.float32)
    tau_min = int(PITCH_RATE / PITCH_FMAX)
    tau_max = int(PITCH_RATE / PITCH_FMIN)
    size = YIN_WINDOW + tau_max
    count = n // hop + 1
    x = samples_16k.astype(np.float32, copy=False)
    fft = 1 << int(np.ceil(np.log2(size + YIN_WINDOW)))
    f0 = np.zeros(count, dtype=np.float32)
    level = np.zeros(count, dtype=np.float32)
    taus = np.arange(tau_max, dtype=np.float64)
    every = _frames(x, size, hop, count)  # views: nothing copied yet
    for start in range(0, count, 2048):
        frames = every[start : start + 2048].astype(np.float64)
        head = frames[:, :YIN_WINDOW]
        # r(τ) = Σ_j x_j x_{j+τ} over the window, every τ at once
        spectrum = np.fft.rfft(frames, fft, axis=1) * np.conj(np.fft.rfft(head, fft, axis=1))
        r = np.fft.irfft(spectrum, fft, axis=1)[:, :tau_max]
        energy = np.cumsum(np.concatenate([np.zeros((frames.shape[0], 1)), frames**2], axis=1), axis=1)
        e0 = energy[:, YIN_WINDOW][:, None]
        e_tau = energy[:, YIN_WINDOW : YIN_WINDOW + tau_max] - energy[:, :tau_max]
        d = np.maximum(e0 + e_tau - 2 * r, 0.0)
        cmnd = np.ones_like(d)
        running = np.cumsum(d[:, 1:], axis=1)
        cmnd[:, 1:] = d[:, 1:] * taus[1:] / np.maximum(running, 1e-12)
        level[start : start + frames.shape[0]] = 10 * np.log10(np.maximum(e0[:, 0] / YIN_WINDOW, 1e-12))
        for i, row in enumerate(cmnd):
            below = np.flatnonzero(row[tau_min:] < YIN_THRESHOLD)
            if not below.size:
                continue
            tau = tau_min + int(below[0])
            while tau + 1 < tau_max and row[tau + 1] < row[tau]:  # down to the dip's bottom
                tau += 1
            shift = 0.0
            if 0 < tau < tau_max - 1:
                a, b, c = row[tau - 1], row[tau], row[tau + 1]
                bend = a - 2 * b + c
                shift = 0.5 * (a - c) / bend if bend > 0 else 0.0
            f0[start + i] = PITCH_RATE / (tau + shift)
    if level.size:
        f0[level < level.max() + SILENCE_DB] = 0.0
    return f0


def median_pitch(f0: np.ndarray) -> float | None:
    """The median of the voiced frames, on a log scale (the middle of the range a voice sits in), or None."""
    voiced = f0[f0 > 0]
    return float(np.exp2(np.median(np.log2(voiced)))) if voiced.size else None
