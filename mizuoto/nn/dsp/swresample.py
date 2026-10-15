"""Resampling as the ffmpeg program does it by default (libswresample, LGPL-2.1, (c) Michael Niedermayer): what
``ffmpeg -i clip -f f32le -ac 1 -ar <rate> -`` gives for a 16-bit clip, the way GPT-SoVITS's ``load_audio`` reads audio.

swr's defaults: a Kaiser-windowed sinc (beta 9) of 32 taps at the lower rate, cutoff 0.97, an exact rational phase
count when the reduced ratio has at most 1024 phases (else 1024 phases, linearly interpolated), in float. The first
``filter_length`` samples are mirrored before the start (``invert_initial_buffer``) and the last ones after the end
(``resample_flush``), so the output is centred and as long as ffmpeg's.

The coefficients are built in double precision and stored as float, as ``build_filter`` does; the dot products follow
the x86 FMA3 kernel (``ff_resample_common_float_fma3``): eight lanes, each a chain of fused multiply-adds over every
eighth tap, then the lanes summed in the kernel's order. A fused multiply-add is computed in double then rounded to
float, which equals the hardware's single rounding except in the rarest double-rounding case. Other CPUs' kernels (SSE,
AVX without FMA) round differently, so ffmpeg itself differs between machines by about one float ulp.
"""

from __future__ import annotations

import math
from functools import lru_cache

import numpy as np

FILTER_SIZE = 32
PHASE_SHIFT = 10
CUTOFF = 0.97
KAISER_BETA = 9.0
LANES = 8
"""The FMA3 kernel's vector width in floats."""

_P1 = (
    -2.2335582639474375249e15,
    -5.5050369673018427753e14,
    -3.2940087627407749166e13,
    -8.4925101247114157499e11,
    -1.1912746104985237192e10,
    -1.0313066708737980747e08,
    -5.9545626019847898221e05,
    -2.4125195876041896775e03,
    -7.0935347449210549190e00,
    -1.5453977791786851041e-02,
    -2.5172644670688975051e-05,
    -3.0517226450451067446e-08,
    -2.6843448573468483278e-11,
    -1.5982226675653184646e-14,
    -5.2487866627945699800e-18,
)
_Q1 = (-2.2335582639474375245e15, 7.8858692566751002988e12, -1.2207067397808979846e10, 1.0377081058062166144e07, -4.8527560179962773045e03, 1.0)
_P2 = (
    -2.2210262233306573296e-04,
    1.3067392038106924055e-02,
    -4.4700805721174453923e-01,
    5.5674518371240761397e00,
    -2.3517945679239481621e01,
    3.1611322818701131207e01,
    -9.6090021968656180000e00,
)
_Q2 = (
    -5.5194330231005480228e-04,
    3.2547697594819615062e-02,
    -1.1151759188741312645e00,
    1.3982595353892851542e01,
    -6.0228002066743340583e01,
    8.5539563258012929600e01,
    -3.1446690275135491500e01,
    1.0,
)


def _poly(coeff: tuple[float, ...], x: float) -> float:
    total = coeff[-1]
    for c in reversed(coeff[:-1]):
        total *= x
        total += c
    return total


def bessel_i0(x: float) -> float:
    """``av_bessel_i0``."""
    if x == 0:
        return 1.0
    x = abs(x)
    if x <= 15:
        y = x * x
        return _poly(_P1, y) / _poly(_Q1, y)
    y = 1 / x - 1.0 / 15
    return math.exp(x) / math.sqrt(x) * (_poly(_P2, y) / _poly(_Q2, y))


def _reduce(num: int, den: int) -> tuple[int, int]:
    g = math.gcd(num, den)
    return num // g, den // g


@lru_cache(maxsize=16)
def _filter_bank(factor: float, tap_count: int, alloc: int, phase_count: int) -> np.ndarray:
    """``build_filter`` for float samples (scale 1), plus ``resample_init``'s extra phase: (phase_count + 1, alloc) float32."""
    bank = np.zeros((phase_count + 1, alloc), dtype=np.float32)
    ph_nb = phase_count if phase_count % 2 else phase_count // 2 + 1
    center = (tap_count - 1) // 2
    factor = min(factor, 1.0)
    sin_lut = [math.sin(math.pi * ph / phase_count) * (1 if center & 1 else -1) for ph in range(ph_nb)] if factor == 1.0 else [0.0] * ph_nb
    norm = 0.0
    for ph in range(ph_nb):
        s = sin_lut[ph]
        tab = []
        for i in range(tap_count):
            x = math.pi * ((float(i - center)) - float(ph) / phase_count) * factor
            if x == 0:
                y = 1.0
            elif factor == 1.0:
                y = s / x
            else:
                y = math.sin(x) / x
            w = 2.0 * x / (factor * tap_count * math.pi)
            y *= bessel_i0(KAISER_BETA * math.sqrt(max(1 - w * w, 0)))
            tab.append(y)
            s = -s
            if not ph:
                norm += y
        for i in range(tap_count):
            bank[ph, i] = np.float32(tab[i] * 1 / norm)
        if phase_count % 2 == 0:
            for i in range(tap_count):
                bank[phase_count - ph, tap_count - 1 - i] = bank[ph, i]
    # resample_init's copies: the extra phase is the first one shifted by a tap (for linear interpolation)
    flat = bank.reshape(-1)
    flat[alloc * phase_count + 1 : alloc * phase_count + alloc] = flat[: alloc - 1]
    flat[alloc * phase_count] = flat[alloc - 1]
    return bank


def _fma_dot(windows: np.ndarray, filters: np.ndarray) -> np.ndarray:
    """(n, alloc) samples · (n, alloc) taps as the FMA3 kernel sums them: (n, 4) lanes after the 256-bit fold."""
    acc = np.zeros((windows.shape[0], LANES), dtype=np.float32)
    for start in range(0, windows.shape[1], LANES):
        x = windows[:, start : start + LANES].astype(np.float64)
        f = filters[:, start : start + LANES].astype(np.float64)
        acc = (x * f + acc.astype(np.float64)).astype(np.float32)
    return acc[:, :4] + acc[:, 4:]


def _fold(v: np.ndarray) -> np.ndarray:
    """The kernel's horizontal sum of four lanes: (v0 + v2) + (v1 + v3)."""
    return (v[:, 0] + v[:, 2]) + (v[:, 1] + v[:, 3])


def resample(samples: np.ndarray, in_rate: int, out_rate: int) -> np.ndarray:
    """Mono float32 samples at ``in_rate`` → at ``out_rate``, as ffmpeg's default resampler on an x86 FMA3 CPU."""
    x = np.asarray(samples, dtype=np.float32).reshape(-1)
    if in_rate == out_rate or x.size == 0:
        return x.copy()
    factor = min(out_rate * CUTOFF / in_rate, 1.0)
    phase_count = 1 << PHASE_SHIFT
    length = max(int(math.ceil(FILTER_SIZE / factor)), 1)
    if length > 1:
        length += length % 2
    exact_num, _ = _reduce(out_rate, in_rate)
    if exact_num <= phase_count:
        phase_count = exact_num
    alloc = (length + 7) // 8 * 8
    bank = _filter_bank(factor, length, alloc, phase_count)
    src_incr, dst_incr = _reduce(out_rate, in_rate * phase_count)
    while dst_incr < (1 << 20) and src_incr < (1 << 20):
        dst_incr *= 2
        src_incr *= 2
    div, mod = divmod(dst_incr, src_incr)
    linear = mod != 0  # resample_linear when the step has a fraction (swr's linear_interp is on)

    # invert_initial_buffer: x[length], …, x[1] before x[0]; the start moved back by (length - 1) // 2 samples
    n = x.size
    head = x[1 : length + 1][::-1] if n > length else np.concatenate([np.zeros(length + 1 - n, np.float32), x[1:][::-1]])
    buf = np.concatenate([head, x])
    start = length - (length - 1) // 2

    def positions(limit: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Every output whose window ends by ``limit`` (in samples of ``buf``): sample index, phase, frac."""
        # output j sits at start·P + j·div + floor(j·mod / src_incr) phases (+ frac j·mod mod src_incr)
        count = int((limit - length - start + 1) * phase_count * src_incr // dst_incr) + 2
        j = np.arange(max(count, 0), dtype=np.int64)
        whole = j * div + (j * mod) // src_incr
        frac = (j * mod) % src_incr
        index = start * phase_count + whole
        sample = index // phase_count
        keep = sample + length <= limit
        return sample[keep], (index % phase_count)[keep], frac[keep]

    # resample_flush: the samples the next output would start from, reflected after the end
    done = positions(buf.size)[0].size
    leftover = buf.size - (start * phase_count + done * div + (done * mod) // src_incr) // phase_count
    reflection = (min(leftover, length) + 1) // 2
    buf = np.concatenate([buf, buf[buf.size - reflection :][::-1], np.zeros(alloc, np.float32)])
    sample, phase, frac = positions(buf.size - alloc)
    windows = buf[sample[:, None] + np.arange(alloc)]
    first = _fma_dot(windows, bank[phase])
    if not linear:
        return _fold(first).astype(np.float32)
    second = _fma_dot(windows, bank[phase + 1])
    weight = (frac.astype(np.float32) * np.float32(np.float32(1.0) / np.float32(src_incr))).astype(np.float32)[:, None]
    mixed = ((second - first).astype(np.float64) * weight.astype(np.float64) + first.astype(np.float64)).astype(np.float32)
    return _fold(mixed).astype(np.float32)
