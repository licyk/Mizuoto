"""Advancing torch's CPU generator by many draws without making them.

Trainers that reproduce upstream's runs replay the random draws upstream spent before training (a model's random
initialisation that the checkpoint then overwrites): only the generator's state afterwards matters. Fish Speech's
replay is about 18 billion 32-bit draws, 40 s of drawing. ``advance`` computes the same state directly.

torch's CPU generator is MT19937 (``at::mt19937``): 624 words, regenerated all at once (the "twist") every 624 draws;
``left`` counts down to the next twist and ``next`` indexes the word the next draw tempers. A twist is 624 steps of a
linear recurrence over GF(2), so the words after ``k`` twists are ``F^(624k)`` of the words now, where ``F`` is one
step. With ``φ`` the recurrence's characteristic polynomial (degree 19937, found by Berlekamp–Massey on one bit of the
sequence), ``F^n = g(F)`` for ``g = t^n mod φ`` on the states the recurrence reaches, so the new words are the XOR of
the windows of the sequence at the exponents ``g`` has. The squares ``t^(2^i) mod φ`` and each jump length's ``g``
are cached (a model's layers repeat, so do the jumps between their real draws): Fish Speech's 204 jumps take about
2 s instead of 40 s of drawing. The rest of the generator's state (the cached normal sample, the seed) is untouched,
as ``random_()`` leaves it. A state whose layout is not the expected one falls back to drawing.
"""

from __future__ import annotations

import functools
import struct
from typing import Any

N = 624
M = 397
MATRIX_A = 0x9908B0DF
UPPER, LOWER = 0x80000000, 0x7FFFFFFF
STATE_BYTES = 5056
"""``CPUGeneratorImplState``: seed (u64), left (i32), seeded (i32), next (u64), 624 words (u64 each), the legacy normal
cache (3 doubles, an int), then the float normal cache (float, bool), padded."""
_WORDS = 24


def _step(window: list[int], i: int) -> int:
    """The word ``i + 624`` of the sequence from words ``i``, ``i + 1`` and ``i + 397`` (``at::mt19937::next_state``)."""
    y = (window[i] & UPPER) | (window[i + 1] & LOWER)
    return window[i + M] ^ (y >> 1) ^ (MATRIX_A if y & 1 else 0)


def _sequence(words: list[int], length: int) -> list[int]:
    seq = list(words)
    for i in range(length - N):
        seq.append(_step(seq, i))
    return seq


@functools.cache
def _phi() -> int:
    """MT19937's characteristic polynomial (bit i: the coefficient of t^i), by Berlekamp–Massey on the top bit of the
    words of a sequence from an arbitrary state."""
    state = [5489]
    for i in range(1, N):
        state.append((1812433253 * (state[-1] ^ (state[-1] >> 30)) + i) & 0xFFFFFFFF)
    bits = [w >> 31 for w in _sequence(state, 2 * 19937 + N + 64)[N:]]
    # Berlekamp–Massey over GF(2), polynomials as ints (bit i: coefficient of x^i); ``window`` has bit i = bits[n - i],
    # so the discrepancy is the parity of the connection polynomial AND it
    c, b = 1, 1
    length, m, window = 0, 1, 0
    for n_, bit in enumerate(bits):
        window = (window << 1) | bit
        if (c & window).bit_count() & 1:
            t = c
            c ^= b << m
            if 2 * length <= n_:
                length, b, m = n_ + 1 - length, t, 1
            else:
                m += 1
        else:
            m += 1
    # c is the connection polynomial (1 + c1 x + … + cL x^L); the characteristic polynomial is its reverse
    return int(f"{c:0{length + 1}b}"[::-1], 2)


def _mulmod(a: int, b: int, phi: int, degree: int) -> int:
    r = 0
    i = 0
    while a:
        if a & 1:
            r ^= b << i
        a >>= 1
        i += 1
    for k in range(r.bit_length() - 1, degree - 1, -1):
        if (r >> k) & 1:
            r ^= phi << (k - degree)
    return r


@functools.cache
def _square(i: int) -> int:
    """``t^(2^i) mod φ``."""
    phi = _phi()
    degree = phi.bit_length() - 1
    return 2 if i == 0 else _mulmod(_square(i - 1), _square(i - 1), phi, degree)


@functools.lru_cache(maxsize=256)
def _powmod(n: int) -> int:
    """``t^n mod φ``, from the cached squares. Cached too: a model's layers repeat, and so do its jumps' lengths."""
    phi = _phi()
    degree = phi.bit_length() - 1
    result, i = 1, 0
    while n:
        if n & 1:
            result = _mulmod(result, _square(i), phi, degree) if result != 1 else _square(i)
        n >>= 1
        i += 1
    return result


def _twist(old: Any) -> Any:
    """The next 624 words of the sequence from the last 624 (``at::mt19937::next_state``, in numpy)."""
    import numpy as np

    def mix(a: Any, b: Any) -> Any:
        y = (a & np.uint32(UPPER)) | (b & np.uint32(LOWER))
        return (y >> np.uint32(1)) ^ np.where(y & np.uint32(1), np.uint32(MATRIX_A), np.uint32(0))

    new = np.empty(N, dtype=np.uint32)
    new[: N - M] = old[M:] ^ mix(old[: N - M], old[1 : N - M + 1])
    for start in range(N - M, N - 1, N - M):  # each word reads the one 227 before it: in chunks of 227
        end = min(start + N - M, N - 1)
        new[start:end] = new[start - (N - M) : end - (N - M)] ^ mix(old[start:end], old[start + 1 : end + 1])
    new[N - 1] = new[M - 1] ^ mix(old[N - 1 : N], new[0:1])[0]
    return new


def jump(words: list[int], steps: int) -> list[int]:
    """The 624 words ``steps`` recurrence steps after ``words`` (``steps`` ≥ 1)."""
    import numpy as np
    from numpy.lib.stride_tricks import sliding_window_view

    degree = _phi().bit_length() - 1
    # one step first: the oldest word's 31 low bits never reach later words, so φ annihilates F only past it
    g = _powmod(steps - 1)
    blocks = [np.array(words, dtype=np.uint32)]
    while len(blocks) * N < 1 + degree + N:
        blocks.append(_twist(blocks[-1]))
    seq = np.concatenate(blocks)[1:]
    positions = [i for i, bit in enumerate(reversed(bin(g)[2:])) if bit == "1"]
    out = np.bitwise_xor.reduce(sliding_window_view(seq, N)[positions], axis=0)
    return [int(w) for w in out]


def advance(generator: Any, draws: int) -> None:
    """Advance a CPU ``torch.Generator`` by ``draws`` 32-bit draws, as ``torch.empty(draws, dtype=torch.int32).random_()``
    would, without making them."""
    import torch

    if draws <= 0:
        return
    state = bytearray(generator.get_state().numpy().tobytes())
    if len(state) != STATE_BYTES:  # another layout: draw for real
        _draw(generator, draws)
        return
    left, _seeded = struct.unpack_from("<ii", state, 8)
    (nxt,) = struct.unpack_from("<Q", state, 16)
    if draws <= left - 1:
        left, nxt = left - draws, nxt + draws
    else:
        rest = draws - (left - 1)  # the first of these twists
        twists = -(-rest // N)
        used = rest - N * (twists - 1)  # draws made from the last block: 1 … 624
        words = list(struct.unpack_from(f"<{N}Q", state, _WORDS))
        words = jump([w & 0xFFFFFFFF for w in words], N * twists)
        struct.pack_into(f"<{N}Q", state, _WORDS, *words)
        left, nxt = N + 1 - used, used
    struct.pack_into("<ii", state, 8, left, _seeded)
    struct.pack_into("<Q", state, 16, nxt)
    generator.set_state(torch.frombuffer(state, dtype=torch.uint8).clone())


def _draw(generator: Any, draws: int) -> None:
    import torch

    chunk = 1 << 26
    buffer = torch.empty(min(draws, chunk), dtype=torch.int32)
    while draws > 0:
        n = min(draws, chunk)
        buffer[:n].random_(generator=generator)
        draws -= n
