"""Reading FLAC files as soundfile reads them, in numpy and the standard library (no libsndfile, no FFmpeg).

Datasets keep their clips as FLAC (``datasets/<slug>/audio/<id>.flac``), and native trainers read them where the
v1 trainers called ``soundfile.read(path, dtype="float32")``. FLAC is lossless, so the samples are the encoder's
integers: ``read_flac`` returns them over ``2 ** (bits - 1)`` as float32 (16-bit over 32768, as libsndfile), the
channels averaged as ``read_wav`` does.

The decoder covers the format (RFC 9639): CONSTANT, VERBATIM, FIXED and LPC subframes, wasted bits, Rice and
Rice2 residuals with escapes, the stereo decorrelations, variable block sizes. CRCs are not checked.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

BLOCK_SIZES = {1: 192, 2: 576, 3: 1152, 4: 2304, 5: 4608, **{n: 256 << (n - 8) for n in range(8, 16)}}
SAMPLE_RATES = {1: 88200, 2: 176400, 3: 192000, 4: 8000, 5: 16000, 6: 22050, 7: 24000, 8: 32000, 9: 44100, 10: 48000, 11: 96000}
SAMPLE_SIZES = {1: 8, 2: 12, 4: 16, 5: 20, 6: 24, 7: 32}


class _Bits:
    """A big-endian bit reader over bytes, reading through a 64-bit window."""

    def __init__(self, data: bytes, pos: int = 0) -> None:
        self.data = data + b"\0" * 8
        self.pos = pos

    def window(self) -> tuple[int, int]:
        """The next bits as an int and how many there are (57 to 64)."""
        i, off = self.pos >> 3, self.pos & 7
        return int.from_bytes(self.data[i : i + 8], "big") & ((1 << (64 - off)) - 1), 64 - off

    def read(self, n: int) -> int:
        if n == 0:
            return 0
        if n > 56:
            hi = self.read(n - 32)
            return (hi << 32) | self.read(32)
        w, avail = self.window()
        self.pos += n
        return w >> (avail - n)

    def signed(self, n: int) -> int:
        v = self.read(n)
        return v - (1 << n) if n and v >> (n - 1) else v

    def unary(self) -> int:
        """Zeros before the next one bit (which is consumed)."""
        count = 0
        while True:
            w, avail = self.window()
            if w:
                n = avail - w.bit_length()
                self.pos += n + 1
                return count + n
            count += avail
            self.pos += avail

    def align(self) -> None:
        self.pos = (self.pos + 7) & ~7

    def rice(self, k: int, n: int, out: list[int]) -> None:
        """``n`` Rice-coded values with parameter ``k``, appended to ``out``."""
        data, pos = self.data, self.pos
        mask = (1 << k) - 1
        append = out.append
        for _ in range(n):
            i, off = pos >> 3, pos & 7
            avail = 64 - off
            w = int.from_bytes(data[i : i + 8], "big") & ((1 << avail) - 1)
            if w and avail - w.bit_length() + 1 + k <= avail:
                q = avail - w.bit_length()
                u = (q << k) | ((w >> (avail - q - 1 - k)) & mask)
                pos += q + 1 + k
            else:  # a long unary part, or the remainder runs past the window
                self.pos = pos
                q = self.unary()
                u = (q << k) | self.read(k)
                pos = self.pos
            append((u >> 1) ^ -(u & 1))
        self.pos = pos


def _utf8_number(bits: _Bits) -> int:
    first = bits.read(8)
    if first < 0x80:
        return first
    extra = 0
    while first & (0x80 >> (extra + 1)):
        extra += 1
    value = first & ((1 << (6 - extra)) - 1)
    for _ in range(extra):
        value = (value << 6) | (bits.read(8) & 0x3F)
    return value


def _residual(bits: _Bits, block: int, order: int) -> list[int]:
    method = bits.read(2)
    if method > 1:
        raise ValueError("Reserved FLAC residual coding method")
    param_bits, escape = (4, 15) if method == 0 else (5, 31)
    partition_order = bits.read(4)
    partitions = 1 << partition_order
    out: list[int] = []
    for p in range(partitions):
        n = (block >> partition_order) - (order if p == 0 else 0)
        k = bits.read(param_bits)
        if k == escape:
            width = bits.read(5)
            out.extend(bits.signed(width) for _ in range(n))
        else:
            bits.rice(k, n, out)
    return out


def _restore(warmup: list[int], residual: list[int], coefs: list[int], shift: int) -> list[int]:
    """``s[i] = e[i] + (Σ c[j] s[i-1-j]) >> shift``."""
    s = list(warmup)
    order = len(coefs)
    rev = coefs[::-1]  # oldest sample first
    for e in residual:
        acc = 0
        tail = s[-order:]
        for c, x in zip(rev, tail):
            acc += c * x
        s.append(e + (acc >> shift))
    return s


FIXED = {0: [], 1: [1], 2: [2, -1], 3: [3, -3, 1], 4: [4, -6, 4, -1]}


def _subframe(bits: _Bits, block: int, bps: int) -> list[int]:
    if bits.read(1):
        raise ValueError("Bad FLAC subframe padding")
    kind = bits.read(6)
    wasted = bits.unary() + 1 if bits.read(1) else 0
    bps -= wasted
    if kind == 0:
        samples = [bits.signed(bps)] * block
    elif kind == 1:
        samples = [bits.signed(bps) for _ in range(block)]
    elif 8 <= kind <= 12:
        order = kind & 7
        warmup = [bits.signed(bps) for _ in range(order)]
        residual = _residual(bits, block, order)
        samples = _restore(warmup, residual, FIXED[order], 0) if order else residual
    elif kind >= 32:
        order = (kind & 31) + 1
        warmup = [bits.signed(bps) for _ in range(order)]
        precision = bits.read(4) + 1
        if precision == 16:
            raise ValueError("Bad FLAC LPC precision")
        shift = bits.signed(5)
        if shift < 0:
            raise ValueError("Negative FLAC LPC shift")
        coefs = [bits.signed(precision) for _ in range(order)]
        samples = _restore(warmup, _residual(bits, block, order), coefs, shift)
    else:
        raise ValueError(f"Reserved FLAC subframe type {kind}")
    if wasted:
        samples = [x << wasted for x in samples]
    return samples


def decode_flac(data: bytes) -> tuple[np.ndarray, int, int]:
    """(int64 samples of shape (frames, channels), sample rate, bits per sample)."""
    if data[:4] != b"fLaC":
        raise ValueError("Not a FLAC stream")
    pos = 4
    rate = channels = bps = 0
    while True:
        header = data[pos]
        length = int.from_bytes(data[pos + 1 : pos + 4], "big")
        if header & 0x7F == 0:  # STREAMINFO
            info = int.from_bytes(data[pos + 4 : pos + 4 + 18], "big")
            rate = (info >> (144 - 80 - 20)) & 0xFFFFF
            channels = ((info >> (144 - 80 - 23)) & 0x7) + 1
            bps = ((info >> (144 - 80 - 28)) & 0x1F) + 1
        pos += 4 + length
        if header & 0x80:
            break
    bits = _Bits(data, pos * 8)
    end = len(data) * 8
    columns: list[list[int]] = [[] for _ in range(channels)]
    while bits.pos + 16 <= end:
        if bits.read(14) != 0x3FFE:
            raise ValueError("Lost FLAC frame sync")
        bits.read(2)  # reserved, blocking strategy
        size_code, rate_code, assignment, size_bits = bits.read(4), bits.read(4), bits.read(4), bits.read(3)
        bits.read(1)
        _utf8_number(bits)
        block = BLOCK_SIZES.get(size_code, 0)
        if size_code == 6:
            block = bits.read(8) + 1
        elif size_code == 7:
            block = bits.read(16) + 1
        if rate_code == 12:
            bits.read(8)
        elif rate_code in (13, 14):
            bits.read(16)
        bits.read(8)  # CRC-8
        frame_bps = SAMPLE_SIZES.get(size_bits, bps)
        if assignment < 8:
            chans = [_subframe(bits, block, frame_bps) for _ in range(assignment + 1)]
        else:
            a = _subframe(bits, block, frame_bps + (1 if assignment == 9 else 0))
            b = _subframe(bits, block, frame_bps + (0 if assignment == 9 else 1))
            if assignment == 8:  # left, side
                chans = [a, [x - y for x, y in zip(a, b)]]
            elif assignment == 9:  # side, right
                chans = [[x + y for x, y in zip(a, b)], b]
            else:  # mid, side
                left, right = [], []
                for m, s in zip(a, b):
                    m = (m << 1) | (s & 1)
                    left.append((m + s) >> 1)
                    right.append((m - s) >> 1)
                chans = [left, right]
        bits.align()
        bits.read(16)  # CRC-16
        for column, chan in zip(columns, chans):
            column.extend(chan)
    return np.array(columns, dtype=np.int64).T.reshape(-1, channels), int(rate), int(bps)


def read_flac(path: str | Path) -> tuple[np.ndarray, int]:
    """Mono float32 samples and the rate, as ``soundfile.read(path, dtype="float32")`` and a mean over channels."""
    ints, rate, bps = decode_flac(Path(path).read_bytes())
    data = ints.astype(np.float32) / np.float32(2 ** (bps - 1))
    return np.ascontiguousarray(data.mean(axis=1), dtype=np.float32), rate


def read_audio(path: str | Path) -> tuple[np.ndarray, int]:
    """A WAV or FLAC file (by its header) as mono float32 samples and the rate."""
    from mizuoto.nn.dsp.wav import read_wav

    with Path(path).open("rb") as f:
        head = f.read(4)
    return read_flac(path) if head == b"fLaC" else read_wav(path)
