"""Decoding a long sequence in overlapping tiles.

A convolutional decoder (a vocoder, a VAE) needs memory in proportion to its input's length. When a whole utterance does
not fit, ``tiled_1d`` decodes it in tiles of ``tile`` frames that overlap by ``overlap`` frames and joins them with a
linear cross-fade over the overlaps, as ComfyUI's ``tiled_scale_multidim`` does for audio VAEs (tile 256, overlap 32).
Each tile sees less context at its edges than the whole would, so the result differs slightly from decoding at once:
only used when the engine and ``compute.oom_fallback`` allow it.

A decoder whose output sample depends on the inputs only within a receptive field needs no fade: ``context_tiled_1d``
decodes each tile with context frames around it (only before it for a causal decoder) and keeps the tile's own samples.
With the context at least the receptive field on each side, and the decoder giving exactly ``upsample`` samples a frame
(the sequence's ends padded as the whole's are), this is the same function as decoding at once; only the sums' order
(other lengths, other kernels) may differ in the last bits.
"""

from __future__ import annotations

from collections.abc import Callable

import torch


def tiled_1d(decode: Callable[[torch.Tensor], torch.Tensor], x: torch.Tensor, upsample: int, tile: int = 256, overlap: int = 32) -> torch.Tensor:
    """``decode(x)`` for ``x`` of shape (..., frames) giving (..., frames × ``upsample``), in tiles along the last axis."""
    frames = x.shape[-1]
    if frames <= tile:
        return decode(x)
    stride = tile - overlap
    out: torch.Tensor | None = None
    weight: torch.Tensor | None = None
    start = 0
    while True:
        end = min(start + tile, frames)
        begin = max(0, end - tile)
        y = decode(x[..., begin:end])
        if out is None:  # a decoder may give a fixed number of samples more or fewer than frames × upsample
            total = frames * upsample + y.shape[-1] - (end - begin) * upsample
            out = torch.zeros((*y.shape[:-1], total), dtype=y.dtype, device=y.device)
            weight = torch.zeros(total, dtype=y.dtype, device=y.device)
        assert weight is not None
        length = min(y.shape[-1], out.shape[-1] - begin * upsample)
        mask = torch.ones(length, dtype=y.dtype, device=y.device)
        fade = min(overlap * upsample, length)
        if begin > 0:
            mask[:fade] = torch.linspace(0, 1, fade, dtype=y.dtype, device=y.device)
        if end < frames:
            mask[length - fade :] = torch.minimum(mask[length - fade :], torch.linspace(1, 0, fade, dtype=y.dtype, device=y.device))
        out[..., begin * upsample : begin * upsample + length] += y[..., :length] * mask
        weight[begin * upsample : begin * upsample + length] += mask
        if end >= frames:
            break
        start += stride
    assert out is not None and weight is not None
    return out / weight.clamp(min=1e-6)


def context_tiled_1d(decode: Callable[[torch.Tensor], torch.Tensor], x: torch.Tensor, upsample: int, tile: int, before: int, after: int = 0) -> torch.Tensor:
    """``decode(x)`` for ``x`` of shape (..., frames) giving exactly (..., frames × ``upsample``), a tile of ``tile`` frames
    at a time, each decoded with up to ``before`` frames before it and ``after`` frames after it whose samples are
    dropped (``after=0`` for a causal decoder)."""
    frames = x.shape[-1]
    if frames <= tile:
        return decode(x)
    pieces = []
    for start in range(0, frames, tile):
        end = min(start + tile, frames)
        begin, stop = max(0, start - before), min(frames, end + after)
        y = decode(x[..., begin:stop])
        pieces.append(y[..., (start - begin) * upsample : (end - begin) * upsample])
    return torch.cat(pieces, dim=-1)
