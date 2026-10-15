"""Reading and resampling GPT-SoVITS's reference clips as ``TTS_infer_pack/TTS.py`` does (MIT).

- ``load``: the WAV as ``torchaudio.load`` gives it (float32), channels averaged.
- ``resample``: ``torchaudio.transforms.Resample`` (``mizuoto.nn.dsp.resample``), or, with ``match_librosa``,
  upstream's torch port of resampy's ``kaiser_best`` interpolation (what ``librosa.resample`` used): the
  filter table is resampy's (ISC), stored in float32 as upstream converts it (``data/resampy_kaiser_best.npy``,
  8192 samples per zero crossing), and the arithmetic is upstream's, chunk by chunk.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import torch

KAISER_BEST = Path(__file__).with_name("data") / "resampy_kaiser_best.npy"
KAISER_BEST_PRECISION = 8192


@lru_cache(maxsize=4)
def _filter(device: str) -> tuple[torch.Tensor, torch.Tensor]:
    import numpy as np

    win = torch.from_numpy(np.load(KAISER_BEST).astype(np.float32)).to(device)
    return win, torch.diff(win, append=win[-1:])


def resample_like_librosa(audio: torch.Tensor, sr0: int, sr1: int) -> torch.Tensor:
    """``resample(..., match_librosa=True)``: (channels, samples) at ``sr0`` → at ``sr1``."""
    if int(sr0) == int(sr1):
        return audio.clone()
    ratio = float(sr1) / float(sr0)
    scale = min(1.0, ratio)
    n_orig = audio.shape[-1]
    n_out = int(n_orig * ratio)
    interp_win, interp_delta = _filter(str(audio.device))
    num_table = KAISER_BEST_PRECISION
    if ratio < 1.0:
        interp_win = interp_win * ratio
        interp_delta = interp_delta * ratio
    index_step = int(scale * num_table)
    nwin = interp_win.numel()
    t_all = torch.arange(n_out, device=audio.device, dtype=torch.float32) / ratio
    out = torch.zeros((audio.shape[0], n_out), device=audio.device, dtype=audio.dtype)
    chunk = 4096
    max_taps = (nwin + index_step - 1) // index_step
    for begin in range(0, n_out, chunk):
        t = t_all[begin : begin + chunk]
        n = torch.floor(t).long()
        frac = scale * (t - n.to(t.dtype))
        index_frac = frac * num_table
        offset = torch.floor(index_frac).long()
        eta = index_frac - offset.to(index_frac.dtype)
        ii = torch.arange(max_taps, device=audio.device).view(1, -1)
        li = offset.view(-1, 1) + ii * index_step
        lmax = torch.minimum(n + 1, (nwin - offset) // index_step).view(-1, 1)
        lm = ii < lmax
        li_safe = li.clamp(0, nwin - 1)
        lw = interp_win[li_safe] + eta.view(-1, 1) * interp_delta[li_safe]
        left_idx = (n.view(-1, 1) - ii).clamp(0, n_orig - 1)
        left = (lw * lm).to(audio.dtype)
        lv = audio[:, left_idx]
        frac_r = scale - frac
        index_frac = frac_r * num_table
        offset = torch.floor(index_frac).long()
        eta = index_frac - offset.to(index_frac.dtype)
        ri = offset.view(-1, 1) + ii * index_step
        rmax = torch.minimum(n_orig - n - 1, (nwin - offset) // index_step).view(-1, 1)
        rm = ii < rmax
        ri_safe = ri.clamp(0, nwin - 1)
        rw = interp_win[ri_safe] + eta.view(-1, 1) * interp_delta[ri_safe]
        right_idx = (n.view(-1, 1) + ii + 1).clamp(0, n_orig - 1)
        right = (rw * rm).to(audio.dtype)
        out[:, begin : begin + t.numel()] = (lv * left.unsqueeze(0)).sum(-1) + (audio[:, right_idx] * right.unsqueeze(0)).sum(-1)
    return out


_TRANSFORMS: dict[tuple[int, int, str], torch.nn.Module] = {}


def resample(audio: torch.Tensor, sr0: int, sr1: int, match_librosa: bool = False) -> torch.Tensor:
    if match_librosa:
        return resample_like_librosa(audio, sr0, sr1)
    from mizuoto.nn.dsp.resample import Resample

    key = (int(sr0), int(sr1), str(audio.device))
    transform = _TRANSFORMS.get(key)
    if transform is None:
        transform = _TRANSFORMS[key] = Resample(sr0, sr1).to(audio.device)
    return transform(audio)


def load(path: str | Path) -> tuple[torch.Tensor, int]:
    """(channels=1, samples) float32 as ``torchaudio.load`` reads a PCM WAV (the host's references are mono)."""
    from mizuoto.nn.dsp.wav import read_wav

    samples, rate = read_wav(path)
    return torch.from_numpy(samples).unsqueeze(0), rate
