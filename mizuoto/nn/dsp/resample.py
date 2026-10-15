"""Resampling, exactly as upstream engines do it.

- ``resample``: ``torchaudio.functional.resample`` (sinc with a Hann or Kaiser window), a port of
  torchaudio's BSD-2-Clause code, identical in 2.3.1 and 2.8.0. The kernel is computed in the
  waveform's dtype. ``Resample`` is the transform, whose kernel is computed in float64 and kept in
  float32 (``kernel_dtype=None``), as ``torchaudio.transforms.Resample`` does.
- ``soxr_resample``: what ``librosa.resample(..., res_type="soxr_hq")`` and ``librosa.load(sr=…)`` do,
  through the ``soxr`` package (wheels on every platform).
"""

from __future__ import annotations

import math
from typing import Any

import torch
from torch.nn import functional as F


def sinc_kernel(
    orig_freq: int,
    new_freq: int,
    gcd: int,
    lowpass_filter_width: int = 6,
    rolloff: float = 0.99,
    resampling_method: str = "sinc_interp_hann",
    beta: float | None = None,
    device: torch.device | str = "cpu",
    dtype: torch.dtype | None = None,
) -> tuple[torch.Tensor, int]:
    if resampling_method not in ("sinc_interp_hann", "sinc_interp_kaiser"):
        raise ValueError(f"Invalid resampling method: {resampling_method}")
    orig_freq = int(orig_freq) // gcd
    new_freq = int(new_freq) // gcd
    if lowpass_filter_width <= 0:
        raise ValueError("Low pass filter width should be positive.")
    base_freq = min(orig_freq, new_freq)
    base_freq *= rolloff
    width = math.ceil(lowpass_filter_width * orig_freq / base_freq)
    idx_dtype = dtype if dtype is not None else torch.float64
    idx = torch.arange(-width, width + orig_freq, dtype=idx_dtype, device=device)[None, None] / orig_freq
    t = torch.arange(0, -new_freq, -1, dtype=dtype, device=device)[:, None, None] / new_freq + idx
    t *= base_freq
    t = t.clamp_(-lowpass_filter_width, lowpass_filter_width)
    if resampling_method == "sinc_interp_hann":
        window = torch.cos(t * math.pi / lowpass_filter_width / 2) ** 2
    else:
        beta_tensor = torch.tensor(float(beta if beta is not None else 14.769656459379492))
        window = torch.i0(beta_tensor * torch.sqrt(1 - (t / lowpass_filter_width) ** 2)) / torch.i0(beta_tensor)
    t *= math.pi
    scale = base_freq / orig_freq
    kernels = torch.where(t == 0, torch.tensor(1.0).to(t), t.sin() / t)
    kernels *= window * scale
    if dtype is None:
        kernels = kernels.to(dtype=torch.float32)
    return kernels, width


def apply_sinc_kernel(waveform: torch.Tensor, orig_freq: int, new_freq: int, gcd: int, kernel: torch.Tensor, width: int) -> torch.Tensor:
    if not waveform.is_floating_point():
        raise TypeError(f"Expected floating point type for waveform tensor, but received {waveform.dtype}.")
    orig_freq = int(orig_freq) // gcd
    new_freq = int(new_freq) // gcd
    shape = waveform.size()
    waveform = waveform.view(-1, shape[-1])
    num_wavs, length = waveform.shape
    waveform = F.pad(waveform, (width, width + orig_freq))
    resampled = F.conv1d(waveform[:, None], kernel, stride=orig_freq)
    resampled = resampled.transpose(1, 2).reshape(num_wavs, -1)
    target_length = torch.ceil(torch.as_tensor(new_freq * length / orig_freq)).long()
    resampled = resampled[..., :target_length]
    return resampled.view(shape[:-1] + resampled.shape[-1:])


def resample(
    waveform: torch.Tensor,
    orig_freq: int,
    new_freq: int,
    lowpass_filter_width: int = 6,
    rolloff: float = 0.99,
    resampling_method: str = "sinc_interp_hann",
    beta: float | None = None,
) -> torch.Tensor:
    """``torchaudio.functional.resample``."""
    if orig_freq <= 0 or new_freq <= 0:
        raise ValueError("Original frequency and desired frequency should be positive")
    if orig_freq == new_freq:
        return waveform
    gcd = math.gcd(int(orig_freq), int(new_freq))
    kernel, width = sinc_kernel(orig_freq, new_freq, gcd, lowpass_filter_width, rolloff, resampling_method, beta, waveform.device, waveform.dtype)
    return apply_sinc_kernel(waveform, orig_freq, new_freq, gcd, kernel, width)


class Resample(torch.nn.Module):
    """``torchaudio.transforms.Resample``: the kernel computed once (float64, kept as float32 by default)."""

    def __init__(
        self,
        orig_freq: int = 16000,
        new_freq: int = 16000,
        resampling_method: str = "sinc_interp_hann",
        lowpass_filter_width: int = 6,
        rolloff: float = 0.99,
        beta: float | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.orig_freq, self.new_freq = orig_freq, new_freq
        self.gcd = math.gcd(int(orig_freq), int(new_freq))
        if orig_freq != new_freq:
            kernel, self.width = sinc_kernel(orig_freq, new_freq, self.gcd, lowpass_filter_width, rolloff, resampling_method, beta, dtype=dtype)
            self.register_buffer("kernel", kernel)

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        if self.orig_freq == self.new_freq:
            return waveform
        return apply_sinc_kernel(waveform, self.orig_freq, self.new_freq, self.gcd, self.get_buffer("kernel"), self.width)


def soxr_resample(samples: Any, orig_sr: int, target_sr: int, res_type: str = "soxr_hq") -> Any:
    """``librosa.resample(y, orig_sr=…, target_sr=…, res_type="soxr_hq")``: numpy audio, resampled along its last axis.

    As librosa does it: ``soxr.resample`` on each 1-D slice, the length fixed to ``ceil(n * ratio)`` (trimmed
    or zero-padded), the input's dtype kept.
    """
    import numpy as np
    import soxr

    y = np.asarray(samples)
    if orig_sr == target_sr:
        return y
    ratio = float(target_sr) / orig_sr
    n_samples = int(np.ceil(y.shape[-1] * ratio))
    y_hat = np.apply_along_axis(soxr.resample, axis=-1, arr=y, in_rate=orig_sr, out_rate=target_sr, quality=res_type)
    if y_hat.shape[-1] > n_samples:
        y_hat = y_hat[..., :n_samples]
    elif y_hat.shape[-1] < n_samples:
        y_hat = np.pad(y_hat, [(0, 0)] * (y_hat.ndim - 1) + [(0, n_samples - y_hat.shape[-1])])
    return np.asarray(y_hat, dtype=y.dtype)
