"""Mel filterbanks and spectrograms, exactly as upstream engines compute them.

- ``librosa_mel``: ``librosa.filters.mel`` (identical maths in librosa 0.10, 0.11 and 1.0), computed in
  float64 with numpy. A port of librosa's ISC-licensed code.
- ``melscale_fbanks``, ``spectrogram``, ``mel_spectrogram``: torchaudio's functions and its
  ``MelSpectrogram`` transform. A port of torchaudio's BSD-2-Clause code, identical in 2.3.1 and 2.8.0.
- ``whisper_log_mel``: openai-whisper's ``log_mel_spectrogram`` (MIT), with its filters computed by
  ``librosa_mel``, as its ``mel_filters.npz`` was made.
- ``hf_mel_filter_bank``, ``hf_whisper_log_mel``: transformers' ``audio_utils.mel_filter_bank`` and
  ``WhisperFeatureExtractor`` (4.57, its torch path; Apache-2.0): the Whisper features of Mizuoto's speech
  recognition, which differ from whisper's in the filterbank's arithmetic and the batch-wise maximum.
- ``matcha_mel``: Matcha-TTS's ``mel_spectrogram`` (MIT), used by CosyVoice. Its filterbank is built per
  call (or held by the caller), never cached by ``fmax`` and device only, the bug that made
  CosyVoice's workers restart to unload.
"""

from __future__ import annotations

import math
import warnings
from typing import Any, Literal

import numpy as np
import torch
from torch.nn import functional as F

# -- librosa ------------------------------------------------------------------------------------


def hz_to_mel(frequencies: np.ndarray | float, htk: bool = False) -> np.ndarray:
    frequencies = np.asanyarray(frequencies)
    if htk:
        return 2595.0 * np.log10(1.0 + frequencies / 700.0)
    f_min = 0.0
    f_sp = 200.0 / 3
    mels = (frequencies - f_min) / f_sp
    min_log_hz = 1000.0
    min_log_mel = (min_log_hz - f_min) / f_sp
    logstep = np.log(6.4) / 27.0
    if frequencies.ndim:
        log_t = frequencies >= min_log_hz
        mels[log_t] = min_log_mel + np.log(frequencies[log_t] / min_log_hz) / logstep
    elif frequencies >= min_log_hz:
        mels = min_log_mel + np.log(frequencies / min_log_hz) / logstep
    return mels


def mel_to_hz(mels: np.ndarray | float, htk: bool = False) -> np.ndarray:
    mels = np.asanyarray(mels)
    if htk:
        return 700.0 * (10.0 ** (mels / 2595.0) - 1.0)
    f_min = 0.0
    f_sp = 200.0 / 3
    freqs = f_min + f_sp * mels
    min_log_hz = 1000.0
    min_log_mel = (min_log_hz - f_min) / f_sp
    logstep = np.log(6.4) / 27.0
    if mels.ndim:
        log_t = mels >= min_log_mel
        freqs[log_t] = min_log_hz * np.exp(logstep * (mels[log_t] - min_log_mel))
    elif mels >= min_log_mel:
        freqs = min_log_hz * np.exp(logstep * (mels - min_log_mel))
    return freqs


def mel_frequencies(n_mels: int = 128, fmin: float = 0.0, fmax: float = 11025.0, htk: bool = False) -> np.ndarray:
    min_mel = hz_to_mel(fmin, htk=htk)
    max_mel = hz_to_mel(fmax, htk=htk)
    return mel_to_hz(np.linspace(min_mel, max_mel, n_mels), htk=htk)


def librosa_mel(
    sr: float,
    n_fft: int,
    n_mels: int = 128,
    fmin: float = 0.0,
    fmax: float | None = None,
    htk: bool = False,
    norm: Literal["slaney"] | None = "slaney",
    dtype: type = np.float32,
) -> np.ndarray:
    """``librosa.filters.mel``: an (n_mels, 1 + n_fft // 2) filterbank."""
    if fmax is None:
        fmax = float(sr) / 2
    n_mels = int(n_mels)
    weights = np.zeros((n_mels, int(1 + n_fft // 2)), dtype=dtype)
    fftfreqs = np.fft.rfftfreq(n=n_fft, d=1.0 / sr)
    mel_f = mel_frequencies(n_mels + 2, fmin=fmin, fmax=fmax, htk=htk)
    fdiff = np.diff(mel_f)
    ramps = np.subtract.outer(mel_f, fftfreqs)
    for i in range(n_mels):
        lower = -ramps[i] / fdiff[i]
        upper = ramps[i + 2] / fdiff[i + 1]
        weights[i] = np.maximum(0, np.minimum(lower, upper))
    if norm == "slaney":
        enorm = 2.0 / (mel_f[2 : n_mels + 2] - mel_f[:n_mels])
        weights *= enorm[:, np.newaxis]
    elif norm is not None:
        raise ValueError(f"Unsupported norm={norm}")
    return weights


# -- torchaudio ---------------------------------------------------------------------------------


def _ta_hz_to_mel(freq: float, mel_scale: str = "htk") -> float:
    if mel_scale == "htk":
        return 2595.0 * math.log10(1.0 + (freq / 700.0))
    f_min = 0.0
    f_sp = 200.0 / 3
    mels = (freq - f_min) / f_sp
    min_log_hz = 1000.0
    min_log_mel = (min_log_hz - f_min) / f_sp
    logstep = math.log(6.4) / 27.0
    if freq >= min_log_hz:
        mels = min_log_mel + math.log(freq / min_log_hz) / logstep
    return mels


def _ta_mel_to_hz(mels: torch.Tensor, mel_scale: str = "htk") -> torch.Tensor:
    if mel_scale == "htk":
        return 700.0 * (10.0 ** (mels / 2595.0) - 1.0)
    f_min = 0.0
    f_sp = 200.0 / 3
    freqs = f_min + f_sp * mels
    min_log_hz = 1000.0
    min_log_mel = (min_log_hz - f_min) / f_sp
    logstep = math.log(6.4) / 27.0
    log_t = mels >= min_log_mel
    freqs[log_t] = min_log_hz * torch.exp(logstep * (mels[log_t] - min_log_mel))
    return freqs


def melscale_fbanks(n_freqs: int, f_min: float, f_max: float, n_mels: int, sample_rate: int, norm: str | None = None, mel_scale: str = "htk") -> torch.Tensor:
    """``torchaudio.functional.melscale_fbanks``: an (n_freqs, n_mels) filterbank, computed in float32."""
    if norm is not None and norm != "slaney":
        raise ValueError('norm must be one of None or "slaney"')
    if mel_scale not in ("slaney", "htk"):
        raise ValueError('mel_scale should be one of "htk" or "slaney".')
    all_freqs = torch.linspace(0, sample_rate // 2, n_freqs)
    m_min = _ta_hz_to_mel(f_min, mel_scale=mel_scale)
    m_max = _ta_hz_to_mel(f_max, mel_scale=mel_scale)
    m_pts = torch.linspace(m_min, m_max, n_mels + 2)
    f_pts = _ta_mel_to_hz(m_pts, mel_scale=mel_scale)
    f_diff = f_pts[1:] - f_pts[:-1]
    slopes = f_pts.unsqueeze(0) - all_freqs.unsqueeze(1)
    zero = torch.zeros(1)
    down_slopes = (-1.0 * slopes[:, :-2]) / f_diff[:-1]
    up_slopes = slopes[:, 2:] / f_diff[1:]
    fb = torch.max(zero, torch.min(down_slopes, up_slopes))
    if norm == "slaney":
        enorm = 2.0 / (f_pts[2 : n_mels + 2] - f_pts[:n_mels])
        fb *= enorm.unsqueeze(0)
    if (fb.max(dim=0).values == 0.0).any():
        warnings.warn(f"At least one mel filterbank has all zero values (n_mels {n_mels}, n_freqs {n_freqs})", stacklevel=2)
    return fb


def spectrogram(
    waveform: torch.Tensor,
    pad: int,
    window: torch.Tensor,
    n_fft: int,
    hop_length: int,
    win_length: int,
    power: float | None,
    normalized: bool | str,
    center: bool = True,
    pad_mode: str = "reflect",
    onesided: bool = True,
) -> torch.Tensor:
    """``torchaudio.functional.spectrogram``."""
    if pad > 0:
        waveform = F.pad(waveform, (pad, pad), "constant")
    frame_length_norm, window_norm = normalized == "frame_length", normalized is True or normalized == "window"
    shape = waveform.size()
    waveform = waveform.reshape(-1, shape[-1])
    spec_f = torch.stft(
        input=waveform,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window=window,
        center=center,
        pad_mode=pad_mode,
        normalized=frame_length_norm,
        onesided=onesided,
        return_complex=True,
    )
    spec_f = spec_f.reshape(shape[:-1] + spec_f.shape[-2:])
    if window_norm:
        spec_f /= window.pow(2.0).sum().sqrt()
    if power is not None:
        if power == 1.0:
            return spec_f.abs()
        return spec_f.abs().pow(power)
    return spec_f


class MelSpectrogram(torch.nn.Module):
    """``torchaudio.transforms.MelSpectrogram`` (hann window, its defaults), with the same buffers.

    Its state dict has torchaudio's keys (``spectrogram.window``, ``mel_scale.fb``), so upstream checkpoints
    that saved the transform inside a model load into it.
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        n_fft: int = 400,
        win_length: int | None = None,
        hop_length: int | None = None,
        f_min: float = 0.0,
        f_max: float | None = None,
        pad: int = 0,
        n_mels: int = 128,
        power: float = 2.0,
        normalized: bool = False,
        center: bool = True,
        pad_mode: str = "reflect",
        norm: str | None = None,
        mel_scale: str = "htk",
    ) -> None:
        super().__init__()
        self.win_length = win_length if win_length is not None else n_fft
        self.hop_length = hop_length if hop_length is not None else self.win_length // 2
        self.n_fft, self.pad, self.power, self.normalized, self.center, self.pad_mode = n_fft, pad, power, normalized, center, pad_mode
        self.spectrogram = torch.nn.Module()
        self.spectrogram.register_buffer("window", torch.hann_window(self.win_length))
        self.mel_scale = torch.nn.Module()
        self.mel_scale.register_buffer("fb", melscale_fbanks(n_fft // 2 + 1, f_min, f_max if f_max is not None else float(sample_rate // 2), n_mels, sample_rate, norm, mel_scale))

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        window = self.spectrogram.get_buffer("window")
        fb = self.mel_scale.get_buffer("fb")
        spec = spectrogram(waveform, self.pad, window, self.n_fft, self.hop_length, self.win_length, self.power, self.normalized, self.center, self.pad_mode)
        return torch.matmul(spec.transpose(-1, -2), fb).transpose(-1, -2)


def mel_spectrogram(waveform: torch.Tensor, sample_rate: int, n_fft: int, n_mels: int, **kwargs: Any) -> torch.Tensor:
    """One call of ``torchaudio.transforms.MelSpectrogram(...)(waveform)`` (keyword arguments as there)."""
    return MelSpectrogram(sample_rate=sample_rate, n_fft=n_fft, n_mels=n_mels, **kwargs).to(waveform.device)(waveform)


# -- Whisper ------------------------------------------------------------------------------------

WHISPER_RATE = 16000
WHISPER_N_FFT = 400
WHISPER_HOP = 160


def whisper_mel_filters(n_mels: int = 80) -> torch.Tensor:
    """The filters of whisper's ``mel_filters.npz``: ``librosa.filters.mel(sr=16000, n_fft=400, n_mels=…)``."""
    return torch.from_numpy(librosa_mel(sr=WHISPER_RATE, n_fft=WHISPER_N_FFT, n_mels=n_mels))


def whisper_log_mel(audio: torch.Tensor, n_mels: int = 80, padding: int = 0, filters: torch.Tensor | None = None) -> torch.Tensor:
    """``whisper.log_mel_spectrogram`` for 16 kHz audio (a tensor; load and resample before)."""
    if padding > 0:
        audio = F.pad(audio, (0, padding))
    window = torch.hann_window(WHISPER_N_FFT).to(audio.device)
    stft = torch.stft(audio, WHISPER_N_FFT, WHISPER_HOP, window=window, return_complex=True)
    magnitudes = stft[..., :-1].abs() ** 2
    fb = (filters if filters is not None else whisper_mel_filters(n_mels)).to(audio.device)
    mel_spec = fb @ magnitudes
    log_spec = torch.clamp(mel_spec, min=1e-10).log10()
    log_spec = torch.maximum(log_spec, log_spec.max() - 8.0)
    return (log_spec + 4.0) / 4.0


# -- transformers' WhisperFeatureExtractor -------------------------------------------------------


def _hf_hertz_to_mel(freq: float) -> float:
    """``transformers.audio_utils.hertz_to_mel(mel_scale="slaney")`` for one frequency."""
    min_log_hertz = 1000.0
    min_log_mel = 15.0
    logstep = 27.0 / np.log(6.4)
    if freq >= min_log_hertz:
        return float(min_log_mel + np.log(freq / min_log_hertz) * logstep)
    return 3.0 * freq / 200.0


def _hf_mel_to_hertz(mels: np.ndarray) -> np.ndarray:
    """``transformers.audio_utils.mel_to_hertz(mel_scale="slaney")`` for an array."""
    min_log_hertz = 1000.0
    min_log_mel = 15.0
    logstep = np.log(6.4) / 27.0
    freq = 200.0 * mels / 3.0
    log_region = mels >= min_log_mel
    freq[log_region] = min_log_hertz * np.exp(logstep * (mels[log_region] - min_log_mel))
    return freq


def hf_mel_filter_bank(num_frequency_bins: int, num_mel_filters: int, min_frequency: float, max_frequency: float, sampling_rate: int) -> np.ndarray:
    """``transformers.audio_utils.mel_filter_bank(..., norm="slaney", mel_scale="slaney")``: (bins, filters), float64."""
    mel_min = _hf_hertz_to_mel(min_frequency)
    mel_max = _hf_hertz_to_mel(max_frequency)
    mel_freqs = np.linspace(mel_min, mel_max, num_mel_filters + 2)
    filter_freqs = _hf_mel_to_hertz(mel_freqs)
    fft_freqs = np.linspace(0, sampling_rate // 2, num_frequency_bins)
    filter_diff = np.diff(filter_freqs)
    slopes = np.expand_dims(filter_freqs, 0) - np.expand_dims(fft_freqs, 1)
    down_slopes = -slopes[:, :-2] / filter_diff[:-1]
    up_slopes = slopes[:, 2:] / filter_diff[1:]
    mel_filters = np.maximum(np.zeros(1), np.minimum(down_slopes, up_slopes))
    enorm = 2.0 / (filter_freqs[2 : num_mel_filters + 2] - filter_freqs[:num_mel_filters])
    mel_filters *= np.expand_dims(enorm, 0)
    return mel_filters


def hf_whisper_filters(n_mels: int, sampling_rate: int = WHISPER_RATE, n_fft: int = WHISPER_N_FFT) -> torch.Tensor:
    """``WhisperFeatureExtractor.mel_filters`` as its torch path uses them: (1 + n_fft // 2, n_mels), float32."""
    return torch.from_numpy(hf_mel_filter_bank(1 + n_fft // 2, n_mels, 0.0, 8000.0, sampling_rate)).to(torch.float32)


def hf_whisper_log_mel(waveform: torch.Tensor, filters: torch.Tensor, n_fft: int = WHISPER_N_FFT, hop_length: int = WHISPER_HOP) -> torch.Tensor:
    """``WhisperFeatureExtractor._torch_extract_fbank_features`` (no dither) of a (batch, samples) float32 batch.

    ``filters`` comes from ``hf_whisper_filters``. Pad or cut the audio to the feature extractor's length first
    (30 s of zeros after the audio). The floor is 8 below each item's maximum.
    """
    window = torch.hann_window(n_fft, device=waveform.device)
    stft = torch.stft(waveform, n_fft, hop_length, window=window, return_complex=True)
    magnitudes = stft[..., :-1].abs() ** 2
    mel_spec = filters.to(waveform.device).T @ magnitudes
    log_spec = torch.clamp(mel_spec, min=1e-10).log10()
    if waveform.dim() == 2:
        max_val = log_spec.max(dim=2, keepdim=True)[0].max(dim=1, keepdim=True)[0]
        log_spec = torch.maximum(log_spec, max_val - 8.0)
    else:
        log_spec = torch.maximum(log_spec, log_spec.max() - 8.0)
    return (log_spec + 4.0) / 4.0


# -- Matcha-TTS (CosyVoice) ---------------------------------------------------------------------


def matcha_mel(
    y: torch.Tensor,
    n_fft: int,
    num_mels: int,
    sampling_rate: int,
    hop_size: int,
    win_size: int,
    fmin: float,
    fmax: float | None,
    center: bool = False,
    basis: torch.Tensor | None = None,
) -> torch.Tensor:
    """``matcha.utils.audio.mel_spectrogram``: log-magnitude mel with a reflect pad and ``center=False``.

    ``basis`` is the filterbank from ``matcha_basis`` for these parameters; without it, one is built.
    """
    if basis is None:
        basis = matcha_basis(sampling_rate, n_fft, num_mels, fmin, fmax)
    basis = basis.to(y.device)
    window = torch.hann_window(win_size).to(y.device)
    pad = int((n_fft - hop_size) / 2)
    y = F.pad(y.unsqueeze(1), (pad, pad), mode="reflect").squeeze(1)
    spec = torch.view_as_real(
        torch.stft(y, n_fft, hop_length=hop_size, win_length=win_size, window=window, center=center, pad_mode="reflect", normalized=False, onesided=True, return_complex=True)
    )
    spec = torch.sqrt(spec.pow(2).sum(-1) + (1e-9))
    spec = torch.matmul(basis, spec)
    return torch.log(torch.clamp(spec, min=1e-5) * 1)


def matcha_basis(sampling_rate: int, n_fft: int, num_mels: int, fmin: float, fmax: float | None) -> torch.Tensor:
    return torch.from_numpy(librosa_mel(sr=sampling_rate, n_fft=n_fft, n_mels=num_mels, fmin=fmin, fmax=fmax)).float()
