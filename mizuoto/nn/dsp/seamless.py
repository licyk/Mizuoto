"""``SeamlessM4TFeatureExtractor`` (transformers, Apache-2.0): w2v-BERT 2.0's input features, in numpy.

Kaldi-style log-mel filterbanks from transformers' ``audio_utils.spectrogram`` (float64 frames: DC offset
removed, pre-emphasis 0.97, a Povey window, a 512-point real FFT kept as complex64, power, 80 triangular
filters on the Kaldi mel scale, floored and logged), normalised per mel bin, padded to an even number of
frames with ones and stacked in pairs (160 features per step). The frame loop is vectorised; each step does
the arithmetic upstream does, in the same order.
"""

from __future__ import annotations

import numpy as np

SAMPLE_RATE = 16000
NUM_MEL_BINS = 80
STRIDE = 2
PADDING_VALUE = 1.0


def _hz_to_mel_kaldi(freq: np.ndarray | float) -> np.ndarray | float:
    return 1127.0 * np.log(1.0 + (np.asarray(freq) / 700.0))


def mel_filters(num_frequency_bins: int = 257, num_mel_filters: int = NUM_MEL_BINS, min_frequency: float = 20, sampling_rate: int = SAMPLE_RATE) -> np.ndarray:
    """``mel_filter_bank(..., norm=None, mel_scale="kaldi", triangularize_in_mel_space=True)``: (bins, filters)."""
    mel_min = _hz_to_mel_kaldi(min_frequency)
    mel_max = _hz_to_mel_kaldi(sampling_rate // 2)
    mel_freqs = np.linspace(mel_min, mel_max, num_mel_filters + 2)
    fft_bin_width = sampling_rate / ((num_frequency_bins - 1) * 2)
    fft_freqs = _hz_to_mel_kaldi(fft_bin_width * np.arange(num_frequency_bins))
    filter_diff = np.diff(mel_freqs)
    slopes = np.expand_dims(mel_freqs, 0) - np.expand_dims(fft_freqs, 1)
    down = -slopes[:, :-2] / filter_diff[:-1]
    up = slopes[:, 2:] / filter_diff[1:]
    return np.maximum(np.zeros(1), np.minimum(down, up))


def povey_window(length: int = 400) -> np.ndarray:
    return np.power(np.hanning(length), 0.85)


_FILTERS = None
_WINDOW = None


def fbank(waveform: np.ndarray) -> np.ndarray:
    """(frames, 80) float32 log-mel features of mono float samples at 16 kHz."""
    global _FILTERS, _WINDOW
    if _FILTERS is None or _WINDOW is None:
        _FILTERS, _WINDOW = mel_filters(), povey_window()
    frame_length, hop, fft_length = 400, 160, 512
    wave = (np.squeeze(waveform) * (2**15)).astype(np.float64)
    num_frames = int(1 + np.floor((wave.size - frame_length) / hop))
    index = np.arange(frame_length)[None, :] + hop * np.arange(num_frames)[:, None]
    frames = wave[index]
    frames = frames - frames.mean(axis=1, keepdims=True)
    frames[:, 1:frame_length] -= 0.97 * frames[:, : frame_length - 1].copy()
    frames[:, 0] *= 1 - 0.97
    frames *= _WINDOW.astype(np.float64)
    buffer = np.zeros((num_frames, fft_length))
    buffer[:, :frame_length] = frames
    spec = np.fft.rfft(buffer, axis=1).astype(np.complex64)
    power = np.abs(spec, dtype=np.float64) ** 2.0
    mel = np.maximum(1.192092955078125e-07, np.dot(_FILTERS.T, power.T))
    return np.asarray(np.log(mel), np.float32).T


def features(waveform: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """``extractor(waveform, sampling_rate=16000, return_tensors="pt")``: (input_features (1, T/2, 160), attention_mask (1, T/2))."""
    x = fbank(np.asarray(waveform, dtype=np.float32))
    x = (x - np.expand_dims(x.mean(0), 0)) / np.sqrt(np.expand_dims(x.var(0, ddof=1), 0) + 1e-7)
    frames = x.shape[0]
    target = frames + (frames % STRIDE)
    mask = np.zeros(target, dtype=np.int32)
    mask[:frames] = 1
    if target != frames:
        x = np.concatenate([x, np.full((target - frames, x.shape[1]), PADDING_VALUE, dtype=x.dtype)], axis=0)
    stacked = x.reshape(1, target // STRIDE, x.shape[1] * STRIDE)
    mask = mask[np.arange(target) % STRIDE == 1][None, :]
    return stacked, mask
