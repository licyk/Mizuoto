"""mizuoto.nn.dsp against the libraries upstream engines call (tier T0).

The fixtures are made by ``scripts/parity/dsp_fixtures.py`` inside the engine environments, with the
real torchaudio, librosa, whisper, soxr and Matcha-TTS. Under the torch version that made them the
outputs must be identical; under another torch, equal within float rounding.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from mizuoto.nn.dsp import kaldi, mel, resample  # noqa: E402
from scripts.parity.dsp_fixtures import FFMPEG_RESAMPLES, KALDI, LIBROSA_MELS, MATCHA, RESAMPLES, SOXR, TA_FBANKS, TA_MELSPECS, pcm16, signal  # noqa: E402

FIXTURES = Path(__file__).resolve().parent.parent / "fixtures" / "dsp"


def load(group: str) -> tuple[dict[str, np.ndarray], dict]:
    with np.load(FIXTURES / f"{group}.npz") as f:
        data = {k.replace("__", "/"): f[k] for k in f.files if k != "meta"}
        meta = json.loads(str(f["meta"]))
    return data, meta


def same(actual, expected: np.ndarray, exact: bool) -> None:
    a = actual.detach().cpu().numpy() if hasattr(actual, "detach") else np.asarray(actual)
    assert a.shape == expected.shape and a.dtype == expected.dtype, (a.shape, expected.shape, a.dtype, expected.dtype)
    if exact:
        np.testing.assert_array_equal(a, expected)
    else:
        np.testing.assert_allclose(a, expected, rtol=1e-5, atol=1e-6)


def exact_torch(meta: dict) -> bool:
    return meta.get("torch") == torch.__version__


def test_librosa_mel_filters_are_identical():
    data, _ = load("librosa")
    for name, sr, n_fft, n_mels, fmin, fmax, htk, norm in LIBROSA_MELS:
        np.testing.assert_array_equal(mel.librosa_mel(sr=sr, n_fft=n_fft, n_mels=n_mels, fmin=fmin, fmax=fmax, htk=htk, norm=norm), data[f"librosa_mel/{name}"], err_msg=name)


def test_torchaudio_filterbanks_and_mel_spectrograms():
    data, meta = load("torchaudio")
    exact = exact_torch(meta)
    for name, n_freqs, f_min, f_max, n_mels, sr, norm, scale in TA_FBANKS:
        same(mel.melscale_fbanks(n_freqs, f_min, f_max, n_mels, sr, norm, scale), data[f"fbank/{name}"], exact)
    for name, sr, n_fft, win, hop, n_mels, power, center, norm, scale, f_min, f_max in TA_MELSPECS:
        t = mel.MelSpectrogram(sr, n_fft, win, hop, f_min, f_max, 0, n_mels, power=power, center=center, norm=norm, mel_scale=scale)
        same(t(torch.from_numpy(signal(sr, 0.5))), data[f"melspec/{name}"], exact)


def test_torchaudio_resampling():
    data, meta = load("torchaudio")
    exact = exact_torch(meta)
    for name, orig, new, method, kw in RESAMPLES:
        x = torch.from_numpy(signal(orig, 0.25))
        same(resample.resample(x, orig, new, resampling_method=method, **kw), data[f"resample/{name}"], exact)
        same(resample.resample(x.double(), orig, new, resampling_method=method, **kw), data[f"resample64/{name}"], exact)
        same(resample.Resample(orig, new, method, **kw)(x), data[f"transform/{name}"], exact)


def test_kaldi_fbank():
    data, meta = load("torchaudio")
    for name, sr, kw in KALDI:
        options: dict = dict(kw)
        same(kaldi.fbank(torch.from_numpy(signal(sr, 0.5)).unsqueeze(0), **options), data[f"kaldi/{name}"], exact_torch(meta))


def test_soxr_as_librosa_calls_it():
    pytest.importorskip("soxr")
    data, _ = load("soxr")
    for name, orig, new in SOXR:
        np.testing.assert_array_equal(resample.soxr_resample(signal(orig, 0.25), orig, new), data[f"soxr/{name}"], err_msg=name)


def test_ffmpeg_resampling():
    """The ffmpeg program's default resampler (libswresample, x86 FMA3 kernel), as GPT-SoVITS's training reads clips."""
    from mizuoto.nn.dsp.swresample import resample as ffmpeg_resample

    data, _ = load("ffmpeg")
    for orig, new in FFMPEG_RESAMPLES:
        np.testing.assert_array_equal(ffmpeg_resample(pcm16(orig), orig, new), data[f"ffmpeg/{orig}-{new}"], err_msg=f"{orig}-{new}")


def test_whisper_log_mel_and_its_filters():
    data, meta = load("whisper")
    exact = exact_torch(meta)
    x = torch.from_numpy(signal(16000, 0.5))
    for n in (80, 128):
        np.testing.assert_array_equal(mel.whisper_mel_filters(n).numpy(), data[f"whisper/filters{n}"])
        same(mel.whisper_log_mel(x, n_mels=n), data[f"whisper/logmel{n}"], exact)


def test_transformers_whisper_features_and_filters():
    data, meta = load("hf_whisper")
    exact = exact_torch(meta)

    def features(x: np.ndarray, n: int):
        padded = np.zeros((1, 480000), dtype=np.float32)
        padded[0, : min(len(x), 480000)] = x[:480000]
        return mel.hf_whisper_log_mel(torch.from_numpy(padded), mel.hf_whisper_filters(n))

    for n in (80, 128):
        np.testing.assert_array_equal(mel.hf_mel_filter_bank(201, n, 0.0, 8000.0, 16000), data[f"hf_whisper/filters{n}"])
        same(features(signal(16000, 0.5), n), data[f"hf_whisper/features{n}"], exact)
    long = np.concatenate([signal(16000, 1.0)] * 31)
    same(features(long, 128)[:, :, -200:], data["hf_whisper/features128_cut"], exact)


def test_matcha_mel_for_each_configuration():
    data, meta = load("matcha")
    for name, n_fft, n_mels, sr, hop, win, fmin, fmax in MATCHA:
        x = torch.from_numpy(signal(sr, 0.5)).unsqueeze(0)
        same(mel.matcha_mel(x, n_fft, n_mels, sr, hop, win, fmin, fmax, center=False), data[f"matcha/{name}"], exact_torch(meta))
