"""mizuoto.nn.dsp against the libraries upstream engines call (tier T0).

The fixtures are made by ``scripts/parity/dsp_fixtures.py`` inside the engine environments, with the
real torchaudio, librosa, whisper, soxr and Matcha-TTS, on an x86-64 Linux machine without AVX-512. On such a
machine, under the torch version that made them, the outputs must be identical. Elsewhere the kernels round
differently (numpy's AVX-512 exp and log, the CRT's libm on Windows, torch's FFT and soxr's SIMD on Arm, another
torch), so there they must agree within that rounding: max abs error <= 1e-4 x max(1, max |ref|) in float32, 1e-12 x
max(1, max |ref|) in float64. The float32 bound is wider than T1's because these ops end in a log, or in differences
of frequencies in Hz, which turn one rounding of an intermediate value into about 1e-5 (CI measured up to 4e-5).
"""

from __future__ import annotations

import importlib
import json
import platform
import sys
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


def same(actual, expected: np.ndarray, exact: bool, err_msg: str = "") -> None:
    a = actual.detach().cpu().numpy() if hasattr(actual, "detach") else np.asarray(actual)
    assert a.shape == expected.shape and a.dtype == expected.dtype, (a.shape, expected.shape, a.dtype, expected.dtype)
    if exact:
        np.testing.assert_array_equal(a, expected, err_msg=err_msg)
    else:
        tolerance = (1e-12 if expected.dtype == np.float64 else 1e-4) * max(1.0, float(np.abs(expected).max()))
        np.testing.assert_allclose(a, expected, rtol=0, atol=tolerance, err_msg=err_msg)


def avx512() -> bool:
    """Whether numpy dispatches to AVX-512 (its float64 exp and log round differently there)."""
    for name in ("numpy._core._multiarray_umath", "numpy.core._multiarray_umath"):  # numpy 2, numpy 1
        try:
            return bool(getattr(importlib.import_module(name), "__cpu_features__", {}).get("AVX512F"))
        except ImportError:
            continue
    return False


# The fixtures' machine: numpy and soxr compute there as on any x86-64 Linux machine with AVX2 and without AVX-512.
FIXTURE_MACHINE = sys.platform == "linux" and platform.machine() == "x86_64" and not avx512() and torch.backends.cpu.get_cpu_capability() == "AVX2"


def exact_torch(meta: dict) -> bool:
    return FIXTURE_MACHINE and meta.get("torch") == torch.__version__


def test_librosa_mel_filters_are_identical():
    data, _ = load("librosa")
    for name, sr, n_fft, n_mels, fmin, fmax, htk, norm in LIBROSA_MELS:
        same(mel.librosa_mel(sr=sr, n_fft=n_fft, n_mels=n_mels, fmin=fmin, fmax=fmax, htk=htk, norm=norm), data[f"librosa_mel/{name}"], FIXTURE_MACHINE, name)


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
        same(resample.soxr_resample(signal(orig, 0.25), orig, new), data[f"soxr/{name}"], FIXTURE_MACHINE, name)


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
        same(mel.whisper_mel_filters(n), data[f"whisper/filters{n}"], FIXTURE_MACHINE)
        same(mel.whisper_log_mel(x, n_mels=n), data[f"whisper/logmel{n}"], exact)


def test_transformers_whisper_features_and_filters():
    data, meta = load("hf_whisper")
    exact = exact_torch(meta)

    def features(x: np.ndarray, n: int):
        padded = np.zeros((1, 480000), dtype=np.float32)
        padded[0, : min(len(x), 480000)] = x[:480000]
        return mel.hf_whisper_log_mel(torch.from_numpy(padded), mel.hf_whisper_filters(n))

    for n in (80, 128):
        same(mel.hf_mel_filter_bank(201, n, 0.0, 8000.0, 16000), data[f"hf_whisper/filters{n}"], FIXTURE_MACHINE)
        same(features(signal(16000, 0.5), n), data[f"hf_whisper/features{n}"], exact)
    long = np.concatenate([signal(16000, 1.0)] * 31)
    same(features(long, 128)[:, :, -200:], data["hf_whisper/features128_cut"], exact)


def test_matcha_mel_for_each_configuration():
    data, meta = load("matcha")
    for name, n_fft, n_mels, sr, hop, win, fmin, fmax in MATCHA:
        x = torch.from_numpy(signal(sr, 0.5)).unsqueeze(0)
        same(mel.matcha_mel(x, n_fft, n_mels, sr, hop, win, fmin, fmax, center=False), data[f"matcha/{name}"], exact_torch(meta))


def test_tiles_with_context_give_the_whole_of_a_convolution_stack():
    """``context_tiled_1d`` over a causal and a centred stack of convolutions with a transposed one: with the context
    the receptive field needs (here 4 frames before, and 4 after for the centred one), the tiles' samples are the
    whole's to float rounding, the length exact; with too little context they are not."""
    from mizuoto.nn.dsp.tiles import context_tiled_1d

    gen = torch.Generator().manual_seed(0)

    class Causal(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.up = torch.nn.ConvTranspose1d(4, 3, 8, stride=4)
            self.conv = torch.nn.Conv1d(3, 1, 9)

        def forward(self, x):
            y = torch.tanh(self.up(x)[..., : x.shape[-1] * 4])  # each sample from its frame and the one before
            return self.conv(torch.nn.functional.pad(y, (8, 0)))  # 8 samples before: 2 frames

    centred = torch.nn.Sequential(torch.nn.ConvTranspose1d(4, 3, 8, stride=4, padding=2), torch.nn.Tanh(), torch.nn.Conv1d(3, 1, 9, padding=4))
    x = torch.randn(1, 4, 1000, generator=gen)
    for model, after in ((Causal(), 0), (centred, 4)):
        for p in model.parameters():
            p.data = torch.randn(p.shape, generator=gen) * 0.3
        with torch.no_grad():
            whole = model(x)
            tiled = context_tiled_1d(model, x, 4, tile=128, before=4, after=after)
            short = context_tiled_1d(model, x, 4, tile=128, before=0, after=0)
        assert tiled.shape == whole.shape == (1, 1, 4000)
        assert float((tiled - whole).abs().max()) < 1e-6
        assert float((short - whole).abs().max()) > 1e-3


def test_a_dac_decoder_tiled_with_its_context_gives_the_whole():
    from mizuoto.nn.codecs.dac import DacDecoder
    from mizuoto.nn.dsp.tiles import context_tiled_1d
    from mizuoto.runtime import ops

    decoder = DacDecoder(8, 16, [4, 3, 2], odd_output_padding=True, tanh=False)
    ops.init_random(decoder, torch.Generator().manual_seed(1), scale=0.3)
    context = decoder.context_frames()
    assert context == 21  # 3 + (2 + 39/4) + (2/4 + 39/12) + (2/12 + 39/24) + 3/24, rounded up
    x = torch.randn(1, 8, 300, generator=torch.Generator().manual_seed(2))
    with torch.no_grad():
        whole = decoder(x)
        tiled = context_tiled_1d(decoder, x, 24, tile=64, before=context, after=context)
    assert tiled.shape == whole.shape and float((tiled - whole).abs().max()) < 1e-5 * float(whole.abs().max())


def _tiles_match(decode, x, hop, before, after, tolerance=1e-5):
    """Tiles of 64 frames with the context give the whole's samples to rounding; with one frame of context they do not."""
    from mizuoto.nn.dsp.tiles import context_tiled_1d

    with torch.no_grad():
        whole = decode(x)
        tiled = context_tiled_1d(decode, x, hop, tile=64, before=before, after=after)
        short = context_tiled_1d(decode, x, hop, tile=64, before=min(before, 1), after=min(after, 1))
    peak = float(whole.abs().max())
    assert tiled.shape == whole.shape and peak > 0
    assert float((tiled - whole).abs().max()) < tolerance * peak
    assert float((short - whole).abs().max()) > 1e-3 * peak


def test_vocos_bigvgan_and_voxcpms_vae_tiled_with_their_context_give_the_whole():
    """The decoders whose out-of-memory fallback tiles with context: their ``context_frames()`` bound their receptive
    field (Vocos with its inverse STFT, BigVGAN with either AMP block, VoxCPM's causal VAE plain and conditioned)."""
    from mizuoto.engines.voxcpm.config import VaeConfig
    from mizuoto.engines.voxcpm.vae import AudioVAE
    from mizuoto.nn.vocoders.bigvgan import BigVGAN
    from mizuoto.nn.vocoders.vocos import Vocos
    from mizuoto.runtime import ops

    gen = torch.Generator().manual_seed(3)
    vocos = Vocos(
        {
            "backbone": {"init_args": {"input_channels": 20, "dim": 16, "intermediate_dim": 32, "num_layers": 3}},
            "head": {"init_args": {"dim": 16, "n_fft": 64, "hop_length": 16, "padding": "center"}},
        }
    )
    ops.init_random(vocos, gen, scale=0.2)
    assert vocos.context_frames() == 3 + 3 * 3 + 2
    _tiles_match(vocos.decode, torch.randn(1, 20, 400, generator=gen), 16, vocos.context_frames(), vocos.context_frames())

    for resblock in ("1", "2"):
        h = {
            "resblock": resblock,
            "upsample_rates": [4, 2],
            "upsample_kernel_sizes": [8, 4],
            "upsample_initial_channel": 16,
            "resblock_kernel_sizes": [3, 5],
            "resblock_dilation_sizes": [[1, 3], [1, 2]],
            "use_tanh_at_final": False,
            "activation": "snakebeta",
            "snake_logscale": True,
            "num_mels": 10,
        }
        bigvgan = BigVGAN(h)
        ops.init_random(bigvgan, gen, scale=0.2)
        with torch.no_grad():
            for name, buffer in bigvgan.named_buffers():  # the anti-aliasing filters are computed, not drawn
                buffer.copy_(BigVGAN(h).get_buffer(name))
            for name, p in bigvgan.named_parameters():
                if name.endswith(("alpha", "beta")):
                    p.zero_()  # exp(0) = 1: Snake at its usual scale
        context = bigvgan.context_frames()
        _tiles_match(bigvgan, torch.randn(1, 10, 300, generator=gen), 8, context, context)

    for config in (
        VaeConfig(encoder_dim=2, encoder_rates=[4, 5], latent_dim=8, decoder_dim=16, decoder_rates=[5, 4], depthwise=True),
        VaeConfig(encoder_dim=2, encoder_rates=[4, 5], latent_dim=8, decoder_dim=16, decoder_rates=[5, 4, 3], out_sample_rate=48000, sr_bin_boundaries=[20000, 40000], version=2),
    ):
        vae = AudioVAE(config)
        ops.init_random(vae, gen, scale=0.3)
        _tiles_match(vae.decode, torch.randn(1, 8, 300, generator=gen), vae.decode_chunk_size, vae.decoder.context_frames(), 0)
