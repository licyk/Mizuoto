"""Reference outputs of the DSP libraries upstream engines call, for mizuoto.nn.dsp's tests.

    ~/.local/share/kodama/envs/f5tts/.venv/bin/python scripts/parity/dsp_fixtures.py torchaudio librosa soxr
    ~/.local/share/kodama/envs/cosyvoice/.venv/bin/python scripts/parity/dsp_fixtures.py whisper
    ~/.local/share/kodama/envs/asr/.venv/bin/python scripts/parity/dsp_fixtures.py hf_whisper
    FFMPEG=<ffmpeg 7.0.2> python scripts/parity/dsp_fixtures.py ffmpeg   # GPT-SoVITS's load_audio (x86, FMA3)

Runs in an engine environment, which has the real torchaudio, librosa, whisper and soxr. It computes
each function for the configurations the engines use, on a fixed test signal, and writes
``tests/fixtures/dsp/<group>.npz`` with the inputs and outputs, plus the library versions. The tests
compare ``mizuoto.nn.dsp`` with these: identical under the same torch version, and within float
rounding otherwise.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

OUT = Path(__file__).resolve().parents[2] / "tests" / "fixtures" / "dsp"


def signal(rate: int, seconds: float = 1.0) -> np.ndarray:
    """Two tones, a chirp and seeded noise, within [-1, 1]."""
    t = np.arange(int(rate * seconds)) / rate
    rng = np.random.default_rng(0)
    x = 0.3 * np.sin(2 * np.pi * 220 * t) + 0.2 * np.sin(2 * np.pi * 1870 * t) + 0.1 * np.sin(2 * np.pi * (100 + 3000 * t) * t) + 0.05 * rng.standard_normal(t.shape)
    return x.astype(np.float32)


# (name, sr, n_fft, n_mels, fmin, fmax, htk, norm): the configurations engines use
LIBROSA_MELS: list[tuple[Any, ...]] = [
    ("qwen3_speaker", 24000, 1024, 128, 0.0, 12000.0, False, "slaney"),
    ("cosyvoice_v1", 22050, 1024, 80, 0.0, 8000.0, False, "slaney"),
    ("cosyvoice_v2", 24000, 1920, 80, 0.0, 8000.0, False, "slaney"),
    ("whisper80", 16000, 400, 80, 0.0, None, False, "slaney"),
    ("whisper128", 16000, 400, 128, 0.0, None, False, "slaney"),
    ("gptsovits_v4", 32000, 1280, 100, 0.0, None, False, "slaney"),
    ("bigvgan22k", 22050, 1024, 80, 0.0, None, False, "slaney"),
    ("htk_none", 24000, 1024, 100, 0.0, 12000.0, True, None),
]
# (name, n_freqs, f_min, f_max, n_mels, sample_rate, norm, mel_scale)
TA_FBANKS: list[tuple[Any, ...]] = [
    ("vocos24k", 513, 0.0, 12000.0, 100, 24000, None, "htk"),
    ("slaney_slaney", 513, 0.0, 12000.0, 100, 24000, "slaney", "slaney"),
    ("indextts_v15", 513, 0.0, 12000.0, 100, 24000, None, "htk"),
]
# (name, sample_rate, n_fft, win_length, hop_length, n_mels, power, center, norm, mel_scale, f_min, f_max)
TA_MELSPECS: list[tuple[Any, ...]] = [
    ("f5_vocos", 24000, 1024, 1024, 256, 100, 1.0, True, None, "htk", 0.0, None),
    ("power2_slaney", 16000, 512, 400, 160, 80, 2.0, True, "slaney", "slaney", 20.0, 7600.0),
]
# (name, orig, new, method, kwargs)
RESAMPLES: list[tuple[Any, ...]] = [
    ("24k_16k", 24000, 16000, "sinc_interp_hann", {}),
    ("44k_24k", 44100, 24000, "sinc_interp_hann", {}),
    ("16k_22k", 16000, 22050, "sinc_interp_hann", {}),
    ("24k_16k_kaiser", 24000, 16000, "sinc_interp_kaiser", {"lowpass_filter_width": 16, "rolloff": 0.85, "beta": 8.555}),
]
SOXR: list[tuple[Any, ...]] = [("44k_24k", 44100, 24000), ("24k_16k", 24000, 16000), ("16k_24k", 16000, 24000)]
# kaldi fbank keyword sets: CAM++ (CosyVoice frontend, IndexTTS), defaults
KALDI: list[tuple[Any, ...]] = [
    ("campplus", 16000, {"num_mel_bins": 80, "dither": 0.0, "sample_frequency": 16000}),
    ("defaults_nodither", 16000, {"dither": 0.0}),
]
MATCHA: list[tuple[Any, ...]] = [
    # n_fft, num_mels, sampling_rate, hop_size, win_size, fmin, fmax
    ("cosyvoice_v2", 1920, 80, 24000, 480, 1920, 0, 8000),
    ("cosyvoice_v1", 1024, 80, 22050, 256, 1024, 0, 8000),
]


def group_torchaudio(out: dict, meta: dict) -> None:
    import torch
    import torchaudio  # ty: ignore[unresolved-import]

    meta["torch"], meta["torchaudio"] = torch.__version__, torchaudio.__version__
    for name, n_freqs, f_min, f_max, n_mels, sr, norm, scale in TA_FBANKS:
        out[f"fbank/{name}"] = torchaudio.functional.melscale_fbanks(n_freqs, f_min, f_max, n_mels, sr, norm, scale).numpy()
    for name, sr, n_fft, win, hop, n_mels, power, center, norm, scale, f_min, f_max in TA_MELSPECS:
        x = torch.from_numpy(signal(sr, 0.5))
        t = torchaudio.transforms.MelSpectrogram(sr, n_fft, win, hop, f_min, f_max, 0, n_mels, power=power, center=center, norm=norm, mel_scale=scale)
        out[f"melspec/{name}"] = t(x).numpy()
    for name, orig, new, method, kw in RESAMPLES:
        x = torch.from_numpy(signal(orig, 0.25))
        out[f"resample/{name}"] = torchaudio.functional.resample(x, orig, new, resampling_method=method, **kw).numpy()
        out[f"resample64/{name}"] = torchaudio.functional.resample(x.double(), orig, new, resampling_method=method, **kw).numpy()
        out[f"transform/{name}"] = torchaudio.transforms.Resample(orig, new, method, **kw)(x).numpy()
    for name, sr, kw in KALDI:
        x = torch.from_numpy(signal(sr, 0.5)).unsqueeze(0)
        out[f"kaldi/{name}"] = torchaudio.compliance.kaldi.fbank(x, **kw).numpy()


def group_librosa(out: dict, meta: dict) -> None:
    import librosa  # ty: ignore[unresolved-import]

    meta["librosa"] = librosa.__version__
    for name, sr, n_fft, n_mels, fmin, fmax, htk, norm in LIBROSA_MELS:
        out[f"librosa_mel/{name}"] = librosa.filters.mel(sr=sr, n_fft=n_fft, n_mels=n_mels, fmin=fmin, fmax=fmax, htk=htk, norm=norm)


def group_soxr(out: dict, meta: dict) -> None:
    import librosa  # ty: ignore[unresolved-import]
    import soxr

    meta["soxr"] = soxr.__version__
    for name, orig, new in SOXR:
        out[f"soxr/{name}"] = librosa.resample(signal(orig, 0.25), orig_sr=orig, target_sr=new, res_type="soxr_hq")


def group_whisper(out: dict, meta: dict) -> None:
    import torch
    import whisper  # ty: ignore[unresolved-import]
    from whisper.audio import mel_filters  # ty: ignore[unresolved-import]

    meta["torch"], meta["whisper"] = torch.__version__, getattr(whisper, "__version__", "?")
    x = torch.from_numpy(signal(16000, 0.5))
    for n in (80, 128):
        out[f"whisper/filters{n}"] = mel_filters("cpu", n).numpy()
        out[f"whisper/logmel{n}"] = whisper.log_mel_spectrogram(x, n_mels=n).numpy()


def group_hf_whisper(out: dict, meta: dict) -> None:
    """transformers' ``WhisperFeatureExtractor`` (the ASR engine's features): filters, and features of audio padded and cut to 30 s."""
    import torch
    import transformers
    from transformers import WhisperFeatureExtractor

    meta["torch"], meta["transformers"] = torch.__version__, transformers.__version__
    for n in (80, 128):
        fe = WhisperFeatureExtractor(feature_size=n)
        out[f"hf_whisper/filters{n}"] = fe.mel_filters
        out[f"hf_whisper/features{n}"] = fe(signal(16000, 0.5), sampling_rate=16000, return_tensors="np").input_features
    long = np.concatenate([signal(16000, 1.0)] * 31)
    out["hf_whisper/features128_cut"] = WhisperFeatureExtractor(feature_size=128)(long, sampling_rate=16000, return_tensors="np").input_features[:, :, -200:]


def group_matcha(out: dict, meta: dict) -> None:
    import torch

    sys.path.insert(
        0,
        str(
            next(Path.home().glob("code_workspace/experiment/kodama-refactor/reference/packages/engines/cosyvoice/kodama_engine_cosyvoice/vendor/CosyVoice/third_party/Matcha-TTS"))
        ),
    )
    import matcha.utils.audio as matcha_audio  # ty: ignore[unresolved-import]
    from matcha.utils.audio import mel_spectrogram  # ty: ignore[unresolved-import]

    meta["torch"] = torch.__version__
    for name, n_fft, n_mels, sr, hop, win, fmin, fmax in MATCHA:
        # Matcha caches the basis and the window by fmax and device only: a second configuration in one
        # process would reuse the first one's window. Each configuration starts as in a fresh process.
        matcha_audio.mel_basis.clear()
        matcha_audio.hann_window.clear()
        x = torch.from_numpy(signal(sr, 0.5)).unsqueeze(0)
        out[f"matcha/{name}"] = mel_spectrogram(x, n_fft, n_mels, sr, hop, win, fmin, fmax, center=False).numpy()


FFMPEG_RESAMPLES = [(8000, 32000), (11025, 32000), (16000, 32000), (22050, 32000), (24000, 32000), (44100, 32000), (48000, 32000), (32000, 16000), (44100, 24000)]
"""(input rate, output rate): ``ffmpeg -i <16-bit WAV> -f f32le -ac 1 -ar <rate> -`` as GPT-SoVITS's training reads clips."""


def pcm16(rate: int, seconds: float = 0.12) -> np.ndarray:
    """``signal`` as a 16-bit clip holds it, back in float (x / 32768)."""
    return np.round(signal(rate, seconds) * 32767).astype(np.int16).astype(np.float32) / np.float32(32768)


def group_ffmpeg(out: dict, meta: dict) -> None:
    import os
    import subprocess
    import tempfile
    import wave

    binary = os.environ.get("FFMPEG", "ffmpeg")
    meta["ffmpeg"] = subprocess.run([binary, "-hide_banner", "-version"], capture_output=True, text=True, check=True).stdout.splitlines()[0]
    with tempfile.TemporaryDirectory() as tmp:
        for orig, new in FFMPEG_RESAMPLES:
            path = Path(tmp) / f"{orig}.wav"
            with wave.open(str(path), "wb") as w:
                w.setnchannels(1)
                w.setsampwidth(2)
                w.setframerate(orig)
                w.writeframes((pcm16(orig) * 32768).astype("<i2").tobytes())
            cmd = [binary, "-nostdin", "-threads", "0", "-i", str(path), "-f", "f32le", "-acodec", "pcm_f32le", "-ac", "1", "-ar", str(new), "-"]
            out[f"ffmpeg/{orig}-{new}"] = np.frombuffer(subprocess.run(cmd, capture_output=True, check=True).stdout, np.float32)


GROUPS = {
    "torchaudio": group_torchaudio,
    "librosa": group_librosa,
    "soxr": group_soxr,
    "whisper": group_whisper,
    "hf_whisper": group_hf_whisper,
    "matcha": group_matcha,
    "ffmpeg": group_ffmpeg,
}


def main(argv: list[str]) -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    for group in argv or list(GROUPS):
        out: dict = {}
        meta: dict = {"python": sys.version.split()[0], "numpy": np.__version__}
        GROUPS[group](out, meta)
        np.savez_compressed(OUT / f"{group}.npz", **{k.replace("/", "__"): v for k, v in out.items()}, meta=np.array(json.dumps(meta)))
        print(group, len(out), "arrays", meta, f"{(OUT / f'{group}.npz').stat().st_size / 1024:.0f} KB")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
