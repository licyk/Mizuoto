import os
import sys

import numpy as np
import pytest

from mizuoto.core.audio import dsp
from mizuoto.core.audio.io import FORMATS, decode, encode, probe, resample
from mizuoto.core.errors import ValidationError
from mizuoto.core.process import WorkerTarget, describe_exit
from mizuoto.core.text.segment import segment

SR = 24000
TONE = (0.5 * np.sin(2 * np.pi * 440 * np.arange(SR) / SR)).astype(np.float32)


@pytest.mark.parametrize("fmt", sorted(FORMATS))
def test_encode_and_decode_every_format(tmp_path, fmt):
    path = encode(TONE, SR, tmp_path / f"x.{fmt}", fmt)
    info = probe(path)
    assert info.sample_rate == (48000 if fmt == "ogg" else SR)
    samples, rate = decode(path, 16000)
    assert rate == 16000 and abs(len(samples) - 16000) < 1600
    assert 0.3 < float(np.max(np.abs(samples))) < 0.7
    assert not list(tmp_path.glob("*.part"))


def test_not_audio(tmp_path):
    bad = tmp_path / "bad.wav"
    bad.write_text("not audio")
    with pytest.raises(ValidationError):
        probe(bad)
    with pytest.raises(ValidationError):
        encode(TONE, SR, tmp_path / "x.aiff", "aiff")


def test_resample_keeps_the_pitch():
    out = resample(TONE, SR, 16000)
    assert abs(len(out) - 16000) < 32
    spectrum = np.abs(np.fft.rfft(out[:16000]))
    assert abs(int(np.argmax(spectrum)) - 440) <= 2


def test_assemble_normalize_peaks():
    a, b = np.ones(100, np.float32) * 0.5, np.ones(100, np.float32) * -0.5
    assert len(dsp.assemble([a, b], 1000, silence_ms=50)) == 250
    assert len(dsp.assemble([a, b], 1000, crossfade_ms=10)) == 190
    assert len(dsp.assemble([], 1000)) == 0
    x = dsp.normalize(np.array([0.1, -0.2, np.nan, 5.0], np.float32), "peak")
    assert np.isfinite(x).all() and float(np.max(np.abs(x))) == pytest.approx(dsp.PEAK_TARGET)
    quiet = TONE * 0.01
    assert float(np.sqrt(np.mean(dsp.normalize(quiet, "loudness") ** 2))) > float(np.sqrt(np.mean(quiet**2)))
    assert len(dsp.peaks(TONE, 10)) == 20


def test_segment():
    assert segment("") == []
    assert segment("今天天气不错。我们出去走走吧！", 8) == ["今天天气不错。", "我们出去走走吧！"]
    assert segment("Dr. Smith arrived. He sat down.", 20) == ["Dr. Smith arrived.", "He sat down."]
    pieces = segment("A [laughter] tag stays whole, even here. " * 10, 30)
    assert all(p.count("[") == p.count("]") for p in pieces)
    assert all(len(p) <= 40 for p in segment("word " * 100, 40))
    assert segment("short one. short two.", 100) == ["short one. short two."]


def test_worker_target_command():
    target = WorkerTarget(python=sys.executable, module="m")  # ty: ignore[invalid-argument-type]
    assert target.command("a") == [sys.executable, "-m", "m", "a"]


def test_describe_exit():
    assert describe_exit(1) == "exit code 1"
    assert describe_exit(None) == "no exit code"
    if os.name != "nt":
        assert "SIGKILL" in describe_exit(-9)


def test_trim_cuts_at_a_quiet_point():
    import numpy as np

    from mizuoto.core.audio.dsp import trim_to

    sr = 1000
    x = np.ones(sr * 10, dtype=np.float32)
    x[8500:8600] = 0.0  # a pause 1.4 s before the 10 s limit... of a 12 s clip
    clip = np.concatenate([x, np.ones(sr * 2, dtype=np.float32)])
    out = trim_to(clip, sr, 10.0)
    assert 8500 <= len(out) <= 8600
    assert len(trim_to(clip, sr, 20.0)) == len(clip)


def test_the_analysis_finds_a_tones_pitch_and_draws_its_band():
    """YIN finds a voiced tone's pitch (and none in silence); the spectrogram's loudest band holds the tone, 0–255 over
    80 dB, its filters Slaney's as librosa's (checked against librosa 0.10: within 4e-9, and identical bytes)."""
    import numpy as np

    from mizuoto.core.audio.analysis import median_pitch, mel_filters, pitch_curve, spectrogram

    sr = 16000
    t = np.arange(sr) / sr
    for hz in (80.0, 220.0, 440.0):
        tone = (0.5 * np.sin(2 * np.pi * hz * t) + 0.2 * np.sin(4 * np.pi * hz * t)).astype(np.float32)
        f0 = pitch_curve(np.concatenate([np.zeros(sr // 2, dtype=np.float32), tone]))
        assert len(f0) == 151 and not f0[:45].any() and (f0[60:145] > 0).all()
        median = median_pitch(f0)
        assert median is not None and abs(median - hz) < hz * 0.005
    assert median_pitch(np.zeros(10)) is None and pitch_curve(np.zeros(10, dtype=np.float32)).size == 0
    spec, fmax = spectrogram((0.5 * np.sin(2 * np.pi * 1000 * t)).astype(np.float32), sr, columns=50)
    assert spec.dtype == np.uint8 and spec.shape[0] == 128 and 50 <= spec.shape[1] <= 52 and fmax == 8000
    band = int(spec[:, 10:40].mean(axis=1).argmax())
    filters = mel_filters(sr, 1024, 128)
    centre = float((filters[band] * np.linspace(0, sr / 2, 513)).sum() / filters[band].sum())
    assert abs(centre - 1000) < 60 and spec.max() == 255
    assert filters.shape == (128, 513) and (filters >= 0).all()
