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
