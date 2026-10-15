from __future__ import annotations

from pathlib import Path

import pytest


@pytest.fixture
def make_services(tmp_path: Path):
    """Build services in a temporary data directory with an empty environment. The dummy engine runs in this
    process, as every engine does."""
    from mizuoto.core.context import build_services

    built = []

    def make(start_background: bool = False, **kwargs):
        env = kwargs.pop("env", {})
        s = build_services(data_dir=tmp_path / "data", environ=dict(env), start_background=start_background, **kwargs)
        built.append(s)
        return s

    yield make
    for s in built:
        s.close()


@pytest.fixture
def services(make_services):
    return make_services()


@pytest.fixture
def wav_file(tmp_path: Path):
    """A 2 s, 22.05 kHz stereo WAV of two tones."""
    import numpy as np

    from mizuoto.core.audio.io import encode

    sr = 22050
    t = np.arange(sr * 2) / sr
    x = (0.3 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)
    path = tmp_path / "src" / "voice.wav"
    return encode(x, sr, path, "wav")
