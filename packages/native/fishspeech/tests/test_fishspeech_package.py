"""mizuoto-fishspeech as a distribution of its own: its version and the Mizuoto versions it works with, its engine
against the manifest Mizuoto ships for it, the rules every engine keeps, and its in-place layers to the bit."""

from __future__ import annotations

from pathlib import Path

import pytest

from tests.core.test_engines import check_devices_mirror, check_gated_distribution, check_options_mirror
from tests.test_architecture import check_native_package

PACKAGE = Path(__file__).resolve().parents[1] / "mizuoto_fishspeech"


def test_version_and_mizuoto_range():
    check_gated_distribution("fishspeech")


def test_follows_the_engine_rules():
    check_native_package(PACKAGE)


def test_engine_mirrors_its_manifest():
    pytest.importorskip("mizuoto_fishspeech")
    import mizuoto_fishspeech
    from mizuoto_fishspeech.version import VERSION

    assert mizuoto_fishspeech.VERSION == VERSION
    check_options_mirror("fishspeech")
    check_devices_mirror("fishspeech")


SNAKE = """
def snake(x, alpha):
    shape = x.shape
    x = x.reshape(shape[0], shape[1], -1)
    x = x + (alpha + 1e-9).reciprocal() * torch.sin(alpha * x).pow(2)
    x = x.reshape(shape)
    return x
"""
"""descript-audio-codec's ``snake`` (``dac/nn/layers.py``), which upstream's codec runs as a TorchScript function."""


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_fish_codecs_snake_matches_the_original(device):
    """The codec's Snake gives upstream's bits: its eager operations (in place) without autocast, and under autocast what
    the TorchScript function computes (float32 throughout; eager autocast would keep ``alpha · x`` and the sine in the
    low precision). The script runs unfused: TorchScript's fuser, from a shape's second call on CUDA, differs from
    either by rounding, and by what the process has run before."""
    torch = pytest.importorskip("torch")
    codec = pytest.importorskip("mizuoto_fishspeech.codec")
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("no CUDA device")
    scripted = torch.jit.CompilationUnit(SNAKE).snake
    namespace: dict = {"torch": torch}
    exec(SNAKE, namespace)  # the same function run eagerly
    eager = namespace["snake"]

    g = torch.Generator().manual_seed(0)
    for dtype in (torch.float32, torch.bfloat16, torch.float16):
        for shape in ((1, 64, 3001), (2, 16, 7)):
            x = (torch.randn(shape, generator=g) * 3).to(device, dtype)
            alpha = (torch.rand(1, shape[1], 1, generator=g) * 2).to(device, dtype)
            with torch.no_grad():
                assert torch.equal(codec.snake(x, alpha), eager(x, alpha)), (dtype, shape)
            assert torch.equal(codec.snake(x, alpha), eager(x, alpha)), (dtype, shape)
            if dtype != torch.float32:  # the codec on a GPU decodes under autocast
                with torch.no_grad(), torch.jit.fuser("none"), torch.autocast(device, dtype=dtype):
                    ours, theirs = codec.snake(x, alpha), [scripted(x, alpha) for _ in range(3)][-1]
                assert ours.dtype == theirs.dtype == torch.float32 and torch.equal(ours, theirs), (dtype, shape, "autocast")
