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


def test_fish_codecs_in_place_snake_matches_the_original():
    torch = pytest.importorskip("torch")
    codec = pytest.importorskip("mizuoto_fishspeech.codec")

    def upstream(x, alpha):
        shape = x.shape
        x = x.reshape(shape[0], shape[1], -1)
        return (x + (alpha + 1e-9).reciprocal() * torch.sin(alpha * x).pow(2)).reshape(shape)

    g = torch.Generator().manual_seed(0)
    for dtype in (torch.float32, torch.bfloat16):
        for shape in ((1, 64, 3001), (2, 16, 7)):
            x = (torch.randn(shape, generator=g) * 3).to(dtype)
            alpha = (torch.rand(1, shape[1], 1, generator=g) * 2).to(dtype)
            with torch.no_grad():
                assert torch.equal(codec.snake(x, alpha), upstream(x, alpha)), (dtype, shape)
            assert torch.equal(codec.snake(x, alpha), upstream(x, alpha)), (dtype, shape)
