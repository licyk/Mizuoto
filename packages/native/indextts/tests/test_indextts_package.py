"""mizuoto-indextts as a distribution of its own: its version and the Mizuoto versions it works with, its engine
against the manifest Mizuoto ships for it, the rules every engine keeps, and its in-place layers to the bit."""

from __future__ import annotations

import math
from pathlib import Path

import pytest

from tests.core.test_engines import check_devices_mirror, check_gated_distribution, check_options_mirror
from tests.test_architecture import check_native_package

PACKAGE = Path(__file__).resolve().parents[1] / "mizuoto_indextts"


def test_version_and_mizuoto_range():
    check_gated_distribution("indextts")


def test_follows_the_engine_rules():
    check_native_package(PACKAGE)


def test_engine_mirrors_its_manifest():
    pytest.importorskip("mizuoto_indextts")
    import mizuoto_indextts
    from mizuoto_indextts.version import VERSION

    assert mizuoto_indextts.VERSION == VERSION
    check_options_mirror("indextts")
    check_devices_mirror("indextts")


def test_gpt2s_in_place_gelu_matches_the_original():
    torch = pytest.importorskip("torch")
    gpt = pytest.importorskip("mizuoto_indextts.gpt")

    def upstream(x):
        return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))

    g = torch.Generator().manual_seed(0)
    for dtype in (torch.float32, torch.float16, torch.bfloat16):
        for shape in ((3, 1, 5120), (1, 50, 5120), (7,)):
            x = (torch.randn(shape, generator=g) * 4).to(dtype)
            assert torch.equal(gpt.gelu_new(x), upstream(x)), (dtype, shape)
