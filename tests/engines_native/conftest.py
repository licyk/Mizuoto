from __future__ import annotations

import pytest

from tests.engines_native.accel_checks import check_mask_info, count_sites

SITES_TESTED = {
    "qwen",
    "qwen3tts.mimi",
    "unet1d",
    "bert",
    "seedvc",
    "hubert",
    "dit",
    "cosyvoice.dit",
    "gptsovits.dit",
    "gptsovits.t2s",
    "asr.whisper",
    "voxcpm.minicpm",
}
"""The sites the built-in tiny engines and trainers reach (``MIZUOTO_TEST_ACCEL=count`` checks that each one goes
through the dispatcher). The gated engines list theirs in their packages' conftests."""


@pytest.fixture(autouse=True)
def _check_mask_info():
    yield from check_mask_info()


@pytest.fixture(scope="session", autouse=True)
def _count_sites():
    yield from count_sites(SITES_TESTED)
