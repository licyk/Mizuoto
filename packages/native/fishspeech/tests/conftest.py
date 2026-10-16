"""The Fish Speech engine's tests run under the same attention checks as the built-in engines' (``tests/engines_native``).
Run them from the repository root, which ``pythonpath`` in its pyproject puts on ``sys.path`` for ``tests.engines_native``."""

from __future__ import annotations

import pytest

from tests.engines_native.accel_checks import check_mask_info, count_sites

SITES_TESTED = {"fishspeech.slow", "fishspeech.step", "fishspeech.fast", "fishspeech.codec"}
"""The attention sites the tiny Fish Speech engine reaches (``MIZUOTO_TEST_ACCEL=count``)."""


@pytest.fixture(autouse=True)
def _check_mask_info():
    yield from check_mask_info()


@pytest.fixture(scope="session", autouse=True)
def _count_sites():
    yield from count_sites(SITES_TESTED)
