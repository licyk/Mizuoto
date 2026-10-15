from __future__ import annotations

import os
from collections import Counter

import pytest

COUNTED: Counter[str] = Counter()

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
    "fishspeech.slow",
    "fishspeech.step",
    "fishspeech.fast",
    "fishspeech.codec",
    "indextts.gpt",
    "indextts.perceiver",
}
"""The sites the tiny engines and trainers reach (``MIZUOTO_TEST_ACCEL=count`` checks that each one goes through the
dispatcher)."""


@pytest.fixture(autouse=True)
def _check_mask_info():
    """Every ``mask_info`` a site passes must describe its dense mask."""
    try:
        from mizuoto.runtime.accel import dispatch
    except ImportError:  # no torch: the tests skip
        yield
        return
    dispatch.DEBUG = True
    try:
        yield
    finally:
        dispatch.DEBUG = False


@pytest.fixture(scope="session", autouse=True)
def _count_sites():
    """``MIZUOTO_TEST_ACCEL=count``: every attention call of the tiny engines goes through a policy that counts its site
    and computes the site's reference (so every test's expectations hold); at the end each site must have been seen."""
    if os.environ.get("MIZUOTO_TEST_ACCEL") != "count":
        yield
        return
    from mizuoto.runtime.accel import dispatch
    from mizuoto.runtime.accel.policy import AccelPolicy, Run

    class Counting(Run):
        transparent = True

        def attend(self, site, q, k, v, mask, causal, scale, enable_gqa, dropout_p, mask_info):
            COUNTED[site.name] += 1
            return site.reference(q, k, v, mask, causal, scale, enable_gqa, dropout_p)

    dispatch.PROCESS_POLICY[0] = Counting(AccelPolicy("sdpa"))
    try:
        yield
    finally:
        dispatch.PROCESS_POLICY[0] = None
    missing = SITES_TESTED - set(COUNTED)
    assert not missing, f"attention sites the tiny engines never reached through mizuoto.runtime.accel: {sorted(missing)}"
