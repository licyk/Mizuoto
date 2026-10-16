"""The attention checks every tiny engine's tests run under, shared by ``tests/engines_native`` and the gated packages'
own tests (``packages/native/*/tests``), whose conftests wrap these generators in fixtures.

- ``mask_info``: every ``mask_info`` a site passes must describe its dense mask (``dispatch.DEBUG``).
- ``MIZUOTO_TEST_ACCEL=count``: every attention call goes through a policy that counts its site and computes the
  site's reference (so every test's expectations hold); at the end each conftest's sites must have been seen.
"""

from __future__ import annotations

import os
from collections import Counter
from collections.abc import Iterator

COUNTED: Counter[str] = Counter()
"""The sites reached in this session, across every directory's tests."""


def check_mask_info() -> Iterator[None]:
    """Turn on the dispatcher's ``mask_info`` check for one test."""
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


def count_sites(sites: set[str]) -> Iterator[None]:
    """For the session, under ``MIZUOTO_TEST_ACCEL=count``: count every site, and check that each of ``sites`` was
    reached. The first caller installs the counting policy and the last one to finish removes it, so the tests of
    every directory are counted however pytest orders them."""
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

    installed = dispatch.PROCESS_POLICY[0] is None
    if installed:
        dispatch.PROCESS_POLICY[0] = Counting(AccelPolicy("sdpa"))
    try:
        yield
    finally:
        if installed:
            dispatch.PROCESS_POLICY[0] = None
    missing = sites - set(COUNTED)
    assert not missing, f"attention sites the tiny engines never reached through mizuoto.runtime.accel: {sorted(missing)}"
