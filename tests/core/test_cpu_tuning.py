"""compute.cpu_allocator and compute.cpu_threads: the host's and its job workers' CPU settings (mizuoto.core.compute.cpu)."""

from __future__ import annotations

import os
import subprocess
import sys

import pytest

from mizuoto.core.compute import cpu

OURS = f"glibc.malloc.mmap_threshold={cpu.MMAP_THRESHOLD}:glibc.malloc.trim_threshold={cpu.TRIM_THRESHOLD}"
VARIABLES = ("GLIBC_TUNABLES", "THP_MEM_ALLOC_ENABLE", "OMP_PLACES", "OMP_PROC_BIND", "GOMP_CPU_AFFINITY", "KMP_AFFINITY")


@pytest.fixture
def clean(monkeypatch):
    """No CPU variables in the environment (restored after the test), mallopt not yet called."""
    for name in VARIABLES:
        monkeypatch.setenv(name, "x")
        monkeypatch.delenv(name)
    monkeypatch.setattr(cpu, "_mallopt_done", False)
    monkeypatch.setattr(cpu, "_applied", set())
    return monkeypatch


def test_tunables_are_added_to_the_environments_and_never_override_its_own():
    assert cpu.tunables(None) == OURS
    assert cpu.tunables("glibc.pthread.rseq=0") == f"glibc.pthread.rseq=0:{OURS}"
    assert cpu.tunables("glibc.malloc.arena_max=2") is None


@pytest.mark.skipif(not cpu.is_glibc(), reason="Linux with glibc")
def test_auto_on_linux_tunes_the_allocator_and_binds_the_threads(clean):
    assert cpu.child_env("auto", "auto", {}) == {"GLIBC_TUNABLES": OURS, "THP_MEM_ALLOC_ENABLE": "1", "OMP_PLACES": "cores", "OMP_PROC_BIND": "close"}
    assert cpu.child_env("system", "off", {}) == {}
    assert cpu.child_env("tuned", "off", {}) == {"GLIBC_TUNABLES": OURS, "THP_MEM_ALLOC_ENABLE": "1"}
    assert cpu.child_env("system", "cores", {}) == {"OMP_PLACES": "cores", "OMP_PROC_BIND": "close"}


@pytest.mark.skipif(not cpu.is_glibc(), reason="Linux with glibc")
def test_a_lowest_training_stage_gets_the_allocator_without_huge_pages(clean):
    """``huge_pages=False`` (a ``training.memory = lowest`` stage) turns torch's huge pages off, also when this process
    applied them (and so passes them on), but not when the user's environment asks for them."""
    assert cpu.child_env("auto", "auto", {}, huge_pages=False)["THP_MEM_ALLOC_ENABLE"] == "0"
    cpu.apply("auto", "auto")
    assert os.environ["THP_MEM_ALLOC_ENABLE"] == "1"
    stage = cpu.child_env("auto", "auto", huge_pages=False)
    assert stage == {"THP_MEM_ALLOC_ENABLE": "0"}
    assert cpu.child_env("auto", "auto") == {}  # an inference worker inherits this process's
    assert "THP_MEM_ALLOC_ENABLE" not in cpu.child_env("auto", "auto", {"THP_MEM_ALLOC_ENABLE": "1"}, huge_pages=False)


@pytest.mark.skipif(not cpu.is_glibc(), reason="Linux with glibc")
def test_an_environment_that_sets_its_own_is_left_alone():
    env = {"GLIBC_TUNABLES": "glibc.malloc.trim_threshold=1", "THP_MEM_ALLOC_ENABLE": "0", "GOMP_CPU_AFFINITY": "0-3"}
    assert cpu.child_env("auto", "auto", env) == {}


def test_other_platforms_change_nothing(clean):
    clean.setattr(cpu, "is_glibc", lambda: False)
    clean.setattr(cpu.sys, "platform", "darwin")
    assert cpu.child_env("auto", "auto", {}) == {}
    assert cpu.apply("auto", "auto") == {}
    assert cpu._mallopt_done is False


@pytest.mark.skipif(not cpu.is_glibc(), reason="Linux with glibc")
def test_apply_sets_this_process_and_what_its_children_inherit(clean):
    env = cpu.apply("auto", "auto")
    assert env["GLIBC_TUNABLES"] == OURS and cpu._mallopt_done
    assert all(os.environ[k] == v for k, v in env.items())
    out = subprocess.run([sys.executable, "-c", "import os; print(os.environ['OMP_PLACES'], os.environ['GLIBC_TUNABLES'])"], capture_output=True, text=True, check=True)
    assert out.stdout.split() == ["cores", OURS]
    assert cpu.apply("auto", "auto") == {}  # the environment carries them now
