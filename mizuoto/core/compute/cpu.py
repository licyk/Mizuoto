"""How the host process and its job workers use the CPU (``compute.cpu_allocator``, ``compute.cpu_threads``). Standard
library only.

Both change where memory and threads live, never what is computed: the results are the same to the bit, only faster
. Both must be in place before torch is imported, so the host applies them when it builds its
runtime (``apply``) and hands them to the processes it starts (``child_env``: training stages, job workers).

**Allocator.** torch takes every CPU tensor from the C library's ``malloc``. glibc's defaults suit small, long-lived
objects: blocks above a threshold that grows as large blocks are freed (128 KB at first) come from ``mmap``, and free
memory at the top of the heap goes back to the system once 128 KB of it piles up. A model's forward allocates and frees
tensors of a few to tens of megabytes at every layer, so each is mapped fresh and the kernel zeroes its pages on first
touch. ``tuned`` (``auto`` on Linux with glibc) fixes the ``mmap`` threshold at 32 MB (glibc's largest) and keeps up to
512 MB of free heap for the next allocation (``mallopt`` here, ``GLIBC_TUNABLES`` for a new process), and has torch put
large tensors on transparent huge pages (``THP_MEM_ALLOC_ENABLE``). F5-TTS runs 10–20 % faster on the CPU; the process
may hold up to 512 MB more than it uses. (Raising only the trim threshold made things slower.) Nothing changes off
glibc (macOS, Windows, musl). Training stages under ``training.memory = lowest`` get the ``malloc`` settings but not
the huge pages (``child_env(huge_pages=False)``): AdamW stepping in RAM allocates gigabytes at every step, and with the
page cache filling the memory each huge page stalls in the kernel's compaction (``defrag=madvise`` compacts
synchronously): Qwen3-TTS 1.7B's fine-tune ran 2.2× slower with them (8× with the optimizer's buffers held across
steps). Other stages keep them: fine-tunes on a GPU ran 5–15 % faster with them.

**Threads.** torch runs its CPU kernels on an OpenMP team of one thread per physical core. Left to the scheduler, two
of them sometimes share a core (its two hardware threads) while another core idles; that process then runs a third
slower for as long as it lives (OmniVoice: 1 process in 6). ``cores`` (``auto`` on Linux) binds the team's threads to
distinct cores (``OMP_PLACES=cores``, ``OMP_PROC_BIND=close``). Another busy program on those cores slows a bound
thread down instead of letting it move: ``off`` leaves placement to the scheduler.

An environment that sets any of these variables itself is left alone.
"""

from __future__ import annotations

import ctypes
import logging
import os
import platform
import sys
import threading

logger = logging.getLogger(__name__)

MMAP_THRESHOLD = 32 * 2**20
TRIM_THRESHOLD = 512 * 2**20
_M_TRIM_THRESHOLD = -1  # mallopt's parameter numbers (malloc.h)
_M_MMAP_THRESHOLD = -3
BINDING = {"OMP_PLACES": "cores", "OMP_PROC_BIND": "close"}
_BINDING_VARIABLES = ("OMP_PLACES", "OMP_PROC_BIND", "GOMP_CPU_AFFINITY", "KMP_AFFINITY")

_lock = threading.Lock()
_mallopt_done = False
_applied: set[str] = set()
"""The variables ``apply`` put in this process's environment (a child may be given others for them)."""


def is_glibc() -> bool:
    return sys.platform.startswith("linux") and platform.libc_ver()[0] == "glibc"


def tunables(existing: str | None) -> str | None:
    """``GLIBC_TUNABLES`` with the tuned malloc settings added to ``existing``; None when ``existing`` sets malloc's own."""
    if existing and "glibc.malloc." in existing:
        return None
    ours = f"glibc.malloc.mmap_threshold={MMAP_THRESHOLD}:glibc.malloc.trim_threshold={TRIM_THRESHOLD}"
    return f"{existing}:{ours}" if existing else ours


def child_env(allocator: str, threads: str, environ: dict[str, str] | None = None, huge_pages: bool = True) -> dict[str, str]:
    """The variables a new process needs for these settings (what ``environ``, default this process's, lacks).
    ``huge_pages=False`` (a ``lowest`` training stage) turns torch's huge pages off unless the environment's own setting
    asks for them."""
    env = os.environ if environ is None else environ
    out: dict[str, str] = {}
    if allocator != "system" and is_glibc():
        value = tunables(env.get("GLIBC_TUNABLES"))
        if value is not None:
            out["GLIBC_TUNABLES"] = value
        ours = "THP_MEM_ALLOC_ENABLE" not in env or (environ is None and "THP_MEM_ALLOC_ENABLE" in _applied)
        want = "1" if huge_pages else "0"
        if ours and env.get("THP_MEM_ALLOC_ENABLE") != want:
            out["THP_MEM_ALLOC_ENABLE"] = want
    if threads != "off" and sys.platform.startswith("linux") and not any(v in env for v in _BINDING_VARIABLES):
        out.update(BINDING)
    return out


def apply(allocator: str, threads: str) -> dict[str, str]:
    """Apply the settings to this process and the processes it starts from now on: ``mallopt`` now, the variables
    torch and OpenMP read when torch is imported (so call this first), and the same in the environment children
    inherit (once torch is imported, only the children). Returns the variables it set."""
    global _mallopt_done
    with _lock:
        if allocator != "system" and is_glibc() and not _mallopt_done and tunables(os.environ.get("GLIBC_TUNABLES")) is not None:
            try:
                libc = ctypes.CDLL("libc.so.6")
                _mallopt_done = libc.mallopt(_M_MMAP_THRESHOLD, MMAP_THRESHOLD) == 1 and libc.mallopt(_M_TRIM_THRESHOLD, TRIM_THRESHOLD) == 1
            except (OSError, AttributeError) as e:
                logger.debug("mallopt unavailable: %s", e)
        env = child_env(allocator, threads)
        if env and "torch" in sys.modules:
            logger.info("compute.cpu_allocator and compute.cpu_threads reach this process at its next start: torch is already loaded")
        os.environ.update(env)
        _applied.update(env)
        return env
