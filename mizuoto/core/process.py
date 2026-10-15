"""Starting and stopping job workers: child processes of this interpreter (training stages).

A job worker is ``python -m <module> <args…>`` (``WorkerTarget``) with the host's own environment, so the child
imports the same Mizuoto; ``kill_process_tree`` stops it and whatever it started.
"""

from __future__ import annotations

import logging
import os
import signal
import subprocess
import sys
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class WorkerTarget:
    """A Python program to run as a job worker: which interpreter, which module, where, with what added environment."""

    python: Path
    module: str
    cwd: Path | None = None
    env: Mapping[str, str] = field(default_factory=dict)
    """Added to the host's environment."""

    def command(self, *args: str) -> list[str]:
        return [str(self.python), "-m", self.module, *args]


def host_target(module: str, cwd: Path | None = None, env: Mapping[str, str] | None = None) -> WorkerTarget:
    """A job worker in the host's own interpreter."""
    return WorkerTarget(Path(sys.executable), module, cwd, dict(env or {}))


def start_process(cmd: Sequence[str], env: Mapping[str, str] | None = None, cwd: Path | None = None) -> subprocess.Popen[str]:
    """Start ``cmd`` in its own process group, with stdout piped as text and stderr merged into it; ``env`` is added
    to the host's environment."""
    kwargs: dict = {}
    if os.name == "nt":
        kwargs["creationflags"] = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
    else:
        kwargs["start_new_session"] = True
    full_env = {**os.environ, **(env or {}), "PYTHONUNBUFFERED": "1", "PYTHONIOENCODING": "utf-8"}
    return subprocess.Popen(
        list(cmd),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        stdin=subprocess.DEVNULL,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
        env=full_env,
        cwd=str(cwd) if cwd else None,
        **kwargs,
    )


def start_target(target: WorkerTarget, *args: str) -> subprocess.Popen[str]:
    """Start a ``WorkerTarget`` with ``args`` after its module."""
    if target.cwd is not None:
        target.cwd.mkdir(parents=True, exist_ok=True)
    return start_process(target.command(*args), env=target.env, cwd=target.cwd)


def kill_process_tree(process: subprocess.Popen | None) -> bool:
    """Terminate a process and every child it created (from the original ``tools/process_utils.py``)."""
    if process is None or process.poll() is not None:
        return False
    pid = process.pid
    if os.name == "nt":
        try:
            subprocess.run(["taskkill", "/t", "/f", "/pid", str(pid)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
        except OSError:
            try:
                process.terminate()
            except OSError:
                pass
    else:
        try:
            group = os.getpgid(pid)
        except OSError:
            group = None
        own = group is not None and group != os.getpgrp()
        try:
            os.killpg(group, signal.SIGTERM) if own and group is not None else os.kill(pid, signal.SIGTERM)
        except OSError:
            pass
        for _ in range(20):
            if process.poll() is not None:
                break
            time.sleep(0.1)
        if process.poll() is None:
            try:
                os.killpg(group, signal.SIGKILL) if own and group is not None else os.kill(pid, signal.SIGKILL)
            except OSError:
                pass
    try:
        process.wait(timeout=5)
    except (OSError, subprocess.TimeoutExpired):
        try:
            process.kill()
        except OSError:
            pass
    logger.debug("Process tree %s terminated", pid)
    return True


WINDOWS_STATUS = {
    0xC0000005: "an access violation",
    0xC000001D: "an illegal instruction",
    0xC0000094: "an integer division by zero",
    0xC00000FD: "a stack overflow",
    0xC0000374: "heap corruption",
    0xC0000409: "a stack buffer overrun",
    0xC000013A: "Ctrl+C",
    0x40010004: "being terminated",
}
"""NTSTATUS codes a crashed Windows process exits with."""


def describe_exit(code: int | None) -> str:
    """A process's exit code in words: ``exit code 1``, ``killed by SIGSEGV (signal 11)``, or a
    Windows crash status such as ``0xC0000005 (an access violation)``."""
    if code is None:
        return "no exit code"
    if code < 0 and os.name != "nt":
        try:
            name = signal.Signals(-code).name
        except ValueError:
            name = "a signal"
        return f"killed by {name} (signal {-code})"
    status = code & 0xFFFFFFFF
    if status in WINDOWS_STATUS:
        return f"exit code 0x{status:08X}, {WINDOWS_STATUS[status]}"
    return f"exit code {code}"
