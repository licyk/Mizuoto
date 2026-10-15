"""The process-wide state an engine call must leave as it found it.

Native engines share the host's process. A test, or ``--debug``, takes ``snapshot()`` before and
after each engine call; ``diff`` names what changed. It covers:

- the Python, numpy and torch RNG states;
- the working directory, ``os.environ`` and ``sys.path``;
- the root logger, warning filters and torch's backend flags.

torch and numpy are read only when they are already loaded (``sys.modules``).
"""

from __future__ import annotations

import hashlib
import logging
import os
import pickle
import random
import sys
import warnings
from typing import Any


def _digest(value: Any) -> str:
    try:
        data = pickle.dumps(value)
    except Exception:
        data = repr(value).encode("utf-8", "replace")
    return hashlib.sha256(data).hexdigest()[:16]


def _rng_digests() -> dict[str, str]:
    states: dict[str, Any] = {"python": random.getstate()}
    np = sys.modules.get("numpy")
    if np is not None:
        states["numpy"] = np.random.get_state()
    torch = sys.modules.get("torch")
    if torch is not None:
        states["torch"] = torch.random.get_rng_state().numpy().tobytes()
        try:
            if torch.cuda.is_available() and torch.cuda.is_initialized():
                states["cuda"] = [s.numpy().tobytes() for s in torch.cuda.get_rng_state_all()]
        except Exception:
            pass
    return {k: _digest(v) for k, v in states.items()}


def _torch_flags() -> dict[str, Any]:
    torch = sys.modules.get("torch")
    if torch is None:
        return {}
    flags: dict[str, Any] = {}
    try:
        flags["cudnn.deterministic"] = torch.backends.cudnn.deterministic
        flags["cudnn.benchmark"] = torch.backends.cudnn.benchmark
        flags["matmul.allow_tf32"] = torch.backends.cuda.matmul.allow_tf32
        flags["cudnn.allow_tf32"] = torch.backends.cudnn.allow_tf32
    except Exception:
        pass
    return flags


def snapshot() -> dict[str, Any]:
    """Digests of the process-wide state requests must not change (the contract suite's no-residue check)."""
    root = logging.getLogger()
    return {
        "cwd": os.getcwd(),
        "env": _digest(sorted(os.environ.items())),
        "rng": _rng_digests(),
        "torch_flags": _torch_flags(),
        "root_level": root.level,
        "root_handlers": len(root.handlers),
        "warning_filters": _digest([(f[0], str(f[1]), getattr(f[2], "__name__", str(f[2])), str(f[3]), f[4]) for f in warnings.filters]),
        "sys_path": hashlib.sha256("\n".join(sys.path).encode()).hexdigest()[:16],
    }


def diff(before: dict[str, Any], after: dict[str, Any]) -> list[str]:
    """The keys (``rng.torch``, ``cwd``…) whose values differ."""
    changed: list[str] = []
    for key in sorted(set(before) | set(after)):
        a, b = before.get(key), after.get(key)
        if isinstance(a, dict) and isinstance(b, dict):
            changed += [f"{key}.{k}" for k in sorted(set(a) | set(b)) if a.get(k) != b.get(k)]
        elif a != b:
            changed.append(key)
    return changed
