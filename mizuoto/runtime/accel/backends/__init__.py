"""The attention backends: torch's own (always registered) and the optional packages (registered when imported)."""

from __future__ import annotations

import threading

_done = False
_lock = threading.Lock()


def register_all() -> None:
    """Register every backend once per process (the optional ones only probe their package when first used)."""
    global _done
    with _lock:
        if _done:
            return
        from mizuoto.runtime.accel.backends import optional, torch_sdpa

        torch_sdpa.register()
        optional.register()
        _done = True
