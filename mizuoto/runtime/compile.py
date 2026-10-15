"""Optional ``torch.compile`` of an engine's heavy, repeatedly called module (``compute.compile``).

An engine wraps the module it calls in a loop (a flow-matching estimator, a vocoder) in ``Compiled`` when
``runtime.options.compile`` is on, and calls ``compiled or module`` otherwise, so the eager path is the code it
always ran. Compilation happens on the first call of each new shape family; with ``dynamic=True`` a new length
does not recompile. When compiling or running the compiled code fails (no C++ compiler, an unsupported op), the
wrapper logs one warning and runs the module eagerly from then on: a request never fails for it.

``Compiled`` is not an ``nn.Module``: held as a plain attribute, it leaves the module tree (state dicts, handles,
device moves) as it was.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from torch import nn

logger = logging.getLogger(__name__)

EAGER_DEVICES = frozenset({"mps"})
"""Device types whose modules always run eagerly: inductor's MPS backend is partial."""


class Compiled:
    def __init__(self, module: nn.Module, name: str, *, dynamic: bool | None = True) -> None:
        import torch

        self.module = module
        self.name = name
        self.failure: str | None = None
        """Why the compiled code is no longer used (the first error), or None."""
        self.calls = 0
        """Calls that ran compiled."""
        self._compiled: Any = torch.compile(module, dynamic=dynamic)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        if self.failure is None and self.calls == 0:
            from mizuoto.runtime.device import module_device

            kind = module_device(self.module).type
            if kind in EAGER_DEVICES:
                self.failure = f"torch.compile is off on {kind.upper()}"
                logger.info("%s: %s, running it eagerly", self.name, self.failure)
        if self.failure is None and not self.module.__dict__.get("_mizuoto_streamed", False):
            # (a module whose units stream runs eagerly: its weights come and go between calls)
            try:
                from mizuoto.runtime.context import accel_policy

                run = accel_policy.get()
                if run is None:
                    out = self._compiled(*args, **kwargs)
                else:  # the graph cannot read the context's attention policy: it traces with this one
                    from mizuoto.runtime.accel.policy import tracing

                    with tracing(run.policy.trace_run):
                        out = self._compiled(*args, **kwargs)
            except Exception as error:  # any compiler failure falls back to eager; running out of memory does not
                from mizuoto.runtime.device import is_oom

                if is_oom(error):
                    raise
                lines = str(error).strip().splitlines()
                self.failure = f"{type(error).__name__}: {lines[0] if lines else ''}"
                logger.warning("torch.compile of %s failed, running it eagerly: %s", self.name, self.failure)
            else:
                self.calls += 1
                return out
        return self.module(*args, **kwargs)


def compile_module(module: nn.Module, enabled: bool, name: str, *, dynamic: bool | None = True) -> Compiled | None:
    """``Compiled(module)`` when ``enabled``, else None (the caller runs ``module`` itself)."""
    return Compiled(module, name, dynamic=dynamic) if enabled else None
