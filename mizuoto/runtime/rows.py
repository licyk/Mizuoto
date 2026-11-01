"""Weights that stream by the row: a stack of experts of which each call reads a few.

A mixture-of-experts layer holds its experts' weights stacked (``[experts, out, in]``) and each token uses a handful of
them. Streamed as a whole unit, the stack would cross the link at every call (Fish Speech's Drama 3: 256 experts, 0.8 GB
a layer, forty layers a token); streamed by the row, a call copies only the experts its tokens were routed to.

A module whose class says ``STREAM_ROWS = True`` declares that its parameters are such stacks (first dimension: the row)
and that its ``forward`` reads them only through ``source(self)``:

- ``source(self)`` is None when the parameters are where the module computes (the unit is resident, or the model is on
  the CPU, which never streams): the module indexes them itself.
- Otherwise it is the ``RowSource`` of the handle streaming the unit: ``take(module, name, row)`` returns that row of the
  parameter on the compute device, copied from its host tensor (page-locked when the streamer pinned it, so the copy is
  asynchronous). The values are the parameter's, so a module computing with gathered rows computes what it would with
  the whole stack resident.

The handle never copies a row module's parameters when its unit is entered (``ModelHandle._enter``); the unit's other
weights stream as usual. A unit's ``reserve`` (the device bytes a streamed call holds) counts those other weights and
the module's ``row_reserve()``: the rows it holds on the device at once. Steps that read rows chosen on the device wait
for the device to learn them, so a handle streaming rows is not graphable (``ModelHandle.graphable``).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol

if TYPE_CHECKING:
    import torch

ATTRIBUTE = "_mizuoto_rows"
"""Set in a row module's ``__dict__`` by the handle streaming its unit: the ``RowSource``."""


class RowSource(Protocol):
    device: str
    """Where the rows are wanted (the handle's load device)."""

    def take(self, module: Any, name: str, row: int) -> torch.Tensor:
        """Row ``row`` of ``module``'s parameter ``name`` on ``device`` (a new tensor, the parameter's dtype)."""
        ...


def source(module: Any) -> RowSource | None:
    """The source of ``module``'s rows while its unit streams, else None (its parameters are where it computes)."""
    return module.__dict__.get(ATTRIBUTE)


def streams_rows(module: Any) -> bool:
    return bool(getattr(type(module), "STREAM_ROWS", False))


def row_reserve(module: Any) -> int:
    """The device bytes ``module`` holds in rows at once while streamed (its ``row_reserve()``, else none)."""
    reserve = getattr(module, "row_reserve", None)
    return int(reserve()) if callable(reserve) else 0
