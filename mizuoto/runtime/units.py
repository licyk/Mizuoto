"""Units: the parts of a model that may stay off the device and stream to it.

A unit is one module whose parameters move together: a decoder layer, a DiT block, a vocoder stage. When a model does
not fit its device, the units the budget leaves out keep their weights in host memory, and each moves to the device
just before the unit runs and back after it (``runtime.stream``). Everything outside the units (embeddings, heads,
norms between blocks, every buffer) is *fixed*: on the device whenever the model is.

- Only the weights of linear and convolution layers stream (as ComfyUI streams only its ops layers), and of layers whose
  class says ``STREAMABLE = True``: their ``forward`` is where they are read. Any other parameter of a unit (norms,
  scales, embeddings: codebooks are read through their ``weight``) stays with the fixed parts.
- A module may name its units: ``memory_units() -> list[nn.Module]`` (ComfyUI's ``get_dynamic_vram__units``).
- Otherwise each child of a ``ModuleList`` or ``Sequential`` of at least ``UNIT_MIN_BYTES`` is a unit (the outermost
  lists: a unit is never split further).
- Kept fixed, whatever the split says:
  - a parameter several modules share unless they are all inside one unit (tied embeddings and heads);
  - the model's first parameter (code reads the model's device from it);
  - units smaller than ``TINY_BYTES``;
  - modules an engine marks ``fixed`` (``ModelHandle.mark``), and everything under a module pinned to the CPU
    (``device.pin_to_cpu``), which never moves at all.

A parameter of a unit is accessed only while the unit's ``forward`` runs (the weights are there then); code that reads a
unit's weight from outside its ``forward`` must mark that module ``fixed``. The tests run every tiny engine with its units
streamed and their resting weights replaced by meta tensors (``strict``), so such a read fails there.
"""

from __future__ import annotations

import functools
from dataclasses import dataclass, field
from typing import Any, Literal

Priority = Literal["hot", "normal", "cold"]
Mark = Literal["fixed", "hot", "normal", "cold"]

TINY_BYTES = 64 * 1024
"""Units smaller than this stay on the device (ComfyUI force-loads modules of 16 KB or less: tiny weights streamed
between giant ones stall the copy streams)."""
UNIT_MIN_BYTES = 1 << 20
"""A list child smaller than this is not worth a transfer of its own: it stays fixed."""


@functools.cache
def streamable() -> tuple[type, ...]:
    """The layers whose weights stream. Weights of other modules stay with the fixed parts: norms, scales, and
    embeddings, which serve as codebooks read through their ``weight`` (``F.embedding(ids, self.codebook.weight)``, a
    cosine search) as often as they are called."""
    from torch import nn

    return (nn.Linear, nn.Conv1d, nn.Conv2d, nn.ConvTranspose1d, nn.ConvTranspose2d)


Slot = tuple[Any, str]
"""(module, name) of a parameter."""


@dataclass(eq=False)
class Unit:
    name: str
    module: Any
    slots: list[Slot]
    bytes: int
    index: int
    priority: Priority = "normal"
    resident: bool = False
    hooks: list[Any] = field(default_factory=list)
    entered: int = 0
    """Depth of calls into the unit's own module (its weights are all there meanwhile)."""

    def leaves(self) -> dict[int, tuple[Any, list[int]]]:
        """The modules holding the unit's parameters: id → (module, indices into ``slots``)."""
        out: dict[int, tuple[Any, list[int]]] = {}
        for i, (m, _) in enumerate(self.slots):
            out.setdefault(id(m), (m, []))[1].append(i)
        return out


@dataclass
class Split:
    units: list[Unit]
    fixed_params: list[Slot]
    buffers: list[Slot]
    excluded: set[int]
    """``id`` of every module under a module pinned to the CPU (never moved)."""

    @property
    def fixed_bytes(self) -> int:
        return sum(_nbytes(getattr(m, n)) for m, n in self.fixed_params) + sum(_nbytes(m._buffers[n]) for m, n in self.buffers)


def _nbytes(t: Any) -> int:
    return 0 if t is None else t.numel() * t.element_size()


def _pinned(module: Any) -> bool:
    return "_apply" in module.__dict__  # device.pin_to_cpu gives the module an instance _apply


def split(root: Any, marks: dict[int, Mark] | None = None) -> Split:
    """``root``'s units, fixed parameters and buffers."""
    from torch import nn

    marks = marks or {}
    excluded: set[int] = set()
    for m in root.modules():
        if _pinned(m):
            excluded.update(id(x) for x in m.modules())

    candidates: list[tuple[str, Any, bool]] = []
    """(name, module, named by ``memory_units``)"""

    def visit(name: str, m: Any) -> None:
        if id(m) in excluded or marks.get(id(m)) == "fixed":
            return
        named = getattr(m, "memory_units", None)
        if callable(named):
            names = {id(c): n for n, c in m.named_modules(prefix=name)}
            for child in named():
                candidates.append((names.get(id(child), f"{name}.?"), child, True))
            return
        if isinstance(m, nn.ModuleList | nn.Sequential) and m is not root:
            for child_name, child in m.named_children():
                candidates.append((f"{name}.{child_name}" if name else child_name, child, False))
            return
        for child_name, child in m.named_children():
            visit(f"{name}.{child_name}" if name else child_name, child)

    visit("", root)

    # who references each parameter (tied weights are one Parameter in several modules)
    owners: dict[int, list[Any]] = {}
    for m in root.modules():
        for p in m._parameters.values():
            if p is not None:
                owners.setdefault(id(p), []).append(m)
    first = next(iter(root.parameters()), None)
    inherited: dict[int, Priority] = {}
    for m in root.modules():  # outermost first, so a deeper mark wins
        mark = marks.get(id(m))
        if mark in ("hot", "normal", "cold"):
            for x in m.modules():
                inherited[id(x)] = mark  # type: ignore[assignment]

    units: list[Unit] = []
    in_unit: set[int] = set()
    for index, (name, module, named) in enumerate(candidates):
        if id(module) in excluded or marks.get(id(module)) == "fixed" or id(module) in in_unit:
            continue
        inside = {id(x) for x in module.modules()}
        if inside & excluded or any(marks.get(i) == "fixed" for i in inside):
            continue
        slots: list[Slot] = []
        for m in module.modules():
            if not isinstance(m, streamable()) and not getattr(type(m), "STREAMABLE", False):
                continue  # other modules' weights stay with the fixed parts (ComfyUI streams only its ops layers)
            for pname, p in m._parameters.items():
                if p is None or p is first:
                    continue
                if not all(id(o) in inside for o in owners[id(p)]):
                    continue  # shared with a module outside the unit
                if any(s[0]._parameters[s[1]] is p for s in slots):
                    continue  # tied inside the unit: one slot
                slots.append((m, pname))
        size = sum(_nbytes(getattr(m, n)) for m, n in slots)
        if size == 0 or size < (TINY_BYTES if named else UNIT_MIN_BYTES):
            continue
        units.append(Unit(name, module, slots, size, index, inherited.get(id(module), "normal")))
        in_unit |= inside

    unit_params = {id(m._parameters[n]) for u in units for m, n in u.slots}
    fixed: list[Slot] = []
    seen: set[int] = set()
    buffers: list[Slot] = []
    for m in root.modules():
        if id(m) in excluded:
            continue
        for pname, p in m._parameters.items():
            if p is not None and id(p) not in unit_params and id(p) not in seen:
                seen.add(id(p))
                fixed.append((m, pname))
        for bname, b in m._buffers.items():
            if b is not None:
                buffers.append((m, bname))
    return Split(units, fixed, buffers, excluded)
