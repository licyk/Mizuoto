"""A model component the manager places on devices.

The equivalent of ComfyUI's ``ModelPatcher``, without the diffusion parts. It wraps one ``nn.Module`` and knows its
size, where it runs and where it rests. The manager puts it on its device whole, or partly: then the module's units
(``runtime.units``) that the budget leaves out keep their weights in host memory, and each streams to the device just
before it runs (a forward pre-hook) and back after (a forward hook; ``runtime.stream``). The engine only reads
``module``; its code runs the same either way.

- **Moves are per tensor**, as ``Module.to`` makes them (``param.data`` replaced, buffers reassigned), made outside
  inference mode so the parameters stay ordinary tensors (an inference tensor takes other CPU kernels).
  A module pinned to the CPU (``device.pin_to_cpu``) never moves.
- **Host copies.** With ``keep_host``, a parameter placed on the device keeps its host tensor, so offloading it costs no
  transfer (ComfyUI's dynamic VRAM: the device as a cache over host memory). Without, offloading copies it back.
  Buffers are always copied back (a module may update one in place).
- **Freeing** drops everything with no copy back.
- **Never changed in place on the device.** A weight that changes after placement goes through the handle
  (``replace``), so a host copy is never stale.
- ``simulated``: the load device is the CPU playing an accelerator (tests and the parity harness: ``DeviceService``'s
  ``simulate``); placing then copies, so a streamed unit really leaves and comes back. ``strict`` (tests) rests streamed
  weights as meta tensors, so code reading a unit's weight outside its ``forward`` fails.
"""

from __future__ import annotations

import threading
from collections.abc import Iterable
from typing import TYPE_CHECKING, Any, Literal

from mizuoto.runtime.units import Mark, Split, Unit, split

if TYPE_CHECKING:
    from mizuoto.runtime.stream import Streamer

Kind = Literal["oneshot", "sequence"]


class ModelHandle:
    def __init__(self, module: Any, *, name: str, load_device: str, offload_device: str = "cpu", key: str | None = None, kind: Kind = "oneshot") -> None:
        self.module = module
        self.name = name
        """Shown in the loaded-models list, e.g. ``qwen3tts/1.7b-base:talker``."""
        self.key = key or name
        """Content identity: components with the same key are the same weights."""
        self.load_device = load_device
        self.offload_device = offload_device
        self.kind: Kind = kind
        """``sequence``: every unit runs once per generated token (an autoregressive LM): a fixed prefix of units stays
        resident. ``oneshot``: units run once per input (encoders, decoders, flow steps): the largest stay resident."""
        self.simulated = False
        self.strict = False
        self.keep_host = False
        self.cap_bytes: int | None = None
        """Device bytes the handle may hold, set when a phase using it ran out of memory; cleared when it is offloaded."""
        self.placement: dict[str, Any] | None = None
        """Set by the manager when it loads the handle partly: why, with the numbers it decided on."""
        self._marks: dict[int, Mark] = {}
        self._split: Split | None = None
        self._placed = False
        """Fixed parameters and buffers on the load device."""
        self._host: dict[tuple[int, str], Any] = {}
        """Host tensors of parameters that are on the device (or streamed), by (module id, name)."""
        self._host_buffers: dict[tuple[int, str], Any] = {}
        """Buffers of a parked handle (``strict``), by (module id, name)."""
        self._streamer: Streamer | None = None
        self._lock = threading.RLock()

    # -- structure --------------------------------------------------------------------

    def mark(self, module: Any, mark: Mark) -> ModelHandle:
        """Mark a submodule: ``fixed`` (never streamed: code reads its weights from outside its ``forward``), ``hot``
        (runs many times per step: resident first), ``cold`` (runs once per request: streamed first)."""
        self._marks[id(module)] = mark
        self._split = None
        return self

    @property
    def parts(self) -> Split:
        if self._split is None:
            self._split = split(self.module, self._marks)
        return self._split

    @property
    def units(self) -> list[Unit]:
        return self.parts.units if self.module is not None else []

    def replace(self, module: Any) -> None:
        """Put ``module`` in place of the handle's module (a voice's own weights): the old one's weights are dropped and
        the new one starts on the host."""
        with self._lock:
            self._drop_hooks()
            self._host.clear()
            self.module = module
            self._split = None
            self._placed = False
            if self.strict:
                self.park()

    # -- sizes ------------------------------------------------------------------------

    @property
    def size_bytes(self) -> int:
        if self.module is None:
            return 0
        p = self.parts
        return p.fixed_bytes + sum(u.bytes for u in p.units)

    @property
    def fixed_bytes(self) -> int:
        """Bytes on the device whenever the handle is placed at all."""
        return self.parts.fixed_bytes if self.module is not None else 0

    @property
    def loaded_bytes(self) -> int:
        """Bytes on the device now."""
        if self.module is None or not self._placed:
            return 0
        return self.fixed_bytes + sum(u.bytes for u in self.units if u.resident)

    @property
    def streamed_bytes(self) -> int:
        """Bytes left in host memory that stream to the device as their units run."""
        if self.module is None or not self._placed:
            return 0
        return sum(u.bytes for u in self.units if not u.resident)

    @property
    def host_bytes(self) -> int:
        return sum(t.numel() * t.element_size() for t in self._host.values())

    @property
    def pinned_bytes(self) -> int:
        return sum(t.numel() * t.element_size() for t in self._host.values() if t.is_pinned())

    def stream_reserve(self, streamed: Iterable[Unit], copies: int) -> int:
        """Device bytes the copies of streamed units need at once: the ``copies`` largest (the running unit and those in
        flight)."""
        sizes = sorted((u.bytes for u in streamed), reverse=True)
        return sum(sizes[:copies])

    # -- where it is ------------------------------------------------------------------

    @property
    def device(self) -> str:
        """Where the weights are now (``none`` once freed)."""
        if self.module is None:
            return "none"
        return self.load_device if self._placed else self.offload_device

    @property
    def placed(self) -> bool:
        return self.module is not None and (self._placed or not self._moves())

    @property
    def loaded(self) -> bool:
        """Wholly on the device."""
        return self.placed and all(u.resident for u in self.units) if self._moves() else self.module is not None

    @property
    def partial(self) -> bool:
        return self._placed and any(not u.resident for u in self.units)

    def _moves(self) -> bool:
        return self.simulated or not _same_device(self.load_device, self.offload_device)

    # -- placement --------------------------------------------------------------------

    def fill(self, budget: int | None, copies: int) -> set[int]:
        """Indices of the units that stay resident within ``budget`` bytes for the whole handle (fixed parts, resident
        units and the reserve for ``copies`` streamed units in flight); all of them for ``None``."""
        units = self.units
        if budget is None:
            return {u.index for u in units}
        order = self._fill_order(units)
        room = budget - self.fixed_bytes
        chosen: list[Unit] = []
        used = 0
        for u in order:
            if used + u.bytes <= room:
                chosen.append(u)
                used += u.bytes
        while chosen:
            streamed = [u for u in units if u not in chosen]
            if used + self.stream_reserve(streamed, copies) <= room:
                break
            last = chosen.pop()
            used -= last.bytes
        return {u.index for u in chosen}

    def graphable(self) -> bool:
        """Whether CUDA graphs may run this handle's steps: wholly on its device, or a ``sequence`` handle whose streamed
        units rest in page-locked memory. Captured, a streamed unit's copy is a memcpy node from its fixed host tensor
        into the graph's own memory, replayed in order with the kernels that read it (one unit's copy at a time, as
        streaming eagerly holds them); a copy from pageable memory cannot be captured."""
        with self._lock:
            if self.loaded:
                return True
            if self.kind != "sequence" or self._streamer is None or not self._placed:
                return False
            hosts = [self._host.get((id(m), n)) for u in self.units if not u.resident for m, n in u.slots]
            return all(h is not None and h.is_pinned() for h in hosts)

    def replayed(self, steps: int) -> None:
        """``steps`` graph replays of a partly loaded handle each copied its streamed units once (``graphable``): count
        them with the streamer's copies, which the output reports (``meta.memory.streamed_mb``)."""
        with self._lock:
            if self._streamer is not None and not self.loaded:
                self._streamer.copied_bytes += self.streamed_bytes * steps

    def minimum_bytes(self, copies: int) -> int:
        """The least device memory the handle runs in: its fixed parts and the copies of its largest units."""
        return self.fixed_bytes + self.stream_reserve(self.units, copies)

    def _fill_order(self, units: list[Unit]) -> list[Unit]:
        rank = {"hot": 0, "normal": 1, "cold": 2}
        if self.kind == "sequence":
            return sorted(units, key=lambda u: (rank[u.priority], u.index))
        return sorted(units, key=lambda u: (rank[u.priority], -u.bytes, u.index))

    def load(self, budget: int | None = None, streamer: Streamer | None = None) -> None:
        """Put the handle on its device: every unit, or those ``fill`` keeps within ``budget`` (the rest stream through
        ``streamer``)."""
        with self._lock:
            if self.module is None or not self._moves():
                return
            copies = streamer.copies_for(self) if streamer is not None else 1
            keep = self.fill(budget, copies)
            if keep != {u.index for u in self.units} and streamer is None:
                raise ValueError(f"{self.name} needs a streamer to load partly")
            self._place_fixed()
            for u in self.units:
                if u.index in keep:
                    self._make_resident(u)
                else:
                    self._make_streamed(u)
            wanted = streamer if any(not u.resident for u in self.units) else None
            if self._streamer is not None and self._streamer is not wanted:
                self._streamer.detach(self)  # whole now: its pins and learnt order go
            self._streamer = wanted
            if self._streamer is not None:
                self._streamer.attach(self)
            self._flag_streamed()

    def partially_unload(self, nbytes: int) -> int:
        """Stream units until ``nbytes`` are free, in reverse fill order; the whole handle when that is not enough or
        it has no streamer. Returns the bytes freed."""
        with self._lock:
            if self.module is None or not self._placed:
                return 0
            if self._streamer is None:
                freed = self.loaded_bytes
                self.offload()
                return freed
            freed = 0
            for u in reversed(self._fill_order([u for u in self.units if u.resident])):
                if freed >= nbytes:
                    break
                self._make_streamed(u)
                freed += u.bytes
            if freed < nbytes:
                freed += self.loaded_bytes
                self.offload()
            elif self._streamer is not None:
                self._streamer.attach(self)
            self._flag_streamed()
            return freed

    def offload(self) -> None:
        """Everything back to the host (host copies reused; else copied back)."""
        with self._lock:
            if self.module is None or not self._moves() or not self._placed:
                return
            for u in self.units:
                self._rest_unit(u)
            self._home(self.parts.fixed_params)
            for m, n in self.parts.buffers:
                b = m._buffers.get(n)
                if b is not None and not _same_device(str(b.device), self.offload_device):
                    m._buffers[n] = _copy(b, self.offload_device)
            self._drop_hooks()
            if self._streamer is not None:
                self._streamer.detach(self)
                self._streamer = None
            self._placed = False
            self.cap_bytes = None
            if self.strict:
                self.park()
            elif not self.keep_host:
                self._host.clear()
            self._flag_streamed()

    def park(self) -> None:
        """(``strict``) Off the device, every weight and buffer rests as a meta tensor (its host tensor kept), so an engine
        that computes with a component it did not place for the phase fails instead of silently using host weights."""
        with self._lock:
            if self.module is None or self._placed:
                return
            p = self.parts
            for m, n in p.fixed_params + [s for u in p.units for s in u.slots]:
                key = (id(m), n)
                current = m._parameters[n]
                if key not in self._host and current.device.type != "meta":
                    self._host[key] = current.data
                if key in self._host:
                    _assign(m, n, _rest(self._host[key], True))
            for m, n in p.buffers:
                b = m._buffers.get(n)
                if b is not None and b.device.type != "meta":
                    self._host_buffers[(id(m), n)] = b
                    m._buffers[n] = _rest(b, True)

    def unpark(self) -> None:
        """Undo ``park``: the host tensors back in the module (the handle leaves the manager, e.g. after a test)."""
        with self._lock:
            if self.module is None:
                return
            if self._placed:
                self.offload()
            p = self.parts
            for m, n in p.fixed_params + [s for u in p.units for s in u.slots]:
                host = self._host.get((id(m), n))
                if host is not None:
                    _assign(m, n, host)
            for m, n in p.buffers:
                b = self._host_buffers.get((id(m), n))
                if b is not None:
                    m._buffers[n] = b
            self._host.clear()
            self._host_buffers.clear()

    def drop_host_copies(self) -> int:
        """Forget the host copies of resident parameters (memory is short); offloading then copies back. Returns bytes."""
        with self._lock:
            if self.strict or not self._placed:  # parked: the host tensors are the only ones; offloaded: they are in use
                return 0
            streamed = {(id(m), n) for u in self.units if not u.resident for m, n in u.slots}
            gone = 0
            for k in [k for k in self._host if k not in streamed]:
                t = self._host.pop(k)
                gone += t.numel() * t.element_size()
            return gone

    def free(self) -> None:
        """Drop the module and every copy of its weights, without copying anything back."""
        with self._lock:
            self._drop_hooks()
            if self._streamer is not None:
                self._streamer.detach(self)
                self._streamer = None
            self._host.clear()
            self.module = None
            self._split = None
            self._placed = False

    def settle(self) -> None:
        """After a request (or an error inside a unit): every streamed unit back at rest."""
        with self._lock:
            if self._streamer is not None:
                for u in self.units:
                    if not u.resident:
                        u.entered = 0
                        for m, n in u.slots:
                            _assign(m, n, _rest(self._host[(id(m), n)], self.strict))
                self._streamer.release_all(self)

    # -- tensors ----------------------------------------------------------------------

    def _target(self) -> str:
        return self.load_device

    def _place_fixed(self) -> None:
        if self._placed:
            return
        for m, n in self.parts.fixed_params:
            self._to_device(m, n)
        for m, n in self.parts.buffers:
            b = self._host_buffers.get((id(m), n), m._buffers.get(n))
            if b is not None:
                m._buffers[n] = _copy(b, self._target()) if (self.simulated or not _same_device(str(b.device), self._target())) else b
        self._placed = True

    def _to_device(self, m: Any, n: str) -> None:
        p = m._parameters[n]
        key = (id(m), n)
        host = self._host.get(key)
        current = p.data
        if host is None:
            if not self.simulated and _same_device(str(current.device), self._target()):
                return  # already there (the engine built it on the device)
            host = current
        device = _copy(host, self._target())
        if self.keep_host:
            self._host[key] = host
        else:
            self._host.pop(key, None)
        _assign(m, n, device)

    def _make_resident(self, u: Unit) -> None:
        if u.resident:
            return
        self._remove_hooks(u)
        for m, n in u.slots:
            self._to_device(m, n)
        u.resident = True

    def _make_streamed(self, u: Unit) -> None:
        """``u``'s weights rest on the host; hooks bring them for each call."""
        for m, n in u.slots:
            key = (id(m), n)
            p = m._parameters[n]
            if key not in self._host:
                current = p.data
                self._host[key] = current if (not self.simulated and _same_device(str(current.device), self.offload_device)) else _copy(current, self.offload_device)
            _assign(m, n, _rest(self._host[key], self.strict))
        u.resident = False
        self._install_hooks(u)

    def _rest_unit(self, u: Unit) -> None:
        for m, n in u.slots:
            key = (id(m), n)
            p = m._parameters[n]
            host = self._host.get(key)
            if host is not None:
                _assign(m, n, host)
            elif not _same_device(str(p.data.device), self.offload_device) or self.simulated:
                _assign(m, n, _copy(p.data, self.offload_device))
        u.resident = False

    def _home(self, slots: list[tuple[Any, str]]) -> None:
        for m, n in slots:
            p = m._parameters[n]
            host = self._host.get((id(m), n))
            if host is not None:
                _assign(m, n, host)
            elif self.simulated or not _same_device(str(p.data.device), self.offload_device):
                _assign(m, n, _copy(p.data, self.offload_device))

    # -- streaming --------------------------------------------------------------------

    def _install_hooks(self, u: Unit) -> None:
        """Hooks on the unit's module (its weights there for the whole call) and on each module holding its parameters
        (for code that enters the unit by another method, ``layer.forward_step(…)``, or calls a leaf directly)."""
        if u.hooks:
            return
        u.entered = 0
        u.hooks = [
            u.module.register_forward_pre_hook(lambda _m, _a, u=u: self._enter(u)),
            u.module.register_forward_hook(lambda _m, _a, _o, u=u: self._exit(u), always_call=True),
        ]
        for module, indices in u.leaves().values():
            if module is u.module:
                continue
            u.hooks.append(module.register_forward_pre_hook(lambda _m, _a, u=u, i=indices: self._enter_leaf(u, i)))
            u.hooks.append(module.register_forward_hook(lambda _m, _a, _o, u=u, i=indices: self._exit_leaf(u, i), always_call=True))

    def _remove_hooks(self, u: Unit) -> None:
        for h in u.hooks:
            h.remove()
        u.hooks = []
        u.entered = 0

    def _drop_hooks(self) -> None:
        if self._split is not None:
            for u in self._split.units:
                self._remove_hooks(u)

    def _fetch(self, u: Unit) -> list[Any]:
        from mizuoto.runtime.context import check_cancel

        assert self._streamer is not None
        check_cancel()
        return self._streamer.fetch(self, u, [self._host[(id(m), n)] for m, n in u.slots])

    def _enter(self, u: Unit) -> None:
        if u.resident or self._streamer is None:
            return
        u.entered += 1
        if u.entered > 1:
            return
        for (m, n), t in zip(u.slots, self._fetch(u), strict=True):
            _assign(m, n, t)

    def _exit(self, u: Unit) -> None:
        if u.resident or u.entered == 0:
            return
        u.entered -= 1
        if u.entered:
            return
        for m, n in u.slots:
            _assign(m, n, _rest(self._host[(id(m), n)], self.strict))
        if self._streamer is not None:
            self._streamer.done(self, u)

    def _enter_leaf(self, u: Unit, indices: list[int]) -> None:
        if u.resident or u.entered or self._streamer is None:
            return
        if _capturing(self._streamer.device):
            # a CUDA graph being captured (``graphable``): this leaf's weights only, copied inside it, so that each weight
            # is copied once per step though a unit's leaves sit in different graphs (around its attention)
            for i in indices:
                m, n = u.slots[i]
                _assign(m, n, _copy(self._host[(id(m), n)], self._streamer.device))
            return
        tensors = self._fetch(u)
        for i in indices:
            m, n = u.slots[i]
            _assign(m, n, tensors[i])

    def _exit_leaf(self, u: Unit, indices: list[int]) -> None:
        if u.resident or u.entered:
            return
        for i in indices:
            m, n = u.slots[i]
            _assign(m, n, _rest(self._host[(id(m), n)], self.strict))
        if self._streamer is not None:
            self._streamer.done(self, u)

    def _flag_streamed(self) -> None:
        """Mark the modules above a streamed unit (``Compiled`` runs those eagerly: its graph would hold the weights)."""
        if self.module is None:
            return
        streamed = {id(u.module) for u in self.units if not u.resident} if self._placed else set()
        for m in self.module.modules():
            if "_mizuoto_streamed" in m.__dict__:
                del m.__dict__["_mizuoto_streamed"]
        if not streamed:
            return
        for m in self.module.modules():
            if any(id(x) in streamed for x in m.modules()):
                m.__dict__["_mizuoto_streamed"] = True

    def __repr__(self) -> str:
        return f"ModelHandle({self.name!r}, {self.size_bytes / 2**20:.0f} MB, on {self.device})"


def _assign(module: Any, name: str, t: Any) -> None:
    """Make ``t`` the data of ``module``'s parameter ``name``, as ``Module._apply`` does: in place when the tensor types
    allow (CPU and CUDA), else a new ``Parameter`` (the meta device)."""
    import torch

    p = module._parameters[name]
    if torch._has_compatible_shallow_copy_type(p, t):
        p.data = t
    else:  # (an inference tensor cannot require grad: weights a request built, e.g. a voice's own)
        module._parameters[name] = torch.nn.Parameter(t, requires_grad=p.requires_grad and not t.is_inference())


def _capturing(device: str) -> bool:
    if not device.startswith("cuda"):
        return False
    import torch

    return torch.cuda.is_current_stream_capturing()


def _copy(t: Any, device: str) -> Any:
    """``t`` on ``device`` as a new ordinary tensor (made outside inference mode, as a loaded parameter is)."""
    import torch

    with torch.inference_mode(False), torch.no_grad():
        return t.to(device, copy=True, non_blocking=device != "cpu" and t.is_pinned())


def _rest(host: Any, strict: bool) -> Any:
    if not strict:
        return host
    import torch

    with torch.inference_mode(False):
        return torch.empty_like(host, device="meta")


def module_bytes(module: Any) -> int:
    return sum(t.numel() * t.element_size() for t in _tensors(module))


def _tensors(module: Any):  # noqa: ANN202 - an iterator of tensors
    yield from module.parameters()
    yield from module.buffers()


def _same_device(a: str, b: str) -> bool:
    if a == b:
        return True
    # ``cuda`` and ``cuda:0`` are the same device when 0 is the current one
    return a.split(":", 1)[0] == b.split(":", 1)[0] and (a.endswith(":0") or ":" not in a) and (b.endswith(":0") or ":" not in b)
