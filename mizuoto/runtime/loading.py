"""Loading weights.

- ``load_state_dict(path)``: safetensors, or a pickle loaded with ``weights_only=True``.
- Key helpers, as ComfyUI's ``state_dict_prefix_replace``.
- ``fold_weight_norm``: weight norm in both of torch's formats, folded into plain weights.
- ``load_into``: strict by default. Every missing or unexpected key is an error naming the key,
  unless the engine lists it as ignorable.

**Weights left in the checkpoint's mapping.** Safetensors files and pickles are read memory-mapped, copy-on-write, so
a checkpoint's tensors are pages of the file. While a model loads through the runtime for a device other than the CPU
(``placing``), ``load_into`` points a parameter at such a tensor (``param.data = …``) instead of copying it into the
memory the module allocated, when the copy would be the same tensor: the same dtype, shape and layout, a tensor
``load_state_dict`` read (never a live module's, which ``state_dict()`` shares) and not yet handed to another
parameter. The load then stages no second copy of the model in RAM: the weights stay page cache the system can drop
and read again, and become the process's own memory only where they are written (copy-on-write) or page-locked
(``cudaHostRegister`` copies the pages it locks). The parameter object stays the module's, so ``requires_grad``, tied
weights and the handles' later ``.data`` swaps are as before; buffers, cast weights and tied parameters are copied.

There the weights only feed the copies to the device. Weights the CPU computes with must be the process's own memory:
CPU kernels can depend on alignment (a float32 matrix-vector product, ``F.linear`` on one row, gives other last bits
for a weight 8 bytes off a 16-byte boundary; torch's allocator aligns to 64 bytes, safetensors to 8), and they ran
slower over the file's pages (IndexTTS 2's decoding on the CPU took 13 % longer, its weights 64-byte aligned). So
``Placement.settle`` copies every such weight that no handle on that device holds (a module an engine runs on the CPU
in any case, such as CosyVoice's CAM++), and everything else (CPU runs, training, previews) copies as before. On
Windows nothing is pointed at: a mapped file cannot be replaced or deleted there while a model is loaded.

**The memory a load leaves behind.** Engines cast a module to the dtype it runs in before loading it, which writes the
module's own allocation; once the checkpoint's tensors replace it, it is freed, but glibc keeps freed heap memory (any
allocation under its mmap threshold, up to 32 MB) in the process: 1.3 GB of Qwen3-TTS's talker stayed resident.
``trim_heap`` (glibc's ``malloc_trim``) gives the free pages back after a load.
"""

from __future__ import annotations

import contextlib
import contextvars
import fnmatch
import io
import sys
import weakref
from collections.abc import Generator, Iterable, Mapping
from pathlib import Path
from typing import Any

import torch
from torch import nn

from mizuoto.runtime.engine import EngineError

StateDict = dict[str, torch.Tensor]

_read: dict[int, weakref.ref[torch.Tensor]] = {}
"""id → tensor, for the checkpoint tensors ``load_state_dict`` read and no parameter took yet."""


def _remember(obj: Any, depth: int = 0) -> None:
    """Note the tensors in a checkpoint (nested in dicts and lists, as pickles hold them) as read from a file."""
    if isinstance(obj, torch.Tensor):
        if obj.device.type == "cpu" and type(obj) is torch.Tensor:
            key = id(obj)
            _read[key] = weakref.ref(obj, lambda _ref, key=key: _read.pop(key, None))
    elif depth < 4 and isinstance(obj, Mapping):
        for v in obj.values():
            _remember(v, depth + 1)
    elif depth < 4 and isinstance(obj, (list, tuple)):
        for v in obj:
            _remember(v, depth + 1)


def _take(t: torch.Tensor) -> bool:
    """Whether ``t`` was read from a checkpoint and not taken yet; it is taken now."""
    ref = _read.get(id(t))
    if ref is None or ref() is not t:
        return False
    del _read[id(t)]
    return True


class Placement:
    """A model loading for ``device``: the parameters left in their checkpoint's mapping."""

    def __init__(self, device: str) -> None:
        self.device = device
        self.adopted: list[nn.Parameter] = []

    @property
    def elsewhere(self) -> bool:
        """Whether the model computes away from the CPU (its weights then compute on copies)."""
        return torch.device(self.device).type != "cpu"

    def settle(self, handles: Iterable[Any]) -> int:
        """Copy into the process's own memory the adopted parameters no handle on a device other than the CPU holds
        (they compute on the CPU); how many."""
        held: set[int] = set()
        for h in handles:
            if torch.device(h.load_device).type != "cpu":
                held.update(id(p) for p in h.module.parameters())
        moved = 0
        for p in self.adopted:
            if id(p) not in held and p.device.type == "cpu":
                p.data = p.data.clone()
                moved += 1
        self.adopted.clear()
        return moved


def trim_heap() -> None:
    """Give the C heap's free pages back to the system (glibc's ``malloc_trim``; nothing elsewhere)."""
    global _malloc_trim
    if _malloc_trim is None:
        _malloc_trim = False
        if sys.platform.startswith("linux"):
            import ctypes
            import ctypes.util

            with contextlib.suppress(OSError, AttributeError, TypeError):
                _malloc_trim = ctypes.CDLL(ctypes.util.find_library("c") or "libc.so.6").malloc_trim
    if _malloc_trim:
        _malloc_trim(0)


_malloc_trim: Any = None

_placement: contextvars.ContextVar[Placement | None] = contextvars.ContextVar("mizuoto_placement", default=None)


@contextlib.contextmanager
def placing(device: str) -> Generator[Placement]:
    """While an engine loads for ``device``: weights may stay in their checkpoint's mapping when it is not the CPU; call
    ``settle`` with the engine's handles once it has loaded."""
    placement = Placement(device)
    token = _placement.set(placement)
    try:
        yield placement
    finally:
        _placement.reset(token)


def load_state_dict(path: str | Path, *, device: str = "cpu", mmap: bool = True) -> Any:
    """A checkpoint's contents: a state dict for safetensors, whatever was saved for a pickle.

    Pickles load with ``weights_only=True``. A file whose zip magic was overwritten (GPT-SoVITS writes
    a version tag in the first two bytes) is read with ``PK`` restored, in memory.
    """
    result = _read_checkpoint(Path(path), device, mmap)
    _remember(result)
    return result


def _read_checkpoint(path: Path, device: str, mmap: bool) -> Any:
    if path.suffix == ".safetensors":
        from safetensors.torch import load_file

        return load_file(str(path), device=device)
    with path.open("rb") as f:
        head = f.read(2)
    if head != b"PK" and _is_tagged_zip(path):
        data = bytearray(path.read_bytes())
        data[:2] = b"PK"
        return torch.load(io.BytesIO(bytes(data)), map_location=device, weights_only=True)
    try:
        return torch.load(str(path), map_location=device, weights_only=True, mmap=mmap)
    except RuntimeError as e:
        if mmap and "mmap" in str(e):
            return torch.load(str(path), map_location=device, weights_only=True)
        raise


def _is_tagged_zip(path: Path) -> bool:
    with path.open("rb") as f:
        head = f.read(4)
    return head[2:4] == b"\x03\x04"


def unwrap(obj: Any, *keys: str) -> StateDict:
    """The first of ``keys`` present in a checkpoint dict (``state_dict``, ``model``, ``weight``…), or the dict itself."""
    if isinstance(obj, Mapping):
        for key in keys:
            if key in obj and isinstance(obj[key], Mapping):
                return dict(obj[key])
    return dict(obj)


def strip_prefix(sd: Mapping[str, torch.Tensor], prefix: str, only: bool = False) -> StateDict:
    """Remove ``prefix`` from the keys that have it; with ``only``, drop the others."""
    out: StateDict = {}
    for k, v in sd.items():
        if k.startswith(prefix):
            out[k[len(prefix) :]] = v
        elif not only:
            out[k] = v
    return out


def replace_prefixes(sd: Mapping[str, torch.Tensor], mapping: Mapping[str, str]) -> StateDict:
    out: StateDict = {}
    for k, v in sd.items():
        for old, new in mapping.items():
            if k.startswith(old):
                k = new + k[len(old) :]
                break
        out[k] = v
    return out


def fold_weight_norm(sd: Mapping[str, torch.Tensor], dim: int = 0, dtype: torch.dtype | None = None, device: torch.device | str | None = None) -> StateDict:
    """Turn ``x.weight_g`` + ``x.weight_v`` and ``x.parametrizations.weight.original0/1`` into ``x.weight``.

    Computed in float32 and stored in ``weight_v``'s dtype, as ``torch.nn.utils.remove_weight_norm`` does; with
    ``dtype``, ``g`` and ``v`` are first cast to it and so is the result (a float16 checkpoint loaded into a float32
    module whose weight norm is computed at run time).

    With a ``device`` other than the CPU, the norm is computed there by ``torch._weight_norm`` in ``dtype`` (else the
    pair's own), as a weight norm left on a model moved there and cast computes it at every call: a GPU's reduction
    and a float16 kernel give other bits than the CPU's float32. The result comes back to the CPU.
    """
    out: StateDict = {}
    pairs: dict[str, dict[str, torch.Tensor]] = {}
    for k, v in sd.items():
        base, _, leaf = k.rpartition(".")
        if leaf in ("weight_g", "weight_v"):
            pairs.setdefault(base, {})[leaf[-1]] = v
        elif leaf in ("original0", "original1") and (base == "parametrizations.weight" or base.endswith(".parametrizations.weight")):
            owner = base[: -len("parametrizations.weight")].rstrip(".")
            pairs.setdefault(owner, {})["g" if leaf == "original0" else "v"] = v
        else:
            out[k] = v
    for base, gv in pairs.items():
        if set(gv) != {"g", "v"}:
            raise EngineError(f"Incomplete weight norm for {base or 'the module'}", {"keys": sorted(gv)})
        g, v = gv["g"], gv["v"]
        if dtype is not None:
            g, v = g.to(dtype), v.to(dtype)
        if device is not None and torch.device(device).type != "cpu":
            out[f"{base}.weight" if base else "weight"] = torch._weight_norm(v.to(device), g.to(device), dim).cpu()
            continue
        compute = torch.float64 if v.dtype == torch.float64 else torch.float32
        out[f"{base}.weight" if base else "weight"] = torch._weight_norm(v.to(compute), g.to(compute), dim).to(v.dtype)
    return out


def load_into(module: nn.Module, sd: Mapping[str, torch.Tensor], *, strict: bool = True, ignore: Iterable[str] = (), what: str = "model") -> None:
    """Load ``sd`` into ``module``: checkpoint tensors the module can use as they are become its parameters' data (see
    the module's docstring), the rest is copied. Keys matching an ``ignore`` pattern (fnmatch) may be missing or extra."""
    patterns = list(ignore)
    taken = _adopt(module, sd)
    result = module.load_state_dict({k: v for k, v in sd.items() if k not in taken}, strict=False)
    missing = [k for k in result.missing_keys if k not in taken and not any(fnmatch.fnmatchcase(k, p) for p in patterns)]
    unexpected = [k for k in result.unexpected_keys if not any(fnmatch.fnmatchcase(k, p) for p in patterns)]
    if strict and (missing or unexpected):
        raise EngineError(
            f"The {what}'s weights do not match its architecture: {len(missing)} missing, {len(unexpected)} unexpected",
            {"missing": missing[:20], "unexpected": unexpected[:20]},
        )


def _adopt(module: nn.Module, sd: Mapping[str, torch.Tensor]) -> set[str]:
    """Point ``module``'s parameters at the checkpoint tensors of ``sd`` they can use as they are; their keys."""
    placement = _placement.get()
    if placement is None or not placement.elsewhere or sys.platform == "win32":
        return set()
    params = dict(module.named_parameters(remove_duplicate=False))
    keys_of: dict[int, int] = {}
    for p in params.values():
        keys_of[id(p)] = keys_of.get(id(p), 0) + 1
    taken: set[str] = set()
    for key, src in sd.items():
        p = params.get(key)
        if p is None or keys_of[id(p)] > 1 or not isinstance(src, torch.Tensor):
            continue  # not a parameter, or a tied one: copied, so the last key wins as before
        if p.device.type != "cpu" or src.device.type != "cpu" or src.dtype != p.dtype or src.shape != p.shape or src.layout != torch.strided:
            continue
        if src.requires_grad or src.is_inference() or not src.is_contiguous() or src.storage_offset() != 0:
            continue
        if src.untyped_storage().nbytes() != src.numel() * src.element_size():
            continue  # a view of a larger storage: the rest would stay alive with it
        if not _take(src):
            continue
        p.data = src
        placement.adopted.append(p)
        taken.add(key)
    return taken
