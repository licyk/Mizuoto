"""Loading weights.

- ``load_state_dict(path)``: safetensors, or a pickle loaded with ``weights_only=True``.
- Key helpers, as ComfyUI's ``state_dict_prefix_replace``.
- ``fold_weight_norm``: weight norm in both of torch's formats, folded into plain weights.
- ``load_into``: strict by default. Every missing or unexpected key is an error naming the key,
  unless the engine lists it as ignorable.
"""

from __future__ import annotations

import fnmatch
import io
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import torch
from torch import nn

from mizuoto.runtime.engine import EngineError

StateDict = dict[str, torch.Tensor]


def load_state_dict(path: str | Path, *, device: str = "cpu", mmap: bool = True) -> Any:
    """A checkpoint's contents: a state dict for safetensors, whatever was saved for a pickle.

    Pickles load with ``weights_only=True``. A file whose zip magic was overwritten (GPT-SoVITS writes
    a version tag in the first two bytes) is read with ``PK`` restored, in memory.
    """
    path = Path(path)
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
    """Copy ``sd`` into ``module``. Keys matching an ``ignore`` pattern (fnmatch) may be missing or extra."""
    patterns = list(ignore)
    result = module.load_state_dict(dict(sd), strict=False)
    missing = [k for k in result.missing_keys if not any(fnmatch.fnmatchcase(k, p) for p in patterns)]
    unexpected = [k for k in result.unexpected_keys if not any(fnmatch.fnmatchcase(k, p) for p in patterns)]
    if strict and (missing or unexpected):
        raise EngineError(
            f"The {what}'s weights do not match its architecture: {len(missing)} missing, {len(unexpected)} unexpected",
            {"missing": missing[:20], "unexpected": unexpected[:20]},
        )
