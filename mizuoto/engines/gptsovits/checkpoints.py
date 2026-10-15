"""Reading GPT-SoVITS' SoVITS checkpoints (``process_ckpt.py``: ``get_sovits_version_from_path_fast``,
``load_sovits_new``; the version rules of ``TTS.init_vits_weights``).

A SoVITS ``.pth`` is a torch zip holding ``weight`` (the state dict, float16 in fine-tunes), ``config`` (the
training hparams as a plain dict), ``info`` and, for LoRA fine-tunes, ``lora_rank``. Newer files carry a version
tag in their first two bytes, written over the zip's ``PK`` magic (``my_save2``); ``load_state_dict`` restores it.

The version is found as upstream finds it:

1. the MD5 of the first 8 KiB names a pretrained base;
2. else a tag (``00``–``08``) in the first two bytes;
3. else (``PK``: an older fine-tune) the file size: v1 under 82,978 KiB, v2 under 700 MiB, else v3.

``TTS.init_vits_weights`` then refines it: the symbol set is v1 when ``enc_p.text_embedding`` has 322 rows; a
v1/v2 model's version is that symbol version; v2Pro/v2ProPlus and v3/v4/v5 keep theirs.
"""

from __future__ import annotations

import hashlib
import io
import pickle
import types
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from mizuoto.runtime.engine import EngineError

HEAD_TO_VERSION: dict[bytes, tuple[str, str, bool]] = {
    b"00": ("v1", "v1", False),
    b"01": ("v2", "v2", False),
    b"02": ("v2", "v3", False),
    b"03": ("v2", "v3", True),
    b"04": ("v2", "v4", True),
    b"05": ("v2", "v2Pro", False),
    b"06": ("v2", "v2ProPlus", False),
    b"07": ("v2", "v5dev", True),
    b"08": ("v2", "v5turbo", True),
}
"""``process_ckpt.head2version``: (symbol version, model version, LoRA) by the two-byte tag."""

PRETRAINED_MD5: dict[str, tuple[str, str, bool]] = {
    "dc3c97e17592963677a4a1681f30c653": ("v2", "v2", False),  # s2G488k.pth (v1 pretrained)
    "43797be674a37c1c83ee81081941ed0f": ("v2", "v3", False),  # s2Gv3.pth
    "6642b37f3dbb1f76882b69937c95a5f3": ("v2", "v2", False),  # s2G2333K.pth
    "4f26b9476d0c5033e04162c486074374": ("v2", "v4", False),  # s2Gv4.pth
    "c7e9fce2223f3db685cdfa1e6368728a": ("v2", "v2Pro", False),  # s2Gv2Pro.pth
    "66b313e39455b57ab1b0bc0b239c9d0a": ("v2", "v2ProPlus", False),  # s2Gv2ProPlus.pth
    "c4ce1f839a0d271e41963f9fb16ade76": ("v2", "v5dev", False),  # s2Gv5dev.pth
    "6bc5cb2a195cc8ec22b2e723cd4d6f47": ("v2", "v5turbo", False),  # s2Gv5turbo.pth
}
"""``process_ckpt.hash_pretrained_dict``: MD5 of a pretrained base's first 8 KiB."""

V2_FAMILY = frozenset({"v1", "v2", "v2Pro", "v2ProPlus"})
"""Model versions built as ``SynthesizerTrn`` (VITS decoder); the others are ``SynthesizerTrnV3`` (CFM + vocoder)."""
V2PRO = frozenset({"v2Pro", "v2ProPlus"})
V3_FAMILY = frozenset({"v3", "v4", "v5", "v5dev", "v5turbo"})


def sovits_version(path: str | Path) -> tuple[str, str, bool]:
    """(symbol version, model version, LoRA) of a SoVITS file, as ``get_sovits_version_from_path_fast``."""
    path = Path(path)
    with path.open("rb") as f:
        head = f.read(8192)
    known = PRETRAINED_MD5.get(hashlib.md5(head).hexdigest())  # upstream's identification, not security
    if known is not None:
        return known
    tag = head[:2]
    if tag != b"PK":
        if tag not in HEAD_TO_VERSION:
            raise EngineError(f"{path.name} is not a GPT-SoVITS SoVITS checkpoint (unknown version tag {tag!r})", {"path": str(path)})
        return HEAD_TO_VERSION[tag]
    size = path.stat().st_size
    if size < 82978 * 1024:
        return "v1", "v1", False
    if size < 700 * 1024 * 1024:
        return "v2", "v2", False
    return "v2", "v3", False


@dataclass
class SovitsCheckpoint:
    path: Path
    version: str
    """The symbol set: ``v1`` (322 symbols) or ``v2``."""
    model_version: str
    """``v1``, ``v2``, ``v2Pro``, ``v2ProPlus``, ``v3``, ``v4``, ``v5dev`` or ``v5turbo``."""
    lora: bool
    lora_rank: int | None
    hps: dict[str, Any]
    """The checkpoint's ``config``, with ``model.semantic_frame_rate`` = ``25hz`` and ``model.version`` set as upstream sets them."""
    weight: dict[str, torch.Tensor]

    @property
    def is_v2pro(self) -> bool:
        return self.model_version in V2PRO

    @property
    def sampling_rate(self) -> int:
        return int(self.hps["data"]["sampling_rate"])

    @property
    def spec_params(self) -> dict[str, int]:
        """``filter_length``, ``hop_length``, ``win_length`` and ``sampling_rate``: the reference spectrogram's settings."""
        data = self.hps["data"]
        return {k: int(data[k]) for k in ("filter_length", "hop_length", "win_length", "sampling_rate")}


def read_sovits(path: str | Path) -> SovitsCheckpoint:
    """A SoVITS checkpoint with its versions resolved (``init_vits_weights`` up to building the model)."""
    path = Path(path)
    _, model_version, lora = sovits_version(path)
    raw = load_checkpoint(path)
    if not isinstance(raw, dict) or "weight" not in raw or "config" not in raw:
        raise EngineError(f"{path.name} is not a GPT-SoVITS SoVITS checkpoint (no weight/config)", {"path": str(path)})
    hps: dict[str, Any] = _plain(raw["config"])
    weight: dict[str, torch.Tensor] = dict(raw["weight"])
    model = hps.setdefault("model", {})
    model["semantic_frame_rate"] = "25hz"
    emb = weight.get("enc_p.text_embedding.weight")
    version = "v1" if emb is not None and emb.shape[0] == 322 else "v2"
    model["version"] = version
    if model_version not in V3_FAMILY:
        if "Pro" not in model_version:
            model_version = version
        else:
            model["version"] = model_version
    else:
        model["version"] = model_version
    rank = raw.get("lora_rank")
    return SovitsCheckpoint(path, version, model_version, lora, int(rank) if rank is not None else None, hps, weight)


def load_checkpoint(path: str | Path) -> Any:
    """A GPT-SoVITS pickle (tagged or not). ``weights_only`` first; a fine-tune whose ``config`` is upstream's
    ``utils.HParams`` is read by a restricted unpickler that knows only tensors, ``OrderedDict`` and ``HParams``
    (as a dict), so nothing in the file can run code.
    """
    from mizuoto.runtime.loading import load_state_dict

    path = Path(path)
    try:
        return load_state_dict(path, mmap=False)
    except pickle.UnpicklingError:
        pass
    data = bytearray(path.read_bytes())
    data[:2] = b"PK"
    return torch.load(io.BytesIO(bytes(data)), map_location="cpu", weights_only=False, pickle_module=_RESTRICTED)


class _HParams(dict):
    """Stands in for ``utils.HParams`` while unpickling: its attributes become keys."""

    def __setstate__(self, state: dict[str, Any]) -> None:
        self.update(state)


_ALLOWED: dict[tuple[str, str], Any] = {
    ("utils", "HParams"): _HParams,
    ("collections", "OrderedDict"): __import__("collections").OrderedDict,
    ("torch._utils", "_rebuild_tensor_v2"): torch._utils._rebuild_tensor_v2,
    ("torch._utils", "_rebuild_parameter"): torch._utils._rebuild_parameter,
    ("torch", "Size"): torch.Size,
}


class _RestrictedUnpickler(pickle.Unpickler):
    def find_class(self, module: str, name: str) -> Any:
        found = _ALLOWED.get((module, name))
        if found is None:
            raise pickle.UnpicklingError(f"GPT-SoVITS checkpoint refers to {module}.{name}, which is not allowed")
        return found


_RESTRICTED = types.ModuleType("mizuoto_gptsovits_pickle")
_RESTRICTED.__dict__.update({"Unpickler": _RestrictedUnpickler, "load": pickle.load, "UnpicklingError": pickle.UnpicklingError, "__name__": "pickle"})


def _plain(obj: Any) -> Any:
    """A config as plain dicts (upstream's ``HParams`` pickles as a dict-like; ``weights_only`` gives dicts already)."""
    if isinstance(obj, dict):
        return {k: _plain(v) for k, v in obj.items()}
    if hasattr(obj, "items"):
        return {k: _plain(v) for k, v in obj.items()}
    return obj
