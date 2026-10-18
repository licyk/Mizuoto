"""Page-locking host copies of weights where they are (ComfyUI's ``pin_memory``: ``cudaHostRegister``).

A copy to an NVIDIA GPU from page-locked memory is asynchronous and runs at the link's speed (13.4 GB/s on an RTX 3090
at PCIe 4.0 x8, against 8.9 GB/s for a Qwen3-TTS talker's pageable host copies); ``Tensor.pin_memory()`` gets there by
allocating page-locked memory and copying into it, 1.5 GB/s, while registering the pages a tensor already has runs at
14 GB/s and copies nothing. So the streamer pins the host copies of streamed weights this way, and a component whose
host copies are kept for its next load (``ModelHandle.keep_host``) has them pinned as it loads, its reloads then
copying at the link's speed.

A registered range must be unregistered before its memory is freed: a later allocation at the same address would
otherwise be taken for page-locked memory and copied from pages it no longer owns. The registration is therefore tied
to the storage: a finalizer, which runs when the storage dies (before its memory is freed), waits for the device (an
asynchronous copy from the pages may still be running), unregisters them and gives their bytes back to the budget.

Only whole storages of at least ``MIN_BYTES`` are registered: smaller ones share pages with other allocations (two
registrations may not overlap) and are too small for their copies' speed to matter. Other devices, ROCm included, are
left to ``pin_memory`` (the streamer) or stay pageable; any refusal leaves the tensor pageable, which is only slower.
"""

from __future__ import annotations

import logging
import threading
import weakref
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from mizuoto.runtime.stream import PinBudget

logger = logging.getLogger(__name__)

MIN_BYTES = 1 << 20

_lock = threading.Lock()
_registered: dict[int, int] = {}
"""data pointer → bytes of each storage registered here"""


def usable(device: str) -> bool:
    """Whether host memory may be registered for copies to ``device`` (an NVIDIA GPU)."""
    if not device.startswith("cuda"):
        return False
    import torch
    import torch.version

    return getattr(torch.version, "hip", None) is None and torch.cuda.is_available()


def pin(t: Any, budget: PinBudget, device: str) -> bool:
    """Page-lock the storage of the host tensor ``t`` in place, within ``budget``, for copies to ``device``. True when
    its pages are page-locked now (by this call or before)."""
    import torch

    if t.device.type != "cpu" or not usable(device):
        return False
    storage = t.untyped_storage()
    ptr, size = storage.data_ptr(), storage.nbytes()
    with _lock:
        if ptr in _registered:
            return True
        if size < MIN_BYTES or t.is_pinned():
            return t.is_pinned()
        if not budget.take(size):
            return False
        cudart = torch.cuda.cudart()
        err = int(cudart.cudaHostRegister(ptr, size, 0))
        if err != 0:
            torch.cuda.cudart().cudaGetLastError()  # clear it, or the next CUDA call reports it
            budget.give(size)
            logger.debug("Could not page-lock %d bytes of host memory (CUDA error %d); they stay pageable", size, err)
            return False
        _registered[ptr] = size
    weakref.finalize(storage, _unpin, ptr, size, budget, device).atexit = False  # the process's exit frees them all
    return True


def pinned_bytes() -> int:
    """Bytes registered here and not yet released."""
    with _lock:
        return sum(_registered.values())


def _unpin(ptr: int, size: int, budget: PinBudget, device: str) -> None:
    import torch

    try:
        if not torch.cuda.is_current_stream_capturing():
            torch.cuda.synchronize(torch.device(device))  # no copy from these pages may still be running
        torch.cuda.cudart().cudaHostUnregister(ptr)
    except Exception as e:  # interpreter shutdown, a device gone: nothing more to do
        logger.debug("Unregistering host memory failed: %s", e)
    finally:
        with _lock:
            _registered.pop(ptr, None)
        budget.give(size)
