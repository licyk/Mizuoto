"""Host weights kept in a file instead of RAM, for when RAM cannot hold them.

A model whose weights stream to the GPU from host memory (``runtime.rows``: Fish Speech Drama 3's 30 GB of experts)
needs them all in RAM, so on a machine with less it could not load. ``scratch`` gives a tensor whose memory is a mapping
of a new file in a folder the core names (``LoadSpec.cache_dir``): the engine fills it as it would fill ``torch.empty``'s
memory, the kernel writes the pages to the file, and from then on they are page cache: kept in RAM while RAM is free,
dropped under pressure (written back, never swapped) and read from the file again when a call needs them. The values
are the same bytes, so everything computed from them is identical; only reading them may be slower.

- Only what does not fit goes there: an engine keeps in RAM what its budget allows and maps the rest (``Plan``), so with
  enough RAM nothing changes.
- The file is unlinked as soon as it is mapped: it lives as long as the tensor and leaves nothing behind (POSIX only;
  ``available`` is False on Windows, where a mapped file cannot be deleted, and everything stays in RAM there).
- Such a tensor is never page-locked (``runtime.pinning`` and the streamer skip it): locking would read the whole file
  into RAM and keep it there. Copies from it are pageable, so somewhat slower than from pinned memory.
- ``is_file_backed`` tells such tensors (and their views) apart; ``ModelHandle`` reports their bytes as ``disk_bytes``,
  not as host memory.
"""

from __future__ import annotations

import logging
import os
import sys
import tempfile
import threading
import weakref
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_lock = threading.Lock()
_mapped: dict[int, int] = {}
"""storage data pointer → bytes, for every storage ``scratch`` mapped and still alive"""


def available() -> bool:
    """Whether weights may be kept in a file here (not on Windows: a mapped file cannot be deleted there)."""
    return sys.platform != "win32"


def scratch(folder: Path, shape: tuple[int, ...], dtype: Any) -> Any:
    """A new, uninitialised CPU tensor of ``shape`` and ``dtype`` whose memory is a mapping of an unlinked file in
    ``folder`` (created as needed). Raises ``OSError`` when the file cannot be made (no space, a read-only folder)."""
    import torch

    numel = 1
    for n in shape:
        numel *= int(n)
    nbytes = numel * torch.empty((), dtype=dtype).element_size()
    folder.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix="weights-", suffix=".bin", dir=folder)
    try:
        os.ftruncate(fd, max(nbytes, 1))
        reserve_blocks(fd, folder, max(nbytes, 1))
        os.close(fd)
        fd = -1
        raw = torch.from_file(name, shared=True, size=max(nbytes, 1), dtype=torch.uint8)
    finally:
        if fd >= 0:
            os.close(fd)
        os.unlink(name)  # the mapping keeps the file alive; nothing stays behind once it goes
    storage = raw.untyped_storage()
    ptr = storage.data_ptr()
    with _lock:
        _mapped[ptr] = nbytes
    weakref.finalize(storage, _forget, ptr).atexit = False
    return raw[:nbytes].view(dtype).view(shape)


def reserve_blocks(fd: int, folder: Path, nbytes: int) -> None:
    """Give the file its disk blocks now. ``ftruncate`` alone makes a sparse file: on a disk that cannot hold it, the
    failure would come later, as a write into the mapping, and kill the process (SIGBUS) instead of raising here.
    ``posix_fallocate`` raises ``OSError`` (ENOSPC) at once; where it does not exist (macOS), the free space is checked."""
    import errno
    import shutil

    if hasattr(os, "posix_fallocate"):
        os.posix_fallocate(fd, 0, nbytes)
        return
    if shutil.disk_usage(folder).free < nbytes + (1 << 30):
        raise OSError(errno.ENOSPC, f"{nbytes} bytes do not fit in {folder}")


def _forget(ptr: int) -> None:
    with _lock:
        _mapped.pop(ptr, None)


def is_file_backed(t: Any) -> bool:
    """Whether ``t``'s memory is a ``scratch`` mapping (the tensor or a view of it)."""
    try:
        ptr = t.untyped_storage().data_ptr()
    except (AttributeError, RuntimeError):
        return False
    with _lock:
        return ptr in _mapped


def mapped_bytes() -> int:
    """Bytes of the ``scratch`` mappings alive in this process."""
    with _lock:
        return sum(_mapped.values())


@dataclass(frozen=True)
class Plan:
    """How many of a model's equal blocks of host weights stay in RAM, the rest in files."""

    in_ram: int
    in_files: int
    block_bytes: int
    ram_available: int | None
    budget: int | None

    @property
    def file_bytes(self) -> int:
        return self.in_files * self.block_bytes


def plan(blocks: int, block_bytes: int, other_bytes: int, ram_available: int | None, reserve: int, margin: int = 2 << 30) -> Plan:
    """Keep ``blocks`` blocks of ``block_bytes`` in RAM as long as they fit beside ``other_bytes`` (the model's other
    weights, in RAM while it loads) within ``ram_available`` less ``reserve`` (``compute.ram_reserve_mb``) and
    ``margin`` (the process and the request's own memory); the rest go to files. All in RAM when the available RAM is
    unknown or file mappings are not ``available()`` here."""
    if ram_available is None or not available() or blocks <= 0 or block_bytes <= 0:
        return Plan(blocks, 0, block_bytes, ram_available, None)
    budget = ram_available - reserve - margin - other_bytes
    in_ram = max(0, min(blocks, budget // block_bytes))
    return Plan(in_ram, blocks - in_ram, block_bytes, ram_available, budget)
