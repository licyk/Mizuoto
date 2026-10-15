"""Key/value caches for decoder layers."""

from __future__ import annotations

import torch


class DynamicCache:
    """Per-layer keys and values that grow by concatenation, as transformers' ``DynamicCache``.

    Concatenating, rather than writing into a preallocated buffer, keeps every attention call the same
    shape as transformers' own, which bit-exact parity needs. A static cache (fixed shapes for CUDA
    graphs) is the accelerated alternative.
    """

    def __init__(self) -> None:
        self.keys: list[torch.Tensor | None] = []
        self.values: list[torch.Tensor | None] = []

    def update(self, layer: int, key: torch.Tensor, value: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        while len(self.keys) <= layer:
            self.keys.append(None)
            self.values.append(None)
        k, v = self.keys[layer], self.values[layer]
        if k is None or v is None:
            self.keys[layer], self.values[layer] = key, value
        else:
            self.keys[layer] = torch.cat([k, key], dim=-2)
            self.values[layer] = torch.cat([v, value], dim=-2)
        out_k, out_v = self.keys[layer], self.values[layer]
        assert out_k is not None and out_v is not None
        return out_k, out_v

    @property
    def length(self) -> int:
        """Positions already cached (0 before the first update)."""
        k = self.keys[0] if self.keys else None
        return 0 if k is None else int(k.shape[-2])

    def select(self, index: torch.Tensor) -> None:
        """Keep the batch rows ``index`` (in that order): beams that survived, sequences still running."""
        self.keys = [None if k is None else k.index_select(0, index) for k in self.keys]
        self.values = [None if v is None else v.index_select(0, index) for v in self.values]


class BufferCache(DynamicCache):
    """``DynamicCache`` over preallocated buffers (``compute.kv_cache = static``): each layer's
    keys and values go into a buffer that doubles when full, and attention reads a view of its filled part. No
    per-step concatenation (no transient copy of the whole cache, no fragmentation); the views have the buffer's
    strides, which some kernels sum in another order, so it is opt-in until checked identical for an engine."""

    def __init__(self, capacity: int = 256) -> None:
        super().__init__()
        self.capacity = capacity
        self._buffers: list[tuple[torch.Tensor, torch.Tensor] | None] = []
        self._filled: list[int] = []

    def update(self, layer: int, key: torch.Tensor, value: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        while len(self._buffers) <= layer:
            self._buffers.append(None)
            self._filled.append(0)
            self.keys.append(None)
            self.values.append(None)
        n = self._filled[layer]
        new = key.shape[-2]
        buf = self._buffers[layer]
        if buf is None or n + new > buf[0].shape[-2]:
            size = max(self.capacity, 1 << max(0, (n + new - 1).bit_length()))
            shape = (*key.shape[:-2], size, key.shape[-1])
            grown = (torch.empty(shape, dtype=key.dtype, device=key.device), torch.empty((*value.shape[:-2], size, value.shape[-1]), dtype=value.dtype, device=value.device))
            if buf is not None and n:
                grown[0][..., :n, :].copy_(buf[0][..., :n, :])
                grown[1][..., :n, :].copy_(buf[1][..., :n, :])
            buf = self._buffers[layer] = grown
        buf[0][..., n : n + new, :].copy_(key)
        buf[1][..., n : n + new, :].copy_(value)
        self._filled[layer] = n + new
        k, v = buf[0][..., : n + new, :], buf[1][..., : n + new, :]
        self.keys[layer], self.values[layer] = k, v
        return k, v

    def select(self, index: torch.Tensor) -> None:
        self._buffers = [None if b is None else (b[0].index_select(0, index), b[1].index_select(0, index)) for b in self._buffers]
        self.keys = [None if b is None else b[0][..., :n, :] for b, n in zip(self._buffers, self._filled, strict=True)]
        self.values = [None if b is None else b[1][..., :n, :] for b, n in zip(self._buffers, self._filled, strict=True)]


def peak_bytes(position_bytes: int, layers: int, positions: int, kind: str = "dynamic", rows: int = 1) -> int:
    """The most device memory a cache of ``layers`` layers holds on its way to ``positions`` positions (for working-memory
    estimates), ``position_bytes`` being one layer's keys and values for one position of one row.

    - ``dynamic``: every layer's keys and values, and one layer's new copy while a step concatenates (each layer's old
      tensors are freed before the next layer's step), not twice the whole cache;
    - ``static`` (``BufferCache``): buffers doubled to a power of two, at least its capacity, and one layer's old buffer
      while it grows.
    """
    layer = position_bytes * rows
    if kind == "static":
        size = max(256, 1 << max(0, (positions - 1).bit_length()))
        return layer * size * layers + layer * size // 2
    return layer * positions * (layers + 1)


def new_cache(kind: str = "dynamic") -> DynamicCache:
    """The cache for ``compute.kv_cache``: ``dynamic`` (transformers' concatenation, the default) or ``static``."""
    return BufferCache() if kind == "static" else DynamicCache()
