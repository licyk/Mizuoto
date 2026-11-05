"""Host weights kept in files when RAM cannot hold them (``runtime.filemap``)."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from torch import nn  # noqa: E402

from mizuoto.runtime import filemap, pinning  # noqa: E402
from mizuoto.runtime.handle import ModelHandle  # noqa: E402

GB = 1 << 30


@pytest.mark.skipif(not filemap.available(), reason="POSIX only")
def test_a_scratch_tensor_holds_what_is_written_and_leaves_no_file(tmp_path):
    t = filemap.scratch(tmp_path, (4, 3, 8), torch.float8_e4m3fn)
    source = torch.randn(4, 3, 8).to(torch.float8_e4m3fn)
    t.copy_(source)
    assert t.dtype == torch.float8_e4m3fn and tuple(t.shape) == (4, 3, 8)
    assert torch.equal(t.view(torch.uint8), source.view(torch.uint8))
    assert filemap.is_file_backed(t) and filemap.is_file_backed(t[2]) and not filemap.is_file_backed(source)
    assert list(tmp_path.iterdir()) == []  # unlinked as soon as mapped
    assert filemap.mapped_bytes() >= t.numel()
    before = filemap.mapped_bytes()
    del t
    import gc

    gc.collect()
    assert filemap.mapped_bytes() == before - 4 * 3 * 8


def test_the_plan_keeps_in_ram_what_fits_and_files_only_the_rest(monkeypatch):
    monkeypatch.setattr(filemap.sys, "platform", "linux")  # the arithmetic; Windows (no file mappings) below
    plenty = filemap.plan(40, GB, other_bytes=8 * GB, ram_available=100 * GB, reserve=10 * GB)
    assert (plenty.in_ram, plenty.in_files, plenty.file_bytes) == (40, 0, 0)
    short = filemap.plan(40, GB, other_bytes=8 * GB, ram_available=29 * GB, reserve=3 * GB)
    assert (short.in_ram, short.in_files) == (29 - 3 - 2 - 8, 40 - 16)
    assert filemap.plan(40, GB, 8 * GB, None, 3 * GB).in_files == 0  # RAM unknown: as before
    assert filemap.plan(40, GB, 60 * GB, 29 * GB, 3 * GB).in_ram == 0
    monkeypatch.setattr(filemap.sys, "platform", "win32")
    assert not filemap.available() and filemap.plan(40, GB, 8 * GB, 29 * GB, 3 * GB).in_files == 0


@pytest.mark.skipif(not filemap.available(), reason="POSIX only")
def test_a_file_backed_weight_is_never_page_locked_and_counts_as_disk(tmp_path, monkeypatch):
    from mizuoto.runtime.stream import PinBudget

    layer = nn.Linear(256, 256, bias=False)
    mapped = filemap.scratch(tmp_path, tuple(layer.weight.shape), layer.weight.dtype)
    mapped.copy_(layer.weight.data)
    monkeypatch.setattr(pinning, "usable", lambda device: True)
    assert pinning.pin(mapped, PinBudget(1 << 40), "cuda:0") is False
    handle = ModelHandle(nn.Sequential(layer), name="t", load_device="cpu")
    handle._host[(id(layer), "weight")] = mapped
    assert handle.disk_bytes == 256 * 256 * 4 and handle.host_bytes == 0
    handle._host[(id(layer), "weight")] = layer.weight.data
    assert handle.disk_bytes == 0 and handle.host_bytes == 256 * 256 * 4


@pytest.mark.skipif(not filemap.available(), reason="POSIX only")
def test_a_full_disk_raises_at_once_and_leaves_nothing(tmp_path, monkeypatch):
    """The file's blocks are reserved before it is mapped: a disk that cannot hold it raises ``OSError`` here, which the
    engine turns into keeping the weights in RAM, instead of a write into the mapping killing the process later."""
    import errno
    import shutil
    from types import SimpleNamespace

    def full(*args):
        raise OSError(errno.ENOSPC, "No space left on device")

    monkeypatch.setattr(filemap.os, "posix_fallocate", full, raising=False)
    with pytest.raises(OSError) as e:
        filemap.scratch(tmp_path, (1024,), torch.float32)
    assert e.value.errno == errno.ENOSPC
    assert list(tmp_path.iterdir()) == []
    monkeypatch.delattr(filemap.os, "posix_fallocate")  # macOS has none: the free space is checked instead
    monkeypatch.setattr(shutil, "disk_usage", lambda folder: SimpleNamespace(free=10))
    with pytest.raises(OSError):
        filemap.scratch(tmp_path, (1024,), torch.float32)
    assert list(tmp_path.iterdir()) == []
