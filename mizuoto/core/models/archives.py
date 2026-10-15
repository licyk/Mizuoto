"""Unpacking downloaded archives (zip, tar, tar.gz) safely: every member must land inside the target."""

from __future__ import annotations

import shutil
import tarfile
import zipfile
from pathlib import Path

from mizuoto.core.errors import DownloadError

ARCHIVE_SUFFIXES = (".zip", ".tar", ".tar.gz", ".tgz", ".tar.bz2", ".tar.xz")


def is_archive(name: str) -> bool:
    return name.lower().endswith(ARCHIVE_SUFFIXES)


def _inside(root: Path, name: str) -> Path:
    target = (root / name).resolve()
    if target != root and root not in target.parents:
        raise DownloadError(f"The archive member {name!r} would land outside {root}", {"member": name})
    return target


def unpack(archive: Path, dest: Path) -> list[Path]:
    """Extract ``archive`` into ``dest``; the regular files written, in order. Links and devices are refused."""
    dest.mkdir(parents=True, exist_ok=True)
    root = dest.resolve()
    written: list[Path] = []
    if archive.name.lower().endswith(".zip"):
        with zipfile.ZipFile(archive) as z:
            for info in z.infolist():
                target = _inside(root, info.filename)
                if info.is_dir():
                    target.mkdir(parents=True, exist_ok=True)
                    continue
                target.parent.mkdir(parents=True, exist_ok=True)
                with z.open(info) as src, open(target, "wb") as out:
                    shutil.copyfileobj(src, out, 1 << 20)
                written.append(target)
        return written
    with tarfile.open(archive) as t:
        for member in t.getmembers():
            target = _inside(root, member.name)
            if member.isdir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            if not member.isfile():
                raise DownloadError(f"The archive member {member.name!r} is not a regular file", {"member": member.name})
            target.parent.mkdir(parents=True, exist_ok=True)
            src = t.extractfile(member)
            assert src is not None
            with src, open(target, "wb") as out:
                shutil.copyfileobj(src, out, 1 << 20)
            written.append(target)
    return written
