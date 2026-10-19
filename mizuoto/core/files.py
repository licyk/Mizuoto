"""File helpers: trash, names and roots."""

from __future__ import annotations

import getpass
import logging
import os
import re
import shutil
import sys
import unicodedata
from collections.abc import Callable
from pathlib import Path, PurePosixPath, PureWindowsPath

from mizuoto.core.clock import new_id

logger = logging.getLogger(__name__)


def remove_path(path: Path) -> None:
    """Delete a file, a symlink or a folder tree."""
    if path.is_symlink() or path.is_file():
        path.unlink(missing_ok=True)
    elif path.is_dir():
        shutil.rmtree(path)


def trash(path: Path, fallback_dir: Path) -> None:
    """Move to the operating system's trash; without one, into ``fallback_dir``."""
    if not path.exists() and not path.is_symlink():
        return
    try:
        from send2trash import send2trash

        send2trash(str(path))
        return
    except Exception as e:
        logger.info("OS trash unavailable for %s (%s); using %s", path.name, e, fallback_dir)
    fallback_dir.mkdir(parents=True, exist_ok=True)
    try:
        shutil.move(str(path), str(fallback_dir / f"{new_id()}-{path.name}"))
    except Exception:
        remove_path(path)


def slugify(name: str, fallback: str = "item") -> str:
    """A folder-safe id: lower case letters, digits and dashes, keeping CJK and other letters."""
    text = unicodedata.normalize("NFKC", name).strip().lower()
    text = re.sub(r"[^\w\-]+", "-", text, flags=re.UNICODE)
    text = re.sub(r"[-_]{2,}", "-", text).strip("-_")
    return text[:64] or fallback


def is_within(path: Path, roots: list[Path]) -> bool:
    """Whether ``path`` (resolved) lies inside one of ``roots``."""
    resolved = path.resolve()
    for root in roots:
        r = root.resolve()
        if resolved == r or r in resolved.parents:
            return True
    return False


def is_absolute_text(value: str) -> bool:
    """Whether a recorded path is absolute on any platform (``/x``, ``C:\\x``, ``\\\\server\\share``): a database
    written on Windows can be opened on Linux and the other way round."""
    return PurePosixPath(value).is_absolute() or bool(PureWindowsPath(value).anchor)


def to_stored(path: Path, base: Path) -> str:
    """``path`` as the database records it: relative to ``base`` with ``/`` separators when it lies inside it, so the
    record follows ``base`` when that folder moves, else absolute. ``base`` is resolved; ``path`` is tried as written
    and resolved."""
    for candidate in (path.absolute(), path.resolve()):
        try:
            rel = candidate.relative_to(base)
        except ValueError:
            continue
        if rel.parts:
            return rel.as_posix()
    return str(path)


def from_stored(value: str, base: Path) -> Path:
    """A recorded path back on disk: a relative one under ``base``, an absolute one as it is."""
    return Path(value) if is_absolute_text(value) else base.joinpath(*value.split("/"))


def relocations(value: str, roots: list[Path], min_parts: int = 2) -> list[Path]:
    """Where a file recorded at ``value`` may be now: its last parts (longest first, down to ``min_parts``) under each
    of ``roots``. ``/old/mizuoto/outputs/2026-10/a.wav`` gives ``<root>/outputs/2026-10/a.wav``, then
    ``<root>/2026-10/a.wav``…; either platform's separators are understood."""
    parts = [p for p in re.split(r"[\\/]", value) if p and not p.endswith(":")]
    seen: set[Path] = set()
    found: list[Path] = []
    for root in roots:
        for start in range(len(parts) - min_parts + 1):
            candidate = root.joinpath(*parts[start:])
            if candidate not in seen:
                seen.add(candidate)
                found.append(candidate)
    return found


def windows_drives(listdrives: Callable[[], list[str]] | None = None) -> list[Path]:
    """Every drive letter (C:\\, D:\\, …), read from the drive list without touching the disks."""
    if listdrives is None:
        listdrives = getattr(os, "listdrives", None)  # Python 3.12+
    if listdrives is not None:
        return [Path(d) for d in listdrives()]
    if sys.platform != "win32":
        return []
    import ctypes

    mask = ctypes.windll.kernel32.GetLogicalDrives()
    return [Path(f"{chr(ord('A') + i)}:\\") for i in range(26) if mask >> i & 1]


def volume_roots() -> list[Path]:
    """The machine's drives (Windows) or mounted volumes (macOS, Linux), for the server file browser.

    Volumes are not resolved: resolving a drive opens it, which stalls on a disconnected network drive.
    """
    if sys.platform == "win32":
        try:
            return windows_drives()
        except (OSError, AttributeError) as e:
            logger.warning("Cannot list the drives: %s", e)
            return []
    try:
        user = getpass.getuser()
    except (KeyError, OSError):
        user = ""
    bases = [Path("/Volumes")] if sys.platform == "darwin" else [p for p in (Path("/media") / user, Path("/run/media") / user, Path("/mnt")) if user or p.name == "mnt"]
    roots: list[Path] = []
    for base in bases:
        try:
            children = sorted(base.iterdir())
        except OSError:
            continue
        for child in children:
            try:
                if child.is_dir() and child.resolve() != Path("/"):
                    roots.append(child)
            except OSError:
                continue
    return roots


def dir_size(path: Path) -> int:
    """Total size in bytes of a file or a folder tree."""
    if path.is_file():
        return path.stat().st_size
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file())


def reveal(path: Path) -> None:
    """Show ``path`` in the operating system's file manager."""
    import subprocess
    import sys

    target = path if path.exists() else path.parent
    if sys.platform == "win32":
        subprocess.Popen(["explorer", "/select,", str(target)] if target.is_file() else ["explorer", str(target)])
    elif sys.platform == "darwin":
        subprocess.Popen(["open", "-R", str(target)] if target.is_file() else ["open", str(target)])
    else:
        subprocess.Popen(["xdg-open", str(target.parent if target.is_file() else target)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
