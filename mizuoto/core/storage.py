"""Where the database's file paths point, and finding a file again after the data folder moved.

``outputs.path`` and ``audio_files.path`` record a file inside the data folder relative to it
(``outputs/2026-10/a.wav``), so the records follow the folder when it is moved, renamed, copied or opened through a
link; a file elsewhere (a server path, an outputs folder set outside the data folder) is recorded absolute.

Databases from before kept absolute paths everywhere. ``reconcile`` (run whenever the data folder is opened at a
place other than the last one) rewrites the records of files inside the folder as relative ones, and ``find`` looks
for a file that is not where its record says: under the data folder and the outputs and cache folders, by the last
parts of the recorded path. A candidate counts only with the recorded size, and content hash where the row keeps
one, and a record found again is rewritten. A file found nowhere is reported missing, never skipped.
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from pydantic import Field

from mizuoto.core import codes
from mizuoto.core.db import Database
from mizuoto.core.errors import NotFoundError, ValidationError
from mizuoto.core.events import EventBus
from mizuoto.core.events.models import OutputsRemovedEvent
from mizuoto.core.files import from_stored, is_absolute_text, relocations, to_stored
from mizuoto.core.record import Record
from mizuoto.core.settings import SettingsService

logger = logging.getLogger(__name__)

FileKind = Literal["output", "audio"]

DATA_DIR_KEY = "data_dir"
"""``meta`` key: where the data folder was opened last, as ``SettingsService.data_dir`` gave it."""


@dataclass(frozen=True)
class _Table:
    name: str
    hashed: bool
    """Whether rows keep the file's SHA-256, which a found file must then match."""


TABLES: dict[FileKind, _Table] = {"output": _Table("outputs", hashed=False), "audio": _Table("audio_files", hashed=True)}


class MissingFile(Record):
    kind: FileKind
    id: str
    name: str
    recorded: str
    """The path as the database records it."""


class PathCheckReport(Record):
    """What ``StorageService.check`` found among the recorded files."""

    data_dir: str
    relative: int = 0
    """Files inside the data folder, where their records say (recorded relative to it, or absolute the old way until
    ``fixed``)."""
    external: int = 0
    """Files outside the data folder, where their records say."""
    relocated: int = 0
    """Files found away from their recorded place (their records rewritten when ``fixed``)."""
    rebased: int = 0
    """Records whose folder was replaced by ``rebase``."""
    missing: list[MissingFile] = Field(default_factory=list)
    """Files found nowhere."""
    pruned: int = 0
    """Records of missing files deleted (``prune_missing``)."""
    fixed: bool = False
    """Whether the records were rewritten, or only checked."""


class PathCheckRequest(Record):
    fix: bool = False
    """Rewrite records: relative for files inside the data folder, the new place for files found elsewhere."""
    prune_missing: bool = False
    """Delete the records of files found nowhere."""
    rebase_from: str | None = None
    """With ``rebase_to``: records under this folder point under ``rebase_to`` instead (an outputs folder that moved
    with its disk)."""
    rebase_to: str | None = None


class StorageService:
    def __init__(self, settings: SettingsService, db: Database) -> None:
        self._settings = settings
        self._db = db
        self.events: EventBus | None = None

    @property
    def base(self) -> Path:
        return self._settings.data_dir

    def store(self, path: Path) -> str:
        """``path`` as a record keeps it: relative inside the data folder, absolute elsewhere."""
        return to_stored(path, self.base)

    def resolve(self, value: str) -> Path:
        """A recorded path on disk, where the record says (whether or not the file is there)."""
        return from_stored(value, self.base)

    def roots(self) -> list[Path]:
        """Folders a recorded file may be found under: the data folder, and the outputs and cache folders when they are
        set elsewhere."""
        found: list[Path] = []
        for root in (self._settings.data_dir, self._settings.outputs_dir, self._settings.cache_dir):
            if root not in found:
                found.append(root)
        return found

    def relocate(self, value: str, size: int | None, sha256: str | None = None, roots: list[Path] | None = None) -> Path | None:
        """The file recorded at ``value``, found by the last parts of its path under ``roots`` (default: ``roots()``)
        with the recorded size and hash; None when none matches."""
        recorded = self.resolve(value)
        for candidate in relocations(value, self.roots() if roots is None else roots):
            if candidate != recorded and _matches(candidate, size, sha256):
                return candidate
        return None

    def find(self, kind: FileKind, row: Any) -> Path | None:
        """A row's file: where its record says, else found again (and the record rewritten); None when missing."""
        path = self.resolve(row["path"])
        if path.is_file():
            return path
        found = self.relocate(row["path"], row["size"], row["sha256"] if TABLES[kind].hashed else None)
        if found is not None:
            stored = self.store(found)
            self._db.execute(f"UPDATE {TABLES[kind].name} SET path = ? WHERE id = ?", (stored, row["id"]))
            logger.info("Found %s again at %s (it was recorded at %s)", row["name"], found, row["path"])
        return found

    def missing_error(self, kind: FileKind, row: Any) -> NotFoundError:
        """The error for a row whose file is found nowhere."""
        path = self.resolve(row["path"])
        return NotFoundError(
            f"{row['name']} is missing: it is not at {path}, nor under the data folder ({self.base})",
            {"kind": kind, "id": row["id"], "reason": codes.FILE_MISSING, "path": str(path), "recorded": row["path"], "hint": "mizuoto output check"},
        )

    # -- the whole database --------------------------------------------------------------

    def reconcile(self) -> PathCheckReport | None:
        """Run ``check(fix=True)`` when the data folder is opened at a new place, or for the first time since records
        became relative; None when it was opened here last. Safe while other processes open the same folder."""
        here = str(self.base)
        if self._db.get_meta(DATA_DIR_KEY) == here:
            return None
        with self._db.transaction(immediate=True):
            last = self._db.get_meta(DATA_DIR_KEY)
            if last == here:  # another process opening the folder did it meanwhile
                return None
            report = self._scan(fix=True, prune=False)
            self._db.set_meta(DATA_DIR_KEY, here)
        if last is not None:
            logger.info("The data folder is at %s now (it was at %s)", here, last)
        if report.relocated or report.missing:
            logger.info(
                "Recorded files: %d found again, %d missing%s",
                report.relocated,
                len(report.missing),
                " (mizuoto output check lists them)" if report.missing else "",
            )
        return report

    def check(self, request: PathCheckRequest | None = None) -> PathCheckReport:
        """Every recorded output and audio file: where it is, rewritten (``fix``), moved (``rebase_*``) or pruned."""
        req = request or PathCheckRequest()
        if (req.rebase_from is None) != (req.rebase_to is None):
            raise ValidationError("A rebase needs both the old folder and the new one")
        writes = req.fix or req.prune_missing or req.rebase_from is not None
        if not writes:
            return self._scan(fix=False, prune=False)
        with self._db.transaction(immediate=True):
            rebased = self._rebase(req.rebase_from, req.rebase_to) if req.rebase_from and req.rebase_to else 0
            report = self._scan(fix=req.fix or rebased > 0, prune=req.prune_missing)
            report.rebased = rebased
        pruned_outputs = [m.id for m in report.missing if m.kind == "output"] if req.prune_missing else []
        if pruned_outputs and self.events is not None:
            self.events.publish(OutputsRemovedEvent(ids=pruned_outputs))
        return report

    def _rebase(self, old: str, new: str) -> int:
        """Point the absolute records under ``old`` under ``new``."""
        old = old.rstrip("/\\")
        count = 0
        for table in TABLES.values():
            for row in self._db.fetchall(f"SELECT id, path FROM {table.name}"):
                value = str(row["path"])
                if is_absolute_text(value) and value.startswith(old) and value[len(old) : len(old) + 1] in ("", "/", "\\"):
                    moved = Path(new).joinpath(*[p for p in value[len(old) :].replace("\\", "/").split("/") if p])
                    self._db.execute(f"UPDATE {table.name} SET path = ? WHERE id = ?", (self.store(moved), row["id"]))
                    count += 1
        return count

    def _scan(self, fix: bool, prune: bool) -> PathCheckReport:
        report = PathCheckReport(data_dir=str(self.base), fixed=fix)
        for kind, table in TABLES.items():
            for row in self._db.fetchall(f"SELECT * FROM {table.name}"):
                value = str(row["path"])
                sha = row["sha256"] if table.hashed else None
                found: Path | None = None
                if not is_absolute_text(value):
                    if self.resolve(value).is_file():
                        report.relative += 1
                        continue
                    found = self.relocate(value, row["size"], sha)
                else:
                    stored = self.store(Path(value))
                    if not is_absolute_text(stored):  # inside the data folder: recorded the old way
                        if fix:
                            self._db.execute(f"UPDATE {table.name} SET path = ? WHERE id = ?", (stored, row["id"]))
                        if self.resolve(stored).is_file():
                            report.relative += 1
                            continue
                        found = self.relocate(stored, row["size"], sha)
                    else:
                        # A copy inside this data folder wins over the file it was copied from: a copied data folder
                        # must not keep reading, or deleting, the original's files.
                        found = self.relocate(value, row["size"], sha, roots=[self.base])
                        if found is None and Path(value).is_file():
                            report.external += 1
                            continue
                        if found is None:
                            found = self.relocate(value, row["size"], sha)
                if found is not None:
                    report.relocated += 1
                    if fix:
                        self._db.execute(f"UPDATE {table.name} SET path = ? WHERE id = ?", (self.store(found), row["id"]))
                    continue
                report.missing.append(MissingFile(kind=kind, id=row["id"], name=row["name"], recorded=value))
                if prune:
                    self._db.execute(f"DELETE FROM {table.name} WHERE id = ?", (row["id"],))
                    report.pruned += 1
        return report


def _matches(path: Path, size: int | None, sha256: str | None) -> bool:
    """Whether ``path`` is a file of the recorded size and content."""
    try:
        st = path.stat()
    except OSError:
        return False
    if not path.is_file() or (size is not None and st.st_size != size):
        return False
    if not sha256:
        return True
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest() == sha256
