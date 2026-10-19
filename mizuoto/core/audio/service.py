"""Audio files given to Mizuoto: references, emotion clips, sources.

Each file is identified by its content (SHA-256). Engines get a canonical copy, mono WAV at the
file's own rate under ``cache/refs/<sha256>.wav``, so every engine can read it, whatever it was
uploaded as. Uploads through the API arrive with milestone M3.
"""

from __future__ import annotations

import hashlib
import re
import shutil
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from pydantic import Field

from mizuoto.core.audio.io import decode, encode, probe, sha256_file
from mizuoto.core.clock import new_id, now_iso
from mizuoto.core.db import Database
from mizuoto.core.errors import InvalidPathError, NotFoundError, ValidationError
from mizuoto.core.files import is_within
from mizuoto.core.record import Record
from mizuoto.core.settings import SettingsService
from mizuoto.core.storage import StorageService


class AudioFile(Record):
    id: str
    name: str
    path: str
    sha256: str
    origin: str
    sample_rate: int
    channels: int
    duration: float
    format: str
    size: int
    created_at: str
    missing: bool = False
    """The file is gone: neither at ``path`` nor anywhere it could have moved to."""


class Peaks(Record):
    """Waveform peaks: ``points`` (min, max) pairs over the whole file, mixed to mono."""

    points: int
    duration: float
    sample_rate: int
    data: list[float]
    """Interleaved min and max, each in -1..1."""


class BrowseEntry(Record):
    name: str
    path: str
    is_dir: bool
    is_audio: bool = False
    size: int | None = None
    mtime: float | None = None


class BrowseListing(Record):
    root: str
    path: str
    parent: str | None
    entries: list[BrowseEntry]
    roots: list[str] = Field(default_factory=list)


AUDIO_EXTENSIONS = {".wav", ".flac", ".mp3", ".ogg", ".opus", ".m4a", ".aac", ".webm", ".wma", ".aiff", ".aif"}
MIN_POINTS = 16
MAX_POINTS = 8192
UPLOAD_NAME = re.compile(r"[^\w.\- ()\[\]]+", re.UNICODE)


class AudioService:
    """Audio file records keep their path relative to the data folder when the file lies inside it (``core/storage``);
    records hand it out absolute, as it is on disk now."""

    def __init__(self, settings: SettingsService, db: Database, storage: StorageService) -> None:
        self._settings = settings
        self._db = db
        self._storage = storage

    @property
    def refs_dir(self) -> Path:
        return self._settings.cache_dir / "refs"

    @property
    def uploads_dir(self) -> Path:
        return self._settings.data_dir / "uploads"

    def upload(self, name: str, chunks: Iterable[bytes]) -> AudioFile:
        """Stream an uploaded body to ``uploads/<id>/<name>``, check it is audio, and record it."""
        clean = UPLOAD_NAME.sub("_", Path(name.replace("\\", "/")).name).strip(" .") or "audio"
        folder = self.uploads_dir / new_id("u")
        folder.mkdir(parents=True, exist_ok=True)
        target = folder / clean
        try:
            with open(target, "wb") as f:
                for chunk in chunks:
                    f.write(chunk)
            if target.stat().st_size == 0:
                raise ValidationError("The upload is empty")
            try:
                probe(target)
            except Exception as e:
                raise ValidationError(f"{clean} is not an audio file Mizuoto can read", {"reason": "audio"}) from e
            return self.register_path(target, origin="upload")
        except BaseException:
            shutil.rmtree(folder, ignore_errors=True)
            raise

    def path_of(self, audio_id: str) -> Path:
        """The file's path; ``not_found`` with ``detail.reason`` ``file_missing`` when it is gone."""
        row = self._fetch(audio_id)
        path = self._storage.find("audio", row)
        if path is None:
            raise self._storage.missing_error("audio", row)
        return path

    # -- server paths (the API checks them; the command line is trusted) ----------------

    def roots(self) -> list[Path]:
        return self._settings.browse_roots()

    def check_root(self, path: Path) -> Path:
        """``path``, which must lie inside ``paths.browse_roots``."""
        path = path.expanduser()
        if not is_within(path, self.roots()):
            raise InvalidPathError(f"{path} is outside the folders the server may browse", {"roots": [str(r) for r in self.roots()]})
        return path

    def browse(self, path: str | None = None, dirs_only: bool = False) -> BrowseListing:
        """A folder's sub-folders and audio files (all files' folders with ``dirs_only``)."""
        roots = self.roots()
        if not path:
            entries = [BrowseEntry(name=str(r), path=str(r), is_dir=True) for r in roots if r.is_dir()]
            return BrowseListing(root="", path="", parent=None, entries=entries, roots=[str(r) for r in roots])
        target = self.check_root(Path(path))
        if not target.is_dir():
            raise NotFoundError(f"No such folder: {target}", {"kind": "folder", "id": str(target)})
        resolved = target.resolve()
        root = next((r for r in roots if resolved == r or r in resolved.parents), roots[0])
        entries: list[BrowseEntry] = []
        try:
            children = sorted(target.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower()))
        except PermissionError as e:
            raise InvalidPathError(f"Cannot read {target}: permission denied") from e
        for child in children:
            if child.name.startswith("."):
                continue
            try:
                is_dir = child.is_dir()
                st = child.stat()
            except OSError:
                continue
            is_audio = not is_dir and child.suffix.lower() in AUDIO_EXTENSIONS
            if not is_dir and (dirs_only or not is_audio):
                continue
            entries.append(BrowseEntry(name=child.name, path=str(child), is_dir=is_dir, is_audio=is_audio, size=None if is_dir else st.st_size, mtime=st.st_mtime))
        parent = None if resolved == root else str(resolved.parent)
        return BrowseListing(root=str(root), path=str(resolved), parent=parent, entries=entries, roots=[str(r) for r in roots])

    def peaks(self, path: Path, points: int = 1024) -> Peaks:
        """Waveform peaks of a file, cached by path, size and modification time."""
        from mizuoto.core.audio.dsp import peaks as compute

        points = max(MIN_POINTS, min(MAX_POINTS, int(points)))
        st = path.stat()
        # Keyed by the path as a record keeps it, so the cache follows the data folder when it moves.
        key = hashlib.sha1(f"{self._storage.store(path)}|{st.st_size}|{st.st_mtime_ns}|{points}".encode()).hexdigest()
        cache = self._settings.cache_dir / "peaks" / key[:2] / f"{key}.json"
        if cache.is_file():
            try:
                return Peaks.model_validate_json(cache.read_text(encoding="utf-8"))
            except ValueError:
                pass
        samples, sr = decode(path)
        result = Peaks(points=points, duration=len(samples) / sr if sr else 0.0, sample_rate=sr, data=compute(samples, points))
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(result.model_dump_json(), encoding="utf-8")
        return result

    def register_path(self, path: Path, origin: str = "path") -> AudioFile:
        """A file on this machine, recorded by content; the same content is recorded once."""
        path = path.expanduser().resolve()
        if not path.is_file():
            raise NotFoundError(f"No such file: {path}", {"kind": "file", "id": str(path)})
        info = probe(path)
        sha = sha256_file(path)
        stored = self._storage.store(path)
        row = self._db.fetchone("SELECT * FROM audio_files WHERE sha256 = ? AND path = ?", (sha, stored))
        if row is not None:
            return self._row(row)
        size = path.stat().st_size
        # The same file recorded where it no longer is (a data folder moved before its records were rewritten) is
        # pointed here rather than recorded twice.
        for row in self._db.fetchall("SELECT * FROM audio_files WHERE sha256 = ? AND size = ?", (sha, size)):
            if not self._storage.resolve(row["path"]).is_file():
                self._db.execute("UPDATE audio_files SET path = ? WHERE id = ?", (stored, row["id"]))
                return self.get(row["id"])
        duration = info.duration
        if duration is None:
            samples, sr = decode(path)
            duration = len(samples) / sr if sr else 0.0
        record = AudioFile(
            id=new_id("a"),
            name=path.name,
            path=str(path),
            sha256=sha,
            origin=origin,
            sample_rate=info.sample_rate,
            channels=info.channels,
            duration=float(duration),
            format=info.codec,
            size=size,
            created_at=now_iso(),
        )
        self._db.execute(
            "INSERT INTO audio_files(id, name, path, sha256, origin, sample_rate, channels, duration, format, size, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                record.id,
                record.name,
                stored,
                record.sha256,
                record.origin,
                record.sample_rate,
                record.channels,
                record.duration,
                record.format,
                record.size,
                record.created_at,
            ),
        )
        return record

    def _fetch(self, audio_id: str) -> Any:
        row = self._db.fetchone("SELECT * FROM audio_files WHERE id = ?", (audio_id,))
        if row is None:
            raise NotFoundError(f"Unknown audio file: {audio_id}", {"kind": "audio", "id": audio_id})
        return row

    def get(self, audio_id: str) -> AudioFile:
        return self._row(self._fetch(audio_id))

    def canonical_wav(self, audio: AudioFile) -> Path:
        """Mono 16-bit WAV at the file's own rate, cached by content."""
        target = self.refs_dir / f"{audio.sha256}.wav"
        if not target.is_file():
            samples, sr = decode(self.path_of(audio.id))
            encode(samples, sr, target, "wav")
        return target

    def trimmed(self, audio: AudioFile, max_seconds: float) -> AudioFile:
        """A copy of ``audio`` cut to ``max_seconds`` at a quiet point (a reference too long for a model)."""
        from mizuoto.core.audio.dsp import trim_to

        target = self.refs_dir / f"{audio.sha256}-max{max_seconds:g}.wav"
        if not target.is_file():
            samples, sr = decode(self.path_of(audio.id))
            encode(trim_to(samples, sr, max_seconds), sr, target, "wav")
        return self.register_path(target, origin="derived")

    def _row(self, r: Any) -> AudioFile:
        path = self._storage.find("audio", r)
        return AudioFile(
            id=r["id"],
            name=r["name"],
            path=str(path or self._storage.resolve(r["path"])),
            missing=path is None,
            sha256=r["sha256"],
            origin=r["origin"],
            sample_rate=r["sample_rate"] or 0,
            channels=r["channels"] or 1,
            duration=r["duration"] or 0.0,
            format=r["format"] or "",
            size=r["size"],
            created_at=r["created_at"],
        )
