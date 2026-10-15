"""Outputs: every synthesized file, with the model, voice, text and parameters that made it."""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from mizuoto.core.clock import new_id, now_iso
from mizuoto.core.db import Database
from mizuoto.core.errors import NotFoundError
from mizuoto.core.events import EventBus
from mizuoto.core.events.models import OutputsAddedEvent, OutputsRemovedEvent
from mizuoto.core.files import slugify, trash
from mizuoto.core.outputs.models import Output, OutputPage, OutputQuery
from mizuoto.core.settings import SettingsService


class OutputService:
    def __init__(self, settings: SettingsService, db: Database, events: EventBus) -> None:
        self._settings = settings
        self._db = db
        self._events = events

    def new_path(self, name_parts: dict[str, str], fmt: str) -> Path:
        """A fresh file under ``outputs/<yyyy-mm>/`` named by ``outputs.naming``, never overwriting."""
        stamp = now_iso()
        parts = {"date": stamp[:19].replace("-", "").replace(":", "").replace("T", "-"), **{k: slugify(v, k) for k, v in name_parts.items()}}
        pattern = self._settings.settings.outputs.naming
        try:
            stem = pattern.format_map(_Missing(parts))
        except (ValueError, IndexError):
            stem = "{date}".format_map(parts)
        stem = slugify(stem, "output")[:96]
        folder = self._settings.outputs_dir / stamp[:7]
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / f"{stem}.{fmt}"
        n = 1
        while path.exists():
            path = folder / f"{stem}-{n}.{fmt}"
            n += 1
        return path

    def add(self, output: Output, job_id: str | None = None) -> Output:
        self._db.execute(
            "INSERT INTO outputs(id, kind, job_id, path, name, engine, model, voice_id, text, language, params, sample_rate, duration, format, size, licence_tags, meta, favourite, created_at)"
            " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                output.id,
                output.kind,
                output.job_id,
                output.path,
                output.name,
                output.engine,
                output.model,
                output.voice_id,
                output.text,
                output.language,
                json.dumps(output.params),
                output.sample_rate,
                output.duration,
                output.format,
                output.size,
                json.dumps(output.licence_tags),
                json.dumps(output.meta),
                int(output.favourite),
                output.created_at,
            ),
        )
        self._events.publish(OutputsAddedEvent(job_id=job_id or output.job_id, ids=[output.id]))
        return output

    @staticmethod
    def new_id() -> str:
        return new_id("o")

    def get(self, output_id: str) -> Output:
        row = self._db.fetchone("SELECT * FROM outputs WHERE id = ?", (output_id,))
        if row is None:
            raise NotFoundError(f"Unknown output: {output_id}", {"kind": "output", "id": output_id})
        return self._row(row)

    def path_of(self, output_id: str) -> Path:
        out = self.get(output_id)
        path = Path(out.path)
        if not path.is_file():
            raise NotFoundError(f"{out.name} is gone from {path.parent}", {"kind": "output", "id": output_id})
        return path

    def list_outputs(self, query: OutputQuery | None = None) -> OutputPage:
        q = query or OutputQuery()
        where: list[str] = []
        params: list[Any] = []
        for column, value in (("voice_id", q.voice_id), ("model", q.model), ("engine", q.engine), ("kind", q.kind), ("job_id", q.job_id)):
            if value is not None:
                where.append(f"{column} = ?")
                params.append(value)
        if q.kind is None and q.job_id is None:
            where.append("kind != 'line'")
        if q.favourite is not None:
            where.append("favourite = ?")
            params.append(int(q.favourite))
        if q.text:
            where.append("(text LIKE ? OR name LIKE ?)")
            params += [f"%{q.text}%", f"%{q.text}%"]
        if q.cursor:
            created, _, oid = q.cursor.partition("|")
            where.append("(created_at < ? OR (created_at = ? AND id < ?))")
            params += [created, created, oid]
        sql = "SELECT * FROM outputs" + (" WHERE " + " AND ".join(where) if where else "") + " ORDER BY created_at DESC, id DESC LIMIT ?"
        rows = self._db.fetchall(sql, (*params, q.limit + 1))
        items = [self._row(r) for r in rows[: q.limit]]
        next_cursor = f"{rows[q.limit - 1]['created_at']}|{rows[q.limit - 1]['id']}" if len(rows) > q.limit else None
        return OutputPage(items=items, next_cursor=next_cursor)

    def update(self, output_id: str, name: str | None = None, favourite: bool | None = None) -> Output:
        self.get(output_id)
        if name is not None:
            self._db.execute("UPDATE outputs SET name = ? WHERE id = ?", (name, output_id))
        if favourite is not None:
            self._db.execute("UPDATE outputs SET favourite = ? WHERE id = ?", (int(favourite), output_id))
        return self.get(output_id)

    def delete(self, ids: list[str]) -> list[str]:
        removed: list[str] = []
        for output_id in ids:
            try:
                out = self.get(output_id)
            except NotFoundError:
                continue
            trash(Path(out.path), self._settings.data_dir / "trash")
            self._db.execute("DELETE FROM outputs WHERE id = ?", (output_id,))
            removed.append(output_id)
        if removed:
            self._events.publish(OutputsRemovedEvent(ids=removed))
        return removed

    def zip(self, ids: list[str]) -> Iterator[bytes]:
        """A zip of outputs, streamed as it is written (names made unique)."""
        import io
        import zipfile

        outs = [self.get(i) for i in ids]

        class _Sink(io.RawIOBase):
            def __init__(self) -> None:
                self.chunks: list[bytes] = []

            def writable(self) -> bool:
                return True

            def write(self, b: Any) -> int:
                self.chunks.append(bytes(b))
                return len(b)

        sink = _Sink()
        used: set[str] = set()
        with zipfile.ZipFile(sink, "w", compression=zipfile.ZIP_STORED) as z:
            for out in outs:
                path = Path(out.path)
                name = f"{out.name}{path.suffix}"
                n = 2
                while name in used:
                    name, n = f"{out.name}-{n}{path.suffix}", n + 1
                used.add(name)
                if path.is_file():
                    z.write(path, name)
                yield from sink.chunks
                sink.chunks.clear()
        yield from sink.chunks

    @staticmethod
    def _row(r: Any) -> Output:
        return Output(
            id=r["id"],
            kind=r["kind"],
            name=r["name"],
            path=r["path"],
            format=r["format"] or "",
            sample_rate=r["sample_rate"] or 0,
            duration=r["duration"] or 0.0,
            size=r["size"] or 0,
            job_id=r["job_id"],
            engine=r["engine"],
            model=r["model"],
            voice_id=r["voice_id"],
            text=r["text"],
            language=r["language"],
            params=json.loads(r["params"] or "{}"),
            meta=json.loads(r["meta"] or "{}"),
            licence_tags=json.loads(r["licence_tags"] or "[]"),
            favourite=bool(r["favourite"]),
            created_at=r["created_at"],
        )


class _Missing(dict):
    """``str.format_map`` that leaves unknown fields empty."""

    def __missing__(self, key: str) -> str:
        return ""
