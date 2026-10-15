"""Raw request bodies: streamed to a temporary file, never parsed as multipart."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

from fastapi import Request
from starlette.concurrency import run_in_threadpool

from mizuoto.core.clock import new_id

CHUNK = 1024 * 1024
RAW_BODY = {"requestBody": {"content": {"application/octet-stream": {"schema": {"type": "string", "format": "binary"}}}, "required": True}}


async def spool(request: Request, folder: Path) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{new_id('in')}.part"
    f = path.open("wb")
    try:
        async for chunk in request.stream():
            if chunk:
                await run_in_threadpool(f.write, chunk)
    except BaseException:
        f.close()
        path.unlink(missing_ok=True)
        raise
    f.close()
    return path


def read_chunks(path: Path) -> Iterator[bytes]:
    with open(path, "rb") as f:
        while True:
            block = f.read(CHUNK)
            if not block:
                return
            yield block
