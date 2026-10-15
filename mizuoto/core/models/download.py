"""Fetching model files: sources, resume, verification.

A file is written to ``<name>.part`` and renamed when its size and SHA-256 match the files manifest.
An interrupted download resumes with a Range request; a server that ignores the range starts it
again. Each source is tried in turn (the configured one first): a file a source lacks, or serves
with the wrong content, comes from the next. Hugging Face, hf-mirror.com and custom endpoints share
the ``{endpoint}/{repo}/resolve/{revision}/{path}`` form; ModelScope has the same form under
``/models/``.
"""

from __future__ import annotations

import hashlib
import logging
import os
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import httpx

from mizuoto.core.engines.models import ModelFiles
from mizuoto.core.errors import DownloadError
from mizuoto.core.record import Record
from mizuoto.core.settings.models import DownloadSettings

logger = logging.getLogger(__name__)

HF_ENDPOINTS = {"huggingface": "https://huggingface.co", "hf-mirror": "https://hf-mirror.com"}
MODELSCOPE_ENDPOINT = "https://www.modelscope.cn"
SOURCES = ("huggingface", "hf-mirror", "modelscope")
CHUNK = 1 << 20
ATTEMPTS = 4


class RemoteFile(Record):
    path: str
    size: int
    sha256: str | None = None


class FilesManifest(Record):
    """``manifests/files/<name>.files.json``: one repository at a pinned revision."""

    repo: str
    revision: str
    files: list[RemoteFile]
    into: str = ""


@dataclass(frozen=True)
class Source:
    name: str
    url: Callable[[str], str]
    """The URL of a file, from its path in the repository."""


MIRROR_REVISIONS = {"modelscope": "master"}
"""The mirror's branch per source (``main`` elsewhere): its files are checked by SHA-256, not pinned by revision."""


def mirror_source(settings: DownloadSettings, files: ModelFiles, prefix: str) -> Source:
    """Mizuoto's mirror: a file at ``<prefix>/<into>/<path without subdir>``, as it lies in the model's folder."""

    def where(p: str) -> str:
        rel = p[len(files.subdir.rstrip("/")) + 1 :] if files.subdir and p.startswith(files.subdir.rstrip("/") + "/") else p
        return "/".join(x.strip("/") for x in (prefix, files.into, rel) if x)

    if settings.source == "modelscope":
        repo = settings.mirror_repo_modelscope
        return Source("mizuoto-model", lambda p: f"{MODELSCOPE_ENDPOINT}/models/{repo}/resolve/{MIRROR_REVISIONS['modelscope']}/{where(p)}")
    repo = settings.mirror_repo
    endpoint = settings.endpoint.rstrip("/") if settings.source == "custom" and settings.endpoint else HF_ENDPOINTS.get(settings.source, HF_ENDPOINTS["huggingface"])
    return Source("mizuoto-model", lambda p: f"{endpoint}/{repo}/resolve/main/{where(p)}")


def sources(settings: DownloadSettings, files: ModelFiles, revision: str, prefix: str | None = None) -> list[Source]:
    """The places to fetch ``files`` from: the mirror first when it is chosen (``prefix``: the folder under
    the models folder), then the configured source, then the others."""
    order = [settings.source, *(s for s in SOURCES if s != settings.source)]
    out: list[Source] = [mirror_source(settings, files, prefix)] if settings.repository == "mizuoto-model" and prefix else []
    for name in order:
        if name == "modelscope":
            if files.modelscope:
                repo, rev = files.modelscope, files.modelscope_revision
                out.append(Source(name, lambda p, repo=repo, rev=rev: f"{MODELSCOPE_ENDPOINT}/models/{repo}/resolve/{rev}/{p}"))
            continue
        endpoint = settings.endpoint.rstrip("/") if name == "custom" else HF_ENDPOINTS[name]
        if files.hf and endpoint:
            repo = files.hf
            out.append(Source(name, lambda p, repo=repo, endpoint=endpoint: f"{endpoint}/{repo}/resolve/{revision}/{p}"))
    return out


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while block := f.read(CHUNK * 4):
            h.update(block)
    return h.hexdigest()


class AnyEvent:
    """Set when any of its events is: a job's cancellation or a sibling download's failure."""

    def __init__(self, *events: threading.Event) -> None:
        self._events = events

    def is_set(self) -> bool:
        return any(e.is_set() for e in self._events)

    def wait(self, timeout: float) -> bool:
        end = time.monotonic() + timeout
        while not self.is_set():
            left = end - time.monotonic()
            if left <= 0:
                return False
            time.sleep(min(0.2, left))
        return True


class Cancelled(Exception):
    """The download was cancelled (the caller turns it into its own cancellation)."""


@dataclass
class Progress:
    """Bytes received across every file of a job, thread-safe."""

    total: int
    on_change: Callable[[int, int], None]
    done: int = 0

    def __post_init__(self) -> None:
        self._lock = threading.Lock()

    def add(self, n: int) -> None:
        with self._lock:
            self.done += n
            done = self.done
        self.on_change(done, self.total)


def fetch(
    client: httpx.Client,
    srcs: list[Source],
    file: RemoteFile,
    dest: Path,
    progress: Progress,
    cancel: threading.Event | AnyEvent,
    verify: bool = True,
    headers: dict[str, str] | None = None,
) -> str:
    """Download ``file`` to ``dest`` from the first source that serves it correctly; return the source's name."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    errors: list[str] = []
    for src in srcs:
        url = src.url(file.path)
        for attempt in range(ATTEMPTS):
            if cancel.is_set():
                raise Cancelled()
            try:
                # A Hugging Face token goes to Hugging Face only, never to a mirror.
                _fetch_once(client, url, file, dest, progress, cancel, verify, headers if src.name == "huggingface" else None)
                return src.name
            except Cancelled:
                raise
            except _WrongContent as e:
                errors.append(f"{src.name}: {e}")
                break
            except httpx.HTTPStatusError as e:
                errors.append(f"{src.name}: HTTP {e.response.status_code}")
                if e.response.status_code in (401, 403, 404):
                    break
            except (httpx.HTTPError, OSError) as e:
                errors.append(f"{src.name}: {type(e).__name__}: {e}")
            delay = min(30.0, 2.0**attempt)
            logger.info("Retrying %s in %.0f s (%s)", file.path, delay, errors[-1])
            if cancel.wait(delay):
                raise Cancelled()
    raise DownloadError(f"Could not download {file.path}", {"file": file.path, "sources": [s.name for s in srcs], "errors": errors[-6:]})


class _WrongContent(Exception):
    pass


def _fetch_once(
    client: httpx.Client,
    url: str,
    file: RemoteFile,
    dest: Path,
    progress: Progress,
    cancel: threading.Event | AnyEvent,
    verify: bool,
    headers: dict[str, str] | None,
) -> None:
    part = dest.with_name(dest.name + ".part")
    pos = part.stat().st_size if part.exists() else 0
    if pos > file.size:
        part.unlink()
        pos = 0
    digest = hashlib.sha256()
    if pos:
        with open(part, "rb") as f:
            while block := f.read(CHUNK * 4):
                digest.update(block)
    progress.add(pos)
    counted = pos
    try:
        if pos < file.size or file.size == 0:
            request_headers = dict(headers or {})
            if pos:
                request_headers["Range"] = f"bytes={pos}-"
            with client.stream("GET", url, headers=request_headers) as r:
                if r.status_code == 416 and pos == file.size:
                    pass
                else:
                    r.raise_for_status()
                    mode = "ab"
                    if pos and r.status_code != 206:
                        # The server ignored the range: start again.
                        progress.add(-counted)
                        counted, pos, digest, mode = 0, 0, hashlib.sha256(), "wb"
                    with open(part, mode) as f:
                        for block in r.iter_bytes():
                            if cancel.is_set():
                                raise Cancelled()
                            f.write(block)
                            digest.update(block)
                            counted += len(block)
                            progress.add(len(block))
        size = part.stat().st_size if part.exists() else 0
        if size != file.size:
            part.unlink(missing_ok=True)
            raise _WrongContent(f"{file.path} is {size} bytes, not {file.size}")
        if verify and file.sha256 and digest.hexdigest() != file.sha256:
            part.unlink(missing_ok=True)
            raise _WrongContent(f"{file.path} does not match its SHA-256")
        os.replace(part, dest)
    except BaseException:
        # Keep the part for resuming, but do not count its bytes twice on the next attempt.
        progress.add(-counted)
        raise
