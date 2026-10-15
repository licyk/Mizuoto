"""Fill a mizuoto-model mirror from the catalogue: ``python scripts/mirror_fetch.py OUT``.

Unlike ``mirror_models.py build``, which lays out what this machine has installed, this puts every
catalogue file under ``OUT`` (archive parts included, kept as published, since a host unpacks them after
download): each at ``<prefix>/<into>/<path without subdir>``, where Mizuoto's mirror source asks for it.
A copy in a models folder with the same size and SHA-256 is hard-linked; anything else is fetched from
the official sources with Mizuoto's own ``fetch`` (resumed, retried, checked against its SHA-256;
``HF_TOKEN`` goes to huggingface.co only). ``MIRROR.json`` is written once every file is there;
``mirror_readme.py`` then writes the README and the gated engines' licence files, and ``mirror_models.py
upload`` publishes the folder.

- ``OUT [id…]``: only these models and assets (default: all).
- ``--models-dir DIR`` (repeatable): folders to take verified copies from (default: the data folder's).
- ``--gated``: include licence-gated engines (their licences ask for the licence text beside the files).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import httpx  # noqa: E402
import mirror_models as mm  # noqa: E402

from mizuoto.core.models.download import Progress, RemoteFile, Source, fetch, sources  # noqa: E402
from mizuoto.core.settings.models import DownloadSettings  # noqa: E402
from mizuoto.version import VERSION  # noqa: E402


def matches(path: Path, f: RemoteFile) -> bool:
    if not path.is_file() or path.stat().st_size != f.size:
        return False
    if not f.sha256:
        return True
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 22), b""):
            h.update(block)
    return h.hexdigest() == f.sha256


def plan(wanted: list[str], gated: bool, models_dirs: list[Path]) -> tuple[list[tuple[str, RemoteFile, list[Source], list[Path]]], list[dict[str, Any]]]:
    """(mirror path, file, official sources, local copies to try) per file, and the MIRROR.json rows."""
    settings = DownloadSettings()
    jobs: list[tuple[str, RemoteFile, list[Source], list[Path]]] = []
    listed: list[dict[str, Any]] = []
    seen: dict[str, str | None] = {}
    for entry_id, prefix, parts in mm.entries(gated):
        if wanted and entry_id not in wanted:
            continue
        for files in parts:
            fm = mm.load(str(files.manifest))
            srcs = sources(settings, files, fm.revision)
            for rel, f in mm.layout(files, fm):
                key = f"{prefix}/{rel}"
                if key in seen:
                    if seen[key] != f.sha256:
                        sys.exit(f"{key} is listed twice with different content")
                    continue
                seen[key] = f.sha256
                # An archive is deleted once unpacked, so a models folder never has it.
                local = [] if files.unpack else [d / prefix / rel for d in models_dirs]
                jobs.append((key, f, srcs, local))
                row = {"path": key, "size": f.size, "sha256": f.sha256, "origin": {"repo": fm.repo, "revision": fm.revision, "path": f.path}}
                listed.append({**row, "archive": True} if files.unpack else row)
    return jobs, listed


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("out", type=Path)
    parser.add_argument("ids", nargs="*", help="Models and assets (default: every one)")
    parser.add_argument("--models-dir", type=Path, action="append", default=[], help="Take verified copies from here (default: the data folder's models folder)")
    parser.add_argument("--gated", action="store_true", help="Include licence-gated engines")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args(argv)
    out: Path = args.out.resolve()
    models_dirs = [d.resolve() for d in args.models_dir]
    if not models_dirs:
        from mizuoto.core.context import open_services

        with open_services() as services:
            models_dirs = [Path(services.settings.models_dir).resolve()]
    token = os.environ.get("HF_TOKEN")
    headers = {"Authorization": f"Bearer {token}"} if token else None

    jobs, listed = plan(args.ids, args.gated, models_dirs)
    total = sum(f.size for _, f, _, _ in jobs)
    print(f"{len(jobs)} files, {total / 1e9:.1f} GB", flush=True)
    lock = threading.Lock()
    counts = {"present": 0, "linked": 0, "fetched": 0}
    failures: list[str] = []
    last = [0.0]

    def report(done: int, _total: int) -> None:
        if time.monotonic() - last[0] > 30:
            last[0] = time.monotonic()
            print(f"  {done / 1e9:.1f} GB downloaded", flush=True)

    progress = Progress(total=total, on_change=report)
    cancel = threading.Event()

    def one(job: tuple[str, RemoteFile, list[Source], list[Path]]) -> None:
        key, f, srcs, local = job
        dest = out / key
        try:
            if matches(dest, f):
                kind = "present"
            else:
                dest.parent.mkdir(parents=True, exist_ok=True)
                src = next((p for p in local if matches(p, f)), None)
                if src is not None:
                    dest.unlink(missing_ok=True)
                    try:
                        os.link(src, dest)
                    except OSError:
                        shutil.copy2(src, dest)
                    kind = "linked"
                else:
                    with httpx.Client(follow_redirects=True, timeout=httpx.Timeout(60, read=300)) as client:
                        name = fetch(client, srcs, f, dest, progress, cancel, verify=True, headers=headers)
                    kind = "fetched"
                    print(f"  {key} ({f.size / 1e6:.1f} MB) from {name}", flush=True)
            with lock:
                counts[kind] += 1
                done = sum(counts.values())
                if done % 50 == 0:
                    print(f"[{done}/{len(jobs)}] {counts}", flush=True)
        except Exception as e:
            with lock:
                failures.append(f"{key}: {e!r}")
            print(f"FAILED {key}: {e!r}", flush=True)

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        list(pool.map(one, sorted(jobs, key=lambda j: -j[1].size)))

    print(f"{counts}; {len(failures)} failed", flush=True)
    for line in failures:
        print("  " + line)
    if failures:
        return 1
    (out / "MIRROR.json").write_text(json.dumps({"mizuoto": VERSION, "built_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "files": listed}, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
