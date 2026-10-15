"""Mizuoto's model mirror (``downloads.repository = "mizuoto-model"``): ``python scripts/mirror_models.py``.

The mirror holds every file Mizuoto downloads, laid out as Mizuoto keeps it: ``<engine>/<model>/<path>``
for a model and ``_assets/<engine>/<asset>/<path>`` for a shared asset, where ``<path>`` is the
file's place in the model's folder (after ``into``, without ``subdir``). A host tries the mirror
first and fills in from the official repositories what it lacks, so a partial mirror works.

- ``build OUT [id…]``: lay the installed catalogue models and assets of the models folder out under
  ``OUT`` (hard links where possible), each file checked against its pinned size and SHA-256, with
  ``MIRROR.json`` (every file: path, size, SHA-256 and its official origin) and a README. Archive parts
  (``unpack``) are not kept after unpacking, so they stay official-only.
- ``upload OUT``: publish ``OUT`` to Hugging Face (``--repo``, default ``licyk/mizuoto-model``) with ``hf upload``
  (``upload-large-folder`` before huggingface_hub 2; it needs ``hf auth login``) and, with ``--modelscope``, to
  ModelScope (``--modelscope-repo``, default ``licyks/mizuoto-model``) with ``modelscope upload``. The defaults are
  ``downloads.mirror_repo`` and ``downloads.mirror_repo_modelscope``'s.
- ``check``: which catalogue files the published mirror lacks (HEAD requests to the URLs a host asks for), on
  Hugging Face or another endpoint, or with ``--modelscope`` on ModelScope; ``--archives`` checks the archive
  parts too (a ``mirror_fetch.py`` mirror has them).

``mirror_fetch.py`` fills a mirror with the whole catalogue instead (archives included) and
``mirror_readme.py`` writes its card and licence files.

Publishing is the mirror owner's step: the weights keep their own licences (``MIRROR.json`` names each
file's origin; gated engines are left out unless ``--gated``).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from mizuoto.core.engines.models import ModelFiles  # noqa: E402
from mizuoto.core.engines.registry import MANIFESTS_DIR, load_manifest  # noqa: E402
from mizuoto.core.models.download import FilesManifest, mirror_source  # noqa: E402
from mizuoto.core.settings.models import DownloadSettings  # noqa: E402
from mizuoto.version import VERSION  # noqa: E402

FILES_DIR = MANIFESTS_DIR / "files"


def manifests() -> list[Any]:
    return [load_manifest(p) for p in sorted(MANIFESTS_DIR.glob("*.toml"))]


def entries(gated: bool) -> list[tuple[str, str, list[ModelFiles]]]:
    """(id, folder under the models folder, parts) of every catalogue model and asset with files."""
    out = []
    for m in manifests():
        if m.licence.gated and not gated:
            continue
        for e in m.models:
            parts = [f for f in [e.files, *e.extra] if f is not None and f.manifest]
            if parts and not e.builtin:
                out.append((e.id, e.id, parts))
        for a in m.assets:
            parts = [f for f in [a.files, *a.extra] if f is not None and f.manifest]
            if parts:
                out.append((a.id, f"_assets/{a.id}", parts))
    return out


def layout(files: ModelFiles, fm: FilesManifest) -> list[tuple[str, Any]]:
    """Each file's path in its model folder (and the mirror), as Mizuoto places it."""
    out = []
    for f in fm.files:
        rel = f.path[len(files.subdir.rstrip("/")) + 1 :] if files.subdir and f.path.startswith(files.subdir.rstrip("/") + "/") else f.path
        out.append(("/".join(x.strip("/") for x in (files.into, rel) if x), f))
    return out


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 22), b""):
            h.update(block)
    return h.hexdigest()


def load(name: str) -> FilesManifest:
    return FilesManifest.model_validate_json((FILES_DIR / name).read_text(encoding="utf-8"))


def build(out: Path, models_dir: Path, wanted: list[str], gated: bool, verify: bool) -> int:
    listed: list[dict[str, Any]] = []
    skipped: dict[str, str] = {}
    for entry_id, prefix, parts in entries(gated):
        if wanted and entry_id not in wanted:
            continue
        folder = models_dir / prefix
        if not folder.is_dir():
            skipped[entry_id] = "not installed"
            continue
        for files in parts:
            fm = load(str(files.manifest))
            if files.unpack:
                skipped[entry_id] = "archive parts stay official-only"
                continue
            for rel, f in layout(files, fm):
                src = folder / rel
                if not src.is_file() or src.stat().st_size != f.size:
                    skipped[entry_id] = f"{rel} is missing or incomplete"
                    continue
                if verify and f.sha256 and sha256(src) != f.sha256:
                    skipped[entry_id] = f"{rel} does not match its SHA-256"
                    continue
                dest = out / prefix / rel
                dest.parent.mkdir(parents=True, exist_ok=True)
                if not dest.exists():
                    try:
                        os.link(src, dest)
                    except OSError:
                        shutil.copy2(src, dest)
                listed.append({"path": f"{prefix}/{rel}", "size": f.size, "sha256": f.sha256, "origin": {"repo": fm.repo, "revision": fm.revision, "path": f.path}})
        print(f"{entry_id}: {'skipped (' + skipped[entry_id] + ')' if entry_id in skipped else 'ok'}", file=sys.stderr)
    out.mkdir(parents=True, exist_ok=True)
    (out / "MIRROR.json").write_text(json.dumps({"mizuoto": VERSION, "built_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "files": listed}, indent=1), encoding="utf-8")
    (out / "README.md").write_text(
        "# mizuoto-model\n\nThe model files Mizuoto (https://github.com/licyk/Mizuoto) downloads, laid out as Mizuoto keeps them:\n"
        "`<engine>/<model>/…` and `_assets/<engine>/<asset>/…`. Each file keeps the licence of the repository it comes from;\n"
        "`MIRROR.json` lists every file with its SHA-256 and its origin (repository, revision, path).\n\n"
        "Use it with `mizuoto config set downloads.repository mizuoto-model`.\n",
        encoding="utf-8",
    )
    total = sum(x["size"] for x in listed)
    print(f"{len(listed)} files, {total / 1e9:.1f} GB in {out}; {len(skipped)} entries skipped", file=sys.stderr)
    return 0


def upload(out: Path, repo: str, modelscope: bool, modelscope_repo: str) -> int:
    if not (out / "MIRROR.json").is_file():
        sys.exit(f"{out} is not a built mirror (no MIRROR.json)")
    hf = shutil.which("hf") or shutil.which("huggingface-cli")
    if hf is None:
        sys.exit("The Hugging Face CLI is needed: pip install -U huggingface_hub")
    # huggingface_hub 2 dropped ``upload-large-folder``: its ``hf upload`` splits and resumes large folders itself.
    large = subprocess.run([hf, "upload-large-folder", "--help"], capture_output=True).returncode == 0
    cmd = [hf, "upload-large-folder", repo, str(out), "--repo-type", "model"] if large else [hf, "upload", repo, str(out), ".", "--repo-type", "model"]
    print("$ " + " ".join(cmd), file=sys.stderr)
    code = subprocess.run(cmd).returncode
    if code == 0 and modelscope:
        ms = shutil.which("modelscope")
        if ms is None:
            sys.exit("The ModelScope CLI is needed: pip install -U modelscope")
        cmd = [ms, "upload", modelscope_repo, str(out), "--repo-type", "model"]
        print("$ " + " ".join(cmd), file=sys.stderr)
        code = subprocess.run(cmd).returncode
    return code


def check(settings: DownloadSettings, gated: bool, archives: bool) -> int:
    import httpx

    missing = 0
    with httpx.Client(follow_redirects=True, timeout=30) as client:
        for entry_id, prefix, parts in entries(gated):
            lacking = []
            for files in parts:
                if files.unpack and not archives:
                    continue
                src = mirror_source(settings, files, prefix)
                for f in load(str(files.manifest)).files:
                    if client.head(src.url(f.path)).status_code != 200:
                        lacking.append(f.path)
            missing += len(lacking)
            print(f"{entry_id}: {'complete' if not lacking else f'{len(lacking)} file(s) missing'}")
    return 1 if missing else 0


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build", help="Lay the installed models out as the mirror")
    b.add_argument("out", type=Path)
    b.add_argument("ids", nargs="*", help="Models and assets (default: every installed one)")
    b.add_argument("--models-dir", type=Path, default=None, help="Default: the data folder's models folder")
    b.add_argument("--gated", action="store_true", help="Include licence-gated engines (their licences may not allow it)")
    b.add_argument("--no-verify", action="store_true", help="Check sizes only, not SHA-256")
    u = sub.add_parser("upload", help="Publish a built mirror")
    u.add_argument("out", type=Path)
    defaults = DownloadSettings()
    u.add_argument("--repo", default=defaults.mirror_repo, help="The Hugging Face repository")
    u.add_argument("--modelscope", action="store_true", help="Also upload to ModelScope")
    u.add_argument("--modelscope-repo", default=defaults.mirror_repo_modelscope, help="The ModelScope repository")
    c = sub.add_parser("check", help="Which files the published mirror lacks")
    c.add_argument("--repo", default=None, help=f"Default: {defaults.mirror_repo}, with --modelscope {defaults.mirror_repo_modelscope}")
    c.add_argument("--endpoint", default="https://huggingface.co", help="A Hugging Face compatible endpoint")
    c.add_argument("--modelscope", action="store_true", help="Check the mirror on ModelScope")
    c.add_argument("--gated", action="store_true")
    c.add_argument("--archives", action="store_true", help="Check the archive parts too")
    args = parser.parse_args(argv)
    if args.cmd == "build":
        models_dir = args.models_dir
        if models_dir is None:
            from mizuoto.core.context import open_services

            with open_services() as services:
                models_dir = services.settings.models_dir
        return build(args.out.resolve(), Path(models_dir).resolve(), args.ids, args.gated, not args.no_verify)
    if args.cmd == "upload":
        return upload(args.out.resolve(), args.repo, args.modelscope, args.modelscope_repo)
    if args.modelscope:
        settings = DownloadSettings(source="modelscope", mirror_repo_modelscope=args.repo or defaults.mirror_repo_modelscope)
    else:
        settings = DownloadSettings(source="custom", endpoint=args.endpoint, mirror_repo=args.repo or defaults.mirror_repo)
    return check(settings, args.gated, args.archives)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
