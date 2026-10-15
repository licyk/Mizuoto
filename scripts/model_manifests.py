"""Write the files manifests of the catalogue models: ``python scripts/model_manifests.py [engine…]``.

For every model and shared asset in ``mizuoto/core/engines/manifests/*.toml`` whose ``files`` (and ``extra``) name a
Hugging Face repository and a ``manifest``, this pins the revision to the current commit and lists
each file with its size and SHA-256 in ``manifests/files/<manifest>``. Git LFS files carry their
SHA-256 in the listing; small files are fetched and hashed. Downloads then verify against it.
"""

from __future__ import annotations

import fnmatch
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import httpx

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from mizuoto.core.engines.models import ModelFiles  # noqa: E402
from mizuoto.core.engines.registry import MANIFESTS_DIR, load_manifest  # noqa: E402

ENDPOINT = "https://huggingface.co"


def selected(path: str, files: ModelFiles) -> bool:
    if files.subdir and not path.startswith(files.subdir.rstrip("/") + "/"):
        return False
    name = path.rsplit("/", 1)[-1]
    if files.include and not any(fnmatch.fnmatch(path, p) for p in files.include):
        return False
    return not any(fnmatch.fnmatch(path, p) or fnmatch.fnmatch(name, p) for p in files.exclude)


def listing(client: httpx.Client, files: ModelFiles) -> dict[str, Any]:
    repo = files.hf
    info = client.get(f"{ENDPOINT}/api/models/{repo}/revision/{files.revision}")
    info.raise_for_status()
    sha = info.json()["sha"]
    tree = client.get(f"{ENDPOINT}/api/models/{repo}/tree/{sha}", params={"recursive": "true"})
    tree.raise_for_status()
    out = []
    for item in tree.json():
        if item.get("type") != "file" or not selected(item["path"], files):
            continue
        lfs = item.get("lfs") or {}
        digest = lfs.get("oid")
        if not digest:
            r = client.get(f"{ENDPOINT}/{repo}/resolve/{sha}/{item['path']}", follow_redirects=True)
            r.raise_for_status()
            digest = hashlib.sha256(r.content).hexdigest()
        out.append({"path": item["path"], "size": int(item["size"]), "sha256": digest})
    out.sort(key=lambda f: f["path"])
    return {"repo": repo, "revision": sha, "files": out}


def main(argv: list[str]) -> int:
    engines = set(argv)
    target = MANIFESTS_DIR / "files"
    target.mkdir(exist_ok=True)
    with httpx.Client(timeout=60, follow_redirects=True) as client:
        for path in sorted(MANIFESTS_DIR.glob("*.toml")):
            manifest = load_manifest(path)
            if engines and manifest.id not in engines:
                continue
            for model in [*manifest.models, *manifest.assets]:
                parts = [f for f in [model.files, *model.extra] if f is not None and f.hf and f.manifest]
                for files in parts:
                    data = listing(client, files)
                    if files.into:
                        data["into"] = files.into
                    out = target / str(files.manifest)
                    out.write_text(json.dumps(data, indent=1) + "\n", encoding="utf-8")
                    total = sum(f["size"] for f in data["files"]) / 1e9
                    print(f"{model.id}: {files.hf}@{data['revision'][:10]}, {len(data['files'])} files, {total:.2f} GB → {out.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
