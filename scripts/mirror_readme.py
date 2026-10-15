"""Write a mizuoto-model mirror's README, LICENSES.md and gated licence files: ``python scripts/mirror_readme.py OUT``.

Reads ``OUT/MIRROR.json`` (``mirror_fetch.py`` or ``mirror_models.py build``) and writes:

- ``README.md``: Hugging Face card metadata, how to use the mirror, the engines' licence notes and a
  table of every model and asset folder (engine licence, files, size, origin repositories);
- ``LICENSES.md``: each engine's licence and link;
- for each engine whose licence ships with Mizuoto (the gated ones), its text as ``<engine>/LICENSE<ext>``,
  and the notice file that licence asks for (Fish Audio's ``NOTICE.txt``): copies of the weights must
  carry them.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import mirror_models as mm  # noqa: E402

from mizuoto.core.engines.registry import MANIFESTS_DIR  # noqa: E402

NOTICES = {"fishspeech": "This model is licensed under the Fish Audio Research License, Copyright © 39 AI, INC. All Rights Reserved.\n"}
"""Notice files a licence requires beside the weights (Fish Audio Research License §III.a)."""


def folders(mirror: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Per model or asset folder: its engine, size, file count and origin repositories."""
    out: dict[str, dict[str, Any]] = {}
    for f in mirror["files"]:
        parts = f["path"].split("/")
        asset = parts[0] == "_assets"
        r = out.setdefault("/".join(parts[: 3 if asset else 2]), {"engine": parts[1] if asset else parts[0], "size": 0, "files": 0, "origins": {}})
        r["size"] += f["size"]
        r["files"] += 1
        r["origins"][f"{f['origin']['repo']}@{f['origin']['revision'][:12]}"] = None
    return out


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("out", type=Path)
    parser.add_argument("--repo", default="licyk/mizuoto-model", help="The Hugging Face repository (for the card's licence link)")
    args = parser.parse_args(argv)
    out: Path = args.out.resolve()
    if not (out / "MIRROR.json").is_file():
        sys.exit(f"{out} is not a built mirror (no MIRROR.json)")
    rows = folders(json.loads((out / "MIRROR.json").read_text(encoding="utf-8")))
    licences = {m.id: m.licence for m in mm.manifests()}
    engines = sorted({r["engine"] for r in rows.values()})

    notes = []
    for engine in engines:
        lic = licences[engine]
        files = []
        if lic.file and (out / engine).is_dir():
            name = f"LICENSE{Path(lic.file).suffix}"
            shutil.copyfile(MANIFESTS_DIR / "licences" / lic.file, out / engine / name)
            files.append(f"`{engine}/{name}`")
            if engine in NOTICES:
                (out / engine / "NOTICE.txt").write_text(NOTICES[engine], encoding="utf-8")
                files.append(f"`{engine}/NOTICE.txt`")
        if files or lic.notice or not lic.commercial:
            line = f"- **{engine}** (`{engine}/`): {lic.name}" + (f" ({', '.join(files)})" if files else "") + "."
            if not lic.commercial and "commercial" not in lic.notice:
                line += " Not for commercial use."
            if lic.notice:
                line += f" {lic.notice.rstrip('.')}."
            notes.append(line)

    total = sum(r["size"] for r in rows.values())
    lines = [
        "---",
        "license: other",
        "license_name: per-model",
        # Hugging Face accepts only an https link here.
        f"license_link: https://huggingface.co/{args.repo}/blob/main/LICENSES.md",
        "tags:",
        "- text-to-speech",
        "- mizuoto",
        "---",
        "",
        "# mizuoto-model",
        "",
        "The model files [Mizuoto](https://github.com/licyk/Mizuoto) downloads, laid out as Mizuoto keeps them:",
        "`<engine>/<model>/…` for a model and `_assets/<engine>/<asset>/…` for a file several models share.",
        "Mizuoto tries this repository first and falls back to each model's official repository, file by file;",
        "every file is checked against the SHA-256 pinned in Mizuoto's catalogue.",
        "",
        "```bash",
        "mizuoto config set downloads.repository mizuoto-model",
        "```",
        "",
        "`MIRROR.json` lists every file with its size, SHA-256 and origin (repository, revision, path).",
        "Files marked `archive` there are kept as published and unpacked by Mizuoto after download.",
        "",
        "## Licences",
        "",
        "The files are unmodified copies; each keeps the licence of the repository it comes from. Using them means",
        "accepting those licences (`LICENSES.md`). In particular:",
        "",
        *notes,
        "",
        f"## Contents ({total / 1e9:.1f} GB)",
        "",
        "| Folder | Engine licence | Files | Size | Origin |",
        "| --- | --- | ---: | ---: | --- |",
    ]
    for prefix, r in rows.items():
        origins = ", ".join(f"`{o}`" for o in r["origins"])
        lines.append(f"| `{prefix}` | {licences[r['engine']].name} | {r['files']} | {r['size'] / 1e9:.2f} GB | {origins} |")
    (out / "README.md").write_text("\n".join([*lines, ""]), encoding="utf-8")

    table = [f"| {e} | {licences[e].name} | {licences[e].url} |" for e in engines]
    (out / "LICENSES.md").write_text(
        "\n".join(
            [
                "# Licences",
                "",
                "| Engine | Licence | Link |",
                "| --- | --- | --- |",
                *table,
                "",
                "Shared assets keep the licence of their own origin repository (see `MIRROR.json`).",
                "",
            ]
        ),
        encoding="utf-8",
    )
    print(f"{len(rows)} folders, {total / 1e9:.1f} GB", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
