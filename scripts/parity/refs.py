"""Lay out the parity corpus's reference clips in the workspace.

    python scripts/parity/refs.py [--voices ~/.local/share/mizuoto/voices]

``corpus/refs.json`` names each clip: a voice of the user's library (its primary reference and
transcript), or a fixture made by the reference stack (a case whose output becomes a clip). Writes
``<workspace>/corpus/refs/<id>.<ext>`` and ``<workspace>/corpus/refs.json`` (file, transcript,
language), which ``stack.py`` reads. Clips that do not exist yet are skipped with a note.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

from stack import WORKSPACE

HERE = Path(__file__).resolve().parent


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--voices", default="~/.local/share/mizuoto/voices")
    args = parser.parse_args(argv)
    spec = json.loads((HERE / "corpus" / "refs.json").read_text(encoding="utf-8"))
    out_dir = WORKSPACE / "corpus" / "refs"
    out_dir.mkdir(parents=True, exist_ok=True)
    table = {}
    for ref_id, entry in spec.items():
        if "voice" in entry:
            folder = Path(args.voices).expanduser() / entry["voice"]
            voice = json.loads((folder / "voice.json").read_text(encoding="utf-8"))
            primary = next(r for r in voice["references"] if r.get("primary")) if voice["references"] else None
            if primary is None:
                print(f"skip {ref_id}: voice {entry['voice']} has no reference")
                continue
            src = folder / primary["file"]
            transcript = entry.get("transcript", primary.get("transcript"))
            language = entry.get("language", primary.get("language"))
        else:
            src = WORKSPACE / "fixtures" / entry["fixture"]
            transcript, language = entry.get("transcript"), entry.get("language")
            if not src.exists():
                print(f"skip {ref_id}: {src} does not exist yet")
                continue
        name = f"{ref_id}{src.suffix}"
        shutil.copyfile(src, out_dir / name)
        table[ref_id] = {"file": name, "transcript": transcript, "language": language}
        print(f"{ref_id}: {name}")
    (WORKSPACE / "corpus" / "refs.json").write_text(json.dumps(table, ensure_ascii=False, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
