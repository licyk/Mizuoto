"""What CI checks and what Release releases, from the files a push or pull request changes: ``python scripts/ci.py``.

- ``scope [--all]``: the changed files (one per line on standard input) → the checks to run. Each distribution is
  checked on its own: a change to a licence-gated engine's package checks that package alone; a change to Mizuoto
  checks it and both gated packages, which run on its runtime; the web UI and the documentation site have checks of
  their own. A change to CI itself, or a list that could not be read (``--all``), checks everything.
- ``release``: the distributions a run of Release publishes. A push to main releases those whose version file it
  changes; a tag releases the one it names (``v<version>`` Mizuoto, ``mizuoto-<id>-v<version>`` a gated engine) and
  must match its version; a manual run releases what its ``packages`` input names (``all``, or names).
- ``lower-bound <id>``: the oldest Mizuoto a gated engine's package says it works with (CI tests the package on it).

With ``GITHUB_OUTPUT`` set, the answers are also written there as ``key=value`` lines for the workflow. Standard
library only.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
NATIVE = "packages/native"

EVERYTHING = (".github/workflows/", "scripts/ci.py")
"""A change here checks everything: it changes the checks themselves."""
NOTHING = ("README.md", "README.zh-CN.md", "AGENTS.md", "CLAUDE.md", "LICENSE", "Dockerfile", ".dockerignore", ".gitignore", ".gitattributes")
"""Files no check reads (the Docker image is built by hand; the READMEs only ship as Mizuoto's long description)."""


@dataclass(frozen=True)
class Distribution:
    name: str
    """Its name on PyPI."""
    id: str
    """``host`` for Mizuoto, else the engine id (``packages/native/<id>``)."""
    version_file: str
    tag_prefix: str


def gated_ids() -> list[str]:
    """The licence-gated engines' packages in ``packages/native``."""
    return sorted(p.name for p in (ROOT / NATIVE).iterdir() if (p / "pyproject.toml").is_file())


def distributions() -> list[Distribution]:
    host = Distribution("mizuoto", "host", "mizuoto/version.py", "v")
    return [host] + [Distribution(f"mizuoto-{g}", g, f"{NATIVE}/{g}/mizuoto_{g}/version.py", f"mizuoto-{g}-v") for g in gated_ids()]


def version_of(dist: Distribution) -> str:
    found = re.search(r'^VERSION\s*=\s*"([^"]+)"', (ROOT / dist.version_file).read_text(encoding="utf-8"), re.M)
    if not found:
        sys.exit(f"No VERSION in {dist.version_file}")
    return found.group(1)


def scope(files: list[str], everything: bool = False) -> dict[str, object]:
    """The checks for ``files``: ``core`` (Mizuoto's Python on every platform), ``web``, ``site``, ``gated`` (the
    engine ids whose packages are checked), and ``release_files`` (whether a version file changed: Release then runs
    the checks for that push, and CI's own run skips them)."""
    gated = gated_ids()
    core = web = site = everything
    checked: set[str] = set(gated) if everything else set()
    for path in files:
        if path.startswith(EVERYTHING):
            core = web = site = True
            checked |= set(gated)
        elif path.startswith(f"{NATIVE}/"):
            engine = path.split("/")[2]
            if engine in gated:
                checked.add(engine)
            else:
                core = True  # a package not (or no longer) there: the core's checks cover packages/native
        elif path.startswith("site/"):
            site = True
        elif path.startswith("mizuoto/webui/"):
            web = True
        elif path in NOTHING:
            continue
        else:
            # Mizuoto's code, its tests (the gated packages' tests share their helpers), the scripts, pyproject.toml,
            # anything new: Mizuoto's checks, the web UI's (its API types come from the code) and both gated packages'
            # (they run on Mizuoto's runtime, at this commit).
            core = web = True
            checked |= set(gated)
    versions = {d.version_file for d in distributions()}
    return {"core": core, "web": web, "site": site, "gated": sorted(checked), "release_files": any(f in versions for f in files)}


def release(event: str, ref_type: str, ref_name: str, files: list[str], packages: str = "") -> list[Distribution]:
    """The distributions this run of Release publishes."""
    dists = distributions()
    if event == "push" and ref_type == "tag":
        # The longest prefix first: "v" would match no gated tag, but mizuoto-<id>-v must not be read as Mizuoto's.
        for dist in sorted(dists, key=lambda d: -len(d.tag_prefix)):
            if ref_name.startswith(dist.tag_prefix) and re.fullmatch(r"\d.*", ref_name[len(dist.tag_prefix) :]):
                tagged = ref_name[len(dist.tag_prefix) :]
                if tagged != version_of(dist):
                    sys.exit(f"Tag {ref_name} does not match {dist.name}'s VERSION {version_of(dist)} in {dist.version_file}")
                return [dist]
        sys.exit(f"Tag {ref_name} names no distribution (v<version>, or mizuoto-<id>-v<version> for {', '.join(gated_ids())})")
    if event == "push":
        return [d for d in dists if d.version_file in files]
    wanted = {p.strip() for p in re.split(r"[,\s]+", packages) if p.strip()} or {"all"}
    if "all" in wanted:
        return dists
    unknown = wanted - {d.name for d in dists}
    if unknown:
        sys.exit(f"Unknown distributions {sorted(unknown)}: {', '.join(d.name for d in dists)}, or all")
    return [d for d in dists if d.name in wanted]


def lower_bound(engine_id: str) -> str:
    text = (ROOT / NATIVE / engine_id / "pyproject.toml").read_text(encoding="utf-8")
    found = re.search(r'"mizuoto>=([^",<]+)', text)
    if not found:
        sys.exit(f"{NATIVE}/{engine_id}/pyproject.toml names no lower bound for mizuoto")
    return found.group(1)


def write_outputs(values: dict[str, object]) -> None:
    """``values`` on standard output as JSON, and as ``key=value`` lines in ``GITHUB_OUTPUT`` when it is set."""
    print(json.dumps(values, indent=1))
    target = os.environ.get("GITHUB_OUTPUT")
    if target:
        with open(target, "a", encoding="utf-8") as out:
            for key, value in values.items():
                out.write(f"{key}={json.dumps(value) if not isinstance(value, str) else value}\n")


def read_files() -> list[str]:
    return [line.strip() for line in sys.stdin.read().splitlines() if line.strip()]


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    s = sub.add_parser("scope", help="The checks for the changed files on standard input")
    s.add_argument("--all", action="store_true", help="Check everything (the changed files are not known)")
    r = sub.add_parser("release", help="The distributions to release (push: the changed files on standard input)")
    r.add_argument("--event", default=os.environ.get("GITHUB_EVENT_NAME", "workflow_dispatch"))
    r.add_argument("--ref-type", default=os.environ.get("GITHUB_REF_TYPE", "branch"))
    r.add_argument("--ref-name", default=os.environ.get("GITHUB_REF_NAME", ""))
    r.add_argument("--packages", default="", help="A manual run's choice: all, or distribution names")
    b = sub.add_parser("lower-bound", help="The oldest Mizuoto a gated engine's package works with")
    b.add_argument("engine")
    args = parser.parse_args(argv)
    if args.command == "scope":
        values = scope([] if args.all else read_files(), everything=args.all)
        gated = values["gated"]
        assert isinstance(gated, list)
        # For the torch job: what to install and test, in one environment.
        values["python"] = bool(values["core"] or gated)
        values["editables"] = " ".join(["-e ."] + [f"-e {NATIVE}/{g}" for g in gated])
        values["test_paths"] = " ".join((["tests"] if values["core"] else []) + [f"{NATIVE}/{g}/tests" for g in gated])
        write_outputs(values)
    elif args.command == "release":
        files = read_files() if args.event == "push" and args.ref_type != "tag" else []
        chosen = release(args.event, args.ref_type, args.ref_name, files, args.packages)
        write_outputs(
            {
                "any": bool(chosen),
                "host": any(d.id == "host" for d in chosen),
                "gated": [d.id for d in chosen if d.id != "host"],
                "gated_ids": " ".join(d.id for d in chosen if d.id != "host"),
                "names": " ".join(d.name for d in chosen),
                "tags": " ".join(f"{d.tag_prefix}{version_of(d)}" for d in chosen),
                "released": [{"name": d.name, "version": version_of(d), "tag": f"{d.tag_prefix}{version_of(d)}"} for d in chosen],
            }
        )
    else:
        write_outputs({"version": lower_bound(args.engine)})
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
