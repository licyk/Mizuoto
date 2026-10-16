"""Build Mizuoto's distributions: ``python scripts/build_wheels.py [all|host|native] [id…] [--out dist]``.

- ``host``: the web UI (``bun install --frozen-lockfile``, ``bun run build``), then the ``mizuoto`` sdist and, from it,
  the wheel, built from a clean copy of ``mizuoto/`` (no ``__pycache__``, no stale ``build/``);
- ``native [id…]``: the licence-gated engines' distributions (``packages/native/<id>``: ``mizuoto-fishspeech``,
  ``mizuoto-indextts``; decision D1), one wheel each, from a clean copy (no ``__pycache__``);
- ``all`` (the default): both.

Each distribution has its own version (``mizuoto/version.py``, ``packages/native/<id>/mizuoto_<id>/version.py``) and
is released on its own; a gated engine's ``pyproject.toml`` names the Mizuoto versions it works with.
Standard library only, plus uv (``uv build``) and bun for the UI.
"""

from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
WEB = ROOT / "mizuoto" / "webui"
NATIVE = ROOT / "packages" / "native"
IGNORED = shutil.ignore_patterns("__pycache__", "*.pyc", "*.egg-info", "build", "dist")
# What the mizuoto distribution is built from, beside the mizuoto/ package.
HOST_FILES = ("pyproject.toml", "README.md", "LICENSE")


def run(cmd: list[str], cwd: Path = ROOT) -> None:
    print(f"\033[36m$ {' '.join(str(c) for c in cmd)}\033[0m", file=sys.stderr, flush=True)
    subprocess.run([str(c) for c in cmd], cwd=cwd, check=True)


def uv() -> str:
    found = shutil.which("uv") or str(Path(sys.executable).parent / "uv")
    if not Path(found).exists() and not shutil.which(found):
        sys.exit("uv is needed: pip install uv")
    return found


def version_of(path: Path) -> str:
    text = path.read_text(encoding="utf-8")
    found = re.search(r'^VERSION\s*=\s*"([^"]+)"', text, re.M)
    if not found:
        sys.exit(f"No VERSION in {path}")
    return found.group(1)


def native_packages() -> list[str]:
    return sorted(p.name for p in NATIVE.iterdir() if (p / "pyproject.toml").is_file())


def versions() -> dict[str, str]:
    """Each distribution's version, by name."""
    found = {"mizuoto": version_of(ROOT / "mizuoto" / "version.py")}
    found |= {f"mizuoto-{name}": version_of(NATIVE / name / f"mizuoto_{name}" / "version.py") for name in native_packages()}
    return found


def build_host(out: Path, skip_web: bool = False) -> None:
    if not skip_web:
        bun = shutil.which("bun")
        if bun is None:
            sys.exit("bun is needed to build the web UI (https://bun.sh); --skip-web uses mizuoto/webui/dist as it is")
        run([bun, "install", "--frozen-lockfile"], cwd=WEB)
        run([bun, "run", "build"], cwd=WEB)
    if not (WEB / "dist" / "index.html").is_file():
        sys.exit("The web UI build wrote no dist/index.html")
    # From a clean copy: setuptools reuses the checkout's build/lib, which still holds modules deleted since the last
    # build, and they went into the wheel. The wheel is then built from the sdist (uv build's default), so the two match.
    with tempfile.TemporaryDirectory(prefix="mizuoto-host-") as tmp:
        copy = Path(tmp) / "mizuoto"
        copy.mkdir()
        for name in HOST_FILES:
            shutil.copy2(ROOT / name, copy / name)
        shutil.copytree(ROOT / "mizuoto", copy / "mizuoto", ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "node_modules"), symlinks=False)
        run([uv(), "build", "--out-dir", str(out), str(copy)])


def build_native(name: str, out: Path) -> None:
    package = NATIVE / name
    if not (package / "pyproject.toml").is_file():
        sys.exit(f"No native engine package {name} (packages/native: {', '.join(native_packages())})")
    with tempfile.TemporaryDirectory(prefix=f"mizuoto-{name}-") as tmp:
        copy = Path(tmp) / name
        shutil.copytree(package, copy, ignore=IGNORED, symlinks=False)
        run([uv(), "build", "--wheel", "--out-dir", str(out), str(copy)])


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("what", nargs="?", default="all", choices=["all", "host", "native"])
    parser.add_argument("engines", nargs="*", help="Gated engines to build (default: every one in packages/native)")
    parser.add_argument("--out", type=Path, default=ROOT / "dist")
    parser.add_argument("--skip-web", action="store_true", help="Use the web UI already built in mizuoto/webui/dist (a Docker stage built it)")
    args = parser.parse_args(argv)
    known = versions()
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    built = []
    if args.what in ("all", "host"):
        build_host(out, args.skip_web)
        built.append("mizuoto")
    if args.what in ("all", "native"):
        for name in args.engines or native_packages():
            build_native(name, out)
            built.append(f"mizuoto-{name}")
    print(f"→ {out}", file=sys.stderr)
    for name in built:
        for wheel in sorted(out.glob(f"{name.replace('-', '_')}-{known[name]}-*.whl")):
            print(f"  {wheel.name}  {wheel.stat().st_size / 1e6:.2f} MB", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
