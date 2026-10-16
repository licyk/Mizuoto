"""Try the built distributions as a user gets them: ``python scripts/smoke_wheels.py [--dist dist]``.

A new virtual environment outside the checkout gets torch (the CPU build: ``--torch-index``), then, with one
``pip install``, the wheels in ``--dist`` and the distributions it lacks from ``--index``, which must leave that torch
in place. The distributions are released on their own, so a release is tried with the others as users have them: a
Mizuoto wheel with the gated engines' released packages, a gated engine's wheel with the released Mizuoto (the newest
its range allows); ``--others none`` installs only the wheels. uv runs without the machine's
configuration (``--no-config``, no ``UV_*``/``PIP_*`` variables). In a temporary data folder it then runs:

- ``mizuoto version``, ``mizuoto env vars`` and ``mizuoto doctor`` (every engine must be available);
- every engine's implementation and trainer imported (``mizuoto.runtime.engine.get_engine``), every module of
  ``mizuoto`` and the gated packages imported, and every data file the checkout has beside the code found installed
  (vocabularies, FSTs, the text front ends' dictionaries, the manifests), with the ones engines read at run time read;
- a dummy-engine ``mizuoto speak``;
- the server: ``/api/v1/app/health``, the UI, and a dummy speak through the API.

Standard library only, plus uv.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import socket
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
NATIVE = ROOT / "packages" / "native"

# Run inside the new environment: every engine and its trainer, every module, every data file, and the files engines
# read at run time actually read. argv[1] is the JSON list of data files expected (paths relative to site-packages).
CHECK = r"""
import importlib, json, pkgutil, sys
from pathlib import Path

import mizuoto
from mizuoto.runtime.engine import BUILTIN, engine_ids, get_engine

expected_data = json.loads(sys.argv[1])
site = Path(mizuoto.__file__).resolve().parent.parent
missing = [name for name in expected_data if not (site / name).is_file()]
if missing:
    sys.exit(f"package data missing from the installed wheels: {missing}")

ids = engine_ids()
gated = sorted(set(ids) - set(BUILTIN))
trainers = []
for engine_id in ids:
    engine = get_engine(engine_id)
    trainer = getattr(engine, "TRAINER", None)
    if trainer:
        module, _, name = trainer.partition(":")
        getattr(importlib.import_module(module), name)
        trainers.append(engine_id)

# Every module, so a module left out of the wheel shows here and not at a user's first request. Modules that need an
# optional extra (GPT-SoVITS's Japanese and Korean front ends) import it lazily, so every module imports.
modules = 0
for package in ["mizuoto", *(f"mizuoto_{g}" for g in gated)]:
    root = importlib.import_module(package)
    for info in pkgutil.walk_packages(root.__path__, package + "."):
        importlib.import_module(info.name)
        modules += 1

# The data engines read at run time, read as they read it.
from mizuoto.text.tokenize.whisper import get_encoding
assert get_encoding().encode("hello world"), "whisper vocabulary"
from mizuoto.engines.gptsovits.text.english import read_cmudict
assert len(read_cmudict()) > 100000, "cmudict"
from mizuoto.engines.f5tts import engine as f5
assert (Path(f5.__file__).with_name("vocab.txt")).stat().st_size > 1000, "F5-TTS vocabulary"
if "indextts" in gated:
    from mizuoto_indextts.text import TextNormalizer
    said = TextNormalizer().normalize("我有3个苹果")
    assert "三" in said, said
print(json.dumps({"engines": ids, "gated": gated, "trainers": trainers, "modules": modules, "data_files": len(expected_data)}))
"""


def run(cmd: list[str], env: dict[str, str] | None = None, cwd: Path | None = None) -> str:
    """Run outside the checkout (from it, ``python -c`` would import the source tree, not the wheel)."""
    shown = " ".join(c if len(c) < 120 else c[:60] + "…" for c in cmd)
    print(f"\033[36m$ {shown}\033[0m", file=sys.stderr, flush=True)
    done = subprocess.run(cmd, env=env, capture_output=True, text=True, cwd=cwd or tempfile.gettempdir())
    if done.returncode != 0:
        sys.exit(f"failed ({done.returncode}):\n{done.stdout}\n{done.stderr}")
    return done.stdout


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def wheel_version(wheel: Path) -> str:
    return wheel.name.split("-")[1]


def expected_data(host: bool, gated: list[str]) -> list[str]:
    """The checkout's data files beside the code (not .py) of the distributions built from it, as paths relative to
    site-packages (those from the index are as they were released)."""
    roots = ([ROOT / "mizuoto"] if host else []) + [NATIVE / g / f"mizuoto_{g}" for g in gated]
    found = []
    for root in roots:
        for path in root.rglob("*"):
            rel = path.relative_to(root.parent)
            parts = rel.parts
            if not path.is_file() or path.suffix in (".py", ".pyc", ".pyi") or "__pycache__" in parts:
                continue
            if parts[:2] == ("mizuoto", "webui"):  # the UI's sources; its build is checked through the server
                continue
            found.append(rel.as_posix())
    return sorted(found)


def request(url: str, body: dict | None = None, origin: str | None = None, timeout: float = 5) -> bytes:
    data = json.dumps(body).encode() if body is not None else None
    headers = {"content-type": "application/json"} if body is not None else {}
    if origin:
        headers["origin"] = origin
    with urllib.request.urlopen(urllib.request.Request(url, data=data, headers=headers), timeout=timeout) as r:
        return r.read()


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dist", type=Path, default=ROOT / "dist")
    parser.add_argument("--python", default="3.11")
    parser.add_argument("--torch-index", default="https://download.pytorch.org/whl/cpu", help="Where torch comes from (the build is the user's choice)")
    parser.add_argument("--torch", default="torch>=2.8", help="Which torch (torch==2.8.0: the oldest Mizuoto supports)")
    parser.add_argument("--index", default="https://pypi.org/simple", help="Where the other dependencies come from")
    parser.add_argument("--others", choices=["index", "none"], default="index", help="The distributions --dist lacks: from --index, or none")
    args = parser.parse_args(argv)
    uv = shutil.which("uv") or str(Path(sys.executable).parent / "uv")
    dist = args.dist.resolve()
    hosts = sorted(dist.glob("mizuoto-[0-9]*-py3-none-any.whl"))
    wheels = sorted(dist.glob("mizuoto_*-py3-none-any.whl"))
    if len(hosts) > 1:
        sys.exit(f"More than one mizuoto wheel in {dist}: {[w.name for w in hosts]}")
    built = [w.name.split("-")[0].removeprefix("mizuoto_") for w in wheels]
    if not hosts and not built:
        sys.exit(f"No wheels in {dist}: python scripts/build_wheels.py")
    if not hosts and args.others == "none":
        sys.exit(f"No mizuoto wheel in {dist}, and --others none: python scripts/build_wheels.py host")
    every = sorted(p.name for p in NATIVE.iterdir() if (p / "pyproject.toml").is_file())
    from_index = ([] if hosts else ["mizuoto"]) + ([f"mizuoto-{g}" for g in every if g not in built] if args.others == "index" else [])
    gated = sorted(set(built) | ({g for g in every if f"mizuoto-{g}" in from_index}))
    expected_versions = {"mizuoto": wheel_version(hosts[0])} if hosts else {}
    expected_versions |= {f"mizuoto-{g}": wheel_version(w) for g, w in zip(built, wheels)}
    with tempfile.TemporaryDirectory(prefix="mizuoto-smoke-") as tmp:
        folder = Path(tmp)
        venv = folder / "venv"
        # uv without the machine's configuration: its UV_DEFAULT_INDEX or uv.toml would win over the torch index.
        uv_env = {k: v for k, v in os.environ.items() if not k.startswith(("UV_", "PIP_")) and k not in ("VIRTUAL_ENV", "PYTHONPATH")}
        if "UV_CACHE_DIR" in os.environ:
            uv_env["UV_CACHE_DIR"] = os.environ["UV_CACHE_DIR"]
        run([uv, "venv", "-q", "--no-config", "--python", args.python, str(venv)], uv_env)
        python = venv / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
        torch_version = [str(python), "-c", "import torch; print(torch.__version__)"]
        run([uv, "pip", "install", "-q", "--no-config", "--python", str(python), "--default-index", args.torch_index, args.torch], uv_env)
        torch_before = run(torch_version, uv_env).strip()
        # One install of the wheels (by path: an index's ``mizuoto`` must not win) and of the released distributions they
        # go with; their dependencies from --index.
        run([uv, "pip", "install", "-q", "--no-config", "--python", str(python), "--default-index", args.index, *map(str, hosts + wheels), *from_index], uv_env)
        torch_after = run(torch_version, uv_env).strip()
        if torch_after != torch_before:
            sys.exit(f"Installing mizuoto replaced torch {torch_before} with {torch_after}: the build is the user's choice")
        mizuoto = [str(venv / "Scripts" / "mizuoto.exe")] if os.name == "nt" else [str(venv / "bin" / "mizuoto")]
        drop = ("PYTHONPATH", "VIRTUAL_ENV", "CUDA_VISIBLE_DEVICES")
        env = {k: v for k, v in os.environ.items() if k not in drop} | {"MIZUOTO_DATA_DIR": str(folder / "data"), "CUDA_VISIBLE_DEVICES": ""}
        about = json.loads(run([*mizuoto, "version", "--json"], env))
        where = run([str(python), "-c", "import mizuoto; print(mizuoto.__file__)"], env).strip()
        if not where.startswith(str(venv)):
            sys.exit(f"mizuoto came from {where}, not the new environment")
        names = ["mizuoto", *(f"mizuoto-{g}" for g in gated)]
        installed = json.loads(run([str(python), "-c", f"import json; from importlib.metadata import version; print(json.dumps({{n: version(n) for n in {names!r}}}))"], env))
        wrong = {n: (installed[n], v) for n, v in expected_versions.items() if installed[n] != v}
        if wrong:
            sys.exit(f"Installed versions are not the wheels' (installed, wheel): {wrong}")
        print(f"mizuoto {about['mizuoto']} from {where}; " + ", ".join(f"{n} {v} ({'wheel' if n in expected_versions else 'index'})" for n, v in installed.items()))
        run([*mizuoto, "env", "vars"], env)
        report = json.loads(run([*mizuoto, "doctor", "--json"], env))
        print(f"torch {report['torch']} ({report['torch_build']}); this machine: {report['recommended_build']}")
        engines = {f["engine"]: f for f in report["findings"] if f.get("check") == "engine"}
        unavailable = {e: f["message"] for e, f in engines.items() if f["level"] == "error"}
        if unavailable or not engines:
            sys.exit(f"doctor: engines not available: {unavailable or 'none listed'}")
        print(f"doctor: {len(engines)} engines available")
        checked = json.loads(run([str(python), "-c", CHECK, json.dumps(expected_data(bool(hosts), built))], env))
        missing = sorted(set(gated) - set(checked["gated"]))
        if missing:
            sys.exit(f"gated engines installed but not registered: {missing}")
        print(
            f"engines {', '.join(checked['engines'])}; trainers {', '.join(checked['trainers'])}; {checked['modules']} modules imported; {checked['data_files']} data files present"
        )
        wav = folder / "hello.wav"
        run([*mizuoto, "speak", "Hello from the wheel.", "-m", "dummy/default", "--preset", "alto", "-o", str(wav)], env)
        if wav.stat().st_size < 1000:
            sys.exit("speak wrote no audio")
        port = free_port()
        base = f"http://127.0.0.1:{port}"
        server = subprocess.Popen([*mizuoto, "webui", "--port", str(port), "--no-open"], env=env, cwd=folder, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
        try:
            deadline = time.monotonic() + 60
            while True:
                try:
                    request(f"{base}/api/v1/app/health", timeout=2)
                    break
                except OSError:
                    if time.monotonic() > deadline or server.poll() is not None:
                        sys.exit("The server did not answer")
                    time.sleep(0.5)
            served = json.loads(request(f"{base}/api/v1/app/version"))
            page = request(f"{base}/").decode()
            if "<div id=" not in page and "<html" not in page.lower():
                sys.exit("The web UI is not served")
            speak = {"model": "dummy/default", "text": "Hello through the API.", "voice_spec": {"preset": "alto"}}
            job = json.loads(request(f"{base}/api/v1/tts/speak?wait=true", speak, origin=base, timeout=60))
            if job.get("state") != "completed":
                sys.exit(f"The API's speak job did not complete: {job}")
            audio = request(f"{base}/api/v1/outputs/{job['result']['output_id']}/content")
            if len(audio) < 1000:
                sys.exit("The API's speak wrote no audio")
            print(f"server {served.get('version')} on port {port}: health, UI and a dummy speak ({len(audio)} bytes) answer")
        finally:
            server.terminate()
            server.wait(timeout=30)
    print("smoke test passed")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
