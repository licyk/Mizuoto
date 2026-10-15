"""Developer tasks, in one place: ``python scripts/dev.py <task>``.

A Python task runner rather than a Makefile, because Mizuoto is developed on Windows as well as
Linux. It uses only the standard library and runs the same tools CI runs; their configuration
lives in ``pyproject.toml``. The web UI and documentation site tasks need bun (https://bun.sh)
.

Run ``python scripts/dev.py`` with no arguments to list the tasks.
"""

import argparse
import inspect
import os
import shutil
import signal
import subprocess
import sys
import threading
from collections.abc import Callable, Sequence
from pathlib import Path
from types import FrameType
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
WEB = ROOT / "mizuoto" / "webui"
SITE = ROOT / "site"
SCHEMA = WEB / "src" / "api" / "schema.d.ts"
DEFAULT_API_PORT = 7869
DEFAULT_WEB_PORT = 5173

# A task takes the options left on the command line, which most ignore.
TASKS: dict[str, Callable[[list[str]], int]] = {}
HELP: dict[str, str] = {}


def task(name: str, help_text: str) -> Callable[[Callable[..., int]], Callable[..., int]]:
    def register(fn: Callable[..., int]) -> Callable[..., int]:
        takes_options = bool(inspect.signature(fn).parameters)

        def call(options: list[str]) -> int:
            if takes_options:
                return fn(options)
            if options:
                print(f"{name} takes no options, but got: {' '.join(options)}", file=sys.stderr)
                return 2
            return fn()

        TASKS[name] = call
        HELP[name] = help_text
        return fn

    return register


def run(command: Sequence[str], cwd: Path = ROOT, stdout: int | None = None, env: dict[str, str] | None = None) -> int:
    """Run one command, echoing it first. Returns its exit status."""
    print(f"\033[36m$ {' '.join(str(c) for c in command)}\033[0m", file=sys.stderr, flush=True)
    return subprocess.call([str(c) for c in command], cwd=cwd, stdout=stdout, env={**os.environ, **(env or {})})


def python(*args: str, stdout: int | None = None, env: dict[str, str] | None = None) -> int:
    """Run a module with the interpreter running this script, not whatever is on PATH."""
    return run([sys.executable, *args], stdout=stdout, env=env)


def first_failure(*results: int) -> int:
    return next((r for r in results if r != 0), 0)


def bun_executable() -> str | None:
    """bun on PATH, or where its installer puts it (~/.bun/bin), which a fresh shell may not have on PATH."""
    found = shutil.which("bun")
    if found:
        return found
    home = Path.home() / ".bun" / "bin" / ("bun.exe" if os.name == "nt" else "bun")
    return str(home) if home.is_file() else None


def bun(*args: str, cwd: Path = WEB) -> int:
    """Run bun in the web UI folder (or ``cwd``). Missing bun is a clear error, not a traceback."""
    exe = bun_executable()
    if exe is None:
        print("bun is needed for the web UI and site tasks. Install it from https://bun.sh, or skip them.", file=sys.stderr)
        return 127
    return run([exe, *args], cwd=cwd)


COLOURS = ["\033[35m", "\033[36m", "\033[33m"]


def run_parallel(processes: list[tuple[str, Sequence[str], Path, dict[str, str]]]) -> int:
    """Run several long-lived commands together, tagging each line with the name of its process.

    They stop together: when one exits, the others are stopped too, and Ctrl+C stops all of them.
    Returns the exit status of the first one that failed.
    """
    started: list[tuple[str, subprocess.Popen[bytes]]] = []
    readers: list[threading.Thread] = []
    lock = threading.Lock()
    stop = threading.Event()
    # Handle the signals here rather than relying on KeyboardInterrupt: a process started in the
    # background inherits SIGINT ignored, and a plain SIGTERM would otherwise orphan the children.
    previous: dict[int, Callable[[int, FrameType | None], Any] | int | None] = {}

    def on_signal(signum: int, _frame: object) -> None:
        print(f"\n\033[33mStopping ({signal.Signals(signum).name}).\033[0m", file=sys.stderr)
        stop.set()

    def pump(name: str, colour: str, proc: subprocess.Popen[bytes]) -> None:
        assert proc.stdout is not None
        for raw in proc.stdout:
            line = raw.decode("utf-8", errors="replace").rstrip()
            with lock:
                print(f"{colour}{name:>3}\033[0m │ {line}", flush=True)

    try:
        for sig in (signal.SIGINT, signal.SIGTERM):
            previous[sig] = signal.signal(sig, on_signal)
        for i, (name, command, cwd, env) in enumerate(processes):
            print(f"\033[36m$ {' '.join(str(c) for c in command)}\033[0m", file=sys.stderr, flush=True)
            proc = subprocess.Popen(
                [str(c) for c in command],
                cwd=cwd,
                env={**os.environ, **env},
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                # Its own process group, so Ctrl+C in this terminal is delivered by us, once.
                start_new_session=os.name != "nt",
            )
            started.append((name, proc))
            reader = threading.Thread(target=pump, args=(name, COLOURS[i % len(COLOURS)], proc), daemon=True)
            reader.start()
            readers.append(reader)

        while not stop.is_set():
            for name, proc in started:
                status = proc.poll()
                if status is not None:
                    print(f"\n\033[33m{name} exited with status {status}; stopping the rest.\033[0m", file=sys.stderr)
                    return status
            stop.wait(0.2)
        return 0
    except KeyboardInterrupt:
        return 0
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, handler)
        _stop_all(started)
        for reader in readers:
            reader.join(timeout=2)


def _stop_all(started: list[tuple[str, subprocess.Popen[bytes]]]) -> None:
    for _, proc in started:
        if proc.poll() is None:
            if os.name != "nt":
                os.killpg(proc.pid, signal.SIGTERM)
            else:
                proc.terminate()
    for _, proc in started:
        try:
            proc.wait(timeout=8)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()


@task("lint", "ruff check and ruff format --check")
def lint() -> int:
    return first_failure(python("-m", "ruff", "check", "."), python("-m", "ruff", "format", "--check", "."))


@task("format", "ruff format and the safe ruff fixes")
def format_code() -> int:
    return first_failure(python("-m", "ruff", "check", "--fix", "."), python("-m", "ruff", "format", "."))


@task("typecheck-py", "ty over mizuoto, the gated engines, tests and scripts")
def typecheck_py() -> int:
    return python("-m", "ty", "check", "--python", sys.executable)


@task("test-py", "pytest (extra options are passed on)")
def test_py(options: list[str]) -> int:
    return python("-m", "pytest", *options)


@task("test-engines", "The engine contract suite: test-engines [engine…] [-- pytest options] (MIZUOTO_TEST_DEVICE=cpu|cuda:N)")
def test_engines(options: list[str]) -> int:
    split = options.index("--") if "--" in options else len(options)
    engines, flags = options[:split], options[split + 1 :]
    env = {"MIZUOTO_CONTRACT": "1", "MIZUOTO_TEST_ENGINES": ",".join(engines)}
    return python("-m", "pytest", "tests/engines", "-m", "engine", "-v", *flags, env=env)


@task("model-manifests", "Pin the catalogue models' files (sizes, SHA-256): model-manifests [engine…]")
def model_manifests(options: list[str]) -> int:
    return python(str(ROOT / "scripts" / "model_manifests.py"), *options)


# -- Release -----------------------------------------------------------------


@task("wheels", "Build mizuoto (wheel, sdist, with the UI) and the licence-gated engines' wheels into dist/: wheels [all|host|native] [engine…]")
def wheels(options: list[str]) -> int:
    return python(str(ROOT / "scripts" / "build_wheels.py"), *options)


@task("smoke", "Install the built wheels over a CPU torch in a new environment and try them (doctor, every engine, speak, server)")
def smoke(options: list[str]) -> int:
    return python(str(ROOT / "scripts" / "smoke_wheels.py"), *options)


# -- Web UI ------------------------------------------------------------------


@task("web-install", "Install the web UI dependencies")
def web_install() -> int:
    return bun("install")


@task("web", "Build the web UI into mizuoto/webui/dist")
def web_build() -> int:
    return bun("run", "build")


@task("web-dev", "Serve only the web UI (expects an API server already running)")
def web_dev() -> int:
    return bun("run", "dev")


@task("dev", "Run the API server and the web UI together, with hot reload (Ctrl+C stops both)")
def dev(options: list[str]) -> int:
    """The Vite dev server proxies /api and /ws to the API server, so one address serves both."""
    parser = argparse.ArgumentParser(prog="python scripts/dev.py dev", description="Run the API server and the web UI together.")
    parser.add_argument("--api-port", type=int, default=DEFAULT_API_PORT, help=f"port for the API server (default: {DEFAULT_API_PORT})")
    parser.add_argument("--web-port", type=int, default=DEFAULT_WEB_PORT, help=f"port for the web UI (default: {DEFAULT_WEB_PORT})")
    parser.add_argument("--data-dir", help="data directory for the API server (settings, database, caches)")
    args = parser.parse_args(options)

    exe = bun_executable()
    if exe is None:
        print("bun is needed for the web UI. Install it from https://bun.sh", file=sys.stderr)
        return 127
    if not (WEB / "node_modules").is_dir():
        print("The web UI dependencies are missing. Run: python scripts/dev.py web-install", file=sys.stderr)
        return 1

    api_command = [sys.executable, "-m", "mizuoto", "webui", "--no-open", "--port", str(args.api_port), "--strict-port"]
    if args.data_dir:
        api_command += ["--data-dir", args.data_dir]
    api_env = {"PYTHONUNBUFFERED": "1", "PYTHONPATH": str(ROOT) + os.pathsep + os.environ.get("PYTHONPATH", "")}
    # The dev server proxies to the API port chosen here.
    web_env = {"MIZUOTO_BACKEND": f"http://127.0.0.1:{args.api_port}", "FORCE_COLOR": "1"}

    # The dev server binds localhost, which is IPv6 on some machines, so it is named as localhost.
    print(f"\n  Web UI  http://localhost:{args.web_port}\n  API     http://127.0.0.1:{args.api_port}\n", file=sys.stderr)
    return run_parallel(
        [
            ("api", api_command, ROOT, api_env),
            ("web", [exe, "run", "dev", "--port", str(args.web_port), "--strictPort"], WEB, web_env),
        ]
    )


@task("test-web", "Run the web UI tests and type-check")
def test_web() -> int:
    return first_failure(bun("run", "test"), bun("run", "typecheck"))


# -- The documentation site --------------------------------------------------


@task("site-install", "Install the documentation site's dependencies")
def site_install() -> int:
    return bun("install", cwd=SITE)


@task("site-dev", "Serve the documentation site with hot reload (http://127.0.0.1:3000)")
def site_dev() -> int:
    return bun("run", "dev", cwd=SITE)


@task("site", "Build the documentation site into site/.output/public")
def site_build() -> int:
    return bun("run", "build", cwd=SITE)


@task("test-site", "Run the documentation site's tests and type-check")
def test_site() -> int:
    return first_failure(bun("run", "test", cwd=SITE), bun("run", "check", cwd=SITE))


# -- The OpenAPI contract ----------------------------------------------------


@task("openapi", "Print the OpenAPI schema on standard output")
def openapi() -> int:
    return python(str(ROOT / "scripts" / "generate_openapi.py"))


@task("typegen", "Regenerate the web UI's types from the OpenAPI schema (commit the result)")
def typegen() -> int:
    tmp = WEB / "openapi.json"
    try:
        with open(tmp, "w", encoding="utf-8") as f:
            status = python(str(ROOT / "scripts" / "generate_openapi.py"), stdout=f.fileno())
        if status != 0:
            return status
        # Defaulted fields of request bodies stay optional; responses list theirs as required.
        return bun("x", "openapi-typescript", tmp.name, "-o", str(SCHEMA.relative_to(WEB)), "--default-non-nullable", "false")
    finally:
        tmp.unlink(missing_ok=True)


@task("typegen-check", "Fail when the committed types are out of date (for CI)")
def typegen_check() -> int:
    status = typegen()
    if status != 0:
        return status
    return run(["git", "diff", "--exit-code", "--", str(SCHEMA.relative_to(ROOT))])


@task("check", "What CI runs: lint, typecheck-py, test-py, test-web, typegen-check")
def check() -> int:
    results = [lint(), typecheck_py(), test_py([]), test_web(), typegen_check()]
    status = first_failure(*results)
    print("\033[32mAll checks passed\033[0m" if status == 0 else "\033[31mSome checks failed\033[0m", file=sys.stderr)
    return status


def main(argv: list[str]) -> int:
    if not argv or argv[0] in ("-h", "--help", "help"):
        width = max(len(n) for n in TASKS)
        print(__doc__.strip().splitlines()[0])
        for name, text in HELP.items():
            print(f"  {name:<{width}}  {text}")
        return 0
    name, options = argv[0], argv[1:]
    if name not in TASKS:
        print(f"Unknown task: {name}. Run without arguments to list the tasks.", file=sys.stderr)
        return 2
    return TASKS[name](options)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
