"""version, env vars."""

from __future__ import annotations

import os
import platform
from importlib import metadata
from typing import Annotated

import typer

from mizuoto.cli.output import print_json, print_table
from mizuoto.version import VERSION

COMPONENTS = ("torch", "fastapi", "uvicorn", "pydantic", "typer", "httpx", "numpy", "av", "mizuoto-fishspeech", "mizuoto-indextts")

ENV_VARS = {
    "MIZUOTO_DATA_DIR": "Data directory holding settings.toml, the database, models, voices and outputs",
    "MIZUOTO_CONFIG_DIR": "Folder for settings.toml, when it should not sit in the data directory",
    "MIZUOTO_LOG_LEVEL": "Log level: DEBUG, INFO, WARNING or ERROR",
    "MIZUOTO_<GROUP>__<FIELD>": "Override one setting, e.g. MIZUOTO_SERVER__PORT=8000 or MIZUOTO_DOWNLOADS__SOURCE=hf-mirror",
}


def _components() -> dict[str, str | None]:
    out: dict[str, str | None] = {}
    for name in COMPONENTS:
        try:
            out[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            out[name] = None
    return out


def version(json_output: Annotated[bool, typer.Option("--json", help="Print JSON")] = False) -> None:
    """Show the version of Mizuoto and its main components."""
    components = _components()
    if json_output:
        print_json({"mizuoto": VERSION, "python": platform.python_version(), "components": components})
        return
    print_table(None, ["Component", "Version"], [("mizuoto", VERSION), ("python", platform.python_version()), *((k, v or "not installed") for k, v in components.items())])


def env_vars(json_output: Annotated[bool, typer.Option("--json", help="Print JSON")] = False) -> None:
    """List the environment variables Mizuoto reads, and those currently set."""
    in_use = {k: ("***" if "TOKEN" in k else v) for k, v in sorted(os.environ.items()) if k.startswith("MIZUOTO_")}
    if json_output:
        print_json({"known": ENV_VARS, "set": in_use})
        return
    print_table("Environment variables", ["Name", "Meaning"], ENV_VARS.items())
    if in_use:
        print_table("Set now", ["Name", "Value"], in_use.items())
