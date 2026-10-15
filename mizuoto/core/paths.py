"""Default locations on disk."""

import os
import sys
from collections.abc import Mapping
from pathlib import Path

ENV_PREFIX = "MIZUOTO_"
DATA_DIR_ENV = f"{ENV_PREFIX}DATA_DIR"
CONFIG_DIR_ENV = f"{ENV_PREFIX}CONFIG_DIR"
APP_DIR_NAME = "mizuoto"


def default_data_dir(environ: Mapping[str, str] | None = None) -> Path:
    """Return the data directory: ``MIZUOTO_DATA_DIR``, or the platform's user data folder."""
    env = os.environ if environ is None else environ
    value = env.get(DATA_DIR_ENV)
    if value:
        return Path(value).expanduser().resolve()
    if sys.platform == "win32":
        base = Path(env.get("APPDATA") or Path.home() / "AppData" / "Roaming")
    elif sys.platform == "darwin":
        base = Path.home() / "Library" / "Application Support"
    else:
        base = Path(env.get("XDG_DATA_HOME") or Path.home() / ".local" / "share")
    return (base / APP_DIR_NAME).resolve()
