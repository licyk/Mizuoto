"""Timestamps and identifiers."""

import secrets
from datetime import datetime, timezone


def now_iso() -> str:
    """The current UTC time as ISO 8601 with milliseconds and a ``Z``, which sorts as text."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"


def new_id(prefix: str = "") -> str:
    """A short random identifier that sorts roughly by creation time."""
    stamp = format(int(datetime.now(timezone.utc).timestamp() * 1000), "x")
    return f"{prefix}{stamp}{secrets.token_hex(4)}"
