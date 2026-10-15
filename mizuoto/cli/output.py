"""Tables, ``--json`` output and services for the command line."""

import json
import sys
from collections.abc import Generator, Iterable, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel
from rich.console import Console
from rich.table import Table

if TYPE_CHECKING:
    from mizuoto.core.context import Services

console = Console()
err_console = Console(stderr=True)


def human_size(size: float | None) -> str:
    if size is None:
        return "-"
    value = float(size)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(value) < 1024 or unit == "TB":
            return f"{value:.0f} {unit}" if unit == "B" else f"{value:.1f} {unit}"
        value /= 1024
    return f"{value:.1f} TB"


def to_jsonable(value: Any) -> Any:
    """Records as the API serialises them; containers recursively."""
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    if isinstance(value, (list, tuple)):
        return [to_jsonable(v) for v in value]
    if isinstance(value, dict):
        return {k: to_jsonable(v) for k, v in value.items()}
    if isinstance(value, Path):
        return str(value)
    return value


def print_json(value: Any) -> None:
    """Print the same shape the API returns."""
    sys.stdout.write(json.dumps(to_jsonable(value), indent=2, ensure_ascii=False) + "\n")


def print_table(title: str | None, columns: Sequence[str], rows: Iterable[Sequence[Any]]) -> None:
    table = Table(title=title, show_lines=False, header_style="bold")
    for column in columns:
        table.add_column(column, overflow="fold")
    for row in rows:
        table.add_row(*["" if v is None else str(v) for v in row])
    console.print(table)


@contextmanager
def open_services() -> "Generator[Services, None, None]":
    """Build the services for one command and close them afterwards. No background threads run."""
    from mizuoto.core.context import open_services as core_open_services

    with core_open_services() as services:
        yield services


def short(text: str | None, width: int = 60) -> str:
    if not text:
        return ""
    flat = " ".join(text.split())
    return flat if len(flat) <= width else flat[: width - 1] + "…"
