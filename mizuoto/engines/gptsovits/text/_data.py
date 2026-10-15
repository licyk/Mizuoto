"""The front end's package data (``data/``): upstream's dictionaries and tables, the large ones gzipped."""

from __future__ import annotations

import gzip
from importlib import resources
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from importlib.abc import Traversable


def data_file(name: str) -> Traversable:
    return resources.files("mizuoto.engines.gptsovits.text").joinpath("data").joinpath(name)


def read_text(name: str) -> str:
    """``data/<name>``, or ``data/<name>.gz`` decompressed."""
    plain = data_file(name)
    if plain.is_file():
        return plain.read_text(encoding="utf-8")
    return gzip.decompress(data_file(name + ".gz").read_bytes()).decode("utf-8")


def read_bytes(name: str) -> bytes:
    plain = data_file(name)
    if plain.is_file():
        return plain.read_bytes()
    return gzip.decompress(data_file(name + ".gz").read_bytes())
