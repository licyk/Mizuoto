"""Application logging."""

import logging
import os

from rich.console import Console
from rich.logging import RichHandler

LOGGER_NAME = "mizuoto"


def setup_logging(level: int | str | None = None) -> logging.Logger:
    """Configure the ``mizuoto`` logger once, writing to stderr through Rich."""
    logger = logging.getLogger(LOGGER_NAME)
    if level is None:
        level = os.environ.get("MIZUOTO_LOG_LEVEL", "INFO").upper()
    logger.setLevel(level)
    if not any(h.name == LOGGER_NAME for h in logger.handlers):
        handler = RichHandler(console=Console(stderr=True), show_time=False, show_path=False, markup=False, rich_tracebacks=False)
        handler.setFormatter(logging.Formatter("%(message)s"))
        handler.name = LOGGER_NAME
        logger.addHandler(handler)
        logger.propagate = False
    return logger
