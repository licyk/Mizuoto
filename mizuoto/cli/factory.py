"""Typer factory and custom classes, after ani2xcur-cli.

The private ``typer._click`` imports are confined to this module. Typer is not pinned from above,
so a release that moves them breaks here first; the CLI tests (tree snapshot, ``--debug``) catch it.
"""

import logging
import os
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any

import typer
from typer import _click
from typer.core import TyperCommand, TyperGroup, TyperOption

from mizuoto.core.paths import DATA_DIR_ENV
from mizuoto.logger import LOGGER_NAME

if TYPE_CHECKING:
    from mizuoto.core.errors import MizuotoError

DEBUG_OPTION_HELP = "Print debug logs"
DATA_DIR_OPTION_HELP = f"Use this data directory (settings, database, models, outputs) instead of {DATA_DIR_ENV} or the default"

# Re-exported for app.py, so it needs no private import of its own.
ClickException = _click.ClickException


def _debug_option_callback(ctx: _click.Context, _param: _click.Parameter, value: bool) -> None:
    """Turn on debug logging."""
    if value and not ctx.resilient_parsing:
        app_logger = logging.getLogger(LOGGER_NAME)
        app_logger.setLevel(logging.DEBUG)
        app_logger.debug("Debug logging enabled")


def _make_debug_option() -> TyperOption:
    """A ``--debug`` option that can be attached at any level."""
    return TyperOption(
        param_decls=["--debug"],
        is_flag=True,
        is_eager=True,
        expose_value=False,
        help=DEBUG_OPTION_HELP,
        callback=_debug_option_callback,
    )


def _has_debug_option(params: list[_click.Parameter] | None) -> bool:
    return any(isinstance(param, TyperOption) and "--debug" in param.opts for param in params or [])


def _with_debug_option(params: list[_click.Parameter] | None) -> list[_click.Parameter]:
    normalized = list(params or [])
    if _has_debug_option(normalized):
        return normalized
    return [_make_debug_option(), *normalized]


def _data_dir_option_callback(ctx: _click.Context, _param: _click.Parameter, value: str | None) -> None:
    """Point this command, and the job workers it starts, at another data directory: through the environment variable
    every way of opening the services reads."""
    if value and not ctx.resilient_parsing:
        os.environ[DATA_DIR_ENV] = str(Path(value).expanduser().resolve())


def _make_data_dir_option() -> TyperOption:
    """A ``--data-dir`` option that can be attached at any level, before or after the command's name."""
    return TyperOption(
        param_decls=["--data-dir"],
        is_eager=True,
        expose_value=False,
        metavar="DIR",
        help=DATA_DIR_OPTION_HELP,
        callback=_data_dir_option_callback,
    )


def _with_shared_options(params: list[_click.Parameter] | None) -> list[_click.Parameter]:
    """``params`` with ``--debug`` and ``--data-dir``, unless the command defines them itself (``webui`` has its own
    ``--data-dir``)."""
    normalized = _with_debug_option(params)
    if any(isinstance(param, TyperOption) and "--data-dir" in param.opts for param in normalized):
        return normalized
    return [*normalized, _make_data_dir_option()]


def report_error(error: "MizuotoError") -> None:
    """Log a domain error's message, and the command that fixes it when the error names one."""
    log = logging.getLogger(LOGGER_NAME)
    log.error("%s", error.message)
    hint = error.detail.get("hint")
    if isinstance(hint, str) and hint:
        log.error("%s", hint)


class DebugTyperCommand(TyperCommand):
    """A Typer command that always has ``--debug`` and turns domain errors into their exit codes."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        kwargs["params"] = _with_shared_options(kwargs.get("params"))
        super().__init__(*args, **kwargs)

    def invoke(self, ctx: _click.Context) -> Any:
        from mizuoto.core.errors import MizuotoError

        try:
            return super().invoke(ctx)
        except MizuotoError as e:
            report_error(e)
            raise typer.Exit(e.exit_code) from None


class AlphabeticalMixedGroup(TyperGroup):
    """A group with ``--debug`` that lists commands and subgroups alphabetically."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        kwargs["params"] = _with_shared_options(kwargs.get("params"))
        super().__init__(*args, **kwargs)

    def list_commands(self, ctx: _click.Context) -> list[str]:  # type: ignore[override]
        return sorted(self.commands.keys())


class DebugOptionTyper(typer.Typer):
    """A Typer app whose commands use DebugTyperCommand by default."""

    def command(self, *args: Any, **kwargs: Any) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
        kwargs.setdefault("cls", DebugTyperCommand)
        return super().command(*args, **kwargs)


def typer_factory(help: str) -> typer.Typer:
    """Create an app or group with the shared settings."""
    return DebugOptionTyper(
        help=help,
        add_completion=True,
        no_args_is_help=True,
        cls=AlphabeticalMixedGroup,
        rich_markup_mode=None,
        rich_help_panel=None,
        pretty_exceptions_enable=False,
    )
