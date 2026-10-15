"""config show | get | set | path."""

from typing import TYPE_CHECKING, Annotated

import typer

from mizuoto.cli.output import console, print_json

if TYPE_CHECKING:
    from mizuoto.core.settings import SettingsService


def _settings() -> "SettingsService":
    from mizuoto.core.settings import SettingsService

    return SettingsService()


def config_show(json_output: Annotated[bool, typer.Option("--json", help="Print JSON")] = False) -> None:
    """Show the effective settings. The access token is shown only as configured or not."""
    view = _settings().view()
    if json_output:
        print_json(view)
        return
    console.print_json(view.model_dump_json())


def config_get(key: Annotated[str, typer.Argument(help="Dotted key, e.g. server.port")]) -> None:
    """Print one setting."""
    print_json(_settings().get_value(key))


def config_set(
    key: Annotated[str, typer.Argument(help="Dotted key, e.g. server.port or tts.default_model")],
    value: Annotated[str, typer.Argument(help="New value. JSON is parsed (numbers, true/false, lists); 'null' clears")],
) -> None:
    """Change one setting and save it."""
    service = _settings()
    service.set_value(key, value)
    shown = "(hidden)" if key.endswith("token") else service.get_value(key)
    console.print(f"{key} = {shown}")
    if f"MIZUOTO_{key.upper().replace('.', '__')}" in service.env_override_names():
        console.print("[yellow]An environment variable overrides this setting.[/yellow]")


def config_path() -> None:
    """Print the path of the settings file."""
    print(_settings().path)
