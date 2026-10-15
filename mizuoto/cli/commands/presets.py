"""preset list | save | delete | default."""

from typing import Annotated

import typer

from mizuoto.cli.commands.speak import parse_pairs
from mizuoto.cli.output import console, open_services, print_json, print_table


def preset_list(
    model: Annotated[str | None, typer.Option("--model", "-m", help="Only those this model can use")] = None,
    json_output: Annotated[bool, typer.Option("--json", help="Print JSON")] = False,
) -> None:
    """List presets."""
    with open_services() as services:
        presets = services.presets.list_presets(model)
        if json_output:
            print_json(presets)
            return
        rows = [(p.id, p.name, p.model or "(any)", "yes" if p.is_default else "", ", ".join(f"{k}={v}" for k, v in {**p.params, **p.engine_options}.items())) for p in presets]
        print_table(None, ["Id", "Name", "Model", "Default", "Values"], rows)


def preset_save(
    name: Annotated[str, typer.Argument(help="The preset's name")],
    model: Annotated[str | None, typer.Option("--model", "-m", help="The model it is for (none: common parameters for any model)")] = None,
    param: Annotated[list[str] | None, typer.Option("--param", "-p", help="A common parameter name=value (repeatable)")] = None,
    option: Annotated[list[str] | None, typer.Option("--option", "-O", help="An engine option name=value (repeatable)")] = None,
    voice: Annotated[str | None, typer.Option("--voice", "-v", help="The voice its default status is for")] = None,
    default: Annotated[bool, typer.Option("--default", help="Apply it whenever the model is picked in the web UI")] = False,
    json_output: Annotated[bool, typer.Option("--json", help="Print JSON")] = False,
) -> None:
    """Save a named set of parameters."""
    from mizuoto.core.presets.models import PresetInput

    with open_services() as services:
        voice_id = services.voices.get(voice).id if voice else None
        preset = services.presets.create(
            PresetInput(name=name, model=model, voice_id=voice_id, params=parse_pairs(param, "--param"), engine_options=parse_pairs(option, "--option"))
        )
        if default:
            preset = services.presets.set_default(preset.id)
        if json_output:
            print_json(preset)
        else:
            console.print(f"Saved {preset.name} ({preset.id})", highlight=False)


def preset_delete(preset: Annotated[str, typer.Argument(help="Preset id or name")]) -> None:
    """Delete a preset."""
    with open_services() as services:
        services.presets.delete(preset)
        console.print("Deleted")


def preset_default(
    preset: Annotated[str, typer.Argument(help="Preset id or name")],
    off: Annotated[bool, typer.Option("--off", help="Stop it being the default")] = False,
) -> None:
    """Make a preset its model's default (applied when the model is picked in the web UI)."""
    with open_services() as services:
        p = services.presets.set_default(preset, not off)
        console.print(f"{p.name}: {'default' if p.is_default else 'not default'}", highlight=False)
