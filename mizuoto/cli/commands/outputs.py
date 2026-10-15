"""output list | info | delete."""

from typing import Annotated

import typer

from mizuoto.cli.output import console, open_services, print_json, print_table, short


def output_list(
    voice: Annotated[str | None, typer.Option("--voice", help="Only this voice")] = None,
    model: Annotated[str | None, typer.Option("--model", "-m", help="Only this model")] = None,
    search: Annotated[str | None, typer.Option("--search", "-s", help="Text contains")] = None,
    limit: Annotated[int, typer.Option(min=1, max=500)] = 30,
    json_output: Annotated[bool, typer.Option("--json", help="Print JSON")] = False,
) -> None:
    """List outputs, newest first."""
    from mizuoto.core.outputs.models import OutputQuery

    with open_services() as services:
        page = services.outputs.list_outputs(OutputQuery(voice_id=voice, model=model, text=search, limit=limit))
        if json_output:
            print_json(page)
            return
        print_table(None, ["Id", "Model", "Seconds", "Text", "File"], [(o.id, o.model, f"{o.duration:.1f}", short(o.text, 40), o.path) for o in page.items])


def output_info(output_id: Annotated[str, typer.Argument(help="Output id")], json_output: Annotated[bool, typer.Option("--json", help="Print JSON")] = False) -> None:
    """Show an output and how it was made."""
    with open_services() as services:
        out = services.outputs.get(output_id)
        if json_output:
            print_json(out)
            return
        console.print_json(out.model_dump_json())


def output_delete(ids: Annotated[list[str], typer.Argument(help="Output ids")], yes: Annotated[bool, typer.Option("--yes", "-y", help="Do not ask")] = False) -> None:
    """Delete outputs (their files go to the trash)."""
    if not yes and not typer.confirm(f"Delete {len(ids)} output(s)?"):
        raise typer.Exit(1)
    with open_services() as services:
        removed = services.outputs.delete(ids)
        console.print(f"Deleted {len(removed)}")
