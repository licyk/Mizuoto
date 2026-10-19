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


def output_check(
    fix: Annotated[bool, typer.Option("--fix", help="Rewrite the records of files found elsewhere, and old absolute ones inside the data folder")] = False,
    prune_missing: Annotated[bool, typer.Option("--prune-missing", help="Delete the records of files found nowhere")] = False,
    rebase: Annotated[tuple[str, str] | None, typer.Option("--rebase", metavar="OLD NEW", help="Records under folder OLD point under NEW instead")] = None,
    json_output: Annotated[bool, typer.Option("--json", help="Print JSON")] = False,
) -> None:
    """Check that every output and audio file is where its record says (they are found again after the data folder
    moved); list the missing ones. Exits 1 when some are missing."""
    from mizuoto.core.storage import PathCheckRequest

    with open_services() as services:
        report = services.storage.check(PathCheckRequest(fix=fix, prune_missing=prune_missing, rebase_from=rebase[0] if rebase else None, rebase_to=rebase[1] if rebase else None))
        if json_output:
            print_json(report)
        else:
            found = "rewritten" if report.fixed else "--fix rewrites their records"
            console.print(f"Data folder: {report.data_dir}", highlight=False, markup=False)
            console.print(f"{report.relative} inside it, {report.external} elsewhere, {report.relocated} found away from their records ({found})", highlight=False)
            if report.rebased:
                console.print(f"{report.rebased} rebased", highlight=False)
            if report.missing:
                print_table(f"Missing ({len(report.missing)})", ["Kind", "Id", "Name", "Recorded at"], [(m.kind, m.id, short(m.name, 40), m.recorded) for m in report.missing])
            if report.pruned:
                console.print(f"Deleted {report.pruned} records of missing files", highlight=False)
        if report.missing and not report.pruned:
            raise typer.Exit(1)
