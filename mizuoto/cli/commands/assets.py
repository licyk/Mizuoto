"""asset list | download | delete."""

from typing import Annotated

import typer

from mizuoto.cli.output import console, human_size, open_services, print_json, print_table


def asset_list(
    engine: Annotated[str | None, typer.Option("--engine", "-e", help="Only this engine's assets")] = None,
    json_output: Annotated[bool, typer.Option("--json", help="Print JSON")] = False,
) -> None:
    """List the shared assets and the models that need them."""
    with open_services() as services:
        assets = services.models.list_assets(engine)
        if json_output:
            print_json(assets)
            return
        rows = [(a.id, a.state, human_size((a.size_mb or 0) * 1024 * 1024), ", ".join(a.used_by)) for a in assets]
        print_table(None, ["Id", "State", "Size", "Used by"], rows)


def asset_download(asset_ids: Annotated[list[str], typer.Argument(help="Asset ids")], json_output: Annotated[bool, typer.Option("--json", help="Print JSON")] = False) -> None:
    """Download shared assets (a model's are downloaded with it)."""
    from mizuoto.cli.progress import finish, job_progress

    with open_services() as services:
        with job_progress(services, enabled=not json_output):
            job = services.models.download_assets(asset_ids, wait=True, origin="cli")
        if json_output:
            print_json(job)
        elif job.state == "completed" and job.result:
            for asset_id, result in job.result.get("assets", {}).items():
                done = "already installed" if result == "installed" else f"{result['files']} files"
                console.print(f"{asset_id}: {done}", highlight=False)
        finish(job)


def asset_delete(asset_id: Annotated[str, typer.Argument(help="Asset id")], yes: Annotated[bool, typer.Option("--yes", "-y", help="Do not ask")] = False) -> None:
    """Delete a shared asset (the models that need it stop working until it is downloaded again)."""
    if not yes and not typer.confirm(f"Delete {asset_id}?"):
        raise typer.Exit(1)
    with open_services() as services:
        services.models.delete_asset(asset_id)
        console.print("Deleted")
