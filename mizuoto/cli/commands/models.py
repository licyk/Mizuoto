"""model list | info | spec | download | verify | delete | import | speakers."""

from pathlib import Path
from typing import Annotated

import typer

from mizuoto.cli.output import console, err_console, human_size, open_services, print_json, print_table


def model_list(
    engine: Annotated[str | None, typer.Option("--engine", "-e", help="Only this engine's models")] = None,
    installed: Annotated[bool | None, typer.Option("--installed/--missing", help="Only installed (or only missing) models")] = None,
    json_output: Annotated[bool, typer.Option("--json", help="Print JSON")] = False,
) -> None:
    """List the models Mizuoto knows, and which are installed."""
    with open_services() as services:
        models = services.models.list_models(engine, installed)
        if json_output:
            print_json(models)
            return
        print_table(None, ["Id", "Variant", "State", "Title"], [(m.id, m.variant, m.state, m.title) for m in models])


def model_info(model_id: Annotated[str, typer.Argument(help="Model id")], json_output: Annotated[bool, typer.Option("--json", help="Print JSON")] = False) -> None:
    """Show a model."""
    with open_services() as services:
        info = services.models.get(model_id)
        if json_output:
            print_json(info)
            return
        print_table(info.title, ["Field", "Value"], [(k, v) for k, v in info.model_dump().items() if v not in (None, "", [])])


def model_spec(model_id: Annotated[str, typer.Argument(help="Model id")], json_output: Annotated[bool, typer.Option("--json", help="Print JSON")] = False) -> None:
    """Show what a model takes: voice modes, languages, parameters and options."""
    with open_services() as services:
        spec = services.tts.model_spec(model_id)
        if json_output:
            print_json(spec)
            return
        console.print(f"[bold]{spec.model}[/bold] · {spec.sample_rate} Hz · voices: {', '.join(spec.voice_modes)} · languages: {', '.join(spec.languages)}", highlight=False)
        console.print(f"Capabilities: {', '.join(spec.capabilities)}", highlight=False)
        print_table(
            "Parameters",
            ["Name", "Kind", "Type", "Range", "Default", "Help"],
            [(p.name, p.kind, p.type, ", ".join(p.choices) if p.choices else f"{p.min}–{p.max}", p.default, p.help) for p in spec.params],
        )


def model_download(
    model_ids: Annotated[list[str], typer.Argument(help="Catalogue model ids")],
    json_output: Annotated[bool, typer.Option("--json", help="Print JSON")] = False,
) -> None:
    """Download catalogue models (resumes an interrupted download; verifies every file)."""
    from mizuoto.cli.progress import finish, job_progress

    with open_services() as services:
        with job_progress(services, enabled=not json_output):
            job = services.models.download(model_ids, wait=True, origin="cli")
        if json_output:
            print_json(job)
        elif job.state == "completed" and job.result:
            for model_id, result in job.result.get("models", {}).items():
                if result == "installed":
                    console.print(f"{model_id}: already installed", highlight=False)
                else:
                    sources = ", ".join(f"{n} from {s}" for s, n in result["sources"].items()) or "nothing to fetch"
                    console.print(f"{model_id}: {result['files']} files, {human_size(result['bytes'])} downloaded ({sources})", highlight=False)
        finish(job)


def model_verify(model_id: Annotated[str, typer.Argument(help="Model id")], json_output: Annotated[bool, typer.Option("--json", help="Print JSON")] = False) -> None:
    """Check every file of an installed model against its SHA-256."""
    with open_services() as services:
        result = services.models.verify(model_id)
        if json_output:
            print_json(result)
        elif result.ok:
            console.print(f"{model_id}: {result.checked} files verified", highlight=False)
        else:
            err_console.print(f"[red]{model_id}: damaged files:[/red] {', '.join(result.bad)}", highlight=False)
        if not result.ok:
            raise typer.Exit(1)


def model_delete(model_ids: Annotated[list[str], typer.Argument(help="Model ids")]) -> None:
    """Delete downloaded or imported models."""
    with open_services() as services:
        for model_id in model_ids:
            services.models.delete(model_id)
            console.print(f"Deleted {model_id}", highlight=False)


def model_import(
    path: Annotated[Path, typer.Argument(help="A folder holding a model")],
    name: Annotated[str | None, typer.Option("--name", help="Import as <engine>/<name> rather than as the catalogue model it matches")] = None,
    move: Annotated[bool, typer.Option("--move", help="Move the folder instead of copying it")] = False,
    json_output: Annotated[bool, typer.Option("--json", help="Print JSON")] = False,
) -> None:
    """Add a model folder from elsewhere (an earlier download, a fine-tune)."""
    with open_services() as services:
        info = services.models.import_model(path, name=name, move=move)
        if json_output:
            print_json(info)
            return
        console.print(f"Imported {info.id} ({info.title}, variant {info.variant})", highlight=False)


def model_speakers(model_id: Annotated[str, typer.Argument(help="Model id")], json_output: Annotated[bool, typer.Option("--json", help="Print JSON")] = False) -> None:
    """List a model's preset speakers."""
    with open_services() as services:
        speakers = services.tts.model_spec(model_id).speakers
        if json_output:
            print_json(speakers)
            return
        for s in speakers:
            console.print(s, highlight=False)
