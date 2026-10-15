"""engine list | info | accept-licence | enable | disable."""

from typing import Annotated

import typer

from mizuoto.cli.output import console, open_services, print_json, print_table


def engine_list(json_output: Annotated[bool, typer.Option("--json", help="Print JSON")] = False) -> None:
    """List the engines, their licences and whether they are available."""
    with open_services() as services:
        engines = services.engines.list_engines()
        if json_output:
            print_json(engines)
            return
        print_table(
            None,
            ["Engine", "Title", "Available", "Licence", "Enabled", "Models"],
            [
                (
                    e.id,
                    e.title,
                    "yes" if e.available else "needs " + ", ".join(e.missing),
                    e.licence.name + ("" if e.licence_accepted else " (not accepted)"),
                    "yes" if e.enabled else "no",
                    len(e.models),
                )
                for e in engines
            ],
        )


def engine_info(engine_id: Annotated[str, typer.Argument(help="Engine id")], json_output: Annotated[bool, typer.Option("--json", help="Print JSON")] = False) -> None:
    """Show an engine: variants, capabilities, parameters and models."""
    with open_services() as services:
        manifest = services.engines.manifest(engine_id)
        status = services.engines.status(engine_id)
        if json_output:
            print_json({"status": status, "manifest": manifest})
            return
        console.print(f"[bold]{manifest.title}[/bold] ({manifest.id}) — {manifest.description}", highlight=False)
        console.print(
            f"Licence: {manifest.licence.name}{'' if status.licence_accepted else ' — not accepted'}   Available: {'yes' if status.available else 'needs ' + ', '.join(status.missing)}",
            highlight=False,
        )
        for v in manifest.variants.values():
            console.print(f"\n[bold]{v.id}[/bold] {v.title} · {v.sample_rate} Hz · {', '.join(v.capabilities)}", highlight=False)
        print_table("Parameters", ["Name", "Engine name", "Range", "Default"], [(k, p.target, f"{p.min}–{p.max}", p.default) for k, p in manifest.params.items()])
        print_table("Models", ["Id", "Variant", "Title"], [(m.id, m.variant, m.title) for m in manifest.models])


def engine_accept_licence(engine_id: Annotated[str, typer.Argument(help="Engine id")], yes: Annotated[bool, typer.Option("--yes", help="Accept without asking")] = False) -> None:
    """Accept a licence-gated engine's licence (after reading it)."""
    with open_services() as services:
        manifest = services.engines.manifest(engine_id)
        if not manifest.licence.gated:
            console.print(f"{manifest.title} needs no licence acceptance ({manifest.licence.name}).")
            return
        console.print(f"{manifest.title} is licensed under {manifest.licence.name}: {manifest.licence.url}", highlight=False)
        if not yes:
            text = services.engines.licence_text(engine_id).text
            if text:
                console.print(text, highlight=False, markup=False)
        if not yes and not typer.confirm("Have you read it, and do you accept it?"):
            raise typer.Exit(1)
        services.engines.accept_licence(engine_id)
        console.print("Accepted.")


def _set_enabled(engine_id: str, enabled: bool) -> None:
    with open_services() as services:
        services.engines.manifest(engine_id)
        services.engines.configure(engine_id, {"enabled": enabled})
        console.print(f"{engine_id}: {'enabled' if enabled else 'disabled'}")


def engine_enable(engine_id: Annotated[str, typer.Argument(help="Engine id")]) -> None:
    """Enable an engine."""
    _set_enabled(engine_id, True)


def engine_disable(engine_id: Annotated[str, typer.Argument(help="Engine id")]) -> None:
    """Disable an engine (its models stay installed)."""
    _set_enabled(engine_id, False)


def engine_licence(engine_id: Annotated[str, typer.Argument(help="Engine id")], json_output: Annotated[bool, typer.Option("--json", help="Print JSON")] = False) -> None:
    """Print an engine's licence."""
    with open_services() as services:
        info = services.engines.licence_text(engine_id)
        if json_output:
            print_json(info)
            return
        console.print(f"{info.name} ({info.key}){' — accepted' if info.accepted else ''}", highlight=False)
        console.print(info.text or f"Read it at {info.url}", highlight=False, markup=False)


def engine_op(
    model: Annotated[str, typer.Argument(help="The model to load, e.g. gptsovits/v2pro-base")],
    op: Annotated[str, typer.Argument(help="An operation the engine declares (mizuoto engine info <engine>)")],
    value: Annotated[list[str] | None, typer.Option("--set", "-s", help="A payload value key=value (repeatable)")] = None,
    json_output: Annotated[bool, typer.Option("--json", help="Print JSON")] = False,
) -> None:
    """Run an engine-specific operation (GPT-SoVITS switch_weights, Fish Speech codec_encode…)."""
    from mizuoto.cli.commands.speak import parse_pairs
    from mizuoto.cli.progress import finish, job_progress

    with open_services() as services:
        with job_progress(services, enabled=not json_output):
            job = services.tts.engine_op(model, op, parse_pairs(value, "--set"), wait=True, origin="cli")
        if json_output:
            print_json(job)
        elif job.state == "completed":
            console.print_json(data=job.result)
        finish(job)
