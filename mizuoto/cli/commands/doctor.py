"""doctor: what stands between this machine and the engines."""

from __future__ import annotations

from typing import Annotated

import typer

from mizuoto.cli.output import console, open_services, print_json


def doctor(
    engine_id: Annotated[str | None, typer.Argument(help="Engine id (default: all)")] = None,
    json_output: Annotated[bool, typer.Option("--json", help="Print JSON")] = False,
) -> None:
    """List the GPUs found (NVIDIA, AMD, Intel, Apple) and check torch against them, with the command that installs
    the right build; list each engine's availability and missing optional extras."""
    with open_services() as services:
        if engine_id is not None:
            services.engines.manifest(engine_id)
        report = services.runtime.doctor(engine_id)
        if json_output:
            print_json(report)
        else:
            from rich.markup import escape

            style = {"ok": "green", "warning": "yellow", "error": "red"}
            built = f"torch {report.torch} ({report.torch_build})" if report.torch else "torch: not installed"
            console.print(f"{built}; this machine's build: {report.recommended_build}", highlight=False, markup=False)
            for f in report.findings:
                text = escape(f.message) + (f"\n         fix: {escape(f.fix)}" if f.fix else "")
                console.print(f"[{style[f.level]}]{f.level:<7}[/{style[f.level]}] {text}", highlight=False, soft_wrap=True)
        if any(f.level == "error" for f in report.findings):
            raise typer.Exit(1)
