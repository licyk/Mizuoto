"""convert, edit: work on a recording."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Annotated

import typer

from mizuoto.cli.output import console, err_console, open_services, print_json


def _finish(services, job, output: Path | None, json_output: bool) -> None:  # noqa: ANN001
    from mizuoto.cli.progress import finish

    if job.state == "completed" and job.result:
        out = services.outputs.get(job.result["output_id"])
        for w in job.result.get("warnings") or []:
            err_console.print(f"[yellow]warning:[/yellow] {w}", highlight=False)
        if output is not None:
            output.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(out.path, output)
        if json_output:
            print_json({"job": job, "output": out})
        else:
            console.print(f"{out.path}  ({out.duration:.2f} s, {out.sample_rate} Hz)", highlight=False)
    elif json_output:
        print_json({"job": job, "output": None})
    finish(job)


def convert(
    source: Annotated[Path, typer.Argument(help="The recording to convert")],
    model: Annotated[str, typer.Option("--model", "-m", help="A model with voice conversion, e.g. cosyvoice/cosyvoice2-0.5b")],
    voice: Annotated[str | None, typer.Option("--voice", "-v", help="The target voice (a library voice)")] = None,
    ref: Annotated[Path | None, typer.Option("--ref", help="Or a clip of the target voice")] = None,
    output: Annotated[Path | None, typer.Option("--output", "-o", help="Also copy the result here")] = None,
    seed: Annotated[int | None, typer.Option("--seed", help="Random seed")] = None,
    json_output: Annotated[bool, typer.Option("--json", help="Print JSON")] = False,
) -> None:
    """Say what a recording says in another voice."""
    from mizuoto.cli.progress import job_progress
    from mizuoto.core.tts.models import CommonParams, ConvertRequest, ReferenceInput, SourceInput, VoiceInput

    with open_services() as services:
        req = ConvertRequest(
            model=model,
            source=SourceInput(path=str(source.expanduser().resolve())),
            voice=voice,
            voice_spec=VoiceInput(references=[ReferenceInput(path=str(ref.expanduser().resolve()))]) if ref else None,
            params=CommonParams(seed=seed),
        )
        with job_progress(services, enabled=not json_output):
            job = services.tts.convert(req, wait=True, origin="cli")
        _finish(services, job, output, json_output)


def edit(
    source: Annotated[Path, typer.Argument(help="The recording to edit")],
    model: Annotated[str, typer.Option("--model", "-m", help="A model with speech editing, e.g. f5tts/v1-base")],
    text: Annotated[str, typer.Option("--text", "-t", help="The recording's whole text after the edit")],
    span: Annotated[list[str], typer.Option("--span", help="START:END[:SECONDS] of a part to replace (repeatable), in seconds")],
    output: Annotated[Path | None, typer.Option("--output", "-o", help="Also copy the result here")] = None,
    seed: Annotated[int | None, typer.Option("--seed", help="Random seed")] = None,
    json_output: Annotated[bool, typer.Option("--json", help="Print JSON")] = False,
) -> None:
    """Replace parts of a recording with new speech in the same voice."""
    from mizuoto.cli.progress import job_progress
    from mizuoto.core.tts.models import CommonParams, EditRequest, EditSpanInput, SourceInput

    spans = []
    for item in span:
        parts = item.split(":")
        try:
            numbers = [float(x) for x in parts]
        except ValueError:
            raise typer.BadParameter(f"--span must be START:END[:SECONDS], got {item!r}") from None
        if len(numbers) not in (2, 3):
            raise typer.BadParameter(f"--span must be START:END[:SECONDS], got {item!r}")
        spans.append(EditSpanInput(start=numbers[0], end=numbers[1], text="", duration=numbers[2] if len(numbers) == 3 else None))
    with open_services() as services:
        req = EditRequest(model=model, source=SourceInput(path=str(source.expanduser().resolve())), transcript=text, edits=spans, params=CommonParams(seed=seed))
        with job_progress(services, enabled=not json_output):
            job = services.tts.edit(req, wait=True, origin="cli")
        _finish(services, job, output, json_output)
