"""asr: transcribe a clip."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from mizuoto.cli.output import console, open_services, print_json


def asr(
    audio: Annotated[Path, typer.Argument(help="An audio file")],
    lang: Annotated[str | None, typer.Option("--lang", "-l", help="Language code; default: detect")] = None,
    model: Annotated[str | None, typer.Option("--model", "-m", help="ASR model (default: tts.asr_model)")] = None,
    json_output: Annotated[bool, typer.Option("--json", help="Print JSON")] = False,
) -> None:
    """Transcribe a clip with the speech recognition engine."""
    from mizuoto.cli.progress import finish, job_progress

    with open_services() as services:
        with job_progress(services, enabled=not json_output):
            job = services.asr.submit(audio.expanduser().resolve(), lang, model, wait=True, origin="cli")
        if job.state == "completed" and job.result:
            if json_output:
                print_json(job.result)
            else:
                console.print(f"[{job.result.get('language')}] {job.result.get('text')}", highlight=False, markup=False)
        finish(job)
