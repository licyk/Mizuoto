"""Show a job's progress in the terminal, fed by the same events the web UI receives."""

from __future__ import annotations

import sys
from collections.abc import Callable, Generator
from contextlib import contextmanager
from typing import TYPE_CHECKING

from rich.progress import BarColumn, Progress, SpinnerColumn, TaskID, TaskProgressColumn, TextColumn, TimeElapsedColumn

from mizuoto.cli.output import err_console

if TYPE_CHECKING:
    from mizuoto.core.context import Services
    from mizuoto.core.events import EventBase
    from mizuoto.core.jobs.models import Job


@contextmanager
def job_progress(services: Services, enabled: bool = True) -> Generator[None, None, None]:
    """A Rich progress bar on stderr for every job that updates while the block runs."""
    from mizuoto.core.events.models import JobLogEvent, JobUpdatedEvent

    if not enabled or not sys.stderr.isatty():
        unsubscribe: Callable[[], None] = services.events.subscribe(lambda e: None)
        try:
            yield
        finally:
            unsubscribe()
        return
    progress = Progress(SpinnerColumn(), TextColumn("{task.description}"), BarColumn(), TaskProgressColumn(), TimeElapsedColumn(), console=err_console, transient=True)
    tasks: dict[str, TaskID] = {}

    def handler(event: EventBase) -> None:
        if isinstance(event, JobUpdatedEvent):
            job = event.job
            detail = job.message or (_step_title(job) if job.step else "")
            if job.transfer is not None:
                detail += " · " + _transfer_text(job)
            label = job.title + (f" · {detail}" if detail else "")
            if job.id not in tasks:
                tasks[job.id] = progress.add_task(label, total=1.0)
            progress.update(tasks[job.id], description=label, completed=job.progress or 0.0, total=1.0 if job.progress is not None else None)
        elif isinstance(event, JobLogEvent):
            for line in event.lines:
                if line.startswith(("error", "Traceback")):
                    progress.console.print(line, style="red", markup=False, highlight=False)

    unsubscribe = services.events.subscribe(handler)
    try:
        with progress:
            yield
    finally:
        unsubscribe()


def _step_title(job: Job) -> str:
    for s in job.steps:
        if s.id == job.step:
            return s.title
    return job.step or ""


def _transfer_text(job: Job) -> str:
    from mizuoto.cli.output import human_size

    t = job.transfer
    assert t is not None
    text = f"{human_size(t.done_bytes)} / {human_size(t.total_bytes)}"
    if t.bytes_per_second:
        text += f" · {human_size(t.bytes_per_second)}/s"
    if t.eta_seconds is not None:
        minutes, seconds = divmod(int(t.eta_seconds), 60)
        text += f" · {minutes}:{seconds:02d} left"
    return text


def finish(job: Job) -> None:
    """Report a foreground job's end, and exit with the domain error's code if it failed."""
    import typer

    from mizuoto.core.errors import error_from_code

    if job.state == "completed":
        return
    if job.state == "cancelled":
        err_console.print("[yellow]Cancelled[/yellow]")
        raise typer.Exit(130)
    error = job.error
    err_console.print(f"[red]{error.message if error else job.state}[/red]", markup=True, highlight=False)
    hint = error.detail.get("hint") if error else None
    if isinstance(hint, str) and hint:
        err_console.print(hint, style="yellow", markup=False, highlight=False)
    raise typer.Exit(error_from_code(error.code, "").exit_code if error else 1)
