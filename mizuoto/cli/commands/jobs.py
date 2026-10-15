"""job list | show | log | cancel | retry."""

from typing import Annotated

import typer

from mizuoto.cli.output import console, open_services, print_json, print_table


def job_list(
    state: Annotated[str | None, typer.Option(help="A state, 'active' or 'finished'")] = None,
    kind: Annotated[str | None, typer.Option(help="speak, design, train, …")] = None,
    limit: Annotated[int, typer.Option(min=1, max=200)] = 30,
    json_output: Annotated[bool, typer.Option("--json", help="Print JSON")] = False,
) -> None:
    """List jobs, newest first."""
    with open_services() as services:
        page = services.jobs.list_jobs(state=state, kind=kind, limit=limit)
        if json_output:
            print_json(page)
            return
        print_table(
            None,
            ["Id", "Kind", "State", "Progress", "Title", "Created"],
            [(j.id, j.kind, j.state, "" if j.progress is None else f"{j.progress:.0%}", j.title, j.created_at[:19]) for j in page.items],
        )


def job_show(
    job_id: Annotated[str, typer.Argument(help="Job id")],
    json_output: Annotated[bool, typer.Option("--json", help="Print JSON")] = False,
) -> None:
    """Show one job."""
    with open_services() as services:
        job = services.jobs.get(job_id)
        if json_output:
            print_json(job)
            return
        print_table(
            job.title,
            ["Field", "Value"],
            [
                ("Id", job.id),
                ("Kind", job.kind),
                ("State", job.state),
                ("Created", job.created_at),
                ("Finished", job.finished_at or ""),
                ("Error", f"{job.error.code}: {job.error.message}" if job.error else ""),
                ("Result", ", ".join(f"{k}={v}" for k, v in (job.result or {}).items() if k != "plan")),
                *((f"Step {s.title}", s.state) for s in job.steps),
            ],
        )


def job_log(job_id: Annotated[str, typer.Argument(help="Job id")]) -> None:
    """Print a job's log."""
    with open_services() as services:
        offset = 0
        while True:
            chunk = services.jobs.read_log(job_id, offset)
            for line in chunk.lines:
                console.print(line, markup=False, highlight=False)
            if chunk.next_offset == offset:
                break
            offset = chunk.next_offset


def job_cancel(job_id: Annotated[str, typer.Argument(help="Job id")]) -> None:
    """Cancel a job of a running server or one queued in the database."""
    with open_services() as services:
        print_json(services.jobs.cancel(job_id))


def job_retry(job_id: Annotated[str, typer.Argument(help="Job id")], json_output: Annotated[bool, typer.Option("--json", help="Print JSON")] = False) -> None:
    """Run a failed, cancelled or interrupted job again, in the foreground."""
    from mizuoto.cli.progress import finish, job_progress

    with open_services() as services:
        old = services.jobs.get(job_id)
        with job_progress(services, enabled=not json_output):
            job = services.jobs.run_now(old.kind, old.request, "cli")
        if json_output:
            print_json(job)
        finish(job)
