"""train new | list | status | run | export | metrics | edit | delete."""

from __future__ import annotations

from typing import Annotated, Any

import typer

from mizuoto.cli.commands.speak import parse_pairs
from mizuoto.cli.output import console, open_services, print_json, print_table, short

STATUS_STYLE = {"done": "green", "running": "cyan", "stale": "yellow", "failed": "red", "pending": "dim"}


def _status(exp: Any) -> None:
    console.print(f"[bold]{exp.key}[/bold] · {exp.method} of {exp.base_model} on {exp.dataset}", highlight=False)
    if exp.hyperparams:
        console.print("Hyperparameters: " + ", ".join(f"{k}={v}" for k, v in exp.hyperparams.items()), highlight=False)
    print_table(
        None,
        ["Stage", "Status", "Finished", "Note"],
        [(s.title, f"[{STATUS_STYLE[s.status]}]{s.status}[/]", (s.finished_at or "-")[:19], short(s.error or "", 60)) for s in exp.stages],
    )
    for a in exp.artifacts:
        console.print(f"Exported {a.kind} {a.id}" + (f" ({a.title})" if a.title else ""), highlight=False)


def train_new(
    name: Annotated[str, typer.Argument(help="The experiment's name")],
    base: Annotated[str, typer.Option("--base", "-m", help="The model to fine-tune, e.g. gptsovits/v2pro-base")],
    dataset: Annotated[str, typer.Option("--dataset", "-d", help="Dataset id or slug")],
    method: Annotated[str | None, typer.Option("--method", help="A training method of the engine (default: its first for the model)")] = None,
    hp: Annotated[list[str] | None, typer.Option("--hp", help="A hyperparameter key=value (repeatable)")] = None,
    voice: Annotated[str | None, typer.Option("--voice", help="Export voice weights to this library voice (default: a new voice)")] = None,
    lang: Annotated[str | None, typer.Option("--lang", "-l", help="The dataset's language, when it does not say")] = None,
    json_output: Annotated[bool, typer.Option("--json", help="Print JSON")] = False,
) -> None:
    """Create an experiment: a base model, a dataset and a method."""
    from mizuoto.core.training.models import ExperimentInput

    with open_services() as services:
        exp = services.training.create(ExperimentInput(name=name, base_model=base, dataset=dataset, method=method, hyperparams=parse_pairs(hp, "--hp"), voice=voice, language=lang))
        if json_output:
            print_json(exp)
            return
        console.print(f"Created {exp.key}; run it with: mizuoto train run {exp.key}", highlight=False)


def train_list(json_output: Annotated[bool, typer.Option("--json", help="Print JSON")] = False) -> None:
    """List the experiments."""
    with open_services() as services:
        exps = services.training.list_experiments()
        if json_output:
            print_json(exps)
            return

        def progress(exp: Any) -> str:
            done = sum(s.status == "done" for s in exp.stages)
            bad = next((s.status for s in exp.stages if s.status in ("running", "failed", "stale")), None)
            return f"{done}/{len(exp.stages)}" + (f" ({bad})" if bad else "")

        print_table(
            None, ["Experiment", "Method", "Base model", "Dataset", "Stages", "Updated"], [(e.key, e.method, e.base_model, e.dataset, progress(e), e.updated_at[:19]) for e in exps]
        )


def train_status(
    experiment: Annotated[str, typer.Argument(help="<engine>/<name>, or a name")], json_output: Annotated[bool, typer.Option("--json", help="Print JSON")] = False
) -> None:
    """Show an experiment's stages and exports."""
    with open_services() as services:
        exp = services.training.get(experiment)
        if json_output:
            print_json(exp)
            return
        _status(exp)


def _run(experiment: str, stages: list[str] | None, restart: bool, json_output: bool) -> None:
    from mizuoto.cli.progress import finish, job_progress
    from mizuoto.core.training.models import RunRequest

    with open_services() as services:
        with job_progress(services, enabled=not json_output):
            job = services.training.run(experiment, RunRequest(stages=stages, restart=restart), wait=True, origin="cli")
        if json_output:
            print_json(job)
        else:
            _status(services.training.get(experiment))
        finish(job)


def train_run(
    experiment: Annotated[str, typer.Argument(help="<engine>/<name>, or a name")],
    stage: Annotated[list[str] | None, typer.Option("--stage", "-s", help="Run only this stage (repeatable; the earlier ones must be done)")] = None,
    restart: Annotated[bool, typer.Option("--restart", help="Run every stage again from the first")] = False,
    json_output: Annotated[bool, typer.Option("--json", help="Print JSON")] = False,
) -> None:
    """Run an experiment's stages: by default every stage from the first not done."""
    _run(experiment, list(stage) if stage else None, restart, json_output)


def train_export(
    experiment: Annotated[str, typer.Argument(help="<engine>/<name>, or a name")],
    json_output: Annotated[bool, typer.Option("--json", help="Print JSON")] = False,
) -> None:
    """Run the export stage again: register the trained weights as a voice's or a model."""
    _run(experiment, ["export"], False, json_output)


def train_metrics(
    experiment: Annotated[str, typer.Argument(help="<engine>/<name>, or a name")],
    stage: Annotated[str | None, typer.Option("--stage", "-s", help="Only this stage")] = None,
    json_output: Annotated[bool, typer.Option("--json", help="Print JSON")] = False,
) -> None:
    """Print the losses the stages reported: first, last and lowest of each."""
    with open_services() as services:
        points = services.training.metrics(experiment, stage)
        if json_output:
            print_json(points)
            return
        series: dict[tuple[str, str], list[tuple[int, float]]] = {}
        for p in points:
            for name, value in p.values.items():
                series.setdefault((p.stage, name), []).append((p.step, value))
        rows = [(s, n, len(v), f"{v[0][1]:.4g}", f"{v[-1][1]:.4g}", f"{min(x for _, x in v):.4g}", v[-1][0]) for (s, n), v in series.items()]
        print_table(None, ["Stage", "Metric", "Points", "First", "Last", "Lowest", "Last step"], rows)


def train_edit(
    experiment: Annotated[str, typer.Argument(help="<engine>/<name>, or a name")],
    hp: Annotated[list[str] | None, typer.Option("--hp", help="A hyperparameter key=value (repeatable; key= resets it)")] = None,
    voice: Annotated[str | None, typer.Option("--voice", help="Export voice weights to this library voice")] = None,
    lang: Annotated[str | None, typer.Option("--lang", "-l", help="The dataset's language")] = None,
) -> None:
    """Change hyperparameters (the stages that read them become stale)."""
    from mizuoto.core.training.models import ExperimentPatch

    with open_services() as services:
        exp = services.training.get(experiment)
        hyperparams = None
        if hp:
            hyperparams = dict(exp.hyperparams)
            for key, value in parse_pairs(hp, "--hp").items():
                if value == "":
                    hyperparams.pop(key, None)
                else:
                    hyperparams[key] = value
        _status(services.training.update(exp.key, ExperimentPatch(hyperparams=hyperparams, voice=voice, language=lang)))


def train_delete(
    experiment: Annotated[str, typer.Argument(help="<engine>/<name>, or a name")], yes: Annotated[bool, typer.Option("--yes", "-y", help="Do not ask")] = False
) -> None:
    """Move an experiment and its work files to the trash (exports stay)."""
    if not yes and not typer.confirm(f"Delete {experiment}?"):
        raise typer.Exit(1)
    with open_services() as services:
        services.training.delete(experiment)
        console.print("Deleted")


def train_methods(
    model: Annotated[str, typer.Argument(help="A base model, e.g. gptsovits/v2pro-base")], json_output: Annotated[bool, typer.Option("--json", help="Print JSON")] = False
) -> None:
    """List the ways a model can be fine-tuned, with their stages and hyperparameters."""
    with open_services() as services:
        methods = services.training.methods(model)
        if json_output:
            print_json(methods)
            return
        if not methods:
            console.print(f"{model}'s engine has no training in Mizuoto", highlight=False)
        for m in methods:
            state = "[green]available[/green]" if m.available else f"[yellow]{m.reason}[/yellow]"
            console.print(f"[bold]{m.id}[/bold] · {m.method.title} → {m.method.export.replace('_', ' ')} · {state}", highlight=False)
            if m.method.description:
                console.print(f"  {m.method.description}", highlight=False)
            console.print("  Stages: " + " → ".join(s.title for s in m.method.stages), highlight=False)
            for name, spec in m.method.hyperparams.items():
                bounds = f" [{_num(spec.min)}, {_num(spec.max)}]" if spec.min is not None or spec.max is not None else ""
                bounds = f" ({', '.join(spec.choices)})" if spec.choices else bounds
                console.print(f"  --hp {name}={spec.default}{bounds}  {spec.help or ''}", highlight=False)


def _num(value: float | None) -> str:
    return "-" if value is None else f"{value:g}"
