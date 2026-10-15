"""dataset list | create | info | import | add | prepare | check | rows | edit-row | delete."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Any

import typer

from mizuoto.cli.output import console, err_console, open_services, print_json, print_table, short


def _warn(warnings: list[str]) -> None:
    for w in warnings:
        err_console.print(f"[yellow]warning:[/yellow] {w}", highlight=False)


def _summary(services: Any, key: str) -> None:
    d = services.datasets.get(key)
    s = d.stats
    console.print(f"{d.name} ({d.slug}): {s.rows} clip(s), {s.seconds / 60:.1f} min, {s.missing_text} without text, {s.excluded} excluded", highlight=False)


def dataset_list(json_output: Annotated[bool, typer.Option("--json", help="Print JSON")] = False) -> None:
    """List the datasets."""
    with open_services() as services:
        datasets = services.datasets.list_datasets()
        if json_output:
            print_json(datasets)
            return
        rows = [(d.slug, d.name, d.language or "-", d.stats.rows, f"{d.stats.seconds / 60:.1f}", d.stats.missing_text, ", ".join(d.stats.speakers) or "-") for d in datasets]
        print_table(None, ["Slug", "Name", "Language", "Clips", "Minutes", "No text", "Speakers"], rows)


def dataset_create(
    name: Annotated[str, typer.Argument(help="The dataset's name")],
    lang: Annotated[str | None, typer.Option("--lang", "-l", help="Its language")] = None,
    consent: Annotated[str, typer.Option("--consent", help="Whose voice it is and how you may use it")] = "",
    description: Annotated[str, typer.Option("--description", help="A note")] = "",
    json_output: Annotated[bool, typer.Option("--json", help="Print JSON")] = False,
) -> None:
    """Create an empty dataset."""
    with open_services() as services:
        d = services.datasets.create(name, lang, consent, description)
        if json_output:
            print_json(d)
            return
        console.print(f"Created {d.name} ({d.slug})", highlight=False)


def dataset_info(dataset: Annotated[str, typer.Argument(help="Dataset id or slug")], json_output: Annotated[bool, typer.Option("--json", help="Print JSON")] = False) -> None:
    """Show a dataset."""
    with open_services() as services:
        d = services.datasets.get(dataset)
        if json_output:
            print_json(d)
            return
        console.print(f"[bold]{d.name}[/bold] ({d.slug}, {d.id}) · {d.language or 'any language'}", highlight=False)
        if d.description:
            console.print(d.description, highlight=False)
        if d.consent:
            console.print(f"Consent: {d.consent}", highlight=False)
        s = d.stats
        console.print(
            f"{s.rows} clip(s), {s.seconds / 60:.1f} min · {s.missing_text} without text · {s.excluded} excluded · speakers: {', '.join(s.speakers) or '-'}", highlight=False
        )
        console.print(f"Fingerprint {d.fingerprint[:16]}", highlight=False)


def dataset_import(
    dataset: Annotated[str, typer.Argument(help="Dataset id or slug")],
    path: Annotated[Path, typer.Argument(help="A folder: a GPT-SoVITS .list, an LJSpeech metadata.csv, a CSV, or audio files")],
    layout: Annotated[str, typer.Option("--layout", help="auto, plain, gptsovits_list, ljspeech or csv")] = "auto",
    json_output: Annotated[bool, typer.Option("--json", help="Print JSON")] = False,
) -> None:
    """Add a folder of clips with their transcripts."""
    if layout not in ("auto", "plain", "gptsovits_list", "ljspeech", "csv"):
        raise typer.BadParameter(f"Unknown layout {layout!r}")
    with open_services() as services:
        d, warnings = services.datasets.import_folder(dataset, path.expanduser().resolve(), layout)  # type: ignore[arg-type]
        if json_output:
            print_json({"dataset": d, "warnings": warnings})
            return
        _warn(warnings)
        _summary(services, d.id)


def dataset_add(
    dataset: Annotated[str, typer.Argument(help="Dataset id or slug")],
    files: Annotated[list[Path] | None, typer.Argument(help="Audio files or folders (a .lab or .txt beside a clip is its text)")] = None,
    voice: Annotated[str | None, typer.Option("--voice", help="Add a library voice's reference clips")] = None,
    output: Annotated[list[str] | None, typer.Option("--output", help="Add a synthesized output (repeatable)")] = None,
    speaker: Annotated[str, typer.Option("--speaker", help="The clips' speaker")] = "",
    lang: Annotated[str | None, typer.Option("--lang", "-l", help="The clips' language")] = None,
    json_output: Annotated[bool, typer.Option("--json", help="Print JSON")] = False,
) -> None:
    """Add clips: files, a voice's references or synthesized outputs."""
    if not files and not voice and not output:
        raise typer.BadParameter("Give files, --voice or --output")
    warnings: list[str] = []
    with open_services() as services:
        d = services.datasets.get(dataset)
        if files:
            d, warnings = services.datasets.add_files(d.id, [f.expanduser().resolve() for f in files], speaker, lang)
        if voice:
            d = services.datasets.add_voice(d.id, voice)
        if output:
            d = services.datasets.add_outputs(d.id, list(output), speaker)
        if json_output:
            print_json({"dataset": d, "warnings": warnings})
            return
        _warn(warnings)
        _summary(services, d.id)


def dataset_prepare(
    dataset: Annotated[str, typer.Argument(help="Dataset id or slug")],
    slice_: Annotated[bool, typer.Option("--slice/--no-slice", help="Cut clips longer than --max-seconds at their silences")] = True,
    max_seconds: Annotated[float, typer.Option("--max-seconds", help="The longest clip to keep whole")] = 12.0,
    min_seconds: Annotated[float, typer.Option("--min-seconds", help="Shorter pieces are dropped")] = 1.0,
    threshold: Annotated[float, typer.Option("--threshold", help="Silence below this level (dB)")] = -34.0,
    transcribe: Annotated[bool, typer.Option("--transcribe/--no-transcribe", help="Fill missing texts with speech recognition")] = True,
    normalize: Annotated[bool, typer.Option("--normalize/--no-normalize", help="Bring each clip's peak to -1 dBFS")] = True,
    lang: Annotated[str | None, typer.Option("--lang", "-l", help="The language for speech recognition")] = None,
    json_output: Annotated[bool, typer.Option("--json", help="Print JSON")] = False,
) -> None:
    """Slice, transcribe and normalise the clips."""
    from mizuoto.cli.progress import finish, job_progress
    from mizuoto.core.datasets.models import PrepareRequest

    req = PrepareRequest(slice=slice_, max_seconds=max_seconds, min_seconds=min_seconds, threshold_db=threshold, transcribe=transcribe, normalize=normalize, language=lang)
    with open_services() as services:
        with job_progress(services, enabled=not json_output):
            job = services.datasets.prepare(dataset, req, wait=True, origin="cli")
        if json_output:
            print_json(job)
        elif job.state == "completed":
            result = job.result or {}
            console.print(", ".join(f"{k} {v}" for k, v in result.items() if isinstance(v, int)) or "Done", highlight=False)
            _summary(services, dataset)
        finish(job)


def dataset_check(dataset: Annotated[str, typer.Argument(help="Dataset id or slug")], json_output: Annotated[bool, typer.Option("--json", help="Print JSON")] = False) -> None:
    """Report what training would see: durations, empty texts, clipping, rates, speakers."""
    with open_services() as services:
        r = services.datasets.check(dataset)
        if json_output:
            print_json(r)
            return
        console.print(f"{r.usable} of {r.rows} clip(s) usable, {r.seconds / 60:.1f} min", highlight=False)
        if r.histogram:
            peak = max(b.count for b in r.histogram) or 1
            for b in r.histogram:
                console.print(f"{b.seconds:5.0f} s {'█' * max(1, round(24 * b.count / peak)) if b.count else '':24} {b.count}", highlight=False)
        for label, ids in (("too short", r.too_short), ("too long", r.too_long), ("without text", r.empty_text), ("clipping", r.clipping)):
            if ids:
                console.print(f"{len(ids)} {label}: {', '.join(ids[:8])}{' …' if len(ids) > 8 else ''}", highlight=False)
        if r.speakers:
            console.print("Speakers: " + ", ".join(f"{k or '-'} ({v})" for k, v in r.speakers.items()), highlight=False)
        if r.languages:
            console.print("Languages: " + ", ".join(f"{k} ({v})" for k, v in r.languages.items()), highlight=False)
        if r.licence_tags:
            console.print("Licence tags: " + ", ".join(r.licence_tags), highlight=False)
        _warn(r.warnings)


def dataset_rows(
    dataset: Annotated[str, typer.Argument(help="Dataset id or slug")],
    q: Annotated[str | None, typer.Option("--search", "-q", help="Text to look for")] = None,
    offset: Annotated[int, typer.Option("--offset", help="Skip this many rows")] = 0,
    limit: Annotated[int, typer.Option("--limit", help="At most this many rows")] = 50,
    json_output: Annotated[bool, typer.Option("--json", help="Print JSON")] = False,
) -> None:
    """List a dataset's clips and texts."""
    with open_services() as services:
        page = services.datasets.rows(dataset, offset, limit, q)
        if json_output:
            print_json(page)
            return
        print_table(
            f"{len(page.items)} of {page.total}",
            ["Id", "Seconds", "Speaker", "Excluded", "Text"],
            [(r.id, f"{r.duration:.1f}", r.speaker or "-", "yes" if r.excluded else "", short(r.text or "-", 70)) for r in page.items],
        )


def dataset_edit_row(
    dataset: Annotated[str, typer.Argument(help="Dataset id or slug")],
    row: Annotated[str, typer.Argument(help="Row id")],
    text: Annotated[str | None, typer.Option("--text", help="The clip's text")] = None,
    speaker: Annotated[str | None, typer.Option("--speaker", help="The clip's speaker")] = None,
    lang: Annotated[str | None, typer.Option("--lang", "-l", help="The clip's language")] = None,
    exclude: Annotated[bool | None, typer.Option("--exclude/--include", help="Leave the clip out of training, or put it back")] = None,
    remove: Annotated[bool, typer.Option("--remove", help="Delete the clip")] = False,
) -> None:
    """Correct a clip's text or speaker, exclude it or delete it."""
    from mizuoto.core.datasets.models import RowPatch

    with open_services() as services:
        if remove:
            services.datasets.delete_rows(dataset, [row])
            console.print("Deleted")
            return
        r = services.datasets.update_row(dataset, row, RowPatch(text=text, speaker=speaker, language=lang, excluded=exclude))
        console.print(f"{r.id}: {short(r.text, 70)}{' (excluded)' if r.excluded else ''}", highlight=False)


def dataset_delete(dataset: Annotated[str, typer.Argument(help="Dataset id or slug")], yes: Annotated[bool, typer.Option("--yes", "-y", help="Do not ask")] = False) -> None:
    """Move a dataset to the trash."""
    if not yes and not typer.confirm(f"Delete {dataset}?"):
        raise typer.Exit(1)
    with open_services() as services:
        services.datasets.delete(dataset)
        console.print("Deleted")
