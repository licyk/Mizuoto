"""project list | new | show | import | script | cast | set | render | export | delete."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Annotated, Any

import typer

from mizuoto.cli.commands.speak import FORMATS, parse_pairs
from mizuoto.cli.output import console, err_console, open_services, print_json, print_table, short

STATE_STYLE = {"rendered": "green", "stale": "yellow", "missing": "dim", "failed": "red", "rendering": "cyan", "empty": "dim"}


def _summary(project: Any) -> str:
    s = project.stats
    return f"{project.name} ({project.slug}): {s.chapters} chapter(s), {s.lines} line(s), {s.rendered} rendered, {s.stale} stale, {s.missing} missing, {s.failed} failed, {s.seconds / 60:.1f} min"


def project_list(json_output: Annotated[bool, typer.Option("--json", help="Print JSON")] = False) -> None:
    """List the projects."""
    with open_services() as services:
        projects = services.projects.list_projects()
        if json_output:
            print_json(projects)
            return
        rows = [(p.slug, p.name, p.model or "-", p.speakers, p.stats.lines, f"{p.stats.rendered}/{p.stats.lines}", f"{p.stats.seconds / 60:.1f}") for p in projects]
        print_table(None, ["Slug", "Name", "Model", "Speakers", "Lines", "Rendered", "Minutes"], rows)


def project_new(
    name: Annotated[str, typer.Argument(help="The project's name")],
    model: Annotated[str | None, typer.Option("--model", "-m", help="The model of speakers that name none")] = None,
    voice: Annotated[str | None, typer.Option("--voice", "-v", help="The narrator's library voice")] = None,
    preset: Annotated[str | None, typer.Option("--preset", help="The narrator's preset speaker")] = None,
    lang: Annotated[str | None, typer.Option("--lang", "-l", help="Its language")] = None,
    script: Annotated[Path | None, typer.Option("--script", "-s", help="A plain-text script to start from (# chapters, Name: lines)")] = None,
    split: Annotated[str, typer.Option("--split", help="line: a line of text is a line; sentence: a line per sentence")] = "line",
    json_output: Annotated[bool, typer.Option("--json", help="Print JSON")] = False,
) -> None:
    """Create a project, with a narrator and optionally a script."""
    from mizuoto.core.projects.models import ProjectInput, ScriptImport, Speaker

    with open_services() as services:
        narrator = Speaker(id="narrator", name="Narrator", voice=voice, preset=preset)
        project = services.projects.create(ProjectInput(name=name, model=model, language=lang, speakers=[narrator]))
        if script is not None:
            project = services.projects.import_script(project.id, ScriptImport(text=script.read_text(encoding="utf-8"), split=split))  # ty: ignore[invalid-argument-type]
        if json_output:
            print_json(project)
            return
        console.print(f"Created {_summary(project)}", highlight=False)


def project_show(
    project: Annotated[str, typer.Argument(help="Project id or slug")],
    json_output: Annotated[bool, typer.Option("--json", help="Print JSON")] = False,
) -> None:
    """Show a project's cast and its lines with their state."""
    with open_services() as services:
        p = services.projects.get(project)
        if json_output:
            print_json(p)
            return
        console.print(f"[bold]{_summary(p)}[/bold]", highlight=False)
        for s in p.speakers:
            who = s.voice and f"voice {s.voice}" or s.preset and f"preset {s.preset}" or "its own voice"
            console.print(f"  [cyan]{s.id}[/cyan] {s.name}: {s.model or p.model or '(no model)'}, {who}{f', {s.instruct!r}' if s.instruct else ''}", highlight=False)
        names = {s.id: s.name for s in p.speakers}
        for i, chapter in enumerate(p.chapters, 1):
            console.print(f"\n[bold]{i}. {chapter.title or '(untitled)'}[/bold] [dim]{chapter.id}[/dim]", highlight=False)
            for line in chapter.lines:
                speaker = names.get(line.speaker or "", "") if line.speaker else ""
                state = f"[{STATE_STYLE[line.state]}]{line.state:9}[/{STATE_STYLE[line.state]}]"
                console.print(f"  {state} [dim]{line.id}[/dim] {speaker + ': ' if speaker else ''}{short(line.text, 90)}", highlight=False)
                if line.error:
                    console.print(f"            [red]{line.error}[/red]", highlight=False)


def project_import(
    project: Annotated[str, typer.Argument(help="Project id or slug")],
    script: Annotated[Path, typer.Argument(help="A plain-text script (# chapters, Name: lines, blank lines between paragraphs)")],
    split: Annotated[str, typer.Option("--split", help="line or sentence")] = "line",
    replace: Annotated[bool, typer.Option("--replace", help="Replace the chapters instead of appending")] = False,
    chapter: Annotated[str | None, typer.Option("--chapter", help="Append the lines to this chapter")] = None,
    json_output: Annotated[bool, typer.Option("--json", help="Print JSON")] = False,
) -> None:
    """Add chapters and lines from a script; new speaker names join the cast."""
    from mizuoto.core.projects.models import ScriptImport

    with open_services() as services:
        p = services.projects.import_script(project, ScriptImport(text=script.read_text(encoding="utf-8"), split=split, replace=replace, chapter=chapter))  # ty: ignore[invalid-argument-type]
        if json_output:
            print_json(p)
            return
        console.print(_summary(p), highlight=False)


def project_script(project: Annotated[str, typer.Argument(help="Project id or slug")]) -> None:
    """Print the project as a plain-text script."""
    with open_services() as services:
        typer.echo(services.projects.script(project), nl=False)


def project_cast(
    project: Annotated[str, typer.Argument(help="Project id or slug")],
    speaker: Annotated[str, typer.Argument(help="A speaker's id (a new id adds one)")],
    name: Annotated[str | None, typer.Option("--name", help="Its display name")] = None,
    model: Annotated[str | None, typer.Option("--model", "-m", help="Its model ('' for the project's)")] = None,
    voice: Annotated[str | None, typer.Option("--voice", "-v", help="A library voice ('' for none)")] = None,
    preset: Annotated[str | None, typer.Option("--preset", help="A preset speaker ('' for none)")] = None,
    lang: Annotated[str | None, typer.Option("--lang", "-l", help="Its language ('' for the project's)")] = None,
    instruct: Annotated[str | None, typer.Option("--instruct", help="How it speaks ('' for nothing)")] = None,
    param: Annotated[list[str] | None, typer.Option("--param", "-p", help="A common parameter name=value (repeatable; replaces them)")] = None,
    option: Annotated[list[str] | None, typer.Option("--option", "-O", help="An engine option name=value (repeatable; replaces them)")] = None,
    remove: Annotated[bool, typer.Option("--remove", help="Remove the speaker (its lines go to the narrator)")] = False,
    json_output: Annotated[bool, typer.Option("--json", help="Print JSON")] = False,
) -> None:
    """Add, change or remove a speaker of the cast."""
    from mizuoto.core.projects.models import Speaker
    from mizuoto.core.tts.models import CommonParams

    with open_services() as services:
        data = services.projects.editable(services.projects.get(project))
        found = next((s for s in data.speakers if s.id == speaker), None)
        if remove:
            if found is None:
                raise typer.BadParameter(f"No speaker {speaker}")
            data.speakers.remove(found)
            for chapter in data.chapters:
                for line in chapter.lines:
                    if line.speaker == speaker:
                        line.speaker = None
        else:
            if found is None:
                found = Speaker(id=speaker, name=name or speaker)
                data.speakers.append(found)
            if name is not None:
                found.name = name
            if model is not None:
                found.model = model or None
            if voice is not None:
                found.voice, found.preset = voice or None, None if voice else found.preset
            if preset is not None:
                found.preset, found.voice = preset or None, None if preset else found.voice
            if lang is not None:
                found.language = lang or None
            if instruct is not None:
                found.instruct = instruct or None
            if param is not None:
                found.params = CommonParams.model_validate(parse_pairs(param, "--param"))
            if option is not None:
                found.engine_options = parse_pairs(option, "--option")
        p = services.projects.update(project, data)
        if json_output:
            print_json(p)
            return
        console.print(_summary(p), highlight=False)


def project_set(
    project: Annotated[str, typer.Argument(help="Project id or slug")],
    name: Annotated[str | None, typer.Option("--name", help="Rename it")] = None,
    model: Annotated[str | None, typer.Option("--model", "-m", help="The default model")] = None,
    lang: Annotated[str | None, typer.Option("--lang", "-l", help="The default language")] = None,
    line_pause: Annotated[int | None, typer.Option("--line-pause", help="Milliseconds after a line")] = None,
    paragraph_pause: Annotated[int | None, typer.Option("--paragraph-pause", help="Milliseconds after a paragraph")] = None,
    chapter_pause: Annotated[int | None, typer.Option("--chapter-pause", help="Milliseconds between chapters")] = None,
    fmt: Annotated[str | None, typer.Option("--format", "-f", help=f"The export format: {', '.join(FORMATS)}")] = None,
    sample_rate: Annotated[int | None, typer.Option("--sample-rate", help="The export rate")] = None,
    normalize: Annotated[str | None, typer.Option("--normalize", help="none, peak or loudness (each take)")] = None,
    json_output: Annotated[bool, typer.Option("--json", help="Print JSON")] = False,
) -> None:
    """Change a project's settings."""
    with open_services() as services:
        data = services.projects.editable(services.projects.get(project))
        changes = {
            "name": name,
            "model": model,
            "language": lang,
            "line_pause_ms": line_pause,
            "paragraph_pause_ms": paragraph_pause,
            "chapter_pause_ms": chapter_pause,
            "format": fmt,
            "sample_rate": sample_rate,
            "normalize": normalize,
        }
        merged = data.model_dump() | {k: v for k, v in changes.items() if v is not None}
        p = services.projects.update(project, type(data).model_validate(merged))
        if json_output:
            print_json(p)
            return
        console.print(_summary(p), highlight=False)


def project_render(
    project: Annotated[str, typer.Argument(help="Project id or slug")],
    chapter: Annotated[list[str] | None, typer.Option("--chapter", "-c", help="Only this chapter (id; repeatable)")] = None,
    line: Annotated[list[str] | None, typer.Option("--line", help="Only this line (id; repeatable)")] = None,
    force: Annotated[bool, typer.Option("--force", help="Render lines that are current too")] = False,
    json_output: Annotated[bool, typer.Option("--json", help="Print JSON")] = False,
) -> None:
    """Render the lines without a current take."""
    from mizuoto.cli.progress import finish, job_progress
    from mizuoto.core.projects.models import RenderRequest

    with open_services() as services:
        with job_progress(services, enabled=not json_output):
            job = services.projects.render(project, RenderRequest(chapters=chapter or None, lines=line or None, force=force), wait=True, origin="cli")
        if json_output:
            print_json(job)
        elif job.state == "completed":
            p = services.projects.get(project)
            console.print(_summary(p), highlight=False)
            for c in p.chapters:
                for ln in c.lines:
                    if ln.state == "failed":
                        err_console.print(f"[red]{ln.id}[/red] {short(ln.text, 50)}: {ln.error}", highlight=False)
        finish(job)


def project_export(
    project: Annotated[str, typer.Argument(help="Project id or slug")],
    output: Annotated[Path | None, typer.Option("--output", "-o", help="Copy the result here; its extension picks the format")] = None,
    chapter: Annotated[list[str] | None, typer.Option("--chapter", "-c", help="Only this chapter (id; repeatable)")] = None,
    fmt: Annotated[str | None, typer.Option("--format", "-f", help=f"One of {', '.join(FORMATS)}")] = None,
    srt: Annotated[Path | None, typer.Option("--srt", help="Write the lines as subtitles here (.srt or .vtt)")] = None,
    allow_missing: Annotated[bool, typer.Option("--allow-missing", help="Export what is rendered: changed lines keep their old take, unrendered ones are left out")] = False,
    json_output: Annotated[bool, typer.Option("--json", help="Print JSON")] = False,
) -> None:
    """Join the takes into one file (the whole project, or chapters)."""
    from mizuoto.cli.progress import finish, job_progress
    from mizuoto.core.projects.models import ExportRequest

    if fmt is None and output is not None and output.suffix.lower().lstrip(".") in FORMATS:
        fmt = output.suffix.lower().lstrip(".")
    if fmt is not None and fmt not in FORMATS:
        raise typer.BadParameter(f"--format must be one of {', '.join(FORMATS)}")
    with open_services() as services:
        req = ExportRequest(chapters=chapter or None, format=fmt, allow_missing=allow_missing)
        with job_progress(services, enabled=not json_output):
            job = services.projects.export(project, req, wait=True, origin="cli")
        if job.state == "completed" and job.result:
            out = services.outputs.get(job.result["output_id"])
            if output is not None:
                output.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(out.path, output)
            if srt is not None:
                srt.parent.mkdir(parents=True, exist_ok=True)
                srt.write_text(services.projects.subtitles(out.id, "vtt" if srt.suffix.lower() == ".vtt" else "srt"), encoding="utf-8")
            if json_output:
                print_json({"job": job, "output": out})
            else:
                console.print(f"{out.name}: {out.duration / 60:.1f} min, {job.result['lines']} line(s) → {output or out.path}", highlight=False)
        elif json_output:
            print_json({"job": job})
        finish(job)


def project_delete(
    project: Annotated[str, typer.Argument(help="Project id or slug")],
    yes: Annotated[bool, typer.Option("--yes", "-y", help="Do not ask")] = False,
) -> None:
    """Move a project to the trash, with its takes."""
    with open_services() as services:
        p = services.projects.get(project)
        if not yes:
            typer.confirm(f"Delete {p.name} and its {sum(len(line.takes) for c in p.chapters for line in c.lines)} take(s)?", abort=True)
        services.projects.delete(p.id)
        console.print(f"Deleted {p.name}", highlight=False)
