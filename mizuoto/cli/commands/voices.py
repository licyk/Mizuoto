"""voice list | info | add | design | from-preset | from-output | edit | remove | add-ref | remove-ref |
set-transcript | retranscribe | fit | prepare | export | import; design."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Annotated

import typer

from mizuoto.cli.output import console, err_console, open_services, print_json, print_table, short


def _warn(warnings: list[str]) -> None:
    for w in warnings:
        err_console.print(f"[yellow]warning:[/yellow] {w}", highlight=False)


def voice_list(
    q: Annotated[str | None, typer.Option("--search", "-q", help="Text to look for in names, descriptions and tags")] = None,
    kind: Annotated[str | None, typer.Option("--kind", help="reference, designed, preset or trained")] = None,
    lang: Annotated[str | None, typer.Option("--lang", "-l", help="Only voices in this language")] = None,
    json_output: Annotated[bool, typer.Option("--json", help="Print JSON")] = False,
) -> None:
    """List the voice library."""
    with open_services() as services:
        voices = services.voices.list_voices(q, kind, lang)
        if json_output:
            print_json(voices)
            return
        print_table(None, ["Slug", "Name", "Kind", "Language", "Clips", "Seconds"], [(v.slug, v.name, v.kind, v.language, v.references, f"{v.duration:.1f}") for v in voices])


def voice_info(voice: Annotated[str, typer.Argument(help="Voice id or slug")], json_output: Annotated[bool, typer.Option("--json", help="Print JSON")] = False) -> None:
    """Show a voice and its reference clips."""
    with open_services() as services:
        v = services.voices.get(voice)
        if json_output:
            print_json(v)
            return
        console.print(f"[bold]{v.name}[/bold] ({v.slug}, {v.id}) · {v.kind} · {v.language or 'any language'}", highlight=False)
        if v.description:
            console.print(v.description, highlight=False)
        if v.preset:
            console.print(f"Preset speaker {v.preset.speaker} of {v.preset.model}", highlight=False)
        if v.design:
            console.print(f"Designed with {v.design.model}: {v.design.description or v.design.attributes}", highlight=False)
        if v.consent:
            console.print(f"Consent: {v.consent.source}" + (f" ({v.consent.note})" if v.consent.note else ""), highlight=False)
        if v.references:
            print_table(
                "References",
                ["Clip", "Seconds", "Rate", "Primary", "Transcript"],
                [(r.sha256[:12], f"{r.duration:.1f}", r.sample_rate, "yes" if r.primary else "", short(r.transcript or "-", 70)) for r in v.references],
            )


def voice_add(
    name: Annotated[str, typer.Argument(help="The voice's name")],
    ref: Annotated[list[Path], typer.Option("--ref", help="A reference clip (repeatable)")],
    text: Annotated[list[str] | None, typer.Option("--text", help="What is said in each --ref, in order")] = None,
    lang: Annotated[str | None, typer.Option("--lang", "-l", help="The voice's language")] = None,
    consent: Annotated[str | None, typer.Option("--consent", help="Whose voice it is and how you may use it")] = None,
    description: Annotated[str, typer.Option("--description", help="A note about the voice")] = "",
    tag: Annotated[list[str] | None, typer.Option("--tag", help="A tag (repeatable)")] = None,
    no_transcribe: Annotated[bool, typer.Option("--no-transcribe", help="Do not fill missing transcripts with speech recognition")] = False,
    json_output: Annotated[bool, typer.Option("--json", help="Print JSON")] = False,
) -> None:
    """Add a voice from reference clips; missing transcripts come from speech recognition."""
    from mizuoto.cli.progress import job_progress

    with open_services() as services:
        with job_progress(services, enabled=not json_output):
            voice, warnings = services.voices.create_from_audio(name, ref, list(text or []), lang, consent, description, tag, transcribe=not no_transcribe)
        if json_output:
            print_json(voice)
            return
        _warn(warnings)
        console.print(f"Added {voice.name} ({voice.slug}) with {len(voice.references)} clip(s)", highlight=False)
        for r in voice.references:
            console.print(f"  {r.sha256[:12]}  {r.duration:.1f} s  {r.transcript or '(no transcript)'}", highlight=False)


def voice_design(
    name: Annotated[str, typer.Argument(help="The voice's name")],
    model: Annotated[str, typer.Option("--model", "-m", help="A voice design model")],
    description: Annotated[str | None, typer.Option("--description", "-d", help="The voice, described")] = None,
    attr: Annotated[list[str] | None, typer.Option("--attr", help="A voice attribute key=value (repeatable)")] = None,
    sample_text: Annotated[str, typer.Option("--sample-text", help="What the voice says in its reference clip")] = "Hello! This is how I sound when I read a few sentences aloud.",
    lang: Annotated[str | None, typer.Option("--lang", "-l", help="Language of the sample text")] = None,
    seed: Annotated[int | None, typer.Option("--seed", help="Random seed")] = None,
    json_output: Annotated[bool, typer.Option("--json", help="Print JSON")] = False,
) -> None:
    """Design a voice from a description; its sample clip lets every cloning model use it."""
    from mizuoto.cli.commands.speak import parse_pairs
    from mizuoto.cli.progress import job_progress

    with open_services() as services:
        with job_progress(services, enabled=not json_output):
            voice, output_id = services.voices.create_from_design(name, model, sample_text, description, parse_pairs(attr, "--attr") or None, lang, seed)
        if json_output:
            print_json(voice)
            return
        ref = voice.references[0]
        console.print(f"Designed {voice.name} ({voice.slug}): {ref.duration:.1f} s sample, output {output_id}", highlight=False)
        console.print(str(services.voices.folder(voice) / ref.file), highlight=False, soft_wrap=True)


def voice_from_preset(
    model: Annotated[str, typer.Argument(help="Model id")],
    speaker: Annotated[str, typer.Argument(help="Preset speaker")],
    name: Annotated[str | None, typer.Option("--name", help="The voice's name")] = None,
    json_output: Annotated[bool, typer.Option("--json", help="Print JSON")] = False,
) -> None:
    """Add a model's preset speaker to the library."""
    with open_services() as services:
        voice = services.voices.create_preset(model, speaker, name)
        if json_output:
            print_json(voice)
            return
        console.print(f"Added {voice.name} ({voice.slug})", highlight=False)


def voice_from_output(
    output_id: Annotated[str, typer.Argument(help="Output id")],
    name: Annotated[str, typer.Argument(help="The voice's name")],
    consent: Annotated[str | None, typer.Option("--consent", help="Whose voice it is and how you may use it")] = None,
    json_output: Annotated[bool, typer.Option("--json", help="Print JSON")] = False,
) -> None:
    """Keep a synthesized clip as a voice (a designed voice when it came from a design)."""
    with open_services() as services:
        voice = services.voices.create_from_output(output_id, name, consent)
        if json_output:
            print_json(voice)
            return
        console.print(f"Added {voice.name} ({voice.slug}, {voice.kind})", highlight=False)


def voice_edit(
    voice: Annotated[str, typer.Argument(help="Voice id or slug")],
    name: Annotated[str | None, typer.Option("--name", help="New name")] = None,
    lang: Annotated[str | None, typer.Option("--lang", "-l", help="Language")] = None,
    description: Annotated[str | None, typer.Option("--description", help="Description")] = None,
    tag: Annotated[list[str] | None, typer.Option("--tag", help="Replace the tags (repeatable)")] = None,
    consent: Annotated[str | None, typer.Option("--consent", help="Consent statement")] = None,
) -> None:
    """Change a voice's name, language, description, tags or consent."""
    with open_services() as services:
        v = services.voices.update(voice, name=name, language=lang, description=description, tags=tag, consent=consent)
        console.print(f"Updated {v.name} ({v.slug})", highlight=False)


def voice_remove(voices: Annotated[list[str], typer.Argument(help="Voice ids or slugs")]) -> None:
    """Move voices to the trash."""
    with open_services() as services:
        for key in voices:
            services.voices.delete(key)
            console.print(f"Removed {key}", highlight=False)


def voice_add_ref(
    voice: Annotated[str, typer.Argument(help="Voice id or slug")],
    clip: Annotated[Path, typer.Argument(help="A reference clip")],
    text: Annotated[str | None, typer.Option("--text", help="What is said in it")] = None,
    lang: Annotated[str | None, typer.Option("--lang", "-l", help="Its language")] = None,
    primary: Annotated[bool, typer.Option("--primary", help="Make it the voice's main clip")] = False,
) -> None:
    """Add a reference clip to a voice."""
    with open_services() as services:
        v, warnings = services.voices.add_reference(voice, clip, text, lang, primary)
        _warn(warnings)
        console.print(f"{v.name} has {len(v.references)} clip(s)", highlight=False)


def voice_remove_ref(voice: Annotated[str, typer.Argument(help="Voice id or slug")], sha: Annotated[str, typer.Argument(help="The clip's SHA-256 (a prefix will do)")]) -> None:
    """Remove a reference clip from a voice."""
    with open_services() as services:
        v = services.voices.remove_reference(voice, sha)
        console.print(f"{v.name} has {len(v.references)} clip(s)", highlight=False)


def voice_set_transcript(
    voice: Annotated[str, typer.Argument(help="Voice id or slug")],
    sha: Annotated[str, typer.Argument(help="The clip's SHA-256 (a prefix will do)")],
    text: Annotated[str, typer.Argument(help="What is said in the clip")],
) -> None:
    """Correct a reference clip's transcript."""
    with open_services() as services:
        services.voices.set_transcript(voice, sha, text)
        console.print("Saved", highlight=False)


def voice_retranscribe(voice: Annotated[str, typer.Argument(help="Voice id or slug")], sha: Annotated[str | None, typer.Argument(help="Only this clip")] = None) -> None:
    """Transcribe a voice's clips again with speech recognition."""
    with open_services() as services:
        v = services.voices.retranscribe(voice, sha)
        for r in v.references:
            console.print(f"{r.sha256[:12]}  {r.transcript}", highlight=False)


def voice_fit(
    voice: Annotated[str, typer.Argument(help="Voice id or slug")],
    model: Annotated[str, typer.Option("--model", "-m", help="Model id")],
    json_output: Annotated[bool, typer.Option("--json", help="Print JSON")] = False,
) -> None:
    """Say whether and how a model can use a voice."""
    with open_services() as services:
        fit = services.voices.fit(voice, model)
        if json_output:
            print_json(fit)
            return
        if fit.usable:
            console.print(f"usable as {fit.mode}" + (f"; first: {', '.join(fit.needs)}" if fit.needs else ""), highlight=False)
        else:
            console.print(f"not usable: {fit.reason}", highlight=False)
        _warn(fit.warnings)


def voice_prepare(
    voice: Annotated[str, typer.Argument(help="Voice id or slug")],
    model: Annotated[str, typer.Option("--model", "-m", help="Model id")],
) -> None:
    """Compute a voice's derived state for a model ahead of time (reference prompt, embedding)."""
    from mizuoto.cli.progress import finish, job_progress

    with open_services() as services:
        with job_progress(services):
            job = services.tts.prepare(voice, model, wait=True, origin="cli")
        if job.state == "completed" and job.result:
            console.print(f"Prepared {voice} for {model}: {', '.join(job.result.get('files') or []) or 'nothing to cache'}", highlight=False)
        finish(job)


def voice_export(voice: Annotated[str, typer.Argument(help="Voice id or slug")], output: Annotated[Path, typer.Option("--output", "-o", help="The zip to write")]) -> None:
    """Write a voice to a zip (its clips and settings)."""
    with open_services() as services:
        path = services.voices.export(voice, output)
        console.print(str(path), highlight=False, soft_wrap=True)


def voice_import(
    archive: Annotated[Path, typer.Argument(help="A zip written by voice export")],
    name: Annotated[str | None, typer.Option("--name", help="Another name for it")] = None,
) -> None:
    """Add a voice from a zip written by voice export."""
    with open_services() as services:
        v = services.voices.import_archive(archive, name)
        console.print(f"Imported {v.name} ({v.slug})", highlight=False)


def design(
    text: Annotated[str, typer.Argument(help="What the designed voice says")],
    model: Annotated[str, typer.Option("--model", "-m", help="A voice design model")],
    description: Annotated[str | None, typer.Option("--description", "-d", help="The voice, described")] = None,
    attr: Annotated[list[str] | None, typer.Option("--attr", help="A voice attribute key=value (repeatable)")] = None,
    lang: Annotated[str | None, typer.Option("--lang", "-l", help="Language")] = None,
    seed: Annotated[int | None, typer.Option("--seed", help="Random seed")] = None,
    output: Annotated[Path | None, typer.Option("--output", "-o", help="Also copy the result here")] = None,
    save_as: Annotated[str | None, typer.Option("--save-as", help="Keep the voice in the library under this name")] = None,
    json_output: Annotated[bool, typer.Option("--json", help="Print JSON")] = False,
) -> None:
    """Speak with a voice made from a description (and keep it with --save-as)."""
    from mizuoto.cli.commands.speak import parse_pairs
    from mizuoto.cli.progress import finish, job_progress
    from mizuoto.core.tts.models import CommonParams, SpeakRequest, VoiceInput

    with open_services() as services:
        attributes = parse_pairs(attr, "--attr") or None
        req = SpeakRequest(model=model, text=text, language=lang, voice_spec=VoiceInput(description=description, attributes=attributes), params=CommonParams(seed=seed))
        with job_progress(services, enabled=not json_output):
            job = services.tts.speak(req, wait=True, origin="cli")
        voice = None
        if job.state == "completed" and job.result:
            out = services.outputs.get(job.result["output_id"])
            if output is not None:
                output.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(out.path, output)
            if save_as:
                voice = services.voices.create_from_output(out.id, save_as, description=description or "")
            if json_output:
                print_json({"job": job, "output": out, "voice": voice})
            else:
                console.print(f"{out.path}  ({out.duration:.2f} s)", highlight=False)
                if voice is not None:
                    console.print(f"Saved as {voice.name} ({voice.slug})", highlight=False)
        finish(job)


def voice_add_weights(
    voice: Annotated[str, typer.Argument(help="Voice id or slug")],
    engine: Annotated[str, typer.Option("--engine", "-e", help="The engine the weights are for, e.g. gptsovits")],
    file: Annotated[list[str], typer.Option("--file", "-f", help="role=path, once per file (GPT-SoVITS: gpt=x.ckpt sovits=y.pth)")],
    move: Annotated[bool, typer.Option("--move", help="Move the files instead of copying them")] = False,
    json_output: Annotated[bool, typer.Option("--json", help="Print JSON")] = False,
) -> None:
    """Give a voice its own fine-tuned weights for an engine (used with that engine's models of the same version)."""
    files: dict[str, Path] = {}
    for item in file:
        role, sep, path = item.partition("=")
        if not sep:
            raise typer.BadParameter(f"--file must be role=path, got {item!r}")
        files[role.strip()] = Path(path).expanduser().resolve()
    with open_services() as services:
        v = services.voices.add_weights(voice, engine, files, move=move)
        if json_output:
            print_json(v)
            return
        w = v.weights[engine]
        console.print(f"{v.name}: {engine} weights ({w.version or 'version unknown'})", highlight=False)


def voice_remove_weights(
    voice: Annotated[str, typer.Argument(help="Voice id or slug")],
    engine: Annotated[str, typer.Option("--engine", "-e", help="The engine whose weights to remove")],
) -> None:
    """Remove a voice's own fine-tuned weights for an engine."""
    with open_services() as services:
        v = services.voices.remove_weights(voice, engine)
        console.print(f"{v.name} has weights for: {', '.join(v.weights) or 'no engine'}", highlight=False)
