"""speak: text to speech in the foreground."""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path
from typing import Annotated, Any

import typer

from mizuoto.cli.output import console, err_console, open_services, print_json

FORMATS = ("wav", "flac", "mp3", "ogg", "m4a")


def parse_pairs(values: list[str] | None, what: str) -> dict[str, Any]:
    """``key=value`` options; values are JSON when they parse (numbers, true/false), else text."""
    out: dict[str, Any] = {}
    for item in values or []:
        key, sep, raw = item.partition("=")
        if not sep or not key.strip():
            raise typer.BadParameter(f"{what} must be key=value, got {item!r}")
        try:
            out[key.strip()] = json.loads(raw)
        except ValueError:
            out[key.strip()] = raw
    return out


def speak(
    text: Annotated[str, typer.Argument(help="The text, or - to read it from stdin")],
    model: Annotated[str | None, typer.Option("--model", "-m", help="Model id, e.g. qwen3tts/1.7b-base (default: tts.default_model)")] = None,
    output: Annotated[Path | None, typer.Option("--output", "-o", help="Also copy the result here; its extension picks the format")] = None,
    voice: Annotated[str | None, typer.Option("--voice", "-v", help="A library voice")] = None,
    ref: Annotated[list[Path] | None, typer.Option("--ref", help="A reference clip to clone (repeatable)")] = None,
    ref_text: Annotated[list[str] | None, typer.Option("--ref-text", help="What is said in each --ref, in order")] = None,
    preset: Annotated[str | None, typer.Option("--preset", help="A preset speaker of the model")] = None,
    describe: Annotated[str | None, typer.Option("--describe", help="Design a voice from this description")] = None,
    attr: Annotated[list[str] | None, typer.Option("--attr", help="A voice attribute key=value (repeatable)")] = None,
    lang: Annotated[str | None, typer.Option("--lang", "-l", help="Language code (zh, en, ja…) or auto")] = None,
    instruct: Annotated[str | None, typer.Option("--instruct", help="How to say it")] = None,
    emotion: Annotated[list[str] | None, typer.Option("--emotion", help="An emotion weight name=value (repeatable)")] = None,
    emotion_audio: Annotated[Path | None, typer.Option("--emotion-audio", help="A clip whose emotion to borrow")] = None,
    emotion_text: Annotated[str | None, typer.Option("--emotion-text", help="The emotion, described")] = None,
    param: Annotated[list[str] | None, typer.Option("--param", "-p", help="A common parameter name=value (repeatable)")] = None,
    option: Annotated[list[str] | None, typer.Option("--option", "-O", help="An engine option name=value (repeatable)")] = None,
    seed: Annotated[int | None, typer.Option("--seed", help="Random seed")] = None,
    speed: Annotated[float | None, typer.Option("--speed", help="1.0 is natural")] = None,
    fmt: Annotated[str | None, typer.Option("--format", "-f", help=f"One of {', '.join(FORMATS)}")] = None,
    sample_rate: Annotated[int | None, typer.Option("--sample-rate", help="Resample the result")] = None,
    batch: Annotated[bool, typer.Option("--batch", help="One output per line of the text")] = False,
    mizuoto_split: Annotated[bool, typer.Option("--split", help="Let Mizuoto split long text, not the engine")] = False,
    lenient: Annotated[bool, typer.Option("--lenient", help="Drop parameters the model does not take, with a warning")] = False,
    use_preset: Annotated[str | None, typer.Option("--use-preset", "-P", help="A parameter preset (mizuoto preset list); --param and --option win over it")] = None,
    stream: Annotated[bool, typer.Option("--stream", help="Write a WAV to stdout as the audio is made (e.g. | ffplay -)")] = False,
    plan_only: Annotated[bool, typer.Option("--plan", help="Show what would happen and stop")] = False,
    json_output: Annotated[bool, typer.Option("--json", help="Print JSON")] = False,
) -> None:
    """Speak text with a model and a voice; the result is kept in the outputs (and copied to -o)."""
    from mizuoto.cli.progress import finish, job_progress
    from mizuoto.core.tts.models import CommonParams, EmotionInput, OutputSpec, ReferenceInput, SpeakRequest, StyleInput, VoiceInput

    if text == "-":
        text = sys.stdin.read()
    if fmt is None and output is not None and output.suffix.lower().lstrip(".") in FORMATS:
        fmt = output.suffix.lower().lstrip(".")
    if fmt is not None and fmt not in FORMATS:
        raise typer.BadParameter(f"--format must be one of {', '.join(FORMATS)}")
    texts = [line.strip() for line in text.splitlines() if line.strip()] if batch else None
    refs = list(ref or [])
    transcripts = list(ref_text or [])
    if len(transcripts) > len(refs):
        raise typer.BadParameter("More --ref-text than --ref")
    voice_spec = None
    if refs or preset or describe or attr:
        voice_spec = VoiceInput(
            references=[ReferenceInput(path=str(p), transcript=transcripts[i] if i < len(transcripts) else None) for i, p in enumerate(refs)],
            preset=preset,
            description=describe,
            attributes={k: str(v) for k, v in parse_pairs(attr, "--attr").items()} or None,
        )
    emotion_input = None
    if emotion or emotion_audio or emotion_text:
        emotion_input = EmotionInput(
            vector={k: float(v) for k, v in parse_pairs(emotion, "--emotion").items()} or None, path=str(emotion_audio) if emotion_audio else None, text=emotion_text
        )
    params = parse_pairs(param, "--param")
    if seed is not None:
        params["seed"] = seed
    if speed is not None:
        params["speed"] = speed
    with open_services() as services:
        model_id = model or services.settings.settings.tts.default_model
        if not model_id:
            raise typer.BadParameter("Name a model with --model (mizuoto model list), or set tts.default_model")
        req = SpeakRequest(
            model=model_id,
            text="" if texts else text,
            texts=texts,
            language=lang,
            voice=voice,
            voice_spec=voice_spec,
            style=StyleInput(instruct=instruct, emotion=emotion_input) if (instruct or emotion_input) else None,
            params=CommonParams.model_validate(params),
            engine_options=parse_pairs(option, "--option"),
            output=OutputSpec(format=fmt, sample_rate=sample_rate),
            segmentation="mizuoto" if mizuoto_split else "engine",
            strict=False if lenient else None,
        )
        if use_preset:
            req = services.presets.apply(use_preset, req)
        if stream and not plan_only:
            _speak_streaming(services, req)
            return
        if plan_only:
            plan = services.tts.plan(req, stream=stream)
            if json_output:
                print_json(plan)
            else:
                console.print_json(plan.model_dump_json())
            return
        with job_progress(services, enabled=not json_output):
            job = services.tts.speak(req, wait=True, origin="cli")
        if job.state == "completed" and job.result:
            outputs = [services.outputs.get(i) for i in job.result.get("output_ids", [])]
            for w in job.result.get("warnings") or []:
                err_console.print(f"[yellow]warning:[/yellow] {w}", highlight=False)
            if output is not None and outputs:
                for n, out in enumerate(outputs):
                    target = output if len(outputs) == 1 else output.with_name(f"{output.stem}-{n + 1}{output.suffix}")
                    target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(out.path, target)
            if json_output:
                print_json({"job": job, "outputs": outputs})
            else:
                for out in outputs:
                    console.print(f"{out.path}  ({out.duration:.2f} s, {out.sample_rate} Hz)", highlight=False)
        elif json_output:
            print_json({"job": job, "outputs": []})
        finish(job)


def _speak_streaming(services: Any, req: Any) -> None:
    """``speak --stream``: the WAV goes to stdout as it is made; everything else to stderr."""
    from mizuoto.cli.progress import finish
    from mizuoto.core.tts.streaming import pcm_sink, wav_header

    if sys.stdout.isatty():
        raise typer.BadParameter("--stream writes audio to stdout: pipe it to a player or a file")
    out = sys.stdout.buffer

    def emit(kind: str, value: Any) -> None:
        if kind == "start":
            out.write(wav_header(value["sample_rate"]))
        elif kind == "audio":
            out.write(value)
        out.flush()

    job = services.tts.speak(req, wait=True, origin="cli", stream=pcm_sink(emit))
    for i in (job.result or {}).get("output_ids", []):
        err_console.print(services.outputs.get(i).path, highlight=False)
    finish(job)
