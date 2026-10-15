"""Run parity cases through a whole Mizuoto stack and save what comes out.

    <python> scripts/parity/stack.py run <cases.jsonl>... --impl reference|native|upstream [--device cpu] [--only ID]
        [--setting compute.compile=true] [--repeat 2]

``<python>`` decides which Mizuoto runs. The reference host (``<workspace>/reference-host/bin/python``,
the worktree at ``v1-engine-workers`` with its engine workers) records the original behaviour. This checkout's
interpreter (``.venv`` or ``<workspace>/native-env``) runs the native engines. This script imports only ``mizuoto``'s
public core API, which both share; ``--impl`` says which one runs (the reference also gets the real engine environments).
``--impl upstream`` is for engines Mizuoto 0.1 never had (VoxCPM): this checkout, with the engine's implementation swapped
for an adapter over the upstream package (``UPSTREAM``), run in an environment where that package is installed.

Each case becomes one ``SpeakRequest`` (or ``ConvertRequest`` / ``EditRequest``) through
``services.tts``, in a temporary data folder that reads the real models (and, for the reference, the
real engine environments) from ``MIZUOTO_PARITY_DATA`` (default: Mizuoto's data folder). It writes
``<workspace>/fixtures/<engine>/<case>/<name>.wav`` and ``<name>.json`` (``<name>`` is ``--name``, default the impl): the output record (applied
parameters, timings, warnings, voice mode), the plan and the case.

``--setting key=value`` (repeatable; the value read as JSON, else as a string) overrides a setting for the whole
run, e.g. ``compute.compile=true``. ``--repeat N`` runs each case N times in the same stack: the first run's audio
is kept, the record lists every run's wall time (``wall_seconds_runs``) and whether the runs' audio was identical.

The workspace is ``MIZUOTO_PARITY_DIR`` (default ``~/code_workspace/experiment/kodama-refactor``).
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
import time
import traceback
from pathlib import Path
from typing import Any


def package(module: str) -> Any:
    """``mizuoto.<module>``, or ``kodama.<module>`` under the reference host: the 0.1 code (tag ``v1-engine-workers``) kept
    the product's old name."""
    import importlib
    import importlib.util

    return importlib.import_module(f"{'mizuoto' if importlib.util.find_spec('mizuoto') else 'kodama'}.{module}")


WORKSPACE = Path(os.environ.get("MIZUOTO_PARITY_DIR", "~/code_workspace/experiment/kodama-refactor")).expanduser()
ENGINES = ("qwen3tts", "asr", "omnivoice", "cosyvoice", "f5tts", "gptsovits", "fishspeech", "indextts", "voxcpm")
UPSTREAM = {"voxcpm": "voxcpm_upstream:UpstreamVoxCpmEngine"}
"""Engines whose reference is their upstream package (no 0.1 implementation): id → the adapter that calls it."""


def load_cases(paths: list[str], only: set[str]) -> list[dict[str, Any]]:
    cases = []
    for p in paths:
        for line in Path(p).read_text(encoding="utf-8").splitlines():
            if line.strip() and not line.lstrip().startswith("//"):
                case = json.loads(line)
                if not only or case["id"] in only:
                    cases.append(case)
    return cases


def refs() -> dict[str, Any]:
    path = WORKSPACE / "corpus" / "refs.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def resolve(obj: Any, table: dict[str, Any]) -> Any:
    """Replace ``{"ref": "<id>"}`` with the clip's path, and its transcript unless the case sets one."""
    if isinstance(obj, dict):
        if "ref" in obj and set(obj) <= {"ref", "transcript", "language"}:
            entry = table[obj["ref"]]
            out = {"path": str(WORKSPACE / "corpus" / "refs" / entry["file"])}
            out["transcript"] = obj["transcript"] if "transcript" in obj else entry.get("transcript")
            if obj.get("language") or entry.get("language"):
                out["language"] = obj.get("language") or entry.get("language")
            return out
        if "source_ref" in obj:
            entry = table[obj.pop("source_ref")]
            obj["path"] = str(WORKSPACE / "corpus" / "refs" / entry["file"])
        return {k: resolve(v, table) for k, v in obj.items()}
    if isinstance(obj, list):
        return [resolve(v, table) for v in obj]
    return obj


def real_data_dir() -> Path:
    """The data folder whose models and environments the runs use (``MIZUOTO_PARITY_DATA``, else the default one)."""
    if os.environ.get("MIZUOTO_PARITY_DATA"):
        return Path(os.environ["MIZUOTO_PARITY_DATA"]).expanduser()
    default_data_dir = package("core.paths").default_data_dir

    return default_data_dir(os.environ)


def copy_voices(s, names: list[str]) -> None:
    """Library voices a case names (``"voices": [...]``: e.g. one with fine-tuned weights), copied without their caches from
    the workspace's ``voices/`` or else the real data folder's."""
    import shutil

    for name in names:
        target = s.settings.voices_dir / name
        if not target.exists():
            source = WORKSPACE / "voices" / name
            source = source if source.is_dir() else real_data_dir() / "voices" / name
            shutil.copytree(source, target, ignore=shutil.ignore_patterns("cache"))


def parse_settings(pairs: list[str]) -> dict[str, Any]:
    """``key=value`` pairs as nested overrides; the value is JSON when it parses (``true``, ``8``), else a string."""
    flat: dict[str, Any] = {}
    for pair in pairs:
        key, sep, raw = pair.partition("=")
        if not sep or not key:
            raise SystemExit(f"--setting needs key=value, not {pair!r}")
        try:
            flat[key.strip()] = json.loads(raw)
        except json.JSONDecodeError:
            flat[key.strip()] = raw
    return _nest(flat)


def _merge(base: dict[str, Any], extra: dict[str, Any]) -> dict[str, Any]:
    for key, value in extra.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _merge(base[key], value)
        else:
            base[key] = value
    return base


def build(device: str, data_dir: Path, impl: str = "native", settings: dict[str, Any] | None = None):
    """Services in ``data_dir`` over the real models; ``impl="reference"`` (the v1 host) also over the real engine environments."""
    build_services = package("core.context").build_services

    if impl == "upstream":
        engines = package("runtime.engine")

        engines.BUILTIN.update(UPSTREAM)
    real = real_data_dir()
    overrides: dict[str, Any] = {
        "paths": {"models_dir": str(real / "models"), **({"envs_dir": str(real / "envs")} if impl == "reference" else {})},
        "engines": {e: {"device": device} for e in ENGINES},
        "tts": {"normalize": "none"},
    }
    _merge(overrides, settings or {})
    s = build_services(data_dir=data_dir, environ={}, settings_overrides=overrides, start_background=True)
    # Licence-gated engines are accepted in this temporary folder only, for the comparison runs.
    for engine in s.engines.ids():
        if s.engines.manifest(engine).licence.gated:
            s.engines.accept_licence(engine)
    return s


def run_case(s, case: dict[str, Any], table: dict[str, Any]) -> tuple[Any, dict[str, Any]]:
    models = package("core.tts.models")
    ConvertRequest, EditRequest, SpeakRequest = models.ConvertRequest, models.EditRequest, models.SpeakRequest
    StreamSink = package("core.tts.service").StreamSink

    kind = case.get("kind", "speak")
    copy_voices(s, case.get("voices", []))
    body = resolve(json.loads(json.dumps(case["request"])), table)
    body.setdefault("output", {}).update({"format": "wav", "normalize": "none"})
    settings = case.get("settings")
    before = _nest({k: s.settings.get_value(k) for k in _keys(settings)}) if settings else None
    if settings:
        s.settings.update(settings)
    started = time.perf_counter()
    chunks: list[int] = []
    try:
        if kind == "speak":
            req = SpeakRequest.model_validate(body)
            plan = s.tts.plan(req)
            sink = None
            if case.get("stream"):  # the audio as it is made; the stored output is still the whole
                sink = StreamSink()
                sink.chunk = lambda samples, rate, segment: chunks.append(len(samples))
            job = s.tts.speak(req, wait=True, stream=sink)
        elif kind == "convert":
            req = ConvertRequest.model_validate(body)
            plan = None
            job = s.tts.convert(req, wait=True)
        else:
            req = EditRequest.model_validate(body)
            plan = None
            job = s.tts.edit(req, wait=True)
    finally:
        if before is not None:
            s.settings.update(before)
    wall = time.perf_counter() - started
    record: dict[str, Any] = {"case": case, "wall_seconds": wall, "job_state": job.state, "error": job.error.model_dump() if job.error else None}
    if plan is not None:
        record["plan"] = plan.model_dump(mode="json")
    if case.get("stream"):
        record["stream_chunks"] = chunks
    if job.state != "completed":
        return None, record
    out = s.outputs.get(job.result["output_id"])
    record["output"] = out.model_dump(mode="json")
    return Path(out.path), record


def _keys(tree: dict[str, Any], prefix: str = ""):
    for k, v in tree.items():
        if isinstance(v, dict):
            yield from _keys(v, f"{prefix}{k}.")
        else:
            yield f"{prefix}{k}"


def _exception_record(case: dict[str, Any], exc: Exception) -> dict[str, Any]:
    return {"case": case, "job_state": "exception", "error": {"message": str(exc), "traceback": traceback.format_exc()}}


def _nest(flat: dict[str, Any]) -> dict[str, Any]:
    tree: dict[str, Any] = {}
    for dotted, value in flat.items():
        node = tree
        *parents, last = dotted.split(".")
        for part in parents:
            node = node.setdefault(part, {})
        node[last] = value
    return tree


def cmd_run(args: argparse.Namespace) -> int:
    cases = load_cases(args.cases, set(args.only or []))
    table = refs()
    name = args.name or args.impl
    tmp = Path(tempfile.mkdtemp(prefix=f"parity-{args.impl}-"))
    failures = 0
    settings = parse_settings(args.setting or [])
    s = build(args.device, tmp, args.impl, settings)
    try:
        for case in cases:
            engine = case["request"]["model"].split("/", 1)[0]
            folder = WORKSPACE / "fixtures" / engine / case["id"]
            folder.mkdir(parents=True, exist_ok=True)
            if args.skip_existing and (folder / f"{name}.json").exists():
                continue
            print(f"[{args.impl}] {case['id']} …", flush=True)
            try:
                path, record = run_case(s, case, table)
                if path is not None and args.repeat > 1:
                    first = path.read_bytes()
                    walls, identical = [record["wall_seconds"]], True
                    for _ in range(args.repeat - 1):
                        again, again_record = run_case(s, case, table)
                        walls.append(again_record["wall_seconds"])
                        identical = identical and again is not None and again.read_bytes() == first
                    record["wall_seconds_runs"] = walls
                    record["repeat_identical"] = identical
            except Exception as exc:  # recorded, so one bad case does not stop the corpus
                path, record = None, _exception_record(case, exc)
            record["impl"] = args.impl
            if settings:
                record["settings"] = settings
            record["device"] = args.device
            record["name"] = name
            if path is not None:
                shutil.copyfile(path, folder / f"{name}.wav")
                runs = ", ".join(f"{w:.1f}s" for w in record.get("wall_seconds_runs", [record["wall_seconds"]]))
                print(f"    ok {record['output']['duration']:.2f}s in {runs}", flush=True)
            else:
                failures += 1
                print(f"    {record['job_state']}: {(record.get('error') or {}).get('message')}", flush=True)
            (folder / f"{name}.json").write_text(json.dumps(record, ensure_ascii=False, indent=1), encoding="utf-8")
    finally:
        s.close()
        shutil.rmtree(tmp, ignore_errors=True)
    return 1 if failures else 0


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="cmd", required=True)
    run = sub.add_parser("run")
    run.add_argument("cases", nargs="+")
    run.add_argument("--impl", required=True, choices=["reference", "native", "upstream"])
    run.add_argument("--device", default="cpu")
    run.add_argument("--only", action="append")
    run.add_argument("--skip-existing", action="store_true")
    run.add_argument("--name", help="File stem for this run (default: the impl), e.g. reference-t28")
    run.add_argument("--setting", action="append", help="A setting for the whole run, key=value (the value as JSON, else a string); repeatable")
    run.add_argument("--repeat", type=int, default=1, help="Run each case this many times in the same stack (timings of a warm run)")
    args = parser.parse_args(argv)
    return cmd_run(args)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
