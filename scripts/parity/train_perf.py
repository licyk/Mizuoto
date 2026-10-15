"""Where a fine-tune's time goes, and whether an optimisation kept what it trains (the performance work's harness).

    <python> scripts/parity/train_perf.py run <engine> --fixture NAME --name NAME [--model ID] [--method M] [--hp k=v]...
        [--profile] [--no-previews] [--reference NAME]
    <python> scripts/parity/train_perf.py compare --a NAME --b NAME [--report out.md]

``run`` trains as ``train.py`` does (a temporary data folder over the real models, the workspace's ``narrator`` dataset,
every stage as a child process), with the base model, method, hyperparameters and language of the parity fixture
``<parity workspace>/train-fixtures/<engine>/<fixture>/native/experiment.json`` unless given. It records each stage's
wall time and, from the metric events as they arrive, the time between training steps; then every metric point and a
SHA-256 of the tensors of every weight file the export wrote (its other values apart). ``--reference NAME`` compares them with an earlier run of this harness
(identical metrics and weights). ``--profile`` runs the stages in this process instead, under cProfile
(``<engine>-<fixture>-<stage>.prof`` and ``.profile.txt``).

Results go to ``<perf workspace>/train/<name>/<engine>-<fixture>.json`` (``MIZUOTO_PERF_DIR``, default
``~/code_workspace/experiment/perf``).
"""

from __future__ import annotations

import argparse
import cProfile
import hashlib
import io
import json
import pstats
import shutil
import statistics
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

from perf import PERF
from stack import WORKSPACE, build

DATASET = WORKSPACE / "train-data" / "datasets" / "narrator"
WEIGHTS = {".safetensors", ".pth", ".pt", ".ckpt", ".bin"}


def _value(raw: str) -> Any:
    try:
        return json.loads(raw)
    except ValueError:
        return raw


class _Timing:
    def __init__(self) -> None:
        self.stages: dict[str, float] = {}
        self.steps: dict[str, list[float]] = {}  # stage -> arrival time of each metric step (from the stage's start)


def _install(timing: _Timing, profile_dir: Path | None, prefix: str) -> None:
    from mizuoto.core.jobs.events import MetricEvent, ResultEvent
    from mizuoto.core.jobs.jsonl import Emitter
    from mizuoto.core.jobs.service import JobContext

    original = JobContext.run_worker

    def run_worker(self: Any, target: Any, request: dict[str, Any], args: Any = None, on_event: Any = None, heartbeat_timeout: float = 0.0) -> dict[str, Any]:
        stage = str(request.get("stage"))
        started = time.perf_counter()
        seen: set[tuple[int, int]] = set()
        times = timing.steps.setdefault(stage, [])

        def event(e: Any) -> None:
            if isinstance(e, MetricEvent) and (e.epoch, e.step) not in seen:
                seen.add((e.epoch, e.step))
                times.append(time.perf_counter() - started)
            if on_event is not None:
                on_event(e)

        try:
            if profile_dir is None:
                return original(self, target, request, args, event, heartbeat_timeout)
            from mizuoto.training.stage import stage_main

            result: dict[str, Any] = {}

            class Forward(Emitter):
                def emit(self, e: Any) -> None:  # ty: ignore[invalid-method-override]
                    if isinstance(e, ResultEvent):
                        result.update(e.data)
                    event(e)

            profiler = cProfile.Profile()
            profiler.enable()
            try:
                stage_main(json.loads(json.dumps(request)), Forward())
            finally:
                profiler.disable()
                base = profile_dir / f"{prefix}-{stage}"
                profiler.dump_stats(str(base.with_suffix(".prof")))
                text = io.StringIO()
                stats = pstats.Stats(profiler, stream=text)
                stats.sort_stats("tottime").print_stats(45)
                stats.sort_stats("cumulative").print_stats(60)
                base.with_suffix(".profile.txt").write_text(text.getvalue(), encoding="utf-8")
            return result
        finally:
            timing.stages[stage] = timing.stages.get(stage, 0.0) + time.perf_counter() - started

    JobContext.run_worker = run_worker


def _digest(obj: Any, h: Any, others: dict[str, str], where: str) -> None:
    import torch

    if isinstance(obj, torch.Tensor):
        t = obj.detach().cpu().contiguous()
        h.update(f"{where}:{t.dtype}:{tuple(t.shape)}".encode())
        h.update(t.reshape(-1).view(torch.uint8).numpy().tobytes() if t.numel() else b"")
    elif isinstance(obj, dict):
        for k in obj:
            _digest(obj[k], h, others, f"{where}/{k}")
    elif isinstance(obj, list | tuple):
        for i, v in enumerate(obj):
            _digest(v, h, others, f"{where}[{i}]")
    else:
        others[where] = repr(obj)[:200]


def _hashes(folder: Path) -> dict[str, Any]:
    """Each weight file's tensors as one SHA-256 (keys, dtypes, shapes and bytes, in the file's order) and its other
    values (a checkpoint's configuration, paths, dates) apart, so a path of the temporary data folder is not taken for a
    different result."""
    import torch
    from safetensors.torch import load_file

    out: dict[str, Any] = {}
    for path in sorted(folder.rglob("*")):
        if not (path.is_file() and path.suffix in WEIGHTS):
            continue
        rel = str(path.relative_to(folder))
        if path.suffix == ".safetensors":
            obj: Any = load_file(str(path))
        else:
            data = bytearray(path.read_bytes())
            if data[:2] != b"PK" and data[2:4] == b"\x03\x04":  # GPT-SoVITS's version tag over the zip magic
                data[:2] = b"PK"
            obj = torch.load(io.BytesIO(bytes(data)), map_location="cpu", weights_only=False)
        h = hashlib.sha256()
        others: dict[str, str] = {}
        _digest(obj, h, others, "")
        out[rel] = {"tensors": h.hexdigest(), "others": others}
    return out


def _repeat_clips(folder: Path, times: int) -> None:
    """Every clip of the dataset ``times`` times (copies under new ids), for a method whose minimum the 53 s of
    ``narrator`` do not reach (CosyVoice's 60 s)."""
    rows = [json.loads(line) for line in (folder / "manifest.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    out = list(rows)
    for n in range(1, times):
        for row in rows:
            copy = dict(row, id=f"{row['id']}-{n}", audio=row["audio"].replace(row["id"], f"{row['id']}-{n}"))
            shutil.copyfile(folder / row["audio"], folder / copy["audio"])
            out.append(copy)
    (folder / "manifest.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in out), encoding="utf-8")


def cmd_run(args: argparse.Namespace) -> int:
    from mizuoto.core.training.models import ExperimentInput

    fixture = json.loads((WORKSPACE / "train-fixtures" / args.engine / args.fixture / "native" / "experiment.json").read_text(encoding="utf-8"))
    model = args.model or fixture["base_model"]
    method = args.method or fixture.get("method")
    hyperparams = dict(fixture.get("hyperparams") or {})
    hyperparams.update({k: _value(v) for k, v in (h.split("=", 1) for h in args.hp or [])})
    language = args.language or fixture.get("language")
    folder = PERF / "train" / args.name
    folder.mkdir(parents=True, exist_ok=True)
    prefix = f"{args.engine}-{args.fixture}"
    timing = _Timing()
    _install(timing, folder if args.profile else None, prefix)
    tmp = Path(tempfile.mkdtemp(prefix="train-perf-"))
    s = build(args.device, tmp, "native")
    try:
        s.settings.update({"training": {"previews": not args.no_previews}})
        shutil.copytree(DATASET, s.datasets.root / DATASET.name)
        if args.clips_repeat > 1:
            _repeat_clips(s.datasets.root / DATASET.name, args.clips_repeat)
        dataset = json.loads((DATASET / "dataset.json").read_text(encoding="utf-8"))
        exp = s.training.create(ExperimentInput(name=args.fixture, base_model=model, dataset=dataset["id"], method=method, hyperparams=hyperparams, language=language))
        print(f"[{args.name}] training {exp.key} ({exp.method}, {hyperparams}) …", flush=True)
        started = time.perf_counter()
        job = s.training.run(exp.key, wait=True)
        wall = time.perf_counter() - started
        exp = s.training.get(exp.key)
        exp_folder = s.training.folder(exp)
        metrics_path = exp_folder / "metrics.jsonl"
        metrics = [json.loads(line) for line in metrics_path.read_text(encoding="utf-8").splitlines() if line.strip()] if metrics_path.is_file() else []
        weights: dict[str, Any] = {}
        for a in exp.artifacts:
            if a.kind == "model":
                info = s.models.get(a.id)
                if info.path:
                    weights.update({f"model/{k}": v for k, v in _hashes(Path(info.path)).items()})
                    shutil.rmtree(info.path, ignore_errors=True)  # registered in the real models folder: taken back out
            elif a.kind == "voice" and s.training.voices is not None:
                voice = s.training.voices.get(a.id)
                weights.update({f"voice/{k}": v for k, v in _hashes(s.training.voices.root / voice.slug / "weights").items()})
        record: dict[str, Any] = {
            "engine": args.engine,
            "fixture": args.fixture,
            "name": args.name,
            "model": model,
            "method": method,
            "hyperparams": hyperparams,
            "profiled": bool(args.profile),
            "state": job.state,
            "error": job.error.model_dump() if job.error else None,
            "wall": wall,
            "stages": timing.stages,
            "steps": timing.steps,
            "metrics": metrics,
            "weights": weights,
        }
        if args.reference:
            ref = json.loads((PERF / "train" / args.reference / f"{prefix}.json").read_text(encoding="utf-8"))
            record["same_metrics"] = ref["metrics"] == metrics
            record["same_weights"] = bool(weights) and {k: v["tensors"] for k, v in ref["weights"].items()} == {k: v["tensors"] for k, v in weights.items()}
            record["other_values_differ"] = {
                k: sorted(n for n in set(v["others"]) | set(ref["weights"][k]["others"]) if v["others"].get(n) != ref["weights"][k]["others"].get(n))
                for k, v in weights.items()
                if k in ref["weights"]
            }
        (folder / f"{prefix}.json").write_text(json.dumps(record, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"    {job.state} in {wall:.1f}s; stages " + ", ".join(f"{k} {v:.1f}s" for k, v in timing.stages.items()), flush=True)
        if args.reference:
            print(f"    same metrics as {args.reference}: {record['same_metrics']}; same weights: {record['same_weights']}", flush=True)
        return 0 if job.state == "completed" and record.get("same_metrics", True) and record.get("same_weights", True) else 1
    finally:
        s.close()
        shutil.rmtree(tmp, ignore_errors=True)


def _step_seconds(times: list[float]) -> float | None:
    gaps = [b - a for a, b in zip(times, times[1:], strict=False)]
    return statistics.median(gaps) if gaps else None


def cmd_compare(args: argparse.Namespace) -> int:
    rows = []
    for path in sorted((PERF / "train" / args.b).glob("*.json")):
        other = PERF / "train" / args.a / path.name
        if not other.exists():
            continue
        a, b = json.loads(other.read_text(encoding="utf-8")), json.loads(path.read_text(encoding="utf-8"))
        cells = []
        for stage in b["stages"]:
            sa, sb = a["stages"].get(stage), b["stages"][stage]
            if sa is not None:
                cells.append(f"{stage} {sa:.1f}→{sb:.1f}")
        step_a, step_b = _step_seconds(a["steps"].get("train", [])), _step_seconds(b["steps"].get("train", []))
        step = f"{step_a:.2f} → {step_b:.2f}" if step_a and step_b else "—"
        same = "yes" if b.get("same_metrics") and b.get("same_weights") else ("no" if b.get("same_metrics") is False or b.get("same_weights") is False else "?")
        rows.append(f"| {b['engine']} | {b['fixture']} | {a['wall']:.1f} → {b['wall']:.1f} | ×{a['wall'] / b['wall']:.2f} | {'; '.join(cells)} | {step} | {same} |")
    head = [
        f"## Training: {args.a} → {args.b}",
        "",
        "| engine | fixture | run s | speed-up | stages (s) | s per train step | identical |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    text = "\n".join(head + rows) + "\n"
    if args.report:
        Path(args.report).write_text(text, encoding="utf-8")
    print(text)
    return 0


def main(argv: list[str]) -> int:
    p = argparse.ArgumentParser(prog="train_perf.py")
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("engine")
    r.add_argument("--fixture", required=True)
    r.add_argument("--name", required=True)
    r.add_argument("--model")
    r.add_argument("--method")
    r.add_argument("--hp", action="append")
    r.add_argument("--language")
    r.add_argument("--device", default="cpu")
    r.add_argument("--profile", action="store_true")
    r.add_argument("--no-previews", action="store_true")
    r.add_argument("--reference")
    r.add_argument("--clips-repeat", type=int, default=1, help="Each dataset clip this many times (copies)")
    c = sub.add_parser("compare")
    c.add_argument("--a", required=True)
    c.add_argument("--b", required=True)
    c.add_argument("--report")
    args = p.parse_args(argv)
    return cmd_run(args) if args.cmd == "run" else cmd_compare(args)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
