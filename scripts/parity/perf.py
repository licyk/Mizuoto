"""Where a request's time goes, and whether an optimisation kept its output (the performance work's harness).

    <python> scripts/parity/perf.py run <cases.jsonl>... [--only ID] [--name NAME] [--repeat N] [--device cpu]
        [--setting key=value] [--profile] [--torch-profile] [--reference native]
    <python> scripts/parity/perf.py compare --a NAME --b NAME [--report out.md]

``run`` runs each case ``--repeat`` times (default 3) through the whole stack in one process (``stack.py``'s
``build``/``run_case``): the first run loads the model and prepares the voice, the later ones are warm. Every run records:

- the wall time, the model's load, the first chunk, the engine's total and the real-time factor;
- **phases:** the time inside each ``ctx.use``/``ctx.run`` phase of the engine (named by ``ctx.run``'s name, else by
  its components), exclusive of phases nested in it, and ``(outside phases)``: the rest of the engine's time (text
  front ends, sampling glue, post-processing);
- the output audio (the first run's kept as ``<engine>-<case>.wav``), compared sample for sample with the other runs
  and with ``--reference``: an earlier run of this harness of that name, else the parity fixture
  ``<parity workspace>/fixtures/<engine>/<case>/<reference>.wav``.

``--profile`` runs the case once more, uncounted, under cProfile (``<case>.prof``, and the top functions by own and
cumulative time in ``<case>.profile.txt``); ``--torch-profile`` under torch's profiler (the operators by their own CPU
time in ``<case>.ops.txt``).

Results go to ``<perf workspace>/<name>/<engine>-<case>.json`` (``MIZUOTO_PERF_DIR``, default
``~/code_workspace/experiment/perf``); ``compare`` writes a Markdown table of the warm runs' times of two names and the
speed-up, with the phases that changed most.
"""

from __future__ import annotations

import argparse
import contextlib
import cProfile
import io
import json
import os
import pstats
import shutil
import statistics
import sys
import tempfile
import threading
import time
from collections.abc import Generator
from pathlib import Path
from typing import Any

from stack import WORKSPACE, build, load_cases, parse_settings, refs, run_case

PERF = Path(os.environ.get("MIZUOTO_PERF_DIR", "~/code_workspace/experiment/perf")).expanduser()


class _Frame:
    def __init__(self, started: float) -> None:
        self.started = started
        self.children = 0.0


class PhaseTimer:
    """Exclusive seconds per phase of the requests running in this process (one at a time)."""

    def __init__(self) -> None:
        self.seconds: dict[str, float] = {}
        self._stack: list[_Frame] = []
        self._pending = threading.local()

    def reset(self) -> None:
        self.seconds = {}
        self._stack = []

    @contextlib.contextmanager
    def timed(self, name: str) -> Generator[None, None, None]:
        frame = _Frame(time.perf_counter())
        self._stack.append(frame)
        try:
            yield
        finally:
            elapsed = time.perf_counter() - frame.started
            self._stack.remove(frame)
            self.seconds[name] = self.seconds.get(name, 0.0) + elapsed - frame.children
            if self._stack:
                self._stack[-1].children += elapsed

    def install(self) -> None:
        from mizuoto.runtime.context import RequestContext

        timer = self
        original_run = RequestContext.run
        original_use = RequestContext.use

        def run(ctx: Any, name: str, handles: Any, fn: Any, *args: Any, **kwargs: Any) -> Any:
            timer._pending.name = name
            return original_run(ctx, name, handles, fn, *args, **kwargs)

        @contextlib.contextmanager
        def use(ctx: Any, handles: Any, memory: int = 0) -> Generator[None, None, None]:
            handles = list(handles)
            name = getattr(timer._pending, "name", None) or "+".join(sorted(h.name.split(":", 1)[-1] for h in handles)) or "(no components)"
            timer._pending.name = None
            with timer.timed(name), original_use(ctx, handles, memory):
                yield

        setattr(RequestContext, "run", run)  # noqa: B010
        setattr(RequestContext, "use", use)  # noqa: B010


def _audio(path: Path) -> Any:
    from mizuoto.nn.dsp.wav import read_wav

    return read_wav(path)[0]


def _same(a: Any, b: Any) -> bool:
    import numpy as np

    return a.shape == b.shape and bool(np.array_equal(a, b))


def _one(s: Any, case: dict[str, Any], table: dict[str, Any], timer: PhaseTimer, profile: str | None, out: Path) -> tuple[Path | None, dict[str, Any]]:
    timer.reset()
    if profile == "python":
        profiler = cProfile.Profile()
        profiler.enable()
        try:
            path, record = run_case(s, case, table)
        finally:
            profiler.disable()
            profiler.dump_stats(str(out.with_suffix(".prof")))
            text = io.StringIO()
            stats = pstats.Stats(profiler, stream=text)
            stats.sort_stats("tottime").print_stats(45)
            stats.sort_stats("cumulative").print_stats(60)
            out.with_suffix(".profile.txt").write_text(text.getvalue(), encoding="utf-8")
    elif profile == "torch":
        from torch.profiler import ProfilerActivity
        from torch.profiler import profile as torch_profile

        with torch_profile(activities=[ProfilerActivity.CPU]) as ops_profile:
            path, record = run_case(s, case, table)
        out.with_suffix(".ops.txt").write_text(ops_profile.key_averages().table(sort_by="self_cpu_time_total", row_limit=50), encoding="utf-8")
    else:
        path, record = run_case(s, case, table)
    run: dict[str, Any] = {"wall": record.get("wall_seconds"), "state": record.get("job_state"), "error": record.get("error")}
    output = record.get("output") or {}
    meta = output.get("meta") or {}
    timings = meta.get("timings") or {}
    run.update({k: timings.get(k) for k in ("load", "first_chunk", "total", "rtf")})
    run["duration"] = output.get("duration")
    params = output.get("params") or {}
    if params.get("attention"):
        run["attention"] = params["attention"]  # the acceleration policy's report
    phases = dict(sorted(timer.seconds.items(), key=lambda kv: -kv[1]))
    run["phases"] = {k: round(v, 3) for k, v in phases.items()}
    if run.get("total") is not None:
        run["phases"]["(outside phases)"] = round(max(0.0, (run["total"] or 0.0) - sum(phases.values())), 3)
    return path, run


def _threads() -> int:
    import torch  # only once a case ran: the host sets the CPU's variables before torch is first imported

    return torch.get_num_threads()


def cmd_run(args: argparse.Namespace) -> int:
    cases = load_cases(args.cases, set(args.only or []))
    table = refs()
    folder = PERF / args.name
    folder.mkdir(parents=True, exist_ok=True)
    timer = PhaseTimer()
    timer.install()
    tmp = Path(tempfile.mkdtemp(prefix="perf-"))
    settings = parse_settings(args.setting or [])
    s = build(args.device, tmp, "native", settings)
    failures = 0
    try:
        for case in cases:
            engine = case["request"]["model"].split("/", 1)[0]
            out = folder / f"{engine}-{case['id']}.json"
            print(f"[{args.name}] {engine}/{case['id']} …", flush=True)
            runs, first = [], None
            for i in range(args.repeat):
                path, run = _one(s, case, table, timer, None, out)
                if path is not None:
                    audio = _audio(path)
                    if first is None:
                        first = audio
                        shutil.copyfile(path, out.with_suffix(".wav"))
                    run["same_as_first"] = _same(audio, first)
                runs.append(run)
                print(f"    run {i + 1}: {run['wall']:.2f}s wall, total {run.get('total') or 0:.2f}s, load {run.get('load') or 0:.2f}s", flush=True)
            if args.profile or args.torch_profile:  # one more run, profiled: its times are not counted
                _one(s, case, table, timer, "python" if args.profile else "torch", out)
            reference = PERF / args.reference / out.with_suffix(".wav").name
            if not reference.exists():
                reference = WORKSPACE / "fixtures" / engine / case["id"] / f"{args.reference}.wav"
            identical = None
            if first is not None and reference.exists():
                identical = _same(first, _audio(reference))
            ok = first is not None and all(r.get("same_as_first", False) for r in runs)
            failures += 0 if ok and identical is not False else 1
            print(f"    identical to {args.reference}: {identical}; runs agree: {ok}", flush=True)
            record = {
                "engine": engine,
                "case": case["id"],
                "name": args.name,
                "threads": _threads(),
                "environment": {k: os.environ[k] for k in ("GLIBC_TUNABLES", "THP_MEM_ALLOC_ENABLE", "OMP_PLACES", "OMP_PROC_BIND") if k in os.environ},
                "settings": settings,
                "reference": args.reference,
                "identical": identical,
                "runs_agree": ok,
                "runs": runs,
            }
            out.write_text(json.dumps(record, ensure_ascii=False, indent=1), encoding="utf-8")
    finally:
        s.close()
        shutil.rmtree(tmp, ignore_errors=True)
    return 1 if failures else 0


def _warm(record: dict[str, Any]) -> dict[str, Any]:
    """The fastest warm run's engine time and phases, and the warm runs' median (the first run when there is only one).
    The fastest, since the CPU's other work only ever slows a run down."""
    runs = [r for r in record["runs"] if r.get("total") is not None]
    warm = runs[1:] or runs
    if not warm:
        return {}
    best = min(warm, key=lambda r: r["total"])
    return {
        "total": best["total"],
        "median": statistics.median(r["total"] for r in warm),
        "first": runs[0]["total"] + (runs[0].get("load") or 0.0),
        "load": runs[0].get("load"),
        "duration": warm[0].get("duration"),
        "phases": best["phases"],
    }


def cmd_compare(args: argparse.Namespace) -> int:
    rows = []
    a_dir, b_dir = PERF / args.a, PERF / args.b
    for path in sorted(b_dir.glob("*.json")):
        other = a_dir / path.name
        if not other.exists():
            continue
        a, b = json.loads(other.read_text(encoding="utf-8")), json.loads(path.read_text(encoding="utf-8"))
        wa, wb = _warm(a), _warm(b)
        if not wa or not wb:
            rows.append(f"| {b['engine']} | {b['case']} | — | — | — | — | — | failed |")
            continue
        changes = []
        for name in sorted(set(wa["phases"]) | set(wb["phases"]), key=lambda n: -abs(wa["phases"].get(n, 0.0) - wb["phases"].get(n, 0.0)))[:3]:
            pa, pb = wa["phases"].get(name, 0.0), wb["phases"].get(name, 0.0)
            if abs(pa - pb) >= 0.05:
                changes.append(f"{name} {pa:.2f}→{pb:.2f}")
        same = "yes" if b["identical"] and b["runs_agree"] else ("no" if b["identical"] is False or not b["runs_agree"] else "?")
        speed = wa["total"] / wb["total"] if wb["total"] else float("nan")
        rows.append(
            f"| {b['engine']} | {b['case']} | {wb['duration'] or 0:.2f} | {wa['total']:.2f} → {wb['total']:.2f} | ×{speed:.2f} | "
            f"{(wa['load'] or 0):.1f} → {(wb['load'] or 0):.1f} | {'; '.join(changes) or '—'} | {same} |"
        )
    head = [
        f"## {args.a} → {args.b}",
        "",
        "Warm request (the fastest of the runs after the first: model loaded, voice prepared), the first run's model load, the",
        "phases that moved most, and whether the audio is identical to the reference and across runs.",
        "",
        "| engine | case | audio s | warm s | speed-up | load s | phases (s) | identical |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    text = "\n".join(head + rows) + "\n"
    if args.report:
        Path(args.report).write_text(text, encoding="utf-8")
    print(text)
    return 0


def cmd_show(args: argparse.Namespace) -> int:
    """Each case's warm phases of one name, longest first."""
    for path in sorted((PERF / args.name).glob("*.json")):
        record = json.loads(path.read_text(encoding="utf-8"))
        w = _warm(record)
        if not w:
            print(f"{record['engine']}/{record['case']}: failed")
            continue
        print(
            f"{record['engine']}/{record['case']}: warm {w['total']:.2f}s (median {w['median']:.2f}s) for {w['duration'] or 0:.2f}s of audio, load {w['load'] or 0:.1f}s, identical {record['identical']}"
        )
        for name, sec in sorted(w["phases"].items(), key=lambda kv: -kv[1]):
            print(f"    {sec:8.3f}  {name}")
    return 0


def main(argv: list[str]) -> int:
    p = argparse.ArgumentParser(prog="perf.py")
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("cases", nargs="+")
    r.add_argument("--only", action="append")
    r.add_argument("--name", required=True)
    r.add_argument("--repeat", type=int, default=3)
    r.add_argument("--device", default="cpu")
    r.add_argument("--setting", action="append")
    r.add_argument("--profile", action="store_true")
    r.add_argument("--torch-profile", action="store_true")
    r.add_argument("--reference", default="native")
    c = sub.add_parser("compare")
    c.add_argument("--a", required=True)
    c.add_argument("--b", required=True)
    c.add_argument("--report")
    sh = sub.add_parser("show")
    sh.add_argument("name")
    args = p.parse_args(argv)
    return {"run": cmd_run, "compare": cmd_compare, "show": cmd_show}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
