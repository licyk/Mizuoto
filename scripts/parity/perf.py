"""Where a request's time goes, and whether an optimisation kept its output (the performance work's harness).

    <python> scripts/parity/perf.py run <cases.jsonl>... [--only ID] [--name NAME] [--repeat N] [--device cpu]
        [--setting key=value] [--profile] [--torch-profile [--keep-trace]] [--deterministic] [--reference native]
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

On a CUDA device (``--device cuda:N``, which every engine's ``engines.<id>.device`` is set to) the harness waits for the
device at each phase boundary, so a phase's time is its work rather than its launches, and records per phase the peak
allocation (``torch.cuda.max_memory_allocated``), the working memory over what was allocated once the phase's weights
were placed, and the working memory the engine asked for (``ctx.use``'s ``memory``), the call that came closest to its own
estimate (``worst_work_mb``, ``worst_estimate_mb``), with the request's peak allocated and reserved memory. The two waits per phase are the harness's own; the engines' code is unchanged.

``--profile`` runs the case once more, uncounted, under cProfile (``<case>.prof``, and the top functions by own and
cumulative time in ``<case>.profile.txt``); ``--torch-profile`` under torch's profiler (the operators by their own CPU
time, or on CUDA by their own device time, in ``<case>.ops.txt``). On CUDA the profile is also summarised per phase in
``<case>.gpu.json`` and in the record: the wall time, the time the device was busy (kernels and copies, overlaps
counted once), kernels and their launches, CUDA graph replays, host waits for the device (``cudaStreamSynchronize``,
``cudaEventSynchronize``, synchronous ``cudaMemcpy``; ``cudaDeviceSynchronize`` apart, since the harness makes two per
phase), and host-to-device and device-to-host copies. ``--keep-trace`` keeps the Chrome trace it is read from.

``--deterministic`` runs with ``torch.use_deterministic_algorithms(True, warn_only=True)`` and
``CUBLAS_WORKSPACE_CONFIG=:4096:8``, both set by the harness (native code never sets them); the operations that have no
deterministic implementation are listed in the record (``nondeterministic``) instead of failing the case.

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
import warnings
from collections.abc import Generator
from pathlib import Path
from typing import Any

from stack import WORKSPACE, build, load_cases, parse_settings, refs, run_case

PERF = Path(os.environ.get("MIZUOTO_PERF_DIR", "~/code_workspace/experiment/perf")).expanduser()


MB = 2**20


def _cuda(device: str) -> Any:
    """torch, when ``device`` is a CUDA device and torch has started CUDA (an engine placed a phase on it); else None."""
    torch = sys.modules.get("torch") if device.startswith("cuda") else None
    return torch if torch is not None and torch.cuda.is_initialized() else None


class _Frame:
    def __init__(self, name: str, started: float) -> None:
        self.name = name
        self.started = started
        self.children = 0.0
        self.placed = 0  # bytes allocated on the device once the phase's weights were placed
        self.peak = 0  # the device's peak allocation during the phase after placement, nested phases included
        self.estimate = 0  # the working memory this call set aside


class PhaseTimer:
    """Exclusive seconds per phase of the requests running in this process (one at a time). On a CUDA device it waits
    for the device at each phase boundary and records each phase's memory (see the module's docstring)."""

    def __init__(self) -> None:
        self.seconds: dict[str, float] = {}
        self.entered: dict[str, int] = {}
        self.memory: dict[str, dict[str, float]] = {}
        self.peak = 0  # the request's peak allocated bytes
        self.peak_reserved = 0
        self.annotate = False  # name each phase in torch's profiler (``record_function``)
        self._stack: list[_Frame] = []
        self._pending = threading.local()

    def reset(self) -> None:
        self.seconds = {}
        self.entered = {}
        self.memory = {}
        self.peak = self.peak_reserved = 0
        self._stack = []

    def fold(self, torch: Any, device: str) -> int:
        """The device's peak since the last reset, folded into the request's and the running phase's, then reset."""
        peak = torch.cuda.max_memory_allocated(device)
        self.peak = max(self.peak, peak)
        self.peak_reserved = max(self.peak_reserved, torch.cuda.max_memory_reserved(device))
        if self._stack:
            self._stack[-1].peak = max(self._stack[-1].peak, peak)
        torch.cuda.reset_peak_memory_stats(device)
        return peak

    @contextlib.contextmanager
    def timed(self, name: str, device: str = "cpu") -> Generator[_Frame, None, None]:
        torch = _cuda(device)
        if torch is not None:
            torch.cuda.synchronize(device)
            self.fold(torch, device)
        frame = _Frame(name, time.perf_counter())
        self._stack.append(frame)
        try:
            if self.annotate:
                from torch.profiler import record_function

                with record_function(f"phase: {name}"):
                    yield frame
            else:
                yield frame
        finally:
            if torch is not None:
                torch.cuda.synchronize(device)
            elapsed = time.perf_counter() - frame.started
            if torch is not None:
                self.fold(torch, device)  # into this frame, still on the stack
            self._stack.remove(frame)
            self.seconds[name] = self.seconds.get(name, 0.0) + elapsed - frame.children
            self.entered[name] = self.entered.get(name, 0) + 1
            if self._stack:
                self._stack[-1].children += elapsed
                self._stack[-1].peak = max(self._stack[-1].peak, frame.peak)

    def placed(self, frame: _Frame, device: str, estimate: int) -> None:
        """The phase's weights are on the device: what is allocated now is where its working memory starts."""
        torch = _cuda(device)
        if torch is None:
            return
        self.fold(torch, device)  # the placement's own peak counts for the request, not for the phase's working memory
        frame.peak = frame.placed = torch.cuda.memory_allocated(device)
        frame.estimate = estimate
        m = self.memory.setdefault(frame.name, {"peak_mb": 0.0, "work_mb": 0.0, "estimate_mb": 0.0, "placed_mb": 0.0})
        m["estimate_mb"] = max(m["estimate_mb"], round(estimate / MB, 1))
        m["placed_mb"] = max(m["placed_mb"], round(frame.placed / MB, 1))

    def closed(self, frame: _Frame, device: str) -> None:
        """The phase ended: its peak and its working memory over what was placed."""
        torch = _cuda(device)
        if torch is None or frame.name not in self.memory:
            return
        self.fold(torch, device)
        m = self.memory[frame.name]
        m["peak_mb"] = max(m["peak_mb"], round(frame.peak / MB, 1))
        m["work_mb"] = max(m["work_mb"], round((frame.peak - frame.placed) / MB, 1))
        # the maxima above may come from different calls (a reference's encoding, a decoding): the call that came
        # closest to its own estimate, with both its numbers
        work = (frame.peak - frame.placed) / MB
        if frame.estimate and work / (frame.estimate / MB) > m.get("worst_ratio", 0.0):
            m["worst_ratio"] = round(work / (frame.estimate / MB), 3)
            m["worst_work_mb"], m["worst_estimate_mb"] = round(work, 1), round(frame.estimate / MB, 1)

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
            device = str(getattr(ctx, "device", "cpu"))
            with timer.timed(name, device) as frame:
                try:
                    with original_use(ctx, handles, memory):
                        timer.placed(frame, device, int(memory))
                        yield
                finally:
                    timer.closed(frame, device)

        setattr(RequestContext, "run", run)  # noqa: B010
        setattr(RequestContext, "use", use)  # noqa: B010


def _audio(path: Path) -> Any:
    from mizuoto.nn.dsp.wav import read_wav

    return read_wav(path)[0]


def _same(a: Any, b: Any) -> bool:
    import numpy as np

    return a.shape == b.shape and bool(np.array_equal(a, b))


_HOST_WAITS = {"cudaStreamSynchronize", "cudaEventSynchronize", "cudaMemcpy", "cuStreamSynchronize", "cuEventSynchronize", "cuMemcpyDtoH_v2"}
_LAUNCHES = {"cudaLaunchKernel", "cudaLaunchKernelExC", "cuLaunchKernel", "cuLaunchKernelEx"}
_DEVICE_WORK = {"kernel", "gpu_memcpy", "gpu_memset"}


def _busy(spans: list[tuple[float, float]]) -> float:
    """The length of the union of ``spans`` (overlapping work on several streams counted once)."""
    total, end = 0.0, float("-inf")
    for a, b in sorted(spans):
        if b > end:
            total += b - max(a, end)
            end = b
    return total


def gpu_summary(trace: Path) -> dict[str, Any]:
    """Per phase, from a Chrome trace of torch's profiler with the phases annotated: wall and device-busy milliseconds,
    kernels, launches, graph replays, host waits and copies. A device event belongs to the innermost phase whose wall span
    contains its start; that holds because the harness waits for the device at each phase boundary."""
    events = json.loads(trace.read_text(encoding="utf-8")).get("traceEvents", [])
    spans = sorted(
        (
            (e["ts"], e["ts"] + e.get("dur", 0), e["name"].removeprefix("phase: "))
            for e in events
            if e.get("cat") == "user_annotation" and str(e.get("name", "")).startswith("phase: ")
        ),
        key=lambda s: (s[0], -s[1]),
    )

    def phase(ts: float) -> str:
        found = "(outside phases)"
        for a, b, name in spans:  # sorted by start: the last match is the innermost
            if a > ts:
                break
            if ts < b:
                found = name
        return found

    out: dict[str, dict[str, Any]] = {}

    def bucket(name: str) -> dict[str, Any]:
        return out.setdefault(
            name,
            {
                "wall_ms": 0.0,
                "gpu_ms": 0.0,
                "kernels": 0,
                "launches": 0,
                "graph_launches": 0,
                "host_waits": 0,
                "device_syncs": 0,
                "h2d": 0,
                "h2d_ms": 0.0,
                "h2d_mb": 0.0,
                "d2h": 0,
                "d2h_ms": 0.0,
                "d2h_mb": 0.0,
                "_spans": [],
            },
        )

    for i, (a, b, name) in enumerate(spans):  # exclusive wall time: each phase's span less those of the phases directly in it
        bucket(name)["wall_ms"] += (b - a) / 1000
        parent = next((spans[j][2] for j in range(i - 1, -1, -1) if spans[j][0] <= a and b <= spans[j][1]), None)
        if parent is not None:
            bucket(parent)["wall_ms"] -= (b - a) / 1000
    for e in events:
        cat, name, ts = e.get("cat"), str(e.get("name", "")), e.get("ts")
        if ts is None:
            continue
        if cat in _DEVICE_WORK:
            b = bucket(phase(ts))
            b["_spans"].append((ts, ts + e.get("dur", 0)))
            if cat == "kernel":
                b["kernels"] += 1
            elif cat == "gpu_memcpy":
                kind = "h2d" if "HtoD" in name else "d2h" if "DtoH" in name else None
                if kind:
                    b[kind] += 1
                    b[f"{kind}_ms"] += e.get("dur", 0) / 1000
                    b[f"{kind}_mb"] += (e.get("args") or {}).get("bytes", 0) / MB
        elif cat in ("cuda_runtime", "cuda_driver"):
            b = bucket(phase(ts))
            if name in _LAUNCHES and cat == "cuda_runtime":
                b["launches"] += 1
            elif name == "cudaGraphLaunch":
                b["graph_launches"] += 1
            elif name in _HOST_WAITS:
                b["host_waits"] += 1
            elif name == "cudaDeviceSynchronize":
                b["device_syncs"] += 1
    if "(outside phases)" in out:
        del out["(outside phases)"]["wall_ms"]  # not a span of its own: the record's phases have it
    for b in out.values():
        b["gpu_ms"] = _busy(b.pop("_spans")) / 1000
        for k, v in b.items():
            if isinstance(v, float):
                b[k] = round(v, 3)
    return dict(sorted(out.items(), key=lambda kv: -kv[1].get("wall_ms", 0.0)))


def _one(
    s: Any, case: dict[str, Any], table: dict[str, Any], timer: PhaseTimer, profile: str | None, out: Path, device: str = "cpu", keep_trace: bool = False
) -> tuple[Path | None, dict[str, Any]]:
    timer.reset()
    torch = _cuda(device)
    if torch is not None:
        torch.cuda.reset_peak_memory_stats(device)
    nondeterministic: set[str] = set()
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

        timer.annotate = True
        try:
            with torch_profile(activities=[ProfilerActivity.CPU, *([ProfilerActivity.CUDA] if torch is not None else [])]) as ops_profile:
                path, record = run_case(s, case, table)
        finally:
            timer.annotate = False
        sort = "self_device_time_total" if torch is not None else "self_cpu_time_total"
        out.with_suffix(".ops.txt").write_text(ops_profile.key_averages().table(sort_by=sort, row_limit=60), encoding="utf-8")
        if torch is not None:
            trace = out.with_suffix(".trace.json")
            ops_profile.export_chrome_trace(str(trace))
            summary = gpu_summary(trace)
            out.with_suffix(".gpu.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
            if not keep_trace:
                trace.unlink()
    else:
        with warnings.catch_warnings(record=True) as caught:  # the deterministic mode's alerts (warn_only): one per operation
            warnings.simplefilter("always")
            path, record = run_case(s, case, table)
        nondeterministic = {str(w.message).split(" does not have a deterministic implementation")[0] for w in caught if "deterministic implementation" in str(w.message)}
    run: dict[str, Any] = {"wall": record.get("wall_seconds"), "state": record.get("job_state"), "error": record.get("error")}
    output = record.get("output") or {}
    meta = output.get("meta") or {}
    timings = meta.get("timings") or {}
    run.update({k: timings.get(k) for k in ("load", "first_chunk", "total", "rtf")})
    run["duration"] = output.get("duration")
    params = output.get("params") or {}
    if params.get("attention"):
        run["attention"] = params["attention"]  # the acceleration policy's report
    for key in ("oom_retries", "fallbacks"):  # phases run again after running out of device memory
        if params.get(key):
            run[key] = params[key]
    if meta.get("memory"):
        run["placement"] = meta["memory"]  # loaded, streamed and partial placements, as the output reports them
    phases = dict(sorted(timer.seconds.items(), key=lambda kv: -kv[1]))
    run["phases"] = {k: round(v, 3) for k, v in phases.items()}
    if run.get("total") is not None:
        run["phases"]["(outside phases)"] = round(max(0.0, (run["total"] or 0.0) - sum(phases.values())), 3)
    run["phase_entries"] = dict(timer.entered)
    if torch is None:
        torch = _cuda(device)  # loaded by this run
    if torch is not None:
        timer.fold(torch, device)
        run["memory"] = {"peak_mb": round(timer.peak / MB, 1), "reserved_mb": round(timer.peak_reserved / MB, 1), "phases": timer.memory}
    if nondeterministic:
        run["nondeterministic"] = sorted(nondeterministic)
    if profile == "torch" and torch is not None:
        run["gpu_profile"] = summary
    return path, run


def _machine(device: str) -> dict[str, Any]:
    """The GPU, driver and torch build a CUDA run measured."""
    import subprocess

    import torch
    from torch.version import cuda

    index = torch.device(device).index or 0
    info: dict[str, Any] = {
        "gpu": torch.cuda.get_device_name(index),
        "capability": ".".join(map(str, torch.cuda.get_device_capability(index))),
        "torch": torch.__version__,
        "cuda": cuda,
        "cudnn": torch.backends.cudnn.version(),
    }
    with contextlib.suppress(OSError, subprocess.SubprocessError):
        query = "driver_version,pcie.link.gen.max,pcie.link.width.max,memory.used,utilization.gpu"
        visible = os.environ.get("CUDA_VISIBLE_DEVICES")
        physical = visible.split(",")[index] if visible else str(index)
        row = subprocess.run(["nvidia-smi", f"--id={physical}", f"--query-gpu={query}", "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=10).stdout.strip()
        info.update(dict(zip(query.split(","), (v.strip() for v in row.split(",")), strict=False)))
    return info


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
    if args.deterministic:
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"  # read when cuBLAS starts, so before any model runs
    s = build(args.device, tmp, "native", settings)
    if args.deterministic:
        s.inference.runtime  # noqa: B018 -- the host's CPU settings first: they must precede torch's import
        import torch

        torch.use_deterministic_algorithms(True, warn_only=True)
    failures = 0
    try:
        for case in cases:
            engine = case["request"]["model"].split("/", 1)[0]
            out = folder / f"{engine}-{case['id']}.json"
            print(f"[{args.name}] {engine}/{case['id']} …", flush=True)
            runs, first = [], None
            for i in range(args.repeat):
                path, run = _one(s, case, table, timer, None, out, args.device)
                if path is not None:
                    audio = _audio(path)
                    if first is None:
                        first = audio
                        shutil.copyfile(path, out.with_suffix(".wav"))
                    run["same_as_first"] = _same(audio, first)
                runs.append(run)
                print(f"    run {i + 1}: {run['wall']:.2f}s wall, total {run.get('total') or 0:.2f}s, load {run.get('load') or 0:.2f}s", flush=True)
            if args.profile or args.torch_profile:  # one more run, profiled: its times are not counted
                _one(s, case, table, timer, "python" if args.profile else "torch", out, args.device, args.keep_trace)
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
                "device": args.device,
                **({"machine": _machine(args.device)} if args.device.startswith("cuda") else {}),
                "deterministic": bool(args.deterministic),
                "stream": bool(case.get("stream")),
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
        "first_chunk": best.get("first_chunk"),
        "peak_mb": (best.get("memory") or {}).get("peak_mb"),
        "median": statistics.median(r["total"] for r in warm),
        "first": runs[0]["total"] + (runs[0].get("load") or 0.0),
        "load": runs[0].get("load"),
        "duration": warm[0].get("duration"),
        "phases": best["phases"],
    }


def _records(folder: Path) -> list[Path]:
    """A round's case records (``<engine>-<case>.json``), without the profiles' ``.gpu.json`` and ``.trace.json``."""
    return [p for p in sorted(folder.glob("*.json")) if not p.name.endswith((".gpu.json", ".trace.json"))]


def cmd_compare(args: argparse.Namespace) -> int:
    rows = []
    a_dir, b_dir = PERF / args.a, PERF / args.b
    for path in _records(b_dir):
        other = a_dir / path.name
        if not other.exists():
            continue
        a, b = json.loads(other.read_text(encoding="utf-8")), json.loads(path.read_text(encoding="utf-8"))
        wa, wb = _warm(a), _warm(b)
        if not wa or not wb:
            rows.append(f"| {b['engine']} | {b['case']} | — | — | — | — | — | — | failed |")
            continue
        changes = []
        for name in sorted(set(wa["phases"]) | set(wb["phases"]), key=lambda n: -abs(wa["phases"].get(n, 0.0) - wb["phases"].get(n, 0.0)))[:3]:
            pa, pb = wa["phases"].get(name, 0.0), wb["phases"].get(name, 0.0)
            if abs(pa - pb) >= 0.05:
                changes.append(f"{name} {pa:.2f}→{pb:.2f}")
        same = "yes" if b["identical"] and b["runs_agree"] else ("no" if b["identical"] is False or not b["runs_agree"] else "?")
        speed = wa["total"] / wb["total"] if wb["total"] else float("nan")
        first = f"{wa['first_chunk'] or 0:.2f} → {wb['first_chunk'] or 0:.2f}" if b.get("stream") and wa.get("first_chunk") and wb.get("first_chunk") else "—"
        rows.append(
            f"| {b['engine']} | {b['case']} | {wb['duration'] or 0:.2f} | {wa['total']:.2f} → {wb['total']:.2f} | ×{speed:.2f} | {first} | "
            f"{(wa['load'] or 0):.1f} → {(wb['load'] or 0):.1f} | {'; '.join(changes) or '—'} | {same} |"
        )
    head = [
        f"## {args.a} → {args.b}",
        "",
        "Warm request (the fastest of the runs after the first: model loaded, voice prepared), the time to its first audio for",
        "streamed cases, the first run's model load, the phases that moved most, and whether the audio is identical to the",
        "reference and across runs.",
        "",
        "| engine | case | audio s | warm s | speed-up | first audio s | load s | phases (s) | identical |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    text = "\n".join(head + rows) + "\n"
    if args.report:
        Path(args.report).write_text(text, encoding="utf-8")
    print(text)
    return 0


def cmd_show(args: argparse.Namespace) -> int:
    """Each case's warm phases of one name, longest first."""
    for path in _records(PERF / args.name):
        record = json.loads(path.read_text(encoding="utf-8"))
        w = _warm(record)
        if not w:
            print(f"{record['engine']}/{record['case']}: failed")
            continue
        extra = (f", first audio {w['first_chunk']:.2f}s" if record.get("stream") and w.get("first_chunk") else "") + (f", peak {w['peak_mb']:.0f} MB" if w.get("peak_mb") else "")
        print(
            f"{record['engine']}/{record['case']}: warm {w['total']:.2f}s (median {w['median']:.2f}s) for {w['duration'] or 0:.2f}s of audio, load {w['load'] or 0:.1f}s{extra}, identical {record['identical']}"
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
    r.add_argument("--keep-trace", action="store_true", help="Keep the CUDA profile's Chrome trace (<case>.trace.json)")
    r.add_argument("--deterministic", action="store_true", help="torch.use_deterministic_algorithms (warn only) and CUBLAS_WORKSPACE_CONFIG=:4096:8")
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
