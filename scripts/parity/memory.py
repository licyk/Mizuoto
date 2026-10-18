"""Memory and speed of the parity cases, one case per process.

    <python> scripts/parity/memory.py run <cases.jsonl>... [--only ID] [--device cpu|cuda:N] [--vram-limit-mb N]
        [--setting key=value] [--name NAME] [--report out.md]
    <python> scripts/parity/memory.py report [--name NAME] [--report out.md]

Each case runs in a fresh child process (so peaks are its own) through the whole stack, as ``stack.py`` runs it, in a
temporary data folder over the real models. The child records:

- **times:** the model's load, the first chunk, the request, the real-time factor (from the output's timings); then,
  with the model loaded, offloading it (``release_idle``), loading it back (``ModelManager.load``) and a second run;
- **memory:** on a GPU the peak allocated and reserved by torch during the first and the second run, and the bytes the
  manager holds per handle; on the CPU the process's peak resident set (``ru_maxrss``) above its size before the run;
- **residency:** ``ModelManager.resident()`` after the first run (loaded, streamed and host bytes once the runtime
  reports them).

``--vram-limit-mb N`` plays a smaller GPU: the child caps torch's allocator (``set_per_process_memory_fraction``, a
process-wide call allowed here only) and sets ``compute.vram_limit_mb`` so the runtime plans for the cap. Before any GPU
run, check that the device has that much free (the GPUs here may belong to another workload).

Results go to ``<workspace>/memory/<name>/<engine>-<case>.json``; ``report`` (also run at the end of ``run``) writes a Markdown
table. The workspace is ``MIZUOTO_PARITY_DIR`` (default ``~/code_workspace/experiment/kodama-refactor``).
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

from stack import WORKSPACE, build, load_cases, parse_settings, refs, run_case

HERE = Path(__file__).resolve().parent


def _rss_mb() -> float:
    if sys.platform == "win32":
        raise SystemExit("memory.py measures the CPU's peak with resource.getrusage, which Windows does not have")
    import resource

    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024  # Linux: KiB


def child(case: dict[str, Any], device: str, settings: dict[str, Any], limit_mb: int | None) -> dict[str, Any]:
    """Run one case twice in this process and measure it."""
    torch = None
    if device.startswith("cuda"):
        import torch as _torch

        torch = _torch
        index = torch.device(device).index or 0
        if limit_mb:
            total = torch.cuda.get_device_properties(index).total_memory
            torch.cuda.set_per_process_memory_fraction(min(1.0, limit_mb * 2**20 / total), index)
    table = refs()
    tmp = Path(tempfile.mkdtemp(prefix="parity-memory-"))
    engine = case["request"]["model"].split("/", 1)[0]
    record: dict[str, Any] = {"case": f"{engine}/{case['id']}", "device": device, "vram_limit_mb": limit_mb, "settings": settings}
    if limit_mb:  # the services read no environment here (``build``): the runtime learns the cap as a setting
        settings = {**settings, "compute": {**settings.get("compute", {}), "vram_limit_mb": limit_mb}}
    s = build(device, tmp, "native", settings)
    try:
        rss_before = _rss_mb()
        if torch is not None and torch.cuda.is_initialized():  # else nothing is allocated yet: the peak starts at zero
            torch.cuda.reset_peak_memory_stats(device)
        started = time.perf_counter()
        path, first = run_case(s, case, table)
        record["wall_seconds"] = time.perf_counter() - started
        record["job_state"] = first["job_state"]
        record["error"] = first.get("error")
        if path is None:
            return record
        meta = first["output"].get("meta") or {}
        record["duration"] = first["output"].get("duration")
        record["timings"] = meta.get("timings")
        record["memory_meta"] = meta.get("memory")
        record["applied"] = meta.get("applied")
        record["warnings"] = meta.get("warnings")
        if torch is not None:
            record["peak_allocated_mb"] = torch.cuda.max_memory_allocated(device) / 2**20
            record["peak_reserved_mb"] = torch.cuda.max_memory_reserved(device) / 2**20
        else:
            record["peak_rss_mb"] = _rss_mb()
            record["rss_before_mb"] = rss_before
        runtime = s.runtime.inference.runtime
        manager = runtime.manager
        record["resident"] = [vars(r) if not hasattr(r, "model_dump") else r.model_dump() for r in manager.resident()]
        # offload, load back, run again
        t = time.perf_counter()
        manager.release_idle(device)
        if torch is not None:
            torch.cuda.synchronize(device)
        record["offload_seconds"] = time.perf_counter() - t
        handles = [e.handle for e in list(manager._entries)]
        t = time.perf_counter()
        manager.load(handles)
        if torch is not None:
            torch.cuda.synchronize(device)
        record["reload_seconds"] = time.perf_counter() - t
        if torch is not None:
            torch.cuda.reset_peak_memory_stats(device)
        started = time.perf_counter()
        _, second = run_case(s, case, table)
        record["wall_seconds_second"] = time.perf_counter() - started
        record["timings_second"] = ((second.get("output") or {}).get("meta") or {}).get("timings")
        if torch is not None:
            record["peak_allocated_mb_second"] = torch.cuda.max_memory_allocated(device) / 2**20
        t = time.perf_counter()
        runtime_service = s.runtime
        runtime_service.unload()
        record["unload_seconds"] = time.perf_counter() - t
    finally:
        s.close()
        shutil.rmtree(tmp, ignore_errors=True)
    return record


def cmd_run(args: argparse.Namespace) -> int:
    cases = load_cases(args.cases, set(args.only or []))
    name = args.name or ("cpu" if args.device == "cpu" else f"{args.device}-{args.vram_limit_mb or 'full'}")
    out = WORKSPACE / "memory" / name
    out.mkdir(parents=True, exist_ok=True)
    failures = 0
    for case in cases:
        print(f"[memory {name}] {case['id']} …", flush=True)
        env = dict(os.environ)
        payload = json.dumps({"case": case, "device": args.device, "settings": parse_settings(args.setting or []), "limit": args.vram_limit_mb})
        proc = subprocess.run([sys.executable, str(HERE / "memory.py"), "_child"], input=payload, capture_output=True, text=True, env=env, cwd=HERE)
        engine = case["request"]["model"].split("/", 1)[0]
        target = out / f"{engine}-{case['id']}.json"
        lines = [line for line in proc.stdout.splitlines() if line.startswith("{")]
        if proc.returncode != 0 or not lines:
            failures += 1
            target.write_text(json.dumps({"case": f"{engine}/{case['id']}", "job_state": "exception", "stderr": proc.stderr[-4000:]}, indent=1), encoding="utf-8")
            print(f"    failed: {proc.stderr.strip().splitlines()[-1:] if proc.stderr.strip() else proc.returncode}", flush=True)
            continue
        record = json.loads(lines[-1])
        target.write_text(json.dumps(record, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"    {record.get('job_state')} {record.get('wall_seconds', 0):.1f}s", flush=True)
    print(report(name, args.report))
    return 1 if failures else 0


def _f(value: Any, fmt: str = "{:.0f}") -> str:
    return "—" if value is None else fmt.format(value)


def report(name: str, path: str | None = None) -> str:
    rows = []
    for f in sorted((WORKSPACE / "memory" / name).glob("*.json")):
        r = json.loads(f.read_text(encoding="utf-8"))
        t = r.get("timings") or {}
        weights = sum(x.get("bytes", 0) for x in r.get("resident") or []) / 2**20
        peak = r.get("peak_allocated_mb") if "peak_allocated_mb" in r else (r.get("peak_rss_mb") or 0) - (r.get("rss_before_mb") or 0) if r.get("peak_rss_mb") else None
        rows.append(
            f"| {r['case']} | {r.get('job_state')} | {_f(weights)} | {_f(peak)} | {_f(t.get('load'), '{:.1f}')} | {_f(t.get('total'), '{:.1f}')} | {_f(t.get('rtf'), '{:.2f}')} "
            f"| {_f(r.get('offload_seconds'), '{:.2f}')} | {_f(r.get('reload_seconds'), '{:.2f}')} | {_f(r.get('unload_seconds'), '{:.2f}')} | {_f(r.get('wall_seconds_second'), '{:.1f}')} |"
        )
    head = (
        f"## memory: {name}\n\n| case | state | weights MB | peak MB | load s | request s | RTF | offload s | reload s | unload s | second run s |\n"
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |\n"
    )
    text = head + "\n".join(rows) + "\n\nPeak: torch's peak allocated on a GPU; on the CPU the process's peak resident set above its size before the run.\n"
    if path:
        Path(path).write_text(text, encoding="utf-8")
    return text


def main(argv: list[str]) -> int:
    if argv[:1] == ["_child"]:
        job = json.loads(sys.stdin.read())
        record = child(job["case"], job["device"], job["settings"], job["limit"])
        sys.stdout.write("\n" + json.dumps(record, default=str) + "\n")
        return 0
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="cmd", required=True)
    run = sub.add_parser("run")
    run.add_argument("cases", nargs="+")
    run.add_argument("--device", default="cpu")
    run.add_argument("--only", action="append")
    run.add_argument("--name")
    run.add_argument("--vram-limit-mb", type=int)
    run.add_argument("--setting", action="append")
    run.add_argument("--report")
    rep = sub.add_parser("report")
    rep.add_argument("--name", required=True)
    rep.add_argument("--report")
    args = parser.parse_args(argv)
    if args.cmd == "report":
        print(report(args.name, args.report))
        return 0
    return cmd_run(args)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
