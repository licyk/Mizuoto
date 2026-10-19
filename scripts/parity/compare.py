"""Compare two runs of the parity cases (tier T3 and the behaviour checks).

    <python> scripts/parity/compare.py <engine>... [--a reference] [--b native] [--report out.md]

For every case folder under ``<workspace>/fixtures/<engine>/`` holding both ``<a>.json`` and
``<b>.json``, compares:

- **Audio:** sample rate, length, and whether the samples are identical; otherwise the largest
  absolute difference and the SNR of b against a.
- **Behaviour:** job state and error code, voice mode, segments, applied parameters, warnings,
  output sample rate and, for cases with ``stream``, the streamed chunks' lengths.
- **Transcripts** (``kind: transcribe`` cases, the ASR engine): the language, the text and every segment with its
  timestamps, exactly; such a case passes when they are all equal.

Prints a Markdown table, and writes it with ``--report``. Exits non-zero when any case differs beyond
``--tolerance`` (default 1e-4, the T3 bound), or in behaviour.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

from stack import WORKSPACE


def read_wav(path: Path) -> tuple[Any, int]:
    from mizuoto.core.audio.io import decode

    samples, rate = decode(path)
    return samples, rate


def behaviour(rec: dict[str, Any]) -> dict[str, Any]:
    out = rec.get("output") or {}
    plan = rec.get("plan") or {}
    meta = out.get("meta") or {}
    return {
        "state": rec.get("job_state"),
        "error": (rec.get("error") or {}).get("code"),
        "voice_mode": plan.get("voice_mode") or meta.get("voice_mode"),
        "segments": plan.get("segments"),
        "params": out.get("params"),
        "warnings": sorted(set(meta.get("warnings") or [])),
        "sample_rate": out.get("sample_rate"),
        "stream_chunks": rec.get("stream_chunks"),
        "transcript": {k: v for k, v in (rec.get("transcript") or {}).items() if k != "sha256"} or None,
    }


def compare_case(folder: Path, a: str, b: str, tolerance: float) -> dict[str, Any] | None:
    ja, jb = folder / f"{a}.json", folder / f"{b}.json"
    if not (ja.exists() and jb.exists()):
        return None
    ra, rb = json.loads(ja.read_text(encoding="utf-8")), json.loads(jb.read_text(encoding="utf-8"))
    ba, bb = behaviour(ra), behaviour(rb)
    diffs = [k for k in ba if ba[k] != bb[k]]
    row: dict[str, Any] = {"case": folder.name, "behaviour_diffs": diffs, "a_wall": ra.get("wall_seconds"), "b_wall": rb.get("wall_seconds")}
    wa, wb = folder / f"{a}.wav", folder / f"{b}.wav"
    if wa.exists() and wb.exists():
        import numpy as np

        xa, sa = read_wav(wa)
        xb, sb = read_wav(wb)
        xa, xb = np.asarray(xa, dtype=np.float64).reshape(-1), np.asarray(xb, dtype=np.float64).reshape(-1)
        row.update(rates=(sa, sb), lengths=(len(xa), len(xb)))
        if sa == sb and len(xa) == len(xb):
            diff = np.abs(xa - xb)
            row["identical"] = bool(diff.max(initial=0.0) == 0.0)
            row["max_abs"] = float(diff.max(initial=0.0))
            noise = float(np.sum((xa - xb) ** 2))
            row["snr_db"] = math.inf if noise == 0 else 10 * math.log10(float(np.sum(xa**2)) / noise)
        else:
            row["identical"] = False
            row["max_abs"] = None
    max_abs = row.get("max_abs")
    if ba["transcript"] is not None:
        row["identical"] = not diffs
    row["pass"] = not diffs and ((max_abs is not None and max_abs <= tolerance) or (ba["state"] != "completed" and ba == bb) or ba["transcript"] is not None)
    return row


def table(rows: list[dict[str, Any]], a: str, b: str) -> str:
    lines = [f"| case | identical | max abs | SNR dB | length {a} / {b} | behaviour differs | wall {a} / {b} (s) | pass |", "| --- | --- | --- | --- | --- | --- | --- | --- |"]
    for r in rows:
        lengths = r.get("lengths")
        snr = r.get("snr_db")
        lines.append(
            "| {case} | {ident} | {mx} | {snr} | {ln} | {bd} | {wa} / {wb} | {ok} |".format(
                case=r["case"],
                ident="yes" if r.get("identical") else "no",
                mx="—" if r.get("max_abs") is None else f"{r['max_abs']:.2e}",
                snr="—" if snr is None else ("∞" if snr == math.inf else f"{snr:.1f}"),
                ln="—" if not lengths else f"{lengths[0]} / {lengths[1]}",
                bd=", ".join(r["behaviour_diffs"]) or "—",
                wa=f"{r['a_wall']:.1f}" if r.get("a_wall") else "—",
                wb=f"{r['b_wall']:.1f}" if r.get("b_wall") else "—",
                ok="✓" if r["pass"] else "✗",
            )
        )
    return "\n".join(lines)


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("engines", nargs="+")
    parser.add_argument("--a", default="reference")
    parser.add_argument("--b", default="native")
    parser.add_argument("--tolerance", type=float, default=1e-4)
    parser.add_argument("--report")
    args = parser.parse_args(argv)
    out, failed = [], 0
    for engine in args.engines:
        rows = []
        for folder in sorted((WORKSPACE / "fixtures" / engine).glob("*/")):
            row = compare_case(folder, args.a, args.b, args.tolerance)
            if row is not None:
                rows.append(row)
        failed += sum(not r["pass"] for r in rows)
        out.append(f"## {engine}: {args.a} vs {args.b}\n\n{table(rows, args.a, args.b) if rows else '(no case has both runs)'}\n")
    text = "\n".join(out)
    print(text)
    if args.report:
        Path(args.report).write_text(text, encoding="utf-8")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
