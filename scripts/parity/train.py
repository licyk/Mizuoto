"""Train a model through a whole Mizuoto stack, reference or native, and compare runs.

    <python> scripts/parity/train.py run <engine> --impl reference|native|upstream --model ID [--method M] [--hp k=v]... [--name N]
    <python> scripts/parity/train.py compare <engine> <name> [--a reference] [--b native] [--weights]

As ``stack.py`` for synthesis: ``<python>`` decides which Mizuoto runs (the reference host drives v1's trainers in the
engine environments; this checkout's interpreter runs the native trainers, as children of its own process). The run
happens in a temporary data folder that reads the real models (and the reference's environments), with the
workspace's ``train-data/datasets/narrator`` (11 clips, the dataset M6's trainers were checked on) copied in. It keeps
``<workspace>/train-fixtures/<engine>/<name>/<impl>/``: ``experiment.json``, ``metrics.jsonl``, the previews, the job's
log and ``artifacts.json`` (the exported voice weights or model, copied beside them: ``artifacts/``), so the exports can
be loaded by the other implementation.

``--impl upstream`` (VoxCPM: no 0.1 trainer) runs upstream's own training script on the same dataset, outside Mizuoto
(``voxcpm_upstream.run_training``), with the native trainer's recipe for the hyperparameters.

``compare`` prints the two runs' metrics step by step with the largest relative difference per metric; ``--weights``
also compares the saved weights (``artifacts/model/*.safetensors`` present in both), tensor by tensor.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any

from stack import WORKSPACE, build, package

DATASET = WORKSPACE / "train-data" / "datasets" / "narrator"


def _value(raw: str) -> Any:
    try:
        return json.loads(raw)
    except ValueError:
        return raw


def cmd_run(args: argparse.Namespace) -> int:
    ExperimentInput = package("core.training.models").ExperimentInput

    name = args.name or args.model.split("/", 1)[1]
    out = WORKSPACE / "train-fixtures" / args.engine / name / args.impl
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    if args.impl == "upstream":
        return run_upstream(args, out)
    tmp = Path(tempfile.mkdtemp(prefix=f"train-{args.impl}-"))
    s = build(args.device, tmp, args.impl)
    try:
        s.settings.update({"training": {"previews": not args.no_previews}})
        shutil.copytree(DATASET, s.datasets.root / DATASET.name)
        dataset = json.loads((DATASET / "dataset.json").read_text(encoding="utf-8"))
        hyperparams = {k: _value(v) for k, v in (h.split("=", 1) for h in args.hp or [])}
        exp = s.training.create(ExperimentInput(name=name, base_model=args.model, dataset=dataset["id"], method=args.method, hyperparams=hyperparams, language=args.language))
        print(f"[{args.impl}] training {exp.key} ({exp.method}) …", flush=True)
        job = s.training.run(exp.key, wait=True)
        exp = s.training.get(exp.key)
        folder = s.training.folder(exp)
        for f in ("experiment.json", "metrics.jsonl"):
            if (folder / f).is_file():
                shutil.copyfile(folder / f, out / f)
        preview = folder / "work" / "preview"
        if preview.is_dir():
            shutil.copytree(preview, out / "preview")
        record = {"impl": args.impl, "job_state": job.state, "error": job.error.model_dump() if job.error else None, "log": s.jobs.read_log(job.id).lines[-200:]}
        artifacts = []
        for a in exp.artifacts:
            entry = a.model_dump(mode="json")
            if a.kind == "model":
                info = s.models.get(a.id)
                if info.path:
                    shutil.copytree(info.path, out / "artifacts" / "model", dirs_exist_ok=True)
                    entry["copied"] = "artifacts/model"
                    # Exported models are registered in the models folder, which is the real one: take it back out.
                    shutil.rmtree(info.path, ignore_errors=True)
            elif a.kind == "voice" and s.training.voices is not None:
                voice = s.training.voices.get(a.id)
                shutil.copytree(s.training.voices.root / voice.slug, out / "artifacts" / "voice", ignore=shutil.ignore_patterns("cache"), dirs_exist_ok=True)
                entry["copied"] = "artifacts/voice"
            artifacts.append(entry)
        (out / "artifacts.json").write_text(json.dumps(artifacts, ensure_ascii=False, indent=1), encoding="utf-8")
        (out / "run.json").write_text(json.dumps(record, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
        print(f"    {job.state}" + (f": {job.error.message}" if job.error else "") + f" → {out}", flush=True)
        return 0 if job.state == "completed" else 1
    finally:
        s.close()
        if args.keep:
            print(f"    data folder kept: {tmp}")
        else:
            shutil.rmtree(tmp, ignore_errors=True)


def run_upstream(args: argparse.Namespace, out: Path) -> int:
    from stack import real_data_dir

    hyperparams = {k: _value(v) for k, v in (h.split("=", 1) for h in args.hp or [])}
    engine, model = args.model.split("/", 1)
    if engine != "voxcpm":
        raise SystemExit(f"--impl upstream trains VoxCPM only, not {engine}")
    from voxcpm_upstream import run_training

    model_dir = real_data_dir() / "models" / engine / model
    variant = json.loads((model_dir / "model.json").read_text(encoding="utf-8")).get("variant", "v2")
    print(f"[upstream] training {args.model} ({args.method or 'full'}) …", flush=True)
    code = run_training(model_dir, DATASET, out, args.method or "full", hyperparams, variant, args.device)
    print(f"    {'completed' if code == 0 else 'failed'} → {out}", flush=True)
    return code


def compare_weights(root: Path, a: str, b: str) -> None:
    import torch
    from safetensors.torch import load_file

    for path in sorted((root / a / "artifacts" / "model").glob("*.safetensors")):
        other = root / b / "artifacts" / "model" / path.name
        if not other.is_file():
            continue
        x, y = load_file(str(path)), load_file(str(other))
        keys = sorted(set(x) | set(y))
        same = [k for k in keys if k in x and k in y and x[k].dtype == y[k].dtype and x[k].shape == y[k].shape and torch.equal(x[k], y[k])]
        worst = max(
            ((k, float((x[k].float() - y[k].float()).abs().max())) for k in keys if k in x and k in y and x[k].shape == y[k].shape), key=lambda kv: kv[1], default=("", 0.0)
        )
        print(f"  {path.name}: {len(same)} of {len(keys)} tensors identical; largest difference {worst[1]:.3g} ({worst[0]})")


def _metrics(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()] if path.is_file() else []


def cmd_compare(args: argparse.Namespace) -> int:
    root = WORKSPACE / "train-fixtures" / args.engine / args.name
    a, b = _metrics(root / args.a / "metrics.jsonl"), _metrics(root / args.b / "metrics.jsonl")
    print(f"{args.engine}/{args.name}: {len(a)} {args.a} points, {len(b)} {args.b} points")
    worst: dict[str, float] = {}
    # Points pair by (stage, step): a run may log only some steps (v1's Fish Speech logs its progress bar's last).
    by_step = {(p["stage"], p["step"]): p for p in b}
    pairs = [(pa, by_step[(pa["stage"], pa["step"])]) for pa in a if (pa["stage"], pa["step"]) in by_step] or list(zip(a, b, strict=False))
    for pa, pb in pairs:
        keys = sorted(set(pa["values"]) & set(pb["values"]))
        cells = []
        for k in keys:
            va, vb = pa["values"][k], pb["values"][k]
            rel = abs(va - vb) / max(abs(va), 1e-12)
            worst[f"{pa['stage']}:{k}"] = max(worst.get(f"{pa['stage']}:{k}", 0.0), rel)
            cells.append(f"{k} {va:.6g}/{vb:.6g}")
        print(
            f"  {pa['stage']} e{pa['epoch']} s{pa['step']}"
            + ("" if (pa["stage"], pa["step"]) == (pb["stage"], pb["step"]) else f" ≠ {pb['stage']} s{pb['step']}")
            + "  "
            + "  ".join(cells)
        )
    for k, v in sorted(worst.items()):
        print(f"  max relative difference {k}: {v:.3g}")
    lr = [(pa["step"], pa.get("lr"), pb.get("lr")) for pa, pb in pairs if pa.get("lr") != pb.get("lr")]
    if lr:
        print(f"  learning rates differ at {len(lr)} steps, first {lr[0]}")
    if args.weights:
        compare_weights(root, args.a, args.b)
    return 0


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="cmd", required=True)
    run = sub.add_parser("run")
    run.add_argument("engine")
    run.add_argument("--impl", required=True, choices=["reference", "native", "upstream"])
    run.add_argument("--model", required=True)
    run.add_argument("--method")
    run.add_argument("--hp", action="append", help="A hyperparameter, name=value (JSON values parsed)")
    run.add_argument("--language")
    run.add_argument("--name")
    run.add_argument("--device", default="cpu")
    run.add_argument("--no-previews", action="store_true")
    run.add_argument("--keep", action="store_true", help="Keep the temporary data folder")
    compare = sub.add_parser("compare")
    compare.add_argument("engine")
    compare.add_argument("name")
    compare.add_argument("--a", default="reference")
    compare.add_argument("--b", default="native")
    compare.add_argument("--weights", action="store_true", help="Also compare the saved weights tensor by tensor")
    args = parser.parse_args(argv)
    return cmd_run(args) if args.cmd == "run" else cmd_compare(args)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
