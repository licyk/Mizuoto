"""Training with the memory tools on gives what training without them gives.

``check_checkpointing`` runs a trainer's stages twice in fresh work folders, once as given and once with every block
checkpointed (``gradient_checkpointing=blocks``), and compares every metric and every weight file the stages wrote.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import torch

from mizuoto.core.jobs.jsonl import Emitter
from mizuoto.training.stage import stage_main


class _Recorder(Emitter):
    def __init__(self) -> None:
        super().__init__()
        self.events: list[Any] = []

    def emit(self, event: Any) -> None:
        self.events.append(event)


def _weights(folder: Path) -> dict[str, Any]:
    from safetensors.torch import load_file

    found: dict[str, Any] = {}
    for path in sorted(folder.rglob("*")):
        rel = str(path.relative_to(folder))
        if path.suffix == ".safetensors":
            found[rel] = load_file(str(path))
        elif path.suffix in (".pt", ".pth", ".ckpt"):
            found[rel] = torch.load(str(path), map_location="cpu", weights_only=False)
    return found


def _same(a: Any, b: Any, where: str) -> None:
    if isinstance(a, torch.Tensor):
        assert isinstance(b, torch.Tensor) and a.dtype == b.dtype and torch.equal(a, b), where
    elif isinstance(a, dict):
        assert isinstance(b, dict) and a.keys() == b.keys(), where
        for k in a:
            _same(a[k], b[k], f"{where}/{k}")
    elif isinstance(a, list | tuple):
        assert len(a) == len(b), where
        for i, (x, y) in enumerate(zip(a, b, strict=True)):
            _same(x, y, f"{where}[{i}]")


def check_checkpointing(request: dict[str, Any], stages: list[str], work: Path) -> int:
    """Run ``stages`` with ``request`` (``workdir`` replaced) plainly and checkpointed; identical. Returns how many
    metric points were compared."""
    runs = []
    for name, extra in (("plain", {}), ("checkpointed", {"gradient_checkpointing": "blocks"})):
        folder = work / name
        logs: list[str] = []
        metrics: list[Any] = []
        for stage in stages:
            r = copy.deepcopy(request)
            r.update(stage=stage, workdir=str(folder))
            r["hyperparams"] = {**r.get("hyperparams", {}), **extra}
            rec = _Recorder()
            stage_main(r, rec)
            metrics += [(e.step, e.values, e.lr) for e in rec.events if type(e).__name__ == "MetricEvent"]
            logs += [e.message for e in rec.events if type(e).__name__ == "LogEvent"]
        runs.append((metrics, _weights(folder), logs))
    (plain, plain_weights, _), (checked, checked_weights, logs) = runs
    assert any("Gradient checkpointing" in m and not m.startswith("Gradient checkpointing: 0 ") for m in logs), logs
    assert plain and checked == plain
    assert plain_weights and plain_weights.keys() == checked_weights.keys()
    for k in plain_weights:
        _same(plain_weights[k], checked_weights[k], k)
    return len(plain)


def check_same_training(request: dict[str, Any], stages: list[str], work: Path, extra: dict[str, Any], log: str) -> int:
    """As ``check_checkpointing``, with ``extra`` hyperparameters instead (a tool that keeps results on the CPU)."""
    runs = []
    for name, more in (("plain", {}), ("tool", extra)):
        folder = work / name
        logs: list[str] = []
        metrics: list[Any] = []
        for stage in stages:
            r = copy.deepcopy(request)
            r.update(stage=stage, workdir=str(folder))
            r["hyperparams"] = {**r.get("hyperparams", {}), **more}
            rec = _Recorder()
            stage_main(r, rec)
            metrics += [(e.step, e.values, e.lr) for e in rec.events if type(e).__name__ == "MetricEvent"]
            logs += [e.message for e in rec.events if type(e).__name__ == "LogEvent"]
        runs.append((metrics, _weights(folder), logs))
    (plain, plain_weights, _), (other, other_weights, logs) = runs
    assert any(log in m for m in logs), logs
    assert plain and other == plain
    assert plain_weights.keys() == other_weights.keys()
    for k in plain_weights:
        _same(plain_weights[k], other_weights[k], k)
    return len(plain)
