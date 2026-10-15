"""Training under ``training.attention``: the stage binds its policy for every thread
of its process, autograd's included, and reports what it used."""

from __future__ import annotations

import io
import json
import threading

import pytest

torch = pytest.importorskip("torch")

from mizuoto.runtime import accel  # noqa: E402
from mizuoto.runtime.accel import dispatch  # noqa: E402
from mizuoto.training import Trainer  # noqa: E402

SITE = accel.Site("test.training")


class AttendingTrainer(Trainer):
    ENGINE = "dummy"

    def stage_train(self, ctx):
        q = torch.randn(1, 2, 5, 8, requires_grad=True)
        out = accel.attention(q, q, q, site=SITE, causal=True)
        out.sum().backward()
        other = []
        thread = threading.Thread(target=lambda: other.append(accel.attention(q.detach(), q.detach(), q.detach(), site=SITE)))
        thread.start()
        thread.join()
        return {"grad": bool(q.grad is not None)}


def run_stage(tmp_path, attention):
    from mizuoto.core.jobs.jsonl import Emitter
    from mizuoto.training.stage import stage_main

    stream = io.StringIO()
    request = {"trainer": f"{__name__}:AttendingTrainer", "stage": "train", "workdir": str(tmp_path), "attention": attention}
    try:
        stage_main(request, Emitter(stream))
    finally:
        dispatch.PROCESS_POLICY[0] = None
    return [json.loads(line.split("{", 1)[0] and "{" + line.split("{", 1)[1]) for line in stream.getvalue().splitlines() if "{" in line]


def test_a_stage_binds_its_attention_for_every_thread(tmp_path):
    events = run_stage(tmp_path, "math")
    logs = [e.get("message", "") for e in events]
    report = next(json.loads(m.removeprefix("attention: ")) for m in logs if m.startswith("attention: {"))
    assert report == {"policy": "math", "used": {"math": 2}, "fallbacks": {}}  # the main thread and the other one
    assert any(e.get("data") == {"grad": True} for e in events)


def test_auto_binds_nothing(tmp_path):
    events = run_stage(tmp_path, "auto")
    assert not any(e.get("message", "").startswith("attention:") for e in events)
