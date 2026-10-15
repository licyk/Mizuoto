"""The dummy engine's native trainer: the whole training path with no model (as v1's dummy trainer).

``speaker`` makes a voice's weights (a pitch, which the dummy engine speaks with); ``full`` makes a model folder.
Losses fall as a real run's would, and ``step_ms`` slows a step down for tests.
"""

from __future__ import annotations

import json
import time
from typing import Any

from mizuoto.runtime.context import Cancelled
from mizuoto.training import StageContext, Trainer


class DummyTrainer(Trainer):
    ENGINE = "dummy"

    def stage_prepare(self, ctx: StageContext) -> dict[str, Any]:
        seconds = sum(r.duration for r in ctx.rows)
        features = {"rows": len(ctx.rows), "seconds": round(seconds, 3), "chars": sum(len(r.text) for r in ctx.rows)}
        (ctx.workdir / "features.json").write_text(json.dumps(features), encoding="utf-8")
        ctx.progress(1.0, f"{len(ctx.rows)} clips")
        return features

    def stage_train(self, ctx: StageContext) -> dict[str, Any]:
        features = json.loads((ctx.workdir / "features.json").read_text(encoding="utf-8"))
        epochs = int(ctx.hp("epochs", 3))
        steps = max(1, features["rows"])
        delay = int(ctx.hp("step_ms", 0)) / 1000
        step = 0
        loss = 1.0
        for epoch in range(1, epochs + 1):
            for _ in range(steps):
                step += 1
                loss = 1.0 / (1 + 0.5 * step)
                ctx.metric(epoch, step, {"loss": loss}, lr=1e-3)
                if delay:
                    time.sleep(delay)
            ctx.progress(epoch / epochs, f"epoch {epoch} of {epochs}")
        if ctx.hp("fail"):
            raise Cancelled() if ctx.hp("fail") == "cancel" else RuntimeError("the dummy training failed on purpose")
        pitch = 120 + (features["seconds"] * 10) % 200
        checkpoint = ctx.workdir / "checkpoint.json"
        checkpoint.write_text(json.dumps({"pitch": pitch, "epochs": epochs}), encoding="utf-8")
        ctx.output(checkpoint, f"epoch {epochs}", "checkpoint")
        return {"steps": step, "loss": loss, "pitch": pitch}

    def stage_export(self, ctx: StageContext) -> dict[str, Any]:
        pitch = json.loads((ctx.workdir / "checkpoint.json").read_text(encoding="utf-8"))["pitch"]
        out = ctx.workdir / "export"
        out.mkdir(exist_ok=True)
        if ctx.request.get("method") == "full":
            folder = out / "model"
            folder.mkdir(exist_ok=True)
            (folder / "weights.json").write_text(json.dumps({"pitch": pitch}), encoding="utf-8")
            return {"artifacts": [{"kind": "model", "engine": "dummy", "variant": "default", "folder": str(folder), "title": ctx.name}]}
        weights = out / "voice.bin"
        weights.write_text(str(pitch), encoding="utf-8")
        return {"artifacts": [{"kind": "voice_weights", "engine": "dummy", "files": {"model": str(weights)}}]}
