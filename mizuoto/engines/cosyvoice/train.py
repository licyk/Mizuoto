"""CosyVoice fine-tuning, native: v1's trainer (upstream's ``examples/libritts/cosyvoice{,2,3}``
recipe) on the native modules, with no onnxruntime, pyarrow, pyworld, transformers, deepspeed or torchrun.

- ``features`` is the recipe's stages 0–3: each clip with text as 16-bit WAV at its own rate (``wavs/``), its speech
  tokens from the model's speech tokenizer and its CAM++ embedding, both native instead of onnxruntime; one JSON line
  per utterance (``utts.jsonl``) instead of the Kaldi files and parquet shards. CosyVoice 3's utterances carry the
  recipe's instruction (``You are a helpful assistant.<|endofprompt|>``).
- ``llm`` and ``flow`` are stage 5, ``cosyvoice/bin/train.py`` on one device from the model's ``llm.pt``/``flow.pt``
  with the recipe's config and Mizuoto's epochs, learning rate and accumulation (``sft.py``); every batch's logged loss
  (divided by the accumulation steps, as upstream logs it) and accuracy are the metrics. Each epoch ends with the
  validation pass and ``exp/<part>/epoch_<N>_whole.pt``, upstream's checkpoint; the newest
  ``training.keep_checkpoints`` are kept.
- ``export`` copies the model folder with the newest epoch's weights in place of the trained parts (without the step
  counters), and speaks a preview cloning a dataset clip.

The stage seeds the process-wide generators (``training.seed``); upstream seeds them in its config (1986, and 0 again
while building a CosyVoice 2/3 flow) and draws its random initialisation before training. Nothing draws here before
training starts, so a run equals upstream's whose generators are seeded with the stage's seed once its model is built.
"""

from __future__ import annotations

import json
import os
import re
import shutil
from pathlib import Path
from typing import Any

from mizuoto.core import codes
from mizuoto.runtime.engine import DEVICE_KINDS, EngineError
from mizuoto.training import Row, StageContext, Trainer

VERSIONS = {"v1_base": 1, "v1_sft": 1, "v1_instruct": 1, "v2": 2, "v3": 3}
V3_INSTRUCT = "You are a helpful assistant.<|endofprompt|>"
"""What CosyVoice 3's recipe prefixes every utterance with (``prepare_data.py --instruct``)."""


class CosyVoiceTrainer(Trainer):
    ENGINE = "cosyvoice"
    DEVICES = DEVICE_KINDS  # upstream trains under torchrun on CUDA only; nothing here needs CUDA

    @staticmethod
    def _version(ctx: StageContext) -> int:
        variant = str(ctx.base_model.get("variant", ""))
        if variant not in VERSIONS:
            raise EngineError(f"Unknown CosyVoice variant {variant}", code=codes.INVALID_INPUT)
        return VERSIONS[variant]

    # -- stages ---------------------------------------------------------------------------------

    def stage_features(self, ctx: StageContext) -> dict[str, Any]:
        """The clips, speaker embeddings and speech tokens (the recipe's stages 0–3)."""
        import torch

        from mizuoto.engines.cosyvoice.sft import load_feature_models, utterance_features
        from mizuoto.nn.dsp.flac import read_audio
        from mizuoto.nn.dsp.wav import write_wav

        version = self._version(ctx)
        model_dir = Path(ctx.base_model["dir"])
        data = ctx.workdir / "data"
        if data.exists():
            shutil.rmtree(data)
        wavs = data / "wavs"
        wavs.mkdir(parents=True)
        rows = [r for r in ctx.rows if r.text.strip()]
        if not rows:
            raise EngineError("No clip has text", code=codes.INVALID_INPUT)
        ctx.progress(0.0, "Loading the speech tokenizer and CAM++")
        tokenizer, campplus = load_feature_models(model_dir, version, ctx.device)
        lines = []
        long = 0
        with torch.inference_mode():
            for i, row in enumerate(rows):
                samples, rate = read_audio(row.audio)
                wav = wavs / f"{row.id}.wav"
                write_wav(wav, samples, rate)
                tokens, embedding = utterance_features(wav, tokenizer, campplus, ctx.device)
                long += not tokens
                item = {
                    "utt": row.id,
                    "wav": str(wav),
                    "text": " ".join(row.text.split()),
                    "spk": re.sub(r"\s+", "_", row.speaker.strip()) or ctx.name,
                    "speech_token": tokens,
                    "utt_embedding": embedding,
                }
                if version == 3:
                    item["instruct"] = V3_INSTRUCT
                lines.append(json.dumps(item, ensure_ascii=False))
                ctx.progress((i + 1) / len(rows), "Speaker embeddings and speech tokens")
        (data / "utts.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")
        if long:
            ctx.log(f"{long} clip(s) longer than 30 s have no speech tokens and are left out of training", "warning")
        return {"clips": len(rows), "shards": -(-len(rows) // 1000)}

    def stage_llm(self, ctx: StageContext) -> dict[str, Any]:
        return self._train(ctx, "llm")

    def stage_flow(self, ctx: StageContext) -> dict[str, Any]:
        return self._train(ctx, "flow")

    def _train(self, ctx: StageContext, part: str) -> dict[str, Any]:
        """``cosyvoice/bin/train.py`` for one part, from the model's own weights (the recipe's stage 5)."""
        from mizuoto.engines.cosyvoice.config import read_config
        from mizuoto.engines.cosyvoice.sft import RECIPES, TrainConfig, build, prepare, read_items, sliding_window, train
        from mizuoto.engines.cosyvoice.tokenizer import text_tokenizer
        from mizuoto.training.memory import blocks, checkpoint_every, checkpoint_units, preset

        version = self._version(ctx)
        model_dir = Path(ctx.base_model["dir"])
        listing = ctx.workdir / "data" / "utts.jsonl"
        if not listing.is_file():
            raise EngineError("Extract the features first", code=codes.INVALID_INPUT)
        epochs = int(ctx.hp("epochs", 10))
        workers = int(ctx.hp("workers", 2))
        cuda = ctx.device.startswith("cuda")
        out = ctx.workdir / "exp" / part
        if out.exists():
            shutil.rmtree(out)
        ctx.progress(None, "Reading the data")
        recipe = RECIPES[version]
        # Workers compute the features on one thread; without them the loop's own process does, on all of its.
        utts = prepare(read_items(listing), recipe, text_tokenizer(model_dir, version), threads=1 if workers > 0 else None)
        kept = sum(u is not None for u in utts)
        if not kept:
            raise EngineError("Every clip was filtered out (too short, too long or too much text for its length)", code=codes.INVALID_INPUT)
        if kept < len(utts):
            ctx.log(f"{len(utts) - kept} clip(s) left out by the recipe's filter", "warning")
        ctx.progress(None, f"Loading the {'language model' if part == 'llm' else 'flow'}")
        model = build(model_dir, version, part, read_config(model_dir))
        every = checkpoint_every(preset(ctx, "gradient_checkpointing"))
        if every:  # the global generators, which every draw here comes from, are replayed in the backward
            n = checkpoint_units(found := blocks(model), (), every)
            ctx.log(f"Gradient checkpointing: {n} of {len(found)} blocks")
        cfg = TrainConfig(
            part=part,
            version=version,
            out=out,
            epochs=epochs,
            lr=float(ctx.hp("learning_rate", 1e-5)),
            accum_grad=int(ctx.hp("grad_accumulation", 2)),
            workers=workers,
            device=ctx.device,
            amp=cuda and ctx.precision != "fp32",
            keep=int(ctx.request.get("keep_checkpoints") or 3),
            window=sliding_window(model_dir) if version > 1 and part == "llm" else None,
        )

        def report(epoch: int, batch: int, values: dict[str, float], lr: float) -> None:
            ctx.metric(epoch + 1, batch, values, lr)
            ctx.progress(min(1.0, (epoch + 0.5) / epochs), f"Epoch {epoch + 1} of {epochs}")

        ctx.progress(0.0, f"Epoch 1 of {epochs}")
        result = train(model, utts, cfg, report, ctx.log)
        return {"epochs": epochs, "checkpoint": result["checkpoint"]}

    def stage_export(self, ctx: StageContext) -> dict[str, Any]:
        """The model folder with the trained parts' newest weights (without the step counters), and a preview."""
        import torch

        version = self._version(ctx)
        model_dir = Path(ctx.base_model["dir"])
        variant = str(ctx.base_model.get("variant", ""))
        folder = ctx.workdir / "export"
        if folder.exists():
            shutil.rmtree(folder)
        top = str(model_dir)
        shutil.copytree(model_dir, folder, ignore=lambda d, names: ["model.json"] if d == top and "model.json" in names else [], copy_function=_link_or_copy)
        trained = []
        for part in ("llm", "flow"):
            newest = _newest(ctx.workdir / "exp" / part)
            if newest is None:
                continue
            state = torch.load(newest, map_location="cpu", weights_only=True)
            state = {k: v for k, v in state.items() if k not in ("epoch", "step")}
            target = folder / f"{part}.pt"
            target.unlink(missing_ok=True)  # a link to the base model's file
            torch.save(state, target)
            del state
            trained.append(part)
        if not trained:
            raise EngineError("Nothing was trained; run the training stages", code=codes.INVALID_INPUT)
        reference = _reference(ctx.rows)
        if reference is not None:
            text = str(ctx.hp("preview_text", "") or "") or reference.text
            ctx.preview(text, reference, "Preview", model_dir=folder, variant=variant)
        return {"artifacts": [{"kind": "model", "folder": str(folder), "variant": variant, "title": ctx.name}], "trained": trained, "version": version}


def _newest(folder: Path) -> Path | None:
    found = sorted(folder.glob("epoch_*_whole.pt"), key=lambda p: int(p.name.split("_")[1])) if folder.is_dir() else []
    return found[-1] if found else None


def _reference(rows: list[Row]) -> Row | None:
    fitting = [r for r in rows if r.text.strip() and 3.0 <= r.duration <= 15.0]
    return min(fitting, key=lambda r: abs(r.duration - 6.0)) if fitting else None


def _link_or_copy(src: str, dst: str) -> str:
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)
    return dst
