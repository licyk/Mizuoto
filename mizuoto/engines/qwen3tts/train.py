"""Qwen3-TTS speaker fine-tuning, native: upstream's ``finetuning/`` (prepare_data.py, sft_12hz.py) on the native modules.

A Base model learns one speaker and becomes a CustomVoice model whose only preset speaker is the new voice, as v1's
trainer (``kodama_engine_qwen3tts.trainer``) makes it:

- ``prepare`` writes the clips at 24 kHz as 16-bit WAV (upstream's dataset asserts the rate) and adds each clip's
  ``audio_codes`` from the model's own 12 Hz speech tokenizer, encoded eight clips at a time with zero padding as
  ``Qwen3TTSTokenizer.encode`` batches them; one clip is every row's ``ref_audio``.
- ``train`` is ``sft_12hz.py``'s loop under Accelerate as v1 runs it: ``TTSDataset``'s items and ``collate_fn``, a
  shuffled ``DataLoader`` drawing from the process-wide generator, AdamW (weight decay 0.01), the loss divided by the
  accumulation steps, gradients clipped to norm 1 and the optimiser stepped every ``grad_accumulation`` batches and at
  the end of each epoch. The loss is upstream's: the talker's next-code loss plus 0.3 × the code predictor's, both
  transformers' ``ForCausalLMLoss`` (which shifts the already shifted labels once more). bf16 on a GPU (fp16 when asked),
  fp32 on a CPU. At the end it writes the checkpoint folder as upstream does: ``tts_model_type = custom_voice``, the
  speaker at codec id 3000 holding the first batch's speaker embedding, the speaker encoder dropped, upstream's keys.
- ``export`` hands the checkpoint to the host as a model and speaks a preview with its preset speaker.
"""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from typing import Any

from mizuoto.core import codes
from mizuoto.runtime.engine import EngineError
from mizuoto.training import Row, StageContext, Trainer

SAMPLE_RATE = 24000
SPEAKER_ID = 3000
"""The codec embedding row upstream's SFT gives the new speaker."""
ENCODE_BATCH = 8
"""Clips per speech-tokenizer call in ``prepare`` (v1's batch)."""


class Qwen3Trainer(Trainer):
    ENGINE = "qwen3tts"

    @classmethod
    def refuses(cls, method: str, model_dir: Path) -> str | None:
        """Upstream's speaker recipe adds the text embeddings to the codec's without the text projection: it trains
        only Base models whose text embeddings are as wide as the talker (the 1.7B; not the 0.6B: 2048 against 1024)."""
        if method != "speaker":
            return None
        try:
            talker = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))["talker_config"]
            text, width = int(talker["text_hidden_size"]), int(talker["hidden_size"])
        except (OSError, ValueError, KeyError, TypeError):
            return None  # the stage reads it again and says what is wrong
        if text == width:
            return None
        return f"Upstream's speaker recipe trains only models whose text embeddings are as wide as the talker ({text} here against {width}): start from the 1.7B Base"

    @staticmethod
    def _model_dir(ctx: StageContext) -> Path:
        if ctx.base_model.get("variant") != "base":
            raise EngineError("Speaker fine-tuning starts from a Qwen3-TTS Base model", code=codes.INVALID_INPUT)
        return Path(ctx.base_model["dir"])

    @staticmethod
    def _speaker(ctx: StageContext) -> str:
        name = str(ctx.hp("speaker", "") or ctx.name).strip().lower().replace(" ", "_")
        return name or "speaker"

    # -- stages ---------------------------------------------------------------------------------

    def stage_prepare(self, ctx: StageContext) -> dict[str, Any]:
        """24 kHz clips, one reference for all, and each clip's codes from the 12 Hz speech tokenizer."""
        import torch

        from mizuoto.engines.qwen3tts.sft import encode_batch, load_codec
        from mizuoto.nn.dsp.flac import read_audio
        from mizuoto.nn.dsp.resample import resample
        from mizuoto.nn.dsp.wav import read_wav, write_wav

        model_dir = self._model_dir(ctx)
        data = ctx.workdir / "data"
        if data.exists():
            shutil.rmtree(data)
        wavs = data / "wavs"
        wavs.mkdir(parents=True)
        rows = [r for r in ctx.rows if r.text.strip()]
        if not rows:
            raise EngineError("No clip has text", code=codes.INVALID_INPUT)
        reference = _reference(rows)
        items: list[dict[str, Any]] = []
        for row in rows:
            audio, sr = read_audio(row.audio)
            samples = torch.from_numpy(audio)
            if sr != SAMPLE_RATE:
                samples = resample(samples, sr, SAMPLE_RATE)
            target = wavs / f"{row.id}.wav"
            write_wav(target, samples.numpy(), SAMPLE_RATE)
            items.append({"audio": str(target), "text": row.text.strip(), "id": row.id})
        ref_audio = str(wavs / f"{reference.id}.wav")
        ctx.progress(0.0, "Encoding the clips")
        codec = load_codec(model_dir, ctx.device)
        lines = []
        with torch.inference_mode():
            for start in range(0, len(items), ENCODE_BATCH):
                ctx.progress(start / len(items))
                chunk = items[start : start + ENCODE_BATCH]
                encoded = encode_batch(codec, [read_wav(i["audio"])[0] for i in chunk], ctx.device)
                for item, code in zip(chunk, encoded):
                    lines.append({"audio": item["audio"], "text": item["text"], "ref_audio": ref_audio, "audio_codes": code.cpu().tolist()})
        del codec
        with open(data / "train.jsonl", "w", encoding="utf-8") as f:
            for line in lines:
                f.write(json.dumps(line, ensure_ascii=False) + "\n")
        return {"clips": len(lines), "reference": reference.id}

    def stage_train(self, ctx: StageContext) -> dict[str, Any]:
        """``sft_12hz.py``: the talker and its code predictor learn the dataset's speaker."""
        import torch
        from safetensors.torch import save_file
        from torch.utils.data import DataLoader

        from mizuoto.engines.qwen3tts.config import load_config
        from mizuoto.engines.qwen3tts.engine import _mistral_fix
        from mizuoto.engines.qwen3tts.sft import Indices, collate, dataset_item, load_model, step
        from mizuoto.text.tokenize.bpe import load_bpe
        from mizuoto.training.memory import checkpoint_every, checkpoint_units, host_optimizer, preset, wants

        model_dir = self._model_dir(ctx)
        data = ctx.workdir / "data"
        gpu = ctx.device.startswith("cuda")
        precision = ctx.precision
        dtype = torch.bfloat16 if gpu and precision in ("auto", "bf16") else torch.float16 if gpu and precision == "fp16" else torch.float32
        accumulation = int(ctx.hp("grad_accumulation", 4))
        attn = str(ctx.hp("attn_implementation", "sdpa"))
        if attn == "flash_attention_2" and ctx.attention == "auto":
            ctx.use_attention("flash")
        elif attn == "eager":
            ctx.log("attn_implementation=eager: the native trainer computes attention as training.attention says", "warning")
        ctx.progress(None, "Loading the model")
        config = load_config(model_dir)
        if config.talker.text_hidden_size != config.talker.decoder.hidden_size:
            # upstream's loop adds the raw text embeddings to the codec's (no text projection): v1 failed on a shape error
            raise EngineError("Upstream's speaker recipe trains only models whose text embeddings are as wide as the talker", code=codes.CAPABILITY_NOT_SUPPORTED)
        model = load_model(model_dir, config, dtype, ctx.device)
        tokenizer = load_bpe(model_dir, mistral_fix=_mistral_fix(model_dir))
        train_data = [json.loads(line) for line in (data / "train.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
        items = [dataset_item(entry, tokenizer) for entry in train_data]
        # upstream's shuffled DataLoader: the same draws from the process-wide generator (its base seed, then the sampler's
        # seed) give the same order
        loader = DataLoader(Indices(len(items)), batch_size=int(ctx.hp("batch_size", 2)), shuffle=True, collate_fn=lambda ids: collate([items[i] for i in ids], config))
        lr = float(ctx.hp("learning_rate", 2e-5))
        if wants(ctx, "optimizer_device", "cpu"):  # the lowest preset: AdamW's state in RAM
            optimizer = host_optimizer(list(model.parameters()), lambda ps: torch.optim.AdamW(ps, lr=lr, weight_decay=0.01))
            ctx.log("The optimizer's state is kept in RAM and stepped on the CPU")
        else:
            optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
        epochs = int(ctx.hp("epochs", 3))
        batches = len(loader)
        total = epochs * batches
        target_speaker_embedding = None
        every = checkpoint_every(preset(ctx, "gradient_checkpointing"))
        if every:
            talker: Any = model.talker
            n = checkpoint_units(found := [*talker.model.layers, *talker.code_predictor.model.layers], (), every)
            ctx.log(f"Gradient checkpointing: {n} of {len(found)} blocks")
        model.train()
        done = 0
        for epoch in range(epochs):
            pending = 0  # Accelerate's GradientState: steps since the last optimiser step, reset at the end of the loader
            for index, batch in enumerate(loader):
                last = index == batches - 1
                pending = 0 if last else pending + 1
                sync = last or pending % accumulation == 0
                batch = {k: v.to(ctx.device) for k, v in batch.items()}
                loss, talker_loss, sub_loss, speaker_embedding = step(model, batch)
                if target_speaker_embedding is None:
                    target_speaker_embedding = speaker_embedding
                (loss / accumulation).backward()
                if sync:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                done += 1
                ctx.metric(epoch + 1, done, {"loss": float(loss.item()), "talker": float(talker_loss.item()), "sub_talker": float(sub_loss.item())})
                ctx.progress(done / max(1, total), f"Epoch {epoch + 1} of {epochs}")
        del optimizer

        # The checkpoint folder, as upstream writes each epoch's.
        checkpoint = ctx.workdir / "checkpoint"
        if checkpoint.exists():
            shutil.rmtree(checkpoint)
        # Everything but the base weights and Mizuoto's sidecar, at the top only (the speech tokenizer has its own model.safetensors).
        top = str(model_dir)
        shutil.copytree(model_dir, checkpoint, ignore=lambda folder, names: [n for n in names if folder == top and n in ("model.json", "model.safetensors")])
        speaker = self._speaker(ctx)
        config_dict = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
        config_dict["tts_model_type"] = "custom_voice"
        talker_config = config_dict.get("talker_config", {})
        talker_config["spk_id"] = {speaker: SPEAKER_ID}
        talker_config["spk_is_dialect"] = {speaker: False}
        config_dict["talker_config"] = talker_config
        (checkpoint / "config.json").write_text(json.dumps(config_dict, indent=2, ensure_ascii=False), encoding="utf-8")
        state_dict = {k: v.detach().to("cpu") for k, v in model.state_dict().items() if not k.startswith("speaker_encoder")}
        if target_speaker_embedding is not None:
            weight = state_dict["talker.model.codec_embedding.weight"]
            weight[SPEAKER_ID] = target_speaker_embedding[0].detach().to(weight.device).to(weight.dtype)
        save_file(state_dict, str(checkpoint / "model.safetensors"))
        return {"steps": done, "epochs": epochs, "speaker": speaker}

    def stage_export(self, ctx: StageContext) -> dict[str, Any]:
        """The checkpoint as a CustomVoice model whose preset speaker is the new voice, and a preview."""
        from mizuoto.engines.qwen3tts.config import VARIANTS

        checkpoint = ctx.workdir / "checkpoint"
        if not (checkpoint / "model.safetensors").is_file():
            raise EngineError("There is no trained checkpoint; run the training stage", code=codes.INVALID_INPUT)
        config = json.loads((checkpoint / "config.json").read_text(encoding="utf-8"))
        variant = VARIANTS.get(("custom_voice", str(config.get("tts_model_size", ""))), "custom_voice")
        speaker = next(iter(config["talker_config"]["spk_id"]))
        folder = ctx.workdir / "export"
        if folder.exists():
            shutil.rmtree(folder)
        shutil.copytree(checkpoint, folder, copy_function=_link_or_copy)
        text = str(ctx.hp("preview_text", "") or "") or (_reference(ctx.rows).text if ctx.rows else "Hello.")
        ctx.preview(text, None, "Preview", model_dir=folder, variant=variant, preset=speaker, language=ctx.request.get("language"))
        return {"artifacts": [{"kind": "model", "folder": str(folder), "variant": variant, "title": ctx.name, "speakers": [speaker]}], "speaker": speaker}


# -- helpers ------------------------------------------------------------------------------------


def _reference(rows: list[Row]) -> Row:
    fitting = [r for r in rows if r.text.strip() and 3.0 <= r.duration <= 15.0]
    return min(fitting or rows, key=lambda r: abs(r.duration - 8.0))


def _link_or_copy(src: str, dst: str) -> str:
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)
    return dst
