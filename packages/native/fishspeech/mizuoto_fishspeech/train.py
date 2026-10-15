"""Fish Speech LoRA fine-tuning, native: v1's trainer (``kodama_engine_fishspeech.trainer``,
upstream's ``docs/en/finetune.md``) on the native modules, without Lightning, Hydra, loralib, transformers or protobuf.

- ``tokens`` is ``tools/vqgan/extract_vq.py`` and ``tools/llama/build_dataset.py`` as v1 runs them: each clip resampled
  to 44.1 kHz and encoded by the model's own codec in float32, two clips at a time on a CPU (eight on a GPU) padded to the
  longer, its codes cut to its own frames; the clips grouped by speaker in path order, their texts cleaned
  (``groups.json`` and a ``.npy`` per clip instead of the protobuf shard).
- ``train`` is ``fish_speech/train.py --config-name text2semantic_finetune +lora@model.model.lora_config=…`` as v1 runs
  it (``finetune.py`` has the maths): upstream's LoRA on the base model, the dataset's samples, AdamW (betas 0.9/0.95,
  eps 1e-5, no weight decay) with the constant schedule after 10 warm-up steps (the first step's rate is 0), gradients
  clipped to norm 1, one batch per step; float32 on a CPU, bf16-true on a GPU (16-mixed for fp16). It saves the LoRA
  weights (``lora/lora.safetensors``).
- ``export`` merges them as ``tools/llama/merge_lora.py`` does and saves what upstream's ``save_pretrained`` writes
  (``config.json`` in the flat ``dual_ar`` form, ``model.pth`` in bfloat16) with the tokenizer, the codec and the chat
  template: a model v1's inference and the native engine both load. Then a preview, cloning a dataset clip.

Seeded alike, a run draws what upstream's draws, in its order (``finetune.py``); upstream's own runs are unseeded (its
config has no ``seed``), so they repeat only their first two steps, which no random number reaches (``B`` starts at
zero and the first step's learning rate is 0). On a GPU, bf16-true differs from Lightning's in one respect: Lightning
also makes bfloat16 the default dtype during each step, which only the top-5 accuracy's division sees.
"""

from __future__ import annotations

import json
import os
import re
import shutil
from pathlib import Path
from typing import Any

from mizuoto.core import codes
from mizuoto.runtime.engine import EngineError
from mizuoto.training import Row, StageContext, Trainer

SAMPLE_RATE = 44100
DEPLOY = ("codec.pth", "chat_template.jinja")
"""What a loadable S2 folder needs besides the merged weights and the tokenizer."""
TOKENIZER_FILES = ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "vocab.json", "merges.txt", "added_tokens.json")
"""The tokenizer as the base model ships it (upstream saves it again with transformers' ``save_pretrained``)."""
PAD_TOKEN = "<|end_of_text|>"
"""The collator's padding token; S2's vocabulary lacks it, so upstream pads with 0 (``F.pad(value=None)``)."""
OPTIMIZER = {"weight_decay": 0.0, "betas": (0.9, 0.95), "eps": 1e-5}
"""``text2semantic_finetune.yaml``'s AdamW."""
CLIP_NORM = 1.0


class FishSpeechTrainer(Trainer):
    ENGINE = "fishspeech"

    @staticmethod
    def _lora(ctx: StageContext) -> Any:
        from mizuoto_fishspeech.finetune import LORA_CONFIGS

        name = str(ctx.hp("lora", "r_8_alpha_16"))
        if name not in LORA_CONFIGS:
            raise EngineError(f"Unknown LoRA config {name}", {"choices": list(LORA_CONFIGS)}, code=codes.INVALID_INPUT)
        return LORA_CONFIGS[name]

    @staticmethod
    def _gpu(ctx: StageContext) -> bool:
        return ctx.device.startswith("cuda")

    # -- stages ---------------------------------------------------------------------------------

    def stage_tokens(self, ctx: StageContext) -> dict[str, Any]:
        """Each clip's codes from the model's codec, grouped by speaker as ``build_dataset`` packs them."""
        import numpy as np
        import torch

        from mizuoto.nn.dsp.flac import read_audio
        from mizuoto.nn.dsp.resample import resample
        from mizuoto.runtime import ops
        from mizuoto.runtime.loading import load_into, load_state_dict, unwrap
        from mizuoto_fishspeech import engine
        from mizuoto_fishspeech.codec import IGNORED, FishCodec, codec_state
        from mizuoto_fishspeech.finetune import pack_text

        model_dir = Path(ctx.base_model["dir"])
        folder = ctx.workdir / "tokens"
        if folder.exists():
            shutil.rmtree(folder)
        folder.mkdir(parents=True)
        rows = [r for r in ctx.rows if r.text.strip()]
        if not rows:
            raise EngineError("No clip has a text", code=codes.INVALID_INPUT)
        gpu = self._gpu(ctx)
        ctx.progress(0.0, "Loading the codec")
        codec = FishCodec(engine.codec_config(model_dir), ops.default)
        load_into(codec, codec_state(unwrap(load_state_dict(model_dir / "codec.pth"), "state_dict")), ignore=IGNORED, what="Fish Speech codec")
        codec = codec.float().to(ctx.device).eval()
        batch = 8 if gpu else 2  # v1's batches: the padding is part of what the codec sees
        seconds = 0.0
        clips: list[tuple[str, Row]] = []
        with torch.inference_mode():
            for start in range(0, len(rows), batch):
                chunk = rows[start : start + batch]
                wavs = []
                for r in chunk:
                    audio, rate = read_audio(r.audio)
                    wav = resample(torch.from_numpy(audio)[None].to(ctx.device), rate, SAMPLE_RATE)[0]
                    seconds += len(wav) / SAMPLE_RATE
                    wavs.append(wav)
                longest = max(len(w) for w in wavs)
                audios = torch.stack([torch.nn.functional.pad(w, (0, longest - len(w))) for w in wavs])[:, None]
                lengths = torch.tensor([len(w) for w in wavs], device=ctx.device, dtype=torch.long)
                indices, frames = codec.encode(audios, lengths)
                for r, feature, n in zip(chunk, indices.cpu().numpy(), frames.tolist()):
                    speaker = re.sub(r"[^\w-]+", "_", r.speaker or "speaker") or "speaker"
                    (folder / speaker).mkdir(exist_ok=True)
                    np.save(folder / speaker / f"{r.id}.npy", feature[:, :n])
                    clips.append((speaker, r))
                ctx.progress(min(1.0, (start + batch) / len(rows)) * 0.9, f"{min(start + batch, len(rows))} of {len(rows)} clips")
        del codec
        # build_dataset's folder tasks: the clips in path order, one group per speaker folder
        groups: dict[str, list[dict[str, Any]]] = {}
        for speaker, r in sorted(clips, key=lambda c: (c[0], f"{c[1].id}.npy")):
            groups.setdefault(speaker, []).append({"id": r.id, "texts": [pack_text(r.text.strip())], "codes": f"{speaker}/{r.id}.npy"})
        index = [{"name": name, "source": "folder", "sentences": sentences} for name, sentences in groups.items()]
        (folder / "groups.json").write_text(json.dumps(index, ensure_ascii=False, indent=1), encoding="utf-8")
        return {"clips": len(clips), "seconds": round(seconds, 1), "speakers": len(index)}

    def stage_train(self, ctx: StageContext) -> dict[str, Any]:
        """Upstream's LoRA fine-tune of the language model; its losses as metrics."""
        from concurrent.futures import ThreadPoolExecutor

        import torch
        from safetensors.torch import save_file

        from mizuoto.text.tokenize.bpe import load_bpe
        from mizuoto_fishspeech.finetune import Draws, Samples, collate, constant_with_warmup, initial_lora, load_groups, lora_state, losses, param_groups, setup_lora

        tokens = ctx.workdir / "tokens"
        if not (tokens / "groups.json").is_file():
            raise EngineError("No clip codes: run the tokens stage", code=codes.INVALID_INPUT)
        model_dir = Path(ctx.base_model["dir"])
        lora = self._lora(ctx)
        steps = int(ctx.hp("steps", 1000))
        max_length = int(ctx.hp("max_length", 4096))
        batch_size = int(ctx.hp("batch_size", 4))
        gpu = self._gpu(ctx)
        precision = ({"auto": "bf16-true", "bf16": "bf16-true", "fp16": "16-mixed"}.get(ctx.precision, "32-true")) if gpu else "32-true"

        ctx.progress(None, "Loading the model")
        tokenizer = load_bpe(model_dir)
        config = model_args(model_dir, tokenizer)
        # the draws upstream makes before training need only the shapes: they run while the weights load
        with ThreadPoolExecutor(1) as pool:
            init = pool.submit(initial_lora, config, lora, Draws())
            model = load_model(model_dir, config)
            # bf16-true casts the whole model next: the frozen copies are made in bfloat16 at once (no 17 GB of float32)
            setup_lora(model, lora, init.result(), torch.bfloat16 if precision == "bf16-true" else None)
        if precision == "bf16-true":
            model = model.to(torch.bfloat16)
        model = model.to(ctx.device)
        model.train()

        optimizer = torch.optim.AdamW(param_groups(model), lr=float(ctx.hp("learning_rate", 1e-4)), **OPTIMIZER)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, constant_with_warmup)
        scaler = torch.amp.GradScaler("cuda") if precision == "16-mixed" else None
        # the data loader's iterator draws its workers' base seed when training starts
        base_seed = int(torch.empty((), dtype=torch.int64).random_().item())
        try:
            pad_id = tokenizer.token_to_id(PAD_TOKEN)
        except KeyError:
            pad_id = 0
        samples = iter(Samples(load_groups(tokens), tokenizer, max_length, model.config.semantic_begin_id, base_seed))
        trainable = [p for group in optimizer.param_groups for p in group["params"]]
        last: dict[str, float] = {}
        for step in range(1, steps + 1):
            batch = collate([next(samples) for _ in range(batch_size)], max_length, pad_id)
            batch = {k: v.to(ctx.device) for k, v in batch.items()}
            lr = float(optimizer.param_groups[0]["lr"])
            if precision == "16-mixed":
                with torch.autocast("cuda", dtype=torch.float16):
                    out = losses(model, batch)
            else:
                out = losses(model, batch)
            optimizer.zero_grad(set_to_none=True)
            if scaler is not None:
                scaler.scale(out["loss"]).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(trainable, CLIP_NORM)
                scaler.step(optimizer)
                scaler.update()
            else:
                out["loss"].backward()
                torch.nn.utils.clip_grad_norm_(trainable, CLIP_NORM)
                optimizer.step()
            scheduler.step()
            last = {k: float(v.item()) for k, v in out.items()}
            ctx.metric(1, step, last, lr)
            ctx.progress(step / steps, f"Step {step} of {steps}")
        del optimizer, scaler

        folder = ctx.workdir / "lora"
        if folder.exists():
            shutil.rmtree(folder)
        folder.mkdir(parents=True)
        save_file(lora_state(model), str(folder / "lora.safetensors"))
        (folder / "lora.json").write_text(json.dumps({"lora": str(ctx.hp("lora", "r_8_alpha_16")), "steps": steps, "precision": precision}), encoding="utf-8")
        return {"steps": steps, "loss": last.get("loss")}

    def stage_export(self, ctx: StageContext) -> dict[str, Any]:
        """The LoRA merged into the base weights (``merge_lora``), saved as upstream saves a model, and a preview."""
        import torch
        from safetensors.torch import load_file

        from mizuoto.runtime import ops
        from mizuoto.text.tokenize.bpe import load_bpe
        from mizuoto_fishspeech.engine import _weights
        from mizuoto_fishspeech.finetune import merged_state
        from mizuoto_fishspeech.model import DualARTransformer, ModelArgs, dual_ar_args, omni_to_dual_ar, remap_keys

        lora_file = ctx.workdir / "lora" / "lora.safetensors"
        if not lora_file.is_file():
            raise EngineError("There are no trained LoRA weights; run the training stage", code=codes.INVALID_INPUT)
        trained = json.loads((ctx.workdir / "lora" / "lora.json").read_text(encoding="utf-8"))
        if trained.get("lora") != str(ctx.hp("lora", "r_8_alpha_16")):
            raise EngineError("The LoRA config changed since training; run the training stage again", code=codes.INVALID_INPUT)
        model_dir = Path(ctx.base_model["dir"])
        folder = ctx.workdir / "export"
        if folder.exists():
            shutil.rmtree(folder)
        folder.mkdir(parents=True)
        ctx.progress(0.1, "Merging")
        data = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
        config = omni_to_dual_ar(data) if data.get("model_type") == "fish_qwen3_omni" else dual_ar_args(data)
        tokenizer = load_bpe(model_dir)  # BaseTransformer.from_pretrained takes the semantic range from the tokenizer
        config["semantic_begin_id"], config["semantic_end_id"] = tokenizer.token_to_id("<|semantic:0|>"), tokenizer.token_to_id("<|semantic:4095|>")
        # upstream's parameter order: the module's (its weights are allocated, never touched)
        with torch.device("meta"):  # the names only: nothing allocated
            keys = list(DualARTransformer(ModelArgs.from_dual_ar(config), ops.default).state_dict(keep_vars=True))
        state = merged_state(remap_keys(_weights(model_dir)), load_file(str(lora_file)), self._lora(ctx), keys)
        (folder / "config.json").write_text(json.dumps(config, indent=4, sort_keys=True, ensure_ascii=False), encoding="utf-8")
        torch.save(state, str(folder / "model.pth"))
        del state
        for name in TOKENIZER_FILES:
            if (model_dir / name).is_file():
                shutil.copy2(model_dir / name, folder / name)
        for name in DEPLOY:
            if (model_dir / name).is_file():
                _link_or_copy(model_dir / name, folder / name)
        ctx.progress(0.8, "Preview")
        reference = next((r for r in ctx.rows if r.text.strip() and 3.0 <= r.duration <= 15.0), next((r for r in ctx.rows if r.text.strip()), None))
        text = str(ctx.hp("preview_text", "") or (reference.text if reference else "Hello."))
        ctx.preview(text, reference, "Preview", model_dir=folder, variant="s2_pro")
        return {"artifacts": [{"kind": "model", "folder": str(folder), "variant": "s2_pro", "title": ctx.name}]}


def model_args(model_dir: Path, tokenizer: Any) -> Any:
    """The model's arguments, the semantic range from the tokenizer (as ``BaseTransformer.from_pretrained``)."""
    from mizuoto_fishspeech.model import ModelArgs

    config = ModelArgs.from_config(json.loads((model_dir / "config.json").read_text(encoding="utf-8")))
    config.semantic_begin_id, config.semantic_end_id = tokenizer.token_to_id("<|semantic:0|>"), tokenizer.token_to_id("<|semantic:4095|>")
    return config


def load_model(model_dir: Path, config: Any) -> Any:
    """The language model holding the checkpoint's own tensors, in their stored dtype (bfloat16 for S2), as upstream's
    ``load_state_dict(assign=True)`` leaves it; the module's own allocations are never touched."""
    from mizuoto.runtime import ops
    from mizuoto_fishspeech.engine import _weights
    from mizuoto_fishspeech.model import DualARTransformer, remap_keys

    model = DualARTransformer(config, ops.default)
    result = model.load_state_dict(remap_keys(_weights(model_dir)), strict=False, assign=True)
    if result.missing_keys or result.unexpected_keys:
        raise EngineError(
            "The Fish Speech language model's weights do not match its architecture",
            {"missing": result.missing_keys[:20], "unexpected": result.unexpected_keys[:20]},
            code=codes.ENGINE_ERROR,
        )
    return model


def _link_or_copy(src: Path, dst: Path) -> None:
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)
