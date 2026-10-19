"""VoxCPM fine-tuning, native: upstream's ``scripts/train_voxcpm_finetune.py`` with its ``conf/<model>`` recipes, full and
LoRA, on the native modules (no transformers, datasets, argbind or tensorboard).

- ``prepare`` reads the clips as upstream's ``datasets`` pipeline does (``sft.read_clip``), tokenises their text and
  drops the clips longer than ``max_batch_tokens // batch_size`` (``compute_sample_lengths``).
- ``train`` is the training loop (``sft.train_steps``) from upstream's state: everything seeded 42 by its
  ``Accelerator``, the construction's draws replayed, the base weights loaded, the VAE frozen. Full fine-tuning trains
  every weight; LoRA trains only the adapters (``LoRALinear`` on the language models and the DiT by default).
  - On CUDA, as upstream: float32 weights under bf16 autocast with a gradient scaler.
  - Elsewhere: float32 throughout. Upstream cannot train off CUDA with its configs: its forward casts the inputs to
    the config's bf16 while the weights stay float32.
  - It ends with upstream's checkpoint: ``model.safetensors`` (float32), or ``lora_weights.safetensors`` +
    ``lora_config.json``.
- ``export`` makes a loadable model folder: the base model's files with the new weights, or with the LoRA files beside
  the base weights (hard links where the file system allows), which the engine applies as upstream's
  ``VoxCPM(lora_weights_path=…)`` does. A preview is spoken with it.
"""

from __future__ import annotations

import json
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from mizuoto.core import codes
from mizuoto.runtime.engine import EngineError
from mizuoto.training import Row, StageContext, Trainer

SEED = 42
"""Upstream's ``Accelerator(seed=42)``: every generator seeded before the model is built."""
BASE_FILES = ("config.json", "audiovae.pth", "audiovae.safetensors", "tokenizer.json", "special_tokens_map.json", "tokenizer_config.json")
"""What upstream's ``save_checkpoint`` copies beside a full fine-tune's weights."""
RECIPES = {
    # conf/voxcpm_v1, conf/voxcpm_v1.5, conf/voxcpm_v2: (steps, batch, accumulation, max_grad_norm, lora rank, lora alpha)
    "v1": (2000, 16, 1, 0.0, 8, 16),
    "v1_5": (2000, 16, 1, 0.0, 8, 16),
    "v2": (1000, 2, 8, 1.0, 32, 32),
}


@dataclass
class Recipe:
    method: str
    steps: int
    learning_rate: float
    batch_size: int
    grad_accum: int
    warmup_steps: int
    weight_decay: float
    max_grad_norm: float
    max_batch_tokens: int
    lora_rank: int
    lora_alpha: float
    lora_dropout: float
    lora_lm: bool
    lora_dit: bool
    lora_proj: bool

    @classmethod
    def of(cls, ctx: StageContext) -> Recipe:
        method = str(ctx.request.get("method") or "full")
        if method not in ("full", "lora"):
            raise EngineError(f"Unknown VoxCPM training method {method}", code=codes.INVALID_INPUT)
        steps, batch, accum, clip, rank, alpha = RECIPES.get(str(ctx.base_model.get("variant")), RECIPES["v2"])

        def number(name: str, recipe: float) -> float:
            value = ctx.hp(name, 0)
            return recipe if value is None or float(value) <= 0 else float(value)

        max_norm = float(ctx.hp("max_grad_norm", -1))
        return cls(
            method=method,
            steps=int(number("steps", steps)),
            learning_rate=number("learning_rate", 1e-4 if method == "lora" else 1e-5),
            batch_size=int(number("batch_size", batch)),
            grad_accum=int(number("grad_accumulation", accum)),
            warmup_steps=int(ctx.hp("warmup_steps", 100)),
            weight_decay=float(ctx.hp("weight_decay", 0.01)),
            max_grad_norm=clip if max_norm < 0 else max_norm,
            max_batch_tokens=int(ctx.hp("max_batch_tokens", 8192)),
            lora_rank=int(number("lora_rank", rank)),
            lora_alpha=number("lora_alpha", alpha),
            lora_dropout=float(ctx.hp("lora_dropout", 0.0)),
            lora_lm=bool(ctx.hp("lora_lm", True)),
            lora_dit=bool(ctx.hp("lora_dit", True)),
            lora_proj=bool(ctx.hp("lora_proj", False)),
        )

    def lora(self) -> Any:
        from mizuoto.engines.voxcpm.model import LoraConfig

        if self.method != "lora":
            return None
        return LoraConfig(enable_lm=self.lora_lm, enable_dit=self.lora_dit, enable_proj=self.lora_proj, r=self.lora_rank, alpha=self.lora_alpha, dropout=self.lora_dropout)


class VoxCpmTrainer(Trainer):
    ENGINE = "voxcpm"

    def stage_prepare(self, ctx: StageContext) -> dict[str, Any]:
        """The clips at the VAE's rate and their tokens, filtered by length."""
        import numpy as np

        from mizuoto.engines.voxcpm.config import load_config
        from mizuoto.engines.voxcpm.sft import Clip, read_clip, sample_lengths
        from mizuoto.engines.voxcpm.text import VoxCpmTokenizer
        from mizuoto.nn.dsp.flac import read_audio

        recipe = Recipe.of(ctx)
        model_dir = Path(ctx.base_model["dir"])
        config = load_config(model_dir)
        tokenizer = VoxCpmTokenizer(model_dir)
        rate = config.vae.sample_rate
        rows = [r for r in ctx.rows if r.text.strip()]
        if not rows:
            raise EngineError("No clip has text", code=codes.INVALID_INPUT)
        data = ctx.workdir / "data"
        if data.exists():
            shutil.rmtree(data)
        data.mkdir(parents=True)
        clips, durations = [], []
        for i, row in enumerate(rows):
            samples, sr = read_audio(row.audio)
            durations.append(len(samples) / sr)  # soundfile's info: frames / rate
            clips.append(Clip(text_ids=tokenizer(row.text), audio=read_clip(row.audio, rate)))
            ctx.progress((i + 1) / len(rows), f"Read {i + 1} of {len(rows)} clips")
        keep = list(range(len(clips)))
        if recipe.max_batch_tokens > 0:
            limit = recipe.max_batch_tokens // recipe.batch_size
            lengths = sample_lengths(clips, durations, rate / _hop(config), config.patch_size)  # the VAE's frames per second
            keep = [i for i, n in enumerate(lengths) if n <= limit]
            if len(keep) < len(clips):
                ctx.log(f"Dropping {len(clips) - len(keep)} of {len(clips)} clips longer than {limit} tokens (max_batch_tokens {recipe.max_batch_tokens})")
        index = []
        for n, i in enumerate(keep):
            np.save(data / f"{n:06d}.npy", clips[i].audio)
            index.append({"id": rows[i].id, "text_ids": clips[i].text_ids, "audio": f"{n:06d}.npy"})
        (data / "index.json").write_text(json.dumps(index), encoding="utf-8")
        if len(index) < recipe.batch_size:
            raise EngineError(f"{len(index)} clips make no whole batch of {recipe.batch_size}: add clips or lower batch_size", {"clips": len(index)}, code=codes.INVALID_INPUT)
        return {"clips": len(index), "dropped": len(clips) - len(index)}

    def stage_train(self, ctx: StageContext) -> dict[str, Any]:
        """Upstream's training loop; each step's losses, learning rate and gradient norm as a metric."""
        import numpy as np
        import torch
        from safetensors.torch import save_file

        from mizuoto.engines.voxcpm.config import load_config
        from mizuoto.engines.voxcpm.model import VoxCpmModel, apply_lora
        from mizuoto.engines.voxcpm.sft import Clip, Loader, Packer, PlainScaler, StepResult, cosine_schedule, replay_construction, train_steps, trainable
        from mizuoto.engines.voxcpm.vae import AudioVAE
        from mizuoto.runtime.loading import fold_weight_norm, load_into, load_state_dict, unwrap
        from mizuoto.training.memory import checkpoint_every, checkpoint_units, frozen_in, preset, stream_frozen, wants

        recipe = Recipe.of(ctx)
        data = ctx.workdir / "data"
        if not (data / "index.json").is_file():
            raise EngineError("Prepare the clips first", code=codes.INVALID_INPUT)
        index = json.loads((data / "index.json").read_text(encoding="utf-8"))
        clips = [Clip(text_ids=item["text_ids"], audio=np.load(data / item["audio"])) for item in index]
        model_dir = Path(ctx.base_model["dir"])
        device = torch.device(ctx.device)
        out = ctx.workdir / "out"
        if out.exists():
            shutil.rmtree(out)
        out.mkdir(parents=True)

        ctx.progress(None, "Loading the model")
        # Upstream's generators after its Accelerator seeded them: the CPU's (construction, loader) and the device's
        # (the forward's draws); on a CPU they are one, as upstream's default generator is.
        cpu_gen = torch.Generator().manual_seed(SEED)
        device_gen = cpu_gen if device.type == "cpu" else torch.Generator(device).manual_seed(SEED)
        config = load_config(model_dir)
        vae = AudioVAE(config.vae)
        model = VoxCpmModel(config)
        lora = recipe.lora()
        layers = apply_lora(model, lora) if lora is not None else []
        replay_construction(model, vae, layers, cpu_gen)
        for layer in layers:
            layer.generator = device_gen
        vae_path = model_dir / "audiovae.safetensors" if (model_dir / "audiovae.safetensors").is_file() else model_dir / "audiovae.pth"
        load_into(vae, fold_weight_norm(unwrap(load_state_dict(vae_path), "state_dict"), device=device), what="AudioVAE")  # as upstream computes it, on the device
        weights = model_dir / "model.safetensors" if (model_dir / "model.safetensors").is_file() else model_dir / "pytorch_model.bin"
        load_into(model, unwrap(load_state_dict(weights), "state_dict"), ignore=["*.lora_A", "*.lora_B"] if lora else (), what="VoxCPM model")
        vae.requires_grad_(False)
        vae.encoder.to(device)  # training encodes every batch; its decoder stays in RAM
        model.to(device)
        model.train()
        params = trainable(model, lora)
        every = checkpoint_every(preset(ctx, "gradient_checkpointing"))
        if every:  # the forward's draws (LoRA's dropout, the CFM's) replayed in the backward: the same gradients
            units = [*model.base_lm.layers, *model.residual_lm.layers, *model.feat_encoder.encoder.layers, *model.feat_decoder.estimator.decoder.layers]
            n = checkpoint_units(units, list({id(g): g for g in (cpu_gen, device_gen)}.values()), every)
            ctx.log(f"Gradient checkpointing: {n} of {len(units)} blocks")

        cuda = device.type == "cuda"
        mixed = {"auto": torch.bfloat16, "bf16": torch.bfloat16, "fp16": torch.float16}.get(ctx.precision) if cuda else None
        if lora is not None and mixed is not None and preset(ctx, "frozen_dtype") == "autocast":
            from mizuoto.engines.voxcpm.model import LoRALinear

            saved = frozen_in(model, mixed, (LoRALinear,))
            ctx.log(f"Frozen weights kept in {str(mixed).removeprefix('torch.')}: {saved / 2**20:.0f} MB saved")
        if lora is not None and wants(ctx, "frozen_device", "cpu"):
            moved = stream_frozen(layers, device)
            ctx.log(f"Frozen LoRA base weights stream from RAM: {moved / 2**20:.0f} MB off the device")
        dtype = mixed or torch.float32  # the inputs: the config's bf16 under autocast on CUDA, else float32
        scaler: Any = torch.amp.GradScaler("cuda") if mixed is not None else PlainScaler()
        optimizer = torch.optim.AdamW(params, lr=recipe.learning_rate, weight_decay=recipe.weight_decay)
        scheduler = cosine_schedule(optimizer, recipe.warmup_steps, recipe.steps)
        packer = Packer(vae, config)
        loader = Loader(clips, recipe.batch_size, cpu_gen)

        def autocast() -> Any:
            return torch.autocast("cuda", dtype=mixed) if mixed is not None else torch.autocast("cpu", enabled=False)

        def on_step(result: StepResult) -> None:
            ctx.metric(loader.epoch + 1, result.step + 1, {**result.losses, "grad_norm": result.grad_norm}, result.lr)
            ctx.progress((result.step + 1) / recipe.steps, f"Step {result.step + 1} of {recipe.steps}")

        ctx.progress(0.0, f"Step 0 of {recipe.steps}")
        train_steps(
            model,
            loader,
            packer,
            optimizer,
            scheduler,
            steps=recipe.steps,
            grad_accum=recipe.grad_accum,
            max_grad_norm=recipe.max_grad_norm,
            dtype=dtype,
            device=device,
            clips=len(clips),
            batch_size=recipe.batch_size,
            autocast=autocast,
            scaler=scaler,
            on_step=on_step,
            check=lambda: None,
            generator=device_gen,
        )
        state = {k: v.detach().contiguous().cpu() for k, v in model.state_dict().items()}
        if lora is not None:
            save_file({k: v for k, v in state.items() if "lora_" in k}, str(out / "lora_weights.safetensors"))
            info = {"base_model": str(model_dir), "lora_config": lora.to_dict(config.version)}
            (out / "lora_config.json").write_text(json.dumps(info, indent=2, ensure_ascii=False), encoding="utf-8")
        else:
            save_file(state, str(out / "model.safetensors"))
        (out / "training_state.json").write_text(json.dumps({"step": recipe.steps}), encoding="utf-8")
        return {"steps": recipe.steps, "method": recipe.method}

    def stage_export(self, ctx: StageContext) -> dict[str, Any]:
        """A model folder the engine (and upstream) loads, and a preview."""
        out = ctx.workdir / "out"
        model_dir = Path(ctx.base_model["dir"])
        lora = (out / "lora_weights.safetensors").is_file()
        if not lora and not (out / "model.safetensors").is_file():
            raise EngineError("Training saved no weights; run the training stage", code=codes.INVALID_INPUT)
        folder = ctx.workdir / "export"
        if folder.exists():
            shutil.rmtree(folder)
        folder.mkdir(parents=True)
        for name in BASE_FILES:
            if (model_dir / name).is_file():
                _link_or_copy(model_dir / name, folder / name)
        if lora:
            base = model_dir / "model.safetensors" if (model_dir / "model.safetensors").is_file() else model_dir / "pytorch_model.bin"
            _link_or_copy(base, folder / base.name)
            shutil.copyfile(out / "lora_weights.safetensors", folder / "lora_weights.safetensors")
            info = json.loads((out / "lora_config.json").read_text(encoding="utf-8"))
            info["base_model"] = str(ctx.base_model.get("id") or info.get("base_model"))
            (folder / "lora_config.json").write_text(json.dumps(info, indent=2, ensure_ascii=False), encoding="utf-8")
        else:
            shutil.copyfile(out / "model.safetensors", folder / "model.safetensors")
        reference = _reference(ctx.rows)  # none fits: VoxCPM speaks the preview in its own voice
        text = str(ctx.hp("preview_text", "") or "") or (reference.text if reference else next((r.text for r in ctx.rows if r.text.strip()), "Hello."))
        ctx.preview(text, reference, "Preview", model_dir=folder)
        return {"artifacts": [{"kind": "model", "folder": str(folder), "variant": str(ctx.base_model.get("variant")), "title": ctx.name}], "lora": lora}


def _hop(config: Any) -> int:
    import math

    return math.prod(config.vae.encoder_rates)


def _link_or_copy(src: Path, dst: Path) -> None:
    """A hard link (the base model's files are never changed in place), else a copy."""
    try:
        os.link(src, dst)
    except OSError:
        shutil.copyfile(src, dst)


def _reference(rows: list[Row]) -> Row | None:
    fitting = [r for r in rows if r.text.strip() and 3.0 <= r.duration <= 15.0]
    return min(fitting, key=lambda r: abs(r.duration - 6.0)) if fitting else None
