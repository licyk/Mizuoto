"""Upstream VoxCPM as a Mizuoto engine, for ``stack.py run --impl upstream``.

Mizuoto 0.1 had no VoxCPM, so its reference is upstream's own package. This engine stands in for the native one:
everything around the engine stays Mizuoto's (planning, reference files, segmentation, joining, output). The engine
itself calls ``voxcpm.VoxCPM.generate`` or ``generate_streaming`` as a user of upstream would, with what Mizuoto's
request maps to:

- the text, with VoxCPM2's ``(description)`` or ``(instruction)`` prefix as upstream's web demo builds it;
- ``prompt_wav_path`` + ``prompt_text`` when the reference has a transcript;
- ``reference_wav_path`` (VoxCPM2) without a transcript, or with one while ``reference_prefix`` is on;
- the parameters by their upstream names; the seed as given, else a 32-bit one from the request's stream.

It needs an interpreter with upstream VoxCPM installed (the VoxCPM reference environment).
"""

from __future__ import annotations

import json
import shutil
import tempfile
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from mizuoto.engines.voxcpm.engine import VoxCpmEngine, control_text
from mizuoto.runtime.engine import AudioChunk, Description, LoadedEngine, LoadSpec


class UpstreamVoxCpmEngine(VoxCpmEngine):
    def load(self, spec: LoadSpec, runtime: Any, ctx: Any) -> LoadedEngine:
        return LoadedUpstreamVoxCpm(spec)


class LoadedUpstreamVoxCpm(LoadedEngine):
    def __init__(self, spec: LoadSpec) -> None:
        super().__init__(spec)
        from voxcpm import VoxCPM

        assert spec.model_dir is not None
        folder = Path(spec.model_dir)
        self._tmp: Path | None = None
        if spec.dtype in ("fp32", "fp16", "bf16"):  # upstream reads the dtype from config.json
            self._tmp = Path(tempfile.mkdtemp(prefix="voxcpm-upstream-"))
            for item in folder.iterdir():
                if item.name == "config.json":
                    config = json.loads(item.read_text(encoding="utf-8"))
                    config["dtype"] = {"fp32": "float32", "fp16": "float16", "bf16": "bfloat16"}[spec.dtype]
                    (self._tmp / item.name).write_text(json.dumps(config), encoding="utf-8")
                elif item.is_file():
                    (self._tmp / item.name).symlink_to(item.resolve())
            folder = self._tmp
        self.model = VoxCPM(str(folder), enable_denoiser=False, optimize=False, device=spec.device)
        self.version = 2 if type(self.model.tts_model).__name__ == "VoxCPM2Model" else 1
        self.sample_rate = int(self.model.tts_model.sample_rate)

    def describe(self) -> Description:
        return Description(sample_rate=self.sample_rate)

    def close(self) -> None:
        super().close()
        if self._tmp is not None:
            shutil.rmtree(self._tmp, ignore_errors=True)

    def synthesize(self, task: Any, ctx: Any) -> Iterator[AudioChunk]:
        import numpy as np

        p, o = task.params, task.options
        text = task.text
        if self.version == 2:
            controls = [c for c in (control_text(task.voice.description or ""), control_text(task.style.instruct or "")) if c]
            if controls:
                text = f"({', '.join(controls)}){text}"
        seed = ctx.seed if ctx.seed is not None else ctx.rng.getrandbits(32)
        kwargs: dict[str, Any] = {
            "text": text,
            "cfg_value": float(p["guidance_scale"]) if p.get("guidance_scale") is not None else 2.0,
            "inference_timesteps": int(p.get("num_steps") or 10),
            "normalize": bool(p.get("text_normalization") or False),
            "max_len": int(p.get("max_new_tokens") or 4096),
            "min_len": int(o["min_len"]) if o.get("min_len") is not None else 2,
            "retry_badcase": bool(o["retry_badcase"]) if o.get("retry_badcase") is not None else True,
            "retry_badcase_max_times": int(o.get("retry_badcase_max_times") or 3),
            "retry_badcase_ratio_threshold": float(o.get("retry_badcase_ratio_threshold") or 6.0),
            "seed": seed,
        }
        mode = "zero_shot"
        if task.voice.references:
            ref = task.voice.references[0]
            transcript = (ref.transcript or "").strip()
            reference_prefix = bool(o["reference_prefix"]) if o.get("reference_prefix") is not None else True
            if transcript:
                kwargs.update(prompt_wav_path=ref.audio_path, prompt_text=transcript)
                mode = "continuation"
            if self.version == 2 and (not transcript or reference_prefix):
                kwargs["reference_wav_path"] = ref.audio_path
                mode = "reference" if mode == "zero_shot" else "ref_continuation"
        dtype = str(self.model.tts_model._dtype()).removeprefix("torch.")
        ctx.applied.update(
            {
                "inference_timesteps": kwargs["inference_timesteps"],
                "cfg_value": kwargs["cfg_value"],
                "normalize": kwargs["normalize"],
                "max_len": kwargs["max_len"],
                "min_len": kwargs["min_len"],
                "retry_badcase": kwargs["retry_badcase"] and not task.stream,
                "retry_badcase_max_times": kwargs["retry_badcase_max_times"],
                "retry_badcase_ratio_threshold": kwargs["retry_badcase_ratio_threshold"],
                "prompt_mode": mode,
                "dtype": dtype,
            }
        )
        if task.stream:
            ctx.applied["seed"] = seed
            pending = None
            stream = self.model.generate_streaming(**{**kwargs, "retry_badcase": False})
            try:
                for chunk in stream:
                    ctx.check_cancel()
                    if pending is not None:
                        yield AudioChunk(samples=pending, sample_rate=self.sample_rate, segment=task.segment)
                    pending = np.asarray(chunk, dtype=np.float32)
            finally:
                stream.close()
            if pending is not None:
                yield AudioChunk(samples=pending, sample_rate=self.sample_rate, segment=task.segment, final=True)
            return
        wav = self.model.generate(**kwargs)
        used = self.model.tts_model.last_successful_seed
        ctx.applied["seed"] = used
        if used != seed:
            ctx.applied["retries"] = used - seed
        yield AudioChunk(samples=np.asarray(wav, dtype=np.float32), sample_rate=self.sample_rate, segment=task.segment, final=True)


# -- training -----------------------------------------------------------------------------


def run_training(model_dir: Path, dataset: Path, out: Path, method: str, hyperparams: dict[str, Any], variant: str, device: str) -> int:
    """Upstream's ``train_voxcpm_finetune.train`` on a Mizuoto dataset folder, with the native trainer's recipe for the
    hyperparameters. Writes ``metrics.jsonl`` as Mizuoto does (stage ``train``, step + 1) and the saved weights in
    ``artifacts/model``. On the CPU the model is a copy whose ``config.json`` says float32: upstream's forward casts its
    inputs to the config's dtype, which fails against float32 weights without CUDA's autocast."""
    import sys

    sys.path.insert(0, str(Path.home() / "code_workspace" / "experiment" / "voxcpm" / "upstream" / "scripts"))
    import train_voxcpm_finetune as T  # ty: ignore[unresolved-import]

    from mizuoto.engines.voxcpm.train import Recipe
    from mizuoto.training.context import StageContext

    class _Quiet:
        def emit(self, event: Any) -> None:
            pass

    ctx = StageContext(_Quiet(), {"workdir": str(out / "ctx"), "hyperparams": hyperparams, "base_model": {"variant": variant}, "method": method})  # ty: ignore[invalid-argument-type]
    recipe = Recipe.of(ctx)
    shutil.rmtree(out / "ctx", ignore_errors=True)
    folder = model_dir
    tmp = None
    if not device.startswith("cuda"):
        tmp = Path(tempfile.mkdtemp(prefix="voxcpm-train-fp32-"))
        for item in model_dir.iterdir():
            if item.name == "config.json":
                config = json.loads(item.read_text(encoding="utf-8"))
                config["dtype"] = "float32"
                (tmp / item.name).write_text(json.dumps(config), encoding="utf-8")
            elif item.is_file():
                (tmp / item.name).symlink_to(item.resolve())
        folder = tmp
    rows = [json.loads(line) for line in (dataset / "manifest.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    manifest = out / "train.jsonl"
    manifest.write_text(
        "".join(
            json.dumps({"audio": str((dataset / r["audio"]).resolve()), "text": r["text"]}, ensure_ascii=False) + "\n" for r in rows if not r.get("excluded") and r["text"].strip()
        ),
        encoding="utf-8",
    )
    metrics: list[dict[str, Any]] = []
    original = T.TrainingTracker.log_metrics

    def log_metrics(self: Any, values: dict[str, float], split: str = "train") -> None:
        if split == "train":
            point = {k: float(v) for k, v in values.items() if k.startswith("loss/") or k == "grad_norm"}
            metrics.append({"stage": "train", "epoch": 0, "step": int(self.step) + 1, "values": point, "lr": float(values["lr"])})
        return original(self, values, split=split)

    T.TrainingTracker.log_metrics = log_metrics
    vae_rate = (json.loads((model_dir / "config.json").read_text(encoding="utf-8")).get("audio_vae_config") or {}).get("sample_rate", 16000)
    args: dict[str, Any] = {
        "pretrained_path": str(folder),
        "train_manifest": str(manifest),
        "sample_rate": vae_rate,
        "batch_size": recipe.batch_size,
        "grad_accum_steps": recipe.grad_accum,
        "num_workers": 0,
        "num_iters": recipe.steps,
        "max_steps": recipe.steps,
        "log_interval": 1,
        "valid_interval": 10**9,
        "save_interval": 10**9,
        "learning_rate": recipe.learning_rate,
        "weight_decay": recipe.weight_decay,
        "warmup_steps": recipe.warmup_steps,
        "max_batch_tokens": recipe.max_batch_tokens,
        "max_grad_norm": recipe.max_grad_norm,
        "save_path": str(out / "checkpoints"),
        "tensorboard": str(out / "tensorboard"),
    }
    lora = recipe.lora()
    if lora is not None:
        args["lora"] = {"enable_lm": lora.enable_lm, "enable_dit": lora.enable_dit, "enable_proj": lora.enable_proj, "r": lora.r, "alpha": lora.alpha, "dropout": lora.dropout}
    try:
        T.train(**args)
    finally:
        T.TrainingTracker.log_metrics = original
        if tmp is not None:
            shutil.rmtree(tmp, ignore_errors=True)
    (out / "metrics.jsonl").write_text("".join(json.dumps(m) + "\n" for m in metrics), encoding="utf-8")
    final = out / "checkpoints" / f"step_{recipe.steps:07d}"
    target = out / "artifacts" / "model"
    target.mkdir(parents=True, exist_ok=True)
    for name in ("model.safetensors", "lora_weights.safetensors", "lora_config.json"):
        if (final / name).is_file():
            shutil.copyfile(final / name, target / name)
    shutil.rmtree(out / "checkpoints", ignore_errors=True)  # optimizer states: gigabytes per save
    return 0
