"""OmniVoice fine-tuning, native: v1's trainer (upstream's ``examples/run_finetune.sh``
and ``run_finetune_lora.sh``) on the native modules, with no transformers, peft, accelerate, webdataset or datasets.

- ``tokens`` is ``extract_audio_tokens`` on the clips' JSONL: the entries shuffled with seed 42, each clip read, made
  mono, resampled to 24 kHz, peak-normalised to 0.9, zero-padded to whole frames and encoded by the model's HiggsAudio
  tokenizer; the codes go into WebDataset-style shards (``audios/shard-N.tar`` of ``<id>.npy``, ``txts/shard-N.jsonl``)
  listed by ``data.lst``, 1000 clips a shard.
- ``train`` is ``omnivoice.cli.train`` with upstream's fine-tune configs (SDPA: length-grouped padding, not
  FlexAttention's packing) as v1 set them: the shards read as ``WebDatasetReader`` and the DataLoader's workers read
  them (shard order shuffled per epoch, a 20000-sample shuffle buffer seeded with the epoch, each worker seeded from the
  loader's base seed as ``seed_worker`` does), ``OmniVoiceSampleProcessor``'s random prompt, masking and condition
  drops, ``StreamLengthGroupDataset``'s buckets, ``PaddingDataCollator``'s batches under a bidirectional 4-D mask; the
  loss is the masked cross-entropy weighted per codebook; AdamW with weight decay, the cosine schedule with warm-up,
  gradient clipping at 1, accelerate's gradient accumulation and mixed precision (CUDA only). LoRA is peft's: rank-r
  adapters (alpha 2r, dropout 0.05) on every projection of the language model, the audio embeddings and heads trained
  in full. It ends with ``out/checkpoint-<steps>`` in upstream's layout: the model (``model.safetensors``) or the
  adapter (``adapter_model.safetensors`` with peft's key names and ``adapter_config.json``).
- ``export`` is ``merge_lora``: the adapter merged into the base weights (or the fine-tuned weights as they are),
  saved as ``model.safetensors`` with upstream's keys, beside the base model's config, text tokenizer, chat template
  and audio tokenizer: a model both v1 and the native engine load. A preview is spoken with it.

Seeded alike (the stage seeds Python's, NumPy's and torch's generators), a run draws what upstream draws, in its order.
Upstream builds the model and its LoRA layers (peft's initialisation draws from the process-wide generator) and then
reseeds everything with the config's seed (42); the reseeded draws are made here on private generators seeded 42: the
loaders' base seeds and the LoRA dropout masks in this process; the data workers' shuffles, prompt ratios and masks on
generators seeded as each worker is.
"""

from __future__ import annotations

import bisect
import io
import json
import math
import random
import shutil
import tarfile
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, TypeVar

from mizuoto.core import codes
from mizuoto.runtime.engine import EngineError
from mizuoto.training import Row, StageContext, Trainer

if TYPE_CHECKING:
    import numpy as np
    import torch
    from torch import nn

    from mizuoto.engines.omnivoice.model import OmniVoiceModel

SAMPLES_PER_SHARD = 1000
EXTRACT_SHUFFLE_SEED = 42
"""``extract_audio_tokens``' ``--shuffle-seed`` default: ``JsonlDatasetReader`` shuffles the entries with it."""
SHUFFLE_BUFFER = 20000
"""``WebDatasetReader``'s ``shuffle_buffer_size``; webdataset's shuffle starts yielding once it holds 100."""
SHUFFLE_INITIAL = 100
NUM_BUCKETS = 20
IGNORE = -100
TOKENIZER_FILES = ("tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt", "special_tokens_map.json", "added_tokens.json", "chat_template.jinja")
DEPLOY = ("audio_tokenizer", "chat_template.jinja")
"""What a loadable model folder needs beside the weights and the text tokenizer (``merge_lora._EXTRA_DEPLOY_PATHS``)."""
TARGETS = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
MODULES_TO_SAVE = ("audio_embeddings", "audio_heads")
T = TypeVar("T")
CODEBOOK_WEIGHTS = (8, 8, 6, 6, 4, 4, 2, 2)
"""``OmniVoiceConfig``'s default ``audio_codebook_weights``."""


@dataclass
class Recipe:
    """Upstream's ``train_config_finetune_sdpa.json`` / ``train_config_finetune_lora.json`` (with ``TrainingConfig``'s
    defaults for what they leave out), and what v1 set from the hyperparameters."""

    method: str
    steps: int
    learning_rate: float
    batch_tokens: int
    accumulation: int
    workers: int
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    drop_cond_ratio: float = 0.1
    prompt_ratio_range: tuple[float, float] = (0.0, 0.3)
    mask_ratio_range: tuple[float, float] = (0.0, 1.0)
    language_ratio: float = 0.8
    use_pinyin_ratio: float = 0.0
    instruct_ratio: float = 0.0
    only_instruct_ratio: float = 0.0
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    seed: int = 42
    warmup_ratio: float = 0.01
    min_sample_tokens: int = 50
    max_sample_tokens: int = 2000
    max_batch_size: int = 64

    @property
    def lora(self) -> bool:
        return self.method == "lora"

    @property
    def warmup(self) -> int:
        return math.ceil(self.steps * self.warmup_ratio)

    @classmethod
    def of(cls, ctx: StageContext) -> Recipe:
        method = _method(ctx)
        rank = int(ctx.hp("lora_rank", 16))
        return cls(
            method=method,
            steps=int(ctx.hp("steps", 1000)),
            learning_rate=float(ctx.hp("learning_rate", 1e-4 if method == "lora" else 1e-5)),
            batch_tokens=int(ctx.hp("batch_tokens", 8192)),
            accumulation=int(ctx.hp("grad_accumulation", 1)),
            workers=max(1, int(ctx.hp("workers", 2))),
            lora_r=rank,
            lora_alpha=2 * rank,
        )


class OmniVoiceTrainer(Trainer):
    ENGINE = "omnivoice"

    # -- stages ---------------------------------------------------------------------------------

    def stage_tokens(self, ctx: StageContext) -> dict[str, Any]:
        """The clips' audio tokens in shards (``extract_audio_tokens``)."""
        import numpy as np
        import torch

        from mizuoto.engines.omnivoice.engine import load_audio_tokenizer
        from mizuoto.nn.dsp.flac import read_audio
        from mizuoto.nn.dsp.resample import resample

        _method(ctx)
        tokens = ctx.workdir / "tokens"
        if tokens.exists():
            shutil.rmtree(tokens)
        train = tokens / "train"
        (train / "audios").mkdir(parents=True)
        (train / "txts").mkdir(parents=True)
        rows = [r for r in ctx.rows if r.text.strip()]
        if not rows:
            raise EngineError("No clip has text", code=codes.INVALID_INPUT)
        entries: list[dict[str, Any]] = []
        for r in rows:
            item: dict[str, Any] = {"id": r.id, "audio_path": r.audio, "text": r.text.strip()}
            language = (r.language or ctx.request.get("language") or "").split("-")[0].lower()
            if language:
                item["language_id"] = language
            entries.append(item)
        random.Random(EXTRACT_SHUFFLE_SEED).shuffle(entries)

        ctx.progress(None, "Loading the audio tokenizer")
        device = torch.device(ctx.device)
        tokenizer, rate = load_audio_tokenizer(Path(ctx.base_model["dir"]))
        tokenizer.to(device)
        hop = tokenizer.hop_length
        layerdrop = torch.Generator()  # HuBERT's layer-drop draws, which never change the codes in eval mode
        writer = ShardWriter(train)
        failed = 0
        for i, entry in enumerate(entries):
            try:
                samples, sr = read_audio(entry["audio_path"])
                wav = torch.from_numpy(np.ascontiguousarray(samples, dtype=np.float32))[None]
                if sr != rate:
                    wav = resample(wav, sr, rate)
                wav = (wav / (wav.abs().max() + 1e-7)) * 0.9
                duration = wav.shape[1] / rate
                pad = (-wav.shape[1]) % hop  # the feature extractor pads to whole frames
                with torch.inference_mode():
                    padded = torch.nn.functional.pad(wav, (0, pad)) if pad else wav
                    codes_ = tokenizer.encode(padded[None].to(device), layerdrop=layerdrop).squeeze(0)
            except Exception as e:  # upstream runs with --skip_errors
                ctx.log(f"Skipping {entry['id']}: {type(e).__name__}: {e}", "warning")
                failed += 1
                continue
            writer.write(entry["id"], codes_.to(torch.int16).cpu().numpy(), {**entry, "audio_duration": duration, "num_tokens": int(codes_.shape[1])})
            ctx.progress((i + 1) / len(entries))
        count = writer.close()
        if not count:
            raise EngineError("No clip was tokenized; see the log", code=codes.ENGINE_ERROR)
        return {"clips": count, "skipped": len(rows) - count}

    def stage_train(self, ctx: StageContext) -> dict[str, Any]:
        """``omnivoice.cli.train`` with upstream's fine-tune config, from the base model; each update's loss as a metric."""
        import torch

        from mizuoto.text.tokenize.bpe import load_bpe
        from mizuoto.training.memory import checkpoint_every, checkpoint_units, host_optimizer, preset, wants

        recipe = Recipe.of(ctx)
        manifest = ctx.workdir / "tokens" / "train" / "data.lst"
        if not manifest.is_file():
            raise EngineError("Make the audio tokens first", code=codes.INVALID_INPUT)
        shards = read_manifest(manifest)
        out = ctx.workdir / "out"
        if out.exists():
            shutil.rmtree(out)
        out.mkdir(parents=True)
        model_dir = Path(ctx.base_model["dir"])
        device = torch.device(ctx.device)
        mixed = _mixed_precision(ctx)

        ctx.progress(None, "Loading the model")
        config = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
        text_tokenizer = load_bpe(model_dir)
        if text_tokenizer.vocab_size != int(config["llm_config"]["vocab_size"]):
            raise EngineError("The text tokenizer does not match the model's vocabulary", code=codes.INVALID_INPUT)
        model = _load_model(model_dir, config)
        # Upstream reseeds every generator with the config's seed after building the model (and its LoRA layers, whose
        # initialisation draws from the process-wide generator): its later draws come from these.
        main_cpu = torch.Generator().manual_seed(recipe.seed)
        main_device = main_cpu if device.type == "cpu" else torch.Generator(device).manual_seed(recipe.seed)
        if recipe.lora:
            add_lora(model, recipe, main_device)
        model.to(device)
        params = _upstream_order(model)
        trainable = [p for p in params if p.requires_grad]
        if wants(ctx, "optimizer_device", "cpu") and mixed != torch.float16:  # the lowest preset: AdamW's state in RAM
            optimizer = host_optimizer(trainable, lambda ps: torch.optim.AdamW(ps, lr=recipe.learning_rate, weight_decay=recipe.weight_decay))
            ctx.log("The optimizer's state is kept in RAM and stepped on the CPU")
        else:
            optimizer = torch.optim.AdamW(trainable, lr=recipe.learning_rate, weight_decay=recipe.weight_decay)
        scheduler = cosine_schedule(optimizer, recipe.warmup, recipe.steps)
        scaler = torch.amp.GradScaler("cuda") if mixed == torch.float16 else None
        processor = SampleProcessor(recipe, text_tokenizer, int(config.get("num_audio_codebook", 8)), int(config.get("audio_mask_id", 1024)))
        loader = Loader(shards, recipe, processor, pad_id(model_dir, text_tokenizer), main_cpu)
        # the model's own codebook weights (its config.json), as ``OmniVoice`` normalises them
        weights = [float(w) / sum(config.get("audio_codebook_weights") or CODEBOOK_WEIGHTS) for w in config.get("audio_codebook_weights") or CODEBOOK_WEIGHTS]

        every = checkpoint_every(preset(ctx, "gradient_checkpointing"))
        if every:  # LoRA's dropout masks come from the main generators: replayed in the backward
            n = checkpoint_units(found := list(model.llm.layers), list({id(g): g for g in (main_cpu, main_device)}.values()), every)
            ctx.log(f"Gradient checkpointing: {n} of {len(found)} blocks")
        model.train()
        step = 0
        micro = 0
        ctx.progress(0.0, f"Step 0 of {recipe.steps}")
        while step < recipe.steps:
            batch = {k: v.to(device) for k, v in loader.next().items()}
            with torch.autocast(device.type, dtype=mixed, enabled=mixed is not None):
                loss = forward_loss(model, batch, weights)
            loss = loss.float()
            scaled = loss / recipe.accumulation
            (scaler.scale(scaled) if scaler is not None else scaled).backward()
            micro += 1
            if micro % recipe.accumulation:
                continue
            if scaler is not None:
                scaler.unscale_(optimizer)
            if recipe.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(params, recipe.max_grad_norm)
            if scaler is not None:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            step += 1
            ctx.metric(loader.epoch + 1, step, {"loss": float(loss.item())}, float(scheduler.get_last_lr()[0]))
            ctx.progress(min(1.0, step / recipe.steps), f"Step {step} of {recipe.steps}")
        checkpoint = out / f"checkpoint-{step}"
        save_checkpoint(model, recipe, model_dir, checkpoint)
        return {"steps": step, "checkpoint": checkpoint.name}

    def stage_export(self, ctx: StageContext) -> dict[str, Any]:
        """A loadable model folder (``merge_lora``), and a preview."""
        from safetensors.torch import save_file

        from mizuoto.runtime.loading import load_state_dict

        _method(ctx)
        checkpoint = _checkpoint(ctx.workdir / "out")
        model_dir = Path(ctx.base_model["dir"])
        folder = ctx.workdir / "export"
        if folder.exists():
            shutil.rmtree(folder)
        folder.mkdir(parents=True)
        if (checkpoint / "adapter_config.json").is_file():
            ctx.progress(None, "Merging the LoRA into the base model")
            state = merge_lora(dict(load_state_dict(model_dir / "model.safetensors", mmap=False)), checkpoint)
            save_file(state, str(folder / "model.safetensors"), metadata={"format": "pt"})
            del state
        else:
            shutil.copyfile(checkpoint / "model.safetensors", folder / "model.safetensors")
        shutil.copyfile(model_dir / "config.json", folder / "config.json")
        _copy_deploy(model_dir, folder)
        reference = _reference(ctx.rows)
        if reference is not None:
            text = str(ctx.hp("preview_text", "") or "") or reference.text
            ctx.preview(text, reference, "Preview", model_dir=folder)
        return {"artifacts": [{"kind": "model", "folder": str(folder), "variant": str(ctx.base_model.get("variant")), "title": ctx.name}], "checkpoint": checkpoint.name}


def _method(ctx: StageContext) -> str:
    method = str(ctx.request.get("method") or "full")
    if method not in ("full", "lora"):
        raise EngineError(f"Unknown OmniVoice training method {method}", code=codes.INVALID_INPUT)
    return method


def _mixed_precision(ctx: StageContext) -> torch.dtype | None:
    """accelerate's mixed precision as v1 asked for it: none off CUDA; bf16 for ``auto``."""
    import torch

    if not ctx.device.startswith("cuda"):
        return None
    return {"auto": torch.bfloat16, "bf16": torch.bfloat16, "fp16": torch.float16}.get(ctx.precision)


def _checkpoint(out: Path) -> Path:
    found = sorted(out.glob("checkpoint-*"), key=lambda p: int(p.name.split("-")[-1]) if p.name.split("-")[-1].isdigit() else -1)
    if not found:
        raise EngineError("Training saved no checkpoint; run the training stage", code=codes.INVALID_INPUT)
    return found[-1]


def _copy_deploy(model_dir: Path, folder: Path) -> None:
    """The base model's text tokenizer (as ``save_pretrained`` writes it back unchanged), chat template and audio tokenizer."""
    for name in (*TOKENIZER_FILES, *DEPLOY):
        src = model_dir / name
        if src.is_dir():
            shutil.copytree(src, folder / name, dirs_exist_ok=True)
        elif src.is_file():
            shutil.copyfile(src, folder / name)


def _reference(rows: list[Row]) -> Row | None:
    fitting = [r for r in rows if r.text.strip() and 3.0 <= r.duration <= 15.0]
    return min(fitting, key=lambda r: abs(r.duration - 6.0)) if fitting else None


# -- token shards ---------------------------------------------------------------------------------


class ShardWriter:
    """``extract_audio_tokens``' output: tar shards of ``<id>.npy`` (int16 codes, ``np.save``), a JSONL of each clip's
    metadata beside them and ``data.lst`` (``<tar> <jsonl> <clips> <seconds>``)."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.lines: list[str] = []
        self.pending: list[tuple[str, bytes, dict[str, Any]]] = []
        self.total = 0

    def _flush(self) -> None:
        if not self.pending:
            return
        index = len(self.lines)
        tar = (self.root / "audios" / f"shard-{index:06d}.tar").absolute()
        jsonl = (self.root / "txts" / f"shard-{index:06d}.jsonl").absolute()
        with tarfile.open(tar, "w") as t:
            for key, data, _ in self.pending:
                info = tarfile.TarInfo(f"{key}.npy")
                info.size = len(data)
                info.mode = 0o444
                t.addfile(info, io.BytesIO(data))
        jsonl.write_text("".join(json.dumps(m, ensure_ascii=False) + "\n" for _, _, m in self.pending), encoding="utf-8")
        seconds = sum(float(m.get("audio_duration", 0.0)) for _, _, m in self.pending)
        self.lines.append(f"{tar} {jsonl} {len(self.pending)} {seconds:.3f}\n")
        self.pending = []

    def write(self, key: str, codes_: np.ndarray, metadata: dict[str, Any]) -> None:
        import numpy as np

        if len(self.pending) >= SAMPLES_PER_SHARD:
            self._flush()
        buffer = io.BytesIO()
        np.save(buffer, codes_)
        self.pending.append((key, buffer.getvalue(), metadata))
        self.total += 1

    def close(self) -> int:
        self._flush()
        (self.root / "data.lst").write_text("".join(self.lines), encoding="utf-8")
        return self.total


@dataclass
class Shard:
    tar: str
    jsonl: str


def read_manifest(path: Path) -> list[Shard]:
    """``webdataset_manifest_reader``."""
    shards = []
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if not parts:
            continue
        if len(parts) != 4:
            raise EngineError(f"Invalid manifest line: {line}", code=codes.INVALID_INPUT)
        shards.append(Shard(parts[0], parts[1]))
    if not shards:
        raise EngineError("No clip was tokenized; make the audio tokens again", code=codes.INVALID_INPUT)
    return shards


def read_shard(shard: Shard) -> Iterator[dict[str, Any]]:
    """A shard's samples in their order, as webdataset's tar reader and ``SampleDecoder`` give them:
    ``{"audio_tokens": int16 tensor (codebooks, frames), "label": its JSONL record}``."""
    import numpy as np
    import torch

    labels = {}
    for line in Path(shard.jsonl).read_text(encoding="utf-8").splitlines():
        if line.strip():
            item = json.loads(line)
            if "id" in item:
                labels[item["id"]] = item
    with tarfile.open(shard.tar) as tar:
        for member in tar:
            if not member.isfile() or not member.name.endswith(".npy"):
                continue
            f = tar.extractfile(member)
            assert f is not None
            key = Path(member.name).name[: -len(".npy")]
            yield {"audio_tokens": torch.from_numpy(np.load(io.BytesIO(f.read()))), "label": labels[key]}


# -- the data pipeline ----------------------------------------------------------------------------


class SampleProcessor:
    """``OmniVoiceSampleProcessor``: the style and text tokens, then the audio codes with a random prompt kept and a
    random share of the rest masked (the loss only there); one in ten samples drops every condition."""

    def __init__(self, recipe: Recipe, tokenizer: Any, codebooks: int, mask_id: int) -> None:
        self.recipe = recipe
        self.tokenizer = tokenizer
        self.codebooks = codebooks
        self.mask_id = mask_id

    def _ids(self, text: str) -> torch.Tensor:
        import torch

        return torch.tensor([self.tokenizer.encode(text)], dtype=torch.long).repeat(self.codebooks, 1)

    def __call__(self, sample: dict[str, Any], rng: random.Random, generator: torch.Generator) -> dict[str, Any]:
        import torch

        r = self.recipe
        label = sample["label"]
        drop_cond = rng.uniform(0, 1) < r.drop_cond_ratio
        if drop_cond:
            prompt_ratio, drop_text, use_language, use_instruct = 0.0, True, False, False
        else:
            prompt_ratio = rng.uniform(*r.prompt_ratio_range)
            drop_text = False
            use_language = rng.uniform(0, 1) < r.language_ratio
            use_instruct = rng.uniform(0, 1) < r.instruct_ratio
            if use_instruct and rng.uniform(0, 1) < r.only_instruct_ratio:
                prompt_ratio = 0.0
        mask_ratio = rng.uniform(*r.mask_ratio_range)
        language = label.get("language_id", "None") if use_language else "None"
        instruct = label.get("instruct", "None") if use_instruct else "None"
        style_inputs = self._ids(f"<|lang_start|>{language}<|lang_end|><|instruct_start|>{instruct}<|instruct_end|>")
        style_labels = torch.full(style_inputs.shape, IGNORE)
        text = label["text_pinyin"] if "text_pinyin" in label and rng.uniform(0, 1) < r.use_pinyin_ratio else label["text"]
        text_inputs = self._ids(f"<|text_start|>{text}<|text_end|>")
        text_labels = torch.full(text_inputs.shape, IGNORE)

        audio_tokens = sample["audio_tokens"].long()
        prompt_length = int(audio_tokens.shape[1] * prompt_ratio)
        audio_inputs = audio_tokens.clone()
        audio_labels = audio_tokens.clone()
        token_mask = torch.rand(audio_tokens[:, prompt_length:].shape, generator=generator) < mask_ratio
        audio_inputs[:, prompt_length:][token_mask] = self.mask_id
        audio_labels[:, prompt_length:][~token_mask] = IGNORE
        if not drop_cond:
            audio_labels[:, :prompt_length] = IGNORE

        if drop_text:
            input_ids, labels = audio_inputs, audio_labels
            audio_mask = torch.ones(input_ids.shape[1], dtype=torch.bool)
        else:
            input_ids = torch.cat([style_inputs, text_inputs, audio_inputs], dim=1)
            labels = torch.cat([style_labels, text_labels, audio_labels], dim=1)
            audio_mask = torch.zeros(input_ids.shape[1], dtype=torch.bool)
            audio_mask[style_inputs.shape[1] + text_inputs.shape[1] :] = True
        return {"input_ids": input_ids, "labels": labels, "audio_mask": audio_mask, "length": int(input_ids.shape[1])}


def shuffle_buffer(samples: Iterator[T], rng: random.Random, size: int = SHUFFLE_BUFFER, initial: int = SHUFFLE_INITIAL) -> Iterator[T]:
    """webdataset's ``shuffle``: a buffer filled two at a time, a random pick from it once it holds ``initial``."""

    def pick(buf: list[T]) -> T:
        k = rng.randint(0, len(buf) - 1)
        sample = buf[k]
        buf[k] = buf[-1]
        buf.pop()
        return sample

    initial = min(initial, size)
    buf: list[T] = []
    for sample in samples:
        buf.append(sample)
        if len(buf) < size:
            try:
                buf.append(next(samples))
            except StopIteration:
                pass
        if len(buf) >= initial:
            yield pick(buf)
    while buf:
        yield pick(buf)


def length_groups(samples: Iterator[dict[str, Any]], recipe: Recipe) -> Iterator[list[dict[str, Any]]]:
    """``StreamLengthGroupDataset`` on processed samples, measured in tokens: 20 buckets between the minimum and
    maximum length, a bucket given out once one more of its longest would pass the token budget or it holds the most
    samples a batch may; the rest at the end, bucket by bucket."""
    import numpy as np

    boundaries = np.linspace(recipe.min_sample_tokens, recipe.max_sample_tokens, NUM_BUCKETS + 1)[1:]
    buckets: list[list[dict[str, Any]]] = [[] for _ in range(NUM_BUCKETS)]
    longest = [0.0] * NUM_BUCKETS
    for sample in samples:
        length = sample["length"]
        if length < recipe.min_sample_tokens or length > recipe.max_sample_tokens:
            continue
        b = bisect.bisect_left(boundaries, length)
        buckets[b].append(sample)
        longest[b] = max(longest[b], length)
        if longest[b] * (len(buckets[b]) + 1) >= recipe.batch_tokens or len(buckets[b]) >= recipe.max_batch_size:
            yield buckets[b]
            buckets[b] = []
            longest[b] = 0.0
    for bucket in buckets:
        if bucket:
            yield bucket


def collate(samples: list[dict[str, Any]], pad: int) -> dict[str, torch.Tensor]:
    """``PaddingDataCollator``: (batch, codebooks, length) padded with the pad token (labels ignored), positions from 0
    per sample, and a 4-D boolean mask where every position sees every real one."""
    import torch
    import torch.nn.functional as F

    longest = max(s["length"] for s in samples)
    valid = torch.zeros(len(samples), longest, dtype=torch.bool)
    ids, labels, audio, positions = [], [], [], []
    for i, s in enumerate(samples):
        extra = longest - s["length"]
        ids.append(F.pad(s["input_ids"], (0, extra), value=pad))
        labels.append(F.pad(s["labels"], (0, extra), value=IGNORE))
        audio.append(F.pad(s["audio_mask"], (0, extra), value=False))
        positions.append(F.pad(torch.arange(s["length"], dtype=torch.long), (0, extra), value=0))
        valid[i, : s["length"]] = True
    batch = len(samples)
    return {
        "input_ids": torch.stack(ids),
        "labels": torch.stack(labels),
        "audio_mask": torch.stack(audio),
        "position_ids": torch.stack(positions),
        "attention_mask": valid[:, None, None, :].expand(batch, 1, longest, longest).contiguous(),
    }


class Loader:
    """The training DataLoader as upstream builds it (``batch_size=None``, ``num_workers`` workers each running the
    reader, the processor, the length groups and the collator), epoch after epoch.

    Each epoch shuffles the shard list with ``Random(epoch)`` and gives worker *w* shards ``w, w + n, …``; the loader
    draws its base seed from the main generator when it starts the epoch, and worker *w* seeds Python's and torch's
    generators with ``(base + w) mod 2**32`` (``seed_worker``). Batches come from the workers in turn, a worker that has
    run out being skipped."""

    def __init__(self, shards: list[Shard], recipe: Recipe, processor: SampleProcessor, pad: int, main: torch.Generator) -> None:
        self.shards = shards
        self.recipe = recipe
        self.processor = processor
        self.pad = pad
        self.main = main
        # More workers than shards made v1 fail (webdataset finds a worker without samples); the spare ones are idle.
        self.workers = min(recipe.workers, len(shards))
        self.epoch = 0
        self._batches = self._start()

    def _worker(self, shards: list[Shard], epoch: int, seed: int) -> Iterator[list[dict[str, Any]]]:
        import torch

        rng = random.Random(seed)
        generator = torch.Generator().manual_seed(seed)
        # webdataset's WebDataset(), made in the worker when the reader starts, draws a seed it does not use here
        # (``random.randint(0, 1000000)``, evaluated even when WDS_SEED is set)
        rng.randint(0, 1000000)
        samples = (sample for shard in shards for sample in read_shard(shard))
        stream = shuffle_buffer(samples, random.Random(epoch))
        yield from length_groups((self.processor(s, rng, generator) for s in stream), self.recipe)

    def _start(self) -> Iterator[dict[str, torch.Tensor]]:
        import torch

        order = list(self.shards)
        random.Random(self.epoch).shuffle(order)
        base = int(torch.empty((), dtype=torch.int64).random_(generator=self.main).item())
        streams = [self._worker(order[w :: self.workers], self.epoch, (base + w) % 2**32) for w in range(self.workers)]
        return self._interleave(streams)

    def _interleave(self, streams: list[Iterator[list[dict[str, Any]]]]) -> Iterator[dict[str, torch.Tensor]]:
        active = list(streams)
        while active:
            for stream in list(active):
                group = next(stream, None)
                if group is None:
                    active.remove(stream)
                    continue
                yield collate(group, self.pad)

    def next(self) -> dict[str, torch.Tensor]:
        batch = next(self._batches, None)
        if batch is None:
            self.epoch += 1
            self._batches = self._start()
            batch = next(self._batches, None)
            if batch is None:
                raise EngineError(f"No clip fits a training sample of {self.recipe.min_sample_tokens}–{self.recipe.max_sample_tokens} tokens", code=codes.INVALID_INPUT)
        return batch


def pad_id(model_dir: Path, tokenizer: Any) -> int:
    """The tokenizer's pad token (else its end-of-sequence token, as upstream's builder falls back)."""
    try:
        config = json.loads((model_dir / "tokenizer_config.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        config = {}
    for key in ("pad_token", "eos_token"):
        token = config.get(key)
        if isinstance(token, dict):
            token = token.get("content")
        if token:
            try:
                return int(tokenizer.token_to_id(token))
            except KeyError:
                continue
    return 0


# -- the model ------------------------------------------------------------------------------------


def _load_model(model_dir: Path, config: dict[str, Any]) -> OmniVoiceModel:
    """The base model in float32 (upstream trains from ``from_pretrained(dtype=float32)``), parameters as ``nn.Module``
    makes them."""
    from mizuoto.engines.omnivoice.model import OmniVoiceModel
    from mizuoto.runtime import ops
    from mizuoto.runtime.loading import load_into, load_state_dict

    model = OmniVoiceModel(config, ops.default)
    load_into(model, load_state_dict(model_dir / "model.safetensors"), what="OmniVoice model")
    return model.float()


def _upstream_order(model: nn.Module) -> list[torch.nn.Parameter]:
    """The parameters in upstream's order (its LM registers the token embedding first), for gradient clipping, whose
    total norm sums the per-tensor norms in this order."""
    llm = model.get_submodule("llm")
    embed = list(llm.get_submodule("embed_tokens").parameters())
    seen = {id(p) for p in embed}
    return [*embed, *(p for p in model.parameters() if id(p) not in seen)]


def _lora_class() -> type:
    """The LoRA layer's class (made here: this module imports no torch until a stage runs)."""
    import torch
    from torch import nn

    class _LoraLinear(nn.Module):
        """peft's ``lora.Linear`` with one adapter: ``base(x) + B(A(dropout(x))) * alpha / r``, ``A`` initialised as
        ``nn.Linear`` (Kaiming uniform), ``B`` zero. The dropout draws its masks from ``generator`` as ``F.dropout``
        draws them on the CPU (Bernoulli noise scaled by 1 / (1 - p))."""

        def __init__(self, base: nn.Linear, r: int, alpha: int, dropout: float, generator: torch.Generator) -> None:
            super().__init__()
            self.base_layer = base
            # peft makes both layers as nn.Linear (each drawing its initialisation), then initialises A again
            self.lora_A = nn.Linear(base.in_features, r, bias=False)
            self.lora_B = nn.Linear(r, base.out_features, bias=False)
            nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B.weight)
            self.lora_A.to(base.weight.device)
            self.lora_B.to(base.weight.device)
            self.scaling = alpha / r
            self.p = dropout
            self.generator = generator

        def _dropout(self, x: torch.Tensor) -> torch.Tensor:
            if not self.training or self.p <= 0:
                return x
            noise = torch.empty_like(x).bernoulli_(1 - self.p, generator=self.generator)
            noise.div_(1 - self.p)
            return x * noise

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            result = self.base_layer(x)
            dtype = result.dtype
            x = x.to(self.lora_A.weight.dtype)
            result = result + self.lora_B(self.lora_A(self._dropout(x))) * self.scaling
            return result.to(dtype)

    return _LoraLinear


def add_lora(model: nn.Module, recipe: Recipe, generator: torch.Generator) -> None:
    """``get_peft_model``: every target projection wrapped in module order (peft's, whose initialisation draws come in
    this order), the rest of the model frozen but for the audio embeddings and heads."""
    cls = _lora_class()
    targets = [name for name, _ in model.named_modules() if name.rsplit(".", 1)[-1] in TARGETS]
    for p in model.parameters():
        p.requires_grad_(False)
    for name in targets:
        parent_name, _, leaf = name.rpartition(".")
        parent = model.get_submodule(parent_name)
        setattr(parent, leaf, cls(getattr(parent, leaf), recipe.lora_r, recipe.lora_alpha, recipe.lora_dropout, generator))
    for name in MODULES_TO_SAVE:
        for p in model.get_submodule(name).parameters():
            p.requires_grad_(True)


def lora_state(model: nn.Module) -> dict[str, torch.Tensor]:
    """The adapter as peft saves it: ``base_model.model.<module>.lora_A.weight``/``lora_B.weight`` and the modules
    trained in full as ``base_model.model.<module>.weight``."""
    out = {}
    for name, module in model.named_modules():
        if hasattr(module, "lora_A") and hasattr(module, "base_layer"):
            out[f"base_model.model.{name}.lora_A.weight"] = module.lora_A.weight.detach().contiguous().cpu()
            out[f"base_model.model.{name}.lora_B.weight"] = module.lora_B.weight.detach().contiguous().cpu()
    for name in MODULES_TO_SAVE:
        for key, t in model.get_submodule(name).state_dict().items():
            out[f"base_model.model.{name}.{key}"] = t.detach().contiguous().cpu()
    return out


def adapter_config(recipe: Recipe, model_dir: Path) -> dict[str, Any]:
    """peft 0.21's ``LoraConfig`` as ``save_pretrained`` writes it."""
    return {
        "alora_invocation_tokens": None,
        "alpha_pattern": {},
        "arrow_config": None,
        "auto_mapping": {"base_model_class": "OmniVoice", "parent_library": "omnivoice.models.omnivoice"},
        "base_model_name_or_path": str(model_dir),
        "bias": "none",
        "corda_config": None,
        "ensure_weight_tying": False,
        "eva_config": None,
        "exclude_modules": None,
        "fan_in_fan_out": False,
        "inference_mode": True,
        "init_lora_weights": True,
        "kasa_config": None,
        "layer_replication": None,
        "layers_pattern": None,
        "layers_to_transform": None,
        "loftq_config": {},
        "lora_alpha": recipe.lora_alpha,
        "lora_bias": False,
        "lora_dropout": recipe.lora_dropout,
        "lora_ga_config": None,
        "megatron_config": None,
        "megatron_core": "megatron.core",
        "modules_to_save": list(MODULES_TO_SAVE),
        "monteclora_config": None,
        "peft_type": "LORA",
        "peft_version": "0.21.2",
        "qalora_group_size": 16,
        "r": recipe.lora_r,
        "rank_pattern": {},
        "revision": None,
        "target_modules": list(TARGETS),
        "target_parameters": None,
        "task_type": None,
        "trainable_token_indices": None,
        "use_bdlora": None,
        "use_dora": False,
        "use_qalora": False,
        "use_rslora": False,
        "velora_config": None,
    }


def model_state(model: nn.Module) -> dict[str, torch.Tensor]:
    return {k: v.detach().contiguous().cpu() for k, v in model.state_dict().items()}


def save_checkpoint(model: nn.Module, recipe: Recipe, model_dir: Path, folder: Path) -> None:
    """``checkpoint-<step>`` as upstream's ``save_checkpoint`` leaves it for ``merge_lora``: the adapter (LoRA) or the
    model (full) with upstream's key names, the text tokenizer and the training config. Optimizer state is not kept:
    Mizuoto does not resume a run."""
    from safetensors.torch import save_file

    folder.mkdir(parents=True, exist_ok=True)
    if recipe.lora:
        save_file(lora_state(model), str(folder / "adapter_model.safetensors"), metadata={"format": "pt"})
        (folder / "adapter_config.json").write_text(json.dumps(adapter_config(recipe, model_dir), indent=2), encoding="utf-8")
    else:
        save_file(model_state(model), str(folder / "model.safetensors"), metadata={"format": "pt"})
        shutil.copyfile(model_dir / "config.json", folder / "config.json")
    for name in TOKENIZER_FILES:
        if (model_dir / name).is_file():
            shutil.copyfile(model_dir / name, folder / name)
    config = {k: v for k, v in recipe.__dict__.items()}
    config["init_from_checkpoint"] = str(model_dir)
    (folder / "train_config.json").write_text(json.dumps(config, indent=4), encoding="utf-8")


def merge_lora(state: dict[str, torch.Tensor], checkpoint: Path) -> dict[str, torch.Tensor]:
    """``PeftModel.from_pretrained(base, checkpoint).merge_and_unload()`` on the base model's state dict: each target
    weight plus ``(B @ A) * alpha / r``, the modules trained in full replaced."""
    from mizuoto.runtime.loading import load_state_dict

    config = json.loads((checkpoint / "adapter_config.json").read_text(encoding="utf-8"))
    scaling = float(config["lora_alpha"]) / float(config["r"])
    adapter = dict(load_state_dict(checkpoint / "adapter_model.safetensors", mmap=False))
    prefix = "base_model.model."
    for key in [k for k in adapter if k.endswith(".lora_A.weight")]:
        module = key[len(prefix) : -len(".lora_A.weight")]
        a = adapter.pop(key)
        b = adapter.pop(f"{prefix}{module}.lora_B.weight")
        weight = f"{module}.weight"
        if weight not in state:
            raise EngineError(f"The adapter names {module}, which the base model lacks", code=codes.INVALID_INPUT)
        state[weight] = state[weight] + (b @ a) * scaling
    for key, value in adapter.items():
        name = key[len(prefix) :]
        if name not in state:
            raise EngineError(f"The adapter names {name}, which the base model lacks", code=codes.INVALID_INPUT)
        state[name] = value
    return {k: v.contiguous() for k, v in state.items()}


def forward_loss(model: OmniVoiceModel, batch: dict[str, torch.Tensor], weights: list[float]) -> torch.Tensor:
    """``OmniVoice.forward`` with labels: each position's text embedding or its codebooks' summed embeddings, the LM
    under the batch's mask and positions, every codebook's logits, and the cross-entropy over the masked codes,
    averaged per codebook and weighted."""
    import torch
    import torch.nn.functional as F

    input_ids, audio_mask = batch["input_ids"], batch["audio_mask"]
    text = model.llm.get_submodule("embed_tokens")(input_ids[:, 0, :])
    shifted = (input_ids * audio_mask.unsqueeze(1)) + model.get_buffer("codebook_layer_offsets").view(1, -1, 1)
    audio = model.audio_embeddings(shifted).sum(dim=1)
    x = torch.where(audio_mask.unsqueeze(-1), audio, text)
    hidden = model.llm(x, batch["position_ids"], batch["attention_mask"])
    b, length, _ = hidden.shape
    logits = model.audio_heads(hidden).view(b, length, model.num_codebooks, model.vocab).permute(0, 2, 1, 3)
    labels = batch["labels"]
    per_token = F.cross_entropy(logits.permute(0, 3, 1, 2), labels, reduction="none", ignore_index=IGNORE)
    valid = (labels != IGNORE).float()
    layer_means = (per_token * valid).sum(dim=(0, 2)) / valid.sum(dim=(0, 2)).clamp(min=1.0)
    return (layer_means * torch.tensor(weights, device=logits.device)).sum()


def cosine_schedule(optimizer: torch.optim.Optimizer, warmup: int, total: int) -> Any:
    """transformers' ``get_cosine_schedule_with_warmup`` (half a cycle)."""
    from torch.optim.lr_scheduler import LambdaLR

    def factor(step: int) -> float:
        if step < warmup:
            return float(step) / float(max(1, warmup))
        progress = float(step - warmup) / float(max(1, total - warmup))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * 0.5 * 2.0 * progress)))

    return LambdaLR(optimizer, factor)
