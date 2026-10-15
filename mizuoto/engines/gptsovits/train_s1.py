"""GPT-SoVITS's GPT (s1) training, native: upstream's ``s1_train.py`` (MIT, (c) 2024 RVC-Boss) as v1's trainer runs it,
without PyTorch Lightning, torchmetrics or pandas.

- The data: ``AR/data/dataset.py``'s ``Text2SemanticDataset`` (``2-name2text.txt``, ``6-name2semantic.tsv``, ``3-bert/``;
  its length and phones-per-second filters; a dataset under 100 clips repeated ``max(2, 100 // n)`` times) and
  ``bucket_sampler.py``'s ``DistributedBucketSampler`` for one replica: 2-second buckets of the sorted lengths, each
  bucket and then the batches shuffled with Python's ``random`` seeded ``epoch`` (a private ``Random`` here).
- The step: ``Text2SemanticLightningModule.training_step`` under manual optimisation: the loss backpropagated every
  batch, ``ScaledAdam`` stepped (then zeroed, then the scheduler stepped) at every batch index > 0 divisible by 4, so
  gradients pile up over five batches first, and over an epoch's end. ``WarmupCosineLRSchedule`` is locked to 0.002
  after its first step, and the optimiser runs its first step at ScaledAdam's own lr, 0.01.
- The model: ``Text2SemanticDecoder.forward_old`` (or, with DPO, ``forward``: the cross-entropy plus the reference-free
  DPO loss against a corrupted copy of each target), computed with the ops upstream's training path calls in its order
  and layout (``multi_head_attention_forward_patched``: packed in-projection, a float mask, SDPA with dropout), every
  dropout 0.1 as upstream's modules apply them, on the process-wide generator. The summed cross-entropy and top-3
  accuracy (torchmetrics' micro ``MulticlassAccuracy`` ignoring EOS) are logged per batch.
- The saving: ``my_model_ckpt``'s half weights, ``{"weight": "model.*" in half precision, "config", "info": "GPT-eN"}``
  every ``save_every_n_epoch`` epochs, to ``<half_weights_save_dir>/<exp_name>-e<N>.ckpt``. Lightning's own resumable
  checkpoint is not written (nothing resumes from it).

Draws, as upstream makes them after ``seed_everything(1234)`` (the stage's seed): the model's construction (one
encoder layer, which upstream deep-copies, then the head), one ``random_`` for the data loader's base seed, then per
batch the dropout masks (and with DPO the corruption's ``randint`` first).
"""

from __future__ import annotations

import math
import random
from collections import OrderedDict, defaultdict
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from mizuoto.engines.gptsovits.t2s import SITE, EncoderLayer, SinePositionalEmbedding, T2SConfig, Text2SemanticDecoder
from mizuoto.runtime.accel import attention

S1_CONFIG: dict[str, Any] = {
    "train": {"seed": 1234, "epochs": 20, "batch_size": 8, "save_every_n_epoch": 1, "precision": "16-mixed", "gradient_clip": 1.0},
    "optimizer": {"lr": 0.01, "lr_init": 0.00001, "lr_end": 0.0001, "warmup_steps": 2000, "decay_steps": 40000},
    "data": {"max_eval_sample": 8, "max_sec": 54, "num_workers": 4, "pad_val": 1024},
    "model": {
        "vocab_size": 1025,
        "phoneme_vocab_size": 732,
        "embedding_dim": 512,
        "hidden_dim": 512,
        "head": 16,
        "linear_units": 2048,
        "n_layer": 24,
        "dropout": 0,
        "EOS": 1024,
        "random_bert": 0,
    },
    "inference": {"top_k": 15},
}
"""``configs/s1longer-v2.yaml``, the config v1's trainer starts from."""

HZ = 25
"""``Text2SemanticDataset.hz``: v1 runs s1_train with ``hz=25hz``."""
MIN_SAMPLES = 100
"""``init_batch``'s ``min_num``: a smaller dataset is repeated."""
DROPOUT = 0.1
"""The encoder layers' and the positional embeddings' dropout (upstream hard-codes it)."""
TOP_K = 3
"""``Text2SemanticLightningModule.top_k``: the logged accuracy is top-3."""
STEP_EVERY = 4
"""``training_step`` steps the optimiser at batch indices > 0 divisible by this."""


# -- the data ---------------------------------------------------------------------------------------------------


def read_semantic(path: Path) -> list[tuple[str, str]]:
    """``pd.read_csv(path, delimiter="\\t")``'s two columns: the header skipped, blank lines skipped."""
    lines = [line for line in path.read_text(encoding="utf-8").split("\n") if line.strip()]
    rows = []
    for line in lines[1:]:
        name, _, semantic = line.partition("\t")
        rows.append((name, semantic.split("\t", 1)[0]))
    return rows


@dataclass
class Item:
    semantic: list[int]
    phonemes: list[int]
    name: str


class Text2SemanticDataset:
    """``AR/data/dataset.py``'s dataset: (semantic ids, phone ids) per clip, BERT features read per item."""

    def __init__(self, phoneme_path: Path, semantic_path: Path, max_sec: int = 54, pad_val: int = 1024, min_ps_ratio: int = 3, max_ps_ratio: int = 25) -> None:
        from mizuoto.engines.gptsovits.text.symbols import cleaned_text_to_sequence

        self.pad = pad_val
        self.bert_dir = phoneme_path.parent / "3-bert"
        phonemes: dict[str, tuple[str, str, str]] = {}
        for line in phoneme_path.read_text(encoding="utf-8").strip("\n").split("\n"):
            parts = line.split("\t")
            if len(parts) == 4:
                phonemes[parts[0]] = (parts[1], parts[2], parts[3])
        items: list[Item] = []
        self.skipped = {"not_in_text": 0, "too_long": 0, "phone_rate": 0}
        for name, semantic_str in read_semantic(semantic_path):
            if name not in phonemes:
                self.skipped["not_in_text"] += 1
                continue
            semantic = [int(i) for i in semantic_str.split(" ")]
            if len(semantic) > max_sec * HZ:
                self.skipped["too_long"] += 1
                continue
            try:
                phone_ids = cleaned_text_to_sequence(phonemes[name][0].split(" "))
            except KeyError:
                self.skipped["not_in_text"] += 1
                continue
            if len(phone_ids) > max_sec * HZ / 2.5:
                self.skipped["phone_rate"] += 1
                continue
            ratio = len(phone_ids) / (len(semantic) / HZ)
            if ratio > max_ps_ratio or ratio < min_ps_ratio:
                self.skipped["phone_rate"] += 1
                continue
            items.append(Item(semantic, phone_ids, name))
        self.clips = len(items)
        if 0 < len(items) < MIN_SAMPLES:
            items = items * max(2, int(MIN_SAMPLES / len(items)))
        self.items = items

    def __len__(self) -> int:
        return len(self.items)

    def sample_length(self, idx: int) -> float:
        return 1.0 * len(self.items[idx].semantic) / HZ

    def __getitem__(self, idx: int) -> dict[str, Any]:
        item = self.items[idx]
        path = self.bert_dir / f"{item.name}.pt"
        bert = torch.load(path, map_location="cpu", weights_only=True) if path.exists() else None
        if bert is not None and bert.shape[-1] != len(item.phonemes):
            raise ValueError(f"The BERT features of {item.name} ({bert.shape[-1]}) do not match its {len(item.phonemes)} phones")
        return {"idx": idx, "phoneme_ids": item.phonemes, "semantic_ids": item.semantic, "bert_feature": bert}

    def collate(self, examples: list[dict[str, Any]]) -> dict[str, Any]:
        """``collate``: both sequences padded at the end (phones with 0, tokens with the pad value), BERT features zero-padded."""
        phone_lens = [len(e["phoneme_ids"]) for e in examples]
        semantic_lens = [len(e["semantic_ids"]) for e in examples]
        phones = torch.zeros(len(examples), max(phone_lens), dtype=torch.int64)
        semantic = torch.full((len(examples), max(semantic_lens)), self.pad, dtype=torch.int64)
        bert = torch.zeros(len(examples), 1024, max(phone_lens), dtype=torch.float32)
        for i, e in enumerate(examples):
            phones[i, : phone_lens[i]] = torch.tensor(e["phoneme_ids"], dtype=torch.int64)
            semantic[i, : semantic_lens[i]] = torch.tensor(e["semantic_ids"], dtype=torch.int64)
            if e["bert_feature"] is not None:
                bert[i, :, : e["bert_feature"].shape[-1]] = e["bert_feature"]
        return {
            "ids": [e["idx"] for e in examples],
            "phoneme_ids": phones,
            "phoneme_ids_len": torch.tensor(phone_lens),
            "semantic_ids": semantic,
            "semantic_ids_len": torch.tensor(semantic_lens),
            "bert_feature": bert,
        }


class BucketSampler:
    """``DistributedBucketSampler`` with one replica: the order of the dataset's indices for an epoch."""

    def __init__(self, dataset: Text2SemanticDataset, batch_size: int, seed: int = 0) -> None:
        self.batch_size = batch_size
        self.seed = seed
        lengths = sorted(((i, dataset.sample_length(i)) for i in range(len(dataset))), key=lambda x: x[1])
        self.buckets: list[list[int]] = []
        current: list[int] = []
        max_sec = 2.0
        for idx, sec in lengths:
            if sec < max_sec:
                current.append(idx)
            else:
                self.buckets.append(current)
                current = [idx]
                max_sec += 2.0
        if current:
            self.buckets.append(current)
        self.size = len(dataset)

    def indices(self, epoch: int) -> list[int]:
        rng = random.Random(epoch + self.seed)  # upstream reseeds the global ``random`` with this
        shuffled: list[int] = []
        for bucket in self.buckets:
            copy = bucket.copy()
            rng.shuffle(copy)
            shuffled.extend(copy)
        n_batch = int(math.ceil(len(shuffled) / self.batch_size))
        batches = [shuffled[b * self.batch_size : (b + 1) * self.batch_size] for b in range(n_batch)]
        rng.shuffle(batches)
        return [i for b in batches for i in b]

    def batches(self, epoch: int) -> Iterator[list[int]]:
        """``BatchSampler(sampler, batch_size, drop_last=False)``."""
        order = self.indices(epoch)
        for start in range(0, len(order), self.batch_size):
            yield order[start : start + self.batch_size]


# -- the optimiser and its schedule --------------------------------------------------------------------------------


class WarmupCosineLRSchedule:
    """``AR/modules/lr_schedulers.py`` as upstream left it: whatever the step, the lr becomes (and stays) 0.002."""

    def __init__(self, optimizer: torch.optim.Optimizer, init_lr: float, peak_lr: float, end_lr: float, warmup_steps: int = 10000, total_steps: int = 400000) -> None:
        self.optimizer = optimizer
        self.init_lr, self.peak_lr, self.end_lr = init_lr, peak_lr, end_lr
        self.warmup_steps, self.total_steps = warmup_steps, total_steps
        self._warmup_rate = (peak_lr - init_lr) / warmup_steps
        self._current_step = 0
        self.lr = init_lr
        self._last_lr = [init_lr]

    def get_last_lr(self) -> list[float]:
        return self._last_lr

    def step(self) -> float:
        if self._current_step < self.warmup_steps:
            lr = self.init_lr + self._warmup_rate * self._current_step
        elif self._current_step > self.total_steps:
            lr = self.end_lr
        else:
            ratio = (self._current_step - self.warmup_steps) / (self.total_steps - self.warmup_steps)
            lr = self.end_lr + 0.5 * (1.0 + math.cos(math.pi * ratio)) * (self.peak_lr - self.end_lr)
        del lr  # upstream computes it, then locks the rate
        self.lr = self.end_lr = 0.002
        self._last_lr = [g["lr"] for g in self.optimizer.param_groups]
        for g in self.optimizer.param_groups:
            g["lr"] = self.end_lr
        self._current_step += 1
        return self.lr


class ScaledAdam(torch.optim.Optimizer):
    """``AR/modules/optim.py``'s ``ScaledAdam`` (from icefall): parameters of one dtype and shape stacked and stepped together,
    each update scaled by the parameter's rms, the scale itself learnt every ``size_update_period`` steps, gradients
    clipped to ``clipping_scale`` × the median norm of the last ``clipping_update_period`` steps."""

    def __init__(
        self,
        params: Any,
        lr: float = 3e-02,
        clipping_scale: float | None = None,
        betas: tuple[float, float] = (0.9, 0.98),
        scalar_lr_scale: float = 0.1,
        eps: float = 1.0e-08,
        param_min_rms: float = 1.0e-05,
        param_max_rms: float = 3.0,
        scalar_max: float = 10.0,
        size_update_period: int = 4,
        clipping_update_period: int = 100,
    ) -> None:
        defaults = {
            "lr": lr,
            "clipping_scale": clipping_scale,
            "betas": betas,
            "scalar_lr_scale": scalar_lr_scale,
            "eps": eps,
            "param_min_rms": param_min_rms,
            "param_max_rms": param_max_rms,
            "scalar_max": scalar_max,
            "size_update_period": size_update_period,
            "clipping_update_period": clipping_update_period,
        }
        super().__init__(params, defaults)

    def _batches(self, params: list[torch.Tensor]) -> list[tuple[list[torch.Tensor], torch.Tensor, dict[str, Any]]]:
        """``batched_params``: (the parameters, their stack with the stacked grads, the first one's state), by sorted key."""
        groups: dict[tuple[Any, ...], list[torch.Tensor]] = defaultdict(list)
        for p in params:
            groups[(str(p.dtype), *p.shape)].append(p)
        keys = list(groups)
        out = []
        for key in sorted(keys):
            batch = groups[key]
            stacked = torch.stack(batch)
            stacked.grad = torch.stack([torch.zeros_like(p) if p.grad is None else p.grad for p in batch])
            out.append((batch, stacked, self.state[batch[0]]))
        return out

    @torch.no_grad()
    def step(self, closure: Callable[[], Any] | None = None) -> Any:
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            batches = self._batches(group["params"])
            clipping_scale = 1.0 if len(batches[0][2]) == 0 else self._clipping_scale(group, batches)
            for _, p, state in batches:
                if len(state) == 0:
                    self._init_state(group, p, state)
                self._step_one_batch(group, p, state, clipping_scale)
            for batch, stacked, _ in batches:
                for i, p in enumerate(batch):
                    p.copy_(stacked[i])
        return loss

    @staticmethod
    def _init_state(group: dict[str, Any], p: torch.Tensor, state: dict[str, Any]) -> None:
        state["step"] = 0
        state["delta"] = torch.zeros_like(p, memory_format=torch.preserve_format)
        if p.numel() > 1:  # upstream's test: the whole stack, so stacked scalars get these too (unused)
            param_rms = (p**2).mean(dim=list(range(1, p.ndim)), keepdim=True).sqrt()
            state["param_rms"] = param_rms
            state["scale_exp_avg_sq"] = torch.zeros_like(param_rms)
            state["scale_grads"] = torch.zeros(group["size_update_period"], *param_rms.shape, device=p.device, dtype=p.dtype)
        state["exp_avg_sq"] = torch.zeros_like(p, memory_format=torch.preserve_format)

    @staticmethod
    def _clipping_scale(group: dict[str, Any], batches: list[tuple[list[torch.Tensor], torch.Tensor, dict[str, Any]]]) -> float:
        clipping_scale = group["clipping_scale"]
        first_p, first_state = batches[0][1], batches[0][2]
        step = first_state["step"]
        if clipping_scale is None or step == 0:
            return 1.0
        period = group["clipping_update_period"]
        tot_sumsq = torch.tensor(0.0, device=first_p.device)
        p = first_p
        for _, p, state in batches:
            grad = p.grad
            assert grad is not None
            if p.numel() == p.shape[0]:
                tot_sumsq += (grad**2).sum()
            else:
                tot_sumsq += ((grad * state["param_rms"]) ** 2).sum()
        tot_norm = tot_sumsq.sqrt()
        if "model_norms" not in first_state:
            first_state["model_norms"] = torch.zeros(period, device=p.device)
        first_state["model_norms"][step % period] = tot_norm
        if step % period == 0:
            sorted_norms = first_state["model_norms"].sort()[0].to("cpu")
            quartiles = [sorted_norms[min(period - 1, (period // 4) * n)].item() for n in range(5)]
            first_state["model_norm_threshold"] = clipping_scale * quartiles[2]
            first_state["num_clipped"] = 0
        if step < period:
            return 1.0
        threshold = first_state.get("model_norm_threshold")
        if threshold is None:
            return 1.0
        ans = min(1.0, (threshold / (tot_norm + 1.0e-20)).item())
        if ans < 1.0:
            first_state["num_clipped"] += 1
        return ans

    def _step_one_batch(self, group: dict[str, Any], p: torch.Tensor, state: dict[str, Any], clipping_scale: float) -> None:
        size_update_period = group["size_update_period"]
        beta1 = group["betas"][0]
        grad = p.grad
        assert grad is not None
        if clipping_scale != 1.0:
            grad = grad * clipping_scale
        step = state["step"]
        state["delta"].mul_(beta1)
        numel = p.numel() // p.shape[0]
        if numel > 1:
            scale_grads = state["scale_grads"]
            scale_grads[step % size_update_period] = (p * grad).sum(dim=list(range(1, p.ndim)), keepdim=True)
            if step % size_update_period == size_update_period - 1:
                state["param_rms"].copy_((p**2).mean(dim=list(range(1, p.ndim)), keepdim=True).sqrt())
                if step > 0:
                    self._size_update(group, scale_grads, p, state)
        if numel == 1:
            self._step_scalar(group, p, state)
        else:
            self._step(group, p, state)
        state["step"] = step + 1

    @staticmethod
    def _size_update(group: dict[str, Any], scale_grads: torch.Tensor, p: torch.Tensor, state: dict[str, Any]) -> None:
        param_rms = state["param_rms"]
        beta1, beta2 = group["betas"]
        size_lr = group["lr"] * group["scalar_lr_scale"]
        size_update_period = scale_grads.shape[0]
        beta2_corr = beta2**size_update_period
        scale_exp_avg_sq = state["scale_exp_avg_sq"]
        scale_exp_avg_sq.mul_(beta2_corr).add_((scale_grads**2).mean(dim=0), alpha=1 - beta2_corr)
        size_step = (state["step"] + 1) // size_update_period
        bias_correction2 = 1 - beta2_corr**size_step
        denom = scale_exp_avg_sq.sqrt() + group["eps"]
        scale_step = -size_lr * (bias_correction2**0.5) * scale_grads.sum(dim=0) / denom
        scale_step.masked_fill_(param_rms < group["param_min_rms"], 0.0)
        scale_step.masked_fill_(param_rms > group["param_max_rms"], -size_lr * size_update_period)
        state["delta"].add_(p * scale_step, alpha=(1 - beta1))

    @staticmethod
    def _step(group: dict[str, Any], p: torch.Tensor, state: dict[str, Any]) -> None:
        grad = p.grad
        assert grad is not None
        lr = group["lr"]
        beta1, beta2 = group["betas"]
        exp_avg_sq = state["exp_avg_sq"]
        exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=(1 - beta2))
        this_step = state["step"] - state.get("zero_step", 0)
        bias_correction2 = 1 - beta2 ** (this_step + 1)
        if bias_correction2 < 0.99:
            exp_avg_sq = exp_avg_sq * (1.0 / bias_correction2)
        denom = exp_avg_sq.sqrt()
        denom += group["eps"]
        grad = grad / denom
        alpha = -lr * (1 - beta1) * state["param_rms"].clamp(min=group["param_min_rms"])
        delta = state["delta"]
        delta.add_(grad * alpha)
        p.add_(delta)

    @staticmethod
    def _step_scalar(group: dict[str, Any], p: torch.Tensor, state: dict[str, Any]) -> None:
        beta1, beta2 = group["betas"]
        lr = group["lr"] * group["scalar_lr_scale"]
        grad = p.grad
        assert grad is not None
        exp_avg_sq = state["exp_avg_sq"]
        exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
        bias_correction2 = 1 - beta2 ** (state["step"] + 1)
        denom = (exp_avg_sq / bias_correction2).sqrt() + group["eps"]
        delta = state["delta"]
        delta.add_(grad / denom, alpha=-lr * (1 - beta1))
        p.clamp_(min=-group["scalar_max"], max=group["scalar_max"])
        p.add_(delta)


# -- the training forward -----------------------------------------------------------------------------------------


def replay_construction_draws(model: dict[str, Any]) -> None:
    """The random initialisation upstream's ``Text2SemanticDecoder`` draws while it is built (its weights are then replaced
    by the pretrained ones), so the dropout masks after it come from the same place in the stream. One encoder layer is
    initialised: ``TransformerEncoder`` deep-copies it."""
    dim, vocab, phones = int(model["embedding_dim"]), int(model["vocab_size"]), int(model["phoneme_vocab_size"])
    hidden = int(model["hidden_dim"])
    nn.Linear(1024, dim)  # bert_proj
    nn.Embedding(phones, dim)  # ar_text_embedding
    nn.Embedding(vocab, dim)  # ar_audio_embedding
    nn.Linear(hidden, hidden)  # the layer's self_attn.out_proj, then its _reset_parameters:
    nn.init.xavier_uniform_(torch.empty(3 * hidden, hidden))  # in_proj_weight
    nn.Linear(hidden, hidden * 4)  # linear1
    nn.Linear(hidden * 4, hidden)  # linear2
    nn.Linear(hidden, vocab, bias=False)  # ar_predict_layer


def make_pad_mask(lengths: torch.Tensor, max_len: int = 0) -> torch.Tensor:
    max_len = max(max_len, int(lengths.max()))
    seq = torch.arange(0, max_len, device=lengths.device)
    return seq.unsqueeze(0).expand(lengths.size(0), max_len) >= lengths.unsqueeze(-1)


def make_pad_mask_left(lengths: torch.Tensor, max_len: int = 0) -> torch.Tensor:
    max_len = max(max_len, int(lengths.max()))
    seq = torch.arange(0, max_len, device=lengths.device).unsqueeze(0).repeat(lengths.size(0), 1)
    seq -= (max_len - lengths).unsqueeze(-1)
    return seq < 0


def make_reject_y(y_o: torch.Tensor, y_lens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """``AR/models/utils.make_reject_y``: per row, a span repeated or dropped (over the padded row, as upstream)."""

    def span(y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        idx, _ = torch.randint(0, len(y), size=(2,)).sort()
        return y[: idx[0]], y[idx[0] : idx[1]], y[idx[1] :]

    rows: list[torch.Tensor] = []
    for b in range(len(y_lens)):
        kind = torch.randint(0, 2, size=(1,))[0]
        pre, mid, post = span(y_o[b])
        rows.append(torch.cat([pre, mid, mid, post]) if kind == 0 else torch.cat([pre, post]))
    lens = [len(r) for r in rows]
    longest = max(lens)
    rows = [torch.cat([r, torch.zeros(longest - len(r), dtype=y_o.dtype, device=y_o.device)], dim=0) for r in rows]
    return torch.stack(rows, dim=0), torch.tensor(lens, device=y_lens.device)


def top_k_accuracy(logits: torch.Tensor, targets: torch.Tensor, ignore_index: int, k: int = TOP_K) -> float:
    """torchmetrics' ``MulticlassAccuracy(top_k=k, average="micro", multidim_average="global", ignore_index=…)`` on (batch,
    classes, time) logits: the share of the targets not ignored that are among the k highest logits."""
    top = logits.topk(k, dim=1).indices
    valid = targets != ignore_index
    hit = (top == targets.unsqueeze(1)).any(dim=1) & valid
    count = valid.sum().float()
    return float(torch.where(count != 0, hit.sum().float() / count, torch.tensor(0.0, device=count.device)).item())


class T2STraining:
    """Upstream's training forward over the native ``Text2SemanticDecoder``'s parameters."""

    def __init__(self, model: Text2SemanticDecoder, config: dict[str, Any]) -> None:
        self.model = model
        self.heads = int(config["model"]["head"])
        self.eos = int(config["model"]["EOS"])
        self.embedding_dropout = float(config["model"]["dropout"])

    @property
    def training(self) -> bool:
        return self.model.training

    def _position(self, module: SinePositionalEmbedding, x: torch.Tensor) -> torch.Tensor:
        pe = module.table(x.size(1), x)
        out = x.unsqueeze(-1) if x.ndim == 2 else x
        out = out * module.x_scale + module.alpha * pe[:, : x.size(1)]
        return F.dropout(out, DROPOUT, self.training)

    def _attention(self, layer: EncoderLayer, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """``MultiheadAttention.forward`` (batch first) → ``multi_head_attention_forward_patched`` without weights."""
        query = x.transpose(1, 0)
        tgt_len, bsz, dim = query.shape
        heads, head_dim = self.heads, dim // self.heads
        attn = layer.self_attn
        proj = F.linear(query, attn.in_proj_weight, attn.in_proj_bias)
        proj = proj.unflatten(-1, (3, dim)).unsqueeze(0).transpose(0, -2).squeeze(-2).contiguous()
        q, k, v = proj[0], proj[1], proj[2]
        q = q.view(tgt_len, bsz * heads, head_dim).transpose(0, 1)
        k = k.view(k.shape[0], bsz * heads, head_dim).transpose(0, 1)
        v = v.view(v.shape[0], bsz * heads, head_dim).transpose(0, 1)
        src_len = k.size(1)
        mask = mask.unsqueeze(0) if mask.size(0) == 1 and mask.dim() == 3 else mask.view(bsz, heads, -1, src_len)
        q = q.view(bsz, heads, tgt_len, head_dim)
        k = k.view(bsz, heads, src_len, head_dim)
        v = v.view(bsz, heads, src_len, head_dim)
        out = attention(q, k, v, site=SITE, mask=mask, dropout_p=DROPOUT if self.training else 0.0)
        out = out.permute(2, 0, 1, 3).contiguous().view(bsz * tgt_len, dim)
        out = F.linear(out, attn.out_proj.weight, attn.out_proj.bias)
        return out.view(tgt_len, bsz, out.size(1)).transpose(1, 0)

    def _layer(self, layer: EncoderLayer, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """``TransformerEncoderLayer.forward``, post-norm."""
        dim = [x.shape[-1]]
        x = F.layer_norm(x + F.dropout(self._attention(layer, x, mask), DROPOUT, self.training), dim, layer.norm1.weight, layer.norm1.bias, layer.norm1.eps)
        ff = F.linear(F.dropout(F.relu(F.linear(x, layer.linear1.weight, layer.linear1.bias)), DROPOUT, self.training), layer.linear2.weight, layer.linear2.bias)
        return F.layer_norm(x + F.dropout(ff, DROPOUT, self.training), dim, layer.norm2.weight, layer.norm2.bias, layer.norm2.eps)

    def _inputs(self, x: torch.Tensor, x_lens: torch.Tensor, y: torch.Tensor, y_lens: torch.Tensor, bert: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """``make_input_data``: [text][tokens] with their positions, the float mask and the targets (EOS-padded)."""
        m = self.model
        x = F.dropout(F.embedding(x, m.ar_text_embedding.word_embeddings.weight), self.embedding_dropout, self.training)
        x = x + F.linear(bert.transpose(1, 2), m.bert_proj.weight, m.bert_proj.bias)
        x = self._position(m.ar_text_position, x)
        x_mask = make_pad_mask_left(x_lens)
        y_mask = make_pad_mask(y_lens)
        y_mask_int = y_mask.type(torch.int64)
        codes = y.type(torch.int64) * (1 - y_mask_int)
        targets = F.pad(codes, (0, 1), value=0) + self.eos * F.pad(y_mask_int, (0, 1), value=1)
        y = targets[:, :-1]
        x_len, y_len = int(x_lens.max()), int(y_lens.max())
        y_emb = F.dropout(F.embedding(y, m.ar_audio_embedding.word_embeddings.weight), self.embedding_dropout, self.training)
        y_pos = self._position(m.ar_audio_position, y_emb)
        padding = torch.concat([x_mask, y_mask], dim=1)
        x_attn = F.pad(torch.zeros((x_len, x_len), dtype=torch.bool, device=x.device), (0, y_len), value=True)
        y_attn = F.pad(torch.triu(torch.ones(y_len, y_len, dtype=torch.bool, device=x.device), diagonal=1), (x_len, 0), value=False)
        mask = torch.concat([x_attn, y_attn], dim=0)
        bsz, src_len = x.shape[0], x_len + y_len
        padding = padding.view(bsz, 1, 1, src_len).expand(-1, self.heads, -1, -1).reshape(bsz * self.heads, 1, src_len)
        mask = mask.logical_or(padding)
        float_mask = torch.zeros_like(mask, dtype=x.dtype)
        float_mask.masked_fill_(mask, float("-inf"))
        return torch.concat([x, y_pos], dim=1), float_mask, targets

    def _logits(self, xy: torch.Tensor, mask: torch.Tensor, x_len: int) -> torch.Tensor:
        for layer in self.model.h.layers:
            xy = self._layer(layer, xy, mask)  # ty: ignore[invalid-argument-type]
        return F.linear(xy[:, x_len - 1 :], self.model.ar_predict_layer.weight)

    def forward_old(self, batch: dict[str, Any]) -> tuple[torch.Tensor, float]:
        """The cross-entropy (summed) and the top-3 accuracy."""
        x, x_lens, y, y_lens, bert = batch["phoneme_ids"], batch["phoneme_ids_len"], batch["semantic_ids"], batch["semantic_ids_len"], batch["bert_feature"]
        xy, mask, targets = self._inputs(x, x_lens, y, y_lens, bert)
        logits = self._logits(xy, mask, int(x_lens.max())).permute(0, 2, 1)
        loss = F.cross_entropy(logits, targets, reduction="sum")
        return loss, top_k_accuracy(logits.detach(), targets, self.eos)

    def forward_dpo(self, batch: dict[str, Any]) -> tuple[torch.Tensor, float]:
        """``forward``: the cross-entropy plus ``dpo_loss(beta=0.2, reference_free=True)`` against a corrupted target."""
        x, x_lens, y, y_lens, bert = batch["phoneme_ids"], batch["phoneme_ids_len"], batch["semantic_ids"], batch["semantic_ids_len"], batch["bert_feature"]
        reject_y, reject_lens = make_reject_y(y, y_lens)
        x_len = int(x_lens.max())
        xy, mask, targets = self._inputs(x, x_lens, y, y_lens, bert)
        logits = self._logits(xy, mask, x_len)
        rxy, rmask, reject_targets = self._inputs(x, x_lens, reject_y, reject_lens, bert)
        reject_logits = self._logits(rxy, rmask, x_len)
        loss_1 = F.cross_entropy(logits.permute(0, 2, 1), targets, reduction="sum")
        acc = top_k_accuracy(logits.permute(0, 2, 1).detach(), targets, self.eos)
        chosen = torch.gather(logits.log_softmax(-1), dim=2, index=targets.unsqueeze(2)).squeeze(2).sum(-1)
        rejected = torch.gather(reject_logits.log_softmax(-1), dim=2, index=reject_targets.unsqueeze(2)).squeeze(2).sum(-1)
        loss_2 = (-F.logsigmoid(0.2 * ((chosen - rejected) - 0))).mean()
        return loss_1 + loss_2, acc


# -- the run --------------------------------------------------------------------------------------------------------


@dataclass
class S1Run:
    """What v1's trainer writes into the config (``stage_gpt``)."""

    exp_dir: Path
    """Upstream's ``exp_dir``: ``2-name2text.txt``, ``3-bert/``, ``6-name2semantic.tsv``."""
    weights_dir: Path
    """``half_weights_save_dir``."""
    output_dir: Path
    """``output_dir`` (v1: ``<exp>/logs_s1_<version>``): only recorded in the config, nothing is written there."""
    pretrained: Path
    exp_name: str
    epochs: int = 15
    batch_size: int = 4
    save_every_n_epoch: int = 15
    dpo: bool = False
    precision: str = "32"
    """``32`` or ``16-mixed`` (CUDA)."""
    device: str = "cpu"


def build_config(run: S1Run, base_config: dict[str, Any]) -> dict[str, Any]:
    """The config upstream's run reads and saves in each checkpoint: ``s1longer-v2.yaml`` with v1's changes. The model's
    sizes are the base checkpoint's (the published ones equal the yaml's)."""
    import copy

    config = copy.deepcopy(S1_CONFIG)
    config["model"].update(base_config.get("model") or {})
    config["train"].update(
        {
            "precision": run.precision,
            "batch_size": run.batch_size,
            "epochs": run.epochs,
            "save_every_n_epoch": run.save_every_n_epoch,
            "if_save_every_weights": True,
            "if_save_latest": True,
            "if_dpo": run.dpo,
            "half_weights_save_dir": str(run.weights_dir),
            "exp_name": run.exp_name,
        }
    )
    config["pretrained_s1"] = str(run.pretrained)
    config["train_semantic_path"] = str(run.exp_dir / "6-name2semantic.tsv")
    config["train_phoneme_path"] = str(run.exp_dir / "2-name2text.txt")
    config["output_dir"] = str(run.output_dir)
    return config


def load_model(pretrained: Path, config: dict[str, Any], device: str) -> Text2SemanticDecoder:
    """``Text2SemanticLightningModule(config)`` with ``pretrained_s1`` loaded (its half weights into float32 parameters)."""
    from mizuoto.engines.gptsovits.checkpoints import load_checkpoint
    from mizuoto.engines.gptsovits.t2s import t2s_state
    from mizuoto.runtime.loading import load_into

    ckpt = load_checkpoint(pretrained)
    model = Text2SemanticDecoder(T2SConfig.from_checkpoint({"model": config["model"], "data": config["data"]}))
    load_into(model, t2s_state(ckpt["weight"]), what="GPT-SoVITS GPT model")
    return model.to(device)


def half_weights(model: Text2SemanticDecoder, config: dict[str, Any], epoch: int) -> dict[str, Any]:
    """``my_model_ckpt.on_train_epoch_end``'s ``to_save_od``."""
    weight: OrderedDict[str, torch.Tensor] = OrderedDict()
    for key, value in model.state_dict().items():
        weight[f"model.{key}"] = value.half().cpu()
    return OrderedDict([("weight", weight), ("config", config), ("info", f"GPT-e{epoch}")])


def train_gpt(
    run: S1Run,
    *,
    on_step: Callable[[int, int, int, dict[str, float], float], None] | None = None,
    check: Callable[[], None] | None = None,
) -> dict[str, Any]:
    """Train and save as upstream's ``s1_train.py``; ``on_step(epoch, step, batches, values, lr)`` after every batch (1-based
    epoch and step). Returns the saved weights and the run's counts."""
    from mizuoto.engines.gptsovits.checkpoints import load_checkpoint

    base_config = load_checkpoint(run.pretrained).get("config") or {}
    config = build_config(run, base_config)
    replay_construction_draws(config["model"])
    model = load_model(run.pretrained, config, run.device)
    trainer = T2STraining(model, config)
    data = Text2SemanticDataset(
        run.exp_dir / "2-name2text.txt", run.exp_dir / "6-name2semantic.tsv", max_sec=int(config["data"]["max_sec"]), pad_val=int(config["data"]["pad_val"])
    )
    if len(data) == 0:
        raise ValueError("No clip is fit for GPT training (all were filtered out by length or phones per second)")
    batch_size = config["train"]["batch_size"] // 2 if run.dpo else config["train"]["batch_size"]
    batch_size = max(min(batch_size, len(data) // 4), 1)
    sampler = BucketSampler(data, batch_size)
    optimizer = ScaledAdam(model.parameters(), lr=0.01, betas=(0.9, 0.95), clipping_scale=2.0, clipping_update_period=1000)
    opt = config["optimizer"]
    scheduler = WarmupCosineLRSchedule(optimizer, opt["lr_init"], opt["lr"], opt["lr_end"], opt["warmup_steps"], opt["decay_steps"])
    cuda = torch.device(run.device).type == "cuda"
    mixed = run.precision == "16-mixed" and cuda
    scaler = torch.amp.GradScaler("cuda") if mixed else None
    forward = trainer.forward_dpo if run.dpo else trainer.forward_old
    torch.empty((), dtype=torch.int64).random_()  # the data loader's base seed (its persistent workers draw nothing else here)
    model.train()
    batches = int(math.ceil(len(data) / batch_size))
    step = 0
    saved: list[Path] = []
    for epoch in range(run.epochs):
        for batch_idx, ids in enumerate(sampler.batches(epoch)):
            if check is not None:
                check()
            batch = data.collate([data[i] for i in ids])
            batch = {k: v.to(run.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            with torch.autocast("cuda", dtype=torch.float16, enabled=mixed):
                loss, acc = forward(batch)
            (scaler.scale(loss) if scaler is not None else loss).backward()
            if batch_idx > 0 and batch_idx % STEP_EVERY == 0:
                if scaler is not None:
                    scaler.unscale_(optimizer)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad()
                scheduler.step()
            step += 1
            if on_step is not None:
                on_step(epoch + 1, step, batches, {"total_loss": float(loss.item()), "top_3_acc": acc}, float(scheduler.get_last_lr()[0]))
        if (epoch + 1) % max(1, run.save_every_n_epoch) == 0:
            run.weights_dir.mkdir(parents=True, exist_ok=True)
            target = run.weights_dir / f"{run.exp_name}-e{epoch + 1}.ckpt"
            torch.save(half_weights(model, config, epoch + 1), str(target))
            saved.append(target)
    return {"weights": [str(p) for p in saved], "clips": data.clips, "samples": len(data), "batch_size": batch_size, "steps": step, "skipped": data.skipped}
