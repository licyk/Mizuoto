"""The maths of Fish Speech's LoRA fine-tune, for ``train.py`` (fish-speech ``models/text2semantic/lora.py`` with
loralib's layers, ``llama.py``'s training forward, ``lit_module.py``, ``datasets/semantic.py``,
``content_sequence.py``, ``tools/llama/build_dataset.py`` and ``merge_lora.py``; Fish Audio Research License).

Everything keeps upstream's operations and their order, so a step on the CPU in fp32 gives v1's loss and gradients:

- **LoRA** is loralib's ``Linear`` and ``Embedding`` without the library: the frozen weight in float32 (loralib's
  layer is made in the default dtype and the weight copied in), ``A`` and ``B`` beside it, the low-rank path added in
  place (``result += (dropout(x) @ Aᵀ @ Bᵀ) · α/r``). Every other parameter keeps the checkpoint's dtype, as upstream's
  ``load_state_dict(assign=True)`` leaves it (bfloat16 for S2), and computes as upstream does with it.
- **Random numbers.** Upstream builds the model with random weights (every layer's own initialisation, then
  ``_init_weights``), overwrites them with the checkpoint, and builds each loralib layer twice over (its weight
  initialised in its constructor and again in ``reset_parameters``) before drawing ``A`` (or an embedding's ``B``).
  ``initial_lora`` replays those draws on the process-wide generator without materialising them (each element of a
  float32 ``normal_`` or ``uniform_`` advances the generator as one raw 32-bit draw does; ``normal_`` draws 16 more when
  the size is not a multiple of 16), from the shapes alone, then draws ``A`` and ``B`` as loralib does. The data
  loader's worker seed comes next, then the dropout masks of each step.
- **Data** is ``AutoTextSemanticInstructionIterableDataset`` as the data loader's single worker runs it: Python's
  ``random`` seeded with the loader's base seed, a speaker group drawn by its number of clips, a run of clips in order
  (all of them when they fit ``max_length // 20``), each clip's text cleaned and its codes interleaved as upstream's
  ``ContentSequence`` encodes them, the clips concatenated, then ``TextDataCollator``'s truncation and padding.
- **Training forward** is ``DualARTransformer.forward``: the codebook embeddings summed *without* the 1/√11 scale
  inference applies (upstream's training path does not scale), each slow and fast layer checkpointed as upstream does,
  the fast transformer on the positions whose label is a semantic token, fed that frame's codes. The loss is
  ``lit_module``'s: the next-token cross-entropy plus the codebooks' cross-entropy.
"""

from __future__ import annotations

import json
import math
import random
import re
from collections import OrderedDict
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from mizuoto.runtime import ops
from mizuoto_fishspeech.model import DualARTransformer, ModelArgs

IM_END = "<|im_end|>"
CODEBOOK_PAD = 0
"""``CODEBOOK_PAD_TOKEN_ID``: the codebook rows of text positions, and of the last label column."""
DATASET_SEED = 42
"""The dataset's ``seed``: the shards' and the groups' order."""
SYSTEM_TEXT = "Speak out the provided text."


@dataclass(frozen=True)
class LoraConfig:
    r: int
    alpha: float
    dropout: float
    targets: tuple[str, ...]


LORA_CONFIGS = {
    "r_8_alpha_16": LoraConfig(8, 16, 0.01, ("attention", "mlp", "embeddings", "output")),
    "r_32_alpha_16_fast": LoraConfig(32, 16, 0.1, ("fast_attention", "fast_mlp", "fast_embeddings", "fast_output")),
}
"""Upstream's ``configs/lora/*.yaml``."""


# -- random numbers -----------------------------------------------------------------------------


class Draws:
    """Advances the process-wide CPU generator by raw 32-bit draws, as the replayed initialisations would. The draws are
    counted and the generator jumped ahead before the next real draw (``flush``; ``mizuoto.training.rng.advance``: the
    same state, without making ~18 G draws)."""

    def __init__(self) -> None:
        self.pending = 0

    def skip(self, count: int) -> None:
        self.pending += count

    def flush(self) -> None:
        from mizuoto.training.rng import advance

        advance(torch.default_generator, self.pending)
        self.pending = 0

    def normal(self, numel: int) -> None:
        """``tensor.normal_()`` on a float32 tensor of ``numel`` elements."""
        if numel < 16:  # the serial path, with its cached second sample: draw it for real
            self.flush()
            torch.empty(numel).normal_()
        else:
            self.skip(numel + (16 if numel % 16 else 0))

    def uniform(self, numel: int) -> None:
        """``tensor.uniform_()`` on a float32 tensor (``kaiming_uniform_``)."""
        self.skip(numel)

    def linear(self, cin: int, cout: int, bias: bool) -> None:
        """``nn.Linear.reset_parameters``."""
        self.uniform(cout * cin)
        if bias:
            self.uniform(cout)


def init_draws(c: ModelArgs, draws: Draws) -> None:
    """What upstream's ``DualARTransformer(config)`` draws: each ``nn.Linear`` and ``nn.Embedding`` initialised by its
    constructor, then ``_init_weights``' ``normal_`` over every one of them."""
    linears: list[tuple[int, int, bool]] = []
    embeddings: list[int] = [c.vocab_size * c.dim, c.codebook_size * c.num_codebooks * c.dim]

    def blocks(a: ModelArgs, n: int) -> None:
        for _ in range(n):
            linears.append((a.dim, (a.n_head + 2 * a.n_local_heads) * a.head_dim, a.attention_qkv_bias))
            linears.append((a.n_head * a.head_dim, a.dim, a.attention_o_bias))
            linears.extend([(a.dim, a.intermediate_size, False), (a.dim, a.intermediate_size, False), (a.intermediate_size, a.dim, False)])

    blocks(c, c.n_layer)
    if not c.tie_word_embeddings:
        linears.append((c.dim, c.vocab_size, False))
    if c.fast_dim != c.dim:
        linears.append((c.dim, c.fast_dim, True))
    embeddings.append(c.codebook_size * c.fast_dim)
    blocks(c.fast(), c.n_fast_layer)
    linears.append((c.fast_dim, c.codebook_size, False))
    for numel in embeddings:  # constructors (their order does not matter: none of these values is kept)
        draws.normal(numel)
    for cin, cout, bias in linears:
        draws.linear(cin, cout, bias)
    for numel in embeddings:  # _init_weights
        draws.normal(numel)
    for cin, cout, _ in linears:
        draws.normal(cin * cout)


# -- LoRA ---------------------------------------------------------------------------------------


def _frozen(layer: nn.Module, weight: torch.Tensor) -> None:
    """The LoRA layer's frozen weight: the base's own tensor when it already has the layer's dtype (bf16-true: the
    checkpoint's bf16 tensors, memory-mapped, so no 8.6 GB copy is made of them), else a copy, as loralib makes one."""
    own = layer.get_parameter("weight")
    if weight.dtype == own.dtype and weight.shape == own.shape:
        layer.weight = nn.Parameter(weight.detach(), requires_grad=False)
    else:
        own.copy_(weight)


class LoraLinear(ops.Linear):
    """loralib's ``Linear`` (``merge_weights``, no ``fan_in_fan_out``): parameters ``weight``, ``bias``, ``lora_A``, ``lora_B``."""

    def __init__(self, base: nn.Linear, config: LoraConfig, a: torch.Tensor, b: torch.Tensor, dtype: torch.dtype | None = None) -> None:
        super().__init__(base.in_features, base.out_features, bias=base.bias is not None, dtype=dtype)
        with torch.no_grad():
            _frozen(self, base.weight)
            if self.bias is not None and base.bias is not None:
                self.bias.copy_(base.bias)
        self.lora_A = nn.Parameter(a)
        self.lora_B = nn.Parameter(b)
        self.scaling = config.alpha / config.r
        self.lora_dropout: nn.Module = nn.Dropout(p=config.dropout) if config.dropout > 0 else nn.Identity()
        self.weight.requires_grad = False

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        result = F.linear(input, self.weight, bias=self.bias)
        result += (self.lora_dropout(input) @ self.lora_A.transpose(0, 1) @ self.lora_B.transpose(0, 1)) * self.scaling
        return result


class LoraEmbedding(ops.Embedding):
    """loralib's ``Embedding``: no dropout."""

    def __init__(self, base: nn.Embedding, config: LoraConfig, a: torch.Tensor, b: torch.Tensor, dtype: torch.dtype | None = None) -> None:
        super().__init__(base.num_embeddings, base.embedding_dim, padding_idx=base.padding_idx, dtype=dtype)
        with torch.no_grad():
            _frozen(self, base.weight)
        self.lora_A = nn.Parameter(a)
        self.lora_B = nn.Parameter(b)
        self.scaling = config.alpha / config.r
        self.weight.requires_grad = False

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        result = F.embedding(input, self.weight, self.padding_idx, self.max_norm, self.norm_type, self.scale_grad_by_freq, self.sparse)
        after_a = F.embedding(input, self.lora_A.transpose(0, 1), self.padding_idx, self.max_norm, self.norm_type, self.scale_grad_by_freq, self.sparse)
        result += (after_a @ self.lora_B.transpose(0, 1)) * self.scaling
        return result


@dataclass(frozen=True)
class LoraTarget:
    path: str
    """The layer's name in the model."""
    embedding: bool
    cin: int
    """``in_features``, or an embedding's ``num_embeddings``."""
    cout: int
    """``out_features``, or an embedding's ``embedding_dim``."""
    bias: bool = False


def lora_targets(c: ModelArgs, config: LoraConfig) -> list[LoraTarget]:
    """The layers upstream's ``setup_lora`` replaces, in the order it makes their loralib layers: the embeddings, then
    the linear layers it collected (the slow model's output, each slow layer's, the fast output, each fast layer's)."""
    targets = set(config.targets)
    slow_attention, slow_mlp, slow_embeddings, slow_output = ("attention" in targets, "mlp" in targets, "embeddings" in targets, "output" in targets)
    fast_attention = slow_attention or "fast_attention" in targets
    fast_mlp = slow_mlp or "fast_mlp" in targets
    fast_embeddings = slow_embeddings or "fast_embeddings" in targets
    fast_output = slow_output or "fast_output" in targets
    out: list[LoraTarget] = []
    if slow_embeddings:
        out += [LoraTarget("embeddings", True, c.vocab_size, c.dim), LoraTarget("codebook_embeddings", True, c.codebook_size * c.num_codebooks, c.dim)]
    if fast_embeddings:
        out.append(LoraTarget("fast_embeddings", True, c.codebook_size, c.fast_dim))

    def layers(prefix: str, a: ModelArgs, n: int, attention: bool, mlp: bool) -> list[LoraTarget]:
        found = []
        for i in range(n):
            if attention:
                found.append(LoraTarget(f"{prefix}.{i}.attention.wqkv", False, a.dim, (a.n_head + 2 * a.n_local_heads) * a.head_dim, a.attention_qkv_bias))
                found.append(LoraTarget(f"{prefix}.{i}.attention.wo", False, a.n_head * a.head_dim, a.dim, a.attention_o_bias))
            if mlp:
                found.append(LoraTarget(f"{prefix}.{i}.feed_forward.w1", False, a.dim, a.intermediate_size))
                found.append(LoraTarget(f"{prefix}.{i}.feed_forward.w2", False, a.intermediate_size, a.dim))
                found.append(LoraTarget(f"{prefix}.{i}.feed_forward.w3", False, a.dim, a.intermediate_size))
        return found

    if slow_output and not c.tie_word_embeddings:
        out.append(LoraTarget("output", False, c.dim, c.vocab_size))
    out += layers("layers", c, c.n_layer, slow_attention, slow_mlp)
    if fast_output:
        out.append(LoraTarget("fast_output", False, c.fast_dim, c.codebook_size))
    out += layers("fast_layers", c.fast(), c.n_fast_layer, fast_attention, fast_mlp)
    return out


def initial_lora(c: ModelArgs, config: LoraConfig, draws: Draws) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
    """The LoRA weights upstream starts from, drawn from the process-wide generator as upstream draws them: after the
    model's random initialisation, each loralib layer's weight initialised twice (its constructor, then
    ``reset_parameters``), then a linear layer's ``A`` (``kaiming_uniform_``, ``B`` zero) or an embedding's ``B``
    (standard normal, ``A`` zero). Needs only the shapes, so it runs while the weights load."""
    init_draws(c, draws)
    out = {}
    for t in lora_targets(c, config):
        if t.embedding:
            draws.normal(t.cin * t.cout)
            draws.normal(t.cin * t.cout)
            a, b = torch.zeros((config.r, t.cin)), torch.empty((t.cout, config.r))
            draws.flush()
            nn.init.normal_(b)
        else:
            draws.linear(t.cin, t.cout, t.bias)
            draws.linear(t.cin, t.cout, t.bias)
            a, b = torch.empty((config.r, t.cin)), torch.zeros((t.cout, config.r))
            draws.flush()
            nn.init.kaiming_uniform_(a, a=math.sqrt(5))
        out[t.path] = (a, b)
    draws.flush()
    return out


def setup_lora(model: DualARTransformer, config: LoraConfig, init: dict[str, tuple[torch.Tensor, torch.Tensor]], dtype: torch.dtype | None = None) -> None:
    """Upstream ``setup_lora`` with the initial LoRA weights given: each layer replaced by its LoRA layer (the frozen
    weight copied in float32, or in ``dtype`` when the whole model is cast to it next, as ``bf16-true`` does: the same
    values without the float32 copies), then only the LoRA weights train (loralib's ``mark_only_lora_as_trainable``)."""
    for path, (a, b) in init.items():
        parent, _, name = path.rpartition(".")
        old = model.get_submodule(path)
        if isinstance(old, nn.Embedding):
            new: nn.Module = LoraEmbedding(old, config, a, b, dtype)
        elif isinstance(old, nn.Linear):
            new = LoraLinear(old, config, a, b, dtype)
        else:
            raise TypeError(f"{path} is a {type(old).__name__}")
        setattr(model.get_submodule(parent) if parent else model, name, new)
    for name, p in model.named_parameters():
        if "lora_" not in name:
            p.requires_grad = False


def lora_state(model: nn.Module) -> dict[str, torch.Tensor]:
    """The LoRA weights by upstream's names (what its checkpoint keeps, without Lightning's ``model.`` prefix)."""
    return {k: v.detach().cpu().contiguous() for k, v in model.state_dict().items() if "lora" in k}


def param_groups(model: nn.Module) -> list[dict[str, Any]]:
    """``TextToSemantic.configure_optimizers``' two groups, in upstream's parameter order (frozen ones left out: AdamW
    and the gradient clipping skip parameters without gradients)."""
    decay, other = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        full = "model." + name
        (other if ".bias" in full or "norm.weight" in full or ".embeddings." in full else decay).append(p)
    return [{"params": decay}, {"params": other, "weight_decay": 0.0}]


def constant_with_warmup(step: int, warmup: int = 10) -> float:
    """``get_constant_schedule_with_warmup_lr_lambda`` (``num_warmup_steps: 10``)."""
    if step < warmup:
        return float(step) / float(max(1, warmup))
    return 1.0


# -- data ---------------------------------------------------------------------------------------


EMOJI = re.compile("[\U0001f600-\U0001f64f\U0001f300-\U0001f5ff\U0001f680-\U0001f6ff\U0001f1e0-\U0001f1ff]+", flags=re.UNICODE)


def pack_text(text: str) -> str:
    """``build_dataset.run_task``'s cleaning: ``{…}`` and ``<…>`` become spaces, whitespace runs one space."""
    text = re.sub(r"\{.*?\}", " ", text)
    text = re.sub(r"<.*?>", " ", text)
    return re.sub(r"\s+", " ", text)


def clean_text(text: str) -> str:
    """``fish_speech.text.clean.clean_text``."""
    text = text.strip()
    text = text.replace("‘", "'").replace("’", "'")
    text = EMOJI.sub("", text)
    return re.sub(r"[,]{2,}", lambda m: m.group()[0], text)


@dataclass
class Sentence:
    texts: list[str]
    codes: Any
    """(codebooks, frames) integers."""


@dataclass
class Group:
    name: str
    sentences: list[Sentence]


def load_groups(folder: Path) -> list[Group]:
    """The packed dataset ``train.py``'s ``tokens`` stage wrote (``groups.json`` and each clip's codes)."""
    import numpy as np

    index = json.loads((folder / "groups.json").read_text(encoding="utf-8"))
    return [Group(g["name"], [Sentence(list(s["texts"]), np.load(folder / s["codes"])) for s in g["sentences"]]) for g in index]


def encode_sentence(text: str, codes: Any, tokenizer: Any, semantic_begin_id: int) -> tuple[torch.Tensor, torch.Tensor]:
    """``pack_sentences``: (tokens, labels), each (codebooks + 1, length), as ``ContentSequence.encode`` makes them."""
    vq = torch.tensor(codes).to(torch.int32)
    # (text or codes, labelled); the end markers take the label setting of the part before them
    parts: list[tuple[str | torch.Tensor, bool]] = [
        (SYSTEM_TEXT, False),
        (f"<|speaker:user|> {text}", False),
        (IM_END, False),
        ("<|speaker:assistant|> <|voice|>", False),
        (vq, True),
        (IM_END, True),
    ]
    all_tokens, all_labels, vq_masks = [], [], []
    for part, labelled in parts:
        if isinstance(part, str):
            tokens = torch.tensor(tokenizer.encode(part), dtype=torch.long)
            vq_masks.append(torch.zeros_like(tokens, dtype=torch.bool))
        else:
            tokens = (part.clone().to(torch.int)[0] + semantic_begin_id).to(torch.long)
            vq_masks.append(torch.ones_like(tokens, dtype=torch.bool))
        all_tokens.append(tokens)
        all_labels.append(tokens.clone() if labelled else torch.full_like(tokens, -100))
    tokens_raw = torch.cat(all_tokens)[:-1]
    labels_raw = torch.cat(all_labels)[1:]
    mask = torch.cat(vq_masks)
    mask_tokens, mask_labels = mask[:-1], mask[1:]
    n = vq.shape[0]
    tokens = torch.zeros((n + 1, len(tokens_raw)), dtype=torch.int)
    tokens[0] = tokens_raw
    tokens[1:, mask_tokens] = vq
    labels = torch.full((n + 1, len(labels_raw)), -100, dtype=torch.int)
    labels[0, :] = labels_raw
    labels[1:, mask_labels] = vq
    labels[1:, -1:] = CODEBOOK_PAD
    return tokens.long(), labels.long()


class Samples:
    """``AutoTextSemanticInstructionIterableDataset`` in its data-loader worker (``causal``, no skipped text): an
    endless stream of {tokens, labels}, drawing from ``random.Random(base_seed)`` as the worker's seeded ``random``."""

    def __init__(self, groups: list[Group], tokenizer: Any, max_length: int, semantic_begin_id: int, base_seed: int) -> None:
        self.groups = list(groups)
        random.Random(DATASET_SEED).shuffle(self.groups)
        self.weights = [len(g.sentences) for g in self.groups]
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.semantic_begin_id = semantic_begin_id
        self.rng = random.Random(base_seed)  # base seed + worker id 0

    def __iter__(self) -> Iterator[dict[str, torch.Tensor]]:
        while True:
            yield self.augment()

    def augment(self) -> dict[str, torch.Tensor]:
        rng = self.rng
        num_samples = self.max_length // 20
        group = rng.choices(self.groups, weights=self.weights, k=1)[0]
        if num_samples >= len(group.sentences):
            samples = group.sentences
        else:
            begin = rng.randint(0, len(group.sentences) - num_samples)
            samples = group.sentences[begin : begin + num_samples]
        all_tokens, all_labels = [], []
        for sentence in samples:
            text = clean_text(rng.choice(sentence.texts))
            rng.random()  # < skip_text_prob (0): the text is never skipped
            tokens, labels = encode_sentence(text, sentence.codes, self.tokenizer, self.semantic_begin_id)
            all_tokens.append(tokens)
            all_labels.append(labels)
        return {"tokens": torch.cat(all_tokens, dim=1), "labels": torch.cat(all_labels, dim=1)}


def collate(examples: list[dict[str, torch.Tensor]], max_length: int, pad_id: int) -> dict[str, torch.Tensor]:
    """``TextDataCollator.batchify``: truncated to ``max_length``, padded to the longest (``attention_masks`` is True
    on padding)."""
    longest = min(max(e["tokens"].size(1) for e in examples), max_length)
    tokens, masks, labels = [], [], []
    for e in examples:
        t, lab = e["tokens"][:, :longest], e["labels"][:, :longest]
        mask = torch.ones((longest,), dtype=torch.bool)
        length = t.size(1)
        mask[:length] = False
        if length < longest:
            t = F.pad(t, (0, longest - length), value=pad_id)
            t[1:, length:] = CODEBOOK_PAD
            lab = F.pad(lab, (0, longest - lab.size(1)), value=-100)
        tokens.append(t)
        masks.append(mask)
        labels.append(lab)
    return {"inputs": torch.stack(tokens), "attention_masks": torch.stack(masks), "labels": torch.stack(labels)}


# -- forward and loss ---------------------------------------------------------------------------


class _NoCache:
    """Training keeps no cache: the layer's keys and values as they are."""

    def update(self, k: torch.Tensor, v: torch.Tensor, active: int) -> tuple[torch.Tensor, torch.Tensor]:
        return k, v


NO_CACHE = _NoCache()


def _run(layer: nn.Module, x: torch.Tensor, freqs_cis: torch.Tensor, mask: torch.Tensor, checkpointed: bool) -> torch.Tensor:
    """A layer, checkpointed as upstream does in training (the same values and draws: the recomputation replays the
    forward's random state and restores the generator after). Without a gradient to carry there is nothing to save."""
    if checkpointed and x.requires_grad:
        return checkpoint(layer, x, freqs_cis, mask, NO_CACHE, False, use_reentrant=True)  # type: ignore[return-value]
    return layer(x, freqs_cis, mask, NO_CACHE, False)


def forward(model: DualARTransformer, inp: torch.Tensor, labels: torch.Tensor, key_padding_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """``DualARTransformer.forward`` in training: (token logits, codebook logits of the semantic positions)."""
    c = model.config
    seq_len = inp.size(2)
    ckpt = c.use_gradient_checkpointing and model.training
    # BaseTransformer.embed: the codebook embeddings summed, not scaled
    embeds = [model.codebook_embeddings(inp[:, i + 1] + i * c.codebook_size) for i in range(c.num_codebooks)]
    vq_sum = torch.stack(embeds, dim=1).sum(dim=1)
    is_semantic = (inp[:, 0] >= c.semantic_begin_id) & (inp[:, 0] <= c.semantic_end_id)
    vq_sum[~is_semantic] = 0
    x = model.embeddings(inp[:, 0]) + vq_sum
    freqs_cis = model.get_buffer("freqs_cis")[:seq_len]
    causal = torch.ones(seq_len, seq_len, dtype=torch.bool, device=inp.device).tril()
    mask = causal[None, None] & key_padding_mask[:, None, None, :].logical_not()
    for layer in model.layers:
        x = _run(layer, x, freqs_cis, mask, ckpt)
    slow_out = model.norm(x)
    embeddings: Any = model.embeddings
    token_logits = F.linear(slow_out, embeddings.weight) if c.tie_word_embeddings else model.output(slow_out)
    hidden = slow_out if c.norm_fastlayer_input else x

    n = c.num_codebooks
    fast_mask = torch.ones(n, n, dtype=torch.bool, device=inp.device).tril()[None, None]
    fast_freqs_cis = model.get_buffer("fast_freqs_cis")[:n]
    token_labels = labels[:, 0]
    codebook_mask = (token_labels >= c.semantic_begin_id) & (token_labels <= c.semantic_end_id)
    h = hidden[codebook_mask]
    if h.shape[0] == 0:  # upstream's dummy input when no position needs the fast transformer
        h = torch.zeros((4, c.dim), device=h.device, dtype=h.dtype)
        codebooks = torch.zeros((h.shape[0], n - 1), device=h.device, dtype=torch.int)
    else:
        codebooks = labels[:, 1:, :].permute(0, 2, 1)[codebook_mask][:, :-1]
    h = model.fast_project_in(h)
    h = torch.cat([h[:, None], model.fast_embeddings(codebooks)], dim=1)
    for layer in model.fast_layers:
        h = _run(layer, h, fast_freqs_cis, fast_mask, ckpt)
    return token_logits, model.fast_output(model.fast_norm(h))


def losses(model: DualARTransformer, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """``TextToSemantic._step``: loss = the next-token cross-entropy + the codebooks'; top-5 accuracy of the codes."""
    c = model.config
    labels = batch["labels"]
    token_logits, codebook_logits = forward(model, batch["inputs"], labels, batch["attention_masks"])
    base_loss = F.cross_entropy(token_logits.view(-1, token_logits.size(-1)), labels[:, 0].reshape(-1), ignore_index=-100)
    token_ids = labels[:, 0]
    semantic_mask = (token_ids >= c.semantic_begin_id) & (token_ids <= c.semantic_end_id)
    codebook_labels = labels[:, 1 : 1 + c.num_codebooks].permute(0, 2, 1)[semantic_mask]
    semantic_loss = F.cross_entropy(codebook_logits.reshape(-1, codebook_logits.size(-1)), codebook_labels.reshape(-1), ignore_index=-100)
    return {"loss": base_loss + semantic_loss, "base_loss": base_loss, "semantic_loss": semantic_loss, "top_5_accuracy": accuracy(codebook_logits, codebook_labels)}


@torch.no_grad()
def accuracy(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    mask = (labels != -100) & (labels != CODEBOOK_PAD)
    if mask.sum() == 0:
        return torch.tensor(0.0, device=logits.device)
    _, indices = logits.topk(5, dim=-1)
    correct = indices.eq(labels.unsqueeze(-1))
    correct[~mask] = 0
    return correct.sum() / mask.sum()


# -- merge ---------------------------------------------------------------------------------------


def merged_state(base: dict[str, torch.Tensor], lora: dict[str, torch.Tensor], config: LoraConfig, keys: list[str]) -> OrderedDict[str, torch.Tensor]:
    """``merge_lora``: the base weights with each LoRA merged as loralib's ``eval()`` does it (in float32: the layer's
    weight is a float32 copy), every weight then in bfloat16 (``model.to(torch.bfloat16)``), in upstream's order."""
    scaling = config.alpha / config.r
    out: OrderedDict[str, torch.Tensor] = OrderedDict()
    for key in keys:
        prefix = key.removesuffix(".weight")
        a, b = lora.get(f"{prefix}.lora_A"), lora.get(f"{prefix}.lora_B")
        weight = base[key]
        if a is not None and b is not None and key.endswith(".weight"):
            merged = weight.to(torch.float32, copy=True)  # one copy, whatever the base's dtype
            delta = b.to(torch.float32) @ a.to(torch.float32)
            embedding = prefix.rsplit(".", 1)[-1].endswith("embeddings")
            merged += (delta.transpose(0, 1) if embedding else delta) * scaling
            weight = merged
        out[key] = weight.to(torch.bfloat16)
    missing = {k.rsplit(".", 1)[0] for k in lora} - {k.removesuffix(".weight") for k in keys}
    if missing:
        raise KeyError(f"LoRA weights for layers the model does not have: {sorted(missing)[:5]}")
    return out
