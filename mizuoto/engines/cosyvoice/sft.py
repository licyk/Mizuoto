"""CosyVoice fine-tuning, the maths (upstream's LibriTTS recipe, Apache-2.0, Alibaba): imported lazily by ``train.py``.

- **Features** as the recipe's ``tools/extract_speech_token.py`` and ``tools/extract_embedding.py`` make them, on the
  native speech tokenizer and CAM++ (``nn.codecs.s3tokenizer``, ``nn.speaker.campplus``) instead of onnxruntime.
- **The data pipeline** of the recipe's yaml (``cosyvoice/dataset``): parquet shards of 1000 utterances, the sampler's
  shard partition over the data loader's workers, ``tokenize``, ``filter``, ``resample``, ``compute_fbank``,
  ``parse_embedding``, the 1000-sample ``shuffle`` and 500-sample ``sort`` buffers, ``dynamic_batch`` (2000 frames)
  and ``padding``. Each worker's pipeline draws from its own generator, seeded as the data loader seeds its worker
  processes (the loader's base seed, drawn from torch's generator, plus the worker's index), and the batches come in
  the loader's round-robin order; with no workers, the pipeline draws from Python's generator between the model's own
  draws. With workers, features are computed on one thread, as in a worker process. With fewer shards than workers
  each worker reads every shard: with the default two workers a dataset under 1000 clips is seen twice an epoch.
- **The losses**: ``TransformerLM.forward`` (v1: Conformer text encoder, Transformer LM, dropout 0.1),
  ``Qwen2LM``/``CosyVoice3LM.forward`` (Qwen2 as transformers 4.51 runs it under a padding mask; half the
  utterances long enough are laid out as the bidirectional-streaming sequence), with upstream's label-smoothing
  KL loss and accuracy; ``MaskedDiffWithXvec`` / ``CausalMaskedDiffWithXvec`` / ``CausalMaskedDiffWithDiT.forward``
  with ``ConditionalCFM.compute_loss`` (random prompt conditions, unified streaming training, classifier-free
  condition drops). The native modules hold the weights; the training forwards here add upstream's dropout layers
  where they draw (encoders, DiT), in upstream's order.
- **The loop** is ``Executor.train_one_epoc`` with ``train_utils``: the loss divided by the accumulation steps,
  gradients clipped to norm 5 over the parameters in upstream's order, Adam, the recipe's scheduler (CosyVoice 1's
  ``WarmupLR``, 2's and 3's constant rate), bf16 autocast with a gradient scaler on CUDA (fp32 elsewhere), a
  validation pass after each epoch on the same data (it draws too) and ``epoch_<N>_whole.pt`` with upstream's keys.
"""

from __future__ import annotations

import functools
import json
import math
import random
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils.rnn import pad_sequence

from mizuoto.engines.cosyvoice.dit import SITE as DIT_SITE
from mizuoto.nn.transformer.wenet import ConformerEncoderLayer, add_optional_chunk_mask, make_pad_mask
from mizuoto.runtime.accel import attention

IGNORE_ID = -1
UTTS_PER_SHARD = 1000
"""``make_parquet_list.py --num_utts_per_parquet 1000``, as v1 ran it."""
MAX_TOKEN_SECONDS = 30
"""``extract_speech_token.py`` gives no tokens for longer clips (and the recipe's ``filter`` then drops them)."""
DROPOUT, POSITIONAL_DROPOUT = 0.1, 0.1
"""Every encoder's ``dropout_rate`` and ``positional_dropout_rate`` in the recipes."""
LLM_ATTENTION_DROPOUT, FLOW_ATTENTION_DROPOUT, DIT_DROPOUT = 0.0, 0.1, 0.1
SIGMA_MIN, TRAINING_CFG_RATE = 1e-6, 0.2
"""``cfm_params`` of every recipe."""
MIX_RATIO = (5, 15)
GRAD_CLIP = 5.0


@dataclass(frozen=True)
class Recipe:
    """What ``examples/libritts/cosyvoice{,2,3}/conf/*.yaml`` sets for the data and the optimiser."""

    sample_rate: int
    feat: dict[str, Any]
    """``feat_extractor``: Matcha's mel spectrogram."""
    num_frames: int
    """``compute_fbank``'s alignment of the waveform (-1: none)."""
    max_length: float
    min_length: float
    scheduler: str
    token_max_length: int = 200
    token_min_length: int = 1
    min_output_input_ratio: float = 0.0005
    max_output_input_ratio: float = 1.0
    shuffle_size: int = 1000
    sort_size: int = 500
    max_frames_in_batch: int = 2000
    warmup_steps: int = 2500


def _feat(sr: int, n_fft: int, hop: int, fmax: float | None) -> dict[str, Any]:
    return {"n_fft": n_fft, "num_mels": 80, "sampling_rate": sr, "hop_size": hop, "win_size": n_fft, "fmin": 0, "fmax": fmax, "center": False}


RECIPES = {
    1: Recipe(22050, _feat(22050, 1024, 256, 8000), -1, 40960, 0, "warmuplr"),
    2: Recipe(24000, _feat(24000, 1920, 480, 8000), 960, 6000, 100, "constantlr"),
    3: Recipe(24000, _feat(24000, 1920, 480, None), 960, 6000, 100, "constantlr"),
}


# -- features (the recipe's stages 1 and 2) ---------------------------------------------------------------------


def load_feature_models(model_dir: Path, version: int, device: str) -> tuple[Any, Any]:
    """The speech tokenizer (on ``device``) and CAM++ (on the CPU, as ``extract_embedding.py`` runs it) from the model's
    ONNX files (converted as the engine converts them)."""
    from mizuoto.engines.cosyvoice.engine import LoadedCosyVoice
    from mizuoto.nn.codecs.s3tokenizer import S3Tokenizer
    from mizuoto.nn.codecs.s3tokenizer import state_dict_from_onnx as tokenizer_state
    from mizuoto.nn.speaker.campplus import CAMPPlus
    from mizuoto.nn.speaker.campplus import state_dict_from_onnx as campplus_state
    from mizuoto.runtime import ops
    from mizuoto.runtime.loading import load_into

    state = LoadedCosyVoice._onnx_weights(model_dir, f"speech_tokenizer_v{version}.onnx", lambda g: tokenizer_state(g, version))
    tokenizer = S3Tokenizer.from_state(version, state, ops.default)
    load_into(tokenizer, state, what="speech tokenizer")
    campplus = CAMPPlus(ops=ops.default)
    load_into(campplus, LoadedCosyVoice._onnx_weights(model_dir, "campplus.onnx", campplus_state), ignore=["*.num_batches_tracked"], what="CAM++")
    return tokenizer.eval().to(device), campplus.eval()


def utterance_features(wav: Path, tokenizer: Any, campplus: Any, device: str) -> tuple[list[int], list[float]]:
    """A clip's speech tokens (``extract_speech_token.py``) and CAM++ embedding (``extract_embedding.py``)."""
    from mizuoto.nn.dsp.kaldi import fbank
    from mizuoto.nn.dsp.mel import whisper_log_mel
    from mizuoto.nn.dsp.resample import Resample
    from mizuoto.nn.dsp.wav import read_wav

    samples, sr = read_wav(wav)
    audio = torch.from_numpy(samples).unsqueeze(0)
    if sr != 16000:
        audio = Resample(sr, 16000)(audio)
    tokens: list[int] = []
    if audio.shape[1] / 16000 <= MAX_TOKEN_SECONDS:
        feat = whisper_log_mel(audio, n_mels=128).to(device)
        codes, lens = tokenizer.quantize(feat, torch.tensor([feat.shape[2]], dtype=torch.int32, device=device))
        tokens = codes[0, : int(lens[0])].flatten().tolist()
    feat = fbank(audio, num_mel_bins=80, dither=0, sample_frequency=16000)
    feat = feat - feat.mean(dim=0, keepdim=True)
    embedding = campplus(feat.unsqueeze(dim=0)).flatten().tolist()
    return tokens, embedding


# -- the data pipeline --------------------------------------------------------------------------------------------


@dataclass
class Utterance:
    """A sample after ``tokenize`` … ``parse_embedding``: what ``shuffle``, ``sort``, ``batch`` and ``padding`` see."""

    utt: str
    text: str
    text_token: list[int]
    instruct_token: list[int] | None
    speech_token: list[int]
    speech_len: int
    """Samples after resampling and ``compute_fbank``'s alignment (``padding`` orders a batch by it)."""
    speech_feat: torch.Tensor
    """(frames, 80), the transposed view ``compute_fbank`` keeps."""
    embedding: torch.Tensor
    """The normalised utterance embedding (``use_spk_embedding: False``)."""


def prepare(items: list[dict[str, Any]], recipe: Recipe, tokenizer: Any, threads: int | None = 1) -> list[Utterance | None]:
    """Each item through ``tokenize``, ``filter``, ``resample``, ``compute_fbank`` and ``parse_embedding`` (``None``:
    dropped by the filter). None of them draws; they run on one thread, as in the loader's worker processes."""
    from mizuoto.nn.dsp.mel import matcha_basis, matcha_mel
    from mizuoto.nn.dsp.resample import Resample
    from mizuoto.nn.dsp.wav import read_wav

    f = recipe.feat
    basis = matcha_basis(f["sampling_rate"], f["n_fft"], f["num_mels"], f["fmin"], f["fmax"])
    resamplers: dict[int, Resample] = {}
    before = torch.get_num_threads()
    if threads is not None:
        torch.set_num_threads(threads)
    try:
        out: list[Utterance | None] = []
        with torch.no_grad():
            for item in items:
                text_token = tokenizer.encode(item["text"])
                instruct = item.get("instruct")
                instruct_token = tokenizer.encode(instruct) if instruct is not None else None
                samples, sr = read_wav(item["wav"])
                speech = torch.from_numpy(samples).unsqueeze(0)
                frames = speech.size(1) / sr * 100
                n_text = len(text_token)
                if (
                    not recipe.min_length <= frames <= recipe.max_length
                    or not recipe.token_min_length <= n_text <= recipe.token_max_length
                    or len(item["speech_token"]) == 0
                    or (frames != 0 and not recipe.min_output_input_ratio <= n_text / frames <= recipe.max_output_input_ratio)
                ):
                    out.append(None)
                    continue
                if sr != recipe.sample_rate:
                    if sr < 16000:
                        out.append(None)
                        continue
                    if sr not in resamplers:
                        resamplers[sr] = Resample(sr, recipe.sample_rate)
                    speech = resamplers[sr](speech)
                peak = speech.abs().max()
                if peak > 1:
                    speech /= peak
                if recipe.num_frames != -1:
                    index = int(math.ceil(speech.shape[1] / recipe.num_frames))
                    speech = torch.concat([speech, torch.zeros(1, index * recipe.num_frames - speech.shape[1])], dim=1)
                feat = matcha_mel(speech, basis=basis, **f).squeeze(dim=0).transpose(0, 1)
                embedding = F.normalize(torch.tensor(item["utt_embedding"], dtype=torch.float32), dim=0)
                out.append(Utterance(item["utt"], item["text"], text_token, instruct_token, list(item["speech_token"]), speech.shape[1], feat, embedding))
        return out
    finally:
        torch.set_num_threads(before)


def shards(count: int) -> list[list[int]]:
    """``make_parquet_list.py``: the utterances in order, 1000 a shard."""
    return [list(range(i, min(i + UTTS_PER_SHARD, count))) for i in range(0, count, UTTS_PER_SHARD)]


def sampled(n_shards: int, epoch: int, worker: int, workers: int, train: bool) -> list[int]:
    """``DistributedSampler.sample`` on one rank: the shards one worker reads (train: shuffled with the epoch)."""
    data = list(range(n_shards))
    if train:
        random.Random(epoch).shuffle(data)
    if len(data) < workers:
        data = (data * math.ceil(workers / len(data)))[:workers]
    return data[worker::workers]


def _shuffle(data: Iterable[Utterance], rng: Any, size: int) -> Iterator[Utterance]:
    buf: list[Utterance] = []
    for sample in data:
        buf.append(sample)
        if len(buf) >= size:
            rng.shuffle(buf)
            yield from buf[: int(size / 2)]
            buf = buf[int(size / 2) :]
    rng.shuffle(buf)
    yield from buf


def _sort(data: Iterable[Utterance], size: int) -> Iterator[Utterance]:
    buf: list[Utterance] = []
    for sample in data:
        buf.append(sample)
        if len(buf) >= size:
            buf.sort(key=lambda x: x.speech_feat.size(0))
            yield from buf
            buf = []
    buf.sort(key=lambda x: x.speech_feat.size(0))
    yield from buf


def _dynamic_batch(data: Iterable[Utterance], max_frames: int) -> Iterator[list[Utterance]]:
    buf: list[Utterance] = []
    longest = 0
    for sample in data:
        frames = sample.speech_feat.size(0)
        longest = max(longest, frames)
        if longest * (len(buf) + 1) > max_frames:
            yield buf
            buf, longest = [sample], frames
        else:
            buf.append(sample)
    if buf:
        yield buf


def padding(sample: list[Utterance]) -> dict[str, Any]:
    """``processor.padding``: the batch ordered by waveform length, longest first, each field padded."""
    order = torch.argsort(torch.tensor([x.speech_len for x in sample], dtype=torch.int32), descending=True).tolist()
    picked = [sample[i] for i in order]
    text_token = [torch.tensor(u.text_token) for u in picked]
    speech_feat = [u.speech_feat for u in picked]
    speech_token = [torch.tensor(u.speech_token) for u in picked]
    batch: dict[str, Any] = {
        "utts": [u.utt for u in picked],
        "text": [u.text for u in picked],
        "text_token_len": torch.tensor([t.size(0) for t in text_token], dtype=torch.int32),
        "text_token": pad_sequence(text_token, batch_first=True, padding_value=0),
        "speech_feat_len": torch.tensor([t.size(0) for t in speech_feat], dtype=torch.int32),
        "speech_feat": pad_sequence(speech_feat, batch_first=True, padding_value=0),
        "embedding": torch.stack([u.embedding for u in picked], dim=0),
        "speech_token_len": torch.tensor([t.size(0) for t in speech_token], dtype=torch.int32),
        "speech_token": pad_sequence(speech_token, batch_first=True, padding_value=0),
    }
    if all(u.instruct_token is not None for u in picked):
        instruct = [torch.tensor(u.instruct_token) for u in picked]
        batch["instruct_token_len"] = torch.tensor([t.size(0) for t in instruct], dtype=torch.int32)
        batch["instruct_token"] = pad_sequence(instruct, batch_first=True, padding_value=0)
    return batch


def _pipeline(utts: list[Utterance | None], groups: list[list[int]], picked: list[int], rng: Any, recipe: Recipe) -> Iterator[dict[str, Any]]:
    def opened() -> Iterator[Utterance]:
        for shard in picked:
            for index in groups[shard]:
                u = utts[index]
                if u is not None:
                    yield u

    for sample in _dynamic_batch(_sort(_shuffle(opened(), rng, recipe.shuffle_size), recipe.sort_size), recipe.max_frames_in_batch):
        yield padding(sample)


def batches(utts: list[Utterance | None], recipe: Recipe, epoch: int, workers: int, train: bool) -> Iterator[dict[str, Any]]:
    """One pass of the data loader (iterate it at once: creating it draws the loader's base seed, as ``iter`` does)."""
    base_seed = int(torch.empty((), dtype=torch.int64).random_().item())
    groups = shards(len(utts))
    if workers <= 0:
        return _pipeline(utts, groups, sampled(len(groups), epoch, 0, 1, train), random, recipe)
    streams = [_pipeline(utts, groups, sampled(len(groups), epoch, k, workers, train), random.Random(base_seed + k), recipe) for k in range(workers)]
    return _round_robin(streams)


def _round_robin(streams: list[Iterator[dict[str, Any]]]) -> Iterator[dict[str, Any]]:
    """The multi-process loader's order: one batch from each worker in turn, a worker leaving when it is done."""
    active = list(streams)
    while active:
        for stream in list(active):
            try:
                yield next(stream)
            except StopIteration:
                active.remove(stream)


# -- the models ---------------------------------------------------------------------------------------------------


_ORDER = {
    ("llm", 2): ("llm_embedding.", "llm.model.model.embed_tokens.", "llm.", "llm_decoder.", "speech_embedding."),
    ("llm", 3): ("llm.model.model.embed_tokens.", "llm.", "llm_decoder.", "speech_embedding."),
    ("flow", 1): ("input_embedding.", "spk_embed_affine_layer.", "encoder.", "encoder_proj.", "decoder.", "length_regulator."),
}
"""Where the native modules register parameters in another order than upstream's (the gradient norm sums in it)."""


def build(model_dir: Path, version: int, part: str, config: dict[str, Any]) -> nn.Module:
    """The part as the engine builds it, with the model's weights (no random initialisation: nothing is drawn)."""
    from mizuoto.engines.cosyvoice.flow import Flow
    from mizuoto.engines.cosyvoice.llm import QwenLM, TransformerLM
    from mizuoto.runtime import ops
    from mizuoto.runtime.loading import load_into

    state = torch.load(str(model_dir / f"{part}.pt"), map_location="cpu", weights_only=True, mmap=True)
    module: nn.Module
    if part == "llm":
        cfg = config["llm"]
        module = TransformerLM(cfg, ops.default) if version == 1 else QwenLM(version, model_dir / "CosyVoice-BlankEN", int(cfg.get("speech_token_size", 6561)), ops.default)
        load_into(module, state, ignore=module.ignorable(), what="CosyVoice language model")
    else:
        module = Flow(version, config["flow"], ops.default)
        load_into(module, state, what="CosyVoice flow")
    return module


def _upstream_order(names: list[str], part: str, version: int) -> list[str]:
    prefixes = _ORDER.get((part, version))
    if prefixes is None:
        return names
    return sorted(names, key=lambda name: next(i for i, p in enumerate(prefixes) if name.startswith(p)))


def ordered_parameters(module: nn.Module, part: str, version: int) -> list[nn.Parameter]:
    """``model.parameters()`` in upstream's order."""
    params = dict(module.named_parameters())
    return [params[name] for name in _upstream_order(list(params), part, version)]


def checkpoint_state(module: nn.Module, part: str, version: int) -> dict[str, torch.Tensor]:
    """``model.state_dict()`` as upstream saves it: its keys in its order, Qwen's tied ``lm_head`` (the token embedding)
    after the decoder's final norm."""
    own = module.state_dict()
    keys = _upstream_order(list(own), part, version)
    if part == "llm" and version > 1:
        last = max(i for i, k in enumerate(keys) if k.startswith("llm."))
        keys.insert(last + 1, "llm.model.lm_head.weight")
        own["llm.model.lm_head.weight"] = own["llm.model.model.embed_tokens.weight"]
    return {k: own[k] for k in keys}


# -- training forwards: the encoders with their dropout -------------------------------------------------------


def _embed(embed: Any, x: torch.Tensor, training: bool) -> tuple[torch.Tensor, torch.Tensor]:
    """``LinearNoSubsampling`` / ``LegacyLinearNoSubsampling`` and ``EspnetRelPositionalEncoding``."""
    out = embed.out
    x = F.dropout(out[1](out[0](x)), DROPOUT, training)
    if len(out) > 3:
        x = out[3](x)
    pe = embed.pos_enc
    pe.extend_pe(x)
    x = x * pe.xscale
    pos = pe.position_encoding(0, x.size(1))
    return F.dropout(x, POSITIONAL_DROPOUT, training), F.dropout(pos, POSITIONAL_DROPOUT, training)


def _attention(att: Any, x: torch.Tensor, mask: torch.Tensor, pos_emb: torch.Tensor, p: float, training: bool) -> torch.Tensor:
    """``RelPositionMultiHeadedAttention`` with its attention-probability dropout."""
    n_batch = x.size(0)
    q = att.linear_q(x).view(n_batch, -1, att.h, att.d_k).transpose(1, 2)
    k = att.linear_k(x).view(n_batch, -1, att.h, att.d_k).transpose(1, 2)
    v = att.linear_v(x).view(n_batch, -1, att.h, att.d_k).transpose(1, 2)
    if att.relative:
        q = q.transpose(1, 2)
        pos = att.linear_pos(pos_emb).view(pos_emb.size(0), -1, att.h, att.d_k).transpose(1, 2)
        q_u = (q + att.pos_bias_u).transpose(1, 2)
        q_v = (q + att.pos_bias_v).transpose(1, 2)
        ac = torch.matmul(q_u, k.transpose(-2, -1))
        bd = torch.matmul(q_v, pos.transpose(-2, -1))
        if ac.shape != bd.shape:
            bd = att.rel_shift(bd)
        scores = (ac + bd) / math.sqrt(att.d_k)
    else:
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(att.d_k)
    if mask.size(2) > 0:
        m = mask.unsqueeze(1).eq(0)[:, :, :, : scores.size(-1)]
        attn = torch.softmax(scores.masked_fill(m, -float("inf")), dim=-1).masked_fill(m, 0.0)
    else:
        attn = torch.softmax(scores, dim=-1)
    out = torch.matmul(F.dropout(attn, p, training), v)
    return att.linear_out(out.transpose(1, 2).contiguous().view(n_batch, -1, att.h * att.d_k))


def _layers(layers: Any, x: torch.Tensor, mask: torch.Tensor, pos_emb: torch.Tensor, p_att: float, training: bool) -> torch.Tensor:
    """WeNet's ``ConformerEncoderLayer`` / ``TransformerEncoderLayer`` (no macaron, no convolution) with dropout."""
    for layer in layers:
        conformer = isinstance(layer, ConformerEncoderLayer)
        norm_att, norm_ff = (layer.norm_mha, layer.norm_ff) if conformer else (layer.norm1, layer.norm2)
        before = layer.normalize_before
        residual = x
        if before:
            x = norm_att(x)
        x = residual + F.dropout(_attention(layer.self_attn, x, mask, pos_emb, p_att, training), DROPOUT, training)
        if not before:
            x = norm_att(x)
        residual = x
        if before:
            x = norm_ff(x)
        ff = layer.feed_forward
        y = F.dropout(ff.w_2(F.dropout(ff.activation(ff.w_1(x)), DROPOUT, training)), DROPOUT, training)
        x = residual + (1.0 * y if conformer else y)
        if not before:
            x = norm_ff(x)
    return x


def encoder_forward(enc: Any, xs: torch.Tensor, xs_lens: torch.Tensor, p_att: float) -> tuple[torch.Tensor, torch.Tensor]:
    """``BaseEncoder.forward`` (no dynamic chunks: the static chunk mask, if any)."""
    masks = ~make_pad_mask(xs_lens, xs.size(1)).unsqueeze(1)
    xs, pos_emb = _embed(enc.embed, xs, enc.training)
    chunk_masks = add_optional_chunk_mask(xs, masks, enc.static_chunk_size)
    xs = _layers(enc.encoders, xs, chunk_masks, pos_emb, p_att, enc.training)
    if enc.normalize_before:
        xs = enc.after_norm(xs)
    return xs, masks


def upsample_encoder_forward(enc: Any, xs: torch.Tensor, xs_lens: torch.Tensor, streaming: bool) -> tuple[torch.Tensor, torch.Tensor]:
    """``UpsampleConformerEncoder.forward``."""
    training = enc.training
    masks = ~make_pad_mask(xs_lens, xs.size(1)).unsqueeze(1)
    xs, pos_emb = _embed(enc.embed, xs, training)
    chunk_masks = add_optional_chunk_mask(xs, masks, enc.static_chunk_size if streaming else 0)
    xs = enc.pre_lookahead_layer(xs)
    xs = _layers(enc.encoders, xs, chunk_masks, pos_emb, FLOW_ATTENTION_DROPOUT, training)
    xs, xs_lens = enc.up_layer(xs.transpose(1, 2).contiguous(), xs_lens)
    xs = xs.transpose(1, 2).contiguous()
    masks = ~make_pad_mask(xs_lens, xs.size(1)).unsqueeze(1)
    xs, pos_emb = _embed(enc.up_embed, xs, training)
    chunk_masks = add_optional_chunk_mask(xs, masks, enc.static_chunk_size * enc.up_layer.stride if streaming else 0)
    xs = _layers(enc.up_encoders, xs, chunk_masks, pos_emb, FLOW_ATTENTION_DROPOUT, training)
    return enc.after_norm(xs), masks


# -- the language models --------------------------------------------------------------------------------------


def label_smoothing_loss(x: torch.Tensor, target: torch.Tensor, size: int) -> torch.Tensor:
    """``LabelSmoothingLoss(smoothing=0, normalize_length=True)``: the KL divergence to the one-hot target, summed over
    every class and averaged over the targets that count."""
    x = x.view(-1, size)
    target = target.view(-1)
    true_dist = torch.zeros_like(x)
    true_dist.fill_(0.0 / (size - 1))
    ignore = target == IGNORE_ID
    total = len(target) - ignore.sum().item()
    target = target.masked_fill(ignore, 0)
    true_dist.scatter_(1, target.unsqueeze(1), 1.0)
    kl = F.kl_div(torch.log_softmax(x, dim=1), true_dist, reduction="none")
    return kl.masked_fill(ignore.unsqueeze(1), 0).sum() / total


def accuracy(pad_outputs: torch.Tensor, pad_targets: torch.Tensor) -> torch.Tensor:
    """``th_accuracy``."""
    pred = pad_outputs.view(pad_targets.size(0), pad_targets.size(1), pad_outputs.size(1)).argmax(2)
    mask = pad_targets != IGNORE_ID
    numerator = torch.sum(pred.masked_select(mask) == pad_targets.masked_select(mask))
    return (numerator / torch.sum(mask)).detach()


def _unpad(x: torch.Tensor, lengths: torch.Tensor) -> list[torch.Tensor]:
    return [x[i, : int(n)] for i, n in enumerate(lengths.cpu())]


def transformer_lm_loss(lm: Any, batch: dict[str, Any], device: torch.device) -> dict[str, torch.Tensor]:
    """``TransformerLM.forward`` (CosyVoice 1)."""
    text_token = batch["text_token"].to(device)
    text_token_len = batch["text_token_len"].to(device)
    speech_token = batch["speech_token"].to(device)
    speech_token_len = batch["speech_token_len"].to(device)
    embedding = batch["embedding"].to(device)
    lm_target = [torch.tensor([IGNORE_ID] * (2 + text_token_len[i]) + speech_token[i, : speech_token_len[i]].tolist() + [lm.speech_token_size]) for i in range(text_token.size(0))]
    target = pad_sequence(lm_target, batch_first=True, padding_value=IGNORE_ID).to(device)
    text = lm.text_embedding(text_token)
    text, mask = encoder_forward(lm.text_encoder, text, text_token_len, LLM_ATTENTION_DROPOUT)
    text_len = mask.squeeze(1).sum(1)
    text = lm.text_encoder_affine_layer(text)
    embedding = lm.spk_embed_affine_layer(F.normalize(embedding, dim=1)).unsqueeze(1)
    sos = lm.llm_embedding.weight[lm.sos].reshape(1, 1, -1)
    task = lm.llm_embedding.weight[lm.task_id].reshape(1, 1, -1)
    speech = lm.speech_embedding(speech_token)
    texts, speeches = _unpad(text, text_len), _unpad(speech, speech_token_len)
    rows = [torch.concat([sos.squeeze(dim=0), embedding[i], texts[i], task.squeeze(dim=0), speeches[i]], dim=0) for i in range(len(texts))]
    lm_input_len = torch.tensor([r.size(0) for r in rows], dtype=torch.int32)
    lm_input = pad_sequence(rows, batch_first=True, padding_value=IGNORE_ID)
    out, _ = encoder_forward(lm.llm, lm_input, lm_input_len.to(device), LLM_ATTENTION_DROPOUT)
    logits = lm.llm_decoder(out)
    size = lm.speech_token_size + 1
    return {"loss": label_smoothing_loss(logits, target, size), "acc": accuracy(logits.view(-1, size), target)}


def causal_mask_451(attention_mask: torch.Tensor, x: torch.Tensor, sliding_window: int | None) -> torch.Tensor | None:
    """transformers 4.51's ``Qwen2Model._update_causal_mask`` for SDPA with no cache: none when nothing is padded, else
    the additive causal mask with the padded keys at the dtype's minimum."""
    batch, length = attention_mask.shape
    if bool(torch.all(attention_mask == 1)) and (sliding_window is None or length < sliding_window):
        return None
    min_dtype = torch.finfo(x.dtype).min
    cache_position = torch.arange(length, device=x.device)
    causal = torch.full((length, length), fill_value=min_dtype, dtype=x.dtype, device=x.device)
    diagonal = torch.arange(length, device=x.device) > cache_position.reshape(-1, 1)
    if sliding_window is not None:
        diagonal.bitwise_or_(torch.arange(length, device=x.device) <= (cache_position.reshape(-1, 1) - sliding_window))
    causal *= diagonal
    causal = causal[None, None, :, :].expand(batch, 1, -1, -1).clone()
    padding = (causal[:, :, :, :length] + attention_mask[:, None, None, :].to(causal.device)) == 0
    causal[:, :, :, :length] = causal[:, :, :, :length].masked_fill(padding, min_dtype)
    return causal


def qwen_hidden(lm: Any, xs: torch.Tensor, xs_lens: torch.Tensor, sliding_window: int | None) -> torch.Tensor:
    """``Qwen2Encoder.forward``: the last hidden state (after the final norm) under the padding mask."""
    masks = ~make_pad_mask(xs_lens, xs.size(1))
    positions = torch.arange(xs.size(1), device=xs.device).unsqueeze(0)
    return lm.llm.model.model(xs, positions, causal_mask_451(masks, xs, sliding_window))


def sliding_window(model_dir: Path) -> int | None:
    """``Qwen2Config.sliding_window`` (4.51 applies it to the mask whether or not ``use_sliding_window`` is set)."""
    return json.loads((model_dir / "CosyVoice-BlankEN" / "config.json").read_text(encoding="utf-8")).get("sliding_window", 4096)


def prepare_lm_input_target(
    lm: Any,
    sos_emb: torch.Tensor,
    text_token: torch.Tensor,
    text_token_emb: torch.Tensor,
    text_token_len: torch.Tensor,
    task_id_emb: torch.Tensor,
    speech_token: torch.Tensor,
    speech_token_emb: torch.Tensor,
    speech_token_len: torch.Tensor,
    instruct: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """``Qwen2LM.prepare_lm_input_target``: each utterance as one sequence, or (half the time, when its speech is long
    enough) as the bidirectional-streaming sequence of 5 text tokens and 15 speech tokens with fill tokens."""
    r_text, r_speech = MIX_RATIO
    texts, speeches = _unpad(text_token, text_token_len), _unpad(speech_token, speech_token_len)
    text_embs, speech_embs = _unpad(text_token_emb, text_token_len), _unpad(speech_token_emb, speech_token_len)
    if instruct is not None:
        instruct_len = instruct[2]
        instruct_embs = _unpad(instruct[1], instruct_len)
    else:
        instruct_embs = [torch.empty(0, text_token_emb.shape[-1]).to(text_embs[0])] * len(texts)
        instruct_len = torch.zeros(len(texts)).to(text_token_len)
    targets, inputs = [], []
    for i in range(len(texts)):
        if random.random() < 0.5 and speech_token_len[i] / text_token_len[i] > r_speech / r_text:
            target: list[int] = [IGNORE_ID]
            parts = [sos_emb.squeeze(dim=0)]
            target += [IGNORE_ID] * instruct_len[i]
            parts.append(instruct_embs[i])
            for j in range(int(((text_token_len[i] + 1) / r_text).ceil().int().item())):
                this_text = texts[i][j * r_text : (j + 1) * r_text].tolist()
                this_speech = speeches[i][j * r_speech : (j + 1) * r_speech].tolist()
                if len(this_text) == r_text:
                    assert len(this_speech) == r_speech
                    target += [IGNORE_ID] * (r_text - 1)
                    target += this_speech
                    target.append(lm.fill_token)
                    parts.append(text_embs[i][j * r_text : (j + 1) * r_text])
                    parts.append(speech_embs[i][j * r_speech : (j + 1) * r_speech])
                else:
                    target += [-1] * len(this_text)
                    target += speeches[i][j * r_speech :].tolist()
                    target.append(lm.eos_token)
                    parts.append(text_embs[i][j * r_text :])
                    parts.append(task_id_emb.squeeze(dim=0))
                    parts.append(speech_embs[i][j * r_speech :])
            this_target, this_input = torch.tensor(target), torch.concat(parts, dim=0)
        else:
            this_target = torch.tensor([IGNORE_ID] * (1 + instruct_len[i] + text_token_len[i]) + speeches[i].tolist() + [lm.eos_token])
            this_input = torch.concat([sos_emb.squeeze(dim=0), instruct_embs[i], text_embs[i], task_id_emb.squeeze(dim=0), speech_embs[i]], dim=0)
        targets.append(this_target)
        inputs.append(this_input)
    lm_input_len = torch.tensor([t.size(0) for t in inputs], dtype=torch.int32)
    lm_input = pad_sequence(inputs, batch_first=True, padding_value=IGNORE_ID)
    lm_target = pad_sequence(targets, batch_first=True, padding_value=IGNORE_ID)
    return lm_target, lm_input, lm_input_len


def qwen_lm_loss(lm: Any, batch: dict[str, Any], device: torch.device, window: int | None = None) -> dict[str, torch.Tensor]:
    """``Qwen2LM.forward`` (CosyVoice 2) and ``CosyVoice3LM``'s (its instruction tokens before the text)."""
    embed = lm.llm.model.embed
    text_token = batch["text_token"].to(device)
    text_token_len = batch["text_token_len"].to(device)
    text_token_emb = embed(text_token)
    speech_token = batch["speech_token"].to(device)
    speech_token_len = batch["speech_token_len"].to(device)
    speech_token_emb = lm.speech_embedding(speech_token)
    table = lm.speech_embedding if lm.version == 3 else lm.llm_embedding
    sos_emb = table.weight[lm.sos].reshape(1, 1, -1)
    task_id_emb = table.weight[lm.task_id].reshape(1, 1, -1)
    instruct = None
    if lm.version == 3:
        instruct_token = batch["instruct_token"].to(device)
        instruct = (instruct_token, embed(instruct_token), batch["instruct_token_len"].to(device))
    lm_target, lm_input, lm_input_len = prepare_lm_input_target(
        lm, sos_emb, text_token, text_token_emb, text_token_len, task_id_emb, speech_token, speech_token_emb, speech_token_len, instruct
    )
    lm_target = lm_target.to(device)
    logits = lm.llm_decoder(qwen_hidden(lm, lm_input, lm_input_len.to(device), window))
    size = lm.llm_decoder.out_features
    return {"loss": label_smoothing_loss(logits, lm_target.to(device), size), "acc": accuracy(logits.view(-1, size), lm_target)}


# -- the flows ------------------------------------------------------------------------------------------------


def _conds(feat: torch.Tensor, feat_len: torch.Tensor) -> torch.Tensor:
    """Half the utterances keep a random prefix of up to 30 % of their mel as the condition."""
    conds = torch.zeros(feat.shape, device=feat.device)
    for i, j in enumerate(feat_len):
        if random.random() < 0.5:
            continue
        index = random.randint(0, int(0.3 * j))
        conds[i, :index] = feat[i, :index]
    return conds.transpose(1, 2)


def cfm_loss(
    estimator: Callable[..., torch.Tensor], x1: torch.Tensor, mask: torch.Tensor, mu: torch.Tensor, spks: torch.Tensor, cond: torch.Tensor, streaming: bool
) -> torch.Tensor:
    """``ConditionalCFM.compute_loss``."""
    b = mu.shape[0]
    t = torch.rand([b, 1, 1], device=mu.device, dtype=mu.dtype)
    z = torch.randn_like(x1)
    y = (1 - (1 - SIGMA_MIN) * t) * z + t * x1
    u = x1 - (1 - SIGMA_MIN) * z
    cfg_mask = torch.rand(b, device=x1.device) > TRAINING_CFG_RATE
    mu = mu * cfg_mask.view(-1, 1, 1)
    spks = spks * cfg_mask.view(-1, 1)
    cond = cond * cfg_mask.view(-1, 1, 1)
    pred = estimator(y, mask, mu, t.squeeze(), spks, cond, streaming=streaming)
    return F.mse_loss(pred * mask, u * mask, reduction="sum") / (torch.sum(mask) * u.shape[1])


def _dit_forward(dit: Any, x: torch.Tensor, mask: torch.Tensor, mu: torch.Tensor, t: torch.Tensor, spks: torch.Tensor, cond: torch.Tensor, streaming: bool = False) -> torch.Tensor:
    """CosyVoice 3's ``DiT.forward`` with its dropout (after each attention projection and inside each feed-forward)."""
    from mizuoto.nn.transformer.dit import apply_rotary

    training = dit.training
    x, mu, cond = x.transpose(1, 2), mu.transpose(1, 2), cond.transpose(1, 2)
    batch, seq_len = x.shape[0], x.shape[1]
    if t.ndim == 0:
        t = t.repeat(batch)
    t = dit.time_embed(t)
    x = dit.input_embed(x, cond, mu, spks)
    rope = dit.rotary_embed.forward_from_seq_len(seq_len)
    if streaming:
        attn_mask = add_optional_chunk_mask(x, mask.bool(), dit.static_chunk_size).unsqueeze(dim=1)
    else:
        attn_mask = add_optional_chunk_mask(x, mask.bool(), 0).repeat(1, x.size(1), 1).unsqueeze(dim=1)
    attn_mask = attn_mask.bool()
    for block in dit.transformer_blocks:
        norm, gate_msa, shift_mlp, scale_mlp, gate_mlp = block.attn_norm(x, emb=t)
        att = block.attn
        q = apply_rotary(att.to_q(norm), rope, 1.0)
        k = apply_rotary(att.to_k(norm), rope, 1.0)
        v = att.to_v(norm)
        head_dim = k.shape[-1] // att.heads
        q = q.view(batch, -1, att.heads, head_dim).transpose(1, 2)
        k = k.view(batch, -1, att.heads, head_dim).transpose(1, 2)
        v = v.view(batch, -1, att.heads, head_dim).transpose(1, 2)
        out = attention(q, k, v, site=DIT_SITE, mask=attn_mask)
        out = out.transpose(1, 2).reshape(batch, -1, att.heads * head_dim).to(q.dtype)
        out = F.dropout(att.to_out[0](out), DIT_DROPOUT, training).masked_fill(~attn_mask[:, 0, -1].unsqueeze(-1), 0.0)
        x = x + gate_msa.unsqueeze(1) * out
        norm = block.ff_norm(x) * (1 + scale_mlp[:, None]) + shift_mlp[:, None]
        ff = block.ff.ff
        x = x + gate_mlp.unsqueeze(1) * ff[2](F.dropout(ff[0](norm), DIT_DROPOUT, training))
    return dit.proj_out(dit.norm_out(x, t)).transpose(1, 2)


def _regulate(regulator: Any, x: torch.Tensor, ylens: torch.Tensor) -> torch.Tensor:
    """``InterpolateRegulator.forward``: the encoder's output stretched to the mel's length."""
    mask = (~make_pad_mask(ylens)).to(x).unsqueeze(-1)
    x = F.interpolate(x.transpose(1, 2).contiguous(), size=int(ylens.max()), mode="linear")
    return regulator.model(x).transpose(1, 2).contiguous() * mask


def flow_loss(flow: Any, batch: dict[str, Any], device: torch.device) -> dict[str, torch.Tensor]:
    """``MaskedDiffWithXvec`` (v1), ``CausalMaskedDiffWithXvec`` (v2) and ``CausalMaskedDiffWithDiT.forward`` (v3)."""
    token = batch["speech_token"].to(device)
    token_len = batch["speech_token_len"].to(device)
    feat = batch["speech_feat"].to(device)
    feat_len = batch["speech_feat_len"].to(device)
    embedding = batch["embedding"].to(device)
    v = flow.version
    streaming = False if v == 1 else random.random() < 0.5
    embedding = flow.spk_embed_affine_layer(F.normalize(embedding, dim=1))
    mask = (~make_pad_mask(token_len)).float().unsqueeze(-1).to(device)
    token = flow.input_embedding(torch.clamp(token, min=0)) * mask
    estimator = flow.decoder.estimator
    if v == 1:
        h, _ = encoder_forward(flow.encoder, token, token_len, FLOW_ATTENTION_DROPOUT)
        h = _regulate(flow.length_regulator, flow.encoder_proj(h), feat_len)
        conds = _conds(feat, feat_len)
        mel_mask = (~make_pad_mask(feat_len)).to(h)
    elif v == 2:
        h, h_lengths = upsample_encoder_forward(flow.encoder, token, token_len, streaming)
        h = flow.encoder_proj(h)
        conds = _conds(feat, feat_len)
        mel_mask = (~make_pad_mask(h_lengths.sum(dim=-1).squeeze(dim=1))).to(h)
    else:
        h = flow.pre_lookahead_layer(token).repeat_interleave(flow.token_mel_ratio, dim=1)
        mel_mask = mask.repeat_interleave(flow.token_mel_ratio, dim=1).squeeze(dim=-1)
        conds = _conds(feat, feat_len)

        def estimator(*args: Any, **kwargs: Any) -> torch.Tensor:
            return _dit_forward(flow.decoder.estimator, *args, **kwargs)

    loss = cfm_loss(estimator, feat.transpose(1, 2).contiguous(), mel_mask.unsqueeze(1), h.transpose(1, 2).contiguous(), embedding, conds, streaming)
    return {"loss": loss}


# -- the loop -------------------------------------------------------------------------------------------------


class WarmupLR(torch.optim.lr_scheduler.LRScheduler):
    """Upstream's ``WarmupLR`` (CosyVoice 1's recipe): the rate grows as the step to ``warmup_steps``, then decays as
    its inverse square root."""

    def __init__(self, optimizer: torch.optim.Optimizer, warmup_steps: int = 25000) -> None:
        self.warmup_steps = warmup_steps
        super().__init__(optimizer)

    # list[Any]: torch 2.8 declares get_lr() -> list[float], later versions list[float | Tensor] (a tensor rate stays one).
    def get_lr(self) -> list[Any]:
        step = self.last_epoch + 1
        return [lr * self.warmup_steps**0.5 * min(step**-0.5, step * self.warmup_steps**-1.5) for lr in self.base_lrs]


class ConstantLR(torch.optim.lr_scheduler.LRScheduler):
    """Upstream's ``ConstantLR`` (CosyVoice 2's and 3's recipes)."""

    def get_lr(self) -> list[Any]:
        return list(self.base_lrs)


@dataclass
class TrainConfig:
    part: str
    """``llm`` or ``flow``."""
    version: int
    out: Path
    """Where ``epoch_<N>_whole.pt`` go."""
    epochs: int = 10
    lr: float = 1e-5
    accum_grad: int = 2
    workers: int = 2
    """The data loader's ``num_workers`` (v1's ``--num_workers``): how the data is shuffled and repeated."""
    device: str = "cpu"
    amp: bool = False
    """bf16 autocast with a gradient scaler (``--use_amp``; CUDA only)."""
    keep: int = 3
    window: int | None = None
    """Qwen2's ``sliding_window`` (CosyVoice 2 and 3)."""


def train(
    model: nn.Module,
    utts: list[Utterance | None],
    cfg: TrainConfig,
    report: Callable[[int, int, dict[str, float], float], None],
    log: Callable[[str], None],
) -> dict[str, Any]:
    """``bin/train.py`` from the start (step 0, epoch 0) for ``cfg.epochs`` epochs: ``train_one_epoc`` then ``cv`` and
    ``epoch_<N>_whole.pt`` each epoch. ``report(epoch, batch, values, lr)`` gets each batch's logged values."""
    recipe = RECIPES[cfg.version]
    device = torch.device(cfg.device)
    model.to(device)
    params = ordered_parameters(model, cfg.part, cfg.version)
    optimizer = torch.optim.Adam(params, lr=cfg.lr)
    scheduler: torch.optim.lr_scheduler.LRScheduler = WarmupLR(optimizer, recipe.warmup_steps) if recipe.scheduler == "warmuplr" else ConstantLR(optimizer)
    scaler = torch.amp.GradScaler("cuda") if cfg.amp else None
    forward = flow_loss if cfg.part == "flow" else transformer_lm_loss if cfg.version == 1 else functools.partial(qwen_lm_loss, window=cfg.window)
    cfg.out.mkdir(parents=True, exist_ok=True)
    step = 0
    reported = 0
    saved: list[Path] = []
    for epoch in range(cfg.epochs):
        model.train()
        for batch_idx, batch in enumerate(batches(utts, recipe, epoch, cfg.workers, train=True)):
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=scaler is not None):
                losses = forward(model, batch, device)
            scaled = losses["loss"] / cfg.accum_grad
            (scaler.scale(scaled) if scaler is not None else scaled).backward()
            losses["loss"] = scaled
            if (batch_idx + 1) % cfg.accum_grad == 0:
                if scaler is not None:
                    scaler.unscale_(optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(params, GRAD_CLIP)
                if not torch.isfinite(grad_norm):
                    log("Infinite gradient norm: the update is skipped")
                elif scaler is not None:
                    scaler.step(optimizer)
                else:
                    optimizer.step()
                if scaler is not None:
                    scaler.update()
                optimizer.zero_grad()
                scheduler.step()
            reported += 1
            report(epoch, reported, {k: float(v.item()) for k, v in losses.items()}, float(optimizer.param_groups[0]["lr"]))
            if (batch_idx + 1) % cfg.accum_grad == 0:
                step += 1
        cv = validate(model, utts, recipe, epoch, cfg.workers, forward, device)
        log(f"Epoch {epoch + 1}: validation " + " ".join(f"{k} {v:.6f}" for k, v in cv.items()))
        path = cfg.out / f"epoch_{epoch}_whole.pt"
        torch.save({**checkpoint_state(model, cfg.part, cfg.version), "epoch": epoch, "step": step}, path)
        saved.append(path)
        for old in saved[: -max(1, cfg.keep)]:
            old.unlink(missing_ok=True)
        saved = saved[-max(1, cfg.keep) :]
    return {"epochs": cfg.epochs, "steps": step, "checkpoint": saved[-1].name if saved else None}


def validate(
    model: nn.Module, utts: list[Utterance | None], recipe: Recipe, epoch: int, workers: int, forward: Callable[..., dict[str, torch.Tensor]], device: torch.device
) -> dict[str, float]:
    """``Executor.cv``: the losses on the validation list (the training list, as v1 runs it), in eval mode."""
    model.eval()
    totals: dict[str, float] = {}
    count = 0
    with torch.inference_mode():
        for batch in batches(utts, recipe, epoch, workers, train=False):
            n = len(batch["utts"])
            count += n
            for k, v in forward(model, batch, device).items():
                totals[k] = totals.get(k, 0.0) + v.mean().item() * n
    return {k: v / max(1, count) for k, v in totals.items()}


def read_items(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
