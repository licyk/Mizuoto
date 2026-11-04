"""Paraformer-large (FunASR's ``Paraformer``): a non-autoregressive recogniser. Ported from FunASR 1.4.21 (MIT,
Copyright (c) 2022 Alibaba Group).

- ``WavFrontend``: Kaldi's filterbank (80 bins, 25/10 ms, Hamming, ``energy_floor`` 0) of the waveform scaled to 16-bit
  range with ``dither`` 1.0, then low frame rate (LFR: 7 frames stacked every 6) and the global CMVN of ``am.mvn``.
  Upstream draws the dither from torch's global generator; here from the generator the caller passes.
- ``SANMEncoder`` (50 layers), ``CifPredictorV2`` (continuous integrate-and-fire: a weight per frame, a token each time
  the running sum crosses 1.0, its acoustic embedding the weighted frames; a tail weight of 0.45 after the last frame),
  ``ParaformerSANMDecoder`` (16 layers over the acoustic embeddings, then one feed-forward layer) and the argmax of its
  log-softmax: Paraformer's inference without a language model or CTC, as the published model is configured.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from mizuoto.engines.asr.paraformer.sanm import DecoderLayer, Encoder, layer_norm
from mizuoto.nn.dsp import kaldi
from mizuoto.runtime import ops as default_ops

BLANK, SOS, EOS = 0, 1, 2
"""``<blank>``, ``<s>`` and ``</s>`` (Paraformer's defaults; its token list starts with them)."""


def load_cmvn(path: Path) -> torch.Tensor:
    """``load_cmvn``: the means (``<AddShift>``) and scales (``<Rescale>``) of a Kaldi nnet ``am.mvn``, (2, dim)."""
    lines = [line for line in path.read_text(encoding="utf-8").splitlines() if line.split()]
    means: list[str] = []
    scales: list[str] = []
    for i, line in enumerate(lines):
        item = line.split()
        if item[0] in ("<AddShift>", "<Rescale>") and i + 1 < len(lines):
            following = lines[i + 1].split()
            if following and following[0] == "<LearnRateCoef>":
                values = following[3 : len(following) - 1]
                if item[0] == "<AddShift>":
                    means = values
                else:
                    scales = values
    if not means or not scales:
        raise ValueError(f"No <AddShift>/<Rescale> statistics in {path}")
    return torch.as_tensor(np.array([np.array(means).astype(np.float32), np.array(scales).astype(np.float32)]), dtype=torch.float32)


def apply_lfr(inputs: torch.Tensor, lfr_m: int, lfr_n: int) -> torch.Tensor:
    """``apply_lfr``: ``lfr_m`` frames side by side every ``lfr_n``, the first repeated before, the last after."""
    t = inputs.shape[0]
    t_lfr = int(np.ceil(t / lfr_n))
    left = inputs[0].repeat((lfr_m - 1) // 2, 1)
    inputs = torch.vstack((left, inputs))
    t = t + (lfr_m - 1) // 2
    feat_dim = inputs.shape[-1]
    last_idx = (t - lfr_m) // lfr_n + 1
    num_padding = lfr_m - (t - last_idx * lfr_n)
    if num_padding > 0:
        num_padding = (2 * lfr_m - 2 * t + (t_lfr - 1 + last_idx) * lfr_n) / 2 * (t_lfr - last_idx)
        inputs = torch.vstack([inputs] + [inputs[-1:]] * int(num_padding))
    return inputs.as_strided((t_lfr, lfr_m * feat_dim), (lfr_n * feat_dim, 1)).clone().type(torch.float32)


def apply_cmvn(inputs: torch.Tensor, cmvn: torch.Tensor) -> torch.Tensor:
    dim = inputs.shape[1]
    inputs += cmvn[0:1, :dim].to(inputs.device)
    inputs *= cmvn[1:2, :dim].to(inputs.device)
    return inputs.type(torch.float32)


class Frontend:
    """``WavFrontend`` (``frontend_conf`` of the model's config.yaml; ``dither`` 1.0 unless the config says)."""

    def __init__(self, conf: dict[str, Any], cmvn: torch.Tensor | None) -> None:
        self.fs = int(conf.get("fs", 16000))
        self.window = str(conf.get("window", "hamming"))
        self.n_mels = int(conf.get("n_mels", 80))
        self.frame_length = float(conf.get("frame_length", 25))
        self.frame_shift = float(conf.get("frame_shift", 10))
        self.lfr_m = int(conf.get("lfr_m", 1))
        self.lfr_n = int(conf.get("lfr_n", 1))
        self.dither = float(conf.get("dither", 1.0))
        self.cmvn = cmvn

    def __call__(self, waveform: torch.Tensor, generator: torch.Generator | None) -> torch.Tensor:
        """``waveform``: float32 samples in [-1, 1] at ``fs``, (samples,): features (frames, n_mels × lfr_m)."""
        length = waveform.shape[0]
        mat = kaldi.fbank(
            (waveform * (1 << 15)).unsqueeze(0),
            num_mel_bins=self.n_mels,
            frame_length=min(self.frame_length, length / self.fs * 1000),
            frame_shift=self.frame_shift,
            dither=self.dither,
            energy_floor=0.0,
            window_type=self.window,
            sample_frequency=self.fs,
            snip_edges=True,
            generator=generator,
        )
        if self.lfr_m != 1 or self.lfr_n != 1:
            mat = apply_lfr(mat, self.lfr_m, self.lfr_n)
        if self.cmvn is not None:
            mat = apply_cmvn(mat, self.cmvn)
        return mat


class Predictor(nn.Module):
    """``CifPredictorV2``."""

    def __init__(self, size: int, l_order: int, r_order: int, threshold: float, tail_threshold: float, ops: Any = default_ops.default) -> None:
        super().__init__()
        self.pad = (l_order, r_order)
        self.cif_conv1d = ops.Conv1d(size, size, l_order + r_order + 1)
        self.cif_output = ops.Linear(size, 1)
        self.threshold = threshold
        self.tail_threshold = tail_threshold

    def forward(self, hidden: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """``hidden`` (batch, time, size), ``mask`` (batch, 1, time) 1 for real frames: the acoustic embeddings, the token
        counts, the weights and the fires."""
        queries = torch.nn.functional.pad(hidden.transpose(1, 2), self.pad, value=0.0)
        output = torch.relu(self.cif_conv1d(queries)).transpose(1, 2)
        alphas = torch.sigmoid(self.cif_output(output))
        alphas = torch.nn.functional.relu(alphas * 1.0 - 0)  # smooth_factor 1, noise_threshold 0, as upstream computes it
        mask = mask.transpose(-1, -2).float()
        alphas = (alphas * mask).squeeze(-1)
        mask = mask.squeeze(-1)
        token_num = alphas.sum(-1)
        if self.tail_threshold > 0.0:
            hidden, alphas, token_num = self._tail(hidden, alphas, mask)
        embeds, fires = cif_v1(hidden, alphas, self.threshold)
        if self.tail_threshold > 0.0:
            embeds = embeds[:, : int(torch.max(token_num).type(torch.int32).item()), :]
        return embeds, token_num, alphas, fires

    def _tail(self, hidden: torch.Tensor, alphas: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """``tail_process_fn`` with the mask: ``tail_threshold`` added after each item's last frame, a zero frame
        appended, the token count floored."""
        b, _, d = hidden.size()
        zeros_t = torch.zeros((b, 1), dtype=torch.float32, device=alphas.device)
        ones_t = torch.ones_like(zeros_t)
        mask = torch.cat([ones_t, mask], dim=1) - torch.cat([mask, zeros_t], dim=1)
        alphas = torch.add(torch.cat([alphas, zeros_t], dim=1), mask * self.tail_threshold)
        hidden = torch.cat([hidden, torch.zeros((b, 1, d), dtype=hidden.dtype).to(hidden.device)], dim=1)
        return hidden, alphas, torch.floor(alphas.sum(dim=-1))


def cif_v1(hidden: torch.Tensor, alphas: torch.Tensor, threshold: float) -> tuple[torch.Tensor, torch.Tensor]:
    """``cif_v1``: the embeddings of the fired tokens (batch, max tokens, size) and the fires (batch, time)."""
    batch_size, len_time = alphas.size()
    device, dtype = alphas.device, alphas.dtype
    prefix_sum = torch.cumsum(alphas, dim=1, dtype=torch.float64).to(torch.float32)
    prefix_sum_floor = torch.floor(prefix_sum)
    shifted_floor = torch.floor(torch.roll(prefix_sum, 1, dims=1))
    shifted_floor[:, 0] = 0
    fire_idxs = (prefix_sum_floor - shifted_floor) > 0
    fires = torch.zeros(batch_size, len_time, dtype=dtype, device=device)
    fires[fire_idxs] = 1
    fires = fires + prefix_sum - prefix_sum_floor
    hidden_size = hidden.size(2)
    hdtype = hidden.dtype
    if fire_idxs.sum() == 0:
        max_label_len = torch.round(alphas.sum(-1)).int().max()
        return torch.zeros(batch_size, int(max_label_len), hidden_size, dtype=hdtype, device=hidden.device), fires
    prefix_sum_hidden = torch.cumsum(alphas.unsqueeze(-1).repeat((1, 1, hidden_size)) * hidden, dim=1)
    frames = prefix_sum_hidden[fire_idxs]
    shift_frames = torch.roll(frames, 1, dims=0)
    batch_len = fire_idxs.sum(1)
    batch_idxs = torch.cumsum(batch_len, dim=0)
    shift_batch_idxs = torch.roll(batch_idxs, 1, dims=0)
    shift_batch_idxs[0] = 0
    shift_frames[shift_batch_idxs] = 0
    remains = fires - torch.floor(fires)
    remain_frames = remains[fire_idxs].unsqueeze(-1).repeat((1, hidden_size)) * hidden[fire_idxs]
    shift_remain_frames = torch.roll(remain_frames, 1, dims=0)
    shift_remain_frames[shift_batch_idxs] = 0
    frames = frames - shift_frames + shift_remain_frames - remain_frames
    max_label_len = int(torch.round(alphas.sum(-1)).int().max())
    frame_fires = torch.zeros(batch_size, max_label_len, hidden_size, dtype=hdtype, device=hidden.device)
    indices = torch.arange(max_label_len, device=hidden.device).expand(batch_size, -1)
    frame_fires[indices < batch_len.unsqueeze(1)] = frames
    return frame_fires, fires


def sequence_mask(lengths: torch.Tensor, maxlen: int | None = None) -> torch.Tensor:
    """``scama.utils.sequence_mask``: (batch, maxlen) float, 1 for the first ``lengths`` positions."""
    maxlen = int(lengths.max()) if maxlen is None else maxlen
    row = torch.arange(0, maxlen, 1, device=lengths.device)
    return (row < lengths.unsqueeze(-1)).type(torch.float32)


class Decoder(nn.Module):
    """``ParaformerSANMDecoder``: ``att_layer_num`` layers with the memory block and cross-attention, one feed-forward
    layer, a norm and the output layer (the token embedding upstream keeps is not used in inference)."""

    def __init__(self, vocab: int, size: int, heads: int, hidden: int, blocks: int, kernel_size: int, ops: Any = default_ops.default) -> None:
        super().__init__()
        self.decoders = nn.ModuleList([DecoderLayer(size, heads, hidden, kernel_size, True, ops) for _ in range(blocks)])
        self.decoders3 = nn.ModuleList([DecoderLayer(size, heads, hidden, kernel_size, False, ops)])
        self.after_norm = layer_norm(size)
        self.output_layer = ops.Linear(size, vocab)

    def forward(self, memory: torch.Tensor, memory_lens: torch.Tensor, embeds: torch.Tensor, embed_lens: torch.Tensor) -> torch.Tensor:
        tgt_mask = sequence_mask(embed_lens)[:, :, None]
        memory_mask = sequence_mask(memory_lens, memory.size(1))[:, None, :]
        x = embeds
        for layer in (*self.decoders, *self.decoders3):
            x = layer(x, tgt_mask, memory, memory_mask)
        return self.output_layer(self.after_norm(x))


class Paraformer(nn.Module):
    """The encoder, the predictor and the decoder, named as the checkpoint names them."""

    def __init__(self, config: dict[str, Any], vocab: int, ops: Any = default_ops.default) -> None:
        super().__init__()
        enc, dec, pred = config["encoder_conf"], config["decoder_conf"], config["predictor_conf"]
        size = int(enc["output_size"])
        self.encoder = Encoder(int(config["input_size"]), size, int(enc["attention_heads"]), int(enc["linear_units"]), int(enc["num_blocks"]), int(enc["kernel_size"]), ops)
        self.predictor = Predictor(size, int(pred["l_order"]), int(pred["r_order"]), float(pred.get("threshold", 1.0)), float(pred.get("tail_threshold", 0.0)), ops)
        self.decoder = Decoder(vocab, size, int(dec["attention_heads"]), int(dec["linear_units"]), int(dec.get("att_layer_num", dec["num_blocks"])), int(dec["kernel_size"]), ops)

    @staticmethod
    def ignorable() -> list[str]:
        return ["decoder.embed.*"]

    def recognise(self, feats: torch.Tensor, generator: torch.Generator | None = None) -> list[int]:
        """One segment's features (frames, dim) on the model's device and dtype: its token ids (sos, eos and blank
        dropped; Paraformer's greedy decoding, ``yseq = argmax`` over the predicted number of tokens).

        Upstream's layer stacks (``MultiSequential``) draw one uniform number per layer at every forward for their layer
        drop, in inference too, from torch's global generator on the CPU: the encoder's first layer and the other 49, the
        decoder's 16 and its last layer (when a token fired). They decide nothing at inference, but they move the
        generator the next segment's dither draws from, so ``generator`` (the transcription's) draws them as well."""
        speech = feats.unsqueeze(0)
        lengths = torch.tensor([speech.shape[1]], device=speech.device)
        mask = sequence_mask(lengths, speech.shape[1]).bool()[:, None, :]
        _layer_drop(generator, len(self.encoder.encoders0), len(self.encoder.encoders))
        encoder_out = self.encoder(speech, mask)
        embeds, token_num, _, _ = self.predictor(encoder_out, mask)
        token_num = token_num.round().long()
        if int(torch.max(token_num)) < 1:
            return []
        _layer_drop(generator, len(self.decoder.decoders), len(self.decoder.decoders3))
        logits = self.decoder(encoder_out, lengths, embeds, token_num)
        scores = torch.log_softmax(logits, dim=-1)[0, : int(token_num[0])]
        return [t for t in (int(t) for t in scores.argmax(dim=-1).tolist()) if t not in (BLANK, SOS, EOS)]


def _layer_drop(generator: torch.Generator | None, *stacks: int) -> None:
    """The uniform numbers upstream's ``MultiSequential`` draws (``torch.empty(len(self)).uniform_()``) per stack."""
    if generator is not None:
        for n in stacks:
            torch.empty(n).uniform_(generator=generator)


def load_config(folder: Path) -> dict[str, Any]:
    """The model's ``config.yaml`` (FunASR's), read without a YAML library: the keys Paraformer uses."""
    return yaml_subset((folder / "config.yaml").read_text(encoding="utf-8"))


def yaml_subset(text: str) -> dict[str, Any]:
    """The plain YAML FunASR's configs use (nested mappings, lists of scalars, scalars), parsed by indentation."""
    root: dict[str, Any] = {}
    stack: list[tuple[int, Any]] = [(-1, root)]
    pending: tuple[int, dict[str, Any], str] | None = None
    for raw in text.splitlines():
        line = raw.split(" #")[0].rstrip() if not raw.lstrip().startswith("#") else ""
        if not line.strip():
            continue
        indent = len(line) - len(line.lstrip())
        content = line.strip()
        if pending is not None:
            p_indent, p_parent, p_key = pending
            pending = None
            if indent > p_indent:
                container: Any = [] if content.startswith("- ") else {}
                p_parent[p_key] = container
                stack.append((p_indent, container))
            else:
                p_parent[p_key] = None
        while stack and indent <= stack[-1][0]:
            stack.pop()
        parent = stack[-1][1]
        if content.startswith("- "):
            item = content[2:].strip()
            if isinstance(parent, list):
                parent.append(_scalar(item))
            continue
        key, _, value = content.partition(":")
        key, value = key.strip(), value.strip()
        if not value:
            pending = (indent, parent, key)
            continue
        if value.startswith("[") and value.endswith("]"):
            parent[key] = [_scalar(v.strip()) for v in value[1:-1].split(",") if v.strip()]
        else:
            parent[key] = _scalar(value)
    if pending is not None:
        pending[1][pending[2]] = None
    return root


def _scalar(value: str) -> Any:
    lowered = value.lower()
    if lowered in ("true", "false"):
        return lowered == "true"
    if lowered in ("null", "none", "~"):
        return None
    for kind in (int, float):
        try:
            return kind(value)
        except ValueError:
            pass
    return value.strip("'\"")


def load_tokens(folder: Path) -> list[str]:
    return list(json.loads((folder / "tokens.json").read_text(encoding="utf-8")))
