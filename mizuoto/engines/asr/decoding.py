"""Whisper's decoding as transformers 4.57.3 runs it for the ASR pipeline (Apache-2.0).

- ``GenerationSettings``: the model's ``generation_config.json`` (``config.json`` fills what it lacks).
- ``Processors``: ``SuppressTokensAtBeginLogitsProcessor``, ``SuppressTokensLogitsProcessor`` and
  ``WhisperTimeStampLogitsProcessor``, in the order ``WhisperGenerationMixin._retrieve_logit_processors`` builds
  them, on log-probabilities for beam search and on logits for greedy decoding, as ``GenerationMixin`` applies them.
- ``beam_search``: ``GenerationMixin._beam_search`` (the vectorised one of 4.50+): ``2 × num_beams`` candidates per
  step from the accumulated log-probabilities, finished hypotheses scored with the length penalty, the early-stop
  heuristic of ``early_stopping=False``, the cache reordered after every step. ``greedy``: ``_sample`` without sampling.
- ``generate``: ``WhisperGenerationMixin.generate`` for 30 s windows (short form): language and task in the prompt,
  timestamps on, and its seek loop — a window whose decoding ends on two timestamps is decoded again from the last
  one (the rest padded with zeros), the tokens after it dropped; the windows of a batch that finish leave it.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from pathlib import Path

import torch
from torch.nn import functional as F

from mizuoto.engines.asr.model import DecoderCache, WhisperModel

StepFn = Callable[[], None]
"""Called before every decoding step (cancellation, progress)."""


@dataclass
class GenerationSettings:
    decoder_start_token_id: int
    eos_token_id: int
    pad_token_id: int
    no_timestamps_token_id: int
    max_length: int
    """Prompt included."""
    max_new_tokens: int | None = None
    """Set when the model's generation config leaves ``max_length`` at transformers' default: the pipeline's 256."""
    suppress_tokens: list[int] = field(default_factory=list)
    begin_suppress_tokens: list[int] = field(default_factory=list)
    max_initial_timestamp_index: int | None = None
    lang_to_id: dict[str, int] = field(default_factory=dict)
    task_to_id: dict[str, int] = field(default_factory=dict)
    is_multilingual: bool = True
    length_penalty: float = 1.0
    early_stopping: bool | str = False

    @property
    def timestamp_begin(self) -> int:
        return self.no_timestamps_token_id + 1

    @classmethod
    def load(cls, folder: Path) -> GenerationSettings:
        config = json.loads((folder / "config.json").read_text(encoding="utf-8"))
        path = folder / "generation_config.json"
        gen = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
        merged = {**{k: v for k, v in config.items() if v is not None}, **{k: v for k, v in gen.items() if v is not None}}
        if "no_timestamps_token_id" not in merged:
            raise ValueError("generation_config.json has no no_timestamps_token_id (timestamps need it)")
        eos = int(merged["eos_token_id"] if "eos_token_id" in merged else merged["bos_token_id"])
        max_target = int(config.get("max_target_positions", 448))
        max_length, max_new_tokens = int(gen.get("max_length") or 20), None
        if max_length == 20:  # transformers' default: the pipeline's max_new_tokens=256 stays in force
            max_length, max_new_tokens = max_target, 256
        return cls(
            decoder_start_token_id=int(merged["decoder_start_token_id"]),
            eos_token_id=eos,
            pad_token_id=int(merged.get("pad_token_id", eos)),
            no_timestamps_token_id=int(merged["no_timestamps_token_id"]),
            max_length=min(max_length, max_target),
            max_new_tokens=max_new_tokens,
            suppress_tokens=[int(t) for t in merged.get("suppress_tokens") or []],
            begin_suppress_tokens=[int(t) for t in merged.get("begin_suppress_tokens") or []],
            max_initial_timestamp_index=merged.get("max_initial_timestamp_index"),
            lang_to_id={str(k): int(v) for k, v in (merged.get("lang_to_id") or {}).items()},
            task_to_id={str(k): int(v) for k, v in (merged.get("task_to_id") or {}).items()},
            is_multilingual=bool(merged.get("is_multilingual", bool(merged.get("lang_to_id")))),
            length_penalty=float(merged.get("length_penalty", 1.0)),
            early_stopping=merged.get("early_stopping", False),
        )


class Processors:
    """The logits processors of one generation call, ``begin_index`` being the prompt's length."""

    def __init__(self, gen: GenerationSettings, vocab_size: int, begin_index: int, device: torch.device) -> None:
        self.gen = gen
        self.begin_index = begin_index
        vocab = torch.arange(vocab_size, device=device)
        self.begin_mask = torch.isin(vocab, torch.tensor(gen.begin_suppress_tokens, dtype=torch.long, device=device)) if gen.begin_suppress_tokens else None
        self.suppress_mask = torch.isin(vocab, torch.tensor(gen.suppress_tokens, dtype=torch.long, device=device)) if gen.suppress_tokens else None

    def __call__(self, input_ids: torch.Tensor, scores: torch.Tensor) -> torch.Tensor:
        if self.begin_mask is not None and input_ids.shape[-1] == self.begin_index:
            scores = torch.where(self.begin_mask, -float("inf"), scores)
        if self.suppress_mask is not None:
            scores = torch.where(self.suppress_mask, -float("inf"), scores)
        return self._timestamps(input_ids, scores)

    def _timestamps(self, input_ids: torch.Tensor, scores: torch.Tensor) -> torch.Tensor:
        """``WhisperTimeStampLogitsProcessor``."""
        gen = self.gen
        tb = gen.timestamp_begin
        processed = scores.clone()
        processed[:, gen.no_timestamps_token_id] = -float("inf")
        for k in range(input_ids.shape[0]):
            sampled = input_ids[k, self.begin_index :]
            seq = sampled.tolist()
            last_was_timestamp = len(seq) >= 1 and seq[-1] >= tb
            penultimate_was_timestamp = len(seq) < 2 or seq[-2] >= tb
            if last_was_timestamp:
                if penultimate_was_timestamp:
                    processed[k, tb:] = -float("inf")
                else:
                    processed[k, : gen.eos_token_id] = -float("inf")
            timestamps = sampled[sampled.ge(tb)]
            if timestamps.numel() > 0:
                timestamp_last = timestamps[-1] if last_was_timestamp and not penultimate_was_timestamp else timestamps[-1] + 1
                processed[k, tb:timestamp_last] = -float("inf")
        if input_ids.shape[1] == self.begin_index:
            processed[:, :tb] = -float("inf")
            if gen.max_initial_timestamp_index is not None:
                processed[:, tb + int(gen.max_initial_timestamp_index) + 1 :] = -float("inf")
        logprobs = F.log_softmax(processed.float(), dim=-1)
        for k in range(input_ids.shape[0]):
            timestamp_logprob = logprobs[k, tb:].logsumexp(dim=-1)
            max_text_token_logprob = logprobs[k, :tb].max()
            if timestamp_logprob > max_text_token_logprob:
                processed[k, :tb] = -float("inf")
        return processed


def _stopping(sequences: torch.Tensor, max_length: int, eos: int) -> torch.Tensor:
    """``MaxLengthCriteria`` | ``EosTokenCriteria`` for each row of (rows, length)."""
    done = torch.full((sequences.shape[0],), sequences.shape[-1] >= max_length, dtype=torch.bool, device=sequences.device)
    return done | (sequences[:, -1] == eos)


def _step_logits(model: WhisperModel, sequences: torch.Tensor, encoder_hidden: torch.Tensor, cache: DecoderCache) -> torch.Tensor:
    """The next-token logits of every row (float32), feeding what the cache does not hold yet."""
    logits = model.decoder(sequences[:, cache.length :], encoder_hidden, cache)
    return logits[:, -1, :].to(copy=True, dtype=torch.float32)


def _gather_beams(tensor: torch.Tensor, beam_indices: torch.Tensor) -> torch.Tensor:
    while len(beam_indices.shape) < len(tensor.shape):
        beam_indices = beam_indices.unsqueeze(-1)
    return torch.take_along_dim(input=tensor, indices=beam_indices, dim=1)


def beam_search(
    model: WhisperModel,
    encoder_hidden: torch.Tensor,
    input_ids: torch.Tensor,
    gen: GenerationSettings,
    processors: Processors,
    num_beams: int,
    step: StepFn,
) -> torch.Tensor:
    """``GenerationMixin._beam_search``: the best sequence of each item, prompt included, (items, length).

    ``encoder_hidden`` and ``input_ids`` are already repeated per beam (``repeat_interleave``).
    """
    device = input_ids.device
    pad, eos = gen.pad_token_id, gen.eos_token_id
    length_penalty, early_stopping, max_length = gen.length_penalty, gen.early_stopping, gen.max_length
    vocab_size = model.config.vocab_size
    batch_size_unflattened, cur_len = input_ids.shape[:2]
    batch_size = batch_size_unflattened // num_beams
    decoder_prompt_len = cur_len
    beams_to_keep = 2 * num_beams
    top_num_beam_mask = torch.cat((torch.ones((num_beams), dtype=torch.bool), torch.zeros((beams_to_keep - num_beams), dtype=torch.bool)), dim=0).to(device)

    running_sequences = torch.full((batch_size, num_beams, max_length), fill_value=pad, dtype=torch.int64, device=device)
    running_sequences[:, :, :cur_len] = input_ids.reshape(batch_size, num_beams, cur_len)
    sequences = running_sequences.detach().clone()
    running_beam_scores = torch.zeros((batch_size, num_beams), dtype=torch.float, device=device)
    running_beam_scores[:, 1:] = -1e9
    beam_scores = torch.full((batch_size, num_beams), fill_value=-1e9, dtype=torch.float, device=device)
    is_sent_finished = torch.zeros((batch_size, num_beams), dtype=torch.bool, device=device)
    is_early_stop_heuristic_unsatisfied = torch.ones((batch_size, 1), dtype=torch.bool, device=device)
    next_token_hits_stopping_criteria = torch.zeros((batch_size, num_beams), dtype=torch.bool, device=device)
    running_beam_indices = torch.full((batch_size, num_beams, max_length - cur_len), fill_value=-1, dtype=torch.int32, device=device)
    beam_indices = running_beam_indices.detach().clone()
    cache = DecoderCache(model.config.decoder_layers)

    finished = False
    while not finished:
        step()
        flat_running_sequences = running_sequences[:, :, :cur_len].reshape(batch_size * num_beams, cur_len)
        logits = _step_logits(model, flat_running_sequences, encoder_hidden, cache)
        if not cache.cross_shared and num_beams > 1:
            cache.cross_shared = _beams_share(cache, batch_size, num_beams)
        log_probs = F.log_softmax(logits, dim=-1)
        log_probs = processors(flat_running_sequences, log_probs)
        log_probs = log_probs.reshape(batch_size, num_beams, -1)
        log_probs = log_probs + running_beam_scores[:, :, None]
        log_probs = torch.reshape(log_probs, (batch_size, num_beams * vocab_size))

        # top-k continuations
        topk_log_probs, topk_indices = torch.topk(log_probs, k=beams_to_keep)
        topk_current_beam_indices = topk_indices // vocab_size
        topk_running_beam_indices = _gather_beams(running_beam_indices, topk_current_beam_indices)
        topk_running_sequences = _gather_beams(running_sequences, topk_current_beam_indices)
        topk_ids = topk_indices % vocab_size
        topk_running_sequences[:, :, cur_len] = topk_ids
        batch_offset = torch.arange(batch_size, device=device).view(-1, 1) * num_beams
        batch_modified_indices = topk_current_beam_indices + batch_offset
        topk_running_beam_indices[:, :, cur_len - decoder_prompt_len] = batch_modified_indices

        flat_topk = topk_running_sequences[:, :, : cur_len + 1].reshape(batch_size * beams_to_keep, cur_len + 1)
        next_token_hits_stopping_criteria = _stopping(flat_topk, max_length, eos).reshape(batch_size, beams_to_keep)

        # the running beams of the next step
        topk_running_log_probs = topk_log_probs + next_token_hits_stopping_criteria.to(torch.float32) * -1.0e9
        next_topk_indices = torch.topk(topk_running_log_probs, k=num_beams)[1]
        running_sequences = _gather_beams(topk_running_sequences, next_topk_indices)
        running_beam_scores = _gather_beams(topk_running_log_probs, next_topk_indices)
        running_beam_indices = _gather_beams(topk_running_beam_indices, next_topk_indices)

        # finished hypotheses
        did_top_num_beams_just_finished = next_token_hits_stopping_criteria & top_num_beam_mask[None, :]
        finished_log_probs = topk_log_probs / ((cur_len + 1 - decoder_prompt_len) ** length_penalty)
        beams_in_batch_are_full = torch.all(is_sent_finished, dim=-1, keepdim=True) & (early_stopping is True)
        finished_log_probs += beams_in_batch_are_full.to(torch.float32) * -1.0e9
        finished_log_probs += (~is_early_stop_heuristic_unsatisfied).to(torch.float32) * -1.0e9
        finished_log_probs += (~did_top_num_beams_just_finished) * -1.0e9
        merged_sequences = torch.cat((sequences, topk_running_sequences), dim=1)
        merged_scores = torch.cat((beam_scores, finished_log_probs), dim=1)
        merged_beam_indices = torch.cat((beam_indices, topk_running_beam_indices), dim=1)
        merged_is_sent_finished = torch.cat((is_sent_finished, did_top_num_beams_just_finished), dim=1)
        topk_merged_indices = torch.topk(merged_scores, k=num_beams)[1]
        sequences = _gather_beams(merged_sequences, topk_merged_indices)
        beam_scores = _gather_beams(merged_scores, topk_merged_indices)
        beam_indices = _gather_beams(merged_beam_indices, topk_merged_indices)
        is_sent_finished = _gather_beams(merged_is_sent_finished, topk_merged_indices)

        beam_idx = running_beam_indices[..., cur_len - decoder_prompt_len].reshape(batch_size * num_beams)
        cache.reorder(beam_idx)
        cur_len = cur_len + 1

        # early-stop heuristic and the stopping condition
        if early_stopping == "never" and length_penalty > 0.0:
            best_hypothetical_length = max_length - decoder_prompt_len
        else:
            best_hypothetical_length = cur_len - decoder_prompt_len
        best_possible_running_score = running_beam_scores[:, :1] / (best_hypothetical_length**length_penalty)
        worst_finished_score = torch.where(is_sent_finished, torch.min(beam_scores, dim=1, keepdim=True)[0], -1.0e9)
        is_early_stop_heuristic_unsatisfied = is_early_stop_heuristic_unsatisfied & torch.any(best_possible_running_score > worst_finished_score, dim=-1, keepdim=True)
        improvement_possible = torch.any(is_early_stop_heuristic_unsatisfied)
        exists_open_beam = ~(torch.all(is_sent_finished) & (early_stopping is True))
        valid_continuations = ~torch.all(next_token_hits_stopping_criteria)
        finished = not bool(improvement_possible & exists_open_beam & valid_continuations)

    sequences = sequences[:, 0, :]
    beam_indices = beam_indices[:, 0, :]
    max_generated_length = int(((beam_indices + 1).bool()).sum(dim=1).max())
    return sequences[:, : decoder_prompt_len + max_generated_length]


def _beams_share(cache: DecoderCache, batch_size: int, num_beams: int) -> bool:
    """Whether the cross-attention keys and values are bit-identical across each item's beams."""
    for kv in cache.cross:
        if kv is None:
            return False
        for t in kv:
            rows = t.reshape(batch_size, num_beams, -1)
            if not torch.equal(rows, rows[:, :1].expand_as(rows)):
                return False
    return True


def greedy(model: WhisperModel, encoder_hidden: torch.Tensor, input_ids: torch.Tensor, gen: GenerationSettings, processors: Processors, step: StepFn) -> torch.Tensor:
    """``GenerationMixin._sample`` without sampling: the processed logits' argmax, finished rows padded."""
    pad, eos = gen.pad_token_id, gen.eos_token_id
    cache = DecoderCache(model.config.decoder_layers)
    unfinished = torch.ones(input_ids.shape[0], dtype=torch.long, device=input_ids.device)
    while True:
        step()
        logits = _step_logits(model, input_ids, encoder_hidden, cache)
        scores = processors(input_ids, logits)
        next_tokens = torch.argmax(scores, dim=-1)
        next_tokens = next_tokens * unfinished + pad * (1 - unfinished)
        input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)
        unfinished = unfinished & ~_stopping(input_ids, gen.max_length, eos)
        if unfinished.max() == 0:
            return input_ids


# -- Whisper's generate ---------------------------------------------------------------------------


def _retrieve_segment(seq: list[int], timestamp_begin: int, seek_num_frames: int, input_stride: int) -> tuple[list[int], int]:
    """``_retrieve_segment``, reduced to what the pipeline uses: the tokens kept and how far to seek."""
    stamps = [t >= timestamp_begin for t in seq]
    single_timestamp_ending = stamps[-2:] == [False, True]
    ends = [i + 1 for i in range(len(seq) - 1) if stamps[i] and stamps[i + 1]]
    if not ends:
        return seq, seek_num_frames
    if single_timestamp_ending:
        return seq, seek_num_frames
    last_slice = ends[-1] + 1
    offset = (seq[last_slice - 2] - timestamp_begin) * input_stride
    return seq[:last_slice], offset if offset > 0 else seek_num_frames


def _strip(seq: list[int], gen: GenerationSettings) -> list[int]:
    """The padding after the end of text removed, then the end of text (``generate_with_fallback``)."""
    if seq and seq[-1] == gen.pad_token_id:
        paddings = seq.count(gen.pad_token_id)
        if gen.pad_token_id == gen.eos_token_id:
            paddings -= 1
        if paddings:
            seq = seq[:-paddings]
    if seq and seq[-1] == gen.eos_token_id:
        seq = seq[:-1]
    return seq


def generate(
    model: WhisperModel,
    features: torch.Tensor,
    prompt: list[int],
    gen: GenerationSettings,
    num_beams: int,
    step: StepFn,
    encoder_hidden: torch.Tensor | None = None,
) -> list[list[int]]:
    """``WhisperGenerationMixin.generate`` on a batch of 30 s windows (batch, mel bins, frames): each window's tokens,
    without the prompt and the end of text, as the pipeline gets them.

    ``encoder_hidden``: the encoder's output for ``features`` when already computed (the first pass reuses it).
    """
    c = model.config
    if gen.max_new_tokens is not None:
        gen = replace(gen, max_length=min(len(prompt) + gen.max_new_tokens, c.max_target_positions))
    device = features.device
    window = c.num_segment_frames
    batch = features.shape[0]
    total = features.shape[-1]
    if total > window:
        raise ValueError("generate takes windows of at most 30 s")
    seek = [0] * batch
    tokens: list[list[int]] = [[] for _ in range(batch)]
    rows = list(range(batch))
    first = True
    processors = Processors(gen, c.vocab_size, len(prompt), device)
    while any(seek[i] < total for i in range(batch)):
        rows = [i for i in rows if seek[i] < total]
        num_frames = {i: min(total - seek[i], window) for i in rows}
        segments = []
        for i in rows:
            part = features[i : i + 1, :, seek[i] : seek[i] + num_frames[i]]
            if part.shape[-1] < window:
                part = F.pad(part, pad=(0, window - part.shape[-1]))
            segments.append(part)
        segment = torch.cat(segments, dim=0)
        hidden = encoder_hidden if first and encoder_hidden is not None and len(rows) == batch else model.encoder(segment)
        first = False
        input_ids = torch.tensor([prompt] * len(rows), dtype=torch.long, device=device)
        if num_beams > 1:
            out = beam_search(model, hidden.repeat_interleave(num_beams, dim=0), input_ids.repeat_interleave(num_beams, dim=0), gen, processors, num_beams, step)
        else:
            out = greedy(model, hidden, input_ids, gen, processors, step)
        for j, i in enumerate(rows):
            seq = _strip(out[j, len(prompt) :].tolist(), gen)
            kept, offset = _retrieve_segment(seq, gen.timestamp_begin, num_frames[i], c.input_stride)
            tokens[i] += kept
            seek[i] += offset
    return tokens


def prompt_tokens(gen: GenerationSettings, language_id: int | None, task: str = "transcribe") -> list[int]:
    """``_retrieve_init_tokens`` with a language and a task, timestamps on: start of transcript, language, task."""
    tokens = [gen.decoder_start_token_id]
    if gen.is_multilingual:
        if language_id is not None:
            tokens.append(language_id)
        if task in gen.task_to_id:
            tokens.append(gen.task_to_id[task])
    return tokens


def detect_language(model: WhisperModel, features: torch.Tensor, gen: GenerationSettings, encoder_hidden: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    """``WhisperGenerationMixin.detect_language``: the language token of each window, and the encoder output.

    One decoder step from the start of transcript without a cache; every token but the languages masked.
    """
    if encoder_hidden is None:
        encoder_hidden = model.encoder(features[:, :, : model.config.num_segment_frames])
    batch = features.shape[0]
    ids = torch.full((batch, 1), gen.decoder_start_token_id, dtype=torch.long, device=features.device)
    mask = torch.ones((batch, 1, 1, 1), dtype=torch.bool, device=features.device)
    logits = model.decoder(ids, encoder_hidden, None, self_mask=mask)[:, -1]
    non_lang = torch.ones_like(logits[0], dtype=torch.bool)
    non_lang[list(gen.lang_to_id.values())] = False
    logits[:, non_lang] = -float("inf")
    return logits.argmax(-1), encoder_hidden
