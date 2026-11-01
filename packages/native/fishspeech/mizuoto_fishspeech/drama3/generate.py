"""Drama 3's generation (upstream ``fish_omni/generate.py``'s ``generate_drama3`` and ``inference.py``'s
``generate_long_drama3``).

The prompt runs through the backbone in blocks of 128 rows; each block's hidden states also teacher-force the tower's
caches (row p labelled with frame p + 1). Then each step draws a frame from the last hidden state (main token, then the
tower's nine codes) and runs the backbone on it, until the end of the turn or ``max_new_tokens``. The one wait for the
device per step reads whether the frame ended the turn (and whether the logits were finite).

Upstream sizes its caches for at most ``GRAPH_CAPACITY`` rows (its CUDA graphs' size) and clamps every chunk's frames to
what is left after the prompt; that limit stays, so a request gives the same frames on any device.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import torch

from mizuoto_fishspeech.drama3.prompt import CHUNK_MAX_NEW_TOKENS, DEFAULT_MAX_NEW_TOKENS, MAX_PROMPT_ROWS, History, PromptError, build_prompt, chunk_prompt, split_text
from mizuoto_fishspeech.drama3.tower import REFERENCE_EPSILON_CUTOFF, Fallback

if TYPE_CHECKING:
    from mizuoto_fishspeech.drama3.backbone import Backbone
    from mizuoto_fishspeech.drama3.config import Drama3Config
    from mizuoto_fishspeech.drama3.tower import CodebookTower

log = logging.getLogger(__name__)
PREFILL_BLOCK = 128
GRAPH_CAPACITY = MAX_PROMPT_ROWS + DEFAULT_MAX_NEW_TOKENS
"""Upstream's cache size (prompt rows and frames): no request goes beyond it."""
DEFAULT_TEMPERATURE = 0.7
DEFAULT_TOP_P = 0.7
DEFAULT_TOP_K = 30
DEFAULT_MAX_NEW = 1024
"""Upstream's defaults (those of Fish Audio's SDK): per chunk."""


@dataclass
class Sampling:
    temperature: float = DEFAULT_TEMPERATURE
    top_p: float = DEFAULT_TOP_P
    top_k: int = DEFAULT_TOP_K
    max_new_tokens: int = DEFAULT_MAX_NEW
    """Frames per chunk; 0 or less: 600 per chunk, 2048 with ``single_chunk``."""
    fallback: Fallback | None = None
    """The reference sampler with this fallback for a repeated main token; None: the default sampler."""
    epsilon_cutoff: float | None = None
    """The reference sampler's cutoff for the main token (None: 0.001); the default sampler takes none."""

    def resolved(self) -> Sampling:
        """Checked as upstream checks a request, the reference sampler's cutoff filled in."""
        if not 0 < self.top_p <= 1:
            raise PromptError("top_p must be in (0, 1]")
        if not 0 < self.temperature < 2:
            raise PromptError("temperature must be in (0, 2)")
        if self.epsilon_cutoff is not None and not 0.0 <= self.epsilon_cutoff < 1.0:
            raise PromptError(f"epsilon_cutoff must be in [0, 1); got {self.epsilon_cutoff}")
        if self.fallback is None:
            # the default sampler keeps the first-ranked token whatever top_k is, so top_k < 1 would silently be greedy
            if self.top_k < 1:
                raise PromptError(f"top_k must be at least 1; got {self.top_k}")
            if self.epsilon_cutoff:
                raise PromptError("epsilon_cutoff needs the reference sampler")
            cutoff = 0.0
        else:
            cutoff = REFERENCE_EPSILON_CUTOFF if self.epsilon_cutoff is None else float(self.epsilon_cutoff)
        return Sampling(self.temperature, self.top_p, self.top_k, self.max_new_tokens, self.fallback, cutoff)


class NonFiniteLogits(FloatingPointError):
    """Upstream refuses to sample from logits that are not finite (invalid audio would follow)."""


def capacity_for(config: Drama3Config) -> int:
    return min(GRAPH_CAPACITY, config.text.max_position_embeddings)


def generate(
    backbone: Backbone,
    tower: CodebookTower,
    config: Drama3Config,
    prompt: torch.Tensor,
    max_new_tokens: int,
    sampling: Sampling,
    generator: torch.Generator,
    check: Callable[[], None] = lambda: None,
    progress: Callable[[int, int], None] | None = None,
) -> tuple[torch.Tensor, str]:
    """The frames after ``prompt`` (CPU [frames, 1 + codebooks], without the end) and why it stopped (``eos`` or
    ``length``)."""
    device = backbone.model.norm.weight.device
    prompt_rows = len(prompt)
    capacity = prompt_rows + max_new_tokens
    if max_new_tokens < 1 or capacity > config.text.max_position_embeddings:
        raise PromptError(f"max_new_tokens={max_new_tokens} does not fit after {prompt_rows} prompt rows")
    states = backbone.new_states(capacity, device)
    tower_state = tower.new_state(capacity, device)
    frames = torch.zeros((capacity, prompt.shape[1]), dtype=torch.long, device=device)
    frames[:prompt_rows] = prompt.to(device)
    positions = torch.arange(capacity, device=device)

    hidden = None
    for start in range(0, prompt_rows, PREFILL_BLOCK):
        check()
        end = min(start + PREFILL_BLOCK, prompt_rows)
        chunk = frames[start:end]
        embeddings = tower.embed_backbone(chunk, backbone.embed(chunk[:, 0]))
        hidden = backbone(embeddings[None], positions[start:end], states)[0]
        paired_end = min(end, prompt_rows - 1)
        if paired_end > start:
            tower.prefill(hidden[: paired_end - start], positions[start:paired_end], frames[start:paired_end], frames[start + 1 : paired_end + 1], tower_state)
    assert hidden is not None
    hidden = hidden[-1:]

    produced, finish = 0, "length"
    for step in range(max_new_tokens):
        check()
        if progress is not None:
            progress(step, max_new_tokens)
        position = prompt_rows - 1 + step
        logits = backbone.compute_logits(hidden)
        sampled, finite = tower.decode_step(
            logits, hidden, positions[position : position + 1], frames[position : position + 1], frames[: position + 1, 0], tower_state, sampling, generator
        )
        main, ok = (int(x) for x in torch.stack((sampled[0, 0], finite.long())).tolist())  # the step's one wait
        if not ok:
            raise NonFiniteLogits("non-finite logits; refusing to sample invalid audio")
        if main == config.eos_token_id:
            finish = "eos"
            break
        frames[position + 1] = sampled[0]
        produced += 1
        if step + 1 < max_new_tokens:
            embeddings = tower.embed_backbone(sampled, backbone.embed(sampled[:, 0]))
            hidden = backbone(embeddings[None], positions[position + 1 : position + 2], states)[0]
    return frames[prompt_rows : prompt_rows + produced].cpu(), finish


def chunks_of(text: str, single_chunk: bool) -> list[str]:
    return [text] if single_chunk else split_text(text) or [text]


def generate_long(
    backbone: Backbone,
    tower: CodebookTower,
    config: Drama3Config,
    tokenizer: Any,
    text: str,
    prompt_texts: Sequence[str],
    prompt_codes: Sequence[torch.Tensor],
    sampling: Sampling,
    single_chunk: bool,
    generator: torch.Generator,
    check: Callable[[], None] = lambda: None,
    progress: Callable[[int, int, int, int], None] | None = None,
    around: Callable[[Callable[[], Any]], Any] | None = None,
) -> Iterator[tuple[str, torch.Tensor]]:
    """Each chunk's text and codes [codebooks, frames], chunk after chunk (a chunk that ended before any frame yields
    nothing). ``around(run)``: runs each chunk's generation (the engine places the models there)."""
    sampling = sampling.resolved()
    prompt_texts = list(prompt_texts)
    prompt_codes = [codes.cpu() for codes in prompt_codes]
    chunks = chunks_of(text, single_chunk)
    capacity = capacity_for(config)
    # bad text in any chunk is refused before generating: failing at a later one would waste the earlier ones
    for chunk in chunks:
        if len(build_prompt(chunk, tokenizer, config, prompt_texts, prompt_codes)) >= capacity:
            raise PromptError("the prompt leaves no room to generate")
    requested = sampling.max_new_tokens
    if requested <= 0:
        requested = DEFAULT_MAX_NEW_TOKENS if single_chunk else CHUNK_MAX_NEW_TOKENS
    history = History()
    for index, chunk in enumerate(chunks):
        prompt = chunk_prompt(chunk, tokenizer, config, prompt_texts, prompt_codes, history)
        limit = capacity - len(prompt)
        if limit < 1:
            raise PromptError(f"prompt has {len(prompt)} rows; no room left to generate")
        budget = min(requested, limit)
        report = (lambda done, total, index=index: progress(index, len(chunks), done, total)) if progress is not None else None

        def run(prompt: torch.Tensor = prompt, budget: int = budget, report: Any = report) -> tuple[torch.Tensor, str]:
            return generate(backbone, tower, config, prompt, budget, sampling, generator, check, report)

        frames, finish = around(run) if around is not None else run()
        if finish != "eos":
            log.warning("Drama 3 reached max_new_tokens (%d) before the end of chunk %d; its audio may be cut short", budget, index + 1)
        if len(frames):
            codes = frames[:, 1:].T.contiguous()
            yield chunk, codes
            history.push(chunk, codes.cpu())
        else:
            log.warning("Drama 3 ended chunk %d before any audio frame", index + 1)
