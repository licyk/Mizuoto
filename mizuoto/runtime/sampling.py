"""Token sampling, reproducing each upstream's order and arithmetic.

``hf_sample`` is transformers' ``_sample`` step for one token:
- logits in float32;
- the processors in ``_get_logits_processor``'s order: repetition penalty, min-new-tokens, suppressed
  tokens, temperature, top-k, top-p;
- softmax, then ``torch.multinomial`` with the request's generator, or argmax when not sampling.

Each processor is the same tensor arithmetic as transformers', so a seeded draw matches. Nothing here waits for the
device: the masks are built once per device, and one-sample draws are ``torch.multinomial``'s own computation without
its validity checks (``multinomial1``), so a decoding loop on a GPU runs ahead of it and can be captured in a CUDA graph.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import torch


@dataclass
class HfSampling:
    do_sample: bool = True
    temperature: float = 1.0
    top_k: int = 0
    top_p: float = 1.0
    repetition_penalty: float = 1.0
    min_new_tokens: int = 0
    eos_token_id: int | None = None
    suppress_tokens: list[int] = field(default_factory=list)
    _masks: dict[Any, torch.Tensor] = field(default_factory=dict, repr=False, compare=False)

    def mask(self, which: str, vocab: int, device: torch.device) -> torch.Tensor:
        """The (vocab,) mask of the suppressed ids (``suppress``) or of EOS (``eos``), built once per device."""
        key = (which, vocab, str(device))
        mask = self._masks.get(key)
        if mask is None:
            ids = self.suppress_tokens if which == "suppress" else [self.eos_token_id]
            mask = self._masks[key] = torch.isin(torch.arange(vocab, device=device), torch.tensor(ids, device=device))
        return mask

    def key(self) -> tuple[Any, ...]:
        """What the processors do, for caching a captured sampler."""
        return (self.do_sample, self.temperature, self.top_k, self.top_p, self.repetition_penalty, self.min_new_tokens, self.eos_token_id, tuple(self.suppress_tokens))


def repetition_penalty(scores: torch.Tensor, generated: torch.Tensor, penalty: float) -> torch.Tensor:
    """``RepetitionPenaltyLogitsProcessor`` on (batch, vocab) scores and (batch, n) generated ids."""
    if generated.shape[-1] == 0:
        return scores
    score = torch.gather(scores, 1, generated)
    score = torch.where(score < 0, score * penalty, score / penalty)
    return scores.scatter(1, generated, score)


def repetition_penalty_seen(scores: torch.Tensor, seen: torch.Tensor, penalty: float) -> torch.Tensor:
    """``repetition_penalty`` given the (batch, vocab) mask of the ids generated so far: the same arithmetic on the same
    elements (a repeated id is gathered and scattered with the same value), with shapes that do not grow."""
    return torch.where(seen, torch.where(scores < 0, scores * penalty, scores / penalty), scores)


def suppress(scores: torch.Tensor, token_ids: torch.Tensor) -> torch.Tensor:
    """``SuppressTokensLogitsProcessor`` (and the min-length EOS mask)."""
    vocab = torch.arange(scores.shape[-1], device=scores.device)
    return torch.where(torch.isin(vocab, token_ids), -math.inf, scores)


def top_k(scores: torch.Tensor, k: int, min_tokens_to_keep: int = 1) -> torch.Tensor:
    k = min(max(k, min_tokens_to_keep), scores.size(-1))
    indices_to_remove = scores < torch.topk(scores, k)[0][..., -1, None]
    return scores.masked_fill(indices_to_remove, -math.inf)


def top_p(scores: torch.Tensor, p: float, min_tokens_to_keep: int = 1) -> torch.Tensor:
    sorted_logits, sorted_indices = torch.sort(scores, descending=False)
    cumulative_probs = sorted_logits.softmax(dim=-1).cumsum(dim=-1)
    sorted_indices_to_remove = cumulative_probs <= (1 - p)
    sorted_indices_to_remove[..., -min_tokens_to_keep:] = 0
    indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
    return scores.masked_fill(indices_to_remove, -math.inf)


def hf_process(scores: torch.Tensor, generated: torch.Tensor | None, s: HfSampling, seen: torch.Tensor | None = None, length: int | None = None) -> torch.Tensor:
    """transformers' logits processors for one step, in its order (``generated``: ids so far, (batch, n)).

    Given ``seen`` (the mask of ``generated``'s ids, see ``repetition_penalty_seen``) and ``length`` (its length), the
    ids themselves are not needed."""
    n = generated.shape[-1] if length is None and generated is not None else int(length or 0)
    if s.repetition_penalty != 1.0:
        if seen is not None:
            scores = repetition_penalty_seen(scores, seen, s.repetition_penalty)
        elif generated is not None:
            scores = repetition_penalty(scores, generated, s.repetition_penalty)
    if s.min_new_tokens > 0 and s.eos_token_id is not None and n < s.min_new_tokens:
        scores = torch.where(s.mask("eos", scores.shape[-1], scores.device), -math.inf, scores)
    if s.suppress_tokens:
        scores = torch.where(s.mask("suppress", scores.shape[-1], scores.device), -math.inf, scores)
    if s.do_sample:
        if s.temperature != 1.0:
            scores = scores / s.temperature
        if s.top_k:
            scores = top_k(scores, s.top_k)
        if s.top_p < 1.0:
            scores = top_p(scores, s.top_p)
    return scores


def multinomial1(probs: torch.Tensor, generator: torch.Generator | None) -> torch.Tensor:
    """``torch.multinomial(probs, 1, generator=generator).squeeze(1)`` for (batch, vocab) probabilities.

    For one sample torch draws ``q ~ Exp(1)`` over ``probs``' shape and takes ``argmax(probs / q)``; this does the same
    on the CPU and CUDA (the same ids and generator state), without the two checks torch makes first, each of which
    waits for the device. Elsewhere it is ``torch.multinomial``. Probabilities with NaN give an arbitrary id here instead
    of an error: callers that can meet them check for them where they wait anyway."""
    if probs.device.type in ("cpu", "cuda"):
        return probs.div(torch.empty_like(probs).exponential_(1, generator=generator)).argmax(-1)
    return torch.multinomial(probs, num_samples=1, generator=generator).squeeze(1)


def multinomial_draws(probs: torch.Tensor, num_samples: int, generator: torch.Generator | None) -> torch.Tensor:
    """``torch.multinomial(probs, num_samples, generator=generator)`` without replacement, (batch, num_samples): torch's
    own computation (``topk(probs / q, num_samples)`` with ``q ~ Exp(1)``) without its checks, on the CPU and CUDA (the
    same ids and generator state); elsewhere ``torch.multinomial``."""
    if probs.device.type in ("cpu", "cuda"):
        return probs.div(torch.empty_like(probs).exponential_(1, generator=generator)).topk(num_samples).indices
    return torch.multinomial(probs, num_samples=num_samples, generator=generator)


def hf_sample(
    logits: torch.Tensor,
    generated: torch.Tensor | None,
    s: HfSampling,
    generator: torch.Generator | None,
    seen: torch.Tensor | None = None,
    length: int | None = None,
    invalid: torch.Tensor | None = None,
) -> torch.Tensor:
    """The next token for each row: (batch,) ids from the last position's (batch, vocab) logits (``hf_process`` for
    ``seen`` and ``length``). When sampling, ``invalid`` (a one-element tensor) is set to whether the probabilities hold
    NaN, where ``torch.multinomial`` raises: the caller checks it when it next waits for the device."""
    scores = hf_process(logits.to(dtype=torch.float32), generated, s, seen, length)
    if s.do_sample:
        probs = torch.softmax(scores, dim=-1)
        if invalid is not None:
            invalid.copy_(probs.isnan().any())
        return multinomial1(probs, generator)
    return torch.argmax(scores, dim=-1)


INVALID_PROBABILITIES = "probability tensor contains either `inf`, `nan` or element < 0"
"""``torch.multinomial``'s error, raised by callers whose ``invalid`` flag is set."""
