"""``compute.compile`` for S2 Pro's decoding steps: the step's small operations fused by ``torch.compile``.

A decoding step launches thousands of small kernels a frame (norms, rotary embeddings, residual adds, the eager
attention, the draws' filtering), each costing the device a few microseconds whatever its size; CUDA graphs remove
their launches, not that. Compiled, each layer's halves around its attention, the attention over the cache, the
step's embeddings and heads, and the draws' probabilities run as a few fused kernels, inside the same CUDA graphs.

Not exact: fusing keeps intermediate values in float32 where eager rounds them to the model's dtype, so the logits
differ in their last bits and the draws soon go elsewhere. The randomness is unchanged (the same uniform draws from
the same generator, in the same order); only the probabilities they are compared with differ.

Each piece is a module-level function of the module it runs, so the 36 slow layers share one compiled graph per
half (their weights are its inputs) and the fast transformer's 4 layers another. The pieces with fixed shapes are
compiled static. The slow attention's key length grows each step: only that dimension is marked dynamic. Compiled with
every dimension dynamic, the head size was symbolic too and inductor computed the scale per element in float64, which
made the attention twice as slow as eager on an RTX 3060; with the length alone dynamic it is twice as fast. Any
compiler error makes every piece run eagerly from then on (one warning); running out of memory still raises.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import torch

    from mizuoto_fishspeech.model import Attention, DualARTransformer, FixedCache, GrowingCache, TransformerBlock

log = logging.getLogger(__name__)


def slow_in(layer: TransformerBlock, h: torch.Tensor, rope: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return layer.attention_in(h, rope)


def slow_out(layer: TransformerBlock, h: torch.Tensor, attended: torch.Tensor) -> torch.Tensor:
    return layer.attention_out(h, attended)


def slow_attend(attention: Attention, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return attention.attend_cached(q, k, v, mask, True)


def fast_in(layer: TransformerBlock, h: torch.Tensor, rope: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return layer.attention_in(h, rope)


def fast_out(layer: TransformerBlock, h: torch.Tensor, attended: torch.Tensor) -> torch.Tensor:
    return layer.attention_out(h, attended)


def fast_attend(attention: Attention, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return attention.attend_cached(q, k, v, mask, False)


def embed(model: DualARTransformer, cur: torch.Tensor) -> torch.Tensor:
    return model.input_embeddings(cur)


def output(model: DualARTransformer, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    result = model.output_step(x)
    return result.logits, result.hidden_states


def fast_head(model: DualARTransformer, x: torch.Tensor) -> torch.Tensor:
    return model.fast_output(model.fast_norm(x))


def probs(logits: torch.Tensor, temperature: torch.Tensor, top_p: torch.Tensor) -> torch.Tensor:
    from mizuoto_fishspeech.generate import TOP_K, logits_to_probs

    return logits_to_probs(logits[0, -1], temperature, top_p, TOP_K)


def pick(probs: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    import torch

    return torch.argmax(probs / -torch.log(q), dim=-1, keepdim=True).to(dtype=torch.int)


PIECES: dict[str, tuple[Any, tuple[tuple[int, int], ...]]] = {
    f.__name__: (f, dynamic)
    for f, dynamic in (
        (slow_in, ()),
        (slow_out, ()),
        (slow_attend, ((2, 2), (3, 2), (4, 3))),  # the keys', the values' and the mask's length
        (fast_in, ()),
        (fast_out, ()),
        (fast_attend, ()),
        (embed, ()),
        (output, ()),
        (fast_head, ()),
        (probs, ()),
        (pick, ()),
    )
}
"""Each piece and the (argument, dimension) pairs it is compiled dynamic in; none: static."""


class Fused:
    """The compiled pieces of one loaded model's decoding steps."""

    def __init__(self, name: str) -> None:
        import torch

        self.name = name
        self.failure: str | None = None
        """Why the pieces run eagerly now (the first compiler error), or None."""
        self.calls = 0
        """Calls that ran compiled."""
        self._compiled = {key: torch.compile(fn, dynamic=None if dynamic else False) for key, (fn, dynamic) in PIECES.items()}

    def run(self, piece: str, *args: Any) -> Any:
        if self.failure is None:
            try:
                from mizuoto.runtime.context import accel_policy

                if PIECES[piece][1]:
                    from torch._dynamo import mark_dynamic

                    for arg, dim in PIECES[piece][1]:
                        mark_dynamic(args[arg], dim)

                active = accel_policy.get()
                if active is None:
                    out = self._compiled[piece](*args)
                else:  # the graph cannot read the context's attention policy: it traces with this one
                    from mizuoto.runtime.accel.policy import tracing

                    with tracing(active.policy.trace_run):
                        out = self._compiled[piece](*args)
            except Exception as error:  # any compiler failure falls back to eager; running out of memory does not
                from mizuoto.runtime.device import is_oom

                if is_oom(error):
                    raise
                lines = str(error).strip().splitlines()
                self.failure = f"{type(error).__name__}: {lines[0] if lines else ''}"
                log.warning("torch.compile of %s failed, decoding eagerly: %s", self.name, self.failure)
                log.debug("torch.compile of %s (piece %s) failed", self.name, piece, exc_info=True)
            else:
                self.calls += 1
                return out
        return PIECES[piece][0](*args)

    # -- a decoding step's parts, as the model's own code runs them -------------------------------

    def slow_attention(self, attention: Attention, cache: GrowingCache, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """``Attention.attend``: the cache's update (eager: it grows), then the compiled attention."""
        k, v = cache.update(k, v, mask.shape[-1])
        return self.run("slow_attend", attention, q, k, v, mask)

    def fast_forward(self, model: DualARTransformer, x: torch.Tensor, pos: int, caches: list[FixedCache]) -> torch.Tensor:
        """``DualARTransformer.forward_generate_fast`` from compiled pieces: each layer's halves and attention, then the head."""
        import torch

        n = model.config.num_codebooks
        x = x.view(x.shape[0], 1, -1)
        mask = torch.ones(n, n, dtype=torch.bool, device=x.device).tril()[None, None, pos : pos + 1, :n]
        freqs_cis = model.get_buffer("fast_freqs_cis")[pos : pos + 1]
        for layer, cache in zip(model.fast_layers, caches, strict=True):
            cache.pos = pos
            q, k, v = self.run("fast_in", layer, x, freqs_cis)
            k, v = cache.update(k, v, mask.shape[-1])
            x = self.run("fast_out", layer, x, self.run("fast_attend", layer.attention, q, k, v, mask))
        return self.run("fast_head", model, x)

    def sample(self, logits: torch.Tensor, temperature: torch.Tensor, top_p: torch.Tensor, generator: torch.Generator) -> torch.Tensor:
        """``generate.sample``: the probabilities compiled, the uniform draw as ever, the pick compiled."""
        import torch

        p = self.run("probs", logits, temperature, top_p)
        q = torch.empty_like(p).uniform_(generator=generator)
        return self.run("pick", p, q)
