"""Drama 3's decoding step as CUDA graphs (``compute.cuda_graphs``; ``mizuoto.runtime.graphs``).

A step is the tower (the frame's draws: the main token from the backbone's logits, then a code per residual layer) and
then the backbone on that frame. Both are thousands of small kernels, so eagerly a step is as slow as the CPU launches
them. Here each is a chain of graphs with the attention calls between them run eagerly: the backbone's softmax layers
and every tower layer attend over keys that grow by one each step, so those calls and their caches stay the eager
code's, and everything else (norms, projections, rotary angles, Gated DeltaNet's recurrent step, the router and the
experts, the draws) is replayed. The graphs replay the kernels the eager step launches, so the frames are the same bit
for bit; the draws come from the graphs' generator, loaded from the request's (``Graphs.drawing``).

Graphs read and write fixed tensors (``Fixed``): the step's inputs (the backbone's last hidden state, the position, the
previous frame, the window of recent main tokens), its outputs (the frame, whether it ended the turn and was finite),
and the Gated DeltaNet layers' states, which the graphs update in place (a request's caches of the attention layers
are its own: only the eager calls read them). The draws' settings (temperature, top-p, top-k, the sampler) are part of
the captured kernels, as eagerly: the tower's graphs are captured again when a request changes them.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import torch

from mizuoto_fishspeech.drama3.tower import RAS_WINDOW

if TYPE_CHECKING:
    from mizuoto.runtime.graphs import Graphs
    from mizuoto_fishspeech.drama3.backbone import Backbone
    from mizuoto_fishspeech.drama3.tower import CodebookTower


class Stretches:
    """A step as graphs with eager calls between them: graph 0, call 0, graph 1, …, call n − 1, graph n. Each call
    writes into a fixed tensor the next graph reads. A stretch's warm-up (``Graphs.capture``) runs it once on whatever
    the stretches before it left in their outputs (they are captured, not run), so it puts back the fixed tensors it
    changes (its ``state``).

    A graph reads the tensors its stretch read while it was captured and writes the ones it made then: a stretch must
    never rebind a name of ``namespace`` that an earlier stretch bound and it reads (its capture would read what its own
    warm-up left there, not the earlier graph's output). Names are bound once per step: per layer, never updated."""

    def __init__(self, graphs: Graphs, stretches: Sequence[tuple[Callable[[], None], Sequence[torch.Tensor]]], calls: Sequence[Callable[..., None]], namespace: dict) -> None:
        if len(stretches) != len(calls) + 1:
            raise ValueError("a graph before and after every eager call")
        self.calls = list(calls)
        self.namespace = namespace
        """The tensors one stretch hands the next (outputs of a graph, inputs of the eager call and the next graph)."""
        pool = graphs.pool()
        self.parts = [graphs.capture(fn, state, pool) for fn, state in stretches]

    def __call__(self, *args: Any) -> None:
        for part, call in zip(self.parts, self.calls, strict=False):
            part.replay()
            call(*args)
        self.parts[-1].replay()


@dataclass
class Fixed:
    """The tensors the graphs of one loaded model read and write, on its device."""

    hidden: torch.Tensor
    """The backbone's last hidden state [1, H]: the tower's input, the backbone step's output."""
    pos: torch.Tensor
    """[1]: the position of the step's row (the tower's), where the backbone step put the frame before it."""
    row: torch.Tensor
    """[1, 1 + codebooks]: the frame at ``pos`` (the backbone step's input; its unseen codes enter the tower)."""
    window: torch.Tensor
    """[RAS_WINDOW]: the main tokens of the rows up to ``pos``, -1 before the first."""
    frame: torch.Tensor
    """[1, 1 + codebooks]: the frame drawn."""
    flags: torch.Tensor
    """[2]: its main token and whether every logits drawn from were finite (the step's one read)."""
    candidates: torch.Tensor
    """The ids the main token is drawn from (``CodebookTower.main_candidates``)."""
    delta: dict[int, Any] = field(default_factory=dict)
    """The Gated DeltaNet layers' states, by layer: a request's own, zeroed as new ones are."""
    backbone_attended: list[torch.Tensor] = field(default_factory=list)
    tower_attended: list[torch.Tensor] = field(default_factory=list)

    @classmethod
    def make(cls, backbone: Backbone, tower: CodebookTower, device: torch.device) -> Fixed:
        dtype = backbone.dtype
        width = 1 + tower.config.num_codebooks
        layers = list(backbone.model.layers)
        c, w = backbone.config, tower.config

        def zeros(*shape: int, dtype: torch.dtype = torch.long) -> torch.Tensor:
            return torch.zeros(shape, dtype=dtype, device=device)

        return cls(
            hidden=zeros(1, c.hidden_size, dtype=dtype),
            pos=zeros(1),
            row=zeros(1, width),
            window=torch.full((RAS_WINDOW,), -1, dtype=torch.long, device=device),
            frame=zeros(1, width),
            flags=zeros(2),
            candidates=tower.main_candidates(device),
            delta={i: layer.mixer.new_state(0, dtype, device) for i, layer in enumerate(layers) if layer.layer_type != "full_attention"},
            backbone_attended=[zeros(1, 1, c.num_attention_heads * c.head_dim, dtype=dtype) for layer in layers if layer.layer_type == "full_attention"],
            tower_attended=[zeros(1, w.n_head * w.head_dim, dtype=tower.codebook_embeddings.weight.dtype) for _ in tower.layers],
        )

    def start(self, hidden: torch.Tensor, frames: torch.Tensor, prompt_rows: int) -> None:
        """The first step's inputs: the prompt's last hidden state, row and main tokens."""
        self.hidden.copy_(hidden)
        self.pos.fill_(prompt_rows - 1)
        self.row.copy_(frames[prompt_rows - 1 : prompt_rows])
        window = frames[max(0, prompt_rows - RAS_WINDOW) : prompt_rows, 0]
        self.window.fill_(-1)
        self.window[RAS_WINDOW - len(window) :].copy_(window)


def sampling_key(sampling: Any) -> tuple:
    """What of a request's sampling the tower's graphs hold (all but the frame budget)."""
    return (sampling.temperature, sampling.top_p, sampling.top_k, sampling.fallback, sampling.epsilon_cutoff)


def backbone_steps(graphs: Graphs, backbone: Backbone, tower: CodebookTower, fixed: Fixed) -> Stretches:
    """The backbone on ``row`` at ``pos`` into ``hidden``: called with the request's layer states and the attended length
    (``pos`` + 1). Graphs from one softmax attention layer to the next; the Gated DeltaNet layers between them run in
    the graphs, on ``fixed.delta``."""
    layers = list(backbone.model.layers)
    sites = [i for i, layer in enumerate(layers) if layer.layer_type == "full_attention"]
    ns: dict[Any, Any] = {}

    def through(h: torch.Tensor, start: int, end: int) -> torch.Tensor:
        for i in range(start, end):
            h = layers[i](h, ns["context"], fixed.delta[i])
        return h

    def enter(j: int, h: torch.Tensor) -> None:
        layer, context = layers[sites[j]], ns["context"]
        ns["residual", j] = h
        *ns["qkv", j], ns["gate", j] = layer.self_attn.attention_in(layer.mixer_input(h), context.cos, context.sin)

    def leave(j: int) -> torch.Tensor:
        layer = layers[sites[j]]
        return layer.finish(ns["residual", j], layer.self_attn.attention_out(fixed.backbone_attended[j], ns["gate", j]))

    def first() -> None:
        embeddings = tower.embed_backbone(fixed.row, backbone.embed(fixed.row[:, 0]))
        ns["context"] = backbone.context(fixed.pos, 0, True)
        enter(0, through(embeddings[None].to(backbone.residual_dtype), 0, sites[0]))

    def middle(j: int) -> Callable[[], None]:
        return lambda: enter(j, through(leave(j - 1), sites[j - 1] + 1, sites[j]))

    def last() -> None:
        fixed.hidden.copy_(backbone.output(through(leave(len(sites) - 1), sites[-1] + 1, len(layers)))[0])

    def call(j: int) -> Callable[..., None]:
        attention = layers[sites[j]].self_attn

        def run(states: list[Any], kv_len: int) -> None:
            fixed.backbone_attended[j].copy_(attention.attend(*ns["qkv", j], fixed.pos, kv_len, states[sites[j]]))

        return run

    def delta(start: int, end: int) -> list[torch.Tensor]:
        return [t for i in range(start, end) for t in (fixed.delta[i].conv, fixed.delta[i].recurrent)]

    stretches: list[tuple[Callable[[], None], Sequence[torch.Tensor]]] = [
        (first, delta(0, sites[0])),
        *((middle(j), delta(sites[j - 1] + 1, sites[j])) for j in range(1, len(sites))),
        (last, [*delta(sites[-1] + 1, len(layers)), fixed.hidden]),
    ]
    return Stretches(graphs, stretches, [call(j) for j in range(len(sites))], ns)


def tower_steps(graphs: Graphs, backbone: Backbone, tower: CodebookTower, fixed: Fixed, sampling: Any) -> Stretches:
    """The frame after ``hidden`` (``CodebookTower.decode_step`` and the output head before it) into ``frame`` and
    ``flags``, then the next step's ``row``, ``pos`` and ``window``: called with the request's tower state and the
    attended length (``pos`` + 1). Graphs from one tower layer's attention to the next."""
    layers = tower.layers
    shared = len(tower.shared_layers)
    ns: dict[Any, Any] = {}

    def enter(i: int, h: torch.Tensor) -> None:
        layer = layers[i]
        if i < shared:
            code, extra = ns["code", 0], ns["unseen"] if i == 0 else None
        else:
            code, extra = ns["code", i - shared], None
        mixed = layer.mix(h, code, ns["semantic"], extra)
        ns["mixed", i] = mixed
        ns["residual", i], (*ns["qkv", i], ns["gate", i]) = layer.transformer_layer.attention_in(mixed, fixed.pos)

    def leave(i: int) -> torch.Tensor:
        layer = layers[i]
        h = layer.finish(ns["mixed", i], layer.transformer_layer.attention_out(ns["residual", i], fixed.tower_attended[i], ns["gate", i]))
        if i >= shared:
            r = i - shared
            ns["code", r + 1], ns["finite", r + 1] = tower.draw_code(tower.codebook_heads[r](h), ns["finite", r], sampling, graphs.generator)
        return h

    def first() -> None:
        logits = backbone.compute_logits(fixed.hidden)
        ns["main"], ns["finite", 0] = tower.draw_main(logits, fixed.window, fixed.candidates, sampling, graphs.generator)
        ns["code", 0], ns["semantic"] = tower.first_code(ns["main"])
        ns["unseen"] = tower.unseen(fixed.row)
        enter(0, fixed.hidden)

    def middle(i: int) -> Callable[[], None]:
        return lambda: enter(i, leave(i - 1))

    def last() -> None:
        leave(len(layers) - 1)
        frame = torch.stack((ns["main"], *(ns["code", r] for r in range(len(tower.residual_layers) + 1))), dim=-1)
        fixed.frame.copy_(frame)
        fixed.flags.copy_(torch.stack((frame[0, 0], ns["finite", len(tower.residual_layers)].long())))
        fixed.row.copy_(frame)
        fixed.pos.add_(1)
        fixed.window.copy_(torch.cat((fixed.window[1:], frame[0, :1])))

    def call(i: int) -> Callable[..., None]:
        attention = layers[i].transformer_layer.attention

        def run(state: Any, end: int) -> None:
            fixed.tower_attended[i].copy_(attention.attend(*ns["qkv", i], fixed.pos, end, state.k[i], state.v[i]))

        return run

    outputs = [fixed.frame, fixed.flags, fixed.row, fixed.pos, fixed.window]
    stretches: list[tuple[Callable[[], None], Sequence[torch.Tensor]]] = [(first, []), *((middle(i), []) for i in range(1, len(layers))), (last, outputs)]
    return Stretches(graphs, stretches, [call(i) for i in range(len(layers))], ns)


def graphable(backbone: Backbone, tower: CodebookTower) -> bool:
    """The step splits at attention calls: the backbone needs a softmax attention layer, the tower a layer."""
    return any(layer.layer_type == "full_attention" for layer in backbone.model.layers) and bool(tower.layers)
