"""CUDA graphs for decoding steps (``compute.cuda_graphs``).

An autoregressive model on a GPU launches thousands of small kernels per step (Qwen3-TTS: about 6 000 per 80 ms frame),
so eagerly the step is as slow as the CPU can launch them. A CUDA graph records a step's kernels once and launches them
all at once. It replays exactly the kernels the eager code launched, with the same arguments, so its results are the
eager code's bit for bit; what it needs is fixed shapes and no waits for the device inside the step.

- ``Graphs`` captures callables that read and write fixed tensors (a model's "ports"). Random draws come from a generator
  the graphs own (``generator``): load the request's into it before a run and write it back after (``drawing``), and
  seeded requests draw the same numbers as eagerly.
- ``LayerSteps`` (and ``DecoderSteps``, its form for ``QwenDecoder``) runs a layer stack's one-token step as one graph
  per stretch between two attention calls. Attention runs eagerly between them, over keys whose length grows each
  step, so the cache and the attention kernel are the eager code's; everything else (norms, projections, rotary
  embeddings, the MLPs) is replayed.

Graphs hold the addresses of the weights they read, so they are only for models wholly on their device (no streamed
units), and a caller drops them when the weights move (``signature``). NVIDIA only for now (ROCm's HIP graphs are not
checked here).
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Generator, Sequence
from contextlib import contextmanager
from typing import Any

import torch
import torch.version

log = logging.getLogger(__name__)


def usable(device: torch.device | str, enabled: bool = True) -> bool:
    """Whether to capture graphs on ``device``: ``compute.cuda_graphs`` on, an NVIDIA GPU, torch's caching allocator.
    Under cudaMallocAsync (``compute.cuda_allocator = async``) a capture's warm-up frees its temporaries the
    stream-ordered way, and replays then fault with an illegal memory access, so the steps run eagerly there."""
    device = torch.device(device)
    if not (bool(enabled) and device.type == "cuda" and getattr(torch.version, "hip", None) is None and torch.cuda.is_available()):
        return False
    return torch.cuda.get_allocator_backend() != "cudaMallocAsync"


def signature(modules: Sequence[torch.nn.Module]) -> tuple[int, ...]:
    """Where the modules' weights are: graphs captured over them are valid while this stays the same."""
    return tuple(t.data_ptr() for m in modules for t in (*m.parameters(), *m.buffers()))


class Graphs:
    """Graphs of one model on one device, with their stream and generator."""

    def __init__(self, device: torch.device | str) -> None:
        self.device = torch.device(device)
        self.stream = torch.cuda.Stream(device=self.device)
        self.generator = torch.Generator(device=self.device)
        self.captures = 0
        self.seconds = 0.0

    def pool(self) -> Any:
        """A memory pool for graphs that are always replayed in the order they were captured."""
        return torch.cuda.graph_pool_handle()

    @contextmanager
    def drawing(self, generator: torch.Generator) -> Generator[torch.Generator, None, None]:
        """Draw from ``generator``'s stream through the graphs' own generator (the one they captured)."""
        self.generator.set_state(generator.get_state())
        try:
            yield self.generator
        finally:
            generator.set_state(self.generator.get_state())

    def capture(self, fn: Callable[[], Any], state: Sequence[torch.Tensor] = (), pool: Any = None) -> torch.cuda.CUDAGraph:
        """Capture ``fn``. It runs once first (outside the graph: lazy initialisation), after which ``state`` (the tensors
        it changes in place) and the generator are put back, so capturing leaves no trace. Raises what ``fn`` or the
        capture raises (a wait for the device inside ``fn``, say); the caller then runs eagerly, drawing from
        ``generator`` (a new one after a failed capture)."""
        import time

        started = time.perf_counter()
        saved = self.generator.get_state()
        before = [t.clone() for t in state]
        current = torch.cuda.current_stream(self.device)
        self.stream.wait_stream(current)
        try:
            with torch.cuda.stream(self.stream):
                fn()
            current.wait_stream(self.stream)
            for t, b in zip(state, before, strict=True):
                t.copy_(b)
            self.generator.set_state(saved)
            graph = torch.cuda.CUDAGraph()
            graph.register_generator_state(self.generator)
            # ``torch.cuda.graph`` without its ``gc.collect()`` and ``empty_cache()`` (they free memory, not needed for a
            # capture, and cost tens of milliseconds each: a layer stack's step is dozens of captures)
            with torch.cuda.stream(self.stream):
                graph.capture_begin(*(() if pool is None else (pool,)), capture_error_mode="thread_local")
                try:
                    fn()
                finally:
                    graph.capture_end()
            # Beginning a capture resets the registered generators' seed and offset tensors with kernels on the
            # capture stream; the first replay (on the current stream) must not overtake them, or it draws from offset 0.
            current.wait_stream(self.stream)
        except BaseException:
            # An aborted capture leaves the generator expecting one (every later draw raises): callers falling back to
            # eager steps draw from a fresh generator in the same state (``generator`` after the call).
            fresh = torch.Generator(device=self.device)
            fresh.set_state(saved)
            self.generator = fresh
            raise
        finally:
            self.generator.set_state(saved)
        self.captures += 1
        self.seconds += time.perf_counter() - started
        return graph


class LayerSteps:
    """A layer stack's one-token step as graphs around each layer's attention call.

    The step is ``h = prologue()``, then for each layer ``i``: ``inputs = before(i, h)`` (captured),
    ``attended = attend(i, inputs, *args)`` (eager: it reads a cache whose length grows), ``h = after(i, h, attended)``
    (captured), and finally ``epilogue(h)`` (captured), which writes what the caller needs into its own tensors
    (``state``: those it writes). ``prologue`` reads only the caller's fixed tensors, filled before each run;
    ``attended_like(i)`` gives a zeroed tensor shaped like layer ``i``'s attention output. Each graph replays what the
    callables launched eagerly, so a step computes what ``prologue``, the layers and ``epilogue`` compute together.
    """

    def __init__(
        self,
        graphs: Graphs,
        layers: int,
        prologue: Callable[[], torch.Tensor],
        before: Callable[[int, torch.Tensor], Any],
        attend: Callable[..., torch.Tensor],
        after: Callable[[int, torch.Tensor, torch.Tensor], torch.Tensor],
        epilogue: Callable[[torch.Tensor], None],
        attended_like: Callable[[int], torch.Tensor],
        state: Sequence[torch.Tensor],
    ) -> None:
        self.layers = layers
        self.attend = attend
        self.attended = [attended_like(i) for i in range(layers)]
        self.inputs: list[Any] = [None] * layers
        self._h: list[Any] = [None] * layers
        pool = graphs.pool()

        def first() -> None:
            self._h[0] = prologue()
            self.inputs[0] = before(0, self._h[0])

        def middle(i: int) -> Callable[[], None]:
            def run() -> None:
                h = after(i - 1, self._h[i - 1], self.attended[i - 1])
                self._h[i] = h
                self.inputs[i] = before(i, h)

            return run

        def last() -> None:
            epilogue(after(layers - 1, self._h[layers - 1], self.attended[layers - 1]))

        parts = [first, *(middle(i) for i in range(1, layers)), last]
        self.parts = [graphs.capture(fn, state if fn is last else (), pool) for fn in parts]

    def __call__(self, *args: Any) -> None:
        for i in range(self.layers):
            self.parts[i].replay()
            self.attended[i].copy_(self.attend(i, self.inputs[i], *args))
        self.parts[-1].replay()


class DecoderSteps(LayerSteps):
    """One-token steps of a ``mizuoto.runtime.lm.qwen.QwenDecoder``: ``x`` (batch, 1, hidden) and ``positions``
    (batch, 1) are the step's inputs; ``epilogue`` gets the normed output. Called with the mask and the cache, which
    must hold the prompt already (its first update keeps the tensor it is given)."""

    def __init__(self, graphs: Graphs, decoder: Any, x: torch.Tensor, positions: torch.Tensor, epilogue: Callable[[torch.Tensor], None], state: Sequence[torch.Tensor]) -> None:
        layers = list(decoder.layers)
        rope: list[Any] = [None]

        def prologue() -> torch.Tensor:
            rope[0] = decoder.rotary_emb(x, positions)
            return x

        super().__init__(
            graphs,
            len(layers),
            prologue,
            lambda i, h: layers[i].attention_in(h, *rope[0]),
            lambda i, qkv, mask, cache: layers[i].self_attn.attend(*qkv, mask, cache),
            lambda i, h, attended: layers[i].attention_out(h, attended),
            lambda h: epilogue(decoder.norm(h)),
            lambda i: x.new_zeros(*x.shape[:2], layers[i].self_attn.o_proj.in_features),
            state,
        )
