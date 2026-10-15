"""Prompting and sampling (fish-speech ``models/text2semantic/inference.py``, ``conversation.py``,
``content_sequence.py``; Fish Audio Research License).

The prompt is upstream's conversation: a system message ("convert the provided text to speech reference to
the following:" with each reference's transcript, tagged ``<|speaker:i|>``, then all the references' codes),
a user message with the text, and an open assistant message in voice modality. Each text part is tokenised
on its own, special tokens recognised inline; a code frame's main token is ``semantic_begin_id + code``.

Every decoding step draws from the request's generator in upstream's order: the main token twice (at the
request's temperature and top-p, then at 1.0 and 0.9 for repetition-aware sampling: when the first draw is a
semantic token already among the last ten, the second is used), then codebooks 1–9 from the fast transformer.
A draw is ``argmax(probs / -log(u))`` with ``u`` uniform, as ``multinomial_sample_one_no_sync``.

Text with ``<|speaker:i|>`` tags is split into turns and grouped (at most five speakers and ``chunk_length``
bytes per group); each group is generated in turn, conditioned on the ones before it.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import torch

if TYPE_CHECKING:
    from mizuoto.runtime.graphs import Graphs
    from mizuoto_fishspeech.model import DualARTransformer

log = logging.getLogger(__name__)
IM_START = "<|im_start|>"
IM_END = "<|im_end|>"
VOICE = "<|voice|>"
RAS_WIN_SIZE = 10
RAS_HIGH_TEMP = 1.0
RAS_HIGH_TOP_P = 0.9
TOP_K = 30
PROMPT_RESERVE = 2048
"""Upstream refuses a prompt longer than ``max_seq_len - 2048``."""

SPEAKER_RE = re.compile(r"(<\|speaker:\d+\|>)")


@dataclass
class Sampling:
    temperature: float = 0.8
    top_p: float = 0.8
    max_new_tokens: int = 1024
    """0: as many as the model's context leaves."""


# -- prompt ------------------------------------------------------------------------------------------


Part = str | torch.Tensor
"""A text part, or the codes (codebooks, frames) of a voice part."""


def system_parts(prompt_texts: list[str], prompt_codes: list[torch.Tensor]) -> list[Part]:
    if not (prompt_texts and prompt_codes):
        return ["convert the provided text to speech"]
    tagged = [t if re.search(r"<\|speaker:\d+\|>", t) else f"<|speaker:{i}|>{t}" for i, t in enumerate(prompt_texts)]
    codes = torch.cat(list(prompt_codes), dim=1)
    return ["convert the provided text to speech reference to the following:\n\nText:\n", "\n".join(tagged), "\n\nSpeech:\n", codes]


def message(role: str, parts: list[Part], voice: bool = False, end: bool = True) -> list[Part]:
    out: list[Part] = [f"{IM_START}{role}\n{VOICE if voice else ''}", *parts]
    if end:
        out.append(IM_END + "\n")
    return out


def encode(parts: list[Part], tokenizer: Any, semantic_begin_id: int, num_codebooks: int) -> torch.Tensor:
    """``encode_for_inference``: (1 + codebooks, length) with the codes under each voice frame."""
    tokens: list[torch.Tensor] = []
    codes: list[torch.Tensor] = []
    vq_mask: list[torch.Tensor] = []
    for part in parts:
        if isinstance(part, str):
            t = torch.tensor(tokenizer.encode(part), dtype=torch.long)
            vq_mask.append(torch.zeros_like(t, dtype=torch.bool))
        else:
            c = part.clone().to(torch.int)
            t = (c[0] + semantic_begin_id).to(torch.long)
            codes.append(c)
            vq_mask.append(torch.ones_like(t, dtype=torch.bool))
        tokens.append(t)
    flat = torch.cat(tokens)
    values = torch.zeros((num_codebooks + 1, len(flat)), dtype=torch.long)
    values[0] = flat
    if codes:
        values[1:, torch.cat(vq_mask)] = torch.cat(codes, dim=1).to(dtype=torch.long)
    return values


def split_turns(text: str) -> list[str]:
    """``split_text_by_speaker``: each ``<|speaker:i|>`` with the text after it (text before the first tag is dropped)."""
    parts = SPEAKER_RE.split(text)
    turns = []
    i = 0
    while i < len(parts):
        part = parts[i].strip()
        if SPEAKER_RE.match(part):
            if i + 1 < len(parts):
                turns.append((part + parts[i + 1]).strip())
                i += 2
            else:
                turns.append(part)
                i += 1
        else:
            i += 1
    return turns


def group_turns(turns: list[str], max_speakers: int, max_bytes: int) -> list[str]:
    """``group_turns_into_batches``."""
    batches: list[str] = []
    current: list[str] = []
    size = 0
    for turn in turns:
        n = len(turn.encode("utf-8"))
        if len(current) >= max_speakers or (size + n > max_bytes and current):
            batches.append("\n".join(current))
            current, size = [turn], n
        else:
            current.append(turn)
            size += n
    if current:
        batches.append("\n".join(current))
    return batches


def batches_of(text: str, chunk_length: int) -> list[str]:
    turns = split_turns(text)
    return group_turns(turns, max_speakers=5, max_bytes=chunk_length) if turns else [text]


# -- sampling ----------------------------------------------------------------------------------------


def logits_to_probs(logits: torch.Tensor, temperature: torch.Tensor, top_p: torch.Tensor, top_k: int) -> torch.Tensor:
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    cum_probs = torch.cumsum(torch.nn.functional.softmax(sorted_logits, dim=-1), dim=-1)
    indices = torch.arange(sorted_logits.shape[-1], device=sorted_logits.device)
    remove = (cum_probs > top_p) | (indices >= top_k)
    remove[..., 0].fill_(False)  # upstream assigns a Python bool, a copy to the device that waits for it
    remove = remove.scatter(dim=-1, index=sorted_indices, src=remove)
    logits = torch.where(remove, float("-Inf"), logits)
    logits = logits / torch.clip(temperature, min=1e-5)
    return torch.nn.functional.softmax(logits, dim=-1)


def sample(logits: torch.Tensor, temperature: torch.Tensor, top_p: torch.Tensor, generator: torch.Generator) -> torch.Tensor:
    probs = logits_to_probs(logits[0, -1], temperature, top_p, TOP_K)
    q = torch.empty_like(probs).uniform_(generator=generator)
    return torch.argmax(probs / -torch.log(q), dim=-1, keepdim=True).to(dtype=torch.int)


class Decoder:
    """Decodes frames for one loaded model, request after request: eagerly, or on an NVIDIA GPU with CUDA graphs
    (``compute.cuda_graphs``; ``mizuoto.runtime.graphs``).

    The prompt's step runs eagerly. Each later step works on fixed tensors (``cur``, ``pos``, ``logits``, ``hidden``,
    ``frame``, ``previous``): the slow transformer's step (captured, as one graph per stretch between its attention
    calls, ``LayerSteps``), then the frame's draws and the fast transformer's ten codebooks (one graph), with one wait
    per frame for whether it ended with ``<|im_end|>``. Graphs replay the eager kernels, so the codes are the same;
    draws come from the graphs' own generator, loaded from the request's and written back.
    """

    def __init__(self, model: DualARTransformer, im_end_id: int, graphs: Callable[[], Graphs | None] | None = None) -> None:
        self.model = model
        self.im_end_id = im_end_id
        self.generator: torch.Generator | None = None
        self.check: Callable[[], None] = lambda: None
        self.max_new_tokens = 0
        self._graphs_for = graphs or (lambda: None)
        self._graphs: Graphs | None = None
        self._tail_graph: Any = None
        self._steps: Any = None
        self._allocated = False
        self.failed = False
        """A capture failed: this decoder runs eagerly from then on."""
        self.graphed = False
        """The last request decoded with graphs."""

    def _allocate(self) -> None:
        """The fixed tensors, on the model's device (where its first parameter is during a request)."""
        model = self.model
        p = next(model.parameters())
        self.device, self.dtype = p.device, p.dtype
        c = model.config
        self.dim = dim = 1 + c.num_codebooks
        # the request's sampling settings, filled per request so that one set of graphs serves every request
        self.temperature = torch.zeros((), device=self.device, dtype=self.dtype)
        self.top_p = torch.zeros((), device=self.device, dtype=self.dtype)
        self.high_temp = torch.tensor(RAS_HIGH_TEMP, device=self.device, dtype=self.dtype)
        self.high_top_p = torch.tensor(RAS_HIGH_TOP_P, device=self.device, dtype=self.dtype)
        bias = torch.full((1, 1, c.vocab_size), float("-inf"), device=self.device, dtype=self.dtype)
        bias[0, 0, c.semantic_begin_id : c.semantic_end_id + 1] = 0.0
        bias[0, 0, self.im_end_id] = 0.0
        self.bias = bias
        self.fast_caches = model.new_fast_caches()
        """Kept across steps as upstream's: each step rewrites the positions it reads."""
        self.cur = torch.zeros((1, dim, 1), dtype=torch.int, device=self.device)
        self.pos = torch.zeros(1, dtype=torch.int, device=self.device)
        self.logits = torch.zeros((1, 1, c.vocab_size), dtype=self.dtype, device=self.device)
        self.hidden = torch.zeros((1, 1, c.fast_dim), dtype=self.dtype, device=self.device)
        self.frame = torch.zeros((dim, 1), dtype=torch.int, device=self.device)
        self.previous = torch.zeros((dim, RAS_WIN_SIZE), dtype=torch.int, device=self.device)
        self.ended = torch.zeros(1, dtype=torch.bool, device=self.device)
        self._allocated = True

    def start(self, s: Sampling, generator: torch.Generator, check: Callable[[], None]) -> None:
        """A new request's sampling settings, generator and cancel check (call where the model runs)."""
        if not self._allocated:
            self._allocate()
        # made as upstream makes them (a Python float to the model's dtype), then copied in
        self.temperature.copy_(torch.tensor(s.temperature, device=self.device, dtype=self.dtype))
        self.top_p.copy_(torch.tensor(s.top_p, device=self.device, dtype=self.dtype))
        self.max_new_tokens = s.max_new_tokens
        self.generator = generator
        self.check = check
        self.graphed = False
        for cache in self.fast_caches:
            cache.k.zero_()
            cache.v.zero_()

    def _frame(self, logits: torch.Tensor, hidden: torch.Tensor, previous: torch.Tensor | None) -> torch.Tensor:
        """The frame (1 + codebooks, 1) from the slow step's logits and the fast transformer's input."""
        model, c, g = self.model, self.model.config, self.generator
        assert g is not None
        biased = logits + self.bias
        main = sample(biased, self.temperature, self.top_p, g)
        high = sample(biased, self.high_temp, self.high_top_p, g)
        if previous is not None:
            in_window = (previous[0] == main).any()
            semantic = (main >= c.semantic_begin_id) & (main <= c.semantic_end_id)
            main = torch.where(in_window & semantic, high, main)
        codebooks = [main]
        fast_caches = self.fast_caches
        model.forward_generate_fast(hidden, 0, fast_caches)
        a = torch.clamp(main - c.semantic_begin_id, min=0, max=c.codebook_size - 1)
        h = model.fast_embeddings(a)
        codebooks.append(a)
        for i in range(1, c.num_codebooks):
            out = model.forward_generate_fast(h, i, fast_caches)
            a = sample(out, self.temperature, self.top_p, g)
            h = model.fast_embeddings(a)
            codebooks.append(a)
        return torch.stack(codebooks, dim=1).T

    def _step(self, x: torch.Tensor, input_pos: torch.Tensor, kv_len: int, caches: list[Any], previous: torch.Tensor | None, math_kernel: bool) -> torch.Tensor:
        """``decode_one_token_ar``: the frame (1 + codebooks, 1) after ``x``."""
        result = self.model.forward_generate(x, input_pos, kv_len, caches, math_kernel)
        return self._frame(result.logits, result.hidden_states, previous)

    # -- a decoding step over the fixed tensors --------------------------------------------------

    def _slow(self, caches: list[Any], kv_len: int) -> None:
        """The slow transformer on ``cur`` at ``pos``: ``logits`` and ``hidden``."""
        mask = self.model.step_mask(kv_len, self.device)
        if self._graphs is not None and self._steps is None:
            self._steps = self._capture(self._layer_steps)
        if self._steps is not None:
            self._steps(mask, caches)
            return
        result = self.model.forward_generate(self.cur, self.pos, kv_len, caches, True, last=True)
        self.logits.copy_(result.logits)
        self.hidden.copy_(result.hidden_states)

    def _layer_steps(self) -> Any:
        from mizuoto.runtime.graphs import LayerSteps

        model = self.model
        layers = list(model.layers)
        rope: list[Any] = [None]

        def prologue() -> torch.Tensor:
            rope[0] = model.get_buffer("freqs_cis")[self.pos]
            return model.input_embeddings(self.cur)

        def epilogue(x: torch.Tensor) -> None:
            result = model.output_step(x)
            self.logits.copy_(result.logits)
            self.hidden.copy_(result.hidden_states)

        assert self._graphs is not None
        heads = model.config.n_head * model.config.head_dim
        return LayerSteps(
            self._graphs,
            len(layers),
            prologue,
            lambda i, h: layers[i].attention_in(h, rope[0]),
            lambda i, qkv, mask, caches: layers[i].attention.attend(*qkv, mask, caches[i], True),
            lambda i, h, attended: layers[i].attention_out(h, attended),
            epilogue,
            lambda i: torch.zeros((1, 1, heads), dtype=self.dtype, device=self.device),
            [self.logits, self.hidden],
        )

    def _tail(self) -> None:
        """The frame's draws and codebooks (``frame``), then the next step's ``cur``, ``pos``, ``previous`` and ``ended``."""
        frame = self._frame(self.logits, self.hidden, self.previous)
        self.frame.copy_(frame)
        self.cur.copy_(frame.view(1, self.dim, -1))
        self.pos.add_(1)
        self.previous.copy_(self.previous.roll(-1, dims=1))
        self.previous[:, -1] = frame.view(self.dim, -1)[:, 0]
        self.ended.copy_(frame[0, -1] == self.im_end_id)

    def _decode_step(self) -> None:
        if self._graphs is not None and self._tail_graph is None:
            state = [self.frame, self.cur, self.pos, self.previous, self.ended, *(t for cache in self.fast_caches for t in (cache.k, cache.v))]
            assert self._graphs is not None
            graphs = self._graphs
            self._tail_graph = self._capture(lambda: graphs.capture(self._tail, state))
        if self._tail_graph is not None:
            self._tail_graph.replay()
        else:
            self._tail()

    def _capture(self, make: Callable[[], Any]) -> Any:
        from mizuoto.runtime.device import is_oom

        try:
            return make()
        except Exception as e:  # any failure to capture means running eagerly (the same computation)
            if is_oom(e):
                raise
            log.warning("Fish Speech: CUDA graphs failed (%s: %s); decoding eagerly", type(e).__name__, e)
            if self._graphs is not None:
                self.generator = self._graphs.generator  # a fresh one, in the same state, after a failed capture
            self.failed = True
            self._graphs = None
            self._tail_graph = self._steps = None
            return None

    # -- one batch of text ------------------------------------------------------------------------

    def generate(self, prompt: torch.Tensor, progress: Callable[[int, int], None] | None = None) -> torch.Tensor:
        """``generate``: the prompt (1 + codebooks, T) followed by the new frames (the last one ``<|im_end|>``)."""
        graphs = None if self.failed else self._graphs_for()
        if graphs is not self._graphs:  # none, or new ones (the weights moved): capture again
            self._graphs, self._tail_graph, self._steps = graphs, None, None
        if graphs is None:
            return self._generate(prompt, progress)
        self.graphed = True
        assert self.generator is not None
        request = self.generator
        with torch.cuda.device(graphs.device), graphs.drawing(request) as own:
            self.generator = own
            try:
                return self._generate(prompt, progress)
            finally:
                self.generator = request

    def _generate(self, prompt: torch.Tensor, progress: Callable[[int, int], None] | None) -> torch.Tensor:
        c = self.model.config
        t = prompt.size(1)
        if t >= c.max_seq_len:
            raise ValueError(f"Input sequence length {t} exceeds max_seq_len {c.max_seq_len}")
        max_new = self.max_new_tokens
        if max_new:
            max_new = min(max_new, c.max_seq_len - t)
        else:
            max_new = c.max_seq_len - t
        dim = self.dim
        caches = self.model.new_caches()
        prompt = prompt.to(self.device)
        input_pos = torch.arange(0, t, device=self.device, dtype=torch.long)
        self.check()
        token = self._step(prompt.view(1, dim, -1), input_pos, t, caches, None, False).clone()
        frames = [token]
        self.cur.copy_(token.view(1, dim, -1))
        self.pos.fill_(t)
        self.previous.zero_()
        for i in range(max_new - 1):
            self.check()
            if progress is not None:
                progress(i + 1, max_new)
            self._slow(caches, t + i + 1)
            self._decode_step()
            frames.append(self.frame.clone())
            if bool(self.ended.item()):  # the step's one wait for the device
                break
        return torch.cat([prompt, *frames], dim=1)


def generate_long(
    model: DualARTransformer,
    tokenizer: Any,
    text: str,
    prompt_texts: list[str],
    prompt_codes: list[torch.Tensor],
    sampling: Sampling,
    chunk_length: int,
    generator: torch.Generator,
    check: Callable[[], None],
    progress: Callable[[int, int, int, int], None] | None = None,
    decoder: Decoder | None = None,
) -> Iterator[torch.Tensor]:
    """``generate_long`` for one sample: the codes (codebooks, frames) of each batch of text, in order. ``decoder``: the
    loaded model's (kept with its graphs), else a new eager one."""
    c = model.config
    im_end_id = tokenizer.token_to_id(IM_END)
    use_prompt = bool(prompt_texts) and bool(prompt_codes)
    conversation: list[Part] = message("system", system_parts(prompt_texts if use_prompt else [], [p.cpu() for p in prompt_codes] if use_prompt else []))
    batches = batches_of(text, chunk_length)
    if decoder is None:
        decoder = Decoder(model, im_end_id)
    decoder.start(sampling, generator, check)
    for index, batch in enumerate(batches):
        conversation += message("user", [batch])
        parts = conversation + message("assistant", [], voice=True, end=False)
        encoded = encode(parts, tokenizer, c.semantic_begin_id, c.num_codebooks)
        if encoded.size(1) > c.max_seq_len - PROMPT_RESERVE:
            raise ValueError(f"Prompt is too long: {encoded.size(1)} > {c.max_seq_len - PROMPT_RESERVE}")
        length = encoded.size(1)
        report = (lambda done, total, index=index: progress(index, len(batches), done, total)) if progress is not None else None
        y = decoder.generate(encoded, report)
        codes = y[1:, length:-1].clone()
        if not bool((codes >= 0).all()):
            raise ValueError("Negative code generated")
        conversation += message("assistant", [codes.cpu()], voice=True)
        yield codes
