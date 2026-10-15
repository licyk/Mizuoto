"""CosyVoice's text-to-speech-token language models (upstream ``llm/llm.py``, Apache-2.0, Alibaba).

- **v1** ``TransformerLM``: a Conformer text encoder and a 14-layer Transformer with relative positions,
  conditioned on the speaker embedding; 4096 speech tokens.
- **v2** ``Qwen2LM``: Qwen2-0.5B (``CosyVoice-BlankEN``'s shape, ``llm.pt``'s weights) over
  [sos | text | task | prompt speech]; 6561 FSQ tokens, three stop tokens.
- **v3** ``CosyVoice3LM``: the same with 200 special speech tokens (sos, eos, task, fill…) in the speech
  vocabulary.

Sampling is upstream's repetition-aware sampling (``ras_sampling``: nucleus sampling, top-p 0.8 and top-k 25,
falling back to sampling the whole distribution when the pick repeats within the last 10 tokens), with the
request's generator. The Qwen2 runs as transformers 4.51 ran it (``runtime.lm.qwen`` with ``sdpa_gqa=False``).
Streamed text (``inference_bistream``) interleaves 5 text tokens with 15 speech tokens, as upstream.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from mizuoto.runtime import ops as default_ops

log = logging.getLogger(__name__)


def nucleus_sampling(scores: torch.Tensor, generator: torch.Generator, top_p: float = 0.8, top_k: int = 25) -> int:
    prob: list[torch.Tensor] = []
    indices: list[torch.Tensor] = []
    cum = 0.0
    values, idx = scores.softmax(dim=0).sort(descending=True, stable=True)
    for i in range(len(idx)):
        if cum < top_p and len(prob) < top_k:
            cum += values[i]  # type: ignore[assignment]
            prob.append(values[i])
            indices.append(idx[i])
        else:
            break
    p = torch.tensor(prob).to(scores)
    ind = torch.tensor(indices, dtype=torch.long).to(scores.device)
    return int(ind[p.multinomial(1, replacement=True, generator=generator)].item())


def ras_sampling(scores: torch.Tensor, decoded: list[int], generator: torch.Generator, top_p: float = 0.8, top_k: int = 25, win_size: int = 10, tau_r: float = 0.1) -> int:
    top = nucleus_sampling(scores, generator, top_p, top_k)
    rep = (torch.tensor(decoded[-win_size:]).to(scores.device) == top).sum().item()
    if rep >= win_size * tau_r:
        scores[top] = -float("inf")
        top = int(scores.softmax(dim=0).multinomial(1, replacement=True, generator=generator).item())
    return top


def nucleus_device(scores: torch.Tensor, generator: torch.Generator, top_p: float = 0.8, top_k: int = 25) -> torch.Tensor:
    """``nucleus_sampling`` on a CUDA device without waiting for it: the same token (a 0-d tensor) and generator state.

    Upstream's loop reads the running sum back at each kept token and draws over the ``n`` it kept. Here the running sum
    is the same 0-d additions in the scores' dtype (so the same rounding) and comparisons, kept as a mask over the first
    ``top_k``; the draw is ``multinomial``'s own (``argmax(p / Exp(1))``) over ``top_k`` exponentials, of which the
    first ``n`` are those an ``n``-element draw gives on CUDA (each element from its own Philox subsequence, the same
    offset step for any size up to thousands). Checked against ``nucleus_sampling`` for every ``n`` from 1 to 25, in
    fp32, fp16 and bf16; not on other devices, which keep the loop."""
    values, idx = scores.softmax(dim=0).sort(descending=True, stable=True)
    k = min(top_k, values.shape[0])
    cum = values[0] + 0.0
    take = [torch.ones((), dtype=torch.bool, device=scores.device)]
    for i in range(1, k):
        take.append(cum < top_p)
        cum = cum + values[i]
    p = values[:k]
    ratio = torch.where(torch.stack(take), p / torch.empty_like(p).exponential_(1, generator=generator), -float("inf"))
    return idx[:k].gather(0, ratio.argmax().unsqueeze(0)).squeeze(0)  # indexing with a 0-d tensor would read it back


def qwen2_config(folder: Path) -> Any:
    from mizuoto.runtime.lm.qwen import QwenConfig

    c = json.loads((folder / "config.json").read_text(encoding="utf-8"))
    return QwenConfig(
        hidden_size=c["hidden_size"],
        intermediate_size=c["intermediate_size"],
        num_layers=c["num_hidden_layers"],
        num_heads=c["num_attention_heads"],
        num_kv_heads=c["num_key_value_heads"],
        head_dim=c.get("head_dim") or c["hidden_size"] // c["num_attention_heads"],
        rms_norm_eps=c.get("rms_norm_eps", 1e-6),
        rope_theta=float(c.get("rope_theta", 1_000_000.0)),
        qk_norm=False,
        attention_bias=True,
        o_bias=False,
        sdpa_gqa=False,
    ), c["vocab_size"]


class _QwenModel(nn.Module):
    """``Qwen2ForCausalLM.model``: token embedding and the decoder (the LM head is never used)."""

    def __init__(self, folder: Path, ops: Any) -> None:
        super().__init__()
        from mizuoto.runtime.lm.qwen import QwenDecoder

        config, vocab = qwen2_config(folder)
        self.model = QwenDecoder(config, ops)
        self.model.add_module("embed_tokens", ops.Embedding(vocab, config.hidden_size))

    def embed(self, ids: torch.Tensor) -> torch.Tensor:
        return self.model.get_submodule("embed_tokens")(ids)


class _QwenEncoder(nn.Module):
    """``Qwen2Encoder``: ``forward_one_step`` with transformers' cache."""

    def __init__(self, folder: Path, ops: Any) -> None:
        super().__init__()
        self.model = _QwenModel(folder, ops)

    kv_cache = "dynamic"
    """``compute.kv_cache`` (the engine sets it at load)."""

    def step(self, xs: torch.Tensor, cache: Any) -> tuple[torch.Tensor, Any]:
        from mizuoto.runtime.lm.cache import new_cache

        if cache is None:
            cache = new_cache(self.kv_cache)
        past = cache.length
        positions = torch.arange(past, past + xs.shape[1], device=xs.device).unsqueeze(0)
        return self.model.model(xs, positions, None, cache), cache


class QwenLM(nn.Module):
    """``Qwen2LM`` (v2) and ``CosyVoice3LM`` (v3)."""

    def __init__(self, version: int, qwen_folder: Path, speech_token_size: int = 6561, ops: Any = default_ops.default) -> None:
        super().__init__()
        self.version = version
        self.speech_token_size = speech_token_size
        hidden = qwen2_config(qwen_folder)[0].hidden_size
        self.llm = _QwenEncoder(qwen_folder, ops)
        if version == 3:
            self.sos, self.eos_token, self.task_id, self.fill_token = (speech_token_size + i for i in range(4))
            self.llm_decoder = ops.Linear(hidden, speech_token_size + 200, bias=False)
            self.speech_embedding = ops.Embedding(speech_token_size + 200, hidden)
            self.stop_token_ids = [speech_token_size + i for i in range(200)]
        else:
            self.sos, self.task_id = 0, 1
            self.eos_token, self.fill_token = speech_token_size, speech_token_size + 2
            self.llm_embedding = ops.Embedding(2, hidden)
            self.llm_decoder = ops.Linear(hidden, speech_token_size + 3)
            self.speech_embedding = ops.Embedding(speech_token_size + 3, hidden)
            self.stop_token_ids = [speech_token_size + i for i in range(3)]
        self.mix_ratio = (5, 15)
        self.graphs_for: Callable[[], Any] = lambda: None
        """The CUDA graphs to decode with now (``compute.cuda_graphs``; the engine sets it), else None."""
        self.steps = _Steps(self)

    @staticmethod
    def ignorable() -> list[str]:
        return ["llm.model.lm_head.weight"]

    def _special(self, index: int) -> torch.Tensor:
        table = self.speech_embedding if self.version == 3 else self.llm_embedding
        return table.weight[index].reshape(1, 1, -1)

    def _sample(self, logp: torch.Tensor, decoded: list[int], generator: torch.Generator, ignore_eos: bool) -> int:
        if ignore_eos:
            logp[self.speech_token_size] = -float("inf")  # upstream's index for every version (v3's sos)
        return ras_sampling(logp, decoded, generator)

    def inference(
        self,
        text: torch.Tensor,
        prompt_text: torch.Tensor,
        prompt_speech_token: torch.Tensor,
        generator: torch.Generator,
        check_cancel: Callable[[], None],
        min_token_text_ratio: float = 2,
        max_token_text_ratio: float = 20,
    ) -> Iterator[int]:
        text_len = text.shape[1] + prompt_text.shape[1]
        text = torch.concat([prompt_text, text], dim=1)
        if self.version == 3 and 151646 not in text:
            raise ValueError("<|endofprompt|> not detected in CosyVoice3 text or prompt_text")
        text_emb = self.llm.model.embed(text)
        speech = (
            self.speech_embedding(prompt_speech_token) if prompt_speech_token.shape[1] != 0 else torch.zeros(1, 0, text_emb.shape[-1], dtype=text_emb.dtype, device=text.device)
        )
        lm_input = torch.concat([self._special(self.sos), text_emb, self._special(self.task_id), speech], dim=1)
        min_len = int((text_len - prompt_text.shape[1]) * min_token_text_ratio)
        max_len = int((text_len - prompt_text.shape[1]) * max_token_text_ratio)
        out: list[int] = []
        cache = None
        self.steps.begin()
        for i in range(max_len):
            check_cancel()
            logp, cache = self.steps.step(lm_input, cache)
            top = self.steps.sample(logp.squeeze(dim=0), out, generator, ignore_eos=i < min_len)
            if top in self.stop_token_ids:
                break
            yield top
            out.append(top)
            lm_input = self.speech_embedding.weight[top].reshape(1, 1, -1)

    def inference_bistream(
        self,
        text: Iterator[torch.Tensor],
        prompt_text: torch.Tensor,
        prompt_speech_token: torch.Tensor,
        generator: torch.Generator,
        check_cancel: Callable[[], None],
    ) -> Iterator[int]:
        """Upstream's ``inference_bistream``: ``text`` yields token ids (1, n) as they come."""
        embed = self.llm.model.embed
        hidden = self.speech_embedding.weight.shape[1]
        speech = self.speech_embedding(prompt_speech_token) if prompt_speech_token.shape[1] != 0 else torch.zeros(1, 0, hidden, dtype=prompt_text.dtype, device=prompt_text.device)
        lm_input = torch.concat([self._special(self.sos)], dim=1)
        out: list[int] = []
        cache = None
        self.steps.begin()
        if self.version == 3:
            if 151646 not in prompt_text:
                raise ValueError("<|endofprompt|> not detected in CosyVoice3 prompt_text")
            eop = prompt_text.flatten().tolist().index(151646)
            lm_input = torch.concat([lm_input, embed(prompt_text[:, : eop + 1])], dim=1)
            prompt_text = prompt_text[:, eop + 1 :]
        text_cache = embed(prompt_text)
        r_text, r_speech = self.mix_ratio
        next_fill = (int(prompt_speech_token.shape[1] / r_speech) + 1) * r_speech - prompt_speech_token.shape[1]
        for piece in text:
            text_cache = torch.concat([text_cache, embed(piece)], dim=1)
            while speech.size(1) != 0:
                if text_cache.size(1) >= r_text:
                    lm_input = torch.concat([lm_input, text_cache[:, :r_text], speech[:, :r_speech]], dim=1)
                    text_cache, speech = text_cache[:, r_text:], speech[:, r_speech:]
                else:
                    break
            if speech.size(1) == 0:
                if (len(out) != 0 and out[-1] == self.fill_token) or (len(out) == 0 and lm_input.size(1) == 1):
                    if text_cache.size(1) >= r_text:
                        lm_text = text_cache[:, :r_text]
                        lm_input = lm_text if len(out) != 0 and out[-1] == self.fill_token else torch.concat([lm_input, lm_text], dim=1)
                        text_cache = text_cache[:, r_text:]
                    else:
                        continue
                while True:
                    check_cancel()
                    logp, cache = self.steps.step(lm_input, cache)
                    if next_fill != -1 and len(out) == next_fill:
                        top = self.fill_token
                        next_fill += r_speech + 1
                    else:
                        top = self.steps.sample(logp.squeeze(dim=0), out, generator, ignore_eos=True)
                    if top == self.fill_token:
                        next_fill = len(out) + r_speech + 1
                    out.append(top)
                    if top >= self.speech_token_size:
                        if top == self.fill_token:
                            break
                        raise ValueError(f"should not get token {top}")
                    yield top
                    lm_input = self.speech_embedding.weight[top].reshape(1, 1, -1)
        lm_input = torch.concat([lm_input, text_cache, self._special(self.task_id)], dim=1)
        while True:
            check_cancel()
            logp, cache = self.steps.step(lm_input, cache)
            top = self.steps.sample(logp.squeeze(dim=0), out, generator, ignore_eos=False)
            out.append(top)
            if top >= self.speech_token_size:
                if top == self.eos_token:
                    break
                raise ValueError(f"should not get token {top}")
            yield top
            lm_input = self.speech_embedding.weight[top].reshape(1, 1, -1)


class _Steps:
    """``QwenLM``'s decoding steps: the language model's step and the token's draw, as upstream computes them.

    On the CPU (and on devices other than CUDA) they are upstream's code. On CUDA the draw waits for the device once, to
    read the token (it is yielded, and repetition-aware sampling decides on the host whether to draw again): the nucleus
    draw is ``nucleus_device``, the masking of EOS and of a repeated token ``fill_`` (upstream's assignment of a Python
    float copies it to the device and waits). With CUDA graphs (``QwenLM.graphs_for``), one-token steps of the Qwen2
    run as graphs around its attention calls (``DecoderSteps``) and the nucleus draw as one graph per EOS mask, drawing
    from the graphs' generator loaded from the request's and written back at each token (CosyVoice 3's streamed flow
    draws from the request's between tokens). Graphs replay the eager kernels: the same tokens.
    """

    def __init__(self, lm: QwenLM) -> None:
        self.lm = lm
        self.graphs: Any = None
        self.failed = False
        self.graphed = False
        """The last request decoded with graphs."""
        self._reset()

    def _reset(self) -> None:
        self._steps: Any = None
        self._pickers: dict[bool, Any] = {}
        self.x: torch.Tensor | None = None
        self.positions: torch.Tensor | None = None
        self.logp: torch.Tensor | None = None
        self.top: torch.Tensor | None = None

    def begin(self) -> None:
        """A request: the graphs to use now (new ones when the weights moved)."""
        graphs = None if self.failed else self.lm.graphs_for()
        if graphs is not self.graphs:
            self.graphs = graphs
            self._reset()
        self.graphed = graphs is not None

    def step(self, lm_input: torch.Tensor, cache: Any) -> tuple[torch.Tensor, Any]:
        """The logits' log-softmax (1, vocabulary) after ``lm_input``, and the cache."""
        lm = self.lm
        if self.graphs is None or cache is None or lm_input.shape[1] != 1:
            y, cache = lm.llm.step(lm_input, cache)
            logp = lm.llm_decoder(y[:, -1]).log_softmax(dim=-1)
            if self.graphs is not None and self.logp is None:  # the fixed tensors, shaped as the eager step's
                self.x, self.logp = torch.zeros_like(lm_input[:, :1]), torch.zeros_like(logp)
                self.positions = torch.zeros((1, 1), dtype=torch.long, device=logp.device)
                self.top = torch.zeros((), dtype=torch.long, device=logp.device)
            return logp, cache
        assert self.x is not None and self.positions is not None and self.logp is not None
        self.x.copy_(lm_input)
        self.positions.fill_(cache.length)
        if self._steps is None:
            self._steps = self._capture(self._decoder_steps)
            if self._steps is None:
                return self.step(lm_input, cache)
        self._steps(None, cache)
        return self.logp, cache

    def _decoder_steps(self) -> Any:
        from mizuoto.runtime.graphs import DecoderSteps

        lm, logp = self.lm, self.logp
        assert logp is not None and self.x is not None and self.positions is not None

        def epilogue(y: torch.Tensor) -> None:
            logp.copy_(lm.llm_decoder(y[:, -1]).log_softmax(dim=-1))

        return DecoderSteps(self.graphs, lm.llm.model.model, self.x, self.positions, epilogue, [logp])

    def sample(self, logp: torch.Tensor, decoded: list[int], generator: torch.Generator, ignore_eos: bool) -> int:
        """``_sample`` (``ras_sampling``): the next token."""
        if logp.device.type != "cuda":
            return self.lm._sample(logp, decoded, generator, ignore_eos)
        graphs, port, top = self.graphs, self.logp, self.top
        if graphs is None or port is None or top is None or logp.data_ptr() != port.data_ptr():
            return self._resample(logp, decoded, generator, int(self._pick(logp, generator, ignore_eos)))
        with torch.cuda.device(graphs.device), graphs.drawing(generator):
            picker = self._pickers.get(ignore_eos)
            if picker is None:
                picker = self._capture(lambda: graphs.capture(lambda: top.copy_(self._pick(port[0], graphs.generator, ignore_eos)), [port, top]))
                if picker is not None:
                    self._pickers[ignore_eos] = picker
            own = graphs.generator  # after a failed capture, a fresh one in the same state (written back all the same)
            if picker is None:
                return self._resample(logp, decoded, own, int(self._pick(logp, own, ignore_eos)))
            picker.replay()
            return self._resample(logp, decoded, own, int(top))  # the token's one wait

    def _pick(self, logp: torch.Tensor, generator: torch.Generator, ignore_eos: bool) -> torch.Tensor:
        if ignore_eos:
            logp[self.lm.speech_token_size].fill_(-float("inf"))  # upstream's index for every version (v3's sos)
        return nucleus_device(logp, generator)

    @staticmethod
    def _resample(logp: torch.Tensor, decoded: list[int], generator: torch.Generator, top: int, win_size: int = 10, tau_r: float = 0.1) -> int:
        """``ras_sampling``'s second half: a token repeated within the window is drawn again from the whole
        distribution (``multinomial``'s computation, without its checks)."""
        if decoded[-win_size:].count(top) >= win_size * tau_r:
            logp[top].fill_(-float("inf"))
            probs = logp.softmax(dim=0)
            top = int(probs.div(torch.empty_like(probs).exponential_(1, generator=generator)).argmax())
        return top

    def _capture(self, make: Callable[[], Any]) -> Any:
        from mizuoto.runtime.device import is_oom

        try:
            return make()
        except Exception as e:  # any failure to capture means running eagerly (the same computation)
            if is_oom(e):
                raise
            log.warning("CosyVoice: CUDA graphs failed (%s: %s); decoding eagerly", type(e).__name__, e)
            self.failed = True
            self.graphs = None
            self._reset()
            return None


class TransformerLM(nn.Module):
    """CosyVoice 1's ``TransformerLM``, sized by the model's yaml (``config["llm"]``)."""

    def __init__(self, cfg: dict[str, Any], ops: Any = default_ops.default) -> None:
        super().__init__()
        from mizuoto.nn.transformer.wenet import Encoder

        te, lm = cfg["text_encoder"], cfg["llm"]
        text_in, llm_in, llm_out = int(cfg["text_encoder_input_size"]), int(cfg["llm_input_size"]), int(cfg["llm_output_size"])
        self.speech_token_size = int(cfg["speech_token_size"])
        self.sos, self.task_id, self.eos_token = 0, 1, self.speech_token_size
        self.text_embedding = ops.Embedding(int(cfg["text_token_size"]), text_in)
        self.text_encoder = Encoder(
            int(te.get("input_size", text_in)),
            int(te["output_size"]),
            int(te["attention_heads"]),
            int(te["linear_units"]),
            int(te["num_blocks"]),
            kind="conformer",
            input_layer=te.get("input_layer", "linear"),
            static_chunk_size=int(te.get("static_chunk_size", 1)),
            ops=ops,
        )
        self.text_encoder_affine_layer = ops.Linear(int(te["output_size"]), llm_in)
        self.llm_embedding = ops.Embedding(2, llm_in)
        self.llm = Encoder(
            int(lm.get("input_size", llm_in)),
            int(lm["output_size"]),
            int(lm["attention_heads"]),
            int(lm["linear_units"]),
            int(lm["num_blocks"]),
            kind="transformer",
            input_layer=lm.get("input_layer", "linear_legacy"),
            static_chunk_size=int(lm.get("static_chunk_size", 1)),
            ops=ops,
        )
        self.llm_decoder = ops.Linear(llm_out, self.speech_token_size + 1)
        self.speech_embedding = ops.Embedding(self.speech_token_size, llm_in)
        self.spk_embed_affine_layer = ops.Linear(int(cfg.get("spk_embed_dim", 192)), llm_in)

    @staticmethod
    def ignorable() -> list[str]:
        return []

    def inference(
        self,
        text: torch.Tensor,
        prompt_text: torch.Tensor,
        prompt_speech_token: torch.Tensor,
        embedding: torch.Tensor,
        generator: torch.Generator,
        check_cancel: Callable[[], None],
    ) -> Iterator[int]:
        device = text.device
        text = torch.concat([prompt_text, text], dim=1)
        text_len = torch.tensor([text.shape[1]], dtype=torch.int32, device=device)
        x = self.text_embedding(text)
        x, mask = self.text_encoder(x, text_len)
        text_len = mask.squeeze(1).sum(1)
        x = self.text_encoder_affine_layer(x)
        hidden = x.shape[-1]
        if embedding.shape[0] != 0:
            emb = self.spk_embed_affine_layer(F.normalize(embedding, dim=1)).unsqueeze(dim=1)
        else:
            emb = torch.zeros(1, 0, hidden, dtype=x.dtype).to(device).to(x.dtype)
        sos = self.llm_embedding.weight[self.sos].reshape(1, 1, -1)
        task = self.llm_embedding.weight[self.task_id].reshape(1, 1, -1)
        speech = self.speech_embedding(prompt_speech_token) if prompt_speech_token.shape[1] != 0 else torch.zeros(1, 0, hidden, dtype=x.dtype).to(device)
        lm_input = torch.concat([sos, emb, x, task, speech], dim=1)
        min_len = int((text_len - prompt_text.shape[1]) * 2)
        max_len = int((text_len - prompt_text.shape[1]) * 20)
        out: list[int] = []
        offset = 0
        cache = torch.zeros((0, 0, 0, 0), device=device)
        for i in range(max_len):
            check_cancel()
            att_mask = torch.tril(torch.ones((1, lm_input.shape[1], lm_input.shape[1]), device=device)).to(torch.bool)
            y, cache = self.llm.forward_chunk(lm_input, offset, cache, att_mask)
            logp = self.llm_decoder(y[:, -1]).log_softmax(dim=-1).squeeze(dim=0)
            if i < min_len:
                logp[self.speech_token_size] = -float("inf")
            top = ras_sampling(logp, out, generator)
            if top == self.eos_token:
                break
            yield top
            out.append(top)
            offset += lm_input.size(1)
            lm_input = self.speech_embedding.weight[top].reshape(1, 1, -1)
