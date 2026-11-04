"""CT-Transformer punctuation (FunASR's ``CTTransformer``): commas, full stops and question marks for unpunctuated text.
Ported from FunASR 1.4.21 (MIT, Copyright (c) 2022 Alibaba Group).

The model is a token embedding, a ``SANMEncoder`` (``input_layer: pe``; ``mizuoto.engines.asr.paraformer.sanm.Encoder``
with every frame real) and a linear layer to the punctuation classes of the config's ``punc_list``
(``<unk>``, ``_`` for none, ``，``, ``。``, ``？``, ``、``), each token's class its ``topk(1)``.

``punctuate`` is ``CTTransformer.inference`` for one text, as ``AutoModel`` calls it after joining the VAD segments'
transcripts (``split_size`` 20, no jieba user dictionary, no ``seg_dict``: what the zh-cn vocab272727 model ships):

- words: the text split on whitespace, then each run of ASCII characters one word and every other character a word of
  its own (``split_words`` without jieba); a word's id is its entry in ``tokens.json``, ``<unk>`` when absent (the
  lookup is case-sensitive, and the vocabulary's English words are lower case);
- mini-sentences of ``split_size`` words, each run with the words carried over from the one before: all but the last
  keep their words up to the last ``。``/``？`` predicted at positions 2…n-2 and carry the rest; with no such stop, a
  mini-sentence longer than 200 words is cut at its last comma there (which becomes ``。``), otherwise all its words are
  carried;
- the text: a word starting with an ASCII character is capitalised (``str.capitalize``: the rest lower-cased) at the
  start of each mini-sentence and after ``。``/``？``, gets a space before it at the start of each mini-sentence (so
  the text can begin with a space) and after another ASCII word, and takes ``,`` ``.`` ``?`` instead of ``，`` ``。``
  ``？``; the end gets a full stop (a trailing ``，``/``、`` becomes ``。``, ``,`` becomes ``.``, and text ending in
  anything but ``。``/``？`` (non-ASCII) or ``.``/``?`` (ASCII) gets ``。``/``.`` appended), its last class set to
  ``sentence_end_id``.

Models with a ``jieba_usr_dict`` (FunASR's word-level punctuation models) split words with jieba and are not ported.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from torch import nn

from mizuoto.engines.asr.paraformer.model import load_config
from mizuoto.engines.asr.paraformer.sanm import Encoder
from mizuoto.runtime import ops as default_ops

CACHE_POP_TRIGGER_LIMIT = 200
"""A mini-sentence (with its carried words) longer than this, without a predicted stop, is cut at its last comma."""


class CTTransformer(nn.Module):
    """``embed``, ``encoder`` and ``decoder``, named as the checkpoint names them."""

    def __init__(self, config: dict[str, Any], vocab: int, ops: Any = default_ops.default) -> None:
        super().__init__()
        conf, enc = config.get("model_conf", {}), config["encoder_conf"]
        self.punc_list: list[str] = [str(p) for p in conf["punc_list"]]
        self.sentence_end_id = int(conf.get("sentence_end_id", 3))
        size = int(enc["output_size"])
        self.embed = ops.Embedding(vocab, int(conf.get("embed_unit", 128)))
        self.encoder = Encoder(int(enc["input_size"]), size, int(enc["attention_heads"]), int(enc["linear_units"]), int(enc["num_blocks"]), int(enc["kernel_size"]), ops)
        self.decoder = ops.Linear(int(conf.get("att_unit", 256)), len(self.punc_list))

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        """``punc_forward`` for one sequence of token ids (1, time): the classes' logits (1, time, classes)."""
        mask = torch.ones(ids.shape, dtype=torch.bool, device=ids.device)[:, None, :]
        return self.decoder(self.encoder(self.embed(ids), mask))


class Vocabulary:
    """``tokens.json`` as ``CharTokenizer`` reads it: a word's id, ``<unk>``'s for a word not in it."""

    def __init__(self, tokens: list[str], unk: str = "<unk>") -> None:
        self.ids: dict[str, int] = {t: i for i, t in enumerate(tokens)}
        self.unk_id = self.ids[unk]

    def encode(self, words: list[str]) -> list[int]:
        return [self.ids.get(w, self.unk_id) for w in words]


def load_vocabulary(folder: Path) -> Vocabulary:
    config = load_config(folder)
    unk = str((config.get("tokenizer_conf") or {}).get("unk_symbol", "<unk>"))
    return Vocabulary(list(json.loads((folder / "tokens.json").read_text(encoding="utf-8"))), unk)


def build(folder: Path, ops: Any = default_ops.default) -> tuple[CTTransformer, Vocabulary]:
    """The model (weights not loaded: ``load_into(model, load_state_dict(folder / "model.pt"))``) and its vocabulary."""
    if (folder / "jieba_usr_dict").exists() or (folder / "seg_dict").exists():
        raise ValueError(f"{folder}: word-level punctuation models (jieba_usr_dict, seg_dict) are not supported")
    vocabulary = load_vocabulary(folder)
    return CTTransformer(load_config(folder), len(vocabulary.ids), ops), vocabulary


def split_words(text: str) -> list[str]:
    """``split_words`` without jieba: whitespace-separated pieces, a run of ASCII characters one word, any other
    character a word."""
    words: list[str] = []
    for piece in text.split():
        current = ""
        for c in piece:
            if len(c.encode()) == 1:
                current += c
            else:
                if current:
                    words.append(current)
                    current = ""
                words.append(c)
        if current:
            words.append(current)
    return words


def split_to_mini_sentences(words: list[Any], limit: int) -> list[list[Any]]:
    """``split_to_mini_sentence``: runs of ``limit`` words, the rest last."""
    if limit <= 1:
        raise ValueError("split_size must be greater than 1")
    if len(words) <= limit:
        return [words]
    return [words[i : i + limit] for i in range(0, len(words), limit)]


def _ascii(word: str) -> bool:
    return len(word[0].encode()) == 1


def punctuate(model: CTTransformer, vocabulary: Vocabulary, text: str, split_size: int = 20) -> tuple[str, list[int]]:
    """``CTTransformer.inference`` for one text: the punctuated text and each word's class (``punc_array``; ``1`` none,
    ``2`` ``，``, ``3`` ``。``, ``4`` ``？``, ``5`` ``、``). Empty or blank text gives ``("", [])`` (upstream's
    ``punc_array`` is None then). The model runs on its own device and dtype; the caller enters inference mode."""
    if not text or not text.strip():
        return "", []
    punc_list, end_id = model.punc_list, model.sentence_end_id
    device = model.decoder.weight.device
    words = split_words(text)
    mini_words = split_to_mini_sentences(words, split_size)
    mini_ids = split_to_mini_sentences(vocabulary.encode(words), split_size)
    cache_words: list[str] = []
    cache_ids: list[int] = []
    out = ""
    out_punc: list[int] = []
    last = len(mini_words) - 1
    for n, (chunk, chunk_ids) in enumerate(zip(mini_words, mini_ids)):
        sentence = cache_words + chunk
        ids = cache_ids + chunk_ids
        logits = model(torch.tensor([ids], dtype=torch.long, device=device))
        _, indices = logits.view(-1, logits.shape[-1]).topk(1, dim=1)
        punc = [int(p) for p in torch.squeeze(indices, dim=1).tolist()]
        if n < last:
            # Keep the words up to the last predicted stop (not among the first two or the last word), carry the rest.
            end = -1
            last_comma = -1
            for i in range(len(punc) - 2, 1, -1):
                if punc_list[punc[i]] in ("。", "？"):
                    end = i
                    break
                if last_comma < 0 and punc_list[punc[i]] == "，":
                    last_comma = i
            if end < 0 and len(sentence) > CACHE_POP_TRIGGER_LIMIT and last_comma >= 0:
                end = last_comma
                punc[end] = end_id
            cache_words, cache_ids = sentence[end + 1 :], ids[end + 1 :]
            sentence, punc = sentence[: end + 1], punc[: end + 1]
        parts: list[str] = []
        for i in range(len(sentence)):
            if (i == 0 or punc_list[punc[i - 1]] in ("。", "？")) and _ascii(sentence[i]):
                sentence[i] = sentence[i].capitalize()
            if i == 0 and _ascii(sentence[i]):
                sentence[i] = " " + sentence[i]
            if i > 0 and _ascii(sentence[i]) and _ascii(sentence[i - 1]):
                sentence[i] = " " + sentence[i]
            parts.append(sentence[i])
            if punc_list[punc[i]] != "_":
                mark = punc_list[punc[i]]
                if _ascii(sentence[i]):
                    mark = {"，": ",", "。": ".", "？": "?"}.get(mark, mark)
                parts.append(mark)
        out += "".join(parts)
        if n == last:
            # The end of the text gets a full stop.
            tail = out[-1]
            if tail in ("，", "、"):
                out = out[:-1] + "。"
            elif tail == ",":
                out = out[:-1] + "."
            elif tail not in ("。", "？") and len(tail.encode()) != 1:
                out += "。"
            elif tail not in (".", "?") and len(tail.encode()) == 1:
                out += "."
            else:
                tail = ""
            if tail and punc:
                punc[-1] = end_id
        out_punc += punc
    return out, out_punc
