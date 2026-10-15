"""G2PW: Mandarin polyphone disambiguation for GPT-SoVITS (upstream ``text/g2pw``; MIT, (c) 2024 RVC-Boss; from PaddleSpeech's
port, Apache-2.0, of GitYCC's g2pW).

Upstream runs ``g2pW.onnx`` (a BERT-base and a conditional softmax head) with onnxruntime and tokenises with
transformers' ``AutoTokenizer``. Here the network is a torch module whose weights come from the ONNX file
(``mizuoto.runtime.convert.onnx``) and the tokeniser is the ``tokenizers`` library's, read from chinese-roberta's
``tokenizer.json`` (the vocabulary G2PW was trained with: 21,128 entries, as bert-base-chinese). Everything around the
network (which characters are polyphonic, the 16-character context, one network call per distinct context, the
traditional-script conversion of the input, bopomofo → pinyin) is upstream's ``onnx_api.py`` and ``dataset.py``.

The model files are the ``gptsovits/g2pw`` asset (``G2PWModel/``): ``g2pW.onnx``, ``POLYPHONIC_CHARS.txt``,
``MONOPHONIC_CHARS.txt``, ``bopomofo_to_pinyin_wo_tune_dict.json`` and ``config.py`` (read as data, not executed).
Upstream's corrections dictionary (``polyphonic.rep`` + ``polyphonic-fix.rep``) ships as ``data/polyphonic.json``.
"""

from __future__ import annotations

import ast
import json
import re
from dataclasses import dataclass
from functools import cache
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    import torch

    from mizuoto.runtime.convert.onnx import OnnxGraph

NON_POLYPHONIC = {"一", "不", "和", "咋", "嗲", "剖", "差", "攢", "倒", "難", "奔", "勁", "拗", "肖", "瘙", "誒", "泊", "听", "噢"}
NON_MONOPHONIC = {"似", "攢"}

DEFAULT_CONFIG: dict[str, Any] = {"use_mask": True, "use_char_phoneme": False, "use_conditional": True}
"""``utils.default_config_dict``'s entries that inference reads."""


def read_config(path: Path) -> dict[str, Any]:
    """``config.py``'s literal assignments (upstream imports it as a module), over upstream's defaults."""
    config = dict(DEFAULT_CONFIG)
    for node in ast.parse(path.read_text(encoding="utf-8")).body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            try:
                config[node.targets[0].id] = ast.literal_eval(node.value)
            except ValueError:
                continue
    return config


@cache
def correction_dict() -> dict[str, list[str]]:
    """``g2pw.get_dict()``: words and characters whose pinyin upstream fixes after G2PW (``polyphonic*.rep``)."""
    from mizuoto.engines.gptsovits.text._data import read_text

    return json.loads(read_text("polyphonic.json"))


def correct_pronunciation(word: str, word_pinyins: list[Any]) -> list[Any]:
    pp_dict = correction_dict()
    new_pinyins = pp_dict.get(word, "")
    if new_pinyins == "":
        for idx, w in enumerate(word):
            w_pinyin = pp_dict.get(w, "")
            if w_pinyin != "":
                word_pinyins[idx] = w_pinyin[0]
        return word_pinyins
    return list(new_pinyins)


# -- tokenisation (dataset.py, utils.py) -------------------------------------------------------------------------------


def wordize_and_map(text: str) -> tuple[list[str], list[int | None], list[tuple[int, int]]]:
    words: list[str] = []
    index_map_from_text_to_word: list[int | None] = []
    index_map_from_word_to_text: list[tuple[int, int]] = []
    while len(text) > 0:
        match_space = re.match(r"^ +", text)
        if match_space:
            space_str = match_space.group(0)
            index_map_from_text_to_word += [None] * len(space_str)
            text = text[len(space_str) :]
            continue
        match_en = re.match(r"^[a-zA-Z0-9]+", text)
        if match_en:
            en_word = match_en.group(0)
            word_start_pos = len(index_map_from_text_to_word)
            index_map_from_word_to_text.append((word_start_pos, word_start_pos + len(en_word)))
            index_map_from_text_to_word += [len(words)] * len(en_word)
            words.append(en_word)
            text = text[len(en_word) :]
        else:
            word_start_pos = len(index_map_from_text_to_word)
            index_map_from_word_to_text.append((word_start_pos, word_start_pos + 1))
            index_map_from_text_to_word += [len(words)]
            words.append(text[0])
            text = text[1:]
    return words, index_map_from_text_to_word, index_map_from_word_to_text


def tokenize_and_map(tokenizer: Any, text: str) -> tuple[list[str], list[int | None], list[tuple[int, int]]]:
    """``tokenizer`` is a ``tokenizers.Tokenizer`` (``tokenize`` = its tokens without special tokens)."""
    words, text2word, word2text = wordize_and_map(text=text)
    tokens: list[str] = []
    index_map_from_token_to_text: list[tuple[int, int]] = []
    for word, (word_start, word_end) in zip(words, word2text, strict=True):
        word_tokens = tokenizer.encode(word, add_special_tokens=False).tokens
        if len(word_tokens) == 0 or word_tokens == ["[UNK]"]:
            index_map_from_token_to_text.append((word_start, word_end))
            tokens.append("[UNK]")
        else:
            current_word_start = word_start
            for word_token in word_tokens:
                word_token_len = len(re.sub(r"^##", "", word_token))
                index_map_from_token_to_text.append((current_word_start, current_word_start + word_token_len))
                current_word_start = current_word_start + word_token_len
                tokens.append(word_token)
    index_map_from_text_to_token = text2word
    for i, (token_start, token_end) in enumerate(index_map_from_token_to_text):
        for token_pos in range(token_start, token_end):
            index_map_from_text_to_token[token_pos] = i
    return tokens, index_map_from_text_to_token, index_map_from_token_to_text


def _truncate(
    max_len: int, text: str, query_id: int, tokens: list[str], text2token: list[int | None], token2text: list[tuple[int, int]]
) -> tuple[str, int, list[str], list[int | None], list[tuple[int, int]]]:
    truncate_len = max_len - 2
    if len(tokens) <= truncate_len:
        return (text, query_id, tokens, text2token, token2text)
    token_position = text2token[query_id]
    assert token_position is not None
    token_start = token_position - truncate_len // 2
    token_end = token_start + truncate_len
    font_exceed_dist = -token_start
    back_exceed_dist = token_end - len(tokens)
    if font_exceed_dist > 0:
        token_start += font_exceed_dist
        token_end += font_exceed_dist
    elif back_exceed_dist > 0:
        token_start -= back_exceed_dist
        token_end -= back_exceed_dist
    start = token2text[token_start][0]
    end = token2text[token_end - 1][1]
    return (
        text[start:end],
        query_id - start,
        tokens[token_start:token_end],
        [i - token_start if i is not None else None for i in text2token[start:end]],
        [(s - start, e - start) for s, e in token2text[token_start:token_end]],
    )


@dataclass
class _Tokenized:
    tokens: list[str]
    text2token: list[int | None]
    token2text: list[tuple[int, int]]
    input_id: list[int] | None = None
    """``[CLS]`` + tokens + ``[SEP]`` as ids, when they fit (else each query is truncated around itself)."""


def _token_ids(tokenizer: Any, tokens: list[str], unk_id: int) -> list[int]:
    ids = []
    for token in tokens:
        token_id = tokenizer.token_to_id(token)
        ids.append(unk_id if token_id is None else token_id)
    return ids


def prepare_input(
    tokenizer: Any,
    texts: list[str],
    query_ids: list[int],
    char2id: dict[str, int],
    char_phoneme_masks: dict[str, list[int]] | None,
    num_labels: int,
    max_len: int = 512,
) -> dict[str, np.ndarray]:
    """``dataset.prepare_onnx_input`` (``window_size=None``): the network's inputs, padded to the longest text."""
    input_ids: list[list[int]] = []
    token_type_ids: list[list[int]] = []
    attention_masks: list[list[int]] = []
    phoneme_masks: list[list[int]] = []
    char_ids: list[int] = []
    position_ids: list[int] = []
    tokenized_cache: dict[str, _Tokenized] = {}
    unk_id = tokenizer.token_to_id("[UNK]")
    full_phoneme_mask = [1] * num_labels
    for idx in range(len(texts)):
        text = texts[idx].lower()
        query_id = query_ids[idx]
        cached = tokenized_cache.get(text)
        if cached is None:
            try:
                tokens, text2token, token2text = tokenize_and_map(tokenizer=tokenizer, text=text)
            except Exception:
                return {}  # upstream: 'warning: text "…" is invalid'
            cached = _Tokenized(tokens, text2token, token2text)
            if len(tokens) <= max_len - 2:
                cached.input_id = _token_ids(tokenizer, ["[CLS]", *tokens, "[SEP]"], unk_id)
            tokenized_cache[text] = cached
        if cached.input_id is not None:
            text_for_query, query_id_for_query, text2token_for_query, input_id = text, query_id, cached.text2token, cached.input_id
        else:
            text_for_query, query_id_for_query, tokens_for_query, text2token_for_query, _ = _truncate(max_len, text, query_id, cached.tokens, cached.text2token, cached.token2text)
            input_id = _token_ids(tokenizer, ["[CLS]", *tokens_for_query, "[SEP]"], unk_id)
        token_type_id = [0] * len(input_id)
        attention_mask = [1] * len(input_id)
        query_char = text_for_query[query_id_for_query]
        phoneme_mask = char_phoneme_masks[query_char] if char_phoneme_masks is not None else full_phoneme_mask
        position = text2token_for_query[query_id_for_query]
        assert position is not None
        input_ids.append(input_id)
        token_type_ids.append(token_type_id)
        attention_masks.append(attention_mask)
        phoneme_masks.append(phoneme_mask)
        char_ids.append(char2id[query_char])
        position_ids.append(position + 1)  # [CLS] comes first
    max_token_length = max(len(seq) for seq in input_ids)

    def pad(sequences: list[list[int]]) -> np.ndarray:
        return np.array([seq + [0] * (max_token_length - len(seq)) for seq in sequences]).astype(np.int64)

    return {
        "input_ids": pad(input_ids),
        "token_type_ids": pad(token_type_ids),
        "attention_masks": pad(attention_masks),
        "phoneme_masks": np.array(phoneme_masks).astype(np.float32),
        "char_ids": np.array(char_ids).astype(np.int64),
        "position_ids": np.array(position_ids).astype(np.int64),
    }


def get_phoneme_labels(polyphonic_chars: list[list[str]]) -> tuple[list[str], dict[str, list[int]]]:
    labels = sorted({phoneme for _char, phoneme in polyphonic_chars})
    index = {label: i for i, label in enumerate(labels)}
    char2phonemes: dict[str, list[int]] = {}
    for char, phoneme in polyphonic_chars:
        char2phonemes.setdefault(char, []).append(index[phoneme])
    return labels, char2phonemes


def get_char_phoneme_labels(polyphonic_chars: list[list[str]]) -> tuple[list[str], dict[str, list[int]]]:
    labels = sorted({f"{char} {phoneme}" for char, phoneme in polyphonic_chars})
    index = {label: i for i, label in enumerate(labels)}
    char2phonemes: dict[str, list[int]] = {}
    for char, phoneme in polyphonic_chars:
        char2phonemes.setdefault(char, []).append(index[f"{char} {phoneme}"])
    return labels, char2phonemes


# -- the network -------------------------------------------------------------------------------------------------------


def _build_model(num_labels: int, num_chars: int, num_pos: int, config: Any) -> Any:
    import torch
    from torch import nn

    from mizuoto.nn.transformer.bert import BertModel
    from mizuoto.runtime import ops

    class G2PWNetwork(nn.Module):
        """``g2pW.onnx``: BERT, then a POS classifier and a phoneme classifier whose softmax is weighted by the character's
        candidates (``phoneme_mask``) times sigmoid(bias + char descriptor + char×POS descriptor)."""

        def __init__(self) -> None:
            super().__init__()
            hidden = config.hidden_size
            self.num_pos = num_pos
            self.bert = BertModel(config, attention="eager")
            self.classifier = ops.default.Linear(hidden, num_labels)
            self.pos_classifier = ops.default.Linear(hidden, num_pos)
            self.descriptor_bias = ops.default.Embedding(1, num_labels)
            self.char_descriptor = ops.default.Embedding(num_chars, num_labels)
            self.second_order_descriptor = ops.default.Embedding(num_chars * num_pos, num_labels)

        def forward(
            self,
            input_ids: torch.Tensor,
            token_type_ids: torch.Tensor,
            attention_mask: torch.Tensor,
            phoneme_mask: torch.Tensor,
            char_ids: torch.Tensor,
            position_ids: torch.Tensor,
        ) -> torch.Tensor:
            hidden = self.bert(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
            batch, length, width = hidden.shape
            # the graph's gather: row b·L + position (b over the token batch, broadcast against the queries)
            index = position_ids + torch.arange(batch, device=hidden.device) * length
            selected = hidden.reshape(batch * length, width)[index]
            pos = self.pos_classifier(selected).argmax(dim=-1)
            weights = self.descriptor_bias.weight[0] + self.char_descriptor(char_ids) + self.second_order_descriptor(char_ids * self.num_pos + pos)
            weights = torch.sigmoid(weights) * phoneme_mask
            logits = self.classifier(selected)
            exp = torch.exp(logits - logits.max(dim=-1, keepdim=True).values) * weights
            return torch.clamp(exp / exp.sum(dim=-1, keepdim=True), 1e-6, 1 - 1e-6)

    return G2PWNetwork()


def state_dict_from_onnx(graph: OnnxGraph) -> dict[str, torch.Tensor]:
    """``g2pW.onnx``'s weights under the network's names. Each anonymous ``MatMul`` weight takes the name of the bias its
    output is added to (``….query.bias`` → ``….query.weight``), transposed to ``Linear``'s layout."""
    import torch

    named = {name for name in graph.initializers if not name[0].isdigit()}
    sd = {name: torch.from_numpy(graph.tensor(name)) for name in named if name != "bert.embeddings.position_ids"}
    consumers: dict[str, list[Any]] = {}
    for node in graph.nodes:
        for name in node.inputs:
            consumers.setdefault(name, []).append(node)
    for node in graph.nodes:
        if node.op_type != "MatMul" or node.inputs[1] not in graph.initializers:
            continue
        bias = next((i for c in consumers.get(node.outputs[0], []) if c.op_type == "Add" for i in c.inputs if i in named and i.endswith(".bias")), None)
        if bias is None:
            raise ValueError(f"g2pW.onnx: no bias follows {node.name}")
        sd[bias[: -len("bias")] + "weight"] = torch.from_numpy(graph.tensor(node.inputs[1])).t().contiguous()
    return sd


# -- the converter (onnx_api.py) ---------------------------------------------------------------------------------------


class G2PW:
    """``G2PWOnnxConverter(style="pinyin", enable_non_tradional_chinese=True)``: ``g2pw(sentences)`` gives, per sentence, a
    pinyin with tone number (``"zhong4"``) or ``None`` per character.

    ``context_chars`` and ``sentence_dedup`` are upstream's environment variables ``g2pw_polyphonic_context_chars`` (16) and
    ``g2pw_sentence_dedup`` (on). ``s2tw`` converts the input to traditional script as upstream's ``OpenCC("s2tw")`` does
    (``opencc`` 1.4.2: ``opencc-python-reimplemented`` converts differently).
    """

    def __init__(
        self,
        model_dir: str | Path,
        tokenizer_file: str | Path,
        *,
        device: str = "cpu",
        context_chars: int = 16,
        sentence_dedup: bool = True,
        s2tw: bool = True,
    ) -> None:
        import torch
        from tokenizers import Tokenizer

        from mizuoto.nn.transformer.bert import BertConfig
        from mizuoto.runtime.convert.onnx import read_onnx
        from mizuoto.runtime.loading import load_into

        model_dir = Path(model_dir)
        self.config = read_config(model_dir / "config.py")
        self.tokenizer = Tokenizer.from_file(str(tokenizer_file))
        self.polyphonic_chars = [line.split("\t") for line in (model_dir / "POLYPHONIC_CHARS.txt").read_text(encoding="utf-8").strip().split("\n")]
        self.monophonic_chars = [line.split("\t") for line in (model_dir / "MONOPHONIC_CHARS.txt").read_text(encoding="utf-8").strip().split("\n")]
        self.labels, self.char2phonemes = (get_char_phoneme_labels if self.config["use_char_phoneme"] else get_phoneme_labels)(self.polyphonic_chars)
        self.chars = sorted(self.char2phonemes.keys())
        self.char2id = {char: idx for idx, char in enumerate(self.chars)}
        self.char_phoneme_masks = (
            {char: [1 if i in self.char2phonemes[char] else 0 for i in range(len(self.labels))] for char in self.char2phonemes} if self.config["use_mask"] else None
        )
        self.polyphonic_chars_new = set(self.chars) - NON_POLYPHONIC
        self.monophonic_chars_dict = {char: phoneme for char, phoneme in self.monophonic_chars}
        for char in NON_MONOPHONIC:
            self.monophonic_chars_dict.pop(char, None)
        self.bopomofo_convert_dict: dict[str, str] = json.loads((model_dir / "bopomofo_to_pinyin_wo_tune_dict.json").read_text(encoding="utf-8"))
        self.context_chars = max(0, int(context_chars))
        self.sentence_dedup = sentence_dedup
        self.cc: Any = None
        if s2tw:
            from opencc import OpenCC

            self.cc = OpenCC("s2tw")

        onnx_path = model_dir / "g2pW.onnx"
        if not onnx_path.is_file():
            onnx_path = model_dir / "g2pw.onnx"
        graph = read_onnx(onnx_path)
        try:
            sd = state_dict_from_onnx(graph)
        finally:
            graph.close()
        vocab, hidden = sd["bert.embeddings.word_embeddings.weight"].shape
        layers = 1 + max(int(k.split(".")[3]) for k in sd if k.startswith("bert.encoder.layer."))
        intermediate = sd["bert.encoder.layer.0.intermediate.dense.weight"].shape[0]
        bert = BertConfig(
            vocab_size=vocab,
            hidden_size=hidden,
            num_hidden_layers=layers,
            num_attention_heads=hidden // 64,
            intermediate_size=intermediate,
            max_position_embeddings=sd["bert.embeddings.position_embeddings.weight"].shape[0],
            type_vocab_size=sd["bert.embeddings.token_type_embeddings.weight"].shape[0],
        )
        num_pos = sd["pos_classifier.weight"].shape[0]
        self.model = _build_model(len(self.labels), len(self.chars), num_pos, bert)
        load_into(self.model, sd, what="G2PW")
        self.model.to(device).eval()
        self.device = torch.device(device)

    def _convert_bopomofo_to_pinyin(self, bopomofo: str) -> str | None:
        tone = bopomofo[-1]
        assert tone in "12345"
        component = self.bopomofo_convert_dict.get(bopomofo[:-1])
        if component:
            return component + tone
        return None  # upstream prints 'Warning: "…" cannot convert to pinyin'

    def __call__(self, sentences: list[str]) -> list[list[str | None]]:
        if self.cc is not None:
            translated = []
            for sent in sentences:
                converted = self.cc.convert(sent)
                assert len(converted) == len(sent)
                translated.append(converted)
            sentences = translated
        texts, model_query_ids, result_query_ids, sent_ids, partial_results = self._prepare_data(sentences)
        if len(texts) == 0:
            return partial_results
        model_input = prepare_input(self.tokenizer, texts, model_query_ids, self.char2id, self.char_phoneme_masks, len(self.labels))
        if not model_input:
            return partial_results
        preds = self._predict_with_sentence_dedup(model_input, texts) if self.sentence_dedup else self._predict(model_input)[0]
        if self.config["use_char_phoneme"]:
            preds = [pred.split(" ")[1] for pred in preds]
        results = partial_results
        for sent_id, query_id, pred in zip(sent_ids, result_query_ids, preds, strict=True):
            results[sent_id][query_id] = self._convert_bopomofo_to_pinyin(pred)
        return results

    def _prepare_data(self, sentences: list[str]) -> tuple[list[str], list[int], list[int], list[int], list[list[str | None]]]:
        from pypinyin import Style, pinyin

        from mizuoto.engines.gptsovits.text.zh_normalization import traditional_to_simplified

        texts: list[str] = []
        model_query_ids: list[int] = []
        result_query_ids: list[int] = []
        sent_ids: list[int] = []
        partial_results: list[list[str | None]] = []
        for sent_id, sent in enumerate(sentences):
            pypinyin_result = pinyin(traditional_to_simplified(sent), neutral_tone_with_five=True, style=Style.TONE3)
            partial_result: list[str | None] = [None] * len(sent)
            polyphonic_indices: list[int] = []
            for i, char in enumerate(sent):
                if char in self.polyphonic_chars_new:
                    polyphonic_indices.append(i)
                elif char in self.monophonic_chars_dict:
                    partial_result[i] = self._convert_bopomofo_to_pinyin(self.monophonic_chars_dict[char])
                else:
                    partial_result[i] = pypinyin_result[i][0]  # char_bopomofo_dict or not, upstream takes pypinyin's
            if polyphonic_indices:
                if self.context_chars > 0:
                    left = max(0, polyphonic_indices[0] - self.context_chars)
                    right = min(len(sent), polyphonic_indices[-1] + self.context_chars + 1)
                    sent_for_predict, query_offset = sent[left:right], left
                else:
                    sent_for_predict, query_offset = sent, 0
                for index in polyphonic_indices:
                    texts.append(sent_for_predict)
                    model_query_ids.append(index - query_offset)
                    result_query_ids.append(index)
                    sent_ids.append(sent_id)
            partial_results.append(partial_result)
        return texts, model_query_ids, result_query_ids, sent_ids, partial_results

    def probabilities(self, model_input: dict[str, np.ndarray]) -> np.ndarray:
        """The network's output for prepared inputs: (queries, labels) float32, as ``session.run`` gives it."""
        import torch

        names = ("input_ids", "token_type_ids", "attention_masks", "phoneme_masks", "char_ids", "position_ids")
        tensors = [torch.from_numpy(model_input[name]).to(self.device) for name in names]
        with torch.inference_mode():
            probs = self.model(*tensors)
        return probs.float().cpu().numpy()

    def _predict(self, model_input: dict[str, np.ndarray]) -> tuple[list[str], list[float]]:
        probs = self.probabilities(model_input)
        preds = np.argmax(probs, axis=1).tolist()
        confidences = [row[index] for index, row in zip(preds, probs.tolist(), strict=True)]
        return [self.labels[pred] for pred in preds], confidences

    def _predict_with_sentence_dedup(self, model_input: dict[str, np.ndarray], texts: list[str]) -> list[str]:
        """One network call per distinct context text (its tokens once, its queries together), as upstream."""
        if len(texts) <= 1:
            return self._predict(model_input)[0]
        grouped: dict[str, list[int]] = {}
        for idx, text in enumerate(texts):
            grouped.setdefault(text, []).append(idx)
        if all(len(indices) == 1 for indices in grouped.values()):
            return self._predict(model_input)[0]
        preds = [""] * len(texts)
        for indices in grouped.values():
            group_input = {name: value[indices] for name, value in model_input.items()}
            if len(indices) > 1:
                for name in ("input_ids", "token_type_ids", "attention_masks"):
                    group_input[name] = group_input[name][:1]
            group_preds, _ = self._predict(group_input)
            for output_idx, pred in zip(indices, group_preds, strict=True):
                preds[output_idx] = pred
        return preds
