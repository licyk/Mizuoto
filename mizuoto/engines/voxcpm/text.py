"""VoxCPM's text: the tokenizer and the optional normaliser (upstream ``model/utils.py``, ``utils/text_normalize.py``).

**Tokenizer.** Upstream loads ``tokenizer.json`` with transformers' ``LlamaTokenizerFast``, which in transformers 5.3
(upstream's lock) is ``LlamaTokenizer``. That class does not run the file's normaliser. It builds a tokenizer of its
own from the file's vocabulary and merges:
- a BPE with byte fallback and no unknown token;
- no normaliser, and a ``Metaspace`` pre-tokenizer that prepends ``▁`` (``always`` while ``legacy`` is true, as the
  models' configs say), unsplit;
- the special tokens of ``tokenizer_config.json``'s ``added_tokens_decoder``, added in id order. A content the
  vocabulary lacks takes the next free id, so VoxCPM2's ``<|im_end|>`` is 73442, not the file's 73440.

This module builds the same tokenizer with ``tokenizers``. Upstream's wrapper (``mask_multichar_chinese_tokens``)
then turns every token that is two or more Han characters (``▁`` removed) into its single characters. Ids come from
``convert_tokens_to_ids``, so a character outside the vocabulary is ``<unk>`` (0). No BOS is added.

**Normaliser** (``normalize=True``). Markdown and emoji are stripped and quotes straightened. Chinese text (any
character in U+4E00–U+9FFF) then runs wetext 0.1.2's ``zh`` TN with ``remove_erhua``, followed by upstream's
blank, symbol and bracket rules. Other text runs wetext's ``en`` TN, then ``inflect`` spells the remaining numbers.
wetext's grammars ship in ``assets/wetext``.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

WETEXT_DIR = Path(__file__).with_name("assets") / "wetext"
HAN = re.compile(r"[一-鿿]+")


class VoxCpmTokenizer:
    def __init__(self, model_dir: Path) -> None:
        from tokenizers import AddedToken, Tokenizer, decoders, pre_tokenizers
        from tokenizers.models import BPE

        model_dir = Path(model_dir)
        data = json.loads((model_dir / "tokenizer.json").read_text(encoding="utf-8"))
        config_path = model_dir / "tokenizer_config.json"
        config: dict[str, Any] = json.loads(config_path.read_text(encoding="utf-8")) if config_path.is_file() else {}
        model = data["model"]
        merges = [tuple(m.split(" ", 1)) if isinstance(m, str) else tuple(m) for m in model["merges"]]
        tokenizer = Tokenizer(BPE(vocab=model["vocab"], merges=merges, fuse_unk=True, byte_fallback=True, dropout=None))
        tokenizer.normalizer = None
        add_prefix_space = config.get("add_prefix_space")
        add_prefix_space = True if add_prefix_space is None else bool(add_prefix_space)
        scheme = ("always" if config.get("legacy", False) else "first") if add_prefix_space else "never"
        tokenizer.pre_tokenizer = pre_tokenizers.Metaspace(replacement="▁", prepend_scheme=scheme, split=False)
        tokenizer.decoder = decoders.Sequence([decoders.Replace("▁", " "), decoders.ByteFallback(), decoders.Fuse()])
        decoder_entries = config.get("added_tokens_decoder") or {}
        added = [decoder_entries[k] for k in sorted(decoder_entries, key=int)]
        tokenizer.add_special_tokens(
            [
                AddedToken(
                    t["content"],
                    single_word=bool(t.get("single_word", False)),
                    lstrip=bool(t.get("lstrip", False)),
                    rstrip=bool(t.get("rstrip", False)),
                    normalized=bool(t.get("normalized", False)),
                    special=True,
                )
                for t in added
            ]
        )
        self.tokenizer = tokenizer
        unk = config.get("unk_token", "<unk>")
        self.unk_id = tokenizer.token_to_id(unk if isinstance(unk, str) else unk.get("content", "<unk>")) or 0
        self.multichar = {token for token in tokenizer.get_vocab(with_added_tokens=True) if len(token) >= 2 and all("一" <= c <= "鿿" for c in token)}

    def tokenize(self, text: str) -> list[str]:
        out: list[str] = []
        for token in self.tokenizer.encode(text, add_special_tokens=False).tokens:
            clean = token.replace("▁", "")
            if clean in self.multichar:
                out.extend(clean)
            else:
                out.append(token)
        return out

    def __call__(self, text: str) -> list[int]:
        ids = []
        for token in self.tokenize(text):
            index = self.tokenizer.token_to_id(token)
            ids.append(self.unk_id if index is None else index)
        return ids


# -- normalisation ----------------------------------------------------------------------


def contains_chinese(text: str) -> bool:
    return bool(HAN.search(text))


def replace_corner_mark(text: str) -> str:
    return text.replace("²", "平方").replace("³", "立方").replace("√", "根号").replace("≈", "约等于").replace("<", "小于")


def remove_bracket(text: str) -> str:
    text = text.replace("（", " ").replace("）", " ")
    text = text.replace("【", " ").replace("】", " ")
    text = text.replace("`", "")
    return text.replace("——", " ")


def spell_out_number(text: str, inflect_parser: Any) -> str:
    out = []
    start = None
    for i, c in enumerate(text):
        if not c.isdigit():
            if start is not None:
                out.append(inflect_parser.number_to_words(text[start:i]))
                start = None
            out.append(c)
        elif start is None:
            start = i
    if start is not None and start < len(text):
        out.append(inflect_parser.number_to_words(text[start:]))
    return "".join(out)


def replace_blank(text: str) -> str:
    """Drop blanks between non-ASCII characters (upstream indexes ``text[i - 1]``, which wraps at 0, as here)."""
    out = []
    for i, c in enumerate(text):
        if c == " ":
            if (text[i + 1].isascii() and text[i + 1] != " ") and (text[i - 1].isascii() and text[i - 1] != " "):
                out.append(c)
        else:
            out.append(c)
    return "".join(out)


def clean_markdown(md_text: str) -> str:
    md_text = re.sub(r"```.*?```", "", md_text, flags=re.DOTALL)
    md_text = re.sub(r"`[^`]*`", "", md_text)
    md_text = re.sub(r"!\[[^\]]*\]\([^\)]+\)", "", md_text)
    md_text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", md_text)
    md_text = re.sub(r"^(\s*)-\s+", r"\1", md_text, flags=re.MULTILINE)
    md_text = re.sub(r"<[^>]+>", "", md_text)
    md_text = re.sub(r"^#{1,6}\s*", "", md_text, flags=re.MULTILINE)
    md_text = re.sub(r"\n\s*\n", "\n", md_text)
    return md_text.strip()


def clean_text(text: str) -> str:
    import regex

    text = clean_markdown(text)
    text = regex.compile(r"\p{Emoji_Presentation}|\p{Emoji}️", flags=regex.UNICODE).sub("", text)
    text = text.replace("\n", " ").replace("\t", " ")
    return text.replace("“", '"').replace("”", '"')


class TextNormalizer:
    """``voxcpm.utils.text_normalize.TextNormalizer.normalize(text)`` (``split=False``)."""

    def __init__(self, fst_dir: Path = WETEXT_DIR) -> None:
        import inflect

        from mizuoto.text.normalize.wetext import WeTextNormalizer

        self.zh = WeTextNormalizer(fst_dir, lang="zh", operator="tn", remove_erhua=True, version="0.1")
        self.en = WeTextNormalizer(fst_dir, lang="en", operator="tn", version="0.1")
        self.inflect = inflect.engine()

    def normalize(self, text: str) -> str:
        lang = "zh" if contains_chinese(text) else "en"
        text = clean_text(text)
        if lang == "zh":
            text = text.replace("=", "等于")
            if re.search(r"([\d$%^*_+≥≤≠×÷?=])", text):
                text = re.sub(r"(?<=[a-zA-Z0-9])-(?=\d)", " - ", text)
            text = self.zh.normalize(text)
            text = replace_blank(text)
            text = replace_corner_mark(text)
            return remove_bracket(text)
        return spell_out_number(self.en.normalize(text), self.inflect)
