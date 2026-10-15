"""CosyVoice's text frontend (upstream ``cli/frontend.py`` and ``utils/frontend_utils.py``, Apache-2.0, Alibaba).

``text_normalize`` as upstream runs it with wetext (the ``ttsfrd`` frontend is not used): text holding
``<|…|>`` markup is left alone; Chinese is normalised by wetext's zh grammar, cleaned (blanks between
Chinese characters, superscripts, brackets) and split at sentence punctuation; other text is normalised by
the en grammar, its digits spelt out with ``inflect`` and split by token count. Pieces that are only
punctuation are dropped.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from pathlib import Path
from typing import Any

CHINESE = re.compile(r"[一-鿿]+")


def contains_chinese(text: str) -> bool:
    return bool(CHINESE.search(text))


def replace_corner_mark(text: str) -> str:
    return text.replace("²", "平方").replace("³", "立方")


def remove_bracket(text: str) -> str:
    text = text.replace("（", "").replace("）", "").replace("【", "").replace("】", "")
    return text.replace("`", "").replace("`", "").replace("——", " ")


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


def split_paragraph(
    text: str, tokenize: Callable[[str], list[int]], lang: str = "zh", token_max_n: int = 80, token_min_n: int = 60, merge_len: int = 20, comma_split: bool = False
) -> list[str]:
    def length(t: str) -> int:
        return len(t) if lang == "zh" else len(tokenize(t))

    punct = ["。", "？", "！", "；", "：", "、", ".", "?", "!", ";"] if lang == "zh" else [".", "?", "!", ";", ":"]
    if comma_split:
        punct.extend(["，", ","])
    if text[-1] not in punct:
        text += "。" if lang == "zh" else "."
    start = 0
    utts: list[str] = []
    for i, c in enumerate(text):
        if c in punct:
            if len(text[start:i]) > 0:
                utts.append(text[start:i] + c)
            if i + 1 < len(text) and text[i + 1] in ['"', "”"]:
                utts.append(utts.pop(-1) + text[i + 1])
                start = i + 2
            else:
                start = i + 1
    final: list[str] = []
    cur = ""
    for utt in utts:
        if length(cur + utt) > token_max_n and length(cur) > token_min_n:
            final.append(cur)
            cur = ""
        cur = cur + utt
    if len(cur) > 0:
        if length(cur) < merge_len and len(final) != 0:
            final[-1] = final[-1] + cur
        else:
            final.append(cur)
    return final


def replace_blank(text: str) -> str:
    out = []
    for i, c in enumerate(text):
        if c == " ":
            if (text[i + 1].isascii() and text[i + 1] != " ") and (text[i - 1].isascii() and text[i - 1] != " "):
                out.append(c)
        else:
            out.append(c)
    return "".join(out)


def is_only_punctuation(text: str) -> bool:
    import regex

    return bool(regex.fullmatch(r"^[\p{P}\p{S}]*$", text))


class TextFrontend:
    """``CosyVoiceFrontEnd.text_normalize`` with the wetext frontend. ``fst_dir`` holds wetext's FSTs."""

    def __init__(self, tokenizer: Any, fst_dir: Path | None) -> None:
        import inflect

        from mizuoto.text.normalize.wetext import WeTextNormalizer

        self.tokenizer = tokenizer
        self.inflect = inflect.engine()
        self.zh = WeTextNormalizer(fst_dir, remove_erhua=False) if fst_dir is not None else None
        self.en = WeTextNormalizer(fst_dir) if fst_dir is not None else None

    def normalize(self, text: str, split: bool = True, text_frontend: bool = True) -> Any:
        """A list of pieces when ``split``, else the normalised text."""
        if "<|" in text and "|>" in text:
            text_frontend = False
        if text_frontend is False or text == "":
            return [text] if split else text
        text = text.strip()
        encode = self.tokenizer.encode
        if contains_chinese(text):
            if self.zh is not None:
                text = self.zh.normalize(text)
            text = text.replace("\n", "")
            text = replace_blank(text)
            text = replace_corner_mark(text)
            text = text.replace(".", "。").replace(" - ", "，")
            text = remove_bracket(text)
            text = re.sub(r"[，,、]+$", "。", text)
            texts = split_paragraph(text, encode, "zh", token_max_n=80, token_min_n=60, merge_len=20, comma_split=False)
        else:
            if self.en is not None:
                text = self.en.normalize(text)
            text = spell_out_number(text, self.inflect)
            texts = split_paragraph(text, encode, "en", token_max_n=80, token_min_n=60, merge_len=20, comma_split=False)
        texts = [t for t in texts if not is_only_punctuation(t)]
        return texts if split else text
