"""OmniVoice's text handling (upstream ``omnivoice/utils/text.py`` and ``models/omnivoice.py``, Apache-2.0, Xiaomi Corp.).

- ``chunk_text_punctuation``: long text cut at sentence punctuation (not after common abbreviations),
  merged up to a length, short pieces folded into their neighbours.
- ``add_punctuation``: a full stop when the text ends without one (``。`` for Chinese).
- ``combine_text``: the reference transcript and the text as one prompt, with upstream's whitespace rules.
- ``tokenize``: the 13 non-verbal tags (``[laughter]``…) encoded on their own, as upstream does.

Text normalisation (``normalize_text``, WeTextProcessing) is not used by Mizuoto's requests and is not ported.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mizuoto.text.tokenize.bpe import BpeTokenizer

SPLIT_PUNCTUATION = set(".,;:!?。，；：！？")
CLOSING_MARKS = set("\"'“”‘’）]》>」】")
END_PUNCTUATION = {";", ":", ",", ".", "!", "?", "…", ")", "]", "}", '"', "'", "“", "”", "‘", "’", "；", "：", "，", "。", "！", "？", "、", "……", "）", "】"}
ABBREVIATIONS = {
    *("Mr.", "Mrs.", "Ms.", "Dr.", "Prof.", "Sr.", "Jr.", "Rev.", "Fr.", "Hon.", "Pres.", "Gov.", "Capt.", "Gen.", "Sen.", "Rep."),
    *("Col.", "Maj.", "Lt.", "Cmdr.", "Sgt.", "Cpl.", "Co.", "Corp.", "Inc.", "Ltd.", "Est.", "Dept.", "St.", "Ave.", "Blvd.", "Rd."),
    *("Mt.", "Ft.", "No.", "Jan.", "Feb.", "Mar.", "Apr.", "Aug.", "Sep.", "Sept.", "Oct.", "Nov.", "Dec.", "i.e.", "e.g.", "vs."),
    *("Vs.", "Etc.", "approx.", "fig.", "def."),
}
NONVERBAL_PATTERN = re.compile(
    r"\[(laughter|sigh|confirmation-en|question-en|question-ah|question-oh|"
    r"question-ei|question-yi|surprise-ah|surprise-oh|surprise-wa|"
    r"surprise-yo|dissatisfaction-hnn)\]"
)


def chunk_text_punctuation(text: str, chunk_len: int, min_chunk_len: int | None = None) -> list[str]:
    sentences: list[list[str]] = []
    current: list[str] = []
    for token in text:
        if not current and sentences and (token in SPLIT_PUNCTUATION or token in CLOSING_MARKS):
            sentences[-1].append(token)
            continue
        current.append(token)
        if token in SPLIT_PUNCTUATION:
            abbreviation = False
            if token == ".":
                so_far = "".join(current).strip()
                if so_far and so_far.split()[-1] in ABBREVIATIONS:
                    abbreviation = True
            if not abbreviation:
                sentences.append(current)
                current = []
    if current:
        sentences.append(current)

    merged: list[list[str]] = []
    chunk: list[str] = []
    for sentence in sentences:
        if len(chunk) + len(sentence) <= chunk_len:
            chunk.extend(sentence)
        else:
            if chunk:
                merged.append(chunk)
            chunk = sentence
    if chunk:
        merged.append(chunk)

    if min_chunk_len is not None:
        first_short = bool(merged) and len(merged[0]) < min_chunk_len
        final: list[list[str]] = []
        for i, piece in enumerate(merged):
            if i == 1 and first_short:
                final[-1].extend(piece)
            elif len(piece) >= min_chunk_len or not final:
                final.append(piece)
            else:
                final[-1].extend(piece)
    else:
        final = merged
    return ["".join(c).strip() for c in final if "".join(c).strip()]


def add_punctuation(text: str) -> str:
    text = text.strip()
    if text and text[-1] not in END_PUNCTUATION:
        text += "。" if any("一" <= c <= "鿿" for c in text) else "."
    return text


def combine_text(text: str, ref_text: str | None = None) -> str:
    full = ref_text.strip() + " " + text.strip() if ref_text else text.strip()
    full = re.sub(r"[\r\n]+", "", full)
    full = full.replace("（", "(").replace("）", ")")
    full = re.sub(r"[ \t]+", " ", full)
    return re.sub(r"(?<=[一-鿿])\s+|\s+(?=[一-鿿])", "", full)


def tokenize(text: str, tokenizer: BpeTokenizer) -> list[int]:
    """``_tokenize_with_nonverbal_tags``: the text between tags and each tag encoded separately."""
    ids: list[int] = []
    last = 0
    for m in NONVERBAL_PATTERN.finditer(text):
        if m.start() > last:
            ids += tokenizer.encode(text[last : m.start()])
        ids += tokenizer.encode(m.group())
        last = m.end()
    if last < len(text):
        ids += tokenizer.encode(text[last:])
    return ids or tokenizer.encode(text)
