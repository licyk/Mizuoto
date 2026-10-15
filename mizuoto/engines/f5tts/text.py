"""F5-TTS's text: characters and pinyin (upstream ``f5_tts.model.utils``, MIT), and its chunking."""

from __future__ import annotations

import re
from pathlib import Path


def read_vocab(path: Path) -> dict[str, int]:
    """``get_tokenizer(path, "custom")``: one token per line, read in text mode, each line's last character dropped."""
    vocab: dict[str, int] = {}
    with Path(path).open(encoding="utf-8") as f:
        for i, line in enumerate(f):
            vocab[line[:-1]] = i
    return vocab


def _is_chinese(c: str) -> bool:
    return "㄀" <= c <= "鿿"


def convert_char_to_pinyin(text_list: list[str], polyphone: bool = True) -> list[list[str]]:
    import rjieba
    from pypinyin import Style, lazy_pinyin

    out = []
    custom_trans = str.maketrans({";": ",", "“": '"', "”": '"', "‘": "'", "’": "'"})
    for text in text_list:
        char_list: list[str] = []
        text = text.translate(custom_trans)
        for seg in rjieba.cut(text):
            seg_byte_len = len(bytes(seg, "UTF-8"))
            if seg_byte_len == len(seg):  # letters and symbols only
                if char_list and seg_byte_len > 1 and char_list[-1] not in " :'\"":
                    char_list.append(" ")
                char_list.extend(seg)
            elif polyphone and seg_byte_len == 3 * len(seg):  # East Asian characters only
                pinyin = lazy_pinyin(seg, style=Style.TONE3, tone_sandhi=True)
                for i, c in enumerate(seg):
                    if _is_chinese(c):
                        char_list.append(" ")
                    char_list.append(pinyin[i])
            else:  # a mix
                for c in seg:
                    if ord(c) < 256:
                        char_list.extend(c)
                    elif _is_chinese(c):
                        char_list.append(" ")
                        char_list.extend(lazy_pinyin(c, style=Style.TONE3, tone_sandhi=True))
                    else:
                        char_list.append(c)
        out.append(char_list)
    return out


def chunk_text(text: str, max_chars: int = 135) -> list[str]:
    """Upstream ``infer.utils_infer.chunk_text``: sentences packed into chunks of at most ``max_chars`` UTF-8 bytes."""
    chunks = []
    current = ""
    for sentence in re.split(r"(?<=[;:,.!?])\s+|(?<=[；：，。！？])", text):
        if not sentence:
            continue
        if len(current.encode("utf-8")) + len(sentence.encode("utf-8")) <= max_chars:
            current += sentence + " " if sentence and len(sentence[-1].encode("utf-8")) == 1 else sentence
        else:
            if current:
                chunks.append(current.strip())
            current = sentence + " " if sentence and len(sentence[-1].encode("utf-8")) == 1 else sentence
    if current:
        chunks.append(current.strip())
    return chunks


def finish_ref_text(text: str) -> str:
    """Upstream ends the reference transcript with ". " (or "。") so the model hears a full stop."""
    text = text.strip()
    if text.endswith(". ") or text.endswith("。"):
        return text
    return text + " " if text.endswith(".") else text + ". "
