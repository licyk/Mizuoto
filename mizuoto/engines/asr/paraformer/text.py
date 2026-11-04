"""Paraformer's tokens to text: FunASR's ``sentence_postprocess`` and ``abbr_dispose`` without timestamps.

Copied from FunASR 1.4.21's ``funasr/utils/postprocess_utils.py`` (MIT, Copyright (c) Alibaba, Inc. and its affiliates),
the branches a transcript without timestamps takes: special tokens dropped; all Chinese characters (and digits, "@")
joined; all alphabetic tokens joined into words (``@@`` continues a word) with spaces; a mix of both handled token by
token; then runs of single letters separated by spaces ("a b c") made one upper-case abbreviation.
"""

from __future__ import annotations

SPECIAL = ("<s>", "</s>", "<unk>", "<OOV>")


def is_chinese(ch: str) -> bool:
    return "一" <= ch <= "鿿" or "0" <= ch <= "9" or ch == "@"


def _cleaned(word: str | list[str]) -> list[str]:
    out = []
    for item in word:
        cur = item.replace(" ", "")
        for special in ("</s>", "<s>", "<unk>", "<OOV>"):
            cur = cur.replace(special, "")
        out.append(cur)
    return out


def is_all_chinese(word: str | list[str]) -> bool:
    items = _cleaned(word)
    return bool(items) and all(is_chinese(ch) for ch in items)


def is_all_alpha(word: str | list[str]) -> bool:
    items = _cleaned(word)
    if not items:
        return False
    for ch in items:
        if ch.isalpha() is False and ch != "'":
            return False
        if ch.isalpha() is True and is_chinese(ch) is True:
            return False
    return True


def abbr_dispose(words: list[str]) -> list[str]:
    size = len(words)
    abbr_begin: list[int] = []
    abbr_end: list[int] = []
    last_num = -1
    for num in range(size):
        if num <= last_num:
            continue
        letter = len(words[num]) == 1 and words[num].encode("utf-8").isalpha()
        if letter and num + 1 < size and words[num + 1] == " " and num + 2 < size and len(words[num + 2]) == 1 and words[num + 2].encode("utf-8").isalpha():
            abbr_begin.append(num)
            num += 2
            abbr_end.append(num)
            while True:
                num += 1
                if num < size and words[num] == " ":
                    num += 1
                    if num < size and len(words[num]) == 1 and words[num].encode("utf-8").isalpha():
                        abbr_end.pop()
                        abbr_end.append(num)
                        last_num = num
                    else:
                        break
                else:
                    break
    out: list[str] = []
    last_num = -1
    for num in range(size):
        if num <= last_num:
            continue
        if num in abbr_begin:
            abbr = words[num].upper()
            num += 1
            while num < size:
                if num in abbr_end:
                    abbr += words[num].upper()
                    last_num = num
                    break
                if words[num].encode("utf-8").isalpha():
                    abbr += words[num].upper()
                num += 1
            out.append(abbr)
        else:
            out.append(words[num])
    return out


def sentence(tokens: list[str]) -> str:
    """``sentence_postprocess(tokens)[0]``: the segment's text."""
    middle = [word for word in tokens if word not in SPECIAL]
    words: list[str] = []
    item = ""
    if is_all_chinese(middle):
        words = [ch.replace(" ", "") for ch in middle]
    elif is_all_alpha(middle):
        for ch in middle:
            if "@@" in ch:
                item += ch.replace("@@", "")
            else:
                item += ch
                words.append(item)
                words.append(" ")
                item = ""
    else:
        alpha_blank = False
        for ch in middle:
            if is_all_chinese(ch):
                if alpha_blank is True:
                    words.pop()
                words.append(ch)
                alpha_blank = False
            elif "@@" in ch:
                item += ch.replace("@@", "")
                alpha_blank = False
            elif is_all_alpha(ch):
                item += ch
                words.append(item)
                words.append(" ")
                item = ""
                alpha_blank = True
            else:
                words.append(ch)
    return "".join(abbr_dispose(words)).strip()


def join_segments(texts: list[str]) -> str:
    """``_join_vad_texts``: the segments' texts (rich tags removed, stripped, empty ones dropped) joined, without a space
    between two Chinese characters (CJK U+3400–U+9FFF) and with one otherwise: the punctuation model's input."""
    import re

    cleaned = [t for t in (re.sub(r"<\|[^|]*\|>", "", text).strip() for text in texts) if t]
    if not cleaned:
        return ""
    joined = cleaned[0]
    for text in cleaned[1:]:
        separator = "" if "㐀" <= joined[-1] <= "鿿" and "㐀" <= text[0] <= "鿿" else " "
        joined += separator + text
    return joined
