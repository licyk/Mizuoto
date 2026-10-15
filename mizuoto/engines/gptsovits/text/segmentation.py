"""How GPT-SoVITS cuts a text before synthesis (upstream ``TTS_infer_pack/text_segmentation_method.py`` and the text half of
``TextPreprocessor``; MIT, (c) 2024 RVC-Boss).

``pre_seg_text(text, lang, method)`` is upstream's ``TextPreprocessor.pre_seg_text``: a short first sentence gets a
leading full stop, the text is cut by ``method`` (``cut0``…``cut5``), empty and punctuation-only pieces are dropped,
pieces under five characters are joined to their neighbours, each piece ends with a full stop, and pieces over 510
characters are split at punctuation for BERT.
"""

from __future__ import annotations

import re
from collections.abc import Callable

PUNCTUATION = {"!", "?", "…", ",", ".", "-", " "}
"""``text_segmentation_method.punctuation``: what a cut that is only these is dropped for."""

SPLITS = {"，", "。", "？", "！", ",", ".", "?", "!", "~", ":", "：", "—", "…"}

CONSECUTIVE = ["!", "?", "…", ",", ".", "-"]
"""``TextPreprocessor``'s module-level ``punctuation``: runs of these collapse to their first."""

_FIRST = re.compile("[" + "".join(re.escape(sep) for sep in SPLITS) + "]")


class EmptyTextError(ValueError):
    """Upstream's ``filter_text`` refuses a text whose every line is empty (``请输入有效文本``)."""


def split_big_text(text: str, max_len: int = 510) -> list[str]:
    """Cuts a piece longer than BERT's limit at punctuation, greedily up to ``max_len`` characters."""
    segments = re.split("([" + "".join(SPLITS) + "])", text)
    result: list[str] = []
    current = ""
    for segment in segments:
        if len(current + segment) > max_len:
            result.append(current)
            current = segment
        else:
            current += segment
    if current:
        result.append(current)
    return result


def split(todo_text: str) -> list[str]:
    """Sentences ending at a mark of ``SPLITS``; the text gets a full stop if it lacks one."""
    todo_text = todo_text.replace("……", "。").replace("——", "，")
    if todo_text[-1] not in SPLITS:
        todo_text += "。"
    i_split_head = i_split_tail = 0
    len_text = len(todo_text)
    todo_texts: list[str] = []
    while True:
        if i_split_head >= len_text:
            break
        if todo_text[i_split_head] in SPLITS:
            i_split_head += 1
            todo_texts.append(todo_text[i_split_tail:i_split_head])
            i_split_tail = i_split_head
        else:
            i_split_head += 1
    return todo_texts


def cut0(inp: str) -> str:
    """No cut."""
    if not set(inp).issubset(PUNCTUATION):
        return inp
    return "\n"


def cut1(inp: str) -> str:
    """Four sentences a piece."""
    inp = inp.strip("\n")
    inps = split(inp)
    split_idx: list[int | None] = list(range(0, len(inps), 4))
    split_idx[-1] = None
    if len(split_idx) > 1:
        opts = ["".join(inps[split_idx[idx] : split_idx[idx + 1]]) for idx in range(len(split_idx) - 1)]
    else:
        opts = [inp]
    opts = [item for item in opts if not set(item).issubset(PUNCTUATION)]
    return "\n".join(opts)


def cut2(inp: str) -> str:
    """About fifty characters a piece."""
    inp = inp.strip("\n")
    inps = split(inp)
    if len(inps) < 2:
        return inp
    opts: list[str] = []
    summ = 0
    tmp_str = ""
    for item in inps:
        summ += len(item)
        tmp_str += item
        if summ > 50:
            summ = 0
            opts.append(tmp_str)
            tmp_str = ""
    if tmp_str != "":
        opts.append(tmp_str)
    if len(opts) > 1 and len(opts[-1]) < 50:
        opts[-2] = opts[-2] + opts[-1]
        opts = opts[:-1]
    opts = [item for item in opts if not set(item).issubset(PUNCTUATION)]
    return "\n".join(opts)


def cut3(inp: str) -> str:
    """At each Chinese full stop."""
    inp = inp.strip("\n")
    opts = list(inp.strip("。").split("。"))
    opts = [item for item in opts if not set(item).issubset(PUNCTUATION)]
    return "\n".join(opts)


def cut4(inp: str) -> str:
    """At each full stop that is not a decimal point."""
    inp = inp.strip("\n")
    opts = re.split(r"(?<!\d)\.(?!\d)", inp.strip("."))
    opts = [item for item in opts if not set(item).issubset(PUNCTUATION)]
    return "\n".join(opts)


def cut5(inp: str) -> str:
    """At every punctuation mark (a decimal point excepted)."""
    inp = inp.strip("\n")
    punds = {",", ".", ";", "?", "!", "、", "，", "。", "？", "！", "：", "…"}
    mergeitems: list[str] = []
    items: list[str] = []
    for i, char in enumerate(inp):
        if char in punds:
            if char == "." and i > 0 and i < len(inp) - 1 and inp[i - 1].isdigit() and inp[i + 1].isdigit():
                items.append(char)
            else:
                items.append(char)
                mergeitems.append("".join(items))
                items = []
        else:
            items.append(char)
    if items:
        mergeitems.append("".join(items))
    opt = [item for item in mergeitems if not set(item).issubset(punds)]
    return "\n".join(opt)


METHODS: dict[str, Callable[[str], str]] = {"cut0": cut0, "cut1": cut1, "cut2": cut2, "cut3": cut3, "cut4": cut4, "cut5": cut5}


def get_method(name: str) -> Callable[[str], str]:
    method = METHODS.get(name)
    if method is None:
        raise ValueError(f"Method {name} not found")
    return method


def replace_consecutive_punctuation(text: str) -> str:
    """``TextPreprocessor.replace_consecutive_punctuation``: ``!!!`` → ``!``, ``,.`` → ``,``."""
    punctuations = "".join(re.escape(p) for p in CONSECUTIVE)
    return re.sub(f"([{punctuations}])([{punctuations}])+", r"\1", text)


def get_first(text: str) -> str:
    return _FIRST.split(text)[0].strip()


def merge_short_text_in_array(texts: list[str], threshold: int) -> list[str]:
    """Joins pieces until each has ``threshold`` characters; a short tail joins the last piece."""
    if len(texts) < 2:
        return texts
    result: list[str] = []
    text = ""
    for ele in texts:
        text += ele
        if len(text) >= threshold:
            result.append(text)
            text = ""
    if len(text) > 0:
        if len(result) == 0:
            result.append(text)
        else:
            result[len(result) - 1] += text
    return result


def filter_text(texts: list[str]) -> list[str]:
    if all(text in (None, " ", "\n", "") for text in texts):
        raise EmptyTextError("请输入有效文本")
    return [text for text in texts if text not in (None, " ", "")]


def pre_seg_text(text: str, lang: str, method: str) -> list[str]:
    """``TextPreprocessor.pre_seg_text``: the pieces synthesised one by one (``lang`` only picks ``.`` or ``。``)."""
    text = text.strip("\n")
    if len(text) == 0:
        return []
    stop = "." if lang == "en" else "。"
    if text[0] not in SPLITS and len(get_first(text)) < 4:
        text = stop + text
    text = get_method(method)(text)
    while "\n\n" in text:
        text = text.replace("\n\n", "\n")
    pieces = merge_short_text_in_array(filter_text(text.split("\n")), 5)
    texts: list[str] = []
    for piece in pieces:
        if len(piece.strip()) == 0:
            continue
        if not re.sub(r"\W+", "", piece):
            continue
        if piece[-1] not in SPLITS:
            piece += stop
        if len(piece) > 510:
            texts.extend(split_big_text(piece))
        else:
            texts.append(piece)
    return texts
