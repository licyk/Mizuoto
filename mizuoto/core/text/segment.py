"""Splitting long text into pieces an engine can speak.

Sentences end at CJK and Latin terminators and line breaks; short sentences are merged up to a
budget, long ones are cut at commas, then at spaces. Inline tags (``[laughter]``, ``<|zh|>``,
``<strong>…``, ``{pin1 yin1}``, ``<word|PRON>``) are never cut and never end a sentence.
"""

from __future__ import annotations

import re

TAG = re.compile(r"\[[^\[\]\n]{1,40}\]|<\|[^|\n]{1,40}\|>|</?[A-Za-z][^<>\n]{0,40}>|<[^<>\n|]{1,40}\|[^<>\n]{1,40}>|\{[^{}\n]{1,60}\}")
# A full stop ends a sentence unless it ends a common abbreviation ("Dr. Smith").
SENTENCE_END = re.compile(r"[。！？!?；;…]+[”’」』\"')）]*|(?<!\bMr)(?<!\bMs)(?<!\bDr)(?<!\bSt)(?<!\bvs)(?<!\bMrs)(?<!\bNo)(?<!\bProf)(?<!\betc)\.(?=\s|$)[\"')]*|\n+")
SOFT_BREAK = re.compile(r"[,，、:：]\s*")
CJK = re.compile(r"[぀-ヿ㐀-䶿一-鿿가-힯]")

DEFAULT_BUDGET_CJK = 100
DEFAULT_BUDGET_OTHER = 250


def default_budget(text: str) -> int:
    """Characters per piece: fewer for CJK text, whose characters carry more speech each."""
    cjk = len(CJK.findall(text))
    return DEFAULT_BUDGET_CJK if cjk * 3 > len(text.strip() or "x") else DEFAULT_BUDGET_OTHER


def _protected(text: str) -> list[tuple[int, int]]:
    return [(m.start(), m.end()) for m in TAG.finditer(text)]


def _inside(pos: int, spans: list[tuple[int, int]]) -> bool:
    return any(a < pos < b for a, b in spans)


def _split(text: str, pattern: re.Pattern[str]) -> list[str]:
    spans = _protected(text)
    pieces: list[str] = []
    start = 0
    for m in pattern.finditer(text):
        if _inside(m.start(), spans) or _inside(m.end(), spans):
            continue
        pieces.append(text[start : m.end()])
        start = m.end()
    pieces.append(text[start:])
    return [p for p in pieces if p.strip()]


def _hard_split(text: str, budget: int) -> list[str]:
    """Cut at spaces (or anywhere, for text without them) outside tags."""
    spans = _protected(text)
    out: list[str] = []
    while len(text) > budget:
        cut = -1
        for i in range(budget, max(budget // 2, 1) - 1, -1):
            if text[i - 1].isspace() and not _inside(i, spans):
                cut = i
                break
        if cut <= 0:
            cut = budget
            while _inside(cut, spans) and cut < len(text):
                cut += 1
        out.append(text[:cut])
        text = text[cut:]
        spans = _protected(text)
    if text.strip():
        out.append(text)
    return out


def sentences(text: str) -> list[str]:
    """``text``'s sentences, unmerged (tags are never cut)."""
    return [s.strip() for s in _split(text.strip(), SENTENCE_END) if s.strip()]


def segment(text: str, max_chars: int | None = None) -> list[str]:
    """``text`` as pieces of at most ``max_chars`` characters (tags excepted), split at sentence ends."""
    text = text.strip()
    if not text:
        return []
    budget = max(8, max_chars or default_budget(text))
    sentences: list[str] = []
    for sentence in _split(text, SENTENCE_END):
        if len(sentence) <= budget:
            sentences.append(sentence)
            continue
        for clause in _split(sentence, SOFT_BREAK):
            sentences.extend([clause] if len(clause) <= budget else _hard_split(clause, budget))
    pieces: list[str] = []
    current = ""
    for s in sentences:
        if current and len(current) + len(s) > budget:
            pieces.append(current.strip())
            current = s
        else:
            current += s
    if current.strip():
        pieces.append(current.strip())
    return pieces
