"""OmniVoice's closed design vocabulary (upstream ``omnivoice/utils/voice_design.py`` and ``_resolve_instruct``, Apache-2.0, Xiaomi Corp.).

An instruction is a comma-separated list of items from six categories (gender, age, pitch, whisper,
English accent, Chinese dialect), at most one per category, all in English (``, ``) or all in Chinese
(``，``): the text's script decides, unless a dialect (Chinese) or an accent (English) does.
"""

from __future__ import annotations

import difflib
import re

from mizuoto.engines.omnivoice.lang_map import LANG_IDS, LANG_NAME_TO_ID

ZH_RE = re.compile(r"[一-鿿]")

CATEGORIES: list[dict[str, str] | set[str]] = [
    {"male": "男", "female": "女"},
    {"child": "儿童", "teenager": "少年", "young adult": "青年", "middle-aged": "中年", "elderly": "老年"},
    {"very low pitch": "极低音调", "low pitch": "低音调", "moderate pitch": "中音调", "high pitch": "高音调", "very high pitch": "极高音调"},
    {"whisper": "耳语"},
    {
        *("american accent", "british accent", "australian accent", "chinese accent", "canadian accent"),
        *("indian accent", "korean accent", "portuguese accent", "russian accent", "japanese accent"),
    },
    {"河南话", "陕西话", "四川话", "贵州话", "云南话", "桂林话", "济南话", "石家庄话", "甘肃话", "宁夏话", "青岛话", "东北话"},
]

EN_TO_ZH: dict[str, str] = {}
ZH_TO_EN: dict[str, str] = {}
MUTUALLY_EXCLUSIVE: list[set[str]] = []
for _cat in CATEGORIES:
    if isinstance(_cat, dict):
        EN_TO_ZH.update(_cat)
        ZH_TO_EN.update({v: k for k, v in _cat.items()})
        MUTUALLY_EXCLUSIVE.append(set(_cat) | set(_cat.values()))
    else:
        MUTUALLY_EXCLUSIVE.append(set(_cat))
ALL_VALID = set(EN_TO_ZH) | set(ZH_TO_EN) | MUTUALLY_EXCLUSIVE[-2] | MUTUALLY_EXCLUSIVE[-1]
VALID_EN = frozenset(i for i in ALL_VALID if not ZH_RE.search(i))
VALID_ZH = frozenset(i for i in ALL_VALID if ZH_RE.search(i))


def resolve_instruct(instruct: str | None, use_zh: bool = False) -> str | None:
    """``_resolve_instruct``: validated, in one language, joined with that language's comma."""
    if instruct is None or not instruct.strip():
        return None
    text = instruct.strip()
    items = [x for x in re.split(r"\s*[,，]\s*", text) if x]
    unknown = []
    normalised = []
    for raw in items:
        n = raw.strip().lower()
        if n in ALL_VALID:
            normalised.append(n)
        else:
            suggestion = difflib.get_close_matches(n, ALL_VALID, n=1, cutoff=0.6)
            unknown.append((raw, n, suggestion[0] if suggestion else None))
    if unknown:
        lines = [f"  '{raw}' -> '{n}' (unsupported; did you mean '{s}'?)" if s else f"  '{raw}' -> '{n}' (unsupported)" for raw, n, s in unknown]
        raise ValueError(
            f"Unsupported instruct items found in {text}:\n"
            + "\n".join(lines)
            + "\n\nValid English items: "
            + ", ".join(sorted(VALID_EN))
            + "\nValid Chinese items: "
            + "，".join(sorted(VALID_ZH))
        )
    has_dialect = any(n.endswith("话") for n in normalised)
    has_accent = any(" accent" in n for n in normalised)
    if has_dialect and has_accent:
        raise ValueError("Cannot mix Chinese dialect and English accent in a single instruct.")
    if has_dialect:
        use_zh = True
    elif has_accent:
        use_zh = False
    normalised = [EN_TO_ZH.get(n, n) for n in normalised] if use_zh else [ZH_TO_EN.get(n, n) for n in normalised]
    conflicts = [hits for cat in MUTUALLY_EXCLUSIVE if len(hits := [n for n in normalised if n in cat]) > 1]
    if conflicts:
        raise ValueError(
            "Conflicting instruct items within the same category: "
            + "; ".join(" vs ".join(f"'{x}'" for x in group) for group in conflicts)
            + ". Each category allows at most one item."
        )
    has_zh = any(any("一" <= c <= "鿿" for c in n) for n in normalised)
    return ("，" if has_zh else ", ").join(normalised)


def resolve_language(language: str | None) -> str | None:
    """``_resolve_language``: an id, or a name mapped to one; ``None`` (language-agnostic) otherwise."""
    if language is None or language.lower() == "none":
        return None
    if language in LANG_IDS:
        return language
    return LANG_NAME_TO_ID.get(language.lower())
