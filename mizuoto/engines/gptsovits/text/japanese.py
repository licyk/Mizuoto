"""Japanese for GPT-SoVITS (upstream ``text/japanese.py``, from CjangCjengh's vits and espnet's ``pyopenjtalk_g2p_prosody``;
MIT, (c) 2024 RVC-Boss).

Phones with prosody marks (``[`` rise, ``]`` fall, ``#`` accent-phrase border, ``_`` pause) from OpenJTalk's full-context
labels. Upstream uses pyopenjtalk's global instance, updated with its user dictionary (``ja_userdic/userdict.csv``,
compiled to ``user.dict`` next to it on first use). Here the front end owns an ``OpenJTalk`` built from the
``gptsovits/open-jtalk`` asset and that user dictionary, compiled once into a cache folder.

The binding must be ``pyopenjtalk`` 0.4.x (built from source: it publishes no wheels). ``pyopenjtalk-plus``, which
has wheels, modifies OpenJTalk's own rules (unknown-word classes, accent sandhi): its labels differ for about one
Japanese sentence in five, so it is not used for parity.
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Any

from mizuoto.engines.gptsovits.text.symbols import PUNCTUATION

_japanese_characters = re.compile(r"[A-Za-z\d々぀-ヿ一-鿿１-９Ａ-Ｚａ-ｚｦ-ﾝ]")
_japanese_marks = re.compile(r"[^A-Za-z\d々぀-ヿ一-鿿１-９Ａ-Ｚａ-ｚｦ-ﾝ]")
_symbols_to_japanese = [(re.compile(x[0]), x[1]) for x in [("％", "パーセント")]]
_CONSECUTIVE_RE = re.compile("([{0}])([{0}])+".format("".join(re.escape(p) for p in PUNCTUATION)))

POST_REP_MAP = {"：": ",", "；": ",", "，": ",", "。": ".", "！": "!", "？": "?", "\n": ".", "·": ",", "、": ",", "...": "…"}

USERDICT = "ja_userdict.csv"


class OpenJTalkMissing(ImportError):
    pass


_COMPILE = """
import sys
from pyopenjtalk.openjtalk import mecab_dict_index
sys.exit(mecab_dict_index(*(a.encode("utf-8") for a in sys.argv[1:4])))
"""
"""The compile step as a child process: ``mecab_dict_index(dict_dir, csv, out)``."""


def compile_user_dict(dict_dir: Path, cache_dir: Path) -> Path:
    """``user.dict`` from upstream's CSV, compiled against ``dict_dir`` once (named by the CSV's and dictionary's digests).

    MeCab's compiler writes its progress to the C ``stdout``, which would end up in whatever this process writes there
    (``mizuoto speak --stream`` writes WAV), so it runs in a child process with its output captured.
    """
    import subprocess
    import sys

    from mizuoto.engines.gptsovits.text._data import read_bytes

    csv = read_bytes(USERDICT)
    digest = hashlib.sha256(csv + str(dict_dir.resolve()).encode() + (dict_dir / "sys.dic").read_bytes()[:4096]).hexdigest()[:16]
    out = cache_dir / f"gptsovits-ja-user-{digest}.dic"
    if out.is_file():
        return out
    cache_dir.mkdir(parents=True, exist_ok=True)
    csv_path = cache_dir / f"gptsovits-ja-user-{digest}.csv"
    csv_path.write_bytes(csv)
    part = out.with_suffix(".part")
    try:
        done = subprocess.run([sys.executable, "-c", _COMPILE, str(dict_dir), str(csv_path), str(part)], capture_output=True, text=True, check=False)
        if done.returncode != 0 or not part.is_file():
            raise RuntimeError(f"Failed to create the Japanese user dictionary: {(done.stderr or done.stdout).strip()[-500:]}")
        part.replace(out)
    finally:
        csv_path.unlink(missing_ok=True)
        part.unlink(missing_ok=True)
    return out


def _numeric_feature_by_regex(pattern: str, s: str) -> int:
    match = re.search(pattern, s)
    if match is None:
        return -50
    return int(match.group(1))


def post_replace_ph(ph: str) -> str:
    return POST_REP_MAP.get(ph, ph)


def text_normalize(text: str) -> str:
    return _CONSECUTIVE_RE.sub(r"\1", text)


class JapaneseG2P:
    """``japanese.g2p(norm_text, with_prosody=True)`` on an OpenJTalk instance of its own.

    ``dict_dir`` is the Open JTalk dictionary (``open_jtalk_dic_utf_8-1.11``); ``cache_dir`` holds the compiled user
    dictionary (``None``: no user dictionary, which upstream also falls back to when compiling fails).
    """

    def __init__(self, dict_dir: str | Path, cache_dir: str | Path | None) -> None:
        try:
            from pyopenjtalk.openjtalk import OpenJTalk  # ty: ignore[unresolved-import]
        except ImportError as e:
            raise OpenJTalkMissing("Japanese needs pyopenjtalk (pip install pyopenjtalk==0.4.1; it is built from source)") from e
        dict_dir = Path(dict_dir)
        userdic = b""
        if cache_dir is not None:
            userdic = str(compile_user_dict(dict_dir, Path(cache_dir))).encode("utf-8")
        self.jtalk: Any = OpenJTalk(dn_mecab=str(dict_dir).encode("utf-8"), userdic=userdic)

    text_normalize = staticmethod(text_normalize)

    def g2p_prosody(self, text: str, drop_unvoiced_vowels: bool = True) -> list[str]:
        """espnet's ``pyopenjtalk_g2p_prosody``."""
        labels = self.jtalk.make_label(self.jtalk.run_frontend(text))
        n_labels = len(labels)
        phones: list[str] = []
        for n in range(n_labels):
            lab_curr = labels[n]
            found = re.search(r"\-(.*?)\+", lab_curr)
            assert found is not None
            p3 = found.group(1)
            if drop_unvoiced_vowels and p3 in "AEIOU":
                p3 = p3.lower()
            if p3 == "sil":
                assert n == 0 or n == n_labels - 1
                if n == 0:
                    phones.append("^")
                elif n == n_labels - 1:
                    e3 = _numeric_feature_by_regex(r"!(\d+)_", lab_curr)
                    if e3 == 0:
                        phones.append("$")
                    elif e3 == 1:
                        phones.append("?")
                continue
            if p3 == "pau":
                phones.append("_")
                continue
            phones.append(p3)
            a1 = _numeric_feature_by_regex(r"/A:([0-9\-]+)\+", lab_curr)
            a2 = _numeric_feature_by_regex(r"\+(\d+)\+", lab_curr)
            a3 = _numeric_feature_by_regex(r"\+(\d+)/", lab_curr)
            f1 = _numeric_feature_by_regex(r"/F:(\d+)_", lab_curr)
            a2_next = _numeric_feature_by_regex(r"\+(\d+)\+", labels[n + 1])
            if a3 == 1 and a2_next == 1 and p3 in "aeiouAEIOUNcl":
                phones.append("#")
            elif a1 == 0 and a2_next == a2 + 1 and a2 != f1:
                phones.append("]")
            elif a2 == 1 and a2_next == 2:
                phones.append("[")
        return phones

    def g2p(self, norm_text: str) -> list[str]:
        """``preprocess_jap(text, with_prosody=True)`` and the punctuation mapping."""
        text = norm_text
        for pattern, replacement in _symbols_to_japanese:
            text = re.sub(pattern, replacement, text)
        text = text.lower()
        sentences = re.split(_japanese_marks, text)
        marks = re.findall(_japanese_marks, text)
        phones: list[str] = []
        for i, sentence in enumerate(sentences):
            if re.match(_japanese_characters, sentence):
                phones += self.g2p_prosody(sentence)[1:-1]
            if i < len(marks):
                if marks[i] == " ":
                    continue
                phones += [marks[i].replace(" ", "")]
        return [post_replace_ph(p) for p in phones]
