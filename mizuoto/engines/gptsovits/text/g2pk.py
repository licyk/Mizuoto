"""Korean pronunciation rules: g2pk2 0.0.3's ``G2p`` (Apache-2.0, Kyubyong Park; ``github.com/kyubyong/g2pK``) as GPT-SoVITS
calls it (``descriptive=False, group_vowels=False, to_syl=True``).

g2pk2's import downloads nltk's CMUdict and its constructor installs packages with pip when MeCab is missing, so the
rules are ported here: idioms, MeCab annotation, numerals, the special rules, the coda × onset table and linking. Its
English step (CMUdict spellings of Latin words) is left out: GPT-SoVITS turns every Latin letter into Hangul before
calling it, and no CMUdict entry is made of apostrophes alone, so the step never changes GPT-SoVITS's input.
MeCab is python-mecab-ko's (``mecab.MeCab()``), the binding g2pk2 uses everywhere but on Windows.
"""

from __future__ import annotations

import re
from typing import Any

BOUND_NOUNS = "군데 권 개 그루 닢 대 두 마리 모 모금 뭇 발 발짝 방 번 벌 보루 살 수 술 시 쌈 움큼 정 짝 채 척 첩 축 켤레 톨 통"


class KoreanMissing(ImportError):
    pass


def _jamo() -> Any:
    try:
        import jamo
    except ImportError as e:
        raise KoreanMissing("Korean needs the jamo and python-mecab-ko packages (Mizuoto's Korean extra)") from e
    return jamo


def parse_table() -> list[tuple[str, str]]:
    from mizuoto.engines.gptsovits.text._data import read_text

    lines = read_text("g2pk_table.csv").splitlines()
    onsets = lines[0].split(",")
    table: list[tuple[str, str]] = []
    for line in lines[1:]:
        cols = line.split(",")
        coda = cols[0]
        for i, onset in enumerate(onsets):
            cell = cols[i]
            if len(cell) == 0 or i == 0:
                continue
            table.append((f"{coda}{onset}", cell.split("(")[0] if "(" in cell else cell))
    return table


def read_idioms() -> list[tuple[str, str]]:
    from mizuoto.engines.gptsovits.text._data import read_text

    idioms: list[tuple[str, str]] = []
    for line in read_text("g2pk_idioms.txt").splitlines():
        line = line.split("#")[0].strip()
        if "===" in line:
            str1, str2 = line.split("===")
            idioms.append((str1, str2))
    return idioms


def annotate(string: str, mecab: Any) -> str:
    h2j = _jamo().h2j
    tokens = mecab.pos(string)
    if string.replace(" ", "") != "".join(token for token, _ in tokens):
        return string
    blanks = [i for i, char in enumerate(string) if char == " "]
    tag_seq_parts = []
    for token, tag in tokens:
        tag = tag.split("+")[-1]
        tag = "B" if tag == "NNBC" else tag[0]
        tag_seq_parts.append("_" * (len(token) - 1) + tag)
    tag_seq = "".join(tag_seq_parts)
    for i in blanks:
        tag_seq = tag_seq[:i] + " " + tag_seq[i:]
    annotated = ""
    for char, tag in zip(string, tag_seq, strict=False):
        annotated += char
        if char == "의" and tag == "J":
            annotated += "/J"
        elif tag == "E":
            if h2j(char)[-1] in "ᆯ":
                annotated += "/E"
        elif tag == "V":
            if h2j(char)[-1] in "ᆫᆬᆷᆱᆰᆲᆴ":
                annotated += "/P"
        elif tag == "B":
            annotated += "/B"
    return annotated


def _words(text: str) -> list[str]:
    return text.split()


def process_num(num: str, sino: bool = True) -> str:
    num = re.sub(",", "", num)
    if num == "0":
        return "영"
    if not sino and num == "20":
        return "스무"
    digit2name = dict(zip("123456789", "일이삼사오육칠팔구", strict=True))
    digit2mod = dict(zip("123456789", _words("한 두 세 네 다섯 여섯 일곱 여덟 아홉"), strict=True))
    digit2dec = dict(zip("123456789", _words("열 스물 서른 마흔 쉰 예순 일흔 여든 아흔"), strict=True))
    spelledout: list[str] = []
    name = ""
    for i, digit in enumerate(num):
        i = len(num) - i - 1
        if sino:
            if i == 0:
                name = digit2name.get(digit, "")
            elif i == 1:
                name = (digit2name.get(digit, "") + "십").replace("일십", "십")
        elif i == 0:
            name = digit2mod.get(digit, "")
        elif i == 1:
            name = digit2dec.get(digit, "")
        if digit == "0":
            if i % 4 == 0:
                last_three = spelledout[-min(3, len(spelledout)) :]
                if "".join(last_three) == "":
                    spelledout.append("")
                    continue
            else:
                spelledout.append("")
                continue
        units = {2: "백", 3: "천", 4: "만", 5: "십", 6: "백", 7: "천", 8: "억", 9: "십", 10: "백", 11: "천", 12: "조", 13: "십", 14: "백", 15: "천"}
        if i in units:
            name = digit2name.get(digit, "") + units[i]
            if i <= 7:
                name = name.replace("일" + units[i], units[i])
        spelledout.append(name)
    return "".join(spelledout)


def convert_num(string: str) -> str:
    for num, bn in set(re.findall(r"([\d][\d,]*)([ㄱ-힣]+)/B", string)):
        spelledout = process_num(num, sino=bn not in BOUND_NOUNS)
        string = string.replace(f"{num}{bn}/B", f"{spelledout}{bn}/B")
    for d, n in zip("0123456789", "영일이삼사오육칠팔구", strict=True):
        string = string.replace(d, n)
    return string


# -- special.py (descriptive=False) ------------------------------------------------------------------------------------


def _special(inp: str) -> str:
    out = re.sub("([ᄌᄍᄎ])ᅧ", r"\1ᅥ", inp)  # jyeo
    # ye: descriptive only
    out = re.sub("([ᄀᄁᄂᄃᄄᄅᄆᄇᄈᄉᄊᄌᄍᄎᄏᄐᄑᄒ])ᅴ", r"\1ᅵ", out)  # consonant_ui
    out = out.replace("/J", "")  # josa_ui
    # vowel_ui: descriptive only
    out = re.sub("([그])ᆮᄋ", r"\1ᄉ", out)  # jamo
    out = re.sub("([으])[ᆽᆾᇀᇂ]ᄋ", r"\1ᄉ", out)
    out = re.sub("([으])[ᆿ]ᄋ", r"\1ᄀ", out)
    out = re.sub("([으])[ᇁ]ᄋ", r"\1ᄇ", out)
    out = re.sub("ᆰ/P([ᄀᄁ])", r"ᆯᄁ", out)  # rieulgiyeok
    for onset, tense in (("ᄀ", "ᄁ"), ("ᄃ", "ᄄ"), ("ᄉ", "ᄊ"), ("ᄌ", "ᄍ")):  # rieulbieub
        out = re.sub(f"([ᆲᆴ])/P{onset}", rf"\1{tense}", out)
    for pattern, repl in (  # verb_nieun
        *((f"([ᆫᆷ])/P{o}", rf"\1{t}") for o, t in (("ᄀ", "ᄁ"), ("ᄃ", "ᄄ"), ("ᄉ", "ᄊ"), ("ᄌ", "ᄍ"))),
        *((f"ᆬ/P{o}", f"ᆫ{t}") for o, t in (("ᄀ", "ᄁ"), ("ᄃ", "ᄄ"), ("ᄉ", "ᄊ"), ("ᄌ", "ᄍ"))),
        *((f"ᆱ/P{o}", f"ᆷ{t}") for o, t in (("ᄀ", "ᄁ"), ("ᄃ", "ᄄ"), ("ᄉ", "ᄊ"), ("ᄌ", "ᄍ"))),
    ):
        out = re.sub(pattern, repl, out)
    out = re.sub("(바)ᆲ(($|[^ᄋᄒ]))", r"\1ᆸ\2", out)  # balb
    out = re.sub("(너)ᆲ([ᄌᄍ]ᅮ|[ᄃᄄ]ᅮ)", r"\1ᆸ\2", out)
    out = re.sub("ᆮᄋ([ᅵᅧ])", r"ᄌ\1", out)  # palatalize
    out = re.sub("ᇀᄋ([ᅵᅧ])", r"ᄎ\1", out)
    out = re.sub("ᆴᄋ([ᅵᅧ])", r"ᆯᄎ\1", out)
    out = re.sub("ᆮᄒ([ᅵ])", r"ᄎ\1", out)
    for pattern, repl in (  # modifying_rieul
        ("ᆯ/E ᄀ", r"ᆯ ᄁ"),
        ("ᆯ/E ᄃ", r"ᆯ ᄄ"),
        ("ᆯ/E ᄇ", r"ᆯ ᄈ"),
        ("ᆯ/E ᄉ", r"ᆯ ᄊ"),
        ("ᆯ/E ᄌ", r"ᆯ ᄍ"),
        ("ᆯ걸", "ᆯ껄"),
        ("ᆯ밖에", "ᆯ빠께"),
        ("ᆯ세라", "ᆯ쎄라"),
        ("ᆯ수록", "ᆯ쑤록"),
        ("ᆯ지라도", "ᆯ찌라도"),
        ("ᆯ지언정", "ᆯ찌언정"),
        ("ᆯ진대", "ᆯ찐대"),
    ):
        out = re.sub(pattern, repl, out)
    return out


LINK1 = [
    ("ᆨᄋ", "ᄀ"),
    ("ᆩᄋ", "ᄁ"),
    ("ᆫᄋ", "ᄂ"),
    ("ᆮᄋ", "ᄃ"),
    ("ᆯᄋ", "ᄅ"),
    ("ᆷᄋ", "ᄆ"),
    ("ᆸᄋ", "ᄇ"),
    ("ᆺᄋ", "ᄉ"),
    ("ᆻᄋ", "ᄊ"),
    ("ᆽᄋ", "ᄌ"),
    ("ᆾᄋ", "ᄎ"),
    ("ᆿᄋ", "ᄏ"),
    ("ᇀᄋ", "ᄐ"),
    ("ᇁᄋ", "ᄑ"),
]
LINK2 = [("ᆪᄋ", "ᆨᄊ"), ("ᆬᄋ", "ᆫᄌ"), ("ᆰᄋ", "ᆯᄀ"), ("ᆱᄋ", "ᆯᄆ"), ("ᆲᄋ", "ᆯᄇ"), ("ᆳᄋ", "ᆯᄊ"), ("ᆴᄋ", "ᆯᄐ"), ("ᆵᄋ", "ᆯᄑ"), ("ᆹᄋ", "ᆸᄊ")]
LINK4 = [("ᇂᄋ", "ᄋ"), ("ᆭᄋ", "ᄂ"), ("ᆶᄋ", "ᄅ")]


def compose(letters: str) -> str:
    j2h = _jamo().j2h
    letters = re.sub("(^|[^ᄀ-ᄒ])([ᅡ-ᅵ])", r"\1ᄋ\2", letters)
    string = letters
    for syl in set(re.findall("[ᄀ-ᄒ][ᅡ-ᅵ][ᆨ-ᇂ]", string)):
        string = string.replace(syl, j2h(*syl))
    for syl in set(re.findall("[ᄀ-ᄒ][ᅡ-ᅵ]", string)):
        string = string.replace(syl, j2h(*syl))
    return string


class G2pK:
    """``g2pk2.G2p()(string)``."""

    def __init__(self) -> None:
        _jamo()
        try:
            import mecab
        except ImportError as e:
            raise KoreanMissing("Korean needs the jamo and python-mecab-ko packages (Mizuoto's Korean extra)") from e
        self.mecab = mecab.MeCab()
        self.table = parse_table()
        self.idioms = read_idioms()

    def __call__(self, string: str) -> str:
        h2j = _jamo().h2j
        for str1, str2 in self.idioms:
            string = re.sub(str1, str2, string)
        string = annotate(string, self.mecab)
        string = convert_num(string)
        inp = _special(h2j(string))
        inp = re.sub("/[PJEB]", "", inp)
        for str1, str2 in self.table:
            inp = re.sub(str1, str2, inp)
        for pairs in (LINK1, LINK2, LINK4):
            for str1, str2 in pairs:
                inp = inp.replace(str1, str2)
        return compose(inp)
