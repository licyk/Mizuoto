"""GPT-SoVITS's phone symbols, version 2 and later (upstream ``text/symbols2.py``, MIT, (c) 2024 RVC-Boss).

The table is upstream's: the sorted set of padding, Chinese initials and finals, Japanese phones, punctuation and
ARPAbet, then the Japanese pitch marks ``[`` ``]``, the Korean jamo and the Cantonese phones, each sorted. A phone's id
is its index here (732 symbols).
"""

from __future__ import annotations

from collections.abc import Iterable

PUNCTUATION = ["!", "?", "…", ",", ".", "-"]
"""``text/symbols.py``'s punctuation, which the language modules use."""


def _words(text: str) -> list[str]:
    return text.split()


PAD = "_"
_INITIALS = _words("AA EE OO b c ch d f g h j k l m n p q r s sh t w x y z zh")
"""Chinese initials (``c``)."""
_FINALS = _words(
    "E1 En1 a1 ai1 an1 ang1 ao1 e1 ei1 en1 eng1 er1 i1 i01 ia1 ian1 iang1 iao1 ie1 in1 ing1 iong1 ir1 iu1 o1 ong1 ou1 u1 ua1 uai1 uan1 uang1 ui1 un1 uo1 v1 van1 ve1 "
    "vn1 E2 En2 a2 ai2 an2 ang2 ao2 e2 ei2 en2 eng2 er2 i2 i02 ia2 ian2 iang2 iao2 ie2 in2 ing2 iong2 ir2 iu2 o2 ong2 ou2 u2 ua2 uai2 uan2 uang2 ui2 un2 uo2 v2 van2 "
    "ve2 vn2 E3 En3 a3 ai3 an3 ang3 ao3 e3 ei3 en3 eng3 er3 i3 i03 ia3 ian3 iang3 iao3 ie3 in3 ing3 iong3 ir3 iu3 o3 ong3 ou3 u3 ua3 uai3 uan3 uang3 ui3 un3 uo3 v3 "
    "van3 ve3 vn3 E4 En4 a4 ai4 an4 ang4 ao4 e4 ei4 en4 eng4 er4 i4 i04 ia4 ian4 iang4 iao4 ie4 in4 ing4 iong4 ir4 iu4 o4 ong4 ou4 u4 ua4 uai4 uan4 uang4 ui4 un4 uo4 "
    "v4 van4 ve4 vn4 E5 En5 a5 ai5 an5 ang5 ao5 e5 ei5 en5 eng5 er5 i5 i05 ia5 ian5 iang5 iao5 ie5 in5 ing5 iong5 ir5 iu5 o5 ong5 ou5 u5 ua5 uai5 uan5 uang5 ui5 un5 "
    "uo5 v5 van5 ve5 vn5"
)
"""Chinese finals with their tones (``v``)."""
_JA = _words("I N U a b by ch cl d dy e f g gy h hy i j k ky m my n ny o p py r ry s sh t ts u v w y z")
"""Japanese phones."""
_PU = _words("! ? … , . - SP SP2 SP3 UNK")
"""Punctuation and the pause symbols."""
_ARPA = _words(
    "AA0 AA1 AA2 AE0 AE1 AE2 AH0 AH1 AH2 AO0 AO1 AO2 AW0 AW1 AW2 AY0 AY1 AY2 B CH D DH EH0 EH1 EH2 ER ER0 ER1 ER2 EY0 EY1 EY2 F G HH IH IH0 IH1 IH2 IY0 IY1 IY2 JH K "
    "L M N NG OW0 OW1 OW2 OY0 OY1 OY2 P R S SH T TH UH0 UH1 UH2 UW0 UW1 UW2 V W Y Z ZH"
)
"""ARPAbet (a set upstream; its order does not matter, the table is sorted)."""
_KO = "ㄱㄴㄷㄹㅁㅂㅅㅇㅈㅊㅋㅌㅍㅎㄲㄸㅃㅆㅉㅏㅓㅗㅜㅡㅣㅐㅔ空停"
"""Korean jamo, plus 空 (space) and 停 (other)."""
_YUE = _words(
    "Ya Ya1 Ya2 Ya3 Ya4 Ya5 Ya6 Yaa Yaa1 Yaa2 Yaa3 Yaa4 Yaa5 Yaa6 Yaai1 Yaai2 Yaai3 Yaai4 Yaai5 Yaai6 Yaak1 Yaak2 Yaak3 Yaak4 Yaak5 Yaak6 Yaam1 Yaam2 Yaam3 Yaam4 "
    "Yaam5 Yaam6 Yaan1 Yaan2 Yaan3 Yaan4 Yaan5 Yaan6 Yaang1 Yaang2 Yaang3 Yaang4 Yaang5 Yaang6 Yaap1 Yaap2 Yaap3 Yaap4 Yaap5 Yaap6 Yaat1 Yaat2 Yaat3 Yaat4 Yaat5 "
    "Yaat6 Yaau1 Yaau2 Yaau3 Yaau4 Yaau5 Yaau6 Yai Yai1 Yai2 Yai3 Yai4 Yai5 Yai6 Yak Yak1 Yak2 Yak3 Yak4 Yak5 Yak6 Yam1 Yam2 Yam3 Yam4 Yam5 Yam6 Yan1 Yan2 Yan3 Yan4 "
    "Yan5 Yan6 Yang1 Yang2 Yang3 Yang4 Yang5 Yang6 Yap1 Yap2 Yap3 Yap4 Yap5 Yap6 Yat1 Yat2 Yat3 Yat4 Yat5 Yat6 Yau Yau1 Yau2 Yau3 Yau4 Yau5 Yau6 Yb Yc Yd Ye Ye1 Ye2 "
    "Ye3 Ye4 Ye5 Ye6 Yei1 Yei2 Yei3 Yei4 Yei5 Yei6 Yek1 Yek2 Yek3 Yek4 Yek5 Yek6 Yeng1 Yeng2 Yeng3 Yeng4 Yeng5 Yeng6 Yeoi1 Yeoi2 Yeoi3 Yeoi4 Yeoi5 Yeoi6 Yeon1 Yeon2 "
    "Yeon3 Yeon4 Yeon5 Yeon6 Yeot1 Yeot2 Yeot3 Yeot4 Yeot5 Yeot6 Yf Yg Yg1 Yg2 Yg3 Yg4 Yg5 Yg6 Ygw Yh Yi1 Yi2 Yi3 Yi4 Yi5 Yi6 Yik1 Yik2 Yik3 Yik4 Yik5 Yik6 Yim1 Yim2 "
    "Yim3 Yim4 Yim5 Yim6 Yin1 Yin2 Yin3 Yin4 Yin5 Yin6 Ying1 Ying2 Ying3 Ying4 Ying5 Ying6 Yip1 Yip2 Yip3 Yip4 Yip5 Yip6 Yit1 Yit2 Yit3 Yit4 Yit5 Yit6 Yiu1 Yiu2 Yiu3 "
    "Yiu4 Yiu5 Yiu6 Yj Yk Yk1 Yk2 Yk3 Yk4 Yk5 Yk6 Ykw Yl Ym Ym1 Ym2 Ym3 Ym4 Ym5 Ym6 Yn Yn1 Yn2 Yn3 Yn4 Yn5 Yn6 Yng Yo Yo1 Yo2 Yo3 Yo4 Yo5 Yo6 Yoe1 Yoe2 Yoe3 Yoe4 "
    "Yoe5 Yoe6 Yoek1 Yoek2 Yoek3 Yoek4 Yoek5 Yoek6 Yoeng1 Yoeng2 Yoeng3 Yoeng4 Yoeng5 Yoeng6 Yoi Yoi1 Yoi2 Yoi3 Yoi4 Yoi5 Yoi6 Yok Yok1 Yok2 Yok3 Yok4 Yok5 Yok6 Yon "
    "Yon1 Yon2 Yon3 Yon4 Yon5 Yon6 Yong1 Yong2 Yong3 Yong4 Yong5 Yong6 Yot1 Yot2 Yot3 Yot4 Yot5 Yot6 You You1 You2 You3 You4 You5 You6 Yp Yp1 Yp2 Yp3 Yp4 Yp5 Yp6 Ys "
    "Yt Yt1 Yt2 Yt3 Yt4 Yt5 Yt6 Yu1 Yu2 Yu3 Yu4 Yu5 Yu6 Yui1 Yui2 Yui3 Yui4 Yui5 Yui6 Yuk Yuk1 Yuk2 Yuk3 Yuk4 Yuk5 Yuk6 Yun1 Yun2 Yun3 Yun4 Yun5 Yun6 Yung1 Yung2 "
    "Yung3 Yung4 Yung5 Yung6 Yut1 Yut2 Yut3 Yut4 Yut5 Yut6 Yw Yyu1 Yyu2 Yyu3 Yyu4 Yyu5 Yyu6 Yyun1 Yyun2 Yyun3 Yyun4 Yyun5 Yyun6 Yyut1 Yyut2 Yyut3 Yyut4 Yyut5 Yyut6 "
    "Yz"
)
"""Cantonese phones (a set upstream)."""

SYMBOLS: list[str] = [*sorted({PAD, *_INITIALS, *_FINALS, *_JA, *_PU, *_ARPA}), "[", "]", *sorted(_KO), *sorted(_YUE)]
SYMBOL_SET = frozenset(SYMBOLS)
SYMBOL_TO_ID = {s: i for i, s in enumerate(SYMBOLS)}


def cleaned_text_to_sequence(phones: Iterable[str]) -> list[int]:
    """Phone symbols → ids (upstream ``text/__init__.py`` for every version but v1)."""
    return [SYMBOL_TO_ID[p] for p in phones]
