"""English tokenisation and part-of-speech tagging as g2p_en calls them, without nltk.

- ``tweet_tokenize``: nltk 3.10.3's ``TweetTokenizer().tokenize`` (``nltk/tokenize/casual.py``, Apache-2.0, (c) NLTK
  Project, author Christopher Potts), its regular expressions copied verbatim (they change between nltk releases).
- ``PerceptronTagger``: nltk's averaged-perceptron tagger at inference (``nltk/tag/perceptron.py``, after Matthew
  Honnibal), reading the ``averaged_perceptron_tagger_eng`` JSON files from the ``gptsovits/nltk-data`` asset, as
  ``nltk.pos_tag`` does in nltk ≥ 3.9. nltk is not imported: g2p_en's import calls ``nltk.download``, and nltk finds
  its data through the process-wide ``nltk.data.path``.
"""

from __future__ import annotations

import html.entities
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import regex

# -- nltk/tokenize/casual.py (3.10.3) ----------------------------------------------------------------------------------

EMOTICONS = r"""
    (?:
      [<>]?
      [:;=8]                     # eyes
      [\-o\*\']?                 # optional nose
      [\)\]\(\[dDpP/\:\}\{@\|\\] # mouth
      |
      [\)\]\(\[dDpP/\:\}\{@\|\\] # mouth
      [\-o\*\']?                 # optional nose
      [:;=8]                     # eyes
      [<>]?
      |
      </?3                       # heart
    )"""

# URL pattern due to John Gruber, modified by Tom Winzig. See
# https://gist.github.com/winzig/8894715

URLS = r"""			# Capture 1: entire matched URL
  (?:
  https?:				# URL protocol and colon
    (?:
      /{1,3}				# 1-3 slashes
      |					#   or
      [a-z0-9%]				# Single letter or digit or '%'
                                       # (Trying not to match e.g. "URI::Escape")
    )
    |					#   or
                                       # looks like domain name followed by a slash:
    [a-z0-9.\-]{1,255}[.]              # bounded to avoid catastrophic backtracking (ReDoS)
    (?:[a-z]{2,13})
    /
  )
  (?:					# One or more:
    [^\s()<>{}\[\]]+			# Run of non-space, non-()<>{}[]
    |					#   or
    \([^\s()]{0,255}?\([^\s()]{1,255}\)[^\s()]{0,255}?\) # balanced parens, one level deep: (...(...)...)
    |
    \([^\s]{1,255}?\)				# balanced parens, non-recursive: (...)
  )+
  (?:					# End with:
    \([^\s()]{0,255}?\([^\s()]{1,255}\)[^\s()]{0,255}?\) # balanced parens, one level deep: (...(...)...)
    |
    \([^\s]{1,255}?\)				# balanced parens, non-recursive: (...)
    |					#   or
    [^\s`!()\[\]{};:'".,<>?«»“”‘’]	# not a space or one of these punct chars
  )
  |					# OR, the following to match naked domains:
  (?:
  	(?<!@)			        # not preceded by a @, avoid matching foo@_gmail.com_
    [a-z0-9]+
    (?:[.\-][a-z0-9]+){0,126}  # bounded to avoid catastrophic backtracking (ReDoS); DNS allows at most 127 labels
    [.]
    (?:[a-z]{2,13})
    \b
    /?
    (?!@)			        # not succeeded by a @,
                            # avoid matching "foo.na" in "foo.na@example.com"
  )
"""

# emoji flag sequence
# https://en.wikipedia.org/wiki/Regional_indicator_symbol
# For regex simplicity, include all possible enclosed letter pairs,
# not the ISO subset of two-letter regional indicator symbols.
# See https://en.wikipedia.org/wiki/ISO_3166-1_alpha-2#Current_codes
# Future regional flag support may be handled with the regex for
# U+1F3F4 🏴 followed by emoji tag sequences:
# r'\U0001F3F4[\U000E0000-\U000E007E]{5}\U000E007F'
FLAGS = r"""
  (?:
    [\U0001F1E6-\U0001F1FF]{2}  # all enclosed letter pairs
    |
    # English flag
    \U0001F3F4\U000E0067\U000E0062\U000E0065\U000E006e\U000E0067\U000E007F
    |
    # Scottish flag
    \U0001F3F4\U000E0067\U000E0062\U000E0073\U000E0063\U000E0074\U000E007F
    |
    # For Wales? Why Richard, it profit a man nothing to give his soul for the whole world … but for Wales!
    \U0001F3F4\U000E0067\U000E0062\U000E0077\U000E006C\U000E0073\U000E007F
  )
"""

# Regex for recognizing phone numbers:
PHONE_REGEX = r"""
    (?:
      (?:            # (international)
        \+?[01]
        [ *\-.\)]*
      )?
      (?:            # (area code)
        [\(]?
        \d{3}
        [ *\-.\)]*
      )?
      \d{3}          # exchange
      [ *\-.\)]*
      \d{4}          # base
    )"""

# The components of the tokenizer:
REGEXPS = (
    URLS,
    # ASCII Emoticons
    EMOTICONS,
    # HTML tags:
    r"""<[^>\s]+>""",
    # ASCII Arrows
    r"""[\-]+>|<[\-]+""",
    # Twitter username:
    r"""(?:@[\w_]+)""",
    # Twitter hashtags:
    r"""(?:\#+[\w_]+[\w\'_\-]*[\w_]+)""",
    # email addresses
    r"""[\w.+-]{1,64}@[\w-]{1,63}\.(?:[\w-]\.?){1,251}[\w-]""",
    # Zero-Width-Joiner and Skin tone modifier emojis
    """.(?:
        [\U0001f3fb-\U0001f3ff]?(?:\u200d.[\U0001f3fb-\U0001f3ff]?)+
        |
        [\U0001f3fb-\U0001f3ff]
    )""",
    # flags
    FLAGS,
    # Remaining word types:
    r"""
    (?:[^\W\d_](?:[^\W\d_]|['\-_])+[^\W\d_]) # Words with apostrophes or dashes.
    |
    (?:[+\-]?\d+[,/.:-]\d+[+\-]?)  # Numbers, including fractions, decimals.
    |
    (?:[\w_]+)                     # Words without apostrophes or dashes.
    |
    (?:\.(?:\s*\.){1,})            # Ellipsis dots.
    |
    (?:\S)                         # Everything else that isn't whitespace.
    """,
)

# Take the main components and add a phone regex as the second parameter
REGEXPS_PHONE = (REGEXPS[0], PHONE_REGEX, *REGEXPS[1:])

######################################################################
# TweetTokenizer.WORD_RE and TweetTokenizer.PHONE_WORD_RE represent
# the core tokenizing regexes. They are compiled lazily.

# WORD_RE performs poorly on these patterns:
HANG_RE = regex.compile(r"([^\p{L}\p{N}])\1{3,}")
ENT_RE = regex.compile(r"&(#?(x?))([^&;\s]+);")
PHONE_WORD_RE = regex.compile(f"({'|'.join(REGEXPS_PHONE)})", regex.VERBOSE | regex.I | regex.UNICODE)


def _replace_html_entities(text: str) -> str:
    """``_replace_html_entities(text)`` with its defaults: entities → characters, unknown ones removed."""

    def convert(match: Any) -> str:
        entity_body = match.group(3)
        number: int | None
        if match.group(1):
            try:
                number = int(entity_body, 16) if match.group(2) else int(entity_body, 10)
                if 0x80 <= number <= 0x9F:
                    return bytes((number,)).decode("cp1252")
            except ValueError:
                number = None
        else:
            number = html.entities.name2codepoint.get(entity_body)
        if number is not None:
            try:
                return chr(number)
            except (ValueError, OverflowError):
                pass
        return ""

    return ENT_RE.sub(convert, text)


def tweet_tokenize(text: str) -> list[str]:
    """``TweetTokenizer()`` (case kept, no length reduction, handles kept, phone numbers matched)."""
    text = _replace_html_entities(text)
    safe_text = HANG_RE.sub(r"\1\1\1", text)
    return PHONE_WORD_RE.findall(safe_text)


# -- nltk/tag/perceptron.py --------------------------------------------------------------------------------------------


class PerceptronTagger:
    """``nltk.tag.PerceptronTagger(lang="eng")`` at inference, from a folder of ``averaged_perceptron_tagger_eng.*.json``."""

    START = ["-START-", "-START2-"]
    END = ["-END-", "-END2-"]

    def __init__(self, folder: str | Path, lang: str = "eng") -> None:
        folder = Path(folder)

        def load(attr: str) -> object:
            return json.loads((folder / f"averaged_perceptron_tagger_{lang}.{attr}.json").read_text(encoding="utf-8"))

        weights, tagdict, classes = load("weights"), load("tagdict"), load("classes")
        assert isinstance(weights, dict) and isinstance(tagdict, dict) and isinstance(classes, list)
        self.weights: dict[str, dict[str, float]] = weights
        self.tagdict: dict[str, str] = tagdict
        self.classes = set(classes)

    @staticmethod
    def normalize(word: str) -> str:
        if "-" in word and word[0] != "-":
            return "!HYPHEN"
        if word.isdigit() and len(word) == 4:
            return "!YEAR"
        if word and word[0].isdigit():
            return "!DIGITS"
        return word.lower()

    def _get_features(self, i: int, word: str, context: list[str], prev: str, prev2: str) -> dict[str, int]:
        features: defaultdict[str, int] = defaultdict(int)

        def add(name: str, *args: str) -> None:
            features[" ".join((name, *args))] += 1

        i += len(self.START)
        add("bias")
        add("i suffix", word[-3:])
        add("i pref1", word[0] if word else "")
        add("i-1 tag", prev)
        add("i-2 tag", prev2)
        add("i tag+i-2 tag", prev, prev2)
        add("i word", context[i])
        add("i-1 tag+i word", prev, context[i])
        add("i-1 word", context[i - 1])
        add("i-1 suffix", context[i - 1][-3:])
        add("i-2 word", context[i - 2])
        add("i+1 word", context[i + 1])
        add("i+1 suffix", context[i + 1][-3:])
        add("i+2 word", context[i + 2])
        return features

    def _predict(self, features: dict[str, int]) -> str:
        scores: defaultdict[str, float] = defaultdict(float)
        for feat, value in features.items():
            if feat not in self.weights or value == 0:
                continue
            for label, weight in self.weights[feat].items():
                scores[label] += value * weight
        return max(self.classes, key=lambda label: (scores[label], label))

    def tag(self, tokens: list[str]) -> list[tuple[str, str]]:
        prev, prev2 = self.START
        output: list[tuple[str, str]] = []
        context = self.START + [self.normalize(w) for w in tokens] + self.END
        for i, word in enumerate(tokens):
            tag = self.tagdict.get(word)
            if not tag:
                tag = self._predict(self._get_features(i, word, context, prev, prev2))
            output.append((word, tag))
            prev2 = prev
            prev = tag
        return output
