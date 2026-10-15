"""Compare mizuoto.text tokenizers with transformers' on a corpus, id for id (tier T0).

    <engine env python> scripts/parity/check_tokenizer.py <model folder> [--corpus file.txt]

Runs in an engine environment, which has transformers. ``mizuoto.text`` is imported from this checkout
(it needs only ``tokenizers``). Each line of the corpus is encoded both ways, as is the whole corpus as
one text, and the result is reported. Exits non-zero on any difference.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CORPUS = ROOT / "scripts" / "parity" / "corpus" / "tokenizer.txt"


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("model")
    parser.add_argument("--corpus", default=str(CORPUS))
    parser.add_argument("--fix-mistral-regex", action="store_true", help="Load transformers' tokenizer as Qwen3-TTS's processor does")
    args = parser.parse_args(argv)
    sys.path.insert(0, str(ROOT))
    from transformers import AutoTokenizer

    from mizuoto.text.tokenize.bpe import load_bpe

    ref = AutoTokenizer.from_pretrained(args.model, **({"fix_mistral_regex": True} if args.fix_mistral_regex else {}))
    ours = load_bpe(Path(args.model), mistral_fix=args.fix_mistral_regex)
    lines = [line for line in Path(args.corpus).read_text(encoding="utf-8").split("\n") if line]
    texts = [*lines, "\n".join(lines)]
    bad = 0
    for text in texts:
        a = ref.encode(text, add_special_tokens=False)
        b = ours.encode(text)
        if a != b:
            bad += 1
            print(json.dumps({"text": text[:80], "transformers": a[:40], "mizuoto": b[:40]}, ensure_ascii=False))
        elif ref.decode(a) != ours.decode(b):
            bad += 1
            print(json.dumps({"text": text[:80], "decode_differs": True}, ensure_ascii=False))
    print(json.dumps({"model": args.model, "texts": len(texts), "differences": bad, "vocab": ours.vocab_size}))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
