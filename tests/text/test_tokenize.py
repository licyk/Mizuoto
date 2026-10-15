"""mizuoto.text.tokenize. Real models are compared with transformers by
scripts/parity/check_tokenizer.py; here a tiny vocabulary checks the construction."""

import json

import pytest

pytest.importorskip("tokenizers")

from mizuoto.text.tokenize.bpe import load_bpe, read_merges


def test_merges_skip_the_version_line(tmp_path):
    (tmp_path / "merges.txt").write_text("#version: 0.2\nĠ t\nĠt he\n\n", encoding="utf-8")
    assert read_merges(tmp_path / "merges.txt") == [("Ġ", "t"), ("Ġt", "he")]


def test_qwen2_files_become_a_tokenizer_with_its_added_tokens(tmp_path):
    letters = {c: i for i, c in enumerate("abcdefghijklmnopqrstuvwxyzĠ")}
    vocab = {**letters, "Ġt": 27, "he": 28, "Ġthe": 29}
    (tmp_path / "vocab.json").write_text(json.dumps(vocab), encoding="utf-8")
    (tmp_path / "merges.txt").write_text("#version: 0.2\nĠ t\nh e\nĠt he\n", encoding="utf-8")
    added = {"30": {"content": "<|im_end|>", "special": True, "normalized": False, "lstrip": False, "rstrip": False, "single_word": False}}
    (tmp_path / "tokenizer_config.json").write_text(json.dumps({"added_tokens_decoder": added}), encoding="utf-8")
    tok = load_bpe(tmp_path)
    ids = tok.encode("a the<|im_end|>")
    assert ids == [0, 29, 30]
    assert tok.decode(ids) == "a the<|im_end|>"
    assert tok.token_to_id("<|im_end|>") == 30
