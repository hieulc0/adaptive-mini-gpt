"""Tests for CharacterTokenizer."""

from __future__ import annotations

from pathlib import Path

from adaptive_mini_gpt.data.tokenizer import CharacterTokenizer


def test_tokenizer_roundtrip() -> None:
    text = "Hello, Adaptive Mini-GPT! 123 \n"
    tokenizer = CharacterTokenizer()
    tokenizer.fit(text)

    encoded = tokenizer.encode(text)
    decoded = tokenizer.decode(encoded)
    assert decoded == text


def test_tokenizer_unknown_character() -> None:
    tokenizer = CharacterTokenizer()
    tokenizer.fit("abc")

    # 'z' is not in vocabulary, should map to <unk>
    encoded = tokenizer.encode("abcz")
    decoded = tokenizer.decode(encoded)
    assert "<unk>" in decoded


def test_tokenizer_save_load_fingerprint(tmp_path: Path) -> None:
    text = "To be, or not to be, that is the question."
    tok1 = CharacterTokenizer().fit(text)
    fp1 = tok1.get_fingerprint()

    save_path = tmp_path / "vocab.json"
    tok1.save(save_path)

    tok2 = CharacterTokenizer.load(save_path)
    fp2 = tok2.get_fingerprint()

    assert fp1 == fp2
    assert tok1.encode(text) == tok2.encode(text)
