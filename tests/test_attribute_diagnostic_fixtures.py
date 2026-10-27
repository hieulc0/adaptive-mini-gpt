"""LOCAL-SAFE: tiny string fixtures only; no corpus generation or neural execution."""

import pytest

from adaptive_mini_gpt.data.attribute_diagnostic import render
from adaptive_mini_gpt.natural_language_learning_m103.tokenizer_m103 import ByteTextTokenizerM103


def test_factorial_changes_one_surface_factor_at_a_time():
    content = "not flammable; conductive"
    same = render("J", content, 0)
    name = render("Y", content, 0)
    syntax = render("J", content, 2)
    both = render("Y", content, 2)
    assert same.replace("J", "Y") == name
    assert syntax.replace("J", "Y") == both
    assert all(content in s for s in (same, name, syntax, both))
    with pytest.raises(ValueError):
        render("J", content, 4)


def test_development_description_roundtrip():
    tok = ByteTextTokenizerM103()
    text = render("Y", "flammable; without conductive", 3)
    ids = tok.encode_prompt(text, "Only affirmed properties hold.", max_length=256)
    assert tok.decode(ids[ids.index(tok.QUESTION_ID) + 1 : -1]) == text
