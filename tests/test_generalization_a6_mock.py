"""LOCAL-SAFE: two string pairs and fake metadata; no corpus/model execution."""

from collections import Counter

import pytest

from adaptive_mini_gpt.data.generalization_a6 import matched_indices, render
from adaptive_mini_gpt.data.joint_acquisition_a5 import verify_observable


def test_same_word_scope_pair_changes_truth():
    a = render("Fake", ("negated", "positive"), 0)
    b = render("Fake", ("positive", "negated"), 0)
    assert Counter(a.question.replace(";", "").split()) == Counter(
        b.question.replace(";", "").split()
    )
    assert a.labels == (0, 1) and b.labels == (1, 0)
    verify_observable(a)
    verify_observable(b)


def test_position_and_grammar_pairs_preserve_meaning():
    a = render("Fake", ("negated", "absent"), 0)
    b = render("Fake", ("negated", "absent"), 4)
    c = render("Fake", ("negated", "absent"), 0, "A record. ")
    pairs = matched_indices([a, b, c])
    assert pairs["frame"] == [(0, 1)] and pairs["prefix"] == [(0, 2)]
    assert a.labels == b.labels == c.labels == (0, 0)
    with pytest.raises(ValueError):
        render("Fake", ("negated", "absent"), 99)
