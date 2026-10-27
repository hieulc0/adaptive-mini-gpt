"""LOCAL-SAFE: tiny pure-text fixtures, no corpus generation or neural execution."""

import random

import pytest

from adaptive_mini_gpt.data.attribute_learning import RULES
from adaptive_mini_gpt.data.frame_curriculum import (
    ARMS,
    DEV_FRAMES,
    SEALED_FRAMES,
    TRAIN_FRAMES,
    draw_factors,
    evaluation_sets,
    training_case,
)
from adaptive_mini_gpt.natural_language_learning_m103.tokenizer_m103 import ByteTextTokenizerM103


def test_paired_factor_streams_keep_labels_and_names_fixed_across_arms():
    rng = random.Random(42)
    for _ in range(3):
        factors = draw_factors(rng)
        rows = [training_case(factors, a) for a in ARMS]
        assert len({r.name for r in rows}) == 1
        assert len({r.statuses for r in rows}) == 1
        assert len({r.labels for r in rows}) == 1


def test_prefix_order_variation_preserves_annotated_properties():
    f = (0, 1, 0, 0, 1, 0, 1, 3)
    plain = training_case(f, "multi")
    varied = training_case(f, "multi_both")
    assert plain.content == "flammable; without conductive"
    assert varied.content == "without conductive; flammable"
    assert varied.prefix and plain.labels == varied.labels == (1, 0)
    tok = ByteTextTokenizerM103()
    ids = tok.encode_prompt(varied.question, RULES, max_length=256)
    assert tok.decode(ids[ids.index(tok.QUESTION_ID) + 1 : -1]) == varied.question


def test_sealed_guard_prevents_generation_and_grammar_families_are_disjoint():
    assert not set(TRAIN_FRAMES) & set(DEV_FRAMES)
    assert not set(TRAIN_FRAMES) & set(SEALED_FRAMES)
    assert not set(DEV_FRAMES) & set(SEALED_FRAMES)
    with pytest.raises(RuntimeError, match="not generated"):
        evaluation_sets(sealed=True)
