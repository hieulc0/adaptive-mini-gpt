"""LOCAL-SAFE: tiny strings/token IDs only, no neural execution or full corpus."""

from collections import Counter

import pytest

from adaptive_mini_gpt.data.attribute_learning import RULES, case
from adaptive_mini_gpt.natural_language_learning_m103.tokenizer_m103 import ByteTextTokenizerM103


def test_negation_binding_changes_labels_without_changing_word_counts():
    a = case("A", ("positive", "negated"))
    b = case("A", ("negated", "positive"))
    assert Counter(a.question.split()) == Counter(b.question.split())
    assert a.labels == (1, 0) and b.labels == (0, 1)


def test_absence_is_not_a_positive_or_an_invented_state_label():
    row = case("Q", ("absent", "negated"), template=3, negation="without")
    assert "flammable" not in row.question
    assert "without conductive" in row.question
    assert row.labels == (0, 0)
    assert not hasattr(row, "state") and not hasattr(row, "action_id")
    with pytest.raises(ValueError):
        case("Q", ("unknown", "positive"))


def test_complete_input_and_context_boundary():
    row = case("Q", ("positive", "negated"), template=2)
    tokenizer = ByteTextTokenizerM103()
    tokens = tokenizer.encode_prompt(row.question, RULES)
    assert tokenizer.decode(tokens[tokens.index(tokenizer.QUESTION_ID) + 1 : -1]) == row.question
    assert (
        tokenizer.decode(
            tokens[tokens.index(tokenizer.RULE_ID) + 1 : tokens.index(tokenizer.SEP_ID)]
        )
        == RULES
    )
    assert tokenizer.encode_prompt(row.question, RULES, max_length=len(tokens)) == tokens
    with pytest.raises(ValueError):
        tokenizer.encode_prompt(row.question, RULES, max_length=len(tokens) - 1)
