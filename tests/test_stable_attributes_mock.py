"""LOCAL-SAFE: tiny string/fake validation only; no neural/tensor execution."""

import pytest

from adaptive_mini_gpt.data.stable_attributes import case
from adaptive_mini_gpt.training.learning_state import validate_consistency_pairs


def test_frame_pair_changes_surface_but_preserves_observed_facts():
    a = case(("positive", "negated"), (0, 0, 0, 0))
    b = case(("positive", "negated"), (1, 1, 1, 1))
    assert a.labels == b.labels == (1, 0)
    assert a.content == "flammable; not conductive"
    assert b.content == "not conductive; flammable"
    assert a.name != b.name and a.question != b.question
    assert case(("absent", "absent"), (0, 0, 0, 0)).content == "present"


@pytest.mark.parametrize("pairs", [((0, 0),), ((0, 2),), ((False, 1),), ((0, 1), (1, 0)), ()])
def test_consistency_indices_reject_ambiguous_or_invalid_pairing(pairs):
    with pytest.raises(ValueError, match="pair"):
        validate_consistency_pairs(pairs, 2)


def test_consistency_pairs_are_explicit_disjoint_indices():
    assert validate_consistency_pairs(((0, 1), (2, 3)), 4) == ((0, 1), (2, 3))
