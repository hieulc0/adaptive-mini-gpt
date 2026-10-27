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


def test_declared_numerical_policy_requires_runtime_match():
    from types import SimpleNamespace

    from adaptive_mini_gpt.training.learning_state import (
        deterministic_numerical_policy,
        require_numerical_policy,
    )

    api = SimpleNamespace(
        are_deterministic_algorithms_enabled=lambda: True,
        backends=SimpleNamespace(cudnn=SimpleNamespace(benchmark=False)),
    )
    require_numerical_policy(
        deterministic_numerical_policy(),
        torch_api=api,
        environ={"CUBLAS_WORKSPACE_CONFIG": ":4096:8"},
    )


@pytest.mark.parametrize("enabled,workspace", [(False, ":4096:8"), (True, None)])
def test_declared_policy_rejects_silent_nondeterministic_resume(enabled, workspace):
    from types import SimpleNamespace

    from adaptive_mini_gpt.training.learning_state import (
        deterministic_numerical_policy,
        require_numerical_policy,
    )

    api = SimpleNamespace(
        are_deterministic_algorithms_enabled=lambda: enabled,
        backends=SimpleNamespace(cudnn=SimpleNamespace(benchmark=False)),
    )
    with pytest.raises(RuntimeError, match="numerical policy"):
        require_numerical_policy(
            deterministic_numerical_policy(),
            torch_api=api,
            environ={"CUBLAS_WORKSPACE_CONFIG": workspace},
        )
