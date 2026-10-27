"""LOCAL-SAFE: two string pairs and fake metadata; no corpus/model execution."""

from collections import Counter
from typing import Any

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


def test_coverage_schedule_balances_new_cells_independent_of_old_slots():
    from types import SimpleNamespace

    from adaptive_mini_gpt.data.generalization_a6 import (
        COVERAGE_PREFIXES,
        coverage_plan,
    )

    status = ("positive", "negated")
    new = [SimpleNamespace(statuses=status)]
    broad = [
        SimpleNamespace(statuses=status, frame=f, prefix=p)
        for f in range(6)
        for p in COVERAGE_PREFIXES
    ]
    # One old index interspersed with two new slots; only new presentations advance cursor.
    stream = coverage_plan([[0, 1, 0] for _ in range(9)], new, broad, 1)
    assert all(batch[1] == len(broad) for batch in stream)
    assert Counter(i for batch in stream for i in batch if i < len(broad)) == Counter(
        {i: 1 for i in range(18)}
    )


def test_stable_acquisition_keeps_single_family_failure():
    import copy

    from adaptive_mini_gpt.data.generalization_a6 import stable_acquisition

    metric = dict(
        exact=1.0,
        family_exact={"f": 1.0},
        status_accuracy={k: 1.0 for k in ("positive", "negated", "absent")},
    )
    curves: list[dict[str, Any]] = [
        dict(updates=s, training=copy.deepcopy(metric)) for s in (1000, 1100, 1200)
    ]
    assert stable_acquisition(curves, ["f"])
    curves[1]["training"]["family_exact"]["f"] = 41 / 42
    assert not stable_acquisition(curves, ["f"])
    curves[1]["training"]["family_exact"]["f"] = float("nan")
    assert not stable_acquisition(curves, ["f"])


def test_structured_diagnostic_preserves_clause_binding_without_label_access():
    from types import SimpleNamespace

    from adaptive_mini_gpt.data.generalization_a6 import structured_features

    rows = [
        SimpleNamespace(content=c)
        for c in ("not flammable; conductive", "flammable; not conductive", "present")
    ]
    a, b, c = structured_features(rows)
    assert a != b and sum(a) == sum(b) == sum(c) == 2
    assert [i for i, v in enumerate(a) if v] == [1, 8]
    assert [i for i, v in enumerate(b) if v] == [0, 9]
    assert [i for i, v in enumerate(c) if v] == [4, 10]
