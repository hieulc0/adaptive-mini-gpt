"""LOCAL-SAFE: tiny fake records, no model/tensor training or benchmark."""

from types import SimpleNamespace

import pytest

from adaptive_mini_gpt.data.replay_a4 import replace_tail, validate_replay


def fake_records():
    statuses = ("positive", "negated", "absent")
    return [
        SimpleNamespace(
            question=f"old {i} {j}",
            statuses=(a, b),
            labels=(int(a == "positive"), int(b == "positive")),
        )
        for i, a in enumerate(statuses)
        for j, b in enumerate(statuses)
    ]


def test_replay_fails_closed_on_viewed_or_inconsistent_examples():
    rows = fake_records()
    assert len(validate_replay(rows, ["other"])) == 9
    with pytest.raises(ValueError, match="protected probe"):
        validate_replay(rows, ["OLD   1 2"])
    rows[0].labels = (0, 0)
    with pytest.raises(ValueError, match="annotation"):
        validate_replay(rows, [])


def test_matched_ratio_keeps_status_order_and_exact_budget():
    old = fake_records()
    lookup = {tuple(r.statuses): i for i, r in enumerate(old)}
    new = [old[i % 9] for i in range(18)]
    indices = list(range(18))
    for count in (0, 3, 6):
        result = replace_tail(indices, new, lookup, 18, count)
        assert len(result) == 18 and sum(i >= 18 for i in result) == count
        assert [new[i].statuses for i in indices] == [
            (old[i - 18] if i >= 18 else new[i]).statuses for i in result
        ]
    with pytest.raises(ValueError, match="budget"):
        replace_tail(indices, new, lookup, 18, 1)
