"""LOCAL-SAFE: scalar gates and tiny fake/string fixtures; no model execution."""

import copy
from types import SimpleNamespace
from typing import Any

import pytest

from adaptive_mini_gpt.data.joint_acquisition_a5 import (
    STATUSES,
    plan,
    stable_joint,
    validate_disjoint,
    verify_observable,
)


def test_observable_labels_and_protected_boundary():
    row = SimpleNamespace(
        question="Rax has properties: not flammable; conductive.",
        statuses=("negated", "positive"),
        labels=(0, 1),
    )
    verify_observable(row)
    validate_disjoint([row], ["other"])
    with pytest.raises(ValueError, match="protected"):
        validate_disjoint([row], ["RAX has properties:  not flammable; conductive."])
    row.labels = (1, 1)
    with pytest.raises(ValueError, match="observable"):
        verify_observable(row)


def test_tiny_fake_plan_pairs_and_balances_families():
    rows = [
        SimpleNamespace(frame=f, statuses=s, name="fake", content=str(s))
        for f in range(5)
        for s in STATUSES
    ]
    batches = plan(rows, 42, 6)
    counts = {}
    for batch in batches:
        assert len(batch) == 18
        for left, right in zip(batch[::2], batch[1::2], strict=True):
            a, b = rows[left], rows[right]
            assert a.statuses == b.statuses and a.content == b.content
            assert a.frame < 2 <= b.frame
            for r in (a, b):
                key = (r.frame, r.statuses)
                counts[key] = counts.get(key, 0) + 1
    assert all(n == (3 if f < 2 else 2) for (f, _), n in counts.items())
    assert batches == plan(rows, 42, 6)


def test_stable_gate_requires_all_predeclared_endpoints():
    metric = dict(
        exact=1.0,
        family_exact={"frame": 1.0},
        status_accuracy={s: 1.0 for s in ("positive", "negated", "absent")},
    )
    curves: list[dict[str, Any]] = [
        dict(updates=s, training=dict(old=copy.deepcopy(metric), new=copy.deepcopy(metric)))
        for s in (1000, 1100, 1200)
    ]
    for c in curves:
        for group in ("old", "new"):
            c["training"][group]["family_exact"] = {
                f"frame_{i}": 1.0 for i in (range(2) if group == "old" else range(2, 5))
            }
    assert stable_joint(curves)
    assert not stable_joint(curves[-2:])
    missing_frame = copy.deepcopy(curves)
    missing_frame[0]["training"]["new"]["family_exact"].pop("frame_4")
    assert not stable_joint(missing_frame)
    curves[0]["training"]["new"]["exact"] = float("nan")
    assert not stable_joint(curves)
