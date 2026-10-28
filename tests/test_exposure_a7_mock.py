"""LOCAL-SAFE: tiny fake index/string/scalar fixtures; no neural operations."""

from types import SimpleNamespace
from typing import Any

import pytest

from adaptive_mini_gpt.data.exposure_a7 import factor_subset, repeated_plan, window_pass


def test_repeated_exposure_preserves_block_and_avoids_aliasing():
    base = [[0, 1], [1, 0]]
    assert repeated_plan(base, 6) == base * 3
    copy = repeated_plan(base, 4)
    copy[0][0] = 99
    assert base[0][0] == 0 and copy[2][0] == 0
    with pytest.raises(ValueError):
        repeated_plan(base, 3)


def test_factor_projection_preserves_every_non_name_factor():
    def row(name):
        return SimpleNamespace(
            name=name,
            question=f"Log. {name}: not flammable.",
            content="not flammable",
            statuses=("negated", "absent"),
            frame=2,
            prefix="Log. ",
        )

    old = SimpleNamespace(question="old")
    small, n, mapping = factor_subset([row("Rax"), row("Vim"), old], 2)
    assert n == 1 and mapping == [0, 0, 1] and small[-1] is old
    bad = row("Vim")
    bad.question += " "
    with pytest.raises((KeyError, ValueError)):
        factor_subset([row("Rax"), bad, old], 2)


def test_late_window_does_not_rewrite_original_gate():
    metric = dict(
        exact=1.0,
        family_exact={"f": 1.0},
        status_accuracy={s: 1.0 for s in ("positive", "negated", "absent")},
    )
    curves: list[dict[str, Any]] = [
        dict(
            updates=i,
            training=metric,
            retention={k: dict(exact=1.0) for k in ("old_known", "old_renamed")},
        )
        for i in (7000, 7100, 7200)
    ]
    assert window_pass(curves, 7200, ["f"]) == dict(acquisition=True, preservation=True)
    assert window_pass(curves, 1200, ["f"]) == dict(acquisition=False, preservation=False)


def test_average_exposure_and_extra_old_dose_are_explicit():
    from fractions import Fraction

    assert Fraction(1200 * 12, 126) == Fraction(7200 * 12, 756)
    assert Fraction(1200 * 12, 378) == Fraction(2400 * 12, 756)
    assert 7200 * 6 == 6 * (1200 * 6)  # rehearsal grows too; no pure new-exposure attribution
