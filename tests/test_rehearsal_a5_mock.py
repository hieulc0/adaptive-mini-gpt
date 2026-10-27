"""LOCAL-SAFE: tiny synthetic fake schedules and scalar gates, no neural calls."""

from collections import Counter
from types import SimpleNamespace
from typing import Any

from adaptive_mini_gpt.data.joint_acquisition_a5 import STATUSES
from adaptive_mini_gpt.data.rehearsal_a5 import SYNTHETIC_NAMES, plans, stable_old


def test_matched_rehearsal_slots_and_sources():
    new = [
        SimpleNamespace(frame=f, statuses=s, name="fake", content=str(s))
        for f in range(2, 5)
        for s in STATUSES
    ]
    historical = [SimpleNamespace(statuses=s) for s in STATUSES]
    synthetic = [
        SimpleNamespace(frame=f, statuses=s, name=n, content=str(s))
        for f in range(2)
        for s in STATUSES
        for n in SYNTHETIC_NAMES
    ]
    rows = new + historical + synthetic
    streams = plans(new, historical, synthetic, 42, 6)
    for arm, batches in streams.items():
        for step, ix in enumerate(batches):
            assert len(ix) == 18
            assert Counter(rows[i].statuses for i in ix) == Counter({s: 2 for s in STATUSES})
            assert sum(i >= len(new) for i in ix) == (
                0 if arm == "none" else 9 if arm == "balanced_9" else 6
            )
            for j, i in enumerate(ix):
                if i < len(new):
                    assert i == streams["none"][step][j]
                elif arm == "historical_6":
                    assert i < len(new) + len(historical)
                else:
                    assert i >= len(new) + len(historical)
    assert streams == plans(new, historical, synthetic, 42, 6)


def test_stable_preservation_requires_each_late_cell():
    good: list[dict[str, Any]] = [
        dict(updates=s, retention={k: dict(exact=1.0) for k in ("old_known", "old_renamed")})
        for s in (1000, 1100, 1200)
    ]
    assert stable_old(good)
    assert not stable_old(good[1:])
    good[0]["retention"]["old_known"]["exact"] = 0.94
    assert not stable_old(good)
