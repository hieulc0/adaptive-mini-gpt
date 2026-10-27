"""Audited old-example replay scheduling; no student-side answer extraction."""

from __future__ import annotations

import itertools

from adaptive_mini_gpt.data.attribute_learning import AttributeCase
from adaptive_mini_gpt.data.stable_attributes import case

OLD_FIXTURE_SHA = "130dc05e42ed9ee0c1122dd95532a16562fc67f5ffa95422f099eff48643f1e5"
DEV_FRAMES = (
    "A note on {name} lists {content}.",
    "{name} is the subject; the listed description is {content}.",
    "These features belong to {name}: {content}.",
)


def normalize(text):
    return " ".join(text.casefold().split())


def validate_replay(rows, blocked):
    if len(rows) != 9:
        raise ValueError("Exactly nine historical training examples required")
    allowed = {"positive", "negated", "absent"}
    by_status = {}
    blocked = {normalize(q) for q in blocked}
    for row in rows:
        status = tuple(row.statuses)
        if len(status) != 2 or any(s not in allowed for s in status) or status in by_status:
            raise ValueError("Historical status coverage must be unique")
        if tuple(row.labels) != (int(status[0] == "positive"), int(status[1] == "positive")):
            raise ValueError("Historical annotation/label mismatch")
        if normalize(row.question) in blocked:
            raise ValueError("Historical replay overlaps a protected probe")
        by_status[status] = row
    if len({normalize(x.question) for x in rows}) != 9:
        raise ValueError("Duplicate historical replay description")
    return by_status


def replace_tail(indices, new_rows, old_index, old_offset, count):
    if count not in (0, 3, 6) or len(indices) != 18:
        raise ValueError("Undeclared replay budget")
    out = list(indices)
    for j in range(len(out) - count, len(out)):
        status = tuple(new_rows[out[j]].statuses)
        if status not in old_index:
            raise ValueError("Missing status-matched historical experience")
        out[j] = old_offset + old_index[status]
    return out


def development():
    """Fresh development families; created only remotely by experiment driver."""
    unique = {}
    for name, statuses, frame, reverse in itertools.product(
        ("J", "K"), itertools.product(("positive", "negated", "absent"), repeat=2), range(3), (0, 1)
    ):
        row = case(statuses, (0, 0, reverse, 0 if name == "J" else 1))
        question = DEV_FRAMES[frame].format(name=name, content=row.content)
        unique.setdefault(
            question, AttributeCase(question, question, row.statuses, row.labels, f"dev_{frame}")
        )
    return list(unique.values())
