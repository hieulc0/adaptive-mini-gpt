"""Finite Stage A.3 development TRAINING corpus; no sealed data or solver."""

from __future__ import annotations

import itertools
import random
from dataclasses import dataclass

from adaptive_mini_gpt.data.attribute_learning import AttributeCase

FRAMES = (
    "For {name}, status: {content}.",
    "{content}. This entry concerns {name}.",
    "Concerning {name}, we record {content}; entry complete.",
)
PREFIXES = ("", "A record. ")
NAMES = ("J", "K")
STATUSES = ("positive", "negated", "absent")
FACTORS = tuple(itertools.product(range(3), range(2), range(2), range(2)))


@dataclass(frozen=True)
class StableCase(AttributeCase):
    content: str
    name: str
    factors: tuple[int, int, int, int]


def case(statuses, factors):
    frame, prefix, reverse, name = factors
    if len(statuses) != 2 or any(s not in STATUSES for s in statuses):
        raise ValueError("Two declared statuses required")
    if factors not in FACTORS:
        raise ValueError("Undeclared development factors")
    parts = [
        w if s == "positive" else "not " + w
        for w, s in zip(("flammable", "conductive"), statuses, strict=True)
        if s != "absent"
    ]
    if reverse:
        parts.reverse()
    content = "; ".join(parts) if parts else "present"
    question = PREFIXES[prefix] + FRAMES[frame].format(name=NAMES[name], content=content)
    return StableCase(
        question,
        question,
        tuple(statuses),
        (int(statuses[0] == "positive"), int(statuses[1] == "positive")),
        f"frame_{frame}",
        content,
        NAMES[name],
        tuple(factors),
    )


def corpus():
    # Identical strings with irrelevant factor choices are deduplicated.
    rows = {}
    for statuses, factors in itertools.product(itertools.product(STATUSES, repeat=2), FACTORS):
        row = case(statuses, factors)
        rows.setdefault(row.question, row)
    return list(rows.values())


def streams(rows, seed, steps):
    """Exactly balanced24-step first views, bijective different-frame second views."""
    index = {r.question: i for i, r in enumerate(rows)}
    rng = random.Random(seed)
    result = {"fixed": [], "augmented": []}
    plans = []
    statuses = tuple(itertools.product(STATUSES, repeat=2))
    for step in range(steps):
        if step % 24 == 0:
            plans = [rng.sample(FACTORS, len(FACTORS)) for _ in statuses]
        order = rng.sample(range(9), 9)
        batch = {k: [] for k in result}
        for j in order:
            f, p, o, n = plans[j][step % 24]
            other = ((f + 1 + (step // 24) % 2) % 3, 1 - p, 1 - o, 1 - n)
            views = ((f, p, o, n), other)
            for v in views:
                batch["augmented"].append(index[case(statuses[j], v).question])
                batch["fixed"].append(index[case(statuses[j], (0, 0, 0, v[3])).question])
        for k in result:
            result[k].append(batch[k])
    return result


def minimal_pairs():
    rows = []
    factors = ((0, 0, 0, 0), (0, 0, 0, 1), (0, 1, 0, 0), (0, 0, 1, 0), (1, 0, 0, 0), (2, 0, 0, 0))
    for statuses in itertools.product(STATUSES, repeat=2):
        rows.extend(case(statuses, f) for f in factors)
    return rows
