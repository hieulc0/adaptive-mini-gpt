"""Stage A research annotations, never a student-side inference solver."""

from __future__ import annotations

import hashlib
import itertools
import json
from dataclasses import asdict, dataclass

TRAITS = ("flammable", "conductive")
TAG_IDS = (10, 15)
RULES = "Only affirmed properties hold. Omitted or negated properties do not hold."


@dataclass(frozen=True)
class AttributeCase:
    case_id: str
    question: str
    statuses: tuple[str, str]
    labels: tuple[int, int]
    family: str


def case(name, statuses, *, template=0, reverse=False, negation="not", case_id="fixture"):
    if len(statuses) != 2 or any(s not in ("positive", "negated", "absent") for s in statuses):
        raise ValueError("Two explicit status annotations required")
    if template not in range(4) or negation not in ("not", "without"):
        raise ValueError("Unknown description grammar")
    clauses = [
        word if status == "positive" else f"{negation} {word}"
        for word, status in zip(TRAITS, statuses, strict=True)
        if status != "absent"
    ]
    if reverse:
        clauses.reverse()
    description = "; ".join(clauses) if clauses else "present"
    if template == 0:
        text = f"Object {name}: {description}."
    elif template == 1:
        text = f"{name} is described as {description}."
    elif template == 2:
        text = f"For object {name}, the description is: {description}."
    else:
        text = f"The properties described for {name} are: {description}."
    return AttributeCase(
        case_id,
        text,
        tuple(statuses),
        (int(statuses[0] == "positive"), int(statuses[1] == "positive")),
        "+".join(statuses),
    )


def generate():
    """Exhaustive finite grammar, no stochastic label generation or oracle input."""
    splits = {}
    for split, names, templates in (
        ("train", ("A", "B", "C", "D"), (0, 1)),
        ("validation", ("E", "F"), (0, 1)),
        ("names", ("U", "V"), (0, 1)),
        ("syntax", ("W", "X"), (2, 3)),
    ):
        unique = {}
        for name, statuses, template, reverse, negation in itertools.product(
            names,
            itertools.product(("positive", "negated", "absent"), repeat=2),
            templates,
            (False, True),
            ("not", "without"),
        ):
            row = case(name, statuses, template=template, reverse=reverse, negation=negation)
            unique.setdefault(row.question, row)
        splits[split] = [
            AttributeCase(f"{split}-{i}", r.question, r.statuses, r.labels, r.family)
            for i, r in enumerate(unique.values())
        ]
    # One example of every annotation combination. Subset of training, never held out.
    splits["tiny"] = []
    for family in sorted({r.family for r in splits["train"]}):
        splits["tiny"].append(next(r for r in splits["train"] if r.family == family))
    return splits


def manifest(splits):
    primary = {s: rows for s, rows in splits.items() if s != "tiny"}
    seen = set()
    for rows in primary.values():
        prompts = {r.question for r in rows}
        if len(prompts) != len(rows) or seen & prompts:
            raise ValueError("Description leakage")
        seen |= prompts
    payload = {s: [asdict(r) for r in rows] for s, rows in splits.items()}
    return {
        "protocol": "attribute-learning-v1",
        "data_sha256": hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest(),
        "splits": {
            s: {
                "cases": len(rows),
                "positives": [sum(r.labels[i] for r in rows) for i in range(2)],
                "families": {
                    f: sum(r.family == f for r in rows) for f in sorted({r.family for r in rows})
                },
            }
            for s, rows in splits.items()
        },
        "boundary": "names: disjoint names, known grammar; syntax: disjoint names and templates; all nine annotation combinations in train. Not composition or unfamiliar-vocabulary transfer.",
    }
