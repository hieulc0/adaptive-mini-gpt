"""A6 development pairs and metadata. Never used as a student-side extractor."""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Any

from adaptive_mini_gpt.data.attribute_learning import AttributeCase
from adaptive_mini_gpt.data.joint_acquisition_a5 import STATUSES, verify_observable

DIAGNOSTIC_FRAMES = (
    "{name} has properties: {content}.",
    "Description of {name}: {content}.",
    "For {name}, status: {content}.",
    "{content}. This entry concerns {name}.",
    "A note on {name} lists {content}.",
    "The following description applies to {name}: {content}.",
)


@dataclass(frozen=True)
class GeneralizationCase(AttributeCase):
    name: str
    content: str
    frame: int
    prefix: str
    # Canonical property states identify matched meaning, never model input.


def render(name, statuses, frame, prefix="", reverse=False, negator="not"):
    if frame not in range(len(DIAGNOSTIC_FRAMES)) or statuses not in STATUSES:
        raise ValueError("Undeclared diagnostic factor")
    if negator not in ("not", "without"):
        raise ValueError("Undeclared negation")
    parts = [
        word if status == "positive" else f"{negator} {word}"
        for word, status in zip(("flammable", "conductive"), statuses, strict=True)
        if status != "absent"
    ]
    if reverse:
        parts.reverse()
    content = "; ".join(parts) if parts else "present"
    question = prefix + DIAGNOSTIC_FRAMES[frame].format(name=name, content=content)
    row = GeneralizationCase(
        question,
        question,
        tuple(statuses),
        (int(statuses[0] == "positive"), int(statuses[1] == "positive")),
        f"frame_{frame}",
        name,
        content,
        frame,
        prefix,
    )
    verify_observable(row)
    return row


def diagnostic(spec: dict[str, Any]):
    """Remote-only scientific generation. Already viewed after A6.1, never training."""
    unique = {}
    for name, status, frame, prefix, reverse, negator in itertools.product(
        spec["names"],
        STATUSES,
        range(len(DIAGNOSTIC_FRAMES)),
        spec["prefixes"],
        (False, True),
        ("not", "without"),
    ):
        row = render(name, status, frame, prefix, reverse, negator)
        unique.setdefault(row.question, row)
    return list(unique.values())


def matched_indices(rows):
    index = {(r.name, r.content, r.frame, r.prefix): i for i, r in enumerate(rows)}
    pairs = {}
    for i, row in enumerate(rows):
        base = index[(row.name, row.content, 0, "")]
        pairs.setdefault("frame", [])
        if row.prefix == "" and row.frame != 0:
            pairs["frame"].append((base, i))
        if row.frame == 0 and row.prefix:
            pairs.setdefault("prefix", []).append((base, i))
        if "; " in row.content:
            reverse = "; ".join(reversed(row.content.split("; ")))
            j = index[(row.name, reverse, row.frame, row.prefix)]
            if i < j:
                pairs.setdefault("order", []).append((i, j))
    return pairs
