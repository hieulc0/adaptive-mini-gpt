"""Development training data, annotations and schedules; no student-side solver."""

from __future__ import annotations

import itertools
import random
import re
from dataclasses import dataclass

from adaptive_mini_gpt.data.attribute_learning import AttributeCase
from adaptive_mini_gpt.data.stable_attributes import FRAMES

NAMES = ("Rax", "Vim")
STATUSES = tuple(itertools.product(("positive", "negated", "absent"), repeat=2))
GRAMMARS = ("{name} has properties: {content}.", "Description of {name}: {content}.", *FRAMES)


@dataclass(frozen=True)
class JointCase(AttributeCase):
    group: str
    frame: int
    name: str
    content: str


def normalized(question):
    return " ".join(question.casefold().split())


def verify_observable(row):
    """Independent dataset check only; this is never a model input or predictor."""
    found = {}
    for negator, word in re.findall(r"\b(?:(not|without) )?(flammable|conductive)\b", row.question):
        if word in found:
            raise ValueError("Duplicate/contradictory property annotation")
        found[word] = "negated" if negator else "positive"
    statuses = tuple(found.get(w, "absent") for w in ("flammable", "conductive"))
    labels = tuple(int(s == "positive") for s in statuses)
    if tuple(row.statuses) != statuses or tuple(row.labels) != labels:
        raise ValueError("Annotation disagrees with observable description")


def corpus():
    """Called only by remote scientific entry points; no sealed data."""
    unique = {}
    for statuses, name, frame, reverse, negator in itertools.product(
        STATUSES, NAMES, range(len(GRAMMARS)), (False, True), ("not", "without")
    ):
        parts = [
            word if status == "positive" else f"{negator} {word}"
            for word, status in zip(("flammable", "conductive"), statuses, strict=True)
            if status != "absent"
        ]
        if reverse:
            parts.reverse()
        content = "; ".join(parts) if parts else "present"
        question = GRAMMARS[frame].format(name=name, content=content)
        row = JointCase(
            question,
            question,
            (statuses[0], statuses[1]),
            (int(statuses[0] == "positive"), int(statuses[1] == "positive")),
            f"frame_{frame}",
            "old" if frame < 2 else "new",
            frame,
            name,
            content,
        )
        verify_observable(row)
        unique.setdefault(question, row)
    return list(unique.values())


def validate_disjoint(rows, protected):
    protected = set(map(normalized, protected))
    seen = set()
    for row in rows:
        verify_observable(row)
        key = normalized(row.question)
        if key in protected or key in seen:
            raise ValueError("Training overlaps historical support/protected probe or duplicate")
        seen.add(key)


def plan(rows, seed, steps):
    """Matched name/content/status per pair, balanced old/new and grammar exposure."""
    index = {(r.frame, r.statuses, r.name, r.content): i for i, r in enumerate(rows)}
    choices = {
        status: sorted({(r.name, r.content) for r in rows if r.statuses == status})
        for status in STATUSES
    }
    if not all(choices.values()):
        raise ValueError("Incomplete status support")
    rng = random.Random(seed)
    result = []
    for step in range(steps):
        order = rng.sample(range(9), 9)
        batch = []
        for j in order:
            status = STATUSES[j]
            name, content = rng.choice(choices[status])
            batch.extend(
                index[(frame, status, name, content)]
                for frame in ((step + j) % 2, 2 + (step + j) % 3)
            )
        result.append(batch)
    return result


def stable_joint(curves):
    wanted = (1000, 1100, 1200)
    selected = [c for c in curves if c["updates"] in wanted]
    if tuple(c["updates"] for c in selected) != wanted:
        return False
    for c in selected:
        for group in ("old", "new"):
            m = c["training"][group]
            expected = {f"frame_{i}" for i in (range(2) if group == "old" else range(2, 5))}
            if set(m["family_exact"]) != expected:
                return False
            values = (m["exact"], *m["family_exact"].values(), *m["status_accuracy"].values())
            if not all(isinstance(v, (int, float)) and 0 <= v <= 1 for v in values):
                return False
            if (
                not m["family_exact"]
                or set(m["status_accuracy"]) != {"positive", "negated", "absent"}
                or m["exact"] < 0.99
                or min(m["family_exact"].values()) < 0.99
                or min(m["status_accuracy"].values()) < 0.98
            ):
                return False
    return True
