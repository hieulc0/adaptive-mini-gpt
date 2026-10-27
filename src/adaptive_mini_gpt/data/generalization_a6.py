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


COVERAGE_FRAMES = (
    "For {name}, status: {content}.",
    "{content}. This entry concerns {name}.",
    "Concerning {name}, we record {content}; entry complete.",
    "Inventory for {name}: {content}.",
    "{name} is catalogued with {content}.",
    "Properties recorded: {content}. Identifier: {name}.",
)
COVERAGE_PREFIXES = ("", "Log. ", "An entry. ")
DEVELOPMENT_FRAMES = (
    "For specimen {name}, affirmed or denied traits are recorded as {content}.",
    "Attribute report ({name}): {content}.",
    "{name}'s characteristics are given here: {content}.",
)


def remap(row, name, frame, prefix, grammar, family):
    question = prefix + grammar.format(name=name, content=row.content)
    result = GeneralizationCase(
        question,
        question,
        tuple(row.statuses),
        tuple(row.labels),
        family,
        name,
        row.content,
        frame,
        prefix,
    )
    verify_observable(result)
    return result


def coverage_corpus(reference):
    unique = {}
    for row in reference:
        for frame, grammar in enumerate(COVERAGE_FRAMES):
            for prefix in COVERAGE_PREFIXES:
                r = remap(row, row.name, frame, prefix, grammar, f"coverage_{frame}")
                unique.setdefault(r.question, r)
    return list(unique.values())


def development_corpus(reference, names):
    unique = {}
    for row in reference:
        for name, (frame, grammar) in itertools.product(names, enumerate(DEVELOPMENT_FRAMES)):
            r = remap(row, name, frame, "", grammar, f"development_{frame}")
            unique.setdefault(r.question, r)
    return list(unique.values())


def coverage_plan(reference_indices, reference_rows, broad, synthetic_offset):
    """Count NEW presentations per status; never alias grammar to replacement cycle."""
    import random
    from collections import defaultdict

    buckets = defaultdict(list)
    for i, row in enumerate(broad):
        buckets[(tuple(row.statuses), row.frame, row.prefix)].append(i)
    cursor = defaultdict(int)
    queues = {}
    rng = random.Random(42)
    result = []
    for ix in reference_indices:
        batch = []
        for i in ix:
            if i >= synthetic_offset:
                batch.append(len(broad) + i - synthetic_offset)
                continue
            if i >= len(reference_rows):
                raise ValueError("Historical examples are not in the synthetic reference")
            status = tuple(reference_rows[i].statuses)
            tick = cursor[status]
            cursor[status] += 1
            frame = tick % len(COVERAGE_FRAMES)
            prefix = COVERAGE_PREFIXES[(tick // len(COVERAGE_FRAMES)) % len(COVERAGE_PREFIXES)]
            key = (status, frame, prefix)
            if not queues.get(key):
                if not buckets[key]:
                    raise ValueError("Missing grammar/prefix/status support")
                queues[key] = rng.sample(buckets[key], len(buckets[key]))
            batch.append(queues[key].pop())
        result.append(batch)
    return result


def stable_acquisition(curves, families):
    """Same binding A5 numerical gates, explicit expected families, fail closed."""
    selected = [c for c in curves if c["updates"] in (1000, 1100, 1200)]
    if [c["updates"] for c in selected] != [1000, 1100, 1200]:
        return False
    for c in selected:
        m = c["training"]
        if set(m["family_exact"]) != set(families) or set(m["status_accuracy"]) != {
            "positive",
            "negated",
            "absent",
        }:
            return False
        if not all(
            0 <= v <= 1
            for v in (m["exact"], *m["family_exact"].values(), *m["status_accuracy"].values())
        ):
            return False
        if (
            m["exact"] < 0.99
            or min(m["family_exact"].values()) < 0.99
            or min(m["status_accuracy"].values()) < 0.98
        ):
            return False
    return True


def structured_features(rows):
    """Privileged explicit clause annotations, diagnostic control ONLY, never student input."""
    result = []
    for row in rows:
        vector = [0.0] * 12
        clauses = row.content.split("; ") if hasattr(row, "content") else []
        # A.4 cases lack content metadata; read ONLY explicit observed clauses for this diagnostic.
        if not clauses:
            import re

            clauses = [
                f"{neg} {word}".strip()
                for neg, word in re.findall(
                    r"\b(?:(not|without) )?(flammable|conductive)\b", row.question
                )
            ]
        if clauses == ["present"]:
            clauses = []
        for slot in range(2):
            if slot >= len(clauses):
                prop, neg = 2, 0
            else:
                tokens = clauses[slot].split()
                prop = ("flammable", "conductive").index(tokens[-1])
                neg = int(tokens[0] in ("not", "without"))
            vector[slot * 6 + 2 * prop + neg] = 1.0
        result.append(vector)
    return result
