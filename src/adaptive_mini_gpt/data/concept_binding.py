"""Researcher-owned simulator/fixtures, never an inference-time answer engine.

Two independent Boolean traits; two entities; explicit positive/negative/absent
traits, reference binding, two context laws and one/two toggling actions. Each
pair swaps the entity/trait binding while preserving the exact word multiset.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import random
import re
from dataclasses import asdict, dataclass

TRAITS = ("flammable", "conductive")
BUNDLES = list(itertools.product((False, True), repeat=2))
COMPOSITION_HOLDOUTS = (((False, False), (False, True)), ((True, False), (True, True)))


@dataclass(frozen=True)
class ConceptCase:
    case_id: str
    pair_id: str
    family: str
    question: str
    rules: str
    tags: tuple[str, ...]
    action_id: int
    state: int
    structured_diagnostic: tuple[int, ...]


def bag(text):
    return sorted(re.findall(r"\w+", text.lower()))


def pair_cases(pair_id, bundles, law, initial, actions, reference, style, absent=False):
    rules = "Absent=false. Actions target query. " + (
        "fire toggles light iff flammable; spark iff conductive."
        if law == 0
        else "fire toggles light iff conductive; spark iff flammable."
    )
    names = ("A", "B") if style < 2 else ("U", "V")
    target = reference % 2
    query = names[target] if reference < 2 else ("first", "second")[target]
    rows = []
    for orientation in (0, 1):
        first, second = bundles if orientation == 0 else bundles[::-1]
        descriptions = []
        for name, bits in zip(names, (first, second), strict=True):
            clauses = []
            for trait, present in zip(TRAITS, bits, strict=True):
                if present:
                    clauses.append(trait)
                elif not absent:
                    clauses.append("not " + trait if style % 2 == 0 else "without " + trait)
            traits = ", ".join(clauses) if clauses else "no listed traits"
            descriptions.append(
                f"{name}: {traits}"
                if style == 0
                else f"{name} has {traits}"
                if style == 1
                else f"For {name}, traits are {traits}"
            )
        sequence = " then ".join(actions)
        question = f"{descriptions[0]}; {descriptions[1]}. Initially {'on' if initial else 'off'}. Do {sequence}. Query {query}."
        chosen = (first, second)[target]
        state = bool(initial)
        for action in actions:
            property_index = (0 if action == "fire" else 1) ^ law
            state ^= chosen[property_index]
        family = (
            "chain"
            if len(actions) == 2
            else "absent"
            if absent
            else "reference"
            if reference >= 2
            else "negation"
        )
        rows.append(
            ConceptCase(
                f"{pair_id}:{orientation}",
                pair_id,
                family,
                question,
                rules,
                tuple(trait for trait, present in zip(TRAITS, chosen, strict=True) if present),
                0 if actions[-1] == "fire" else 2,
                int(state),
                tuple(
                    map(
                        int,
                        (
                            *chosen,
                            initial,
                            law,
                            actions.count("fire") % 2,
                            actions.count("spark") % 2,
                        ),
                    )
                ),
            )
        )
    assert bag(rows[0].question + rows[0].rules) == bag(rows[1].question + rows[1].rules)
    return rows


def generate(seed=42, pairs_per_split=128):
    rng = random.Random(seed)
    buckets = {"train": [], "validation": [], "description": [], "composition": []}
    pair_number = 0
    for bundles in itertools.combinations(BUNDLES, 2):
        for law, initial, actions, reference, absent in itertools.product(
            range(2),
            range(2),
            [("fire",), ("spark",), ("fire", "spark"), ("spark", "fire")],
            range(4),
            (False, True),
        ):
            # Exact balanced-label pairs prevent the bag-of-words ceiling being
            # hidden by a constant or skewed outcome slice.
            rows = pair_cases("probe", bundles, law, initial, actions, reference, 0, absent)
            if rows[0].state == rows[1].state:
                continue
            pair_number += 1
            holdout = bundles in COMPOSITION_HOLDOUTS
            split = "composition" if holdout else "train"
            style = 2 if holdout else (law + initial + reference) % 2
            buckets[split].append(
                pair_cases(
                    f"{split}:{seed}:{pair_number}",
                    bundles,
                    law,
                    initial,
                    actions,
                    reference,
                    style,
                    absent,
                )
            )
            if not holdout:
                buckets["description"].append(
                    pair_cases(
                        f"description:{seed}:{pair_number}",
                        bundles,
                        law,
                        initial,
                        actions,
                        reference,
                        2,
                        absent,
                    )
                )
    # Validation uses a disjoint set of procedural settings from training.
    rng.shuffle(buckets["train"])
    validation = buckets["train"][: max(8, pairs_per_split // 4)]
    buckets["train"] = buckets["train"][len(validation) :]
    buckets["validation"] = validation
    result = {}
    for split, pairs in buckets.items():
        rng.shuffle(pairs)
        result[split] = [row for pair in pairs[:pairs_per_split] for row in pair]
    # Tiny is a separate development fitting diagnostic, never called held-out.
    result["tiny"] = [row for pair in buckets["train"][:4] for row in pair]
    train_questions = {(r.question, r.rules) for r in result["train"]}
    for split in ("validation", "description", "composition"):
        assert not train_questions.intersection((r.question, r.rules) for r in result[split])
    return result


def manifest(splits, *, seed):
    records = {name: [asdict(c) for c in rows] for name, rows in splits.items()}
    return {
        "data_sha256": hashlib.sha256(json.dumps(records, sort_keys=True).encode()).hexdigest(),
        "splits": {name: len(rows) for name, rows in splits.items()},
        "protocol": "concept-binding-v1",
        "seed": seed,
    }
