"""Stage A.11-R2: Entity-Conditioned Counterfactual Diagnostic Suite (EB-1..4).

Constructs counterfactual minimal pairs specifically designed to test multi-entity binding:
- EB-1_QUERY_SWAP: Identical prefix description, opposite queried target (multiset equal).
- EB-2_CLAUSE_ORDER_SWAP: Swapped clause order, same queried target (multiset equal).
- EB-3_PROPERTY_ASSIGNMENT_SWAP: Swapped property binding across entities (multiset equal).
- EB-4_THREE_ENTITY_DISTRACTOR: Two distractor entities + one queried entity (tests 3-way binding).

Includes an independent researcher-only structured rule verifier that parses prompts
symbolically and asserts 100% label entailment.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from typing import Any

from adaptive_mini_gpt.data.attribute_learning import AttributeCase

TRAITS = ("flammable", "conductive")


@dataclass(frozen=True)
class EntityBindingCase(AttributeCase):
    pair_id: str
    member: str  # "A" or "B"
    family: str  # "EB-1", "EB-2", "EB-3", "EB-4"
    target_entity: str
    other_entities: tuple[str, ...]
    is_counterfactual: bool = True


def words_bow(text: str) -> list[str]:
    return re.findall(r"\w+", text.lower())


def verify_pair_multiset_equality(case_a: EntityBindingCase, case_b: EntityBindingCase) -> bool:
    """Verifies that case_a and case_b have the EXACT same word multiset."""
    return Counter(words_bow(case_a.question)) == Counter(words_bow(case_b.question))


# Entity pools per seed (completely disjoint from sealed suite names: Drav, Sorn, Kael, Vond, Lira, Meko)
TRAIN_ENTITIES = ["Vim", "Rax", "Kip"]
DEV_ENTITIES_BY_SEED: dict[int, list[str]] = {
    42: ["Tuv", "Wex", "Zen"],
    43: ["Bax", "Cip", "Gid"],
    44: ["Tor", "Lum", "Fen"],
}

# Grammatical templates
EB1_TEMPLATES = [
    "Object {e1} is {d1}, while Object {e2} is {d2}. Query target: {query}, not {other}.",
    "Regarding {e1}, the core is {d1}; whereas for {e2}, the core is {d2}. Query target: {query}, not {other}.",
    "While {e1} is described as {d1}, {e2} is described as {d2}. Target is {query}, not {other}.",
    "Notice indicates {e1} is {d1}; in contrast, {e2} is {d2}. Analysis focuses on {query}, not {other}.",
]

EB2_TEMPLATES = [
    ("Object {e1} is {d1}; Object {e2} is {d2}. Query target: {query}.",
     "Object {e2} is {d2}; Object {e1} is {d1}. Query target: {query}."),
    ("Regarding {e1}, it is {d1}; regarding {e2}, it is {d2}. Query target: {query}.",
     "Regarding {e2}, it is {d2}; regarding {e1}, it is {d1}. Query target: {query}."),
    ("While {e1} is {d1}, {e2} is {d2}. Target is {query}.",
     "While {e2} is {d2}, {e1} is {d1}. Target is {query}."),
    ("For {e1}, the core is {d1}; for {e2}, the core is {d2}. Analysis focuses on {query}.",
     "For {e2}, the core is {d2}; for {e1}, the core is {d1}. Analysis focuses on {query}."),
]

EB3_TEMPLATES = [
    ("For {e1}, it is {d1}; for {e2}, it is {d2}. Query target: {query}.",
     "For {e1}, it is {d2}; for {e2}, it is {d1}. Query target: {query}."),
    ("Object {e1} has core {d1}; Object {e2} has core {d2}. Target is {query}.",
     "Object {e1} has core {d2}; Object {e2} has core {d1}. Target is {query}."),
    ("Notice for {e1} shows {d1}; notice for {e2} shows {d2}. Analysis focuses on {query}.",
     "Notice for {e1} shows {d2}; notice for {e2} shows {d1}. Analysis focuses on {query}."),
    ("While {e1} is reported {d1}, {e2} is reported {d2}. Query target: {query}.",
     "While {e1} is reported {d2}, {e2} is reported {d1}. Query target: {query}."),
]

EB4_TEMPLATES = [
    "Object {e1} is {d1}, Object {e2} is {d2}, and Object {e3} is {d3}. Query target: {query}, not {other}.",
    "Regarding {e1}, it is {d1}; for {e2}, it is {d2}; for {e3}, it is {d3}. Query target: {query}, not {other}.",
    "Record indicates {e1} is {d1}; meanwhile {e2} is {d2}; whereas {e3} is {d3}. Target is {query}, not {other}.",
    "While {e1} has core {d1}, {e2} exhibits {d2}, and {e3} demonstrates {d3}. Focus on {query}, not {other}.",
]


# Structured configurations: (statuses_1, statuses_2, labels_1, labels_2, desc_1, desc_2)
TWO_ENTITY_CONFIGS = [
    # 1. Opposite single properties
    (("positive", "absent"), ("absent", "positive"), (1, 0), (0, 1), "flammable", "conductive"),
    (("absent", "positive"), ("positive", "absent"), (0, 1), (1, 0), "conductive", "flammable"),
    # 2. Opposite with negation
    (("positive", "negated"), ("negated", "positive"), (1, 0), (0, 1), "flammable and not conductive", "not flammable and conductive"),
    (("negated", "positive"), ("positive", "negated"), (0, 1), (1, 0), "not flammable and conductive", "flammable and not conductive"),
    # 3. Dual affirmed vs dual negated
    (("positive", "positive"), ("negated", "negated"), (1, 1), (0, 0), "flammable and conductive", "not flammable and not conductive"),
    (("negated", "negated"), ("positive", "positive"), (0, 0), (1, 1), "not flammable and not conductive", "flammable and conductive"),
    # 4. Asymmetric dual
    (("positive", "positive"), ("positive", "negated"), (1, 1), (1, 0), "flammable and conductive", "flammable and not conductive"),
    (("positive", "negated"), ("positive", "positive"), (1, 0), (1, 1), "flammable and not conductive", "flammable and conductive"),
    (("negated", "positive"), ("negated", "negated"), (0, 1), (0, 0), "not flammable and conductive", "not flammable and not conductive"),
]


def generate_eb1_query_swap(
    entities: list[str],
    split_prefix: str = "dev",
    target_count: int = 36,
) -> list[tuple[EntityBindingCase, EntityBindingCase]]:
    """EB-1: Identical prefix description, opposite queried target."""
    pairs: list[tuple[EntityBindingCase, EntityBindingCase]] = []
    e1, e2 = entities[0], entities[1]
    pair_idx = 0

    while len(pairs) < target_count:
        for tmpl in EB1_TEMPLATES:
            for st1, st2, lbl1, lbl2, d1, d2 in TWO_ENTITY_CONFIGS:
                pair_idx += 1
                pid = f"{split_prefix}_eb1_{pair_idx:03d}"

                q_a = tmpl.format(e1=e1, e2=e2, d1=d1, d2=d2, query=e1, other=e2)
                q_b = tmpl.format(e1=e1, e2=e2, d1=d1, d2=d2, query=e2, other=e1)

                case_a = EntityBindingCase(
                    case_id=f"{pid}_A",
                    question=q_a,
                    statuses=st1,
                    labels=lbl1,
                    family="EB-1",
                    pair_id=pid,
                    member="A",
                    target_entity=e1,
                    other_entities=(e2,),
                )
                case_b = EntityBindingCase(
                    case_id=f"{pid}_B",
                    question=q_b,
                    statuses=st2,
                    labels=lbl2,
                    family="EB-1",
                    pair_id=pid,
                    member="B",
                    target_entity=e2,
                    other_entities=(e1,),
                )
                assert verify_pair_multiset_equality(case_a, case_b)
                pairs.append((case_a, case_b))
                if len(pairs) >= target_count:
                    return pairs
    return pairs


def generate_eb2_clause_order_swap(
    entities: list[str],
    split_prefix: str = "dev",
    target_count: int = 36,
) -> list[tuple[EntityBindingCase, EntityBindingCase]]:
    """EB-2: Swapped clause order, same queried target."""
    pairs: list[tuple[EntityBindingCase, EntityBindingCase]] = []
    e1, e2 = entities[0], entities[1]
    pair_idx = 0

    while len(pairs) < target_count:
        for tmpl_a, tmpl_b in EB2_TEMPLATES:
            for st1, _st2, lbl1, _, d1, d2 in TWO_ENTITY_CONFIGS:
                pair_idx += 1
                pid = f"{split_prefix}_eb2_{pair_idx:03d}"

                # Query is e1 in both
                q_a = tmpl_a.format(e1=e1, e2=e2, d1=d1, d2=d2, query=e1)
                q_b = tmpl_b.format(e1=e1, e2=e2, d1=d1, d2=d2, query=e1)

                case_a = EntityBindingCase(
                    case_id=f"{pid}_A",
                    question=q_a,
                    statuses=st1,
                    labels=lbl1,
                    family="EB-2",
                    pair_id=pid,
                    member="A",
                    target_entity=e1,
                    other_entities=(e2,),
                )
                case_b = EntityBindingCase(
                    case_id=f"{pid}_B",
                    question=q_b,
                    statuses=st1,
                    labels=lbl1,
                    family="EB-2",
                    pair_id=pid,
                    member="B",
                    target_entity=e1,
                    other_entities=(e2,),
                )
                assert verify_pair_multiset_equality(case_a, case_b)
                pairs.append((case_a, case_b))
                if len(pairs) >= target_count:
                    return pairs
    return pairs


def generate_eb3_property_assignment_swap(
    entities: list[str],
    split_prefix: str = "dev",
    target_count: int = 36,
) -> list[tuple[EntityBindingCase, EntityBindingCase]]:
    """EB-3: Swapped property binding across entities, same queried target."""
    pairs: list[tuple[EntityBindingCase, EntityBindingCase]] = []
    e1, e2 = entities[0], entities[1]
    pair_idx = 0

    while len(pairs) < target_count:
        for tmpl_a, tmpl_b in EB3_TEMPLATES:
            for st1, st2, lbl1, lbl2, d1, d2 in TWO_ENTITY_CONFIGS:
                if lbl1 == lbl2:
                    continue  # Ensure label changes across pair
                pair_idx += 1
                pid = f"{split_prefix}_eb3_{pair_idx:03d}"

                # Target is e1 in both
                q_a = tmpl_a.format(e1=e1, e2=e2, d1=d1, d2=d2, query=e1)
                q_b = tmpl_b.format(e1=e1, e2=e2, d1=d1, d2=d2, query=e1)

                case_a = EntityBindingCase(
                    case_id=f"{pid}_A",
                    question=q_a,
                    statuses=st1,
                    labels=lbl1,
                    family="EB-3",
                    pair_id=pid,
                    member="A",
                    target_entity=e1,
                    other_entities=(e2,),
                )
                case_b = EntityBindingCase(
                    case_id=f"{pid}_B",
                    question=q_b,
                    statuses=st2,
                    labels=lbl2,
                    family="EB-3",
                    pair_id=pid,
                    member="B",
                    target_entity=e1,
                    other_entities=(e2,),
                )
                assert verify_pair_multiset_equality(case_a, case_b)
                pairs.append((case_a, case_b))
                if len(pairs) >= target_count:
                    return pairs
    return pairs


def generate_eb4_three_entity_distractor(
    entities: list[str],
    split_prefix: str = "dev",
    target_count: int = 36,
) -> list[tuple[EntityBindingCase, EntityBindingCase]]:
    """EB-4: Two distractor entities + one queried entity with unfamiliar names."""
    pairs: list[tuple[EntityBindingCase, EntityBindingCase]] = []
    if len(entities) < 3:
        raise ValueError("EB-4 requires at least 3 entities")
    e1, e2, e3 = entities[0], entities[1], entities[2]
    pair_idx = 0

    three_configs = [
        # (st1, st2, st3, lbl1, lbl2, lbl3, d1, d2, d3)
        (
            ("positive", "absent"), ("absent", "positive"), ("negated", "negated"),
            (1, 0), (0, 1), (0, 0),
            "flammable", "conductive", "not flammable and not conductive",
        ),
        (
            ("positive", "negated"), ("negated", "positive"), ("positive", "positive"),
            (1, 0), (0, 1), (1, 1),
            "flammable and not conductive", "not flammable and conductive", "flammable and conductive",
        ),
        (
            ("positive", "positive"), ("positive", "absent"), ("absent", "positive"),
            (1, 1), (1, 0), (0, 1),
            "flammable and conductive", "flammable", "conductive",
        ),
        (
            ("absent", "positive"), ("positive", "negated"), ("negated", "absent"),
            (0, 1), (1, 0), (0, 0),
            "conductive", "flammable and not conductive", "not flammable",
        ),
    ]

    while len(pairs) < target_count:
        for tmpl in EB4_TEMPLATES:
            for st1, st2, _st3, lbl1, lbl2, _lbl3, d1, d2, d3 in three_configs:
                pair_idx += 1
                pid = f"{split_prefix}_eb4_{pair_idx:03d}"

                # Query e1 vs e2 with e3 as 3rd distractor
                q_a = tmpl.format(e1=e1, e2=e2, e3=e3, d1=d1, d2=d2, d3=d3, query=e1, other=e2)
                q_b = tmpl.format(e1=e1, e2=e2, e3=e3, d1=d1, d2=d2, d3=d3, query=e2, other=e1)

                case_a = EntityBindingCase(
                    case_id=f"{pid}_A",
                    question=q_a,
                    statuses=st1,
                    labels=lbl1,
                    family="EB-4",
                    pair_id=pid,
                    member="A",
                    target_entity=e1,
                    other_entities=(e2, e3),
                )
                case_b = EntityBindingCase(
                    case_id=f"{pid}_B",
                    question=q_b,
                    statuses=st2,
                    labels=lbl2,
                    family="EB-4",
                    pair_id=pid,
                    member="B",
                    target_entity=e2,
                    other_entities=(e1, e3),
                )
                assert verify_pair_multiset_equality(case_a, case_b)
                pairs.append((case_a, case_b))
                if len(pairs) >= target_count:
                    return pairs
    return pairs


def build_entity_binding_suite(seed: int = 42, split: str = "dev") -> dict[str, Any]:
    """Builds the 144-pair (288-case) Entity-Conditioned Diagnostic Suite."""
    if split == "train":
        entities = TRAIN_ENTITIES
        prefix = f"train_s{seed}"
    else:
        entities = DEV_ENTITIES_BY_SEED.get(seed, DEV_ENTITIES_BY_SEED[42])
        prefix = f"dev_s{seed}"

    eb1 = generate_eb1_query_swap(entities, split_prefix=prefix, target_count=36)
    eb2 = generate_eb2_clause_order_swap(entities, split_prefix=prefix, target_count=36)
    eb3 = generate_eb3_property_assignment_swap(entities, split_prefix=prefix, target_count=36)
    eb4 = generate_eb4_three_entity_distractor(entities, split_prefix=prefix, target_count=36)

    assert len(eb1) == 36
    assert len(eb2) == 36
    assert len(eb3) == 36
    assert len(eb4) == 36

    all_pairs = eb1 + eb2 + eb3 + eb4
    assert len(all_pairs) == 144

    cases: list[EntityBindingCase] = []
    for a, b in all_pairs:
        cases.extend([a, b])
    assert len(cases) == 288

    return {
        "pairs": all_pairs,
        "cases": cases,
        "by_family": {
            "EB-1": eb1,
            "EB-2": eb2,
            "EB-3": eb3,
            "EB-4": eb4,
        },
        "num_pairs": len(all_pairs),
        "num_cases": len(cases),
    }


def structured_oracle_verify(question: str) -> tuple[int, int]:
    """Independent researcher-only symbolic rule verifier.

    Parses natural language prompt into symbolic bindings and derives ground truth
    labels strictly via deterministic linguistic rules (never exposed to student).
    """
    query_match = re.search(
        r"(?:Query target|Target|Analysis focuses on|Focus on):?\s*(?:is\s*)?(\w+)",
        question,
        re.IGNORECASE,
    )
    if not query_match:
        raise ValueError(f"Could not extract query target from: {question}")
    target_entity = query_match.group(1).strip()
    prefix = question[: query_match.start()]

    non_entities = {
        "regarding", "while", "notice", "object", "analysis",
        "record", "focus", "target", "query", "for", "the",
    }
    tokens = re.findall(r"\b[A-Z][a-z0-9]+\b", prefix)
    entities = [t for t in tokens if t.lower() not in non_entities]
    seen: list[str] = []
    for e in entities:
        if e not in seen:
            seen.append(e)

    positions: list[tuple[int, str]] = []
    for e in seen:
        for m in re.finditer(rf"\b{re.escape(e)}\b", prefix):
            positions.append((m.start(), e))
    positions.sort()

    target_clause: str | None = None
    for i, (pos, ent) in enumerate(positions):
        if ent.lower() == target_entity.lower():
            next_pos = positions[i + 1][0] if i + 1 < len(positions) else len(prefix)
            if i + 1 < len(positions):
                sub = prefix[pos:next_pos]
                cut = max(sub.rfind(";"), sub.rfind("."))
                if cut != -1:
                    target_clause = prefix[pos : pos + cut]
                else:
                    cut2 = sub.rfind(",")
                    if cut2 != -1:
                        target_clause = prefix[pos : pos + cut2]
                    else:
                        target_clause = prefix[pos:next_pos]
            else:
                target_clause = prefix[pos:next_pos]
            break

    if target_clause is None:
        raise ValueError(f"Target entity '{target_entity}' not found in: {prefix}")

    flammable_status = 0
    conductive_status = 0

    flam_match = re.search(r"(\b(?:not|without)\s+)?flammable\b", target_clause, re.IGNORECASE)
    if flam_match:
        flammable_status = 1 if not flam_match.group(1) else 0

    cond_match = re.search(r"(\b(?:not|without)\s+)?conductive\b", target_clause, re.IGNORECASE)
    if cond_match:
        conductive_status = 1 if not cond_match.group(1) else 0

    return (flammable_status, conductive_status)
