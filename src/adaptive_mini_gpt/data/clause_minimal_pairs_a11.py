"""Semantic Minimal Pairs for Stage A.11 (Clause-Sensitive Relational Learning).

Constructs counterfactual minimal pairs where each pair (case_a, case_b) shares the
EXACT same word multiset (100% Bag-of-Words overlap) or swaps the queried target across clauses.
Guarantees mathematically that pure Bag-of-Words linear models cannot score above 50%
on pair exact accuracy.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from typing import Any

import torch

from adaptive_mini_gpt.data.attribute_learning import AttributeCase

TRAITS = ("flammable", "conductive")


@dataclass(frozen=True)
class MinimalPairCase(AttributeCase):
    pair_id: str
    member: str  # "A" or "B"
    family: str  # "MP-1", "MP-2", "MP-3"
    entity: str
    distractor_part: str
    distractor_adj: str
    is_counterfactual: bool = True


def words_bow(text: str) -> list[str]:
    return re.findall(r"\w+", text.lower())


def verify_pair_multiset_equality(case_a: MinimalPairCase, case_b: MinimalPairCase) -> bool:
    """Verifies that case_a and case_b have the EXACT same word multiset."""
    return Counter(words_bow(case_a.question)) == Counter(words_bow(case_b.question))


# Distractor components and adjectives
TRAIN_PARTS = ["casing", "chassis", "housing", "frame"]
DEV_PARTS = ["shell", "panel", "conduit", "shroud"]

TRAIN_ADJS = ["heavy", "dense", "fragile", "rigid"]
DEV_ADJS = ["porous", "metallic", "elastic", "viscous"]

LONG_RANGE_MODIFIERS = [
    "decidedly and undeniably",
    "clearly and unambiguously",
    "definitely and consistently",
    "certainly and demonstrably",
]

MP2_TEMPLATES = [
    "Regarding {e1}, the core is {d1}; whereas for {e2}, the core is {d2}. Target is {query}, not {other}.",
    "While {e1} has a core that is {d1}, {e2} exhibits a core that is {d2}. Query target is {query}, not {other}.",
    "Notice indicates {e1} is {d1}; in contrast, {e2} is {d2}. Analysis focuses on {query}, not {other}.",
    "Record for {e1} shows traits {d1}; meanwhile, {e2} demonstrates {d2}. The evaluated target is {query}, not {other}.",
]


def generate_mp1_negation_transfer(
    entities: list[str],
    parts: list[str],
    adjs: list[str],
    split_prefix: str = "dev",
) -> list[tuple[MinimalPairCase, MinimalPairCase]]:
    """Family MP-1: Moves 'not' between the distractor clause and target property.

    Pair A: "The <part> is not <adj>; for <entity>, the core is <prop1> and <prop2>."
    Pair B: "The <part> is <adj>; for <entity>, the core is not <prop1> and <prop2>."
    """
    pairs: list[tuple[MinimalPairCase, MinimalPairCase]] = []
    pair_idx = 0

    status_patterns = [
        # (prop1_a, prop2_a, prop1_b, prop2_b, label_a, label_b)
        ("flammable", "conductive", "flammable", "conductive", (1, 1), (0, 1), "neg_prop1"),
        ("flammable", "conductive", "flammable", "conductive", (1, 1), (1, 0), "neg_prop2"),
        ("flammable", "conductive", "flammable", "conductive", (1, 0), (0, 0), "neg_prop1_abs2"),
        ("flammable", "conductive", "flammable", "conductive", (0, 1), (0, 0), "neg_prop2_abs1"),
    ]

    for entity in entities:
        for p_idx, part in enumerate(parts):
            adj = adjs[p_idx % len(adjs)]
            for p1, p2, _, _, lbl_a, lbl_b, mode in status_patterns:
                pair_idx += 1
                pid = f"{split_prefix}_mp1_{pair_idx:03d}"

                if mode == "neg_prop1":
                    q_a = f"The {part} is not {adj}; for {entity}, the core is {p1} and {p2}."
                    q_b = f"The {part} is {adj}; for {entity}, the core is not {p1} and {p2}."
                    st_a = ("positive", "positive")
                    st_b = ("negated", "positive")
                elif mode == "neg_prop2":
                    q_a = f"The {part} is not {adj}; for {entity}, the core is {p1} and {p2}."
                    q_b = f"The {part} is {adj}; for {entity}, the core is {p1} and not {p2}."
                    st_a = ("positive", "positive")
                    st_b = ("positive", "negated")
                elif mode == "neg_prop1_abs2":
                    q_a = f"The {part} is not {adj}; for {entity}, the core is {p1}."
                    q_b = f"The {part} is {adj}; for {entity}, the core is not {p1}."
                    st_a = ("positive", "absent")
                    st_b = ("negated", "absent")
                else:  # neg_prop2_abs1
                    q_a = f"The {part} is not {adj}; for {entity}, the core is {p2}."
                    q_b = f"The {part} is {adj}; for {entity}, the core is not {p2}."
                    st_a = ("absent", "positive")
                    st_b = ("absent", "negated")

                case_a = MinimalPairCase(
                    case_id=f"{pid}_A",
                    question=q_a,
                    statuses=st_a,
                    labels=lbl_a,
                    family="MP-1",
                    pair_id=pid,
                    member="A",
                    entity=entity,
                    distractor_part=part,
                    distractor_adj=adj,
                )
                case_b = MinimalPairCase(
                    case_id=f"{pid}_B",
                    question=q_b,
                    statuses=st_b,
                    labels=lbl_b,
                    family="MP-1",
                    pair_id=pid,
                    member="B",
                    entity=entity,
                    distractor_part=part,
                    distractor_adj=adj,
                )
                assert verify_pair_multiset_equality(case_a, case_b)
                pairs.append((case_a, case_b))

    return pairs


def generate_mp2_multi_entity_binding(
    entities: list[str],
    split_prefix: str = "dev",
) -> list[tuple[MinimalPairCase, MinimalPairCase]]:
    """Family MP-2: Two entities described in two clauses, followed by target query.

    Pair A: "<clause_e1>; <clause_e2>. Target is <e1>, not <e2>."
    Pair B: "<clause_e1>; <clause_e2>. Target is <e2>, not <e1>."
    """
    pairs: list[tuple[MinimalPairCase, MinimalPairCase]] = []
    if len(entities) < 2:
        return pairs

    pair_idx = 0
    configurations = [
        # (e1_st, e2_st, e1_lbl, e2_lbl, e1_desc, e2_desc)
        (
            ("positive", "absent"),
            ("absent", "positive"),
            (1, 0),
            (0, 1),
            "flammable",
            "conductive",
        ),
        (
            ("positive", "negated"),
            ("negated", "positive"),
            (1, 0),
            (0, 1),
            "flammable and not conductive",
            "not flammable and conductive",
        ),
        (
            ("positive", "positive"),
            ("negated", "negated"),
            (1, 1),
            (0, 0),
            "flammable and conductive",
            "not flammable and not conductive",
        ),
        (
            ("absent", "positive"),
            ("positive", "absent"),
            (0, 1),
            (1, 0),
            "conductive",
            "flammable",
        ),
    ]

    for i in range(len(entities)):
        for j in range(len(entities)):
            if i == j:
                continue
            e1, e2 = entities[i], entities[j]
            for tmpl in MP2_TEMPLATES:
                for st1, st2, lbl1, lbl2, desc1, desc2 in configurations:
                    pair_idx += 1
                    pid = f"{split_prefix}_mp2_{pair_idx:03d}"

                    q_a = tmpl.format(e1=e1, e2=e2, d1=desc1, d2=desc2, query=e1, other=e2)
                    q_b = tmpl.format(e1=e1, e2=e2, d1=desc1, d2=desc2, query=e2, other=e1)

                    case_a = MinimalPairCase(
                        case_id=f"{pid}_A",
                        question=q_a,
                        statuses=st1,
                        labels=lbl1,
                        family="MP-2",
                        pair_id=pid,
                        member="A",
                        entity=e1,
                        distractor_part=e2,
                        distractor_adj=desc2,
                    )
                    case_b = MinimalPairCase(
                        case_id=f"{pid}_B",
                        question=q_b,
                        statuses=st2,
                        labels=lbl2,
                        family="MP-2",
                        pair_id=pid,
                        member="B",
                        entity=e2,
                        distractor_part=e1,
                        distractor_adj=desc1,
                    )
                    assert verify_pair_multiset_equality(case_a, case_b)
                    pairs.append((case_a, case_b))
                    if len(pairs) >= 32:
                        return pairs

    return pairs


def generate_mp3_long_range_scope(
    entities: list[str],
    parts: list[str],
    adjs: list[str],
    split_prefix: str = "dev",
) -> list[tuple[MinimalPairCase, MinimalPairCase]]:
    """Family MP-3: Long-range negation separated by 3-5 modifier words with distractor clauses.

    Pair A: "Although the outer <part> is not <adj>, for <entity> the core remains <mod> flammable and conductive."
    Pair B: "Although the outer <part> is <adj>, for <entity> the core remains <mod> not flammable and conductive."
    """
    pairs: list[tuple[MinimalPairCase, MinimalPairCase]] = []
    pair_idx = 0

    configs = [
        # (mod, prop_desc_a, prop_desc_b, st_a, st_b, lbl_a, lbl_b)
        (
            LONG_RANGE_MODIFIERS[0],
            "flammable and conductive",
            "not flammable and conductive",
            ("positive", "positive"),
            ("negated", "positive"),
            (1, 1),
            (0, 1),
        ),
        (
            LONG_RANGE_MODIFIERS[1],
            "flammable and conductive",
            "flammable and not conductive",
            ("positive", "positive"),
            ("positive", "negated"),
            (1, 1),
            (1, 0),
        ),
        (
            LONG_RANGE_MODIFIERS[2],
            "flammable",
            "not flammable",
            ("positive", "absent"),
            ("negated", "absent"),
            (1, 0),
            (0, 0),
        ),
        (
            LONG_RANGE_MODIFIERS[3],
            "conductive",
            "not conductive",
            ("absent", "positive"),
            ("absent", "negated"),
            (0, 1),
            (0, 0),
        ),
    ]

    for entity in entities:
        for p_idx, part in enumerate(parts):
            adj = adjs[p_idx % len(adjs)]
            for mod, desc_a, desc_b, st_a, st_b, lbl_a, lbl_b in configs:
                pair_idx += 1
                pid = f"{split_prefix}_mp3_{pair_idx:03d}"

                q_a = f"Although the outer {part} is not {adj}, for {entity} the core remains {mod} {desc_a}."
                q_b = f"Although the outer {part} is {adj}, for {entity} the core remains {mod} {desc_b}."

                case_a = MinimalPairCase(
                    case_id=f"{pid}_A",
                    question=q_a,
                    statuses=st_a,
                    labels=lbl_a,
                    family="MP-3",
                    pair_id=pid,
                    member="A",
                    entity=entity,
                    distractor_part=part,
                    distractor_adj=adj,
                )
                case_b = MinimalPairCase(
                    case_id=f"{pid}_B",
                    question=q_b,
                    statuses=st_b,
                    labels=lbl_b,
                    family="MP-3",
                    pair_id=pid,
                    member="B",
                    entity=entity,
                    distractor_part=part,
                    distractor_adj=adj,
                )
                assert verify_pair_multiset_equality(case_a, case_b)
                pairs.append((case_a, case_b))

    return pairs


def build_semantic_minimal_pairs_suite() -> dict[str, Any]:
    """Builds the complete Semantic Minimal Pairs suite with train and dev splits."""
    train_entities = ["Vim", "Rax"]
    dev_entities = ["Tuv", "Wex"]

    # 1. Training minimal pairs (32 + 32 + 32 = 96 pairs = 192 cases)
    tr_mp1 = generate_mp1_negation_transfer(train_entities, TRAIN_PARTS, TRAIN_ADJS, "train")
    tr_mp2 = generate_mp2_multi_entity_binding(train_entities, "train")
    tr_mp3 = generate_mp3_long_range_scope(train_entities, TRAIN_PARTS, TRAIN_ADJS, "train")

    # 2. Development minimal pairs (32 + 32 + 32 = 96 pairs = 192 cases)
    dev_mp1 = generate_mp1_negation_transfer(dev_entities, DEV_PARTS, DEV_ADJS, "dev")
    dev_mp2 = generate_mp2_multi_entity_binding(dev_entities, "dev")
    dev_mp3 = generate_mp3_long_range_scope(dev_entities, DEV_PARTS, DEV_ADJS, "dev")

    def flatten_pairs(pairs: list[tuple[MinimalPairCase, MinimalPairCase]]) -> list[MinimalPairCase]:
        res = []
        for a, b in pairs:
            res.extend([a, b])
        return res

    all_train_pairs = tr_mp1 + tr_mp2 + tr_mp3
    all_dev_pairs = dev_mp1 + dev_mp2 + dev_mp3
    train_cases = flatten_pairs(all_train_pairs)
    dev_cases = flatten_pairs(all_dev_pairs)

    return {
        "train_pairs": all_train_pairs,
        "dev_pairs": all_dev_pairs,
        "train_cases": train_cases,
        "dev_cases": dev_cases,
        "by_family": {
            "train": {"MP-1": tr_mp1, "MP-2": tr_mp2, "MP-3": tr_mp3},
            "dev": {"MP-1": dev_mp1, "MP-2": dev_mp2, "MP-3": dev_mp3},
        },
        "num_train_pairs": len(all_train_pairs),
        "num_dev_pairs": len(all_dev_pairs),
        "num_train_cases": len(train_cases),
        "num_dev_cases": len(dev_cases),
    }


def evaluate_pair_exact_accuracy(
    predictions: Any,
    eval_batch: Any,
    cases: list[Any],
) -> dict[str, float]:
    """Computes pair_exact_accuracy: 1.0 iff BOTH members of the pair are correct."""
    # Support both model + eval_batch or direct prediction list
    if hasattr(predictions, "forward_with_strategy"):
        model = predictions
        tokens, actions, _targets = eval_batch
        with torch.no_grad():
            out, _, _, _ = model.forward_with_strategy(tokens, actions)
            # Evaluate attribute head predictions on TAG_IDS
            from adaptive_mini_gpt.data.attribute_learning import TAG_IDS
            binary_preds = (out[:, TAG_IDS] >= 0).long().tolist()
    else:
        binary_preds = [list(p) for p in predictions]

    pair_map: dict[str, dict[str, Any]] = {}
    for c, pred in zip(cases, binary_preds, strict=True):
        pid = getattr(c, "pair_id", None)
        member = getattr(c, "member", None)
        if not pid or not member:
            continue
        if pid not in pair_map:
            pair_map[pid] = {}
        is_correct = (list(pred) == list(c.labels))
        pair_map[pid][member] = {
            "correct": is_correct,
            "family": getattr(c, "family", "unknown"),
        }

    total_pairs = len(pair_map)
    pair_correct_count = 0
    fam_counts: dict[str, int] = {}
    fam_correct: dict[str, int] = {}

    for _pid, members in pair_map.items():
        fam = members.get("A", {}).get("family", "unknown")
        fam_counts[fam] = fam_counts.get(fam, 0) + 1
        both_ok = members.get("A", {}).get("correct", False) and members.get("B", {}).get("correct", False)
        if both_ok:
            pair_correct_count += 1
            fam_correct[fam] = fam_correct.get(fam, 0) + 1

    overall_pair_acc = pair_correct_count / total_pairs if total_pairs > 0 else 0.0
    results: dict[str, float] = {
        "pair_exact_accuracy": overall_pair_acc,
        "pair_exact_flip_rate": overall_pair_acc,
    }
    for fam, cnt in fam_counts.items():
        results[f"pair_exact_{fam.lower()}"] = fam_correct.get(fam, 0) / cnt if cnt > 0 else 0.0

    return results
