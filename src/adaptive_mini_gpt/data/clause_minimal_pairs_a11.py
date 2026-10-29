"""Semantic Minimal Pairs for Stage A.11 (Clause-Sensitive Relational Learning).

Constructs counterfactual minimal pairs where each pair (case_a, case_b) shares the
EXACT same word multiset (100% Bag-of-Words overlap) or swaps the queried target across clauses.
Guarantees mathematically that pure Bag-of-Words linear models cannot score above 50%
on pair exact accuracy.
"""

from __future__ import annotations

import re
from collections import Counter, defaultdict
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


def compute_paired_metrics_from_predictions(
    cases: list[Any],
    predictions: Any,
) -> dict[str, Any]:
    """Computes deterministic paired metrics from a unified set of predictions.

    Supports predictions as:
    - list of predictions aligned with cases
    - dict of predictions keyed by case_id
    - torch.Tensor of logits or binary values

    Enforces the mathematical pigeonhole lower bound:
        pair_both_correct_acc + 1e-9 >= max(0.0, 2.0 * case_exact_acc - 1.0)
    on the overall set and across each family.
    """
    if isinstance(predictions, dict):
        raw_list = [predictions[getattr(c, "case_id", f"case_{i}")] for i, c in enumerate(cases)]
    elif isinstance(predictions, torch.Tensor):
        if predictions.dtype.is_floating_point:
            raw_list = (predictions >= 0).long().tolist()
        else:
            raw_list = predictions.long().tolist()
    else:
        raw_list = list(predictions)

    if len(cases) != len(raw_list):
        raise ValueError(f"Mismatch between cases ({len(cases)}) and predictions ({len(raw_list)})")

    binary_preds = [[int(v) for v in p] for p in raw_list]

    # Evaluate case-level correctness
    total_elements = 0
    correct_elements = 0
    case_matches: list[bool] = []

    for c, pred in zip(cases, binary_preds, strict=True):
        true_labels = list(c.labels)
        pred_labels = [int(v) for v in pred]
        is_exact = (pred_labels == true_labels)
        case_matches.append(is_exact)
        for p_val, t_val in zip(pred_labels, true_labels, strict=True):
            correct_elements += int(p_val == t_val)
            total_elements += 1

    # Group into pairs
    pair_map: dict[str, dict[str, Any]] = defaultdict(dict)
    for c, pred, is_exact in zip(cases, binary_preds, case_matches, strict=True):
        pid = getattr(c, "pair_id", None)
        member = getattr(c, "member", None)
        if pid and member in ("A", "B"):
            pair_map[pid][member] = {
                "case": c,
                "pred": [int(v) for v in pred],
                "correct": is_exact,
                "family": getattr(c, "family", "unknown"),
            }

    complete_pairs = {pid: m for pid, m in pair_map.items() if "A" in m and "B" in m}

    def _eval_pair_group(p_dict: dict[str, dict[str, Any]]) -> dict[str, Any]:
        num_p = len(p_dict)
        if num_p == 0:
            return {
                "n_cases": 0,
                "n_pairs": 0,
                "n_label_changing_pairs": 0,
                "per_property_binary_acc": 0.0,
                "case_exact_acc": 0.0,
                "pair_both_correct_acc": 0.0,
                "pair_counterfactual_flip_acc": 0.0,
                "confusion_matrices": {},
            }

        g_elem_total = 0
        g_elem_correct = 0
        g_case_correct = 0
        g_both_correct = 0
        g_label_changing = 0
        g_flip_correct = 0

        # Confusion statistics per property: traits = ("flammable", "conductive")
        tp_counts = [0, 0]
        tn_counts = [0, 0]
        fp_counts = [0, 0]
        fn_counts = [0, 0]

        for _pid, m in p_dict.items():
            for mem in ("A", "B"):
                c_case = m[mem]["case"]
                c_pred = m[mem]["pred"]
                c_true = list(c_case.labels)
                if m[mem]["correct"]:
                    g_case_correct += 1
                for p_idx, (p_val, t_val) in enumerate(zip(c_pred, c_true, strict=True)):
                    g_elem_correct += int(p_val == t_val)
                    g_elem_total += 1
                    if p_val == 1 and t_val == 1:
                        tp_counts[p_idx] += 1
                    elif p_val == 0 and t_val == 0:
                        tn_counts[p_idx] += 1
                    elif p_val == 1 and t_val == 0:
                        fp_counts[p_idx] += 1
                    elif p_val == 0 and t_val == 1:
                        fn_counts[p_idx] += 1

            true_a = list(m["A"]["case"].labels)
            true_b = list(m["B"]["case"].labels)
            pred_a = m["A"]["pred"]
            pred_b = m["B"]["pred"]
            both_ok = (m["A"]["correct"] and m["B"]["correct"])

            if both_ok:
                g_both_correct += 1

            if true_a != true_b:
                g_label_changing += 1
                if both_ok and (pred_a != pred_b):
                    g_flip_correct += 1

        n_c = num_p * 2
        case_acc = g_case_correct / n_c
        both_acc = g_both_correct / num_p
        flip_acc = g_flip_correct / g_label_changing if g_label_changing > 0 else 0.0
        prop_acc = g_elem_correct / g_elem_total if g_elem_total > 0 else 0.0

        confusion = {}
        for p_idx, trait in enumerate(TRAITS):
            tp = tp_counts[p_idx]
            tn = tn_counts[p_idx]
            fp = fp_counts[p_idx]
            fn = fn_counts[p_idx]
            prec = tp / max(1, tp + fp)
            rec = tp / max(1, tp + fn)
            confusion[trait] = {
                "TP": tp,
                "TN": tn,
                "FP": fp,
                "FN": fn,
                "precision": float(prec),
                "recall": float(rec),
            }

        # Mathematical bound assertion: Pigeonhole principle
        pigeonhole_bound = max(0.0, 2.0 * case_acc - 1.0)
        assert both_acc + 1e-9 >= pigeonhole_bound, (
            f"Pigeonhole violation: pair_both_correct_acc={both_acc:.4f} < {pigeonhole_bound:.4f} "
            f"(case_acc={case_acc:.4f})"
        )

        return {
            "n_cases": n_c,
            "n_pairs": num_p,
            "n_label_changing_pairs": g_label_changing,
            "per_property_binary_acc": prop_acc,
            "case_exact_acc": case_acc,
            "pair_both_correct_acc": both_acc,
            "pair_counterfactual_flip_acc": flip_acc,
            "confusion_matrices": confusion,
        }

    overall_paired = _eval_pair_group(complete_pairs)
    total_cases = len(cases)
    overall_case_acc = sum(case_matches) / total_cases if total_cases > 0 else 0.0
    overall_prop_acc = correct_elements / total_elements if total_elements > 0 else 0.0

    families = sorted({getattr(c, "family", "unknown") for c in cases if getattr(c, "family", None)})
    by_family: dict[str, dict[str, Any]] = {}
    for fam in families:
        fam_pairs = {
            pid: m for pid, m in complete_pairs.items()
            if m["A"]["family"] == fam or m["B"]["family"] == fam
        }
        by_family[fam] = _eval_pair_group(fam_pairs)

    return {
        "n_cases": total_cases,
        "n_pairs": len(complete_pairs),
        "n_label_changing_pairs": overall_paired["n_label_changing_pairs"],
        "per_property_binary_acc": overall_prop_acc,
        "case_exact_acc": overall_case_acc,
        "pair_both_correct_acc": overall_paired["pair_both_correct_acc"],
        "pair_counterfactual_flip_acc": overall_paired["pair_counterfactual_flip_acc"],
        "confusion_matrices": overall_paired.get("confusion_matrices", {}),
        "by_family": by_family,
    }


# Backwards compatibility alias
compute_paired_metrics = compute_paired_metrics_from_predictions


def evaluate_pair_exact_accuracy(
    predictions: Any,
    eval_batch: Any,
    cases: list[Any],
) -> dict[str, Any]:
    """Computes pair metrics on Semantic Minimal Pairs using compute_paired_metrics."""
    if hasattr(predictions, "forward_with_strategy"):
        model = predictions
        tokens, actions, _targets = eval_batch
        with torch.no_grad():
            _, tag_logits, _, _ = model.forward_with_strategy(tokens, actions)
            assert tag_logits is not None
            preds = (tag_logits[:, (10, 15)] >= 0).long().tolist()
        return compute_paired_metrics(cases, preds)

    return compute_paired_metrics(cases, predictions)


def build_balanced_relational_plan(
    single_coverage_plan: list[list[int]],
    len_broad: int,
    train_pairs: list[Any],
    synthetic_offset: int,
    total_steps: int = 7200,
) -> tuple[list[list[int]], int]:
    """Constructs balanced 18-case batches: 6 single-clause, 6 minimal-pair (3 pairs), 6 rehearsal."""
    from adaptive_mini_gpt.data.exposure_a7 import repeated_plan

    relational_offset = len_broad
    num_train_pairs = len(train_pairs)

    rel_plan: list[list[int]] = []
    for b_idx, ix in enumerate(single_coverage_plan):
        syn_items = [i for i in ix if i >= synthetic_offset]
        reindexed_syn = [len_broad + (2 * num_train_pairs) + (i - synthetic_offset) for i in syn_items]

        new_items = [i for i in ix if i < synthetic_offset]
        selected_single = new_items[:6]

        p1 = (b_idx * 3) % num_train_pairs
        p2 = (b_idx * 3 + 1) % num_train_pairs
        p3 = (b_idx * 3 + 2) % num_train_pairs
        pair_items = [
            relational_offset + (2 * p1),
            relational_offset + (2 * p1 + 1),
            relational_offset + (2 * p2),
            relational_offset + (2 * p2 + 1),
            relational_offset + (2 * p3),
            relational_offset + (2 * p3 + 1),
        ]

        batch = selected_single + pair_items + reindexed_syn
        assert len(batch) == 18, f"Batch size mismatch: {len(batch)} != 18"
        rel_plan.append(batch)

    return repeated_plan(rel_plan, total_steps), len_broad + (2 * num_train_pairs)


def build_a11r2_balanced_plan(
    single_coverage_plan: list[list[int]],
    len_broad: int,
    mp_train_pairs: list[Any],
    eb_train_pairs: list[Any],
    synthetic_offset: int,
    total_steps: int = 7200,
) -> tuple[list[list[int]], int]:
    """Constructs balanced 24-case batches for Stage A.11-R2:

    6 single-clause, 6 minimal-pair (3 pairs), 6 entity-binding (3 pairs), 6 rehearsal.
    """
    from adaptive_mini_gpt.data.exposure_a7 import repeated_plan

    mp_offset = len_broad
    num_mp_pairs = len(mp_train_pairs)

    eb_offset = len_broad + (2 * num_mp_pairs)
    num_eb_pairs = len(eb_train_pairs)

    syn_offset_in_dataset = eb_offset + (2 * num_eb_pairs)

    rel_plan: list[list[int]] = []
    for b_idx, ix in enumerate(single_coverage_plan):
        syn_items = [i for i in ix if i >= synthetic_offset]
        reindexed_syn = [syn_offset_in_dataset + (i - synthetic_offset) for i in syn_items]

        new_items = [i for i in ix if i < synthetic_offset]
        selected_single = new_items[:6]

        p1_mp = (b_idx * 3) % num_mp_pairs
        p2_mp = (b_idx * 3 + 1) % num_mp_pairs
        p3_mp = (b_idx * 3 + 2) % num_mp_pairs
        mp_items = [
            mp_offset + (2 * p1_mp),
            mp_offset + (2 * p1_mp + 1),
            mp_offset + (2 * p2_mp),
            mp_offset + (2 * p2_mp + 1),
            mp_offset + (2 * p3_mp),
            mp_offset + (2 * p3_mp + 1),
        ]

        p1_eb = (b_idx * 3) % num_eb_pairs
        p2_eb = (b_idx * 3 + 1) % num_eb_pairs
        p3_eb = (b_idx * 3 + 2) % num_eb_pairs
        eb_items = [
            eb_offset + (2 * p1_eb),
            eb_offset + (2 * p1_eb + 1),
            eb_offset + (2 * p2_eb),
            eb_offset + (2 * p2_eb + 1),
            eb_offset + (2 * p3_eb),
            eb_offset + (2 * p3_eb + 1),
        ]

        batch = selected_single + mp_items + eb_items + reindexed_syn
        assert len(batch) == 24, f"Batch size mismatch: {len(batch)} != 24"
        rel_plan.append(batch)

    return repeated_plan(rel_plan, total_steps), syn_offset_in_dataset
