"""Stage A.12: High-Diversity Entity-Binding & In-Batch Paired Curriculum.

Provides:
1. TRAIN_ENTITIES_A12: 28 synthetic training entity names strictly disjoint
   from development and sealed suites.
2. In-batch counterfactual pair coupling: Minibatches constructed such that both
   halves of each sampled pair (case_a, case_b) are guaranteed to co-occur in the same batch.
3. build_a12_balanced_plan: Balanced 24-case batches with coupled pair placement.
4. build_a12_training_suites: Multi-entity minimal-pair and EB suites with diverse names.
"""

from __future__ import annotations

import random
from typing import Any

from adaptive_mini_gpt.data.clause_minimal_pairs_a11 import (
    TRAIN_ADJS,
    TRAIN_PARTS,
    MinimalPairCase,
    generate_mp1_negation_transfer,
    generate_mp2_multi_entity_binding,
    generate_mp3_long_range_scope,
)
from adaptive_mini_gpt.data.entity_binding_a11r2 import (
    generate_eb1_query_swap,
    generate_eb2_clause_order_swap,
    generate_eb3_property_assignment_swap,
    generate_eb4_three_entity_distractor,
)

# 28 synthetic training entity names (strictly disjoint from dev and sealed suites)
TRAIN_ENTITIES_A12 = [
    "Vim", "Rax", "Kip", "Jor", "Qel", "Zyp", "Mox", "Fex",
    "Hul", "Dak", "Ryn", "Gom", "Pek", "Yor", "Vek", "Zan",
    "Tep", "Lox", "Nok", "Coz", "Biv", "Wup", "Jad", "Fyn",
    "Gek", "Pel", "Mub", "Rix",
]

# Forbidden names that must never enter training entity pools
ALL_DEV_NAMES = {"tuv", "wex", "zen", "bax", "cip", "gid", "tor", "lum", "fen"}
ALL_SEALED_NAMES = {
    "drav", "sorn", "kael", "vond", "lira", "meko",
    "zor", "kex", "plin", "vex", "taro", "nul",
}


def verify_training_names_disjointness(names: list[str]) -> bool:
    """Verifies that none of the provided names overlap with dev or sealed suites."""
    for n in names:
        n_low = n.lower()
        if n_low in ALL_DEV_NAMES:
            raise ValueError(f"Training name '{n}' overlaps with dev entities")
        if n_low in ALL_SEALED_NAMES:
            raise ValueError(f"Training name '{n}' overlaps with sealed entities")
    return True


def build_a12_diverse_mp_train_pairs(
    entities: list[str] | None = None,
    seed: int = 42,
    num_pairs_per_family: int = 32,
) -> list[tuple[MinimalPairCase, MinimalPairCase]]:
    """Builds diverse training minimal pairs across the expanded entity name pool."""
    if entities is None:
        entities = TRAIN_ENTITIES_A12
    verify_training_names_disjointness(entities)

    rng = random.Random(seed)
    pairs: list[tuple[MinimalPairCase, MinimalPairCase]] = []

    # MP-1 Negation transfer
    mp1_full = generate_mp1_negation_transfer(entities[:8], TRAIN_PARTS, TRAIN_ADJS, "train_div")
    if len(mp1_full) >= num_pairs_per_family:
        pairs.extend(mp1_full[:num_pairs_per_family])
    else:
        pairs.extend(mp1_full)

    # MP-2 Multi-entity binding (permuted entity pairs across the full pool)
    mp2_pairs: list[tuple[MinimalPairCase, MinimalPairCase]] = []
    e_pool = list(entities)
    rng.shuffle(e_pool)
    for i in range(0, len(e_pool) - 1, 2):
        sub_entities = [e_pool[i], e_pool[i + 1]]
        mp2_sub = generate_mp2_multi_entity_binding(sub_entities, f"train_div_{i}")
        mp2_pairs.extend(mp2_sub)
        if len(mp2_pairs) >= num_pairs_per_family:
            break
    pairs.extend(mp2_pairs[:num_pairs_per_family])

    # MP-3 Long-range scope
    mp3_full = generate_mp3_long_range_scope(entities[8:16], TRAIN_PARTS, TRAIN_ADJS, "train_div")
    if len(mp3_full) >= num_pairs_per_family:
        pairs.extend(mp3_full[:num_pairs_per_family])
    else:
        pairs.extend(mp3_full)

    return pairs


def build_a12_diverse_eb_train_pairs(
    entities: list[str] | None = None,
    seed: int = 42,
    num_pairs_per_family: int = 36,
) -> list[tuple[Any, Any]]:
    """Builds diverse entity-binding training pairs across the expanded entity name pool."""
    if entities is None:
        entities = TRAIN_ENTITIES_A12
    verify_training_names_disjointness(entities)

    rng = random.Random(seed)
    e_pool = list(entities)
    rng.shuffle(e_pool)

    # Generate each family with rotated subsets of entities
    eb1 = generate_eb1_query_swap(e_pool[:8], "train_div")[:num_pairs_per_family]
    eb2 = generate_eb2_clause_order_swap(e_pool[8:16], "train_div")[:num_pairs_per_family]
    eb3 = generate_eb3_property_assignment_swap(e_pool[16:24], "train_div")[:num_pairs_per_family]
    eb4 = generate_eb4_three_entity_distractor(e_pool, "train_div")[:num_pairs_per_family]

    return eb1 + eb2 + eb3 + eb4


def build_a12_balanced_plan(
    single_coverage_plan: list[list[int]],
    len_broad: int,
    mp_train_pairs: list[Any],
    eb_train_pairs: list[Any],
    synthetic_offset: int,
    total_steps: int = 7200,
) -> tuple[list[list[int]], int]:
    """Constructs balanced 24-case batches with strict in-batch counterfactual pair coupling.

    Batch composition (24 items total):
    - 6 single-clause items (new vocabulary/grammar coverage)
    - 6 minimal-pair items: 3 coupled counterfactual pairs [(A1, B1), (A2, B2), (A3, B3)]
    - 6 entity-binding items: 3 coupled counterfactual pairs [(A4, B4), (A5, B5), (A6, B6)]
    - 6 rehearsal items: historical frames

    Guarantees that both halves of every sampled pair appear in the same minibatch,
    forcing gradient updates through the query target distinction.
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

        # Sample 3 minimal pairs and place both halves in the batch
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

        # Sample 3 entity-binding pairs and place both halves in the batch
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
