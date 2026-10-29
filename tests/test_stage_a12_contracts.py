"""Stage A.12: Stable Entity Binding & Autonomous Practice Local Contracts.

Local-safe tests enforcing:
1. Query-sensitivity diagnostic math (isolating query blindness vs sensitive routing).
2. Attention margin computation on synthetic attention weight tensors.
3. Expanded training entity name pool (28 names) strict disjointness from dev and sealed suites.
4. In-batch counterfactual pair coupling: Minibatch construction guarantees that
   both halves of every sampled pair (case_a, case_b) co-occur in the same batch.
5. Strict per-seed qualification gate evaluation: Gate 3C and 3D fail if any single seed collapses.
6. Query-span cross readout prefix masking logic on synthetic tokens (0 neural forward/backward passes).
7. Pretraining corpus isolation with repeated-name induction patterns.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

from adaptive_mini_gpt.data.clause_minimal_pairs_a11 import MinimalPairCase
from adaptive_mini_gpt.data.entity_binding_a12 import (
    ALL_DEV_NAMES,
    ALL_SEALED_NAMES,
    TRAIN_ENTITIES_A12,
    build_a12_balanced_plan,
    build_a12_diverse_eb_train_pairs,
    build_a12_diverse_mp_train_pairs,
    verify_training_names_disjointness,
)
from adaptive_mini_gpt.data.pretraining_a12 import (
    generate_a12_relational_pretraining_corpus,
    verify_a12_pretraining_corpus_isolation,
)

SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from diagnose_stage_a12_seeds import compute_query_sensitivity  # noqa: E402 # pyright: ignore[reportMissingImports]
from run_stage_a12 import check_a12_qualification_gates  # noqa: E402 # pyright: ignore[reportMissingImports]


def test_query_sensitivity_calculation():
    """Verifies that query-sensitivity math correctly detects query blindness vs sensitivity."""
    case_1a = MinimalPairCase(
        case_id="case_1a",
        question="Prefix text. Query target: Alpha.",
        statuses=("positive", "negated"),
        labels=(1, 0),
        family="MP-2",
        pair_id="p1",
        member="A",
        entity="Alpha",
        distractor_part="shell",
        distractor_adj="dense",
    )
    case_1b = MinimalPairCase(
        case_id="case_1b",
        question="Prefix text. Query target: Beta.",
        statuses=("negated", "positive"),
        labels=(0, 1),
        family="MP-2",
        pair_id="p1",
        member="B",
        entity="Beta",
        distractor_part="shell",
        distractor_adj="dense",
    )

    case_2a = MinimalPairCase(
        case_id="case_2a",
        question="Prefix 2 text. Query target: Gamma.",
        statuses=("positive", "negated"),
        labels=(1, 0),
        family="MP-2",
        pair_id="p2",
        member="A",
        entity="Gamma",
        distractor_part="panel",
        distractor_adj="heavy",
    )
    case_2b = MinimalPairCase(
        case_id="case_2b",
        question="Prefix 2 text. Query target: Delta.",
        statuses=("negated", "positive"),
        labels=(0, 1),
        family="MP-2",
        pair_id="p2",
        member="B",
        entity="Delta",
        distractor_part="panel",
        distractor_adj="heavy",
    )

    pairs = [(case_1a, case_1b), (case_2a, case_2b)]

    # Scenario 1: Query blind (predicts [1, 0] for all cases regardless of target)
    blind_preds = {
        "case_1a": [1, 0],
        "case_1b": [1, 0],
        "case_2a": [1, 0],
        "case_2b": [1, 0],
    }
    sens_blind = compute_query_sensitivity(pairs, blind_preds)
    assert sens_blind["query_sensitivity_rate"] == 0.0
    assert sens_blind["query_blind_rate"] == 1.0
    assert sens_blind["query_blind_pairs"] == 2

    # Scenario 2: Fully sensitive (predicts flipped labels when target changes)
    sensitive_preds = {
        "case_1a": [1, 0],
        "case_1b": [0, 1],
        "case_2a": [1, 0],
        "case_2b": [0, 1],
    }
    sens_active = compute_query_sensitivity(pairs, sensitive_preds)
    assert sens_active["query_sensitivity_rate"] == 1.0
    assert sens_active["query_blind_rate"] == 0.0
    assert sens_active["query_sensitive_pairs"] == 2


def test_attention_margin_computation_synthetic():
    """Verifies that attention margin calculation (target - distractor) computes correct differences."""
    # Synthetic attention matrix: (B=1, H=2, T=5, T=5)
    attn = torch.zeros((1, 2, 5, 5))
    # Query token is at position 4. It attends to target tokens [1] and distractor tokens [2].
    # Head 0: attends 0.6 to target (idx 1), 0.2 to distractor (idx 2) -> margin = +0.4
    attn[0, 0, 4, 1] = 0.6
    attn[0, 0, 4, 2] = 0.2

    # Head 1: attends 0.1 to target (idx 1), 0.5 to distractor (idx 2) -> margin = -0.4
    attn[0, 1, 4, 1] = 0.1
    attn[0, 1, 4, 2] = 0.5

    alpha_query = attn[0, :, 4, :]  # (2, 5)
    target_idx = [1]
    other_idx = [2]

    margin_head0 = (alpha_query[0, target_idx].mean() - alpha_query[0, other_idx].mean()).item()
    margin_head1 = (alpha_query[1, target_idx].mean() - alpha_query[1, other_idx].mean()).item()

    assert pytest.approx(margin_head0, abs=1e-5) == 0.4
    assert pytest.approx(margin_head1, abs=1e-5) == -0.4


def test_expanded_training_entities_disjointness():
    """Verifies that the expanded 28 training entity names are strictly disjoint from dev and sealed names."""
    assert len(TRAIN_ENTITIES_A12) >= 24
    assert len(TRAIN_ENTITIES_A12) == 28

    # Assert no overlaps
    for name in TRAIN_ENTITIES_A12:
        assert name.lower() not in ALL_DEV_NAMES, f"Overlap with dev: {name}"
        assert name.lower() not in ALL_SEALED_NAMES, f"Overlap with sealed: {name}"

    assert verify_training_names_disjointness(TRAIN_ENTITIES_A12)

    # Injected violation test
    with pytest.raises(ValueError, match="overlaps with dev"):
        verify_training_names_disjointness(TRAIN_ENTITIES_A12 + ["Tuv"])

    with pytest.raises(ValueError, match="overlaps with sealed"):
        verify_training_names_disjointness(TRAIN_ENTITIES_A12 + ["Drav"])


def test_in_batch_counterfactual_pair_coupling():
    """Verifies that build_a12_balanced_plan guarantees both pair halves co-occur in the same batch."""
    # Synthetic coverage plan with 10 batches of 12 items
    cov_plan = [[j for j in range(6)] + [50 + j for j in range(6)] for _ in range(10)]
    len_broad = 50

    mp_pairs = build_a12_diverse_mp_train_pairs(TRAIN_ENTITIES_A12, seed=42)
    eb_pairs = build_a12_diverse_eb_train_pairs(TRAIN_ENTITIES_A12, seed=42)

    plan, _ = build_a12_balanced_plan(
        single_coverage_plan=cov_plan,
        len_broad=len_broad,
        mp_train_pairs=mp_pairs,
        eb_train_pairs=eb_pairs,
        synthetic_offset=len_broad,
        total_steps=10,
    )

    mp_offset = len_broad
    eb_offset = len_broad + (2 * len(mp_pairs))

    for b_idx, batch in enumerate(plan):
        assert len(batch) == 24, f"Batch {b_idx} size is not 24: {len(batch)}"

        # Check MP items co-occurrence
        mp_items = [idx for idx in batch if mp_offset <= idx < eb_offset]
        assert len(mp_items) == 6, f"Batch {b_idx} does not have 6 MP items"
        # Each pair (2*p, 2*p + 1) must co-occur
        for k in range(0, 6, 2):
            item_a = mp_items[k]
            item_b = mp_items[k + 1]
            assert item_b == item_a + 1, f"Coupled pair mismatch: {item_a}, {item_b}"
            assert (item_a - mp_offset) % 2 == 0, f"Unpaired start: {item_a}"

        # Check EB items co-occurrence
        eb_items = [idx for idx in batch if eb_offset <= idx < eb_offset + (2 * len(eb_pairs))]
        assert len(eb_items) == 6, f"Batch {b_idx} does not have 6 EB items"
        for k in range(0, 6, 2):
            item_a = eb_items[k]
            item_b = eb_items[k + 1]
            assert item_b == item_a + 1, f"Coupled EB pair mismatch: {item_a}, {item_b}"
            assert (item_a - eb_offset) % 2 == 0, f"Unpaired EB start: {item_a}"


def test_per_seed_qualification_gate_evaluator():
    """Verifies that check_a12_qualification_gates requires ALL individual seeds to pass Gate 3C."""
    # Scenario 1: Historical Stage A.11-R2 failure (Seed 42: 75%, Seed 43: 87.5%, Seed 44: 0.0%)
    failing_cohort = [
        {
            "training": {"exact": 1.0},
            "old_known": {"exact": 1.0},
            "old_renamed": {"exact": 1.0},
            "development_familiar": {"exact": 0.90},
            "development_novel": {"exact": 0.90},
            "dev_minimal_pairs": {
                "paired_metrics": {
                    "pair_both_correct_acc": 0.875,
                    "by_family": {"MP-2": {"pair_both_correct_acc": 0.75}},
                }
            },
            "dev_entity_binding": {
                "paired_metrics": {
                    "pair_both_correct_acc": 0.82,
                    "by_family": {
                        "EB-1_QUERY_SWAP": {"pair_both_correct_acc": 0.80},
                        "EB-2_CLAUSE_ORDER_SWAP": {"pair_both_correct_acc": 0.80},
                        "EB-3_PROPERTY_ASSIGNMENT_SWAP": {"pair_both_correct_acc": 0.80},
                        "EB-4_THREE_ENTITY_DISTRACTOR": {"pair_both_correct_acc": 0.80},
                    },
                }
            },
        },
        {
            "training": {"exact": 1.0},
            "old_known": {"exact": 1.0},
            "old_renamed": {"exact": 1.0},
            "development_familiar": {"exact": 0.90},
            "development_novel": {"exact": 0.90},
            "dev_minimal_pairs": {
                "paired_metrics": {
                    "pair_both_correct_acc": 0.958,
                    "by_family": {"MP-2": {"pair_both_correct_acc": 0.875}},
                }
            },
            "dev_entity_binding": {
                "paired_metrics": {
                    "pair_both_correct_acc": 0.85,
                    "by_family": {
                        "EB-1_QUERY_SWAP": {"pair_both_correct_acc": 0.85},
                        "EB-2_CLAUSE_ORDER_SWAP": {"pair_both_correct_acc": 0.85},
                        "EB-3_PROPERTY_ASSIGNMENT_SWAP": {"pair_both_correct_acc": 0.85},
                        "EB-4_THREE_ENTITY_DISTRACTOR": {"pair_both_correct_acc": 0.85},
                    },
                }
            },
        },
        {  # Collapsed Seed 44
            "training": {"exact": 1.0},
            "old_known": {"exact": 1.0},
            "old_renamed": {"exact": 1.0},
            "development_familiar": {"exact": 0.90},
            "development_novel": {"exact": 0.90},
            "dev_minimal_pairs": {
                "paired_metrics": {
                    "pair_both_correct_acc": 0.667,
                    "by_family": {"MP-2": {"pair_both_correct_acc": 0.00}},
                }
            },
            "dev_entity_binding": {
                "paired_metrics": {
                    "pair_both_correct_acc": 0.00,
                    "by_family": {
                        "EB-1_QUERY_SWAP": {"pair_both_correct_acc": 0.00},
                        "EB-2_CLAUSE_ORDER_SWAP": {"pair_both_correct_acc": 0.00},
                        "EB-3_PROPERTY_ASSIGNMENT_SWAP": {"pair_both_correct_acc": 0.00},
                        "EB-4_THREE_ENTITY_DISTRACTOR": {"pair_both_correct_acc": 0.00},
                    },
                }
            },
        },
    ]

    gates_fail, _ = check_a12_qualification_gates(failing_cohort)
    assert not gates_fail["gate_3c_multi_entity_binding_mp2"], "Gate 3C must fail when Seed 44 collapses"
    assert not gates_fail["gate_3d_entity_binding_suite"], "Gate 3D must fail when Seed 44 collapses"

    # Scenario 2: All seeds passing
    passing_cohort = [
        dict(failing_cohort[0]),
        dict(failing_cohort[1]),
        {
            "training": {"exact": 1.0},
            "old_known": {"exact": 1.0},
            "old_renamed": {"exact": 1.0},
            "development_familiar": {"exact": 0.88},
            "development_novel": {"exact": 0.88},
            "dev_minimal_pairs": {
                "paired_metrics": {
                    "pair_both_correct_acc": 0.875,
                    "by_family": {"MP-2": {"pair_both_correct_acc": 0.8125}},
                }
            },
            "dev_entity_binding": {
                "paired_metrics": {
                    "pair_both_correct_acc": 0.83,
                    "by_family": {
                        "EB-1_QUERY_SWAP": {"pair_both_correct_acc": 0.80},
                        "EB-2_CLAUSE_ORDER_SWAP": {"pair_both_correct_acc": 0.80},
                        "EB-3_PROPERTY_ASSIGNMENT_SWAP": {"pair_both_correct_acc": 0.80},
                        "EB-4_THREE_ENTITY_DISTRACTOR": {"pair_both_correct_acc": 0.80},
                    },
                }
            },
        },
    ]

    gates_pass, _ = check_a12_qualification_gates(passing_cohort)
    assert all(gates_pass.values()), f"All gates must pass when all seeds qualify: {gates_pass}"


def test_query_span_prefix_extraction_on_synthetic_tokens():
    """Verifies that prefix masking on synthetic tokens accurately finds query markers."""
    q_id = 260
    a_id = 261
    
    # Construct prompt: [QUESTION_ID, "Object Alpha is red. Query target: Alpha.", ANSWER_ID]
    prefix_text = "Object Alpha is red. "
    query_text = "Query target: Alpha."

    row = [q_id] + [ord(c) for c in prefix_text + query_text] + [a_id]
    tokens = torch.tensor([row], dtype=torch.long)
    B, L = tokens.size()

    prefix_mask = torch.zeros((B, L), dtype=torch.bool)
    query_marker_bytes = [
        b"Query target",
        b"Target is",
        b"Analysis focuses on",
        b"The evaluated target",
        b"Focus on",
    ]

    row_list = tokens[0].tolist()
    q_locs = [i for i, tok in enumerate(row_list) if tok == q_id]
    a_locs = [i for i, tok in enumerate(row_list) if tok == a_id]
    assert q_locs and a_locs

    q_start = q_locs[0] + 1
    q_end = a_locs[0]
    q_slice = bytes([t for t in row_list[q_start:q_end] if 0 <= t < 256])

    earliest_pos = None
    for marker in query_marker_bytes:
        pos = q_slice.find(marker)
        if pos != -1:
            if earliest_pos is None or pos < earliest_pos:
                earliest_pos = pos

    assert earliest_pos is not None
    assert earliest_pos == len(prefix_text.encode("utf-8"))
    prefix_end = q_start + earliest_pos
    prefix_mask[0, q_start:prefix_end] = True

    # Assert that all prefix bytes are True, and query bytes are False
    prefix_indices = list(range(q_start, prefix_end))
    query_indices = list(range(prefix_end, q_end))

    assert prefix_mask[0, prefix_indices].all()
    assert not prefix_mask[0, query_indices].any()


def test_a12_pretraining_corpus_isolation():
    """Verifies that generate_a12_relational_pretraining_corpus passes isolation tests."""
    corpus = generate_a12_relational_pretraining_corpus(100, seed=42, induction_ratio=0.5)
    iso = verify_a12_pretraining_corpus_isolation(corpus)
    assert iso["status"] == "PASS"
    assert iso["num_sentences"] == 100
    assert iso["forbidden_words_checked"] > 30
