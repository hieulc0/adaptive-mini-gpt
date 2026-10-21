"""Unit tests for M26 Autonomous Curriculum Selection and Multi-Domain Readiness.

Validates:
  1. Lineage Registry integrity and checkpoint immutability.
  2. Expanded language comprehension dataset partitioning and zero test leakage.
  3. Logic reasoning generalization separation (unseen variables vs unseen structures).
  4. Curriculum selector strategies: Round-Robin, Largest Gap, and Progress & Retention Aware.
  5. Dynamic replay scheduling under positive retention drift.
"""

from __future__ import annotations

import pytest

from adaptive_mini_gpt.autonomous.curriculum_selector import (
    CurriculumProposal,
    CurriculumSelector,
    CurriculumStrategy,
    CycleResult,
    InterventionType,
)
from adaptive_mini_gpt.autonomous.lineage_registry import (
    CANONICAL_LINEAGE_REGISTRY,
    LineageRecord,
    LineageRole,
    get_candidate_root_lineage,
)
from adaptive_mini_gpt.data.language_data import (
    ABSTENTION_TARGET,
    TEST_DOCUMENTS,
    TRAIN_DOCUMENTS,
    VAL_DOCUMENTS,
    build_language_benchmark_samples,
    parse_language_completion,
    verify_language_answer,
)
from adaptive_mini_gpt.data.logic_data import (
    CONTRADICTION_TARGET,
    LOGIC_ABSTENTION_TARGET,
    TEST_VARS,
    TRAIN_VARS,
    VAL_VARS,
    LogicGeneralizationType,
    generate_logic_samples_for_split,
    parse_logic_completion,
    verify_logic_answer,
)
from adaptive_mini_gpt.evaluation.multi_domain import (
    BenchmarkSample,
    BenchmarkSplit,
    DomainType,
)


# =====================================================================
# 1. Lineage Registry Integrity Tests (Phase 1)
# =====================================================================


def test_canonical_lineage_registry_integrity():
    assert "m18_operational_planner" in CANONICAL_LINEAGE_REGISTRY
    assert "base_pretrain_seed_42" in CANONICAL_LINEAGE_REGISTRY
    assert "base_pretrain_seed_43" in CANONICAL_LINEAGE_REGISTRY
    assert "base_pretrain_seed_44" in CANONICAL_LINEAGE_REGISTRY

    # Verify root lineages are marked as NEURAL_BASELINE and CANONICAL_CANDIDATE_ROOT
    for seed in (42, 43, 44):
        rec = get_candidate_root_lineage(seed)
        assert rec.role == LineageRole.NEURAL_BASELINE
        assert rec.operational_status == "CANONICAL_CANDIDATE_ROOT"
        assert rec.param_count == 895744
        assert rec.tokenizer_fingerprint == "f698e45e209b7fe6"

    # Verify M18 operational planner is distinct from neural weights
    planner = CANONICAL_LINEAGE_REGISTRY["m18_operational_planner"]
    assert planner.role == LineageRole.OPERATIONAL_SYSTEM
    assert planner.param_count == 0


# =====================================================================
# 2. Expanded Language Dataset & Split Disjointness Tests (Phase 2)
# =====================================================================


def test_expanded_language_dataset_disjointness():
    # 1. Document count check: Train must have at least 8 docs, Val at least 3, Test at least 3
    assert len(TRAIN_DOCUMENTS) >= 8
    assert len(VAL_DOCUMENTS) >= 3
    assert len(TEST_DOCUMENTS) == 3

    train_ids = {d.doc_id for d in TRAIN_DOCUMENTS}
    val_ids = {d.doc_id for d in VAL_DOCUMENTS}
    test_ids = {d.doc_id for d in TEST_DOCUMENTS}

    # Zero ID intersection
    assert len(train_ids.intersection(val_ids)) == 0
    assert len(train_ids.intersection(test_ids)) == 0
    assert len(val_ids.intersection(test_ids)) == 0

    # Zero text or fact leakage into test documents
    for test_doc in TEST_DOCUMENTS:
        for train_doc in TRAIN_DOCUMENTS:
            assert test_doc.topic.lower() != train_doc.topic.lower()
            assert test_doc.text != train_doc.text


def test_expanded_language_sample_generation():
    train_samples = build_language_benchmark_samples(BenchmarkSplit.TRAIN)
    val_samples = build_language_benchmark_samples(BenchmarkSplit.VAL)
    test_samples = build_language_benchmark_samples(BenchmarkSplit.TEST)

    # Substantially expanded diversity: Train >= 50 samples
    assert len(train_samples) >= 50
    assert len(val_samples) >= 20
    assert len(test_samples) == 22

    # Verify answerable vs unanswerable split
    unans_train = [s for s in train_samples if s.is_unanswerable]
    assert len(unans_train) >= 15

    # Test verifier on unanswerable question
    unans_sample = unans_train[0]
    assert verify_language_answer(ABSTENTION_TARGET, unans_sample) is True
    assert verify_language_answer("Random hallucinated answer", unans_sample) is False


# =====================================================================
# 3. Logic Generalization Separation Tests (Phase 2)
# =====================================================================


def test_logic_generalization_separation():
    test_samples = generate_logic_samples_for_split(
        BenchmarkSplit.TEST,
        count_per_category=10,
        include_unseen_structures=True,
    )

    unseen_vars = [
        s for s in test_samples
        if s.metadata.get("generalization_type") == LogicGeneralizationType.UNSEEN_VARIABLES.value
    ]
    unseen_structs = [
        s for s in test_samples
        if s.metadata.get("generalization_type") == LogicGeneralizationType.UNSEEN_STRUCTURES.value
    ]
    contra_samples = [
        s for s in test_samples
        if s.metadata.get("generalization_type") == LogicGeneralizationType.CONTRADICTION_ABSTENTION.value
    ]

    # Both categories must be populated
    assert len(unseen_vars) > 0
    assert len(unseen_structs) > 0
    assert len(contra_samples) > 0

    # Ensure no train variables leak into test prompts
    for s in test_samples:
        for tv in TRAIN_VARS:
            assert f" {tv} " not in s.prompt and f"Item {tv} " not in s.prompt

    # Test unseen structures (Modus Tollens, Disjunctive Syllogism, Biconditional)
    categories = {s.metadata.get("category") for s in unseen_structs}
    assert "modus_tollens" in categories
    assert "disjunctive_syllogism" in categories
    assert "biconditional_equivalence" in categories


# =====================================================================
# 4. Curriculum Selector Strategy Tests (Phase 3)
# =====================================================================


def test_curriculum_selector_round_robin():
    selector = CurriculumSelector(strategy=CurriculumStrategy.ROUND_ROBIN)

    p1 = selector.select_next_experiment(cycle_index=1, current_metrics={"arithmetic": 0.1, "language": 0.0, "logic": 0.5})
    assert p1.target_domain == DomainType.ARITHMETIC
    assert p1.strategy == CurriculumStrategy.ROUND_ROBIN

    p2 = selector.select_next_experiment(cycle_index=2, current_metrics={"arithmetic": 0.2, "language": 0.0, "logic": 0.5})
    assert p2.target_domain == DomainType.LANGUAGE

    p3 = selector.select_next_experiment(cycle_index=3, current_metrics={"arithmetic": 0.2, "language": 0.1, "logic": 0.5})
    assert p3.target_domain == DomainType.LOGIC


def test_curriculum_selector_largest_gap():
    selector = CurriculumSelector(strategy=CurriculumStrategy.LARGEST_GAP, target_accuracy=0.90)

    # Language has 0% accuracy (gap = 0.90), arithmetic has 40% (gap = 0.50), logic has 60% (gap = 0.30)
    p = selector.select_next_experiment(
        cycle_index=1,
        current_metrics={"arithmetic": 0.40, "language": 0.00, "logic": 0.60},
    )
    assert p.target_domain == DomainType.LANGUAGE
    assert p.strategy == CurriculumStrategy.LARGEST_GAP
    assert p.sampling_ratios["language"] == 0.80


def test_curriculum_selector_progress_and_retention_aware():
    selector = CurriculumSelector(
        strategy=CurriculumStrategy.PROGRESS_AND_RETENTION_AWARE,
        target_accuracy=0.90,
    )

    # 1. Normal state with low retention drift
    p_norm = selector.select_next_experiment(
        cycle_index=1,
        current_metrics={"arithmetic": 0.10, "language": 0.00, "logic": 0.50},
        retention_drift=+0.02,
    )
    assert p_norm.replay_ratio == 0.10
    assert p_norm.strategy == CurriculumStrategy.PROGRESS_AND_RETENTION_AWARE

    # 2. Retention emergency: positive drift > 0.10 nats must trigger replay buffering (25%)
    p_drift = selector.select_next_experiment(
        cycle_index=2,
        current_metrics={"arithmetic": 0.15, "language": 0.05, "logic": 0.52},
        retention_drift=+0.42,  # Like observed in M25 Candidate D
    )
    assert p_drift.replay_ratio == 0.25
    assert p_drift.intervention == InterventionType.REPLAY_STABILIZED

    # 3. Test recording outcome and gate verification
    result = selector.record_cycle_outcome(
        proposal=p_norm,
        pre_metrics={"language": 0.00},
        post_metrics={"language": 0.06},
        observed_retention_drift=+0.03,
    )
    assert result.observed_gain == 0.06
    assert result.is_promoted_to_candidate is True
    assert len(selector.cycle_history) == 1
