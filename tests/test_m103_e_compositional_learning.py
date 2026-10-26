"""Unit Tests for Milestone M103-E: Compositional Learning & Retention."""

import pytest
import torch

from adaptive_mini_gpt.natural_language_learning_m103.capacity_auditor_m103_e import (
    DynamicCapacityAuditorM103_E,
)
from adaptive_mini_gpt.natural_language_learning_m103.concept_learner_m103_e import (
    ConceptLearnerM103_E,
)
from adaptive_mini_gpt.natural_language_learning_m103.text_student_m103 import (
    TextStudentM103,
)


def test_capacity_audit_and_parameter_identity():
    """Verify forensic parameter reconciliation: 1,178,474 unique vs 1,309,546 registered params."""
    student = TextStudentM103(enable_persistent_growth=True, device=torch.device("cpu"))
    audit = DynamicCapacityAuditorM103_E.audit_student(student)

    # 1. Unique storage parameters (with tied embeddings)
    assert audit.total_unique_params == 1178474
    # 2. Registered parameter objects (including untied base_model.lm_head)
    assert audit.total_registered_params == 1309546
    assert audit.untyped_orphaned_lm_head_params == 131072
    assert audit.discrepancy == 0
    assert audit.reconciliation_certified is True
    assert audit.storage_identity_verified is True
    assert audit.optimizer_deduplication_verified is True
    assert audit.checkpoint_roundtrip_verified is True

    # 3. Verify shared storage between tied student.lm_head and base_model.wte
    assert student.lm_head.weight.data_ptr() == student.base_model.wte.weight.data_ptr()


def test_balanced_parity_dataset_partitioning():
    """Verify balanced parity training and strictly disjoint held-out validation."""
    learner = ConceptLearnerM103_E()
    datasets = learner.build_m103_e_datasets(mode="raw_bytes")

    train_items = datasets["train"]
    val_in_range = datasets["val_in_range"]
    test_extrap = datasets["test_extrapolation"]

    train_nums = {learner.extract_numeric_value(it.question) for it in train_items if it.question}
    val_nums = {learner.extract_numeric_value(it.question) for it in val_in_range if it.question}
    extrap_nums = {learner.extract_numeric_value(it.question) for it in test_extrap if it.question}

    # Verify train set contains BOTH odd and even values: {2, 3, 4, 6, 7, 8, 10, 11, 12, 14}
    assert any(n % 2 == 0 for n in train_nums if n is not None)
    assert any(n % 2 == 1 for n in train_nums if n is not None)

    # Verify in-range held-out set is {5, 9, 13}
    assert val_nums == {5, 9, 13}
    # Zero overlap between train and in-range held-out numbers
    assert not train_nums.intersection(val_nums)

    # Verify extrapolation is outside [2..14]
    assert all(n > 14 for n in extrap_nums if n is not None)
    assert not train_nums.intersection(extrap_nums)


def test_forensic_audit_leakage_mitigation():
    """Audit mitigation: verify unseen numbers for renamed entities and disjoint corrective splits."""
    learner = ConceptLearnerM103_E()
    datasets = learner.build_m103_e_datasets(mode="raw_bytes")

    # 1. Renamed entities in test_hypothetical MUST use unseen numbers [7, 13, 19]
    hypo_items = datasets["test_hypothetical"]
    renamed_items = [it for it in hypo_items if it.entity_name in ["monolith", "prism"]]
    renamed_nums = {learner.extract_numeric_value(it.question) for it in renamed_items}
    assert renamed_nums == {7, 13, 19}

    # 2. Corrective splits MUST be disjoint: train [2, 4, 6] vs held-out [8, 10, 12]
    corr_train_nums = {learner.extract_numeric_value(it.question) for it in datasets["corrective_train"]}
    corr_val_nums = {learner.extract_numeric_value(it.question) for it in datasets["corrective_held_out"]}
    assert corr_train_nums == {2, 4, 6}
    assert corr_val_nums == {8, 10, 12}
    assert not corr_train_nums.intersection(corr_val_nums)


def test_representation_modes_and_context_length_compliance():
    """Verify all representation modes format valid sequences within the 256 context length."""
    learner = ConceptLearnerM103_E()
    modes = ["raw_bytes", "structured_tokens", "scratchpad_compositional"]

    for mode in modes:
        datasets = learner.build_m103_e_datasets(mode=mode)
        for cat, items in datasets.items():
            for it in items:
                assert len(it.full_tokens) > 0
                assert len(it.full_tokens) <= 256, f"Sequence exceeded 256 tokens in mode {mode}, cat {cat}"
                assert it.prompt_len > 0
                assert it.prompt_len < len(it.full_tokens)


def test_learning_to_learn_dataset_integrity():
    """Verify Grid-Beta navigation dataset items format and disjoint train/held-out split."""
    learner = ConceptLearnerM103_E()
    datasets = learner.build_m103_e_datasets()

    grid_train = datasets["grid_train"]
    grid_held_out = datasets["grid_held_out"]

    train_xs = {learner.extract_numeric_value(it.question) for it in grid_train}
    test_xs = {learner.extract_numeric_value(it.question) for it in grid_held_out}

    assert len(train_xs) > 0
    assert len(test_xs) > 0
    assert not train_xs.intersection(test_xs)
    assert all("coord" in it.canonical_answer for it in grid_train)
    assert all("coord" in it.canonical_answer for it in grid_held_out)
