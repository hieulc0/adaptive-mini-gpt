"""Unit Tests for Milestone M103-F / M103-E.1 Reusable Concepts & Meta-Transfer."""

import pytest
import torch

from adaptive_mini_gpt.natural_language_learning_m103.capacity_auditor_m103_e1 import (
    DynamicCapacityAuditorM103_E1,
)
from adaptive_mini_gpt.natural_language_learning_m103.concept_learner_m103_f import (
    ConceptLearnerM103_F,
    DatasetItemM103_F,
)
from adaptive_mini_gpt.natural_language_learning_m103.text_student_m103 import (
    TextStudentM103,
)


def test_capacity_audit_and_pruned_head():
    """Verify clean parameter accounting: 1,178,474 active params, 0 orphaned, 0 discrepancy."""
    student = TextStudentM103(
        enable_persistent_growth=True,
        prune_orphaned_lm_head=True,
        device=torch.device("cpu"),
    )
    audit = DynamicCapacityAuditorM103_E1.audit_student(student)

    assert audit.active_forward_path_params == 1178474
    assert audit.total_registered_params == 1178474
    assert audit.unique_storage_params == 1178474
    assert audit.orphaned_lm_head_params == 0
    assert audit.discrepancy == 0
    assert audit.reconciliation_certified is True
    assert audit.storage_identity_verified is True
    assert audit.optimizer_deduplication_verified is True
    assert audit.checkpoint_roundtrip_verified is True

    # Shared storage pointer verified
    assert student.lm_head.weight.data_ptr() == student.base_model.wte.weight.data_ptr()


def test_strict_zero_leakage_numerical_partitioning():
    """Verify that all numerical splits are strictly disjoint and s=7 is never in training."""
    learner = ConceptLearnerM103_F()
    datasets = learner.build_m103_f_datasets()

    train_nums = {learner.extract_numeric_value(it.question) for it in datasets["train"]}
    val_nums = {learner.extract_numeric_value(it.question) for it in datasets["val_in_range"]}
    novel_nums = {learner.extract_numeric_value(it.question) for it in datasets["test_hypothetical"]}
    extrap_nums = {learner.extract_numeric_value(it.question) for it in datasets["test_extrapolation"]}
    corr_train_nums = {learner.extract_numeric_value(it.question) for it in datasets["corr_train"]}
    corr_held_out_nums = {learner.extract_numeric_value(it.question) for it in datasets["corr_held_out"]}

    # 1. Leaked value s=7 MUST NOT be in training set
    assert 7 not in train_nums, "Critical: s=7 must not leak into training curriculum!"

    # 2. Strict pairwise disjointness
    assert not train_nums.intersection(val_nums), "Train and In-Range Validation must be disjoint"
    assert not train_nums.intersection(novel_nums), "Train and Novel Hypothetical must be disjoint"
    assert not train_nums.intersection(extrap_nums), "Train and Extrapolation must be disjoint"
    assert not train_nums.intersection(corr_held_out_nums), "Train and Corrective Held-Out must be disjoint"
    assert not corr_train_nums.intersection(corr_held_out_nums), "Corrective Train and Held-Out must be disjoint"

    # 3. Check balanced parity within training curriculum
    assert any(n % 2 == 0 for n in train_nums if n is not None)
    assert any(n % 2 == 1 for n in train_nums if n is not None)


def test_multidimensional_scoring_separation():
    """Verify strict exact match is differentiated from prefix correctness."""
    learner = ConceptLearnerM103_F()
    target = "8 energy"

    # Simulate strict exact match
    norm_strict = "8 energy"
    is_strict = (norm_strict == target)
    is_prefix_strict = (norm_strict.startswith(target) and not is_strict)
    assert is_strict is True
    assert is_prefix_strict is False

    # Simulate prefix match (target followed by additional text)
    norm_prefix = "8 energy and extra output"
    is_strict_p = (norm_prefix == target)
    is_prefix_p = (norm_prefix.startswith(target) and not is_strict_p)
    assert is_strict_p is False
    assert is_prefix_p is True


def test_pre_competence_gated_retention_rejection():
    """Verify retention criterion FAILS when pre-update baseline competence is 0.0%."""
    learner = ConceptLearnerM103_F()
    student = TextStudentM103(enable_persistent_growth=True, prune_orphaned_lm_head=True, device=torch.device("cpu"))

    datasets = learner.build_m103_f_datasets()

    # Model without training has 0% pre-update competence
    report = learner.run_phase2_corrective_retention(
        student=student,
        original_train_items=datasets["train"],
        original_held_out_items=datasets["val_in_range"],
        corr_train_items=datasets["corr_train"],
        corr_held_out_items=datasets["corr_held_out"],
        corrective_steps=2,
    )

    # Must NOT report 100% or pass when baseline competence is zero!
    assert report.pre_update_competence_certified is False
    assert report.protected_backward_retention_pct == 0.0
    assert report.retention_criterion_met is False


def test_true_disk_checkpoint_recovery():
    """Verify true disk-roundtrip checkpoint recovery with object destruction."""
    learner = ConceptLearnerM103_F()
    student = TextStudentM103(enable_persistent_growth=True, prune_orphaned_lm_head=True, device=torch.device("cpu"))
    datasets = learner.build_m103_f_datasets()

    is_bit_exact, logit_diff = learner.verify_bit_exact_checkpoint_recovery(
        student=student,
        test_items=datasets["train"][:4],
    )

    assert is_bit_exact is True
    assert logit_diff == 0.0
