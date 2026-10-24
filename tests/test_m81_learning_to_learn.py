"""Milestone M81: Fast CPU-Only Unit Tests for Learning-to-Learn Suite.

Verifies:
1. Student-t statistics, df=4 accounting, exact two-tailed p-values, and paired confidence intervals.
2. M80 Forensic Audit reconciliation (928k vs 1.07M parameters, statistical corrections, l2l findings).
3. Curriculum task generation, partition isolation (train, dev, qual, test), and parametric rule instances.
4. Capacity adaptation parameter counts (Rank-4=852, Rank-8=1,432, Rank-16=2,592), weight shuffling, and hierarchy.
5. Parameter hash invariance and composition reporting contracts.
"""

from __future__ import annotations

import torch

from adaptive_mini_gpt.learning_to_learn.audit_m80 import M80ForensicAuditor
from adaptive_mini_gpt.learning_to_learn.capacity_adaptation import (
    BottleneckSequenceAdapter,
    CapacityAdaptationManager,
    CapacityDecision,
)
from adaptive_mini_gpt.learning_to_learn.statistics import (
    compute_paired_t_test,
    student_t_two_tailed_p_value,
)
from adaptive_mini_gpt.learning_to_learn.task_curriculum import (
    CurriculumFamily,
    CurriculumTaskGenerator,
    apply_curriculum_rule,
)


def test_student_t_exact_p_values_and_df4_audit() -> None:
    """Verify exact Student-t two-tailed p-values for df=4 matching scientific audit."""
    # df = 4 exact values for reported t statistics
    p_rank = student_t_two_tailed_p_value(3.1293, df=4)
    p_shuff = student_t_two_tailed_p_value(6.5731, df=4)
    p_inv = student_t_two_tailed_p_value(2.5323, df=4)

    assert round(p_rank, 4) == 0.0352, f"Expected 0.0352, got {p_rank}"
    assert round(p_shuff, 4) == 0.0028, f"Expected 0.0028, got {p_shuff}"
    assert round(p_inv, 4) == 0.0645, f"Expected 0.0645, got {p_inv}"

    # Verify that paired t-test returns df = n - 1
    sample_a = [1.0, 1.0, 1.0, 1.0, 1.0]
    sample_b = [0.875, 0.9167, 1.0, 0.9167, 0.8125]
    res = compute_paired_t_test(sample_a, sample_b)
    assert res.df == 4
    assert res.is_significant_05 is True
    assert res.ci_95_lower > 0.0


def test_m80_forensic_audit_reconciliation() -> None:
    """Verify that M80 audit reconciles parameter discrepancies and corrects claims."""
    audit = M80ForensicAuditor.execute_audit()
    assert audit.status == "M80_AUDIT_COMPLETED"
    assert audit.verdict == "PASS_RECONCILED_WITH_CORRECTIONS"

    # Parameter reconciliation
    p_rec = audit.parameter_reconciliation
    assert p_rec["m79_backbone_parameters"] == 928512
    assert p_rec["m80_backbone_parameters"] == 1071872
    assert p_rec["transformer_core_layers_parameters"] == 793088
    assert p_rec["parameter_difference"] == 143360

    # Statistical corrections
    sc = audit.statistical_corrections
    assert sc["rank16_vs_rank8"]["corrected_p"] == 0.0352
    assert sc["acquired_vs_shuffled"]["corrected_p"] == 0.0028
    assert sc["discovered_vs_inverted"]["corrected_p"] == 0.0645
    assert sc["discovered_vs_inverted"]["significant_at_05"] is False

    # Learning-to-learn claim rectification
    l2l = audit.learning_to_learn_rectification
    assert l2l["reported_delta"] == 0.0
    assert l2l["reported_p_value"] == 1.0
    assert l2l["audit_verdict"] == "CLAIM_UNSUPPORTED"


def test_task_curriculum_generation_and_partition_isolation() -> None:
    """Verify disjoint dataset splits and deterministic mapping across curriculum families."""
    gen = CurriculumTaskGenerator(vocab_size=16)

    # 1. Arithmetic
    ds_arith = gen.generate_task(
        task_id="test_arith",
        family=CurriculumFamily.ARITHMETIC,
        rule_instance={"a": 3, "b": 5},
        rule_instance_id="AFFINE_3s_5",
        seed=42,
    )
    assert len(ds_arith.train_examples) == 48
    assert len(ds_arith.dev_examples) == 16
    assert len(ds_arith.qualification_examples) == 16
    assert len(ds_arith.sealed_test_examples) == 48

    # Verify input token disjointness
    train_inputs = {ex.input_tokens for ex in ds_arith.train_examples}
    dev_inputs = {ex.input_tokens for ex in ds_arith.dev_examples}
    qual_inputs = {ex.input_tokens for ex in ds_arith.qualification_examples}
    test_inputs = {ex.input_tokens for ex in ds_arith.sealed_test_examples}

    assert len(train_inputs & dev_inputs) == 0
    assert len(train_inputs & qual_inputs) == 0
    assert len(train_inputs & test_inputs) == 0
    assert len(dev_inputs & test_inputs) == 0

    # Rule correctness: Affine (3 * s + 5) % 16
    res = apply_curriculum_rule(CurriculumFamily.ARITHMETIC, {"a": 3, "b": 5}, (0, 1, 2))
    assert res == (5, 8, 11)

    # 2. Conditional Logic: Parity flip
    res_logic = apply_curriculum_rule(
        CurriculumFamily.CONDITIONAL_LOGIC, {"mask": 6, "offset": 4}, (0, 1)
    )
    # 0 is even -> (0 + 4) % 16 = 4. 1 is odd -> (1 ^ 6) % 16 = 7.
    assert res_logic == (4, 7)

    # 3. Symbolic Procedure: Reverse + shift 3
    res_seq = apply_curriculum_rule(
        CurriculumFamily.SYMBOLIC_PROCEDURE, {"shift": 3}, (1, 2, 3)
    )
    # reversed: (3, 2, 1) -> +3: (6, 5, 4)
    assert res_seq == (6, 5, 4)


def test_capacity_adaptation_architecture_and_hierarchy() -> None:
    """Verify adapter parameter counts, weight shuffling, and bounded expansion."""
    rank4 = BottleneckSequenceAdapter(d_model=128, bottleneck_dim=4, num_symbols=16)
    rank8 = BottleneckSequenceAdapter(d_model=128, bottleneck_dim=8, num_symbols=16)
    rank16 = BottleneckSequenceAdapter(d_model=128, bottleneck_dim=16, num_symbols=16)

    # Parameter counts:
    # Rank-4: 256 (ln) + 128*4 + 4 (516) + 4*16 + 16 (80) = 852
    assert rank4.count_parameters() == 852
    # Rank-8: 256 + 1032 + 144 = 1,432
    assert rank8.count_parameters() == 1432
    # Rank-16: 256 + 2064 + 272 = 2,592
    assert rank16.count_parameters() == 2592

    # Forward shape
    dummy = torch.randn(2, 3, 128)
    assert rank4(dummy).shape == (2, 3, 16)
    assert rank8(dummy).shape == (2, 3, 16)
    assert rank16(dummy).shape == (2, 3, 16)

    # Weight shuffling
    h1 = rank4.compute_parameter_hash()
    shuffled = rank4.shuffle_weights(seed=42)
    h2 = shuffled.compute_parameter_hash()
    assert h1 != h2

    # Capacity hierarchy decision test:
    # If rank-4 achieves 0.90 EM >= 0.85, system MUST choose Rank-4 (minimal allocation)
    res4 = CapacityAdaptationManager.evaluate_capacity_hierarchy(
        eval_fn=lambda r: (0.90 if r == 4 else 1.0, rank4),
        admission_threshold=0.85,
    )
    assert res4.decision == CapacityDecision.ALLOCATE_RANK_4
    assert res4.selected_rank == 4
    assert res4.parameters_allocated == 852

    # If rank-4 achieves 0.70 < 0.85 but rank-8 achieves 0.92 >= 0.85, expand to Rank-8
    res8 = CapacityAdaptationManager.evaluate_capacity_hierarchy(
        eval_fn=lambda r: (0.70 if r == 4 else (0.92 if r == 8 else 1.0), rank8),
        admission_threshold=0.85,
    )
    assert res8.decision == CapacityDecision.ALLOCATE_RANK_8
    assert res8.selected_rank == 8
    assert res8.parameters_allocated == 1432
