"""Unit tests for Milestone M76: Causal Neural Skill Composition and Reusable Procedural Knowledge.

Verifies:
1. Forensic scientific audit of M75 and hindsight oracle diagnosis.
2. Primitive skill definitions, qualification threshold, and parameter freezing.
3. Compound procedural task generation, noncommutativity, and split isolation.
4. Two-stage neural composition pipeline, teacher-forced diagnostic, and error decomposition.
5. Empirical retention safeguards (double SHA-256 base invariance, primitive hash integrity, carry/logic probes, rollback).
"""

from __future__ import annotations

import torch

from adaptive_mini_gpt.model import MiniGPT, ModelConfig
from adaptive_mini_gpt.neural_skill_composition import (
    CompoundTaskGenerator,
    CompoundTaskType,
    M75ScientificAuditor,
    M76SafeguardsAuditor,
    NeuralCompositionPipeline,
    PrimitiveSkillType,
    QualifiedPrimitiveRecord,
    apply_primitive_mapping,
)
from adaptive_mini_gpt.neural_weight_learning.sequence_adapter import Rank16SequenceAdapter


def test_m75_forensic_audit_and_hindsight_diagnosis() -> None:
    """Verify Phase 1 M75 audit metrics, hindsight diagnosis, and paired CIs."""
    audit_report = M75ScientificAuditor.run_full_audit()
    assert audit_report["status"] == "M75_AUDIT_VERIFIED"

    # Reconciled metrics
    rec = audit_report["reconciled_metrics"]
    assert rec["NO_HISTORY_UTILITY"] == 0.5366
    assert rec["RELEVANT_HISTORY_UTILITY"] == 0.5576
    assert rec["SHUFFLED_HISTORY_UTILITY"] == 0.5890
    assert rec["IRRELEVANT_HISTORY_UTILITY"] == 0.4354

    # Hindsight diagnosis
    hindsight = audit_report["hindsight_audit"]
    assert hindsight["verdict"] == "RESTRICTED_DIAGNOSTIC_NOT_TRUE_UPPER_BOUND"
    assert "fixed 40-step budget" in hindsight["root_cause"]

    # Paired Student's t intervals
    intervals = audit_report["paired_intervals"]
    rel_no = [p for p in intervals if p["comparison_id"] == "RELEVANT_vs_NO_HISTORY"][0]
    assert rel_no["spans_zero"] is True
    assert rel_no["verdict"] == "INCONCLUSIVE_SPANS_ZERO"

    rel_shuf = [p for p in intervals if p["comparison_id"] == "RELEVANT_vs_SHUFFLED_HISTORY"][0]
    assert rel_shuf["mean_delta"] < 0.0
    assert rel_shuf["verdict"] == "FAILED_SHUFFLED_SUPERIOR"


def test_compound_task_generation_and_noncommutativity() -> None:
    """Verify Phase 3 & 6 compound tasks, split isolation, and noncommutativity."""
    gen = CompoundTaskGenerator(vocab_size=16)

    # 1. Noncommutative verification: REFLECT_THEN_SHIFT3 vs SHIFT3_THEN_REFLECT
    # Test token: 5
    # Reflect then shift mod 3: 15 - 5 = 10 -> (10 + 3) % 16 = 13
    # Shift mod 3 then reflect: (5 + 3) % 16 = 8 -> 15 - 8 = 7
    # 13 != 7! Noncommutative!
    test_seq = (5, 0, 15)
    r_then_s = apply_primitive_mapping(
        PrimitiveSkillType.CAESAR_SHIFT_MOD3,
        apply_primitive_mapping(PrimitiveSkillType.COMPLEMENT_REFLECT, test_seq),
    )
    s_then_r = apply_primitive_mapping(
        PrimitiveSkillType.COMPLEMENT_REFLECT,
        apply_primitive_mapping(PrimitiveSkillType.CAESAR_SHIFT_MOD3, test_seq),
    )
    assert r_then_s == (13, 2, 3)
    assert s_then_r == (7, 12, 13)
    assert r_then_s != s_then_r

    # 2. Split isolation
    task = gen.generate_compound_task(
        task_id="comp_test_01",
        compound_type=CompoundTaskType.REFLECT_THEN_SHIFT3,
        seed=42,
        n_train=20,
        n_dev=10,
        n_qual=10,
        n_sealed_test=20,
    )
    train_in = {ex.input_tokens for ex in task.train_examples}
    dev_in = {ex.input_tokens for ex in task.dev_examples}
    qual_in = {ex.input_tokens for ex in task.qualification_examples}
    test_in = {ex.input_tokens for ex in task.sealed_test_examples}

    assert train_in.isdisjoint(dev_in)
    assert train_in.isdisjoint(qual_in)
    assert train_in.isdisjoint(test_in)
    assert dev_in.isdisjoint(qual_in)
    assert dev_in.isdisjoint(test_in)
    assert qual_in.isdisjoint(test_in)

    # 3. Ground-truth intermediate verification
    for ex in task.train_examples:
        inter = apply_primitive_mapping(task.stage1_primitive, ex.input_tokens)
        final = apply_primitive_mapping(task.stage2_primitive, inter)
        assert ex.intermediate_tokens == inter
        assert ex.target_tokens == final


def test_neural_composition_pipeline_and_error_diagnostics() -> None:
    """Verify Phase 3 & 5 cascaded composition, teacher-forcing, and error categories."""
    device = torch.device("cpu")
    cfg = ModelConfig(
        vocab_size=100,
        d_model=64,
        n_layers=2,
        n_heads=2,
        d_ff=128,
        context_length=32,
    )
    model = MiniGPT(cfg).to(device)

    ad_a = Rank16SequenceAdapter(d_model=64, bottleneck_dim=16, num_symbols=16).to(device)
    ad_b = Rank16SequenceAdapter(d_model=64, bottleneck_dim=16, num_symbols=16).to(device)

    gen = CompoundTaskGenerator(vocab_size=16)
    task = gen.generate_compound_task(
        task_id="comp_diag_test",
        compound_type=CompoundTaskType.REFLECT_THEN_SHIFT3,
        seed=100,
        n_train=4,
        n_dev=4,
        n_qual=4,
        n_sealed_test=4,
    )

    report = NeuralCompositionPipeline.execute_composition(
        model=model,
        adapter_stage1=ad_a,
        adapter_stage2=ad_b,
        examples=task.sealed_test_examples,
        device=device,
        pipeline_name="TEST_COMPOSITION",
    )

    assert report.num_examples == 4
    assert 0.0 <= report.cascaded_exact_match <= 1.0
    assert 0.0 <= report.token_accuracy <= 1.0
    assert len(report.detailed_diagnostics) == 4

    for diag in report.detailed_diagnostics:
        assert diag.error_category in (
            "SUCCESS",
            "STAGE_1_DEFECT",
            "ERROR_PROPAGATION",
            "STAGE_2_DEFECT",
            "DUAL_DEFECT",
        )
        assert len(diag.predicted_intermediate) == 3
        assert len(diag.predicted_final) == 3
        assert len(diag.teacher_forced_final) == 3


def test_safeguards_primitive_invariance_and_accounting() -> None:
    """Verify Phase 10 & 11 safeguards, primitive freezing, and accounting."""
    cfg = ModelConfig(
        vocab_size=64,
        d_model=64,
        n_layers=2,
        n_heads=2,
        d_ff=128,
        context_length=32,
    )
    model = MiniGPT(cfg)

    # 1. Base model double SHA-256 invariance
    unnamed = M76SafeguardsAuditor.compute_unnamed_param_hash(model)
    named = M76SafeguardsAuditor.compute_named_param_hash(model)
    assert M76SafeguardsAuditor.verify_base_invariance(model, unnamed, named) is True

    # 2. Arithmetic carry & operational logic probes
    assert M76SafeguardsAuditor.verify_arithmetic_carry_retention() == 1.0
    assert M76SafeguardsAuditor.verify_chained_operational_logic() == 1.0

    # 3. Bit-exact rollback
    ad = Rank16SequenceAdapter(d_model=64, bottleneck_dim=16, num_symbols=16)
    assert M76SafeguardsAuditor.verify_bit_exact_rollback(ad) is True

    # 4. Primitive invariance verification
    h1 = ad.compute_parameter_hash()
    rec = QualifiedPrimitiveRecord(
        skill_id="prim_test",
        rule_family="COMPLEMENT_REFLECT",
        adapter=ad,
        qualification_exact_match=0.85,
        qualification_loss=0.45,
        trainable_parameters=ad.count_parameters(),
        parameter_hash=h1,
        training_steps=40,
        is_frozen=True,
    )
    expected_hashes = {"prim_test": h1}
    assert M76SafeguardsAuditor.verify_primitive_invariance({PrimitiveSkillType.COMPLEMENT_REFLECT: rec}, expected_hashes) is True

    # 5. Hardware accounting
    hw = M76SafeguardsAuditor.audit_hardware_accounting(model, num_primitives=3)
    assert hw["composition_additional_parameters"] == 0
    assert hw["monolithic_additional_parameters"] == 2592
    assert hw["parameter_savings_ratio"] == 1.0
