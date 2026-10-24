"""Deterministic unit tests for Milestone M77: Autonomous Neural Skill-Plan Discovery."""

from __future__ import annotations

import torch

from adaptive_mini_gpt.model import MiniGPT, ModelConfig
from adaptive_mini_gpt.neural_plan_discovery import (
    AbstentionEvaluator,
    AutonomousPlanDiscoveryEngine,
    CandidatePlan,
    CausalPlanBenchmark,
    IdentifiableTaskDataset,
    M76ScientificAuditor,
    M77SafeguardsAuditor,
    PlanIdentifiabilityGenerator,
    PlanSearchResult,
    PlanSpace,
    PlanType,
)
from adaptive_mini_gpt.neural_skill_composition.compound_tasks import CompoundTaskType
from adaptive_mini_gpt.neural_skill_composition.primitive_skills import (
    PrimitiveSkillType,
    QualifiedPrimitiveRecord,
)
from adaptive_mini_gpt.neural_weight_learning.sequence_adapter import Rank16SequenceAdapter


def _build_dummy_primitives() -> dict[PrimitiveSkillType, QualifiedPrimitiveRecord]:
    """Construct mock qualified primitive records for deterministic CPU unit testing."""
    repertoire: dict[PrimitiveSkillType, QualifiedPrimitiveRecord] = {}
    for p_type in [
        PrimitiveSkillType.COMPLEMENT_REFLECT,
        PrimitiveSkillType.CAESAR_SHIFT_MOD3,
        PrimitiveSkillType.AFFINE_TRANSFORM,
    ]:
        adapter = Rank16SequenceAdapter(d_model=64, bottleneck_dim=16, num_symbols=16)
        for p in adapter.parameters():
            p.requires_grad = False
        repertoire[p_type] = QualifiedPrimitiveRecord(
            skill_id=f"prim_{p_type.value}",
            rule_family=p_type.value,
            adapter=adapter,
            qualification_exact_match=0.85,
            qualification_loss=0.30,
            trainable_parameters=adapter.count_parameters(),
            parameter_hash=adapter.compute_parameter_hash(),
            training_steps=40,
            is_frozen=True,
        )
    return repertoire


def test_m76_forensic_audit_and_parameter_lineage() -> None:
    """Audit M76 empirical outcomes and reconcile 928k vs 974k backbone parameters."""
    audit = M76ScientificAuditor.execute_audit()
    assert audit.status == "M76_AUDIT_VERIFIED"
    assert audit.verdict == "PASS_RECONCILED"

    # Verify metrics
    assert audit.reconciled_metrics["NEURAL_COMPOSITION_EM"] == 0.4479
    assert audit.reconciled_metrics["WRONG_ORDER_EM"] == 0.1302
    assert audit.reconciled_metrics["CAUSAL_ORDERING_DELTA"] == 0.3177

    # Parameter reconciliation
    p_rec = audit.parameter_reconciliation
    assert p_rec["core_backbone_parameters"] == 928512
    assert p_rec["expanded_total_parameters"] == 974658
    assert p_rec["task_head_parameters"] == 46146
    assert p_rec["core_backbone_parameters"] + p_rec["task_head_parameters"] == 974658

    # Decomposition vs aggregate reconciliation
    d_rec = audit.decomposition_reconciliation
    assert d_rec["error_decomposition_success_pct"] == 44.44
    assert d_rec["aggregate_exact_match_pct"] == 44.79

    # Lifetime cost accounting
    l_cost = audit.lifetime_cost_accounting
    assert l_cost["total_primitive_acquisition_steps"] == 120
    assert l_cost["composition_additional_training_steps"] == 0
    assert l_cost["composition_additional_parameters"] == 0


def test_plan_space_and_identifiability() -> None:
    """Verify candidate plan enumeration and identifiable task generation."""
    plans = PlanSpace.generate_candidate_plans(max_depth=2, include_abstain=True)
    # 1 abstain + 3 single + 6 two-stage = 10 plans
    assert len(plans) == 10
    abstain_plans = [p for p in plans if p.plan_type == PlanType.ABSTAIN]
    assert len(abstain_plans) == 1
    assert abstain_plans[0].depth == 0

    single_plans = [p for p in plans if p.plan_type == PlanType.SINGLE_PRIMITIVE]
    assert len(single_plans) == 3

    two_stage_plans = [p for p in plans if p.plan_type == PlanType.COMPOSED_TWO_STAGE]
    assert len(two_stage_plans) == 6

    # Test task generator with N=4 demonstrations
    gen = PlanIdentifiabilityGenerator(vocab_size=16)
    task = gen.generate_task(
        task_id="test_reflect_shift",
        compound_type=CompoundTaskType.REFLECT_THEN_SHIFT3,
        seed=42,
        n_demo=4,
        n_dev=8,
        n_sealed_test=16,
    )
    assert len(task.demonstrations) == 4
    assert len(task.development_examples) == 8
    assert len(task.sealed_test_examples) == 16
    assert task.ground_truth_plan_stages == (
        PrimitiveSkillType.COMPLEMENT_REFLECT.value,
        PrimitiveSkillType.CAESAR_SHIFT_MOD3.value,
    )

    # Shuffled demonstrations break pairings
    shuffled_demos = PlanIdentifiabilityGenerator.create_shuffled_demonstrations(task.demonstrations, seed=99)
    assert len(shuffled_demos) == 4


def test_plan_discovery_engine_selection_and_abstention() -> None:
    """Verify plan search evaluation, confidence margins, and fallback when N=0."""
    config = ModelConfig(vocab_size=100, d_model=64, n_layers=2, n_heads=2, d_ff=128, context_length=32)
    model = MiniGPT(config)
    device = torch.device("cpu")
    primitives = _build_dummy_primitives()

    gen = PlanIdentifiabilityGenerator(vocab_size=16)

    # 1. N=0 demonstrations -> Must abstain
    task_n0 = gen.generate_task(
        task_id="test_n0",
        compound_type=CompoundTaskType.REFLECT_THEN_SHIFT3,
        seed=42,
        n_demo=0,
    )
    res_n0 = AutonomousPlanDiscoveryEngine.discover_plan(
        model=model,
        task_id=task_n0.task_id,
        demonstrations=task_n0.demonstrations,
        primitives=primitives,
        device=device,
    )
    assert res_n0.is_abstained is True
    assert res_n0.selected_plan.plan_type == PlanType.ABSTAIN
    assert res_n0.fallback_requested is True

    # 2. Unfamiliar task -> Must abstain
    task_unfamiliar = gen.generate_task(
        task_id="test_unfamiliar",
        unfamiliar_rule="BITWISE_XOR_MASK",
        seed=42,
        n_demo=4,
    )
    res_unfamiliar = AutonomousPlanDiscoveryEngine.discover_plan(
        model=model,
        task_id=task_unfamiliar.task_id,
        demonstrations=task_unfamiliar.demonstrations,
        primitives=primitives,
        device=device,
    )
    assert res_unfamiliar.is_abstained is True
    assert res_unfamiliar.fallback_requested is True


def test_causal_plan_benchmark_and_safeguards() -> None:
    """Verify causal plan benchmark execution, safeguards, carry, and rollback."""
    config = ModelConfig(vocab_size=100, d_model=64, n_layers=2, n_heads=2, d_ff=128, context_length=32)
    model = MiniGPT(config)
    device = torch.device("cpu")
    primitives = _build_dummy_primitives()

    init_hash = M77SafeguardsAuditor.compute_model_hash(model)

    gen = PlanIdentifiabilityGenerator(vocab_size=16)
    task = gen.generate_task(
        task_id="test_benchmark_task",
        compound_type=CompoundTaskType.REFLECT_THEN_SHIFT3,
        seed=42,
        n_demo=2,
        n_dev=4,
        n_sealed_test=8,
    )

    arms = CausalPlanBenchmark.evaluate_all_arms(
        model=model,
        task=task,
        primitives=primitives,
        device=device,
        seed=42,
    )

    assert "A_AUTONOMOUS_PLAN_SEARCH" in arms
    assert "B_PROVIDED_CORRECT_PLAN" in arms
    assert "C_FIXED_ORDER" in arms
    assert "D_RANDOM_VALID_PLAN" in arms
    assert "E_SHUFFLED_DEMONSTRATIONS" in arms
    assert "F_SINGLE_SKILL" in arms
    assert "G_WRONG_ORDER" in arms
    assert "H_FRESH_COMPOUND_ADAPTER" in arms
    assert "I_REFERENCE_ORACLE" in arms

    # Check oracle is perfect
    assert arms["I_REFERENCE_ORACLE"].sealed_exact_match == 1.0

    # Run safeguards audit
    report = M77SafeguardsAuditor.execute_audit(
        model=model,
        initial_hash=init_hash,
        primitives=primitives,
        device=device,
    )
    assert report.base_invariance_verified is True
    assert report.primitives_frozen_verified is True
    assert report.arithmetic_carry_retention_rate == 1.0
    assert report.chained_operational_logic_rate == 1.0
    assert report.bit_exact_rollback_verified is True
    assert report.hardware_accounting["plan_discovery_additional_parameters"] == 0
