"""Unit and integration tests for Milestone M78: Reliable Autonomous Neural Plan Discovery."""

from __future__ import annotations

import torch
import pytest

from adaptive_mini_gpt.model import MiniGPT, ModelConfig
from adaptive_mini_gpt.neural_plan_discovery.plan_space import CandidatePlan, PlanSpace, PlanType
from adaptive_mini_gpt.neural_skill_composition.compound_tasks import (
    CompoundProceduralExample,
    CompoundTaskType,
)
from adaptive_mini_gpt.neural_skill_composition.primitive_skills import (
    PrimitiveSkillType,
    QualifiedPrimitiveRecord,
)
from adaptive_mini_gpt.neural_weight_learning.sequence_adapter import Rank16SequenceAdapter
from adaptive_mini_gpt.reliable_plan_discovery.audit_m77 import M77ScientificAuditor
from adaptive_mini_gpt.reliable_plan_discovery.expanded_task_generator import (
    ExpandedPlanTaskGenerator,
    apply_expanded_unfamiliar_rule,
)
from adaptive_mini_gpt.reliable_plan_discovery.likelihood_plan_engine import (
    LikelihoodPlanDiscoveryEngine,
    PlanExecutionStatus,
)
from adaptive_mini_gpt.reliable_plan_discovery.causal_plan_benchmark_v2 import (
    CausalPlanBenchmarkV2,
)


@pytest.fixture
def tiny_model() -> MiniGPT:
    """Fixture providing deterministic MiniGPT model."""
    torch.manual_seed(42)
    config = ModelConfig(
        vocab_size=256,
        d_model=64,
        n_layers=2,
        n_heads=2,
        d_ff=128,
        context_length=16,
        dropout=0.0,
    )
    return MiniGPT(config)


@pytest.fixture
def mock_primitives(tiny_model: MiniGPT) -> dict[PrimitiveSkillType, QualifiedPrimitiveRecord]:
    """Fixture providing initialized Rank16 adapters for all primitive skills."""
    torch.manual_seed(42)
    device = torch.device("cpu")
    primitives: dict[PrimitiveSkillType, QualifiedPrimitiveRecord] = {}
    for p_type in [
        PrimitiveSkillType.COMPLEMENT_REFLECT,
        PrimitiveSkillType.CAESAR_SHIFT_MOD3,
        PrimitiveSkillType.AFFINE_TRANSFORM,
    ]:
        adapter = Rank16SequenceAdapter(d_model=64, bottleneck_dim=16, num_symbols=16).to(device)
        for p in adapter.parameters():
            p.requires_grad = False
        rec = QualifiedPrimitiveRecord(
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
        primitives[p_type] = rec
    return primitives


def test_m77_forensic_audit_and_parameter_lineage():
    """Verify Phase 1 forensic audit of M77 empirical metrics and root causes."""
    audit = M77ScientificAuditor.execute_audit()
    assert audit.status == "AUDIT_VERIFIED"
    assert audit.reconciled_metrics["AUTONOMOUS_PLAN_ACCURACY"] == 0.25
    assert audit.reconciled_metrics["AUTONOMOUS_EXACT_MATCH"] == 0.1146
    assert audit.reconciled_metrics["COMPOSABLE_ABSTENTION_RATE"] == 0.75
    assert audit.reconciled_metrics["UNFAMILIAR_REJECTION_RATE"] == 1.0
    assert audit.reconciled_metrics["ADDITIONAL_PARAMETERS_ALLOCATED"] == 0
    assert audit.reconciled_metrics["ADDITIONAL_TRAINING_STEPS"] == 0

    assert audit.parameter_reconciliation["core_backbone_parameters"] == 928512
    assert audit.parameter_reconciliation["task_head_parameters"] == 46146
    assert audit.parameter_reconciliation["expanded_total_parameters"] == 974658

    assert audit.retention_verification["arithmetic_carry_retention_rate"] == 1.0
    assert audit.retention_verification["operational_logic_retention_rate"] == 1.0
    assert len(audit.prescriptions_for_m78) >= 5


def test_expanded_task_generator_and_partition_isolation():
    """Verify Phase 3 task generator creates isolated partitions across all 4 unfamiliar rules."""
    gen = ExpandedPlanTaskGenerator(vocab_size=16)

    # Test all 4 unfamiliar rules
    for rule in ExpandedPlanTaskGenerator.UNFAMILIAR_RULES:
        task = gen.generate_single_task(
            task_id=f"test_{rule}",
            unfamiliar_rule=rule,
            seed=42,
            n_demo=4,
            n_dev=8,
            n_sealed_test=16,
        )
        assert len(task.demonstrations) == 4
        assert len(task.development_examples) == 8
        assert len(task.sealed_test_examples) == 16

        # Verify strict partition isolation
        demo_inputs = set(d.input_tokens for d in task.demonstrations)
        dev_inputs = set(d.input_tokens for d in task.development_examples)
        test_inputs = set(d.input_tokens for d in task.sealed_test_examples)

        assert demo_inputs.isdisjoint(dev_inputs)
        assert demo_inputs.isdisjoint(test_inputs)
        assert dev_inputs.isdisjoint(test_inputs)

    # Test balanced suite generation
    suite = gen.generate_balanced_suite(seed=42, n_demo=4, composable_replications=2, uncomposable_replications=1)
    assert len(suite) == (4 * 2) + (4 * 1)  # 8 composable + 4 uncomposable = 12 tasks


def test_likelihood_plan_engine_scoring_and_decisions(
    tiny_model: MiniGPT, mock_primitives: dict[PrimitiveSkillType, QualifiedPrimitiveRecord]
):
    """Verify Phase 2 likelihood planner calculates continuous scores and handles N=0."""
    device = torch.device("cpu")
    gen = ExpandedPlanTaskGenerator(vocab_size=16)

    # Case 1: N=0 demonstrations -> immediate abstention with ABSTAINED_NO_EVIDENCE
    res_zero = LikelihoodPlanDiscoveryEngine.discover_plan(
        model=tiny_model,
        task_id="task_zero",
        demonstrations=[],
        primitives=mock_primitives,
        device=device,
    )
    assert res_zero.is_abstained is True
    assert res_zero.execution_status == PlanExecutionStatus.ABSTAINED_NO_EVIDENCE
    assert res_zero.demonstrations_used == 0

    # Case 2: Unfamiliar rule -> evaluated with likelihood -> abstains when below threshold
    uncomp_task = gen.generate_single_task(
        task_id="task_uncomp",
        unfamiliar_rule="NONLINEAR_SBOX",
        seed=101,
        n_demo=4,
    )
    res_uncomp = LikelihoodPlanDiscoveryEngine.discover_plan(
        model=tiny_model,
        task_id=uncomp_task.task_id,
        demonstrations=uncomp_task.demonstrations,
        primitives=mock_primitives,
        device=device,
        min_log_likelihood=-0.1,  # Strict threshold triggers unsupported abstention
        min_token_accuracy=0.99,
    )
    assert res_uncomp.is_abstained is True
    assert res_uncomp.execution_status == PlanExecutionStatus.ABSTAINED_UNSUPPORTED


def test_paired_statistical_comparison():
    """Verify Phase 4 paired Student-t test and confidence interval calculation."""
    arm_a = [0.45, 0.44, 0.46, 0.43, 0.45]
    control = [0.12, 0.11, 0.13, 0.10, 0.12]

    comp = CausalPlanBenchmarkV2.compute_paired_comparison(
        arm_a_vals=arm_a,
        control_vals=control,
        control_name="WRONG_ORDER",
    )
    assert comp.mean_difference > 0.30
    assert comp.ci_lower > 0.0
    assert comp.is_significant is True
    assert comp.t_statistic > 10.0
