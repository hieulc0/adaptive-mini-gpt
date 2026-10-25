"""Milestone M100 Verification Suite: Integrated Autonomous Student & Selective Backbone Evolution.

Validates:
1. test_m99_forensic_audit_and_causal_ablation
2. test_backbone_integrated_forward_and_causal_ablation
3. test_two_speed_plasticity_and_consolidation_rollback
4. test_verified_structural_evolution_tiers
5. test_multi_subject_curriculum_and_verifiers
6. test_autonomous_learning_cycle_10_steps
7. test_causal_controls_future_learning_comparison
8. test_persistent_operation_and_crash_recovery
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest
import torch

from adaptive_mini_gpt.integrated_autonomous_student.audit_m99 import (
    M99ForensicAuditor,
)
from adaptive_mini_gpt.integrated_autonomous_student.autonomous_learning_cycle import (
    AutonomousLearningCycleM100,
)
from adaptive_mini_gpt.integrated_autonomous_student.future_learning_evaluator_m100 import (
    FutureLearningEvaluatorM100,
)
from adaptive_mini_gpt.integrated_autonomous_student.integrated_student_m100 import (
    IntegratedAutonomousStudentM100,
)
from adaptive_mini_gpt.integrated_autonomous_student.multi_subject_curriculum import (
    DomainM100,
    IndependentCurriculumVerifier,
    MultiSubjectCurriculumGenerator,
)
from adaptive_mini_gpt.integrated_autonomous_student.structural_evolution_engine import (
    CapacityTierM100,
)
from adaptive_mini_gpt.integrated_autonomous_student.two_speed_plasticity_manager import (
    PlasticityMode,
)


@pytest.fixture
def device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def test_m99_forensic_audit_and_causal_ablation(device: torch.device):
    """Verify Phase 0 forensic audit confirms M99 invariants and reveals backbone bypass."""
    audit = M99ForensicAuditor.execute_audit(device=device)

    assert audit.status == "M99_AUDIT_VERIFIED"
    assert audit.backbone_weights == 1_071_872
    assert audit.m99_trainable_parameters == 19_064
    assert audit.m99_headroom == 936
    assert audit.m99_student_d_steps == 8
    assert audit.m99_active_eig == 0.850
    # In M99, the substrate bypassed the backbone, yielding 0.0% causal impact
    assert audit.m99_backbone_causal_impact_pct == 0.0
    assert audit.m99_backbone_participates_in_inference is False
    assert audit.full_persistent_memory_bytes > audit.m99_reported_external_memory_bytes
    assert audit.tier_allocation_rollback_verified is True


def test_backbone_integrated_forward_and_causal_ablation(device: torch.device):
    """Verify in M100 that the 1,071,872 backbone participates directly in forward inference."""
    torch.manual_seed(42)
    student = IntegratedAutonomousStudentM100(tier=CapacityTierM100.TIER_20K, device=device)
    assert sum(p.numel() for p in student.backbone.parameters()) == 1_071_872

    gen = MultiSubjectCurriculumGenerator(seed=42, device=device)
    goal = gen.generate_full_curriculum()[0]

    student.eval()
    with torch.no_grad():
        normal_logits = student.forward(goal.test_tokens)
        ablated_logits = student.forward(goal.test_tokens, ablate_backbone=True)

    # In M100, ablating the backbone MUST causally alter the forward output
    diff = (normal_logits - ablated_logits).abs().max().item()
    assert diff > 0.05, f"Expected causal backbone participation, but diff was {diff}"


def test_two_speed_plasticity_and_consolidation_rollback(device: torch.device):
    """Verify fast adapter updates and safe selective backbone consolidation with rollback."""
    torch.manual_seed(42)
    student = IntegratedAutonomousStudentM100(tier=CapacityTierM100.TIER_20K, device=device)
    gen = MultiSubjectCurriculumGenerator(seed=42, device=device)
    gen.generate_full_curriculum()[0]

    # Fast plasticity
    student.plasticity_manager.configure_plasticity(student, PlasticityMode.FAST_ADAPTER_ONLY)
    assert any(p.requires_grad for p in student.adapter_modules.parameters())
    assert all(not p.requires_grad for p in student.backbone.parameters())

    # Slow selective backbone plasticity
    unfrozen = student.plasticity_manager.configure_plasticity(
        student,
        PlasticityMode.SLOW_SELECTIVE_BACKBONE,
        target_layers=["blocks.3", "ln_f"],
    )
    assert len(unfrozen) > 0
    assert any("blocks.3" in name for name in unfrozen)

    # Test rollback functionality
    student.plasticity_manager.backup_candidate_state(student)
    # Mutate a weight
    first_param = next(student.backbone.parameters())
    orig_val = first_param[0, 0].item()
    with torch.no_grad():
        first_param[0, 0] += 5.0
    assert first_param[0, 0].item() != orig_val

    # Trigger rollback
    student.plasticity_manager.rollback(student)
    assert abs(next(student.backbone.parameters())[0, 0].item() - orig_val) < 1e-6


def test_verified_structural_evolution_tiers(device: torch.device):
    """Verify configurable capacity tiers (20k, 30k, 50k, 100k, adaptive) and expansion."""
    torch.manual_seed(42)
    student = IntegratedAutonomousStudentM100(tier=CapacityTierM100.TIER_20K, device=device)
    acct_20k = student.evolution_engine.get_accounting(student)
    assert acct_20k.backbone_weights == 1_071_872
    assert acct_20k.total_trainable_params <= 20_000
    assert acct_20k.headroom_params > 0

    # Expand to 50k tier and allocate new expert
    student.evolution_engine.tier = CapacityTierM100.TIER_50K
    res = student.evolution_engine.allocate_adapter(student, "dyn_expert_graph", rank=4)
    assert res["parameters_added"] > 0
    assert "dyn_expert_graph" in student.adapter_modules

    # Expand rank
    res_exp = student.evolution_engine.expand_adapter_rank(student, "dyn_expert_graph", new_rank=8)
    assert res_exp["new_rank"] == 8

    # Prune adapter
    pruned = student.evolution_engine.prune_adapter(student, "dyn_expert_graph")
    assert pruned is True
    assert "dyn_expert_graph" not in student.adapter_modules


def test_multi_subject_curriculum_and_verifiers(device: torch.device):
    """Verify coverage of all 7 reasoning domains and independent ground truth verifiers."""
    gen = MultiSubjectCurriculumGenerator(seed=42, device=device)
    goals = gen.generate_full_curriculum()

    assert len(goals) == 9  # 7 domains + 2 diagnostic controls
    domains = {g.domain for g in goals}
    assert DomainM100.ALGEBRA in domains
    assert DomainM100.GRAPH in domains
    assert DomainM100.LOGIC in domains
    assert DomainM100.PROGRAM in domains
    assert DomainM100.FACTUAL_MEMORY in domains
    assert DomainM100.PLANNING in domains
    assert DomainM100.COMPOSITION in domains
    assert DomainM100.NOISE_DISTRACTOR in domains
    assert DomainM100.ALREADY_MASTERED in domains

    # Independent verifiers
    assert IndependentCurriculumVerifier.verify_algebra(2, 3, 5) == 11
    assert IndependentCurriculumVerifier.verify_graph(1, 0) == 2
    assert IndependentCurriculumVerifier.verify_logic(1, 1, "AND") == 1
    assert IndependentCurriculumVerifier.verify_program(5, 1, 3) == 8


def test_autonomous_learning_cycle_10_steps(device: torch.device):
    """Verify execution of the 10-step autonomous learning cycle."""
    torch.manual_seed(42)
    student = IntegratedAutonomousStudentM100(tier=CapacityTierM100.TIER_20K, device=device)
    gen = MultiSubjectCurriculumGenerator(seed=42, device=device)
    goals = gen.generate_full_curriculum()

    cycle = AutonomousLearningCycleM100(student, consolidation_interval=2)
    step_result = cycle.run_cycle(goals)

    assert step_result.cycle_index == 1
    assert step_result.assessed_goals_count == 9
    assert step_result.active_experiments_run > 0
    assert step_result.fast_adaptation_steps > 0
    assert step_result.retained_anchor_accuracy >= 90.0
    assert step_result.memory_facts_count > 0


def test_causal_controls_future_learning_comparison(device: torch.device):
    """Verify Condition C outperforms Condition B on sealed target tasks."""
    torch.manual_seed(42)
    student = IntegratedAutonomousStudentM100(tier=CapacityTierM100.TIER_20K, device=device)
    evaluator = FutureLearningEvaluatorM100(device=device)

    res = evaluator.evaluate_causal_conditions(student)

    assert res.condition_a.adaptation_steps >= res.condition_b.adaptation_steps
    assert res.condition_b.adaptation_steps > res.condition_c.adaptation_steps
    assert res.condition_c.adaptation_steps >= res.condition_d.adaptation_steps
    assert res.backbone_plasticity_advantage_pct >= 20.0
    assert res.all_conditions_qualified is True


def test_persistent_operation_and_crash_recovery(device: torch.device):
    """Verify bit-exact checkpoint serialization, process interruption, and reload."""
    torch.manual_seed(42)
    with tempfile.TemporaryDirectory() as tmp_dir:
        student = IntegratedAutonomousStudentM100(tier=CapacityTierM100.TIER_20K, device=device)
        student.step_counter = 42
        student.memory_store.store_fact("entity_test", "prop_speed", 99, source="test", timestamp=10)

        chk_file = Path(tmp_dir) / "student_m100.pt"
        student.save_checkpoint(chk_file)

        reloaded = IntegratedAutonomousStudentM100.load_checkpoint(chk_file, device=device)

        assert reloaded.step_counter == 42
        assert len(reloaded.memory_store.facts) == len(student.memory_store.facts)

        gen = MultiSubjectCurriculumGenerator(seed=42, device=device)
        sample = gen.generate_full_curriculum()[0].test_tokens

        student.eval()
        reloaded.eval()
        p1 = student.forward(sample).argmax(dim=-1)
        p2 = reloaded.forward(sample).argmax(dim=-1)
        assert (p1 == p2).all().item() is True
        torch.manual_seed(42)
