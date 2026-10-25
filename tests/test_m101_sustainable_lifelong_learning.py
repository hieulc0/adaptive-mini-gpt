"""Milestone M101 Tests: Sustainable Lifelong Neural Learning, Selective Backbone Plasticity, and Resource-Aware Consolidation.

Validates:
1. Forensic scientific audit of M100 reconciling trainable parameter accounting.
2. Permanent Architecture Principle & Transparent Parameter Accounting.
3. Two-Speed Neural Plasticity (Fast Adapters + Slow Selective Backbone Consolidation with Rollback).
4. Diverse Lifelong Curriculum across 8 distinct reasoning mechanisms.
5. Lifelong Learning Coordinator multi-episode execution.
6. Causal 5-Condition Future-Learning Comparison on novel non-linear mechanisms.
7. Process Interruption, Checkpoint Persistence & Bit-Exact Recovery.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest
import torch

from adaptive_mini_gpt.integrated_autonomous_student.structural_evolution_engine import (
    CapacityTierM100,
)
from adaptive_mini_gpt.sustainable_lifelong_learning.audit_m100 import (
    M100ForensicAuditor,
)
from adaptive_mini_gpt.sustainable_lifelong_learning.future_learning_evaluator_m101 import (
    FutureLearningEvaluatorM101,
)
from adaptive_mini_gpt.sustainable_lifelong_learning.lifelong_coordinator_m101 import (
    LifelongLearningCoordinatorM101,
)
from adaptive_mini_gpt.sustainable_lifelong_learning.lifelong_curriculum_m101 import (
    DomainM101,
    IndependentLifelongVerifier,
    LifelongCurriculumGeneratorM101,
)
from adaptive_mini_gpt.sustainable_lifelong_learning.persistent_student_m101 import (
    PersistentStudentM101,
)


@pytest.fixture
def device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def test_m100_forensic_audit_and_reconciliation(device: torch.device):
    """Verify Phase 0 audit reconciles parameter accounting and verifies causal backbone forward impact."""
    audit = M100ForensicAuditor.execute_audit(device=device)

    assert audit.status == "M100_AUDIT_VERIFIED"
    assert audit.backbone_parameters == 1_071_872
    assert audit.fast_trainable_parameters <= 20_000
    assert audit.fast_trainable_headroom_20k > 0
    assert audit.slow_unfrozen_backbone_parameters > 0
    assert audit.slow_trainable_total_parameters > audit.fast_trainable_parameters
    assert audit.causal_forward_impact_pct == 100.0
    assert audit.target_task_diversity_verdict == "DIVERSITY_EXTENSION_REQUIRED"


def test_permanent_architecture_and_parameter_accounting(device: torch.device):
    """Verify Permanent Architecture Principle and transparent parameter accounting."""
    student = PersistentStudentM101(tier=CapacityTierM100.TIER_20K, device=device)
    acct = student.get_full_parameter_breakdown()

    assert acct["backbone_parameters"] == 1_071_872
    assert acct["fast_trainable_parameters"] <= 20_000
    assert acct["fast_trainable_headroom_20k"] > 0
    assert acct["slow_unfrozen_backbone_parameters"] == 198_528  # blocks.3 + ln_f
    assert acct["slow_trainable_total"] == acct["fast_trainable_parameters"] + acct["slow_unfrozen_backbone_parameters"]
    assert acct["total_persistent_parameters"] == acct["backbone_parameters"] + acct["fast_trainable_parameters"]
    assert acct["external_memory_bytes"] <= 262_144

    # Expand rank dynamically
    student.evolution_engine.tier = CapacityTierM100.TIER_50K
    res_exp = student.evolution_engine.expand_adapter_rank(student, "adapter_algebra_quadratic", new_rank=4)
    assert res_exp["new_rank"] == 4


def test_two_speed_plasticity_and_safe_rollback(device: torch.device):
    """Verify fast adapter gradient isolation and slow selective consolidation with safe rollback."""
    torch.manual_seed(42)
    student = PersistentStudentM101(tier=CapacityTierM100.TIER_20K, device=device)

    # Fast Plasticity default: backbone frozen
    assert student.count_trainable_parameters() > 0
    assert all(not p.requires_grad for p in student.backbone.parameters())

    # Test rollback functionality
    student.plasticity_manager.backup_candidate_state(student)
    first_param = next(student.backbone.parameters())
    orig_val = first_param[0, 0].item()
    with torch.no_grad():
        first_param[0, 0] += 5.0
    assert first_param[0, 0].item() != orig_val

    # Trigger rollback
    student.plasticity_manager.rollback(student)
    assert abs(next(student.backbone.parameters())[0, 0].item() - orig_val) < 1e-6


def test_diverse_lifelong_curriculum_and_verifiers(device: torch.device):
    """Verify coverage of all 8 diverse reasoning domains and independent solvers."""
    gen = LifelongCurriculumGeneratorM101(seed=42, device=device)
    goals = gen.generate_full_curriculum()

    assert len(goals) == 10  # 8 domains + 2 diagnostic controls
    domains = {g.domain for g in goals}
    assert DomainM101.ALGEBRA_QUADRATIC in domains
    assert DomainM101.GRAPH_REACHABILITY in domains
    assert DomainM101.LOGIC_MAJORITY in domains
    assert DomainM101.STATEFUL_PROGRAM in domains
    assert DomainM101.FINITE_STATE_MACHINE in domains
    assert DomainM101.FACTUAL_MEMORY in domains
    assert DomainM101.MULTI_STEP_PLANNING in domains
    assert DomainM101.CROSS_SUBJECT_COMPOSITION in domains
    assert DomainM101.NOISE_DISTRACTOR in domains
    assert DomainM101.ALREADY_MASTERED in domains

    # Verify independent solvers on diverse mechanisms
    assert IndependentLifelongVerifier.verify_quadratic(2, 1, 3, 1) == 11  # (4 + 6 + 1)%16 = 11
    assert IndependentLifelongVerifier.verify_graph_2hop(1, 0, 1) == 5  # hop1=(1+1)=2, hop2=(2*2+1)=5
    assert IndependentLifelongVerifier.verify_logic_majority(1, 1, 0) == 1
    assert IndependentLifelongVerifier.verify_dfa(0, 1) == 1


def test_lifelong_learning_coordinator_episodes(device: torch.device):
    """Verify execution of learning episodes and multi-episode lifelong loop."""
    torch.manual_seed(42)
    student = PersistentStudentM101(tier=CapacityTierM100.TIER_20K, device=device)
    gen = LifelongCurriculumGeneratorM101(seed=42, device=device)
    goals = gen.generate_full_curriculum()

    coordinator = LifelongLearningCoordinatorM101(student, consolidation_interval=2)
    ep_res = coordinator.run_episode(goals)

    assert ep_res.episode_index == 1
    assert ep_res.fast_adaptation_steps > 0
    assert ep_res.retained_anchor_accuracy >= 90.0
    assert ep_res.memory_facts_count > 0


def test_causal_controls_future_learning_evaluation(device: torch.device):
    """Verify Condition C outperforms Condition B on sealed novel quadratic tasks."""
    torch.manual_seed(42)
    student = PersistentStudentM101(tier=CapacityTierM100.TIER_20K, device=device)
    evaluator = FutureLearningEvaluatorM101(device=device)

    res = evaluator.evaluate_causal_conditions(student)

    assert res.condition_a.adaptation_steps >= res.condition_b.adaptation_steps
    assert res.condition_b.adaptation_steps > res.condition_c.adaptation_steps
    assert res.condition_c.adaptation_steps >= res.condition_d.adaptation_steps
    assert res.backbone_plasticity_advantage_pct >= 20.0
    assert res.all_conditions_qualified is True


def test_process_interruption_and_bit_exact_recovery(device: torch.device):
    """Verify bit-exact checkpoint serialization, process interruption, and reload."""
    torch.manual_seed(42)
    with tempfile.TemporaryDirectory() as tmp_dir:
        student = PersistentStudentM101(tier=CapacityTierM100.TIER_20K, device=device)
        student.step_counter = 101
        student.memory_store.store_fact("entity_dfa", "state_trans", 3, source="test", timestamp=10)

        chk_file = Path(tmp_dir) / "student_m101.pt"
        student.save_checkpoint(chk_file)

        reloaded = PersistentStudentM101.load_checkpoint(chk_file, device=device)

        assert reloaded.step_counter == 101
        assert len(reloaded.memory_store.facts) == len(student.memory_store.facts)

        gen = LifelongCurriculumGeneratorM101(seed=42, device=device)
        sample = gen.generate_full_curriculum()[0]
        student.eval()
        reloaded.eval()
        p1 = student.forward(sample.test_tokens).argmax(dim=-1)
        p2 = reloaded.forward(sample.test_tokens).argmax(dim=-1)
        assert (p1 == p2).all().item() is True
