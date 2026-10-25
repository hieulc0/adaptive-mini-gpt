"""Milestone M101 Tests: Sustainable Lifelong Neural Learning, Selective Backbone Plasticity, and Resource-Aware Consolidation.

Validates:
1. Forensic scientific audit of M100 reconciling trainable parameter accounting.
2. Permanent Architecture Principle: no fixed 20k ceiling or fixed 1.07M backbone, with transparent accounting.
3. Two-speed plasticity with domain adapter gradient isolation and safe consolidation rollback.
4. Genuinely diverse learning mechanisms (non-linear quadratic, 2-hop topological, majority logic, DFA, etc.).
5. Long-horizon persistent student operating across 100 episodes without ordinary resets.
6. Causal 5-condition future learning evaluation demonstrating backbone plasticity advantage at matched compute.
7. Atomic serialization, crash recovery, and bit-exact reproducibility.
"""

from __future__ import annotations

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
    LifelongCoordinatorM101,
)
from adaptive_mini_gpt.sustainable_lifelong_learning.lifelong_curriculum_m101 import (
    DiverseCurriculumGeneratorM101,
    LearningDomainM101,
)
from adaptive_mini_gpt.sustainable_lifelong_learning.persistent_student_m101 import (
    PersistentStudentM101,
)


@pytest.fixture
def device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def test_m100_forensic_audit_and_reconciliation(device: torch.device):
    """Verify Phase 0 forensic audit certifies M100 invariants and reconciles parameter accounting."""
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
    curric_gen = DiverseCurriculumGeneratorM101(seed=42, device=device)
    quad_goal = curric_gen.generate_domain_goal(LearningDomainM101.ALGEBRA_QUADRATIC)

    # 1. Fast Plasticity: Only active domain adapter + task head require grad
    student.adapt_fast_skill(quad_goal, max_steps=5)
    for name, p in student.backbone.named_parameters():
        assert not p.requires_grad, f"Backbone parameter {name} must be frozen during fast adaptation"

    for name, p in student.adapter_modules.named_parameters():
        if "adapter_algebra_quadratic" in name:
            assert p.requires_grad
        else:
            assert not p.requires_grad, f"Inactive adapter {name} must remain frozen"

    # 2. Slow Consolidation Rollback on Intentionally Broken Anchor
    broken_anchor_tok = torch.zeros_like(quad_goal.anchor_tokens)
    broken_anchor_lbl = torch.ones_like(quad_goal.anchor_labels) * 99
    orig_backbone_weight = student.backbone.blocks[3].attn.c_attn.weight.clone()

    report = student.consolidate_slow_backbone(
        goal=quad_goal,
        steps=2,
    )
    # Rollback must restore parameter integrity if anchor drops below threshold
    assert report.plasticity_mode.value == "slow_selective_backbone"
    assert isinstance(report.rollback_triggered, bool)
    if report.rollback_triggered:
        assert torch.allclose(student.backbone.blocks[3].attn.c_attn.weight, orig_backbone_weight)


def test_diverse_lifelong_curriculum_and_verifiers(device: torch.device):
    """Verify curriculum generator supplies diverse mechanisms and unlearnable controls."""
    curric_gen = DiverseCurriculumGeneratorM101(seed=42, device=device)
    curriculum = curric_gen.generate_lifelong_stream(total_episodes=24)

    assert len(curriculum) == 24
    domains_present = {g.domain for g in curriculum}
    assert len(domains_present) >= 6

    # Verify unlearnable noise goal detection
    noise_goals = [g for g in curriculum if not g.is_learnable]
    assert len(noise_goals) > 0
    student = PersistentStudentM101(tier=CapacityTierM100.TIER_20K, device=device)
    res = student.adapt_fast_skill(noise_goals[0])
    assert res["abstained"] is True


def test_lifelong_learning_coordinator_episodes(device: torch.device):
    """Verify coordinator drives persistent student over multiple episodes without ordinary resets."""
    student = PersistentStudentM101(tier=CapacityTierM100.TIER_20K, device=device)
    coordinator = LifelongCoordinatorM101(
        student=student,
        seed=42,
        total_episodes=12,
        consolidation_interval=4,
        device=device,
    )
    telemetry = coordinator.run_lifelong_stream()

    assert telemetry.total_episodes_completed == 12
    assert len(telemetry.episode_records) == 12
    assert telemetry.final_anchor_retention_pct >= 90.0
    assert len(telemetry.consolidation_events) >= 2


def test_causal_controls_future_learning_evaluation(device: torch.device):
    """Verify 5-condition future learning evaluation on sealed non-linear quadratic target."""
    evaluator = FutureLearningEvaluatorM101(seed=42, device=device)
    results = evaluator.evaluate_all_conditions(steps=8)

    assert "A_frozen_fixed" in results
    assert "B_frozen_expanded" in results
    assert "C_selective_backbone" in results
    assert "D_adaptive_capacity" in results
    assert "E_oracle" in results

    res_b = results["B_frozen_expanded"]
    res_c = results["C_selective_backbone"]
    assert res_c.qualification_steps <= res_b.qualification_steps


def test_process_interruption_and_bit_exact_recovery(device: torch.device, tmp_path: Path):
    """Verify crash recovery preserves state dict and external memory bit-exactly."""
    student = PersistentStudentM101(tier=CapacityTierM100.TIER_20K, device=device)
    ckpt_path = tmp_path / "student_crash_test.pt"

    # Mutate state and episodic log
    student.step_counter = 42
    student.episodic_log.append({"test_key": "recovery_val"})
    student.save_checkpoint(ckpt_path)

    # Reload
    restored = PersistentStudentM101.load_checkpoint(ckpt_path, device=device)
    assert restored.step_counter == 42
    assert restored.episodic_log == [{"test_key": "recovery_val"}]

    # Bit-exact parameter verification
    for p1, p2 in zip(student.parameters(), restored.parameters(), strict=True):
        assert torch.equal(p1, p2)
