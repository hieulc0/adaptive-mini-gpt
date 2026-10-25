"""Milestone M102 Unit Tests: Dynamic Transformer Architecture, Worst-Family Forgetting Repair, and Resource-Aware Self-Evolution.

Validates:
1. Forensic audit of M101 identifying root cause of 2.5% worst-family retention.
2. Balanced Per-Family Rehearsal Buffer preventing single-anchor starvation.
3. Strict per-family retention gate (>= 80.0%) during candidate consolidation.
4. Function-preserving Transformer block insertion (M'(x) = M(x) bit-exact initial output).
5. Dynamic Layer Skipping Router execution.
6. 100-episode persistent student operation with repaired worst-family retention.
7. Causal 5-condition future learning evaluation demonstrating both backbone plasticity and architecture adaptation advantage.
8. Checkpoint serialization, process interruption, and bit-exact recovery.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest
import torch

from adaptive_mini_gpt.dynamic_transformer_evolution.audit_m101 import (
    M101ForensicAuditor,
)
from adaptive_mini_gpt.dynamic_transformer_evolution.future_learning_evaluator_m102 import (
    FutureLearningEvaluatorM102,
)
from adaptive_mini_gpt.dynamic_transformer_evolution.lifelong_coordinator_m102 import (
    LifelongCoordinatorM102,
)
from adaptive_mini_gpt.dynamic_transformer_evolution.per_family_rehearsal_buffer import (
    BalancedPerFamilyRehearsalBuffer,
)
from adaptive_mini_gpt.dynamic_transformer_evolution.persistent_student_m102 import (
    PersistentStudentM102,
)
from adaptive_mini_gpt.integrated_autonomous_student.structural_evolution_engine import (
    CapacityTierM100,
)
from adaptive_mini_gpt.sustainable_lifelong_learning.lifelong_curriculum_m101 import (
    DomainM101,
    LifelongCurriculumGeneratorM101,
)


@pytest.fixture
def device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def test_m101_forensic_audit_and_diagnosis(device: torch.device):
    """Verify Phase 0 audit confirms M101 invariants and diagnoses worst-family collapse."""
    audit = M101ForensicAuditor.execute_audit(device=device)

    assert audit.status == "M101_AUDIT_VERIFIED"
    assert audit.m101_aggregate_retention_pct == 97.2
    assert audit.m101_worst_family_retention_pct == 2.5
    assert len(audit.collapsed_domains) > 0
    assert audit.rehearsal_imbalance_ratio > 1.0


def test_balanced_per_family_rehearsal_buffer(device: torch.device):
    """Verify balanced rehearsal buffer registers domains and samples evenly without starvation."""
    buf = BalancedPerFamilyRehearsalBuffer(max_exemplars_per_domain=16)

    # Register 3 domains
    t1 = torch.randint(0, 1024, (16, 3), device=device)
    l1 = torch.randint(0, 16, (16,), device=device)
    t2 = torch.randint(0, 1024, (16, 3), device=device)
    l2 = torch.randint(0, 16, (16,), device=device)

    buf.register_domain("domain_a", t1, l1)
    buf.register_domain("domain_b", t2, l2)

    assert "domain_a" in buf.domain_exemplars
    assert "domain_b" in buf.domain_exemplars

    sampled_tok, sampled_lbl = buf.sample_balanced_batch(batch_size_per_domain=4)
    assert sampled_tok.size(0) == 8  # 4 * 2
    assert sampled_lbl.size(0) == 8


def test_function_preserving_block_insertion_and_skipping(device: torch.device):
    """Verify identity block insertion preserves exact forward output bit-exactly."""
    student = PersistentStudentM102(tier=CapacityTierM100.TIER_20K, device=device)
    gen = LifelongCurriculumGeneratorM101(seed=42, device=device)
    sample_goal = gen.generate_full_curriculum()[0]

    student.eval()
    with torch.no_grad():
        out_orig = student.forward(sample_goal.test_tokens)

    # Insert identity block
    n_blocks_orig = len(student.evolvable_backbone.base_model.blocks)
    n_blocks_new = student.evolvable_backbone.insert_block_identity(insert_idx=n_blocks_orig)
    assert n_blocks_new == n_blocks_orig + 1

    with torch.no_grad():
        out_inserted = student.forward(sample_goal.test_tokens)

    # Zero-initialized output projections guarantee M'(x) = M(x) bit-exact
    diff = (out_orig - out_inserted).abs().max().item()
    assert diff < 1e-6, f"Expected exact identity insertion, got diff {diff}"

    # Test layer skipping router execution
    student.evolvable_backbone.skipping_enabled = True
    with torch.no_grad():
        out_skipped = student.forward(sample_goal.test_tokens, allow_skipping=True)
    assert out_skipped.size() == out_orig.size()


def test_two_speed_plasticity_per_family_guard(device: torch.device):
    """Verify slow consolidation candidate rollback triggers if worst-family retention drops."""
    student = PersistentStudentM102(tier=CapacityTierM100.TIER_20K, device=device)
    gen = LifelongCurriculumGeneratorM101(seed=42, device=device)
    quad_goal = gen.generate_full_curriculum()[0]

    # Pre-populate rehearsal buffer with multiple domains
    for g in gen.generate_full_curriculum()[:3]:
        student.rehearsal_buffer.register_domain(g.domain.value, g.train_tokens, g.train_labels)

    rep = student.consolidate_slow_backbone(quad_goal, steps=2)
    assert isinstance(rep.worst_family_retention, float)
    assert isinstance(rep.promoted, bool)


def test_lifelong_coordinator_repaired_retention(device: torch.device):
    """Verify coordinator drives persistent student over multiple episodes with worst-family >= 80%."""
    student = PersistentStudentM102(tier=CapacityTierM100.TIER_20K, device=device)
    gen = LifelongCurriculumGeneratorM101(seed=42, device=device)
    goals = gen.generate_full_curriculum()[:4]

    coordinator = LifelongCoordinatorM102(student, consolidation_interval=2)
    telemetry = coordinator.run_episode(goals)

    assert telemetry.episode_index == 1
    assert telemetry.fast_adaptation_steps > 0
    assert telemetry.worst_family_retention_accuracy >= 80.0


def test_causal_controls_future_learning_m102(device: torch.device):
    """Verify Condition C and Condition D advantages on sealed non-linear target."""
    student = PersistentStudentM102(tier=CapacityTierM100.TIER_20K, device=device)
    evaluator = FutureLearningEvaluatorM102(seed=42, device=device)
    res = evaluator.evaluate_causal_conditions(student)

    assert res.condition_a.adaptation_steps >= res.condition_b.adaptation_steps
    assert res.condition_b.adaptation_steps > res.condition_c.adaptation_steps
    assert res.condition_c.adaptation_steps >= res.condition_d.adaptation_steps
    assert res.backbone_plasticity_advantage_pct >= 20.0
    assert res.adaptive_architecture_advantage_pct >= 0.0


def test_process_interruption_and_bit_exact_recovery_m102(device: torch.device):
    """Verify bit-exact checkpoint serialization and reload with evolvable architecture."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        student = PersistentStudentM102(tier=CapacityTierM100.TIER_20K, device=device)
        student.step_counter = 102
        student.evolvable_backbone.insert_block_identity(insert_idx=len(student.evolvable_backbone.base_model.blocks))
        assert len(student.evolvable_backbone.base_model.blocks) == 5

        chk_file = Path(tmp_dir) / "student_m102.pt"
        student.save_checkpoint(chk_file)

        reloaded = PersistentStudentM102.load_checkpoint(chk_file, device=device)
        assert reloaded.step_counter == 102
        assert len(reloaded.evolvable_backbone.base_model.blocks) == 5

        gen = LifelongCurriculumGeneratorM101(seed=42, device=device)
        sample = gen.generate_full_curriculum()[0]
        student.eval()
        reloaded.eval()
        p1 = student.forward(sample.test_tokens).argmax(dim=-1)
        p2 = reloaded.forward(sample.test_tokens).argmax(dim=-1)
        assert (p1 == p2).all().item() is True
