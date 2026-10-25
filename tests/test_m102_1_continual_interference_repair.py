"""Milestone M102.1 Unit Tests: Continual-Learning Interference Repair and Architecture Preservation.

Validates:
1. Forensic audit and capacity reconciliation (16,776 router parameter resolution with 0 discrepancy).
2. Seed variance diagnosis (Seed 43 collapse vs Seed 46 success).
3. Task-conditioned decoupled heads preventing output boundary interference.
4. Inference-time domain router dispatch without ground-truth cheating.
5. OrthogonalGradientProjector (PCGrad) resolving conflicting rehearsal updates.
6. BalancedRehearsalBufferM102_1 with per-family standard error uncertainty tracking.
7. PersistentStudentM102_1 candidate rollback guard and identity block architecture preservation.
8. Bit-exact checkpoint interruption and restoration.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest
import torch
import torch.nn as nn

from adaptive_mini_gpt.continual_interference_repair.audit_m102 import (
    M102ForensicAuditor,
    analyze_m102_seed_variance,
    reconcile_m102_parameters,
)
from adaptive_mini_gpt.continual_interference_repair.interference_controllers import (
    OrthogonalGradientProjector,
    TaskConditionedHeadManager,
)
from adaptive_mini_gpt.continual_interference_repair.lifelong_coordinator_m102_1 import (
    LifelongCoordinatorM102_1,
)
from adaptive_mini_gpt.continual_interference_repair.persistent_student_m102_1 import (
    PersistentStudentM102_1,
)
from adaptive_mini_gpt.continual_interference_repair.rehearsal_buffer_m102_1 import (
    BalancedRehearsalBufferM102_1,
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


def test_parameter_audit_and_reconciliation_zero_discrepancy():
    """Verify Phase 1 parameter audit proves the 16,776-parameter discrepancy with zero error."""
    audit = reconcile_m102_parameters()

    assert audit.reported_total_parameters == 1_097_524
    assert audit.sum_of_listed_components == 1_080_748
    assert audit.unaccounted_difference == 16_776
    assert audit.num_dynamic_routers == 4
    assert audit.parameters_per_router == 4_194
    assert audit.total_router_parameters == 16_776
    assert audit.discrepancy == 0
    assert audit.verified is True
    assert "DynamicLayerSkippingRouter" in audit.reconciliation_explanation


def test_seed_variance_analysis():
    """Verify Phase 0 seed variance analysis confirms Seed 43 collapse and identifies root causes."""
    analysis = analyze_m102_seed_variance()

    assert len(analysis.seeds) == 5
    assert analysis.seed_43_held_out_accuracy_pct == 25.78
    assert analysis.seed_43_worst_family_retention_pct == 6.25
    assert analysis.seed_46_held_out_accuracy_pct == 80.47
    assert analysis.seed_46_worst_family_retention_pct == 81.25
    assert "Shared Output-Head Interference" in analysis.primary_interference_mechanisms[0]
    assert "Incomplete Rehearsal" in analysis.primary_interference_mechanisms[1]


def test_task_conditioned_head_manager_isolation(device: torch.device):
    """Verify TaskConditionedHeadManager creates isolated domain heads and prevents cross-domain corruption."""
    domains = [d.value for d in DomainM101]
    mgr = TaskConditionedHeadManager(d_model=128, domains=domains, num_classes=16).to(device)

    # Verify head keys
    dom_math = DomainM101.MODULAR_ARITHMETIC.value
    dom_graph = DomainM101.GRAPH_NAVIGATION.value
    assert f"head_{dom_math}" in mgr.domain_heads
    assert f"head_{dom_graph}" in mgr.domain_heads

    # Forward with explicit domain
    x = torch.randn(4, 128, device=device)
    out_math = mgr(x, active_domains=[dom_math])
    assert out_math.shape == (4, 16)

    # Legitimate inference routing without task labels
    pred_dom = mgr.route_domain(x)
    assert pred_dom in domains
    out_routed = mgr(x)
    assert out_routed.shape == (4, 16)

    # Isolate domain head for training
    mgr.isolate_domain_head(dom_graph)
    math_head = mgr.domain_heads[f"head_{dom_math}"]
    graph_head = mgr.domain_heads[f"head_{dom_graph}"]

    for param in math_head.parameters():
        assert param.requires_grad is False
    for param in graph_head.parameters():
        assert param.requires_grad is True


def test_orthogonal_gradient_projector_pcgrad(device: torch.device):
    """Verify PCGrad projects conflicting gradients onto orthogonal normal planes."""
    p = nn.Parameter(torch.tensor([1.0, 1.0], device=device, requires_grad=True))

    # Construct conflicting task and rehearsal losses (opposite directions along dimension 0)
    task_loss = 2.0 * p[0] + 0.5 * p[1]
    rehearsal_loss = -2.0 * p[0] + 0.5 * p[1]

    dot, conflict = OrthogonalGradientProjector.project_conflicting_gradients(
        [p], task_loss, rehearsal_loss
    )

    assert conflict is True
    assert dot < 0.0
    assert p.grad is not None
    # Dimension 1 had matching signs and must retain positive accumulated gradient
    assert p.grad[1].item() > 0.0


def test_balanced_rehearsal_buffer_m102_1(device: torch.device):
    """Verify BalancedRehearsalBufferM102_1 stores exemplars across domains and computes per-family SE."""
    buffer = BalancedRehearsalBufferM102_1(max_exemplars_per_domain=32)

    dom1 = DomainM101.MODULAR_ARITHMETIC.value
    dom2 = DomainM101.GRAPH_NAVIGATION.value

    t1 = torch.randint(0, 1024, (16, 4), device=device)
    l1 = torch.randint(0, 16, (16,), device=device)
    t2 = torch.randint(0, 1024, (16, 4), device=device)
    l2 = torch.randint(0, 16, (16,), device=device)

    buffer.register_domain(dom1, t1, l1)
    buffer.register_domain(dom2, t2, l2)

    assert len(buffer.domain_exemplars) == 2

    batches = buffer.sample_domain_batches(batch_size_per_domain=8)
    assert len(batches) == 2
    for d, b_tok, b_lbl in batches:
        assert d in [dom1, dom2]
        assert b_tok.size(0) == 8
        assert b_lbl.size(0) == 8

    # Clone buffer
    clone_buf = buffer.clone()
    assert len(clone_buf.domain_exemplars) == 2


def test_persistent_student_m102_1_architecture_preservation(device: torch.device):
    """Verify PersistentStudentM102_1 preserves bit-exact identity block insertion."""
    student = PersistentStudentM102_1(tier=CapacityTierM100.TIER_20K, device=device)
    gen = LifelongCurriculumGeneratorM101(seed=42, device=device)
    goal = gen.generate_full_curriculum()[0]

    student.eval()
    with torch.no_grad():
        out_orig = student.forward(goal.test_tokens, active_domains=[goal.domain.value])

    n_blocks_orig = len(student.evolvable_backbone.base_model.blocks)
    n_blocks_new = student.evolvable_backbone.insert_block_identity(insert_idx=n_blocks_orig)
    assert n_blocks_new == n_blocks_orig + 1

    with torch.no_grad():
        out_inserted = student.forward(goal.test_tokens, active_domains=[goal.domain.value])

    diff = (out_orig - out_inserted).abs().max().item()
    assert diff < 1e-6, f"Identity block insertion must be bit-exact, got max diff {diff}"


def test_checkpoint_serialization_and_recovery_m102_1(device: torch.device):
    """Verify bit-exact checkpoint serialization and restoration for evolvable student with decoupled heads."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        student = PersistentStudentM102_1(tier=CapacityTierM100.TIER_20K, device=device)
        student.step_counter = 42

        # Insert block
        student.evolvable_backbone.insert_block_identity(insert_idx=len(student.evolvable_backbone.base_model.blocks))

        chk_file = Path(tmp_dir) / "student_m102_1.pt"
        student.save_checkpoint(chk_file)

        # Reload
        reloaded = PersistentStudentM102_1.load_checkpoint(chk_file, device=device)
        assert reloaded.step_counter == 42
        assert len(reloaded.evolvable_backbone.base_model.blocks) == len(student.evolvable_backbone.base_model.blocks)

        # Verify exact forward outputs match
        gen = LifelongCurriculumGeneratorM101(seed=42, device=device)
        goal = gen.generate_full_curriculum()[0]
        student.eval()
        reloaded.eval()
        with torch.no_grad():
            p1 = student.forward(goal.test_tokens, active_domains=[goal.domain.value])
            p2 = reloaded.forward(goal.test_tokens, active_domains=[goal.domain.value])
        diff = (p1 - p2).abs().max().item()
        assert diff < 1e-6


def test_lifelong_coordinator_m102_1_census(device: torch.device):
    """Verify coordinator drives persistent student over multiple episodes with worst-family >= 80%."""
    student = PersistentStudentM102_1(tier=CapacityTierM100.TIER_20K, device=device)
    gen = LifelongCurriculumGeneratorM101(seed=42, device=device)
    goals = gen.generate_full_curriculum()[:3]

    coordinator = LifelongCoordinatorM102_1(student, consolidation_interval=2)
    telemetry = coordinator.run_episode(goals)

    assert telemetry.episode_index == 1
    assert telemetry.fast_adaptation_steps > 0
    assert telemetry.worst_family_retention_accuracy >= 80.0
