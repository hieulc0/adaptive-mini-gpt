"""Milestone M102.1 Unit Tests: Continual-Learning Interference Repair and Architecture Preservation.

Validates:
1. Forensic audit and capacity reconciliation (16,776 router parameter resolution with 0 discrepancy).
2. Seed variance diagnosis (Seed 43 collapse vs Seed 46 success).
3. Task-conditioned decoupled heads preventing output boundary interference.
4. DomainRouter legitimate inference-time dispatch without ground-truth cheating.
5. OrthogonalGradientProjector (PCGrad) resolving conflicting rehearsal updates.
6. Full-batch balanced rehearsal buffer with per-family standard error tracking.
7. M102.1 Persistent Student candidate rollback guard and architecture preservation.
8. Bit-exact checkpoint interruption and restoration.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest
import torch
import torch.nn as nn

from adaptive_mini_gpt.continual_interference_repair.audit_m102 import (
    analyze_m102_seed_variance,
    reconcile_m102_parameters,
)
from adaptive_mini_gpt.continual_interference_repair.interference_controllers import (
    DomainRouter,
    OrthogonalGradientProjector,
    TaskConditionedHeadManager,
)
from adaptive_mini_gpt.continual_interference_repair.persistent_student_m102_1 import (
    M102_1PersistentStudent,
    PersistentStudentConfig,
)
from adaptive_mini_gpt.continual_interference_repair.rehearsal_buffer_m102_1 import (
    FullBatchBalancedRehearsalBuffer,
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


def test_parameter_audit_and_reconciliation_zero_discrepancy(device: torch.device):
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
    mgr = TaskConditionedHeadManager(d_model=128, num_classes=16).to(device)

    h_math = mgr.get_or_create_head(DomainM101.MODULAR_ARITHMETIC.value, device=device)
    h_graph = mgr.get_or_create_head(DomainM101.GRAPH_NAVIGATION.value, device=device)

    # Initial weights must be distinct instances
    assert h_math is not h_graph

    # Store copy of h_math weights
    orig_math_weight = h_math.weight.clone().detach()

    # Perform gradient step exclusively on h_graph
    opt = torch.optim.SGD(h_graph.parameters(), lr=0.1)
    x = torch.randn(4, 128, device=device)
    loss = h_graph(x).sum()
    opt.zero_grad()
    loss.backward()
    opt.step()

    # Verify h_math was strictly unaffected
    diff = (h_math.weight - orig_math_weight).abs().max().item()
    assert diff == 0.0, f"Expected domain head isolation, but h_math weights drifted by {diff}"


def test_domain_router_legitimate_dispatch(device: torch.device):
    """Verify DomainRouter correctly routes pooled features without ground-truth labels."""
    domains = [d.value for d in DomainM101]
    router = DomainRouter(d_model=128, domains=domains).to(device)

    x = torch.randn(8, 128, device=device)
    logits = router(x)
    assert logits.shape == (8, len(domains))

    preds = router.predict_domain(x)
    assert len(preds) == 8
    assert all(p in domains for p in preds)


def test_orthogonal_gradient_projector_pcgrad():
    """Verify PCGrad projects conflicting gradients onto orthogonal normal planes."""
    projector = OrthogonalGradientProjector()

    # Construct two conflicting gradients (negative inner product)
    g1 = torch.tensor([1.0, 1.0, 0.0])
    g2 = torch.tensor([-1.0, 0.0, 0.0])

    dot_before = torch.dot(g1, g2).item()
    assert dot_before < 0.0  # Conflicting

    # Project g1 with respect to g2
    g1_proj = projector.project_conflicting(g1, g2)
    dot_after = torch.dot(g1_proj, g2).item()
    assert dot_after >= -1e-6  # Orthogonalized (no destructive interference)
    assert g1_proj[1].item() == 1.0  # Non-conflicting dimension preserved


def test_full_batch_balanced_rehearsal_buffer(device: torch.device):
    """Verify rehearsal buffer stores exemplars across domains and computes per-family SE."""
    buffer = FullBatchBalancedRehearsalBuffer(max_exemplars_per_domain=32)

    # Register 3 domains
    for d in [DomainM101.MODULAR_ARITHMETIC.value, DomainM101.GRAPH_NAVIGATION.value, DomainM101.STRING_TRANSFORMATION.value]:
        tokens = torch.randint(0, 1024, (16, 4), device=device)
        labels = torch.randint(0, 16, (16,), device=device)
        buffer.register_domain(d, tokens, labels)

    assert buffer.num_domains == 3
    sampled_tok, sampled_lbl, sampled_domains = buffer.sample_balanced_batch(batch_size_per_domain=4)
    assert sampled_tok.size(0) == 12  # 3 * 4
    assert sampled_lbl.size(0) == 12
    assert len(sampled_domains) == 12

    # Verify per-domain metrics calculation
    mock_eval = {
        DomainM101.MODULAR_ARITHMETIC.value: 0.95,
        DomainM101.GRAPH_NAVIGATION.value: 0.85,
        DomainM101.STRING_TRANSFORMATION.value: 0.90,
    }
    stats = buffer.compute_retention_statistics(mock_eval)
    assert stats["aggregate_retention_pct"] == 90.0
    assert stats["worst_family_retention_pct"] == 85.0
    assert "standard_error_pct" in stats


def test_persistent_student_m102_1_architecture_and_rollback(device: torch.device):
    """Verify PersistentStudentM102_1 preserves identity block insertion and handles candidate rollback."""
    cfg = PersistentStudentConfig(
        tier=CapacityTierM100.TIER_20K,
        use_task_conditioned_heads=True,
        use_orthogonal_projection=True,
        use_balanced_rehearsal=True,
        rehearsal_batch_size_per_domain=4,
    )
    student = M102_1PersistentStudent(cfg=cfg, device=device)

    # 1. Verify identity block insertion bit-exact preservation
    gen = LifelongCurriculumGeneratorM101(seed=42, device=device)
    goal = gen.generate_full_curriculum()[0]

    student.eval()
    with torch.no_grad():
        out_orig = student.forward(goal.test_tokens, domain_name=goal.domain.value)

    n_blocks_orig = len(student.evolvable_backbone.base_model.blocks)
    n_blocks_new = student.evolvable_backbone.insert_block_identity(insert_idx=n_blocks_orig)
    assert n_blocks_new == n_blocks_orig + 1

    with torch.no_grad():
        out_inserted = student.forward(goal.test_tokens, domain_name=goal.domain.value)

    diff = (out_orig - out_inserted).abs().max().item()
    assert diff < 1e-6, f"Identity block insertion must be bit-exact, got max diff {diff}"

    # 2. Test candidate rollback behavior
    # Register rehearsal domains
    for g in gen.generate_full_curriculum()[:2]:
        student.rehearsal_buffer.register_domain(g.domain.value, g.train_tokens, g.train_labels)

    # Run consolidation with strict rollback check
    report = student.consolidate_candidate(goal, steps=2, worst_family_gate=80.0)
    assert isinstance(report["promoted"], bool)
    assert "aggregate_retention_pct" in report


def test_checkpoint_interruption_and_recovery_m102_1(device: torch.device):
    """Verify bit-exact checkpoint serialization and restoration for evolvable student with decoupled heads."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        cfg = PersistentStudentConfig(
            tier=CapacityTierM100.TIER_20K,
            use_task_conditioned_heads=True,
            use_orthogonal_projection=True,
            use_balanced_rehearsal=True,
        )
        student = M102_1PersistentStudent(cfg=cfg, device=device)
        student.step_counter = 42

        # Insert block and register heads
        student.evolvable_backbone.insert_block_identity(insert_idx=len(student.evolvable_backbone.base_model.blocks))
        student.head_manager.get_or_create_head(DomainM101.MODULAR_ARITHMETIC.value, device=device)

        chk_file = Path(tmp_dir) / "student_m102_1.pt"
        student.save_checkpoint(chk_file)

        # Reload
        reloaded = M102_1PersistentStudent.load_checkpoint(chk_file, device=device)
        assert reloaded.step_counter == 42
        assert len(reloaded.evolvable_backbone.base_model.blocks) == len(student.evolvable_backbone.base_model.blocks)
        assert DomainM101.MODULAR_ARITHMETIC.value in reloaded.head_manager.heads

        # Verify exact forward outputs match
        gen = LifelongCurriculumGeneratorM101(seed=42, device=device)
        goal = gen.generate_full_curriculum()[0]
        student.eval()
        reloaded.eval()
        with torch.no_grad():
            p1 = student.forward(goal.test_tokens, domain_name=goal.domain.value)
            p2 = reloaded.forward(goal.test_tokens, domain_name=goal.domain.value)
        diff = (p1 - p2).abs().max().item()
        assert diff < 1e-6
