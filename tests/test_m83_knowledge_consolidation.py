"""Milestone M83: Fast CPU-Only Unit Tests for Knowledge Consolidation Suite.

Verifies:
1. M82 forensic audit reconciliation (M82 step reduction, Random policy paired uncertainty df=4, target gates).
2. ConsolidationCandidateAnalyzer multi-metric compatibility (weight, activation, agreement, gradient).
3. ConsolidatedSharedAdapter parameter accounting, forward pass, and >= 20% compression guarantee.
4. LearnedConsolidationController architecture, parameter count (< 1,000 params), and action selection.
5. Bit-exact rollback restoration upon trial failure.
"""

from __future__ import annotations

import copy
import torch
import torch.nn as nn

from adaptive_mini_gpt.knowledge_consolidation.audit_m82 import M82ForensicAuditor
from adaptive_mini_gpt.knowledge_consolidation.candidate_analyzer import (
    ConsolidationCandidateAnalyzer,
)
from adaptive_mini_gpt.knowledge_consolidation.consolidation_policy import (
    ConsolidationAction,
    ConsolidationState,
    LearnedConsolidationController,
)
from adaptive_mini_gpt.knowledge_consolidation.consolidated_adapter import (
    ConsolidatedSharedAdapter,
)
from adaptive_mini_gpt.learning_to_learn.capacity_adaptation import (
    BottleneckSequenceAdapter,
)


def test_m82_forensic_audit_and_reconciliation() -> None:
    """Verify that M82 audit reconciles claims, evaluates Random paired uncertainty, and sets gates."""
    audit = M82ForensicAuditor.execute_audit()
    assert audit.status == "M82_AUDIT_COMPLETED"
    assert audit.verdict == "PASS_RECONCILED_WITH_M83_CORRECTIONS"

    # Verify M82 metrics
    m82_m = audit.m82_reported_metrics
    assert m82_m["m81_experienced_median_steps"] == 45
    assert m82_m["m82_learned_controller_median_steps"] == 30
    assert m82_m["relative_step_reduction_pct"] == 33.33

    # Verify audit of Random policy comparison
    rnd = audit.random_policy_paired_uncertainty_audit
    assert rnd["df"] == 4
    assert rnd["is_significant_05"] is False
    assert round(rnd["p_value"], 2) == 0.76

    # Verify M83 target gates are defined
    gates = audit.m83_target_gates
    assert "gate_parameter_compression" in gates
    assert "gate_skill_retention" in gates
    assert "gate_bit_exact_rollback" in gates


def test_consolidation_candidate_analyzer() -> None:
    """Verify multi-metric compatibility evaluation between adapter pairs."""
    adapter_a = BottleneckSequenceAdapter(d_model=128, bottleneck_dim=8, num_symbols=16)
    adapter_b = BottleneckSequenceAdapter(d_model=128, bottleneck_dim=8, num_symbols=16)

    # Identical adapter compatibility test
    probe_h = torch.randn(16, 4, 128)
    probe_y = torch.randint(0, 16, (16, 4))

    compat = ConsolidationCandidateAnalyzer.analyze_pair(
        skill_id_a="SKILL_A",
        adapter_a=adapter_a,
        skill_id_b="SKILL_B",
        adapter_b=adapter_a,  # Compared to self
        probe_h=probe_h,
        probe_y_a=probe_y,
        probe_y_b=probe_y,
    )

    assert compat.weight_similarity == 1.0
    assert compat.representation_similarity == 1.0
    assert compat.functional_agreement == 1.0
    assert compat.gradient_alignment == 1.0
    assert compat.composite_compatibility_score == 1.0
    assert compat.is_redundant_candidate is True
    assert compat.is_conflicting_candidate is False


def test_consolidated_adapter_architecture_and_compression() -> None:
    """Verify consolidated adapter forward pass, residuals, and parameter compression."""
    adapter = ConsolidatedSharedAdapter(d_model=128, shared_rank=8, residual_rank=2, num_symbols=16)

    # 1. Base shared trunk parameter count
    trunk_params = adapter.count_parameters()
    assert trunk_params == 1432

    # 2. Register 3 skill residuals
    adapter.register_skill_residual("skill_1")
    adapter.register_skill_residual("skill_2")
    adapter.register_skill_residual("skill_3")

    total_params = adapter.count_parameters()
    # Each residual is: 8*2 + 2 = 18; 2*16 + 16 = 48 -> 66 params
    assert total_params == 1432 + 3 * 66
    assert total_params == 1630

    # 3. Compression calculation verification (3 skills: 4,296 -> 1,630)
    savings = ConsolidatedSharedAdapter.calculate_compression_savings(num_skills=3)
    assert savings["baseline_uncompressed_parameters"] == 4296
    assert savings["consolidated_parameters"] == 1630
    assert savings["relative_compression_pct"] > 60.0
    assert savings["meets_target_20pct"] is True

    # 4. Forward pass execution
    dummy_input = torch.randn(4, 4, 128)
    out_shared = adapter(dummy_input)
    assert out_shared.shape == (4, 4, 16)

    out_skill1 = adapter(dummy_input, skill_id="skill_1")
    assert out_skill1.shape == (4, 4, 16)


def test_learned_consolidation_controller() -> None:
    """Verify consolidation policy architecture, parameter budget, and action decisions."""
    ctrl = LearnedConsolidationController(in_features=6, hidden_dim=24)

    # 1. Parameter budget check (< 1,000 parameters)
    assert ctrl.count_parameters() < 1000
    assert ctrl.count_parameters() == 939

    # 2. Forward pass dimensions
    state = ConsolidationState(
        weight_similarity=0.85,
        representation_similarity=0.90,
        functional_agreement=0.95,
        gradient_alignment=0.80,
        compression_potential=0.62,
        memory_budget_pressure=0.75,
    )
    state_tensor = state.to_tensor()
    assert state_tensor.shape == (6,)

    logits = ctrl(state_tensor)
    assert logits.shape == (1, 3)

    # 3. Action selection
    action = ctrl.select_decision(state, deterministic=True)
    assert isinstance(action, ConsolidationAction)


def test_bit_exact_rollback() -> None:
    """Verify bit-exact state restoration upon trial failure."""
    adapter = BottleneckSequenceAdapter(d_model=128, bottleneck_dim=8, num_symbols=16)
    initial_hash = adapter.compute_parameter_hash()
    saved_state = copy.deepcopy(adapter.state_dict())

    # Perturb weights
    with torch.no_grad():
        for p in adapter.parameters():
            p.add_(torch.randn_like(p))

    assert adapter.compute_parameter_hash() != initial_hash

    # Rollback
    adapter.load_state_dict(saved_state)
    assert adapter.compute_parameter_hash() == initial_hash
