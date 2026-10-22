"""Unit tests for M38: Controlled Neural Structural Evolution."""

import pytest
import torch

from adaptive_mini_gpt.curriculum_discovery.structural_evolution import (
    StructuralEvolutionGate,
    StructuralModificationType,
    StructuralProposal,
)
from adaptive_mini_gpt.model.config import ModelConfig
from adaptive_mini_gpt.model.transformer import MiniGPT
from adaptive_mini_gpt.topology.benchmark_suite import (
    SEQUENTIAL_EPISODES,
    SOURCE_FACTUAL_TASKS,
    UNSEEN_ARITHMETIC_TASKS,
    generate_carry_retention_suite,
)
from adaptive_mini_gpt.topology.gated_skip import (
    ArchitectureSpec,
    GatedSkipConnection,
    TopologyEvolutionManager,
)


def _create_small_test_model() -> MiniGPT:
    """Create a minimal 4-layer MiniGPT for fast unit testing."""
    config = ModelConfig(
        vocab_size=128,
        context_length=64,
        d_model=32,
        n_layers=4,
        n_heads=2,
        d_ff=64,
        dropout=0.0,
    )
    return MiniGPT(config)


def test_gated_skip_connection_numerical_equivalence() -> None:
    """Test that zero-initialized skip connection yields bit-for-bit identical outputs."""
    model = _create_small_test_model()
    model.eval()

    test_input = torch.randint(0, 128, (2, 8), dtype=torch.long)

    # Output without skip
    with torch.no_grad():
        out_parent = model(test_input)
        parent_logits = out_parent[0] if isinstance(out_parent, tuple) else out_parent

    # Attach skip connection Block 1 -> Block 3
    skip = TopologyEvolutionManager.attach_gated_skip(
        model, source_layer=1, target_layer=3, gate_type="scalar", name="test_skip"
    )
    assert skip.get_alpha_value() == 0.0

    # Output with skip at alpha=0
    is_equiv, max_diff = TopologyEvolutionManager.verify_numerical_equivalence(
        model, test_input, atol=1e-6
    )
    assert is_equiv is True
    assert max_diff <= 1e-6


def test_gated_skip_connection_reversibility() -> None:
    """Test that setting enabled=False or removing skip connection restores parent behavior."""
    model = _create_small_test_model()
    model.eval()

    test_input = torch.randint(0, 128, (2, 8), dtype=torch.long)

    # Initial baseline
    with torch.no_grad():
        out_base = model(test_input)
        base_logits = out_base[0] if isinstance(out_base, tuple) else out_base

    # Attach and manually change alpha
    skip = TopologyEvolutionManager.attach_gated_skip(
        model, source_layer=1, target_layer=3, gate_type="scalar", name="test_skip"
    )
    skip.alpha.data.fill_(1.5)

    # Output with active non-zero alpha
    with torch.no_grad():
        out_active = model(test_input)
        active_logits = out_active[0] if isinstance(out_active, tuple) else out_active

    # They should differ
    assert not torch.allclose(base_logits, active_logits, atol=1e-4)

    # Now disable skip
    skip.disable()
    with torch.no_grad():
        out_disabled = model(test_input)
        disabled_logits = out_disabled[0] if isinstance(out_disabled, tuple) else out_disabled

    # Disabled must match base logits exactly
    assert torch.allclose(base_logits, disabled_logits, atol=1e-6)

    # Remove skip
    TopologyEvolutionManager.remove_gated_skip(model, name="test_skip")
    with torch.no_grad():
        out_removed = model(test_input)
        removed_logits = out_removed[0] if isinstance(out_removed, tuple) else out_removed

    assert torch.allclose(base_logits, removed_logits, atol=1e-6)


def test_architecture_spec_export() -> None:
    """Test exporting versioned architecture specifications."""
    model = _create_small_test_model()
    TopologyEvolutionManager.attach_gated_skip(
        model, source_layer=1, target_layer=3, gate_type="scalar", name="skip_l1_l3"
    )

    spec = TopologyEvolutionManager.export_architecture_spec(
        model, parent_hash="test_parent_hash", version="1.1.0"
    )
    assert isinstance(spec, ArchitectureSpec)
    assert spec.version == "1.1.0"
    assert spec.topology_type == "gated_skip_inter_block"
    assert len(spec.connections) == 1
    assert spec.connections[0]["name"] == "skip_l1_l3"
    assert spec.connections[0]["source_layer"] == 1
    assert spec.connections[0]["target_layer"] == 3
    assert spec.reversibility_verified is True


def test_structural_evolution_gate_decisions() -> None:
    """Test that StructuralEvolutionGate correctly evaluates structural proposals."""
    gate = StructuralEvolutionGate(max_parameter_growth_ratio=0.20, min_expected_gain=0.15)
    base_params = 1_000_000

    # 1. Proposal with sub-threshold gain (< 0.15)
    p_low_gain = StructuralProposal.create(
        modification_type=StructuralModificationType.WEIGHT_ADJUSTMENT,
        target_module="backbone",
        evidence_of_need="Minor gain",
        expected_gain=0.08,
        parameter_delta=0,
        compute_overhead_ratio=0.0,
        current_checkpoint_hash="abc123",
    )
    approved, reason = gate.evaluate_proposal(p_low_gain, base_params)
    assert approved is False
    assert "below minimum gate" in reason

    # 2. Proposal exceeding growth ceiling (> 20%)
    p_high_growth = StructuralProposal.create(
        modification_type=StructuralModificationType.EXPAND_DIMENSIONS,
        target_module="all_layers",
        evidence_of_need="High capacity expansion",
        expected_gain=0.40,
        parameter_delta=300_000,
        compute_overhead_ratio=0.35,
        current_checkpoint_hash="abc123",
    )
    approved, reason = gate.evaluate_proposal(p_high_growth, base_params)
    assert approved is False
    assert "exceeds ceiling" in reason

    # 3. Valid Gated Skip Connection proposal (1 param, +0.20 gain)
    p_valid_skip = StructuralProposal.create(
        modification_type=StructuralModificationType.GATED_SKIP_CONNECTION,
        target_module="blocks.1_to_blocks.3",
        evidence_of_need="Inter-block gradient flow",
        expected_gain=0.20,
        parameter_delta=1,
        compute_overhead_ratio=0.01,
        current_checkpoint_hash="abc123",
    )
    approved, reason = gate.evaluate_proposal(p_valid_skip, base_params)
    assert approved is True
    assert "approved" in reason.lower()


def test_cross_domain_retention_matrix_suite() -> None:
    """Test retention matrix task suites and carry transitions generation."""
    suite = generate_carry_retention_suite()
    assert len(suite) == 200
    for a, b, c_in in suite:
        assert 0 <= a <= 9
        assert 0 <= b <= 9
        assert c_in in (0, 1)

    assert len(SEQUENTIAL_EPISODES) == 4
    subjects = [ep["subject"] for ep in SEQUENTIAL_EPISODES]
    assert subjects == ["physics", "biology", "computing", "history"]


def test_topology_evolution_manager_parameter_freezing() -> None:
    """Test freezing base model parameters while keeping skip connection trainable."""
    model = _create_small_test_model()
    TopologyEvolutionManager.attach_gated_skip(
        model, source_layer=1, target_layer=3, gate_type="scalar", name="skip_l1_l3"
    )

    trainable, frozen = TopologyEvolutionManager.freeze_for_structural_training(
        model, target_connection="skip_l1_l3"
    )
    assert trainable == 1
    assert frozen > 0
    for name, p in model.named_parameters():
        if "skip_connections.skip_l1_l3" in name:
            assert p.requires_grad is True
        else:
            assert p.requires_grad is False
