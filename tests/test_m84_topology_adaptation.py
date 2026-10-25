"""Milestone M84 Test Suite: Autonomous Neural Connection Pruning and Regrowth.

Verifies:
1. M83 Forensic Audit and Amortized Overhead Analysis.
2. Dynamic Sparse Neural Connections & Autograd Gradient Saliency for Regrowth.
3. Magnitude Pruning, Gradient Regrowth, and Capacity-Preserving Rewiring.
4. Bit-Exact Topology State Rollback via SHA-256 State Fingerprints.
5. Learned Topology Controller Parameter Bounds (<1,000 params, exactly 964 params).
6. Sparse Adapter Training Engine across Comparative Arms.
7. M84 Master Benchmark Smoke Test and Report Generation.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from adaptive_mini_gpt.model import MiniGPT, ModelConfig
from adaptive_mini_gpt.topology_adaptation.audit_m83 import (
    M83ForensicAuditor,
)
from adaptive_mini_gpt.topology_adaptation.benchmark import (
    M84TopologyAdaptationBenchmark,
)
from adaptive_mini_gpt.topology_adaptation.sparse_adapter import (
    DynamicSparseSequenceAdapter,
    PruningCriterion,
    RegrowthCriterion,
    SparseMaskedLinear,
    TopologyStats,
)
from adaptive_mini_gpt.topology_adaptation.sparse_trainer import (
    SparseAdapterTrainer,
    TopologyTrainingStrategy,
)
from adaptive_mini_gpt.topology_adaptation.topology_controller import (
    LearnedTopologyController,
    TopologyAction,
    TopologyState,
)


def test_m83_forensic_audit() -> None:
    """Audit M83 claims and confirm net vs marginal compression accounting."""
    audit = M83ForensicAuditor.execute_audit()

    assert audit.reported_compression == pytest.approx(0.5857, abs=1e-3)
    assert audit.net_compression_1cluster == pytest.approx(0.3671, abs=1e-3)
    assert audit.net_compression_5cluster == pytest.approx(0.5419, abs=1e-3)
    assert audit.controller_parameters == 939
    assert audit.net_compression_1cluster < audit.reported_compression


def test_sparse_masked_linear_unmasked_gradients() -> None:
    """Verify that SparseMaskedLinear provides non-zero unmasked gradients for inactive weights."""
    linear = SparseMaskedLinear(in_features=16, out_features=8, bias=True)
    assert linear.count_total_connections() == 128
    assert linear.count_active_connections() == 128
    assert linear.get_sparsity() == 0.0

    # Prune 50% of connections
    linear.mask[4:, :] = 0.0
    linear.enforce_mask()
    assert linear.count_active_connections() == 64
    assert linear.get_sparsity() == pytest.approx(0.5, abs=1e-4)

    # Forward pass
    x = torch.randn(4, 16, requires_grad=True)
    out = linear(x)
    loss = out.sum()
    loss.backward()

    # In standard masked autograd, grad would be zero for masked rows.
    # In MaskedLinearFunction, grad_weight is unmasked (x^T @ grad_out), enabling RigL regrowth!
    assert linear.weight.grad is not None
    assert torch.any(linear.weight.grad[4:, :] != 0.0)


def test_dynamic_sparse_sequence_adapter_pruning() -> None:
    """Verify magnitude pruning on DynamicSparseSequenceAdapter."""
    adapter = DynamicSparseSequenceAdapter(d_model=128, bottleneck_dim=8, num_symbols=16)
    stats_init = adapter.get_topology_stats()

    # 128 * 8 (down) + 8 * 16 (up) = 1024 + 128 = 1152 connections
    assert stats_init.total_connections == 1152
    assert stats_init.active_connections == 1152
    assert stats_init.sparsity_ratio == 0.0

    # Prune 30% connections
    pruned_count = adapter.prune_connections(prune_ratio=0.30, criterion=PruningCriterion.MAGNITUDE)
    assert pruned_count > 0

    stats_pruned = adapter.get_topology_stats()
    assert stats_pruned.sparsity_ratio == pytest.approx(0.30, abs=0.02)
    assert stats_pruned.active_connections + stats_pruned.pruned_connections == 1152

    # Enforce masks ensures pruned weights are strictly 0.0
    adapter.enforce_masks()
    down_zeroed = (adapter.down_proj.weight[adapter.down_proj.mask == 0.0] == 0.0).all()
    up_zeroed = (adapter.up_proj.weight[adapter.up_proj.mask == 0.0] == 0.0).all()
    assert down_zeroed
    assert up_zeroed


def test_dynamic_sparse_sequence_adapter_regrowth() -> None:
    """Verify gradient-salient connection regrowth on pruned adapter."""
    adapter = DynamicSparseSequenceAdapter(d_model=128, bottleneck_dim=8, num_symbols=16)
    adapter.prune_connections(prune_ratio=0.40, criterion=PruningCriterion.MAGNITUDE)
    stats_pruned = adapter.get_topology_stats()
    init_active = stats_pruned.active_connections

    # Forward & backward pass to compute gradients on inactive links
    x = torch.randn(2, 4, 128)
    out = adapter(x)
    loss = out.sum()
    loss.backward()

    # Regrow 20 connections using gradient saliency
    regrown_count = adapter.regrow_connections(regrow_count=20, criterion=RegrowthCriterion.GRADIENT_SALIENT)
    assert regrown_count == 20

    stats_regrown = adapter.get_topology_stats()
    assert stats_regrown.active_connections == init_active + 20


def test_dynamic_sparse_sequence_adapter_rewiring() -> None:
    """Verify coupled prune-and-regrow rewiring maintaining active connection budget."""
    adapter = DynamicSparseSequenceAdapter(d_model=128, bottleneck_dim=8, num_symbols=16)
    adapter.prune_connections(prune_ratio=0.30, criterion=PruningCriterion.MAGNITUDE)
    stats_before = adapter.get_topology_stats()

    # Compute gradients
    x = torch.randn(2, 4, 128)
    out = adapter(x)
    loss = out.sum()
    loss.backward()

    pruned_cnt, regrown_cnt = adapter.rewire_topology(
        rewire_fraction=0.10,
        prune_criterion=PruningCriterion.MAGNITUDE,
        regrow_criterion=RegrowthCriterion.GRADIENT_SALIENT,
        seed=42,
    )
    assert pruned_cnt == regrown_cnt
    stats_after = adapter.get_topology_stats()
    assert stats_after.active_connections == stats_before.active_connections


def test_bit_exact_rollback() -> None:
    """Verify deterministic rollback of weights and masks using SHA-256 state hash."""
    adapter = DynamicSparseSequenceAdapter(d_model=128, bottleneck_dim=8, num_symbols=16)
    adapter.prune_connections(prune_ratio=0.25, criterion=PruningCriterion.MAGNITUDE)
    adapter.enforce_masks()

    checkpoint = adapter.save_topology_state()
    saved_hash = adapter.compute_topology_hash()

    # Mutate weights and masks
    with torch.no_grad():
        adapter.down_proj.mask.bernoulli_(0.5)
        adapter.down_proj.weight.add_(torch.randn_like(adapter.down_proj.weight))
    perturbed_hash = adapter.compute_topology_hash()
    assert saved_hash != perturbed_hash

    # Rollback
    adapter.load_topology_state(checkpoint)
    restored_hash = adapter.compute_topology_hash()
    assert saved_hash == restored_hash


def test_learned_topology_controller() -> None:
    """Verify controller parameter bound (<1,000 params) and decision execution."""
    controller = LearnedTopologyController(in_features=6, hidden_dim=24)
    param_count = controller.count_parameters()

    # Strict Gate Requirement: strictly < 1,000 parameters
    assert param_count < 1000
    assert param_count == 964

    # Evaluate decision output
    state = TopologyState(
        current_sparsity=0.25,
        mean_weight_magnitude=0.15,
        gradient_norm=0.08,
        loss_slope=-0.02,
        prediction_entropy=1.2,
        budget_pressure=0.20,
    )
    action = controller.select_action(state, deterministic=True)
    assert isinstance(action, TopologyAction)
    assert action in (
        TopologyAction.MAINTAIN_TOPOLOGY,
        TopologyAction.PRUNE_LEAST_IMPORTANT,
        TopologyAction.REGROW_GRADIENT_SALIENT,
        TopologyAction.BALANCED_REWIRE,
    )


def test_sparse_trainer_smoke() -> None:
    """Smoke test training across comparative strategies."""
    controller = LearnedTopologyController(in_features=6, hidden_dim=24)
    train_h = torch.randn(8, 4, 128)
    train_y = torch.randint(0, 16, (8, 4))
    qual_h = torch.randn(4, 4, 128)
    qual_y = torch.randint(0, 16, (4, 4))

    strategies_to_test = [
        TopologyTrainingStrategy.ORIGINAL_DENSE,
        TopologyTrainingStrategy.STATIC_MAGNITUDE_PRUNING,
        TopologyTrainingStrategy.RIGL_GRADIENT_REGROWTH,
        TopologyTrainingStrategy.LEARNED_TOPOLOGY_CONTROLLER,
    ]

    for strat in strategies_to_test:
        adapter = DynamicSparseSequenceAdapter(d_model=128, bottleneck_dim=8, num_symbols=16)
        outcome = SparseAdapterTrainer.train_adapter(
            adapter=adapter,
            train_h=train_h,
            train_y=train_y,
            qual_h=qual_h,
            qual_y=qual_y,
            strategy=strat,
            target_sparsity=0.30,
            total_steps=5,
            lr=0.01,
            rewire_freq=2,
            rewire_end_step=4,
            controller=controller,
            seed=42,
        )
        assert outcome.strategy == strat
        assert len(outcome.history_loss) == 5
        assert outcome.final_loss > 0.0


def test_m84_benchmark_smoke() -> None:
    """Smoke test full M84 master benchmark on a single seed."""
    torch.manual_seed(42)
    cfg = ModelConfig(
        vocab_size=1024,
        context_length=128,
        d_model=128,
        n_heads=4,
        n_layers=2,
        d_ff=512,
    )
    model = MiniGPT(cfg)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    device = torch.device("cpu")

    result = M84TopologyAdaptationBenchmark.run_benchmark(
        model=model,
        device=device,
        seeds=[42],
        target_sparsity=0.30,
    )

    assert "report_data" in result
    assert "markdown_summary" in result
    assert result["status"] in ("PASS", "PARTIAL")

    report_data = result["report_data"]
    assert report_data["milestone"] == "M84"
    assert "gates" in report_data
    assert report_data["gates"]["gate_capacity_and_overhead_accounting"]["controller_parameters"] == 964
    assert report_data["gates"]["gate_bit_exact_rollback"]["passed"] is True
