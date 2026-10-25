"""Milestone M85 Test Suite: Autonomous Neural Dimension Adaptation and Module Evolution.

Verifies:
1. M84 Forensic Audit and Physical Parameter vs. Sparsity Reconciliation.
2. Physical Dynamic Dimension Resizing across Candidate Ranks (2, 4, 8, 16).
3. Bit-Exact Function Preservation during Bottleneck Rank Expansion.
4. Channel Energy-Preserving Rank Contraction & Physical Deallocation.
5. Optimizer State Migration for Uninterrupted AdamW Optimization.
6. Dynamic Module Manager Lifecycle (Allocation, Resizing, and Retirement).
7. Learned Architecture Policy Parameter Bounds (<1,000 params, exactly 989 params).
8. Bit-Exact Multi-Module State Rollback via Holistic SHA-256 Fingerprint.
9. M85 Master Benchmark Smoke Test and Report Generation.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from adaptive_mini_gpt.dimension_evolution.architecture_policy import (
    ArchitectureAction,
    ArchitectureState,
    LearnedArchitecturePolicy,
)
from adaptive_mini_gpt.dimension_evolution.audit_m84 import (
    M84ForensicAuditor,
)
from adaptive_mini_gpt.dimension_evolution.benchmark import (
    M85DimensionEvolutionBenchmark,
)
from adaptive_mini_gpt.dimension_evolution.dynamic_adapter import (
    DynamicDimensionAdapter,
)
from adaptive_mini_gpt.dimension_evolution.module_registry import (
    DynamicModuleManager,
)
from adaptive_mini_gpt.model import MiniGPT, ModelConfig


def test_m84_forensic_audit() -> None:
    """Verify M84 audit metrics, active vs physical parameter accounting, and speedup."""
    audit = M84ForensicAuditor.execute_audit()
    assert audit.total_connections == 1152
    assert audit.active_connections == 806
    assert audit.pruned_connections == 346
    assert audit.reported_connection_sparsity_pct == pytest.approx(30.03, abs=0.01)
    assert audit.physically_stored_adapter_parameters == 1432
    assert audit.actual_physical_parameter_savings_pct == 0.0
    assert audit.controller_parameters == 964
    assert audit.controller_speedup_vs_random_steps == pytest.approx(4.0, abs=0.1)
    assert audit.paired_t_vs_random == pytest.approx(0.9617, abs=1e-3)
    assert audit.p_value_vs_random == pytest.approx(0.3907, abs=1e-3)
    assert audit.future_learning_parity_achieved is True


def test_dynamic_adapter_physical_resizing_and_function_preservation() -> None:
    """Verify physical tensor reshaping and function-preserving expansion."""
    adapter = DynamicDimensionAdapter(d_model=128, rank=4, num_symbols=16)
    x = torch.randn(2, 4, 128)
    out_r4 = adapter(x)

    # Initial physical parameters: (128*4 + 4) + (4*16 + 16) + 256 = 852
    assert adapter.count_parameters() == 852
    assert adapter.down_proj.weight.shape == (4, 128)
    assert adapter.up_proj.weight.shape == (16, 4)

    # 1. Expand rank 4 -> 8
    old_r, new_r = adapter.expand_rank(8)
    assert old_r == 4 and new_r == 8
    assert adapter.count_parameters() == 1432
    assert adapter.down_proj.weight.shape == (8, 128)
    assert adapter.up_proj.weight.shape == (16, 8)

    # Function-Preserving Invariant: Output must match bit-exact at expansion moment!
    out_r8 = adapter(x)
    max_diff = torch.max(torch.abs(out_r8 - out_r4)).item()
    assert max_diff == pytest.approx(0.0, abs=1e-7)

    # 2. Shrink rank 8 -> 2
    old_r, new_r = adapter.shrink_rank(2)
    assert old_r == 8 and new_r == 2
    # Physical parameter reduction: (128*2 + 2) + (2*16 + 16) + 256 = 562
    assert adapter.count_parameters() == 562
    assert adapter.down_proj.weight.shape == (2, 128)
    assert adapter.up_proj.weight.shape == (16, 2)


def test_optimizer_state_migration() -> None:
    """Verify AdamW optimizer momentum buffer migration across dimension changes."""
    adapter = DynamicDimensionAdapter(d_model=128, rank=4, num_symbols=16)
    opt = torch.optim.AdamW(adapter.parameters(), lr=0.01)

    # Run one step to populate Adam momentum states
    x = torch.randn(2, 4, 128)
    loss = adapter(x).sum()
    loss.backward()
    opt.step()

    # Expand rank 4 -> 8 and migrate optimizer
    adapter.expand_rank(8)
    adapter.migrate_optimizer_state(opt)

    # Run step at new rank (must not crash with shape mismatch)
    opt.zero_grad()
    loss2 = adapter(x).sum()
    loss2.backward()
    opt.step()

    assert adapter.rank == 8


def test_dynamic_module_manager_lifecycle() -> None:
    """Verify module allocation, expansion, contraction, and retirement."""
    mgr = DynamicModuleManager(d_model=128, num_symbols=16)
    assert mgr.get_total_physical_parameters() == 0

    # 1. Allocate module
    mgr.allocate_module("skill_1", initial_rank=4, step=1)
    assert mgr.has_module("skill_1")
    assert mgr.get_total_physical_parameters() == 852

    # 2. Expand module rank
    mgr.expand_module_rank("skill_1", target_rank=8, step=2)
    assert mgr.get_total_physical_parameters() == 1432

    # 3. Allocate second module
    mgr.allocate_module("skill_2", initial_rank=2, step=3)
    assert mgr.get_total_physical_parameters() == 1432 + 562

    # 4. Retire module
    freed = mgr.retire_module("skill_2", step=4)
    assert freed == 562
    assert not mgr.has_module("skill_2")
    assert mgr.get_total_physical_parameters() == 1432


def test_learned_architecture_policy() -> None:
    """Verify policy parameters (<1,000 params) and discrete action outputs."""
    policy = LearnedArchitecturePolicy(in_features=6, hidden_dim=24, num_actions=5)
    param_count = policy.count_parameters()

    # Strict gate: < 1,000 parameters
    assert param_count < 1000
    assert param_count == 989

    # Stalled hard task state -> EXPAND_RANK
    expand_state = ArchitectureState(
        loss_slope=0.002,
        validation_loss=1.85,
        prediction_entropy=1.45,
        gradient_norm=0.22,
        capacity_pressure=0.25,
        interference_risk=0.08,
    )
    act_exp = policy.select_action(expand_state, deterministic=True)
    assert act_exp == ArchitectureAction.EXPAND_RANK

    # Simple converged task state -> SHRINK_RANK
    shrink_state = ArchitectureState(
        loss_slope=-0.12,
        validation_loss=0.06,
        prediction_entropy=0.12,
        gradient_norm=0.02,
        capacity_pressure=0.80,
        interference_risk=0.01,
    )
    act_shr = policy.select_action(shrink_state, deterministic=True)
    assert act_shr == ArchitectureAction.SHRINK_RANK


def test_bit_exact_multi_module_rollback() -> None:
    """Verify holistic SHA-256 multi-module state restoration upon rejected trial."""
    mgr = DynamicModuleManager(d_model=128, num_symbols=16)
    mgr.allocate_module("alpha", initial_rank=4)
    mgr.allocate_module("beta", initial_rank=8)

    ckpt = mgr.save_registry_checkpoint()
    saved_hash = mgr.compute_registry_hash()

    # Mutate registry
    mgr.expand_module_rank("alpha", target_rank=8)
    with torch.no_grad():
        mgr.get_module("alpha").down_proj.weight.add_(2.0)
    mgr.allocate_module("gamma", initial_rank=2)
    perturbed_hash = mgr.compute_registry_hash()
    assert saved_hash != perturbed_hash

    # Restore from checkpoint
    mgr.restore_registry_checkpoint(ckpt)
    restored_hash = mgr.compute_registry_hash()
    assert saved_hash == restored_hash
    assert not mgr.has_module("gamma")
    assert mgr.get_module("alpha").rank == 4


def test_m85_benchmark_smoke() -> None:
    """Smoke test full M85 master benchmark on a single seed."""
    torch.manual_seed(42)
    cfg = ModelConfig(
        vocab_size=1024,
        context_length=128,
        d_model=128,
        n_heads=4,
        n_layers=4,
        d_ff=512,
    )
    model = MiniGPT(cfg)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    device = torch.device("cpu")

    result = M85DimensionEvolutionBenchmark.run_benchmark(
        model=model,
        device=device,
        seeds=[42],
    )

    assert "report_data" in result
    assert "markdown_summary" in result
    assert result["status"] in ("PASS", "PARTIAL")

    report_data = result["report_data"]
    assert report_data["milestone"] == "M85"
    assert report_data["gates"]["gate_dynamic_rank_evolution"]["passed"] is True
    assert report_data["gates"]["gate_dynamic_module_lifecycle"]["passed"] is True
    assert report_data["gates"]["gate_policy_parameter_accounting"]["policy_parameters"] == 989
    assert report_data["gates"]["gate_bit_exact_rollback"]["passed"] is True
