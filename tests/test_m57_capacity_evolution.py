"""Unit tests for Milestone M57: Controlled Neural Capacity Evolution."""
from __future__ import annotations

import copy
import pytest
import torch

from adaptive_mini_gpt.causal_learning_audit.adapter import PersistentCandidateAdapter
from adaptive_mini_gpt.model.config import ModelConfig
from adaptive_mini_gpt.model.transformer import MiniGPT
from adaptive_mini_gpt.capacity_evolution.audit import M56ScientificAuditor
from adaptive_mini_gpt.capacity_evolution.capacity_controller import (
    AdaptiveCapacityController,
    CapacityAction,
)
from adaptive_mini_gpt.capacity_evolution.epistemic_diagnostic import CalibratedSPRTDiagnosticAuditor
from adaptive_mini_gpt.capacity_evolution.evolvable_adapter import EvolvableResidualSkillAdapter
from adaptive_mini_gpt.capacity_evolution.retention_rollback import CapacityRetentionAuditor
from adaptive_mini_gpt.capacity_evolution.task_complexity import (
    compute_observable_complexity,
    create_balanced_hierarchy_environment,
    create_deep_chain_environment,
    create_dense_cluster_exceptions_environment,
    create_incompatible_parity_environment,
    create_wide_branching_tree_environment,
)


class DummyTokenizer:
    def encode(self, text: str) -> list[int]:
        return [hash(w) % 100 for w in text.split()] or [0]

    def decode(self, tokens: list[int]) -> str:
        return " ".join(str(t) for t in tokens)


@pytest.fixture
def base_model_and_tokenizer():
    config = ModelConfig(
        vocab_size=1024,
        d_model=128,
        n_layers=2,
        n_heads=2,
        d_ff=256,
        context_length=128,
        dropout=0.0,
    )
    model = MiniGPT(config)
    tokenizer = DummyTokenizer()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    return model, tokenizer, device


def test_m56_scientific_audit_and_reconciliation():
    """Verify reproduction of M56 metrics and documented forensic explanations."""
    summary = M56ScientificAuditor.audit_m56_execution()
    assert summary.reporting_hierarchy_residual_gain == pytest.approx(0.0811, abs=0.01)
    assert summary.supply_chain_residual_loss == pytest.approx(-0.0769, abs=0.01)
    assert summary.adamw_scale_invariance_identified is True
    assert "AdamW" in summary.gradient_scaling_root_cause
    assert summary.epistemic_discrepancy_resolved is True
    assert "base rate" in summary.epistemic_reconciliation_cause.lower() or "mismatch rate" in summary.epistemic_reconciliation_cause.lower()


def test_task_complexity_disentanglement():
    """Verify that graph diameter (depth) and node count (size) are disentangled."""
    deep_env = create_deep_chain_environment(seed=42)
    wide_env = create_wide_branching_tree_environment(seed=42)
    hier_env = create_balanced_hierarchy_environment(seed=42)
    dense_env = create_dense_cluster_exceptions_environment(seed=42)
    parity_env = create_incompatible_parity_environment(seed=42)

    c_deep = compute_observable_complexity(deep_env)
    c_wide = compute_observable_complexity(wide_env)
    c_hier = compute_observable_complexity(hier_env)

    # Disentanglement check: Both have 8 nodes, but DeepChain has d=7 while WideTree has d=2!
    assert c_deep.num_nodes == 8
    assert c_wide.num_nodes == 8
    assert c_deep.graph_diameter == 7
    assert c_wide.graph_diameter == 2
    assert c_deep.structural_complexity_index > c_wide.structural_complexity_index

    # Hierarchical 6-node isomorphic check
    assert c_hier.num_nodes == 6
    assert c_hier.graph_diameter == 5

    # Queries and minimal pairs check
    assert len(deep_env.held_out_test_queries) >= 16
    assert len(wide_env.held_out_test_queries) >= 16
    assert len(dense_env.held_out_test_queries) >= 16


def test_evolvable_residual_adapter_zero_init_and_rollback():
    """Verify parameter counts, zero-initialized expansion identity, and state rollback."""
    source_adapter = PersistentCandidateAdapter(d_model=128, bottleneck_dim=32)
    adapter = EvolvableResidualSkillAdapter(
        source_adapter=source_adapter,
        d_model=128,
        initial_bottleneck_dim=16,
    )
    source_adapter.eval()
    adapter.eval()

    # 1. Initial parameter counts
    counts_16 = adapter.count_parameters()
    assert counts_16["trainable"] == 2371  # 256 LN + 2064 Linear + 51 Linear = 2371
    assert counts_16["frozen"] == 4483

    # 2. Forward pass output before expansion
    dummy_h = torch.randn(2, 8, 128)
    with torch.no_grad():
        logits_before = adapter(dummy_h)

    # 3. Expand capacity to 32 dimensions
    new_trainable = adapter.expand_capacity(target_bottleneck_dim=32)
    assert new_trainable == 4483
    assert adapter.current_bottleneck_dim == 32

    # 4. Zero-Initialization Check: Output at step 0 must match pre-expansion output exactly!
    with torch.no_grad():
        logits_after_init = adapter(dummy_h)
    torch.testing.assert_close(logits_before, logits_after_init, atol=1e-5, rtol=1e-5)

    # 5. Expand further to wide capacity (64 dimensions)
    adapter.expand_capacity(target_bottleneck_dim=64)
    assert adapter.count_parameters()["trainable"] == 8707

    # 6. Rollback verification: revert back to snapshot
    adapter.rollback_to_snapshot()
    assert adapter.current_bottleneck_dim == 32
    assert adapter.count_parameters()["trainable"] == 4483


def test_bounded_adaptive_capacity_controller():
    """Verify that controller selects KEEP, EXPAND, or RESET according to evidence."""
    controller = AdaptiveCapacityController(
        stagnation_loss_delta_threshold=0.05,
        incompatible_loss_threshold=1.15,
        complexity_expansion_threshold=6.0,
    )

    deep_env = create_deep_chain_environment(seed=42)
    c_deep = compute_observable_complexity(deep_env)

    hier_env = create_balanced_hierarchy_environment(seed=42)
    c_hier = compute_observable_complexity(hier_env)

    # Case 1: High complexity task (diameter=7 >= 5) with stalled loss progress -> EXPAND
    d_expand = controller.evaluate_action(
        step_n=4,
        val_loss_initial=0.85,
        val_loss_current=0.83,  # delta = 0.02 < 0.05 (stalled)
        complexity=c_deep,
    )
    assert d_expand.action == CapacityAction.EXPAND
    assert "Expanding residual bottleneck" in d_expand.rationale

    # Case 2: Incompatible negative transfer with severe loss -> RESET
    d_reset = controller.evaluate_action(
        step_n=4,
        val_loss_initial=1.25,
        val_loss_current=1.24,  # > 1.15 and delta < 0.02
        complexity=c_deep,
    )
    assert d_reset.action == CapacityAction.RESET
    assert "resetting to fresh adapter" in d_reset.rationale

    # Case 3: Manageable task with healthy progress -> KEEP
    d_keep = controller.evaluate_action(
        step_n=4,
        val_loss_initial=0.80,
        val_loss_current=0.60,  # delta = 0.20 >= 0.05
        complexity=c_hier,
    )
    assert d_keep.action == CapacityAction.KEEP


def test_calibrated_sprt_diagnostic():
    """Verify calibrated SPRT diagnostic runs bounded graph episodes without errors."""
    auditor = CalibratedSPRTDiagnosticAuditor(budget=8, seed=42)
    env = create_balanced_hierarchy_environment(seed=42)
    from adaptive_mini_gpt.reliable_transfer.noise_aware_belief import SequentialProbabilityRatioTest
    sprt = SequentialProbabilityRatioTest(assumed_noise_floor=0.05)

    succ, q_used, lat = auditor.run_graph_episode(env, "graph_contrastive_targeted", 0.05, sprt)
    assert q_used <= 8
    assert lat >= 0.0

    succ_rand, q_used_rand, lat_rand = auditor.run_graph_episode(env, "graph_uniform_random", 0.05, sprt)
    assert q_used_rand <= 8


def test_capacity_retention_and_rollback_auditor(base_model_and_tokenizer):
    """Verify retention verification and reversible rollback protocol."""
    model, tokenizer, device = base_model_and_tokenizer
    auditor = CapacityRetentionAuditor(model, tokenizer, device)

    source_adapter = PersistentCandidateAdapter(d_model=128, bottleneck_dim=32).to(device)
    evolved_adapter = EvolvableResidualSkillAdapter(
        source_adapter=source_adapter,
        d_model=128,
        initial_bottleneck_dim=16,
    ).to(device)

    env_src = create_balanced_hierarchy_environment(seed=42)
    env_tgt = create_deep_chain_environment(seed=43)

    result = auditor.audit_evolution_retention(
        source_env=env_src,
        target_env=env_tgt,
        initial_source_adapter=source_adapter,
        evolved_adapter=evolved_adapter,
    )

    assert result.source_isolated_forgetting_f1 == pytest.approx(0.0000, abs=1e-5)
    assert result.arithmetic_carry_retention == 1.0
    assert result.chained_logic_retention == 1.0
    assert result.base_parameter_hash_invariant is True
    assert result.rollback_tested is True
    assert result.rollback_successful is True
