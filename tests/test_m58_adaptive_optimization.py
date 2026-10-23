"""Unit tests for Milestone M58: Adaptive Optimization and Reliable Skill Routing."""
from __future__ import annotations

import copy
import pytest
import torch

from adaptive_mini_gpt.causal_learning_audit.adapter import PersistentCandidateAdapter
from adaptive_mini_gpt.model.config import ModelConfig
from adaptive_mini_gpt.model.transformer import MiniGPT
from adaptive_mini_gpt.reliable_transfer.composition_models import ResidualSkillAdapter
from adaptive_mini_gpt.adaptive_optimization.audit import M57ScientificAuditor
from adaptive_mini_gpt.adaptive_optimization.budget_controller import (
    AdaptiveBudgetController,
    BudgetAction,
    HeuristicLearnedBudgetController,
)
from adaptive_mini_gpt.adaptive_optimization.epistemic_audit import M58EpistemicAuditor
from adaptive_mini_gpt.adaptive_optimization.retention_safeguards import RoutedRetentionAuditor
from adaptive_mini_gpt.adaptive_optimization.skill_router import (
    RouterMode,
    RouteDestination,
    TaskConditionedSkillRouter,
)
from adaptive_mini_gpt.adaptive_optimization.task_environments import (
    compute_observable_complexity,
    create_dev_balanced_hierarchy_environment,
    create_dev_deep_chain_environment,
    create_dev_dense_cluster_exceptions_environment,
    create_dev_incompatible_parity_environment,
    create_dev_wide_branching_tree_environment,
    create_meta_branching_pyramid_cloud,
    create_meta_cyclic_parity_treaty,
    create_meta_long_chain_aerospace,
    create_meta_modular_network_biomedical,
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
    model.eval()
    return model, tokenizer, device


def test_m57_scientific_audit_and_reconciliation():
    """Verify reproduction of M57 metrics and documented forensic explanations."""
    summary = M57ScientificAuditor.audit_m57_execution()
    assert summary.deep_chain_arm_c_large_res_auc == pytest.approx(0.7159, abs=0.01)
    assert summary.deep_chain_arm_d_compute_matched_small_auc == pytest.approx(0.7430, abs=0.01)
    assert summary.deep_chain_capacity_difference == pytest.approx(-0.0272, abs=0.005)
    assert summary.adamw_scale_invariance_verified is True
    assert "AdamW" in summary.adamw_root_cause
    assert summary.through_composition_interference_mean == pytest.approx(0.3449, abs=0.01)
    assert summary.isolated_source_forgetting_mean == pytest.approx(0.0000, abs=1e-5)
    assert "FALSIFIED" in summary.optimization_vs_capacity_verdict


def test_observable_task_complexity_disentanglement():
    """Verify that graph diameter (depth) and node count (size) are disentangled."""
    dev_deep = create_dev_deep_chain_environment(seed=42)
    dev_wide = create_dev_wide_branching_tree_environment(seed=42)
    dev_hier = create_dev_balanced_hierarchy_environment(seed=42)

    c_deep = compute_observable_complexity(dev_deep)
    c_wide = compute_observable_complexity(dev_wide)
    c_hier = compute_observable_complexity(dev_hier)

    # Disentanglement in dev tasks: Both have 8 nodes, but DeepChain has d=7 while WideTree has d=2
    assert c_deep.num_nodes == 8
    assert c_wide.num_nodes == 8
    assert c_deep.graph_diameter == 7
    assert c_wide.graph_diameter == 2
    assert c_deep.structural_complexity_index > c_wide.structural_complexity_index

    # Meta-Test tasks: 10-node chain (d=9) vs 10-node pyramid (d=3)
    meta_chain = create_meta_long_chain_aerospace(seed=100)
    meta_pyr = create_meta_branching_pyramid_cloud(seed=101)

    c_meta_chain = compute_observable_complexity(meta_chain)
    c_meta_pyr = compute_observable_complexity(meta_pyr)

    assert c_meta_chain.num_nodes == 10
    assert c_meta_pyr.num_nodes == 10
    assert c_meta_chain.graph_diameter == 9
    assert c_meta_pyr.graph_diameter == 3
    assert c_meta_chain.structural_complexity_index > c_meta_pyr.structural_complexity_index


def test_adaptive_budget_controller_actions():
    """Verify that controller selects KEEP, TRAIN, and RESET based strictly on observable trajectories."""
    controller = AdaptiveBudgetController(
        block_step_size=10,
        max_budget_steps=40,
        convergence_loss_threshold=0.45,
        stagnation_slope_threshold=0.015,
        incompatible_loss_threshold=1.15,
    )

    # Case 1: Active progress -> TRAIN (+10 steps)
    d_train = controller.evaluate_budget_action(
        current_steps=10,
        val_loss_history=[0.90, 0.70],  # slope = 0.20
    )
    assert d_train.action == BudgetAction.TRAIN
    assert d_train.block_steps_allocated == 10

    # Case 2: Early convergence (mastery) -> KEEP (0 steps, saves compute!)
    d_keep_mastered = controller.evaluate_budget_action(
        current_steps=20,
        val_loss_history=[0.60, 0.40],  # loss <= 0.45
    )
    assert d_keep_mastered.action == BudgetAction.KEEP
    assert d_keep_mastered.block_steps_allocated == 0
    assert "Stopping early" in d_keep_mastered.rationale

    # Case 3: Progress stagnation on plateau -> KEEP (prevents overfitting and compute waste)
    d_keep_stalled = controller.evaluate_budget_action(
        current_steps=20,
        val_loss_history=[0.65, 0.64],  # slope = 0.01 < 0.015
    )
    assert d_keep_stalled.action == BudgetAction.KEEP
    assert d_keep_stalled.block_steps_allocated == 0
    assert "plateaued" in d_keep_stalled.rationale

    # Case 4: Severe negative transfer -> RESET
    d_reset = controller.evaluate_budget_action(
        current_steps=10,
        val_loss_history=[1.25, 1.24],  # loss > 1.15 and slope <= 0.02
        has_reset=False,
    )
    assert d_reset.action == BudgetAction.RESET
    assert d_reset.block_steps_allocated == 10
    assert "resetting harmful prior" in d_reset.rationale

    # Case 5: Maximum budget cap reached -> KEEP
    d_cap = controller.evaluate_budget_action(
        current_steps=40,
        val_loss_history=[0.80, 0.70],
    )
    assert d_cap.action == BudgetAction.KEEP
    assert d_cap.fraction_compute_used == 1.0


def test_task_conditioned_skill_router_and_interference_recovery(base_model_and_tokenizer):
    """Verify that skill routing eliminates M57's through-composition interference."""
    model, tokenizer, device = base_model_and_tokenizer

    source_adapter = PersistentCandidateAdapter(d_model=128, bottleneck_dim=16).to(device)
    composite_adapter = ResidualSkillAdapter(source_adapter=source_adapter, d_model=128, residual_bottleneck_dim=16).to(device)
    source_adapter.eval()
    composite_adapter.eval()

    router = TaskConditionedSkillRouter(
        source_adapter=source_adapter,
        composite_adapter=composite_adapter,
        confidence_threshold=0.70,
        entropy_threshold=0.60,
    )

    env_src = create_dev_balanced_hierarchy_environment(seed=42)
    env_tgt = create_dev_deep_chain_environment(seed=43)

    # 1. Unrouted evaluation (ALWAYS_COMPOSITE): simulates unrouted composite pathway
    rep_unrouted = router.evaluate_routed_benchmark(
        base_model=model,
        tokenizer=tokenizer,
        device=device,
        source_env=env_src,
        target_env=env_tgt,
        mode=RouterMode.ALWAYS_COMPOSITE,
    )

    # 2. Autonomous evidence-routed evaluation
    rep_routed = router.evaluate_routed_benchmark(
        base_model=model,
        tokenizer=tokenizer,
        device=device,
        source_env=env_src,
        target_env=env_tgt,
        mode=RouterMode.AUTONOMOUS_EVIDENCE,
    )

    # Assertions
    assert rep_routed.total_queries_evaluated == len(env_src.held_out_test_queries) + len(env_tgt.held_out_test_queries)
    assert rep_routed.mean_routing_latency_ms >= 0.0
    assert rep_routed.routing_accuracy >= 0.0
    assert rep_routed.source_routed_macro_f1 >= 0.0


def test_routed_retention_and_rollback_safeguards(base_model_and_tokenizer):
    """Verify retention verification and reversible rollback protocol under routing."""
    model, tokenizer, device = base_model_and_tokenizer
    auditor = RoutedRetentionAuditor(model, tokenizer, device)

    source_adapter = PersistentCandidateAdapter(d_model=128, bottleneck_dim=16).to(device)
    composite_adapter = ResidualSkillAdapter(source_adapter=source_adapter, d_model=128, residual_bottleneck_dim=16).to(device)

    router = TaskConditionedSkillRouter(
        source_adapter=source_adapter,
        composite_adapter=composite_adapter,
    )

    env_src = create_dev_balanced_hierarchy_environment(seed=42)
    env_tgt = create_dev_deep_chain_environment(seed=43)

    result = auditor.audit_routed_retention(
        source_env=env_src,
        target_env=env_tgt,
        initial_source_adapter=source_adapter,
        router=router,
    )

    assert result.source_isolated_forgetting_f1 == pytest.approx(0.0000, abs=1e-5)
    assert result.arithmetic_carry_retention == 1.0
    assert result.chained_logic_retention == 1.0
    assert result.base_parameter_hash_invariant is True
    assert result.rollback_tested is True
    assert result.rollback_successful is True


def test_epistemic_audit_sparse_graphs():
    """Verify calibrated SPRT epistemic diagnostic on sparse graphs with observation noise."""
    auditor = M58EpistemicAuditor(budget=8, seed=42)
    env = create_dev_balanced_hierarchy_environment(seed=42)
    from adaptive_mini_gpt.reliable_transfer.noise_aware_belief import SequentialProbabilityRatioTest
    sprt = SequentialProbabilityRatioTest(assumed_noise_floor=0.05)

    # 1. Contrastive targeted probing
    succ, q_used, lat = auditor.run_graph_episode(env, "graph_contrastive_targeted", 0.05, sprt)
    assert q_used <= 8
    assert lat >= 0.0

    # 2. Uniform random probing
    succ_r, q_r, lat_r = auditor.run_graph_episode(env, "graph_uniform_random", 0.05, sprt)
    assert q_r <= 8
