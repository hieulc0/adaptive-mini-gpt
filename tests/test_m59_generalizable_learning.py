"""Unit tests for Milestone M59: Generalizable Learning Decisions and Reliable Skill Routing."""
from __future__ import annotations

import copy
import pytest
import torch
import torch.nn as nn

from adaptive_mini_gpt.causal_learning_audit.adapter import PersistentCandidateAdapter
from adaptive_mini_gpt.model.config import ModelConfig
from adaptive_mini_gpt.model.transformer import MiniGPT
from adaptive_mini_gpt.reliable_transfer.composition_models import ResidualSkillAdapter
from adaptive_mini_gpt.generalizable_learning.audit import M58ScientificAuditor
from adaptive_mini_gpt.generalizable_learning.budget_controller import (
    BudgetAction,
    ControllerBudgetDecision,
    ExperientialBudgetController,
    FixedBudgetController,
    M58AdaptiveBudgetController,
    ValidationProgressCostController,
)
from adaptive_mini_gpt.generalizable_learning.epistemic_audit import CorrectedEpistemicAuditor
from adaptive_mini_gpt.generalizable_learning.experiential_memory import (
    ExperientialMemoryStore,
    LearningEpisodeRecord,
)
from adaptive_mini_gpt.generalizable_learning.retention_safeguards import RoutedRetentionAuditor
from adaptive_mini_gpt.generalizable_learning.skill_router import (
    EvidenceBasedSkillRouter,
    RouteDestination,
    RouterMode,
    compute_contrastive_representation_loss,
)
from adaptive_mini_gpt.generalizable_learning.task_suite import (
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
    create_meta_sparse_lattice_logistics,
    create_train_cluster_mesh,
    create_train_genomics_hierarchy,
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


def test_m58_scientific_audit_and_reconciliation():
    """Verify forensic reconciliation of M58 discoveries."""
    audit = M58ScientificAuditor.audit_m58_execution()
    # Split Reconciliation
    assert audit.dev_adaptive_advantage == pytest.approx(0.0306, abs=1e-4)
    assert audit.sealed_adaptive_advantage == pytest.approx(-0.0268, abs=1e-4)
    assert audit.grand_adaptive_advantage == pytest.approx(0.0019, abs=1e-4)

    # Efficiency Reconciliation
    assert audit.mean_of_per_task_efficiencies == pytest.approx(0.4233, abs=1e-4)
    assert audit.ratio_of_aggregate_means_efficiency == pytest.approx(0.390857, abs=1e-4)
    assert audit.true_step_budget_efficiency_gain_pct == pytest.approx(14.92, abs=0.1)

    # Task Accounting
    assert audit.total_target_tasks_evaluated == 8
    assert audit.source_task_name == "dev_balanced_hierarchy"
    assert "8 target environments" in audit.accounting_resolution

    # SPRT Diagnostic
    assert audit.sprt_false_alarm_rate_targeted == 1.00
    assert "transitive reachability" in audit.sprt_false_alarm_root_cause

    # Routing Tradeoff
    assert audit.always_composite_joint_f1 == pytest.approx(0.5220, abs=0.001)
    assert audit.autonomous_routed_joint_f1 == pytest.approx(0.5200, abs=0.001)
    assert audit.routing_accuracy_autonomous == pytest.approx(0.4792, abs=0.001)
    assert "FAILED" in audit.routing_joint_tradeoff_verdict


def test_observable_task_complexity_multi_split():
    """Verify topological complexity computation and depth vs size disentanglement."""
    # Meta-Train
    t_genomics = create_train_genomics_hierarchy(seed=10)
    t_mesh = create_train_cluster_mesh(seed=11)
    c_gen = compute_observable_complexity(t_genomics)
    c_mesh = compute_observable_complexity(t_mesh)
    assert c_gen.num_nodes == 6
    assert c_mesh.num_nodes == 6
    assert c_gen.graph_diameter == 5
    assert c_mesh.graph_diameter == 1 or c_mesh.graph_diameter == 2

    # Dev tasks
    d_deep = create_dev_deep_chain_environment(seed=42)
    d_wide = create_dev_wide_branching_tree_environment(seed=42)
    c_deep = compute_observable_complexity(d_deep)
    c_wide = compute_observable_complexity(d_wide)
    assert c_deep.num_nodes == 8
    assert c_wide.num_nodes == 8
    assert c_deep.graph_diameter == 7
    assert c_wide.graph_diameter == 2
    assert c_deep.structural_complexity_index > c_wide.structural_complexity_index

    # Sealed Meta-Test
    m_chain = create_meta_long_chain_aerospace(seed=100)
    m_pyr = create_meta_branching_pyramid_cloud(seed=101)
    m_lat = create_meta_sparse_lattice_logistics(seed=104)
    c_m_chain = compute_observable_complexity(m_chain)
    c_m_pyr = compute_observable_complexity(m_pyr)
    c_m_lat = compute_observable_complexity(m_lat)
    assert c_m_chain.num_nodes == 10
    assert c_m_pyr.num_nodes == 10
    assert c_m_chain.graph_diameter == 9
    assert c_m_pyr.graph_diameter == 3
    assert c_m_lat.graph_diameter == 4


def test_budget_controllers_and_decisions():
    """Verify actions across Fixed, M58, ProgressCost, and Experiential controllers."""
    # 1. Fixed Controller
    fc = FixedBudgetController(target_steps=20)
    d_fc_1 = fc.evaluate_budget_action(0, [1.0])
    assert d_fc_1.action == BudgetAction.TRAIN
    assert d_fc_1.block_steps_allocated == 10
    d_fc_2 = fc.evaluate_budget_action(20, [0.5])
    assert d_fc_2.action == BudgetAction.KEEP
    assert d_fc_2.block_steps_allocated == 0

    # 2. M58 Controller
    m58_c = M58AdaptiveBudgetController()
    d_train = m58_c.evaluate_budget_action(10, [0.90, 0.70])
    assert d_train.action == BudgetAction.TRAIN
    d_reset = m58_c.evaluate_budget_action(10, [1.25, 1.24])
    assert d_reset.action == BudgetAction.RESET
    d_keep = m58_c.evaluate_budget_action(20, [0.40, 0.38])
    assert d_keep.action == BudgetAction.KEEP

    # 3. ProgressCostController
    pcc = ValidationProgressCostController(min_marginal_return=0.005)
    # Stalled progress: 0.60 -> 0.58 (drop = 0.02 / 10 = 0.002 < 0.005)
    d_pcc_stop = pcc.evaluate_budget_action(20, [0.60, 0.58])
    assert d_pcc_stop.action == BudgetAction.KEEP
    assert "below economic cost" in d_pcc_stop.rationale

    # 4. ExperientialController
    store = ExperientialMemoryStore()
    store.populate_seed_experiences()
    exp_c = ExperientialBudgetController(memory_store=store)
    deep_env = create_dev_deep_chain_environment(seed=42)
    c_deep = compute_observable_complexity(deep_env)
    d_exp = exp_c.evaluate_budget_action(0, [0.75], complexity=c_deep)
    assert d_exp.action == BudgetAction.TRAIN
    assert exp_c.recommendation is not None
    assert exp_c.recommendation.recommended_budget_steps == 40


def test_experiential_memory_store_retrieval():
    """Verify retrieval of past learning episodes based on topology and validation loss."""
    store = ExperientialMemoryStore()
    store.populate_seed_experiences()
    assert len(store.records) == 3

    deep_env = create_meta_long_chain_aerospace(seed=100)
    c_deep = compute_observable_complexity(deep_env)

    # Retrieval for deep chain: matches deep chain experience
    rec_deep = store.query_recommendation(c_deep, sample_loss=0.70, sample_acc=0.65)
    assert rec_deep.recommended_action == "TRAIN"
    assert rec_deep.recommended_budget_steps == 40
    assert rec_deep.negative_transfer_warning is False

    # Retrieval for incompatible parity: high loss triggers negative transfer warning and RESET
    parity_env = create_meta_cyclic_parity_treaty(seed=102)
    c_par = compute_observable_complexity(parity_env)
    rec_par = store.query_recommendation(c_par, sample_loss=1.25, sample_acc=0.33)
    assert rec_par.recommended_action == "RESET"
    assert rec_par.negative_transfer_warning is True


def test_evidence_based_skill_router_joint_performance(base_model_and_tokenizer):
    """Verify evidence-based skill router and joint source/target evaluation."""
    model, tokenizer, device = base_model_and_tokenizer

    source_adapter = PersistentCandidateAdapter(d_model=128, bottleneck_dim=16).to(device)
    composite_adapter = ResidualSkillAdapter(source_adapter=source_adapter, d_model=128, residual_bottleneck_dim=16).to(device)
    source_adapter.eval()
    composite_adapter.eval()

    router = EvidenceBasedSkillRouter(
        source_adapter=source_adapter,
        composite_adapter=composite_adapter,
        d_model=128,
    ).to(device)

    env_src = create_dev_balanced_hierarchy_environment(seed=42)
    env_tgt = create_dev_deep_chain_environment(seed=43)

    # Centroid calibration
    dummy_src = [torch.randn(1, 10, 128, device=device) for _ in range(3)]
    dummy_tgt = [torch.randn(1, 10, 128, device=device) for _ in range(3)]
    router.calibrate_centroids(dummy_src, dummy_tgt)
    assert router.source_centroid is not None
    assert router.target_centroid is not None

    # Train minimal gate
    training_data = [
        (torch.randn(1, 10, 128, device=device), RouteDestination.SOURCE_SKILL),
        (torch.randn(1, 10, 128, device=device), RouteDestination.COMPOSITE_SKILL),
    ]
    router.train_minimal_gate(training_data, steps=5)

    # Evaluate all routing modes
    for mode in [
        RouterMode.ALWAYS_SOURCE,
        RouterMode.ALWAYS_COMPOSITE,
        RouterMode.M58_CONFIDENCE_ENTROPY,
        RouterMode.DETERMINISTIC_VERIFIED_EVIDENCE,
        RouterMode.MINIMAL_TRAINABLE_ROUTER,
        RouterMode.ORACLE,
    ]:
        rep = router.evaluate_routed_benchmark(
            base_model=model,
            tokenizer=tokenizer,
            device=device,
            source_env=env_src,
            target_env=env_tgt,
            mode=mode,
        )
        assert rep.joint_source_target_f1 == pytest.approx((rep.source_routed_macro_f1 + rep.target_routed_macro_f1) / 2.0, abs=1e-5)
        assert rep.routing_accuracy >= 0.0
        assert rep.mean_routing_latency_ms >= 0.0


def test_contrastive_representation_loss():
    """Verify Phase 5 contrastive representation loss between skill representations."""
    z_src = torch.randn(4, 128)
    z_tgt = torch.randn(4, 128)
    loss = compute_contrastive_representation_loss(z_src, z_tgt, margin=0.5)
    assert loss.dim() == 0
    assert float(loss.item()) >= 0.0


def test_corrected_sprt_epistemic_audit():
    """Verify corrected SPRT with transitive ground truth eliminates false alarms."""
    auditor = CorrectedEpistemicAuditor(budget=8, seed=42)

    # Compare corrected oracle vs flawed oracle on known environment at 0% noise
    diag_corr = auditor.evaluate_regime(0.0, "graph_contrastive_targeted", episodes_per_world=5, use_corrected_oracle=True)
    diag_flawed = auditor.evaluate_regime(0.0, "graph_contrastive_targeted", episodes_per_world=5, use_corrected_oracle=False)

    # Flawed oracle exhibits 100% false alarms on known world
    assert diag_flawed.known_world_false_alarm_rate == 1.0

    # Corrected oracle resolves this: false alarm rate is dramatically reduced
    assert diag_corr.known_world_false_alarm_rate < 0.20
    assert diag_corr.detection_macro_f1 > diag_flawed.detection_macro_f1


def test_routed_retention_and_rollback_safeguards(base_model_and_tokenizer):
    """Verify retention of frozen base parameters and reversible rollback."""
    model, tokenizer, device = base_model_and_tokenizer
    auditor = RoutedRetentionAuditor(model, tokenizer, device)

    source_adapter = PersistentCandidateAdapter(d_model=128, bottleneck_dim=16).to(device)
    composite_adapter = ResidualSkillAdapter(source_adapter=source_adapter, d_model=128, residual_bottleneck_dim=16).to(device)

    router = EvidenceBasedSkillRouter(
        source_adapter=source_adapter,
        composite_adapter=composite_adapter,
        d_model=128,
    ).to(device)

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
    assert "PASS" in result.verdict
