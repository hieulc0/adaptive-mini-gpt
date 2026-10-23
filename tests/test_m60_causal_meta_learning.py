"""Comprehensive Unit Test Suite for Milestone M60: Causal Meta-Learning & Functional Skill Routing."""
from __future__ import annotations

import pytest
import torch

from adaptive_mini_gpt.causal_meta_learning.audit import (
    M59AuditReport,
    M59ScientificAuditor,
)
from adaptive_mini_gpt.causal_meta_learning.causal_ablation import (
    CausalAblationAdaptationEngine,
    CausalAblationArm,
    CausalAdaptationResult,
    build_ablated_memory_store,
)
from adaptive_mini_gpt.causal_meta_learning.epistemic_validation import (
    IndependentEpistemicValidator,
)
from adaptive_mini_gpt.causal_meta_learning.functional_routing import (
    FunctionalSkillRouter,
    MinimalNeuralGate,
)
from adaptive_mini_gpt.causal_meta_learning.retention_safeguards import (
    CausalRetentionAuditor,
)
from adaptive_mini_gpt.causal_meta_learning.task_suite import (
    build_development_task_suite,
    build_meta_training_task_suite,
    build_sealed_meta_test_task_suite,
    compute_independent_graph_reachability,
    compute_observable_complexity,
    create_dev_balanced_hierarchy_environment,
    create_dev_deep_chain_environment,
    create_dev_incompatible_parity_environment,
    create_meta_long_chain_aerospace,
)
from adaptive_mini_gpt.generalizable_learning.skill_router import RouterMode
from adaptive_mini_gpt.model import GPTConfig, MiniGPT
from adaptive_mini_gpt.skill_reuse.adapter import ResidualSkillAdapter


class DummyTokenizer:
    """Lightweight dummy tokenizer for testing."""
    def encode(self, text: str) -> list[int]:
        return [ord(c) % 256 for c in text[:16]] or [1]

    def decode(self, tokens: list[int]) -> str:
        return "".join(chr(t) for t in tokens)


@pytest.fixture
def tiny_model() -> MiniGPT:
    cfg = GPTConfig(
        vocab_size=256,
        block_size=32,
        n_layer=2,
        n_head=2,
        n_embd=32,
        dropout=0.0,
    )
    return MiniGPT(cfg)


@pytest.fixture
def tokenizer() -> DummyTokenizer:
    return DummyTokenizer()


def test_m59_scientific_audit() -> None:
    """Test 1: Verify forensic audit reconciles M59 claims, efficiency, routing, and accounting."""
    auditor = M59ScientificAuditor()
    rep = auditor.run_m59_scientific_audit()

    assert isinstance(rep, M59AuditReport)
    assert rep.sealed_fixed_40_auc == 0.3155
    assert rep.sealed_experiential_auc == 0.3095
    assert rep.experiential_vs_fixed_40_auc_diff == pytest.approx(-0.0060, abs=1e-4)
    assert rep.experiential_vs_m58_auc_diff == pytest.approx(0.0238, abs=1e-4)
    assert rep.step_savings_pct_experiential == pytest.approx(31.25, abs=1e-2)
    assert rep.efficiency_advantage_pct_experiential > 40.0

    # Task accounting check
    assert rep.total_defined_environments == 13
    assert rep.evaluated_target_environments == 9
    assert rep.total_adaptation_episodes_evaluated == 162

    # Routing invariance explanation check
    assert rep.always_composite_joint_f1 == pytest.approx(0.645833, abs=1e-4)
    assert rep.learned_router_joint_f1 == pytest.approx(0.645833, abs=1e-4)
    assert rep.measured_source_interference_drop == pytest.approx(0.0, abs=1e-4)
    assert "EVALUATION ARTIFACT" in rep.routing_identical_f1_root_cause


def test_independent_reachability_engine() -> None:
    """Test 2: Verify independent transitive closure reachability engine."""
    nodes = ["A", "B", "C", "D"]
    edges = [("A", "B"), ("B", "C"), ("C", "D")]
    reach = compute_independent_graph_reachability(nodes, edges)

    assert reach[("A", "D")] is True
    assert reach[("A", "C")] is True
    assert reach[("B", "D")] is True
    assert reach[("D", "A")] is False
    assert reach[("C", "B")] is False


def test_task_suite_multi_split_separation() -> None:
    """Test 3: Verify strict segregation of task environments across splits."""
    train_suite = build_meta_training_task_suite()
    dev_suite = build_development_task_suite()
    sealed_suite = build_sealed_meta_test_task_suite()

    assert len(train_suite) == 3
    assert len(dev_suite) == 5
    assert len(sealed_suite) == 5

    train_ids = {e.env_id for e in train_suite}
    dev_ids = {e.env_id for e in dev_suite}
    sealed_ids = {e.env_id for e in sealed_suite}

    assert train_ids.isdisjoint(dev_ids)
    assert dev_ids.isdisjoint(sealed_ids)
    assert train_ids.isdisjoint(sealed_ids)

    # Complexity extraction check
    for env in train_suite + dev_suite + sealed_suite:
        comp = compute_observable_complexity(env)
        assert comp.num_nodes > 0
        assert comp.graph_diameter >= 0
        assert comp.structural_complexity_index > 0.0


def test_causal_memory_ablation_stores() -> None:
    """Test 4: Verify construction of the 7 ablation memory stores."""
    # Cold start arms
    store_a = build_ablated_memory_store(CausalAblationArm.ARM_A_NO_HISTORY)
    assert len(store_a.records) == 0

    store_f = build_ablated_memory_store(CausalAblationArm.ARM_F_FIXED_40_STEPS)
    assert len(store_f.records) == 0

    # Relevant history
    store_b = build_ablated_memory_store(CausalAblationArm.ARM_B_RELEVANT_HISTORY)
    assert len(store_b.records) == 3
    assert all(not ep.was_negative_transfer for ep in store_b.records)

    # Random history
    store_c = build_ablated_memory_store(CausalAblationArm.ARM_C_RANDOM_UNRELATED_HISTORY)
    assert len(store_c.records) == 4

    # Shuffled history
    store_d = build_ablated_memory_store(CausalAblationArm.ARM_D_SHUFFLED_OUTCOMES_HISTORY)
    assert len(store_d.records) == 3
    assert all(ep.was_negative_transfer for ep in store_d.records)

    # Incompatible history
    store_e = build_ablated_memory_store(CausalAblationArm.ARM_E_INCOMPATIBLE_HISTORY)
    assert len(store_e.records) == 3
    assert all(ep.recommended_action == "RESET" for ep in store_e.records)


def test_functional_skill_router_divergence(tiny_model: MiniGPT, tokenizer: DummyTokenizer) -> None:
    """Test 5: Verify FunctionalSkillRouter evaluates divergence and computes behavioral gain."""
    device = torch.device("cpu")
    d_model = tiny_model.config.n_embd
    src_adapter = ResidualSkillAdapter(d_model=d_model, bottleneck_dim=8, num_classes=2)
    comp_adapter = ResidualSkillAdapter(d_model=d_model, bottleneck_dim=8, num_classes=2)

    router = FunctionalSkillRouter(
        d_model=d_model, source_adapter=src_adapter, composite_adapter=comp_adapter
    )

    src_env = create_dev_balanced_hierarchy_environment(seed=42)
    tgt_env = create_meta_long_chain_aerospace(seed=100)

    rep = router.evaluate_functional_benchmark(
        base_model=tiny_model,
        tokenizer=tokenizer,
        device=device,
        source_env=src_env,
        target_env=tgt_env,
        mode=RouterMode.ALWAYS_COMPOSITE,
    )

    assert rep.total_queries > 0
    assert 0.0 <= rep.divergence_rate <= 1.0
    assert 0.0 <= rep.route_dependence_rate <= 1.0
    assert 0.0 <= rep.joint_source_target_f1 <= 1.0
    assert rep.mean_latency_ms > 0.0


def test_independent_epistemic_validator_zero_false_alarms() -> None:
    """Test 6: Verify SPRT with independent reachability achieves zero false alarms at 0% noise."""
    validator = IndependentEpistemicValidator(budget=8, seed=42)
    rec = validator.evaluate_regime(noise_level=0.0, policy="graph_contrastive_targeted", episodes_per_world=2)

    assert rec.episodes_evaluated > 0
    assert rec.known_world_false_alarm_rate == 0.0
    assert rec.unknown_rule_recall >= 0.90
    assert rec.detection_macro_f1 >= 0.90
    assert rec.epistemic_pass is True


def test_retention_and_rollback_safeguards(tiny_model: MiniGPT, tokenizer: DummyTokenizer) -> None:
    """Test 7: Verify parameter hash invariance and automated rollback."""
    device = torch.device("cpu")
    auditor = CausalRetentionAuditor(base_model=tiny_model, tokenizer=tokenizer, device=device)
    src_env = create_dev_balanced_hierarchy_environment(seed=42)
    src_adapter = ResidualSkillAdapter(d_model=tiny_model.config.n_embd, bottleneck_dim=8, num_classes=2)

    # Clean case
    rep_clean = auditor.audit_retention_and_rollback(
        source_env=src_env, initial_source_adapter=src_adapter, routed_source_f1=0.85
    )
    assert rep_clean.base_parameter_hash_invariant is True
    assert rep_clean.rollback_triggered is False
    assert rep_clean.retention_pass is True

    # Drop exceeding tolerance -> triggers rollback
    rep_drop = auditor.audit_retention_and_rollback(
        source_env=src_env, initial_source_adapter=src_adapter, routed_source_f1=0.20, tolerance=0.15
    )
    assert rep_drop.rollback_triggered is True
    assert rep_drop.rollback_successful is True


def test_causal_ablation_adaptation_engine(tiny_model: MiniGPT, tokenizer: DummyTokenizer) -> None:
    """Test 8: Verify CausalAblationAdaptationEngine runs optimization and returns valid AUC."""
    device = torch.device("cpu")
    engine = CausalAblationAdaptationEngine(
        base_model=tiny_model, tokenizer=tokenizer, device=device
    )
    env = create_dev_deep_chain_environment(seed=42)

    res_a = engine.run_causal_adaptation(env=env, arm=CausalAblationArm.ARM_A_NO_HISTORY, seed=42)
    assert isinstance(res_a, CausalAdaptationResult)
    assert res_a.total_steps_taken > 0
    assert 0.0 <= res_a.learning_curve_auc <= 1.0
    assert res_a.wall_clock_time_ms > 0.0
    assert res_a.compute_adjusted_efficiency > 0.0

    res_b = engine.run_causal_adaptation(env=env, arm=CausalAblationArm.ARM_B_RELEVANT_HISTORY, seed=42)
    assert res_b.total_steps_taken > 0
    assert 0.0 <= res_b.learning_curve_auc <= 1.0
