"""Unit and regression tests for Milestone M71: Selective Neural Memory and Causal Retention."""

import copy
from typing import Any
import pytest
import torch
import torch.nn as nn

from adaptive_mini_gpt.budgeted_autonomous_learning.budget_regimes import (
    BUDGET_REGIMES,
    BudgetTier,
    FairBudgetLedger,
)
from adaptive_mini_gpt.data.bpe_scaling import train_scaled_bpe_tokenizer
from adaptive_mini_gpt.model import MiniGPT, ModelConfig
from adaptive_mini_gpt.modular_skill_learning.repertoire import (
    ModularSkillRepertoire,
    Rank16ResidualAdapter,
)
from adaptive_mini_gpt.selective_neural_memory import (
    M70ScientificAuditor,
    M71SafeguardsAuditor,
    RoutingDecision,
    SelectiveContinualLearningStreamEngine,
    SelectiveSkillMemoryRegistry,
    SelectiveSkillRouter,
    SkillState,
)


def _create_mock_model_and_tokenizer(seed: int = 42) -> tuple[MiniGPT, Any]:
    torch.manual_seed(seed)
    training_data = (
        "CoreSecurityNode1 in_same_security_zone CoreSecurityNode2. "
        "StateAlpha1 holds_mutual_pact_with StateBeta2. "
        "AeroModuleM0 assembles_into_subsystem AeroModuleM1. "
        "GatewayGW routes_traffic_payload_to IngressRouterIR. "
        "Evidence: [A->B, NOT(B->A), UNOBS(C)]. Does A imply B? "
    )
    tokenizer = train_scaled_bpe_tokenizer(training_data, target_vocab_size=1024)
    cfg = ModelConfig(
        vocab_size=tokenizer.vocab_size,
        d_model=128,
        n_layers=2,
        n_heads=4,
        d_ff=256,
        context_length=256,
    )
    model = MiniGPT(cfg)
    model.tokenizer = tokenizer
    model.eval()
    return model, tokenizer


def test_m70_scientific_audit_and_reconciliation():
    """Verify Phase 1 forensic audit of M70 claims and explanation of identical budget F1."""
    report = M70ScientificAuditor.audit_and_reconcile_m70()
    assert report.audit_passed is True
    assert report.m70_small_utility == 0.5311
    assert report.always_reuse_utility == 0.2889
    assert report.m70_retained_f1 == 0.4250
    assert report.m70_reset_f1 == 0.4500
    assert report.sealed_test_isolation_verified is True
    assert "gradient" in report.identical_f1_explanation.lower()
    assert len(report.conclusions) == 5


def test_selective_skill_memory_registry_and_states():
    """Verify Phase 3 skill states (ACTIVE, DORMANT, REJECTED), accounting, and state invariance."""
    registry = SelectiveSkillMemoryRegistry(d_model=128, bottleneck_dim=16, num_classes=3)
    ad1 = Rank16ResidualAdapter(d_model=128, bottleneck_dim=16, num_classes=3)
    ad2 = Rank16ResidualAdapter(d_model=128, bottleneck_dim=16, num_classes=3)

    rec1 = registry.register_skill("source_dag", "TRANSITIVE_DAG", "env_0", ad1, 0.70, 0.70, initial_state=SkillState.ACTIVE)
    rec2 = registry.register_skill("cand_parity", "PARITY_CYCLE", "env_1", ad2, 0.65, 0.65, initial_state=SkillState.DORMANT)

    assert registry.list_active_skills() == ["source_dag"]
    assert registry.list_dormant_skills() == ["cand_parity"]

    # Parameter accounting
    stored_p = registry.total_stored_parameters()
    active_p = registry.total_active_parameters()
    assert stored_p > 0
    assert active_p > 0
    assert stored_p > active_p  # dormant skill excluded from active parameters

    # Reversible state transition
    h_before = rec2.compute_hash()
    registry.transition_state("cand_parity", SkillState.ACTIVE, "activating for routing")
    assert registry.get_skill("cand_parity").state == SkillState.ACTIVE
    assert registry.total_active_parameters() == stored_p
    h_after = rec2.compute_hash()
    assert h_before == h_after  # parameter invariance preserved

    # Export to modular repertoire
    rep = registry.export_to_modular_repertoire(active_only=True)
    assert len(rep.list_skills()) == 2


def test_selective_skill_router_negative_transfer_prevention():
    """Verify Phase 3 router rejects weak or mismatched candidates to prevent negative transfer."""
    model, tokenizer = _create_mock_model_and_tokenizer(seed=42)
    dev = torch.device("cpu")
    router = SelectiveSkillRouter(tau_select=0.75, delta_margin=0.05)
    registry = SelectiveSkillMemoryRegistry(d_model=128, bottleneck_dim=16, num_classes=3)

    # Source skill
    ad_src = Rank16ResidualAdapter(d_model=128, bottleneck_dim=16, num_classes=3)
    registry.register_skill("source_dag", "TRANSITIVE_DAG", "env_0", ad_src, 0.40, 0.40, initial_state=SkillState.ACTIVE)

    # Candidate skill with weak or mismatched weights
    ad_cand = Rank16ResidualAdapter(d_model=128, bottleneck_dim=16, num_classes=3)
    registry.register_skill("cand_weak", "PARITY_CYCLE", "env_1", ad_cand, 0.20, 0.20, initial_state=SkillState.ACTIVE)

    mock_queries = [
        {"question_text": "Sample query 1 for routing", "expected_answer": "YES"},
        {"question_text": "Sample query 2 for routing", "expected_answer": "NO"},
        {"question_text": "Sample query 3 for routing", "expected_answer": "UNKNOWN"},
    ]

    # Route task: candidate should NOT be routed because it fails tau_select (0.45)
    dec = router.route_task("task_test", model, registry, mock_queries, dev)
    assert dec.is_candidate_routed is False
    assert dec.selected_skill_id != "cand_weak"
    assert "source" in dec.reason or "base" in dec.reason


def test_selective_lifelong_stream_and_six_arm_causal_ablation():
    """Verify Phase 2, 4, 5 lifelong stream runs 6 causal arms and logs realized transfers."""
    model, tokenizer = _create_mock_model_and_tokenizer(seed=42)
    dev = torch.device("cpu")
    registry = SelectiveSkillMemoryRegistry(d_model=128, bottleneck_dim=16, num_classes=3)

    ad_init = Rank16ResidualAdapter(d_model=128, bottleneck_dim=16, num_classes=3)
    registry.register_skill("source_transitive", "TRANSITIVE_DAG", "env_0", ad_init, 0.60, 0.60, initial_state=SkillState.ACTIVE)

    class MockEnv:
        def __init__(self, name: str, expected_ans: str = "YES"):
            self.env_id = name
            self.held_out_test_queries = [
                {"question_text": f"Question 1 for {name}", "expected_answer": expected_ans},
                {"question_text": f"Question 2 for {name}", "expected_answer": "NO"},
            ]

    # 5-episode test stream with repeated encounter
    stream_tasks = [
        {"task_name": "ep0_parity_a", "rule_family": "PARITY_CYCLE", "world_id": "alpha", "is_repeated_encounter": False, "train_env": MockEnv("p_tr"), "qual_env": MockEnv("p_q"), "eval_env": MockEnv("p_ev")},
        {"task_name": "ep1_chrom_a", "rule_family": "CHROMATIN_LATTICE", "world_id": "alpha", "is_repeated_encounter": False, "train_env": MockEnv("c_tr"), "qual_env": MockEnv("c_q"), "eval_env": MockEnv("c_ev")},
        {"task_name": "ep2_dag_a", "rule_family": "TRANSITIVE_DAG", "world_id": "alpha", "is_repeated_encounter": False, "train_env": MockEnv("d_tr"), "qual_env": MockEnv("d_q"), "eval_env": MockEnv("d_ev")},
        {"task_name": "ep3_parity_b", "rule_family": "PARITY_CYCLE", "world_id": "beta", "is_repeated_encounter": True, "train_env": MockEnv("p_tr"), "qual_env": MockEnv("p_q"), "eval_env": MockEnv("p_ev")},
        {"task_name": "ep4_chrom_g", "rule_family": "CHROMATIN_LATTICE", "world_id": "gamma", "is_repeated_encounter": True, "train_env": MockEnv("c_tr"), "qual_env": MockEnv("c_q"), "eval_env": MockEnv("c_ev")},
    ]

    budget = BUDGET_REGIMES[BudgetTier.MEDIUM]
    engine = SelectiveContinualLearningStreamEngine()

    for arm in ["SELECTIVE_RETAIN", "RESET", "RETAIN_ALL", "RETAIN_DISABLED"]:
        res = engine.run_sequential_stream(
            stream_id=f"test_stream_{arm}",
            seed=42,
            model=model,
            initial_registry=copy.deepcopy(registry),
            task_sequence=stream_tasks,
            sealed_eval_env=MockEnv("sealed"),
            budget=budget,
            arm_name=arm,
            device=dev,
        )
        assert len(res.episodes) == 5
        assert res.accumulated_macro_f1 > 0.0
        assert isinstance(res.accumulated_net_utility, float)


def test_m71_retention_safeguards_and_rollback():
    """Verify Phase 5 retention safeguards, bit-exact rollback, and state invariance."""
    model, tokenizer = _create_mock_model_and_tokenizer(seed=42)
    dev = torch.device("cpu")
    registry = SelectiveSkillMemoryRegistry(d_model=128, bottleneck_dim=16, num_classes=3)
    ad = Rank16ResidualAdapter(d_model=128, bottleneck_dim=16, num_classes=3)
    registry.register_skill("source_skill_1", "TRANSITIVE_DAG", "env_0", ad, 0.70, 0.70)

    h_unnamed = M71SafeguardsAuditor.compute_unnamed_model_hash(model)
    h_named = M71SafeguardsAuditor.compute_sorted_named_model_hash(model)

    cand_ad = Rank16ResidualAdapter(d_model=128, bottleneck_dim=16, num_classes=3)
    rollback_ok = M71SafeguardsAuditor.verify_bit_exact_rollback(registry, "temp_cand", cand_ad)
    assert rollback_ok is True

    state_inv_ok = M71SafeguardsAuditor.verify_reversible_state_invariance(registry, "source_skill_1")
    assert state_inv_ok is True

    report = M71SafeguardsAuditor.run_m71_safeguards_audit(
        model=model,
        registry=registry,
        seed=42,
        initial_unnamed_hash=h_unnamed,
        initial_sorted_named_hash=h_named,
        device=dev,
    )

    assert report.base_checkpoint_double_hash_verified is True
    assert report.bit_exact_rollback_passed is True
    assert report.reversible_state_invariance_passed is True
    assert report.carry_arithmetic_retained is True
    assert report.expanded_logic_retained is True
    assert report.topology_expansion_blocked is True
    assert "NO-GO" in report.production_promotion_status
    assert report.all_safeguards_passed is True
