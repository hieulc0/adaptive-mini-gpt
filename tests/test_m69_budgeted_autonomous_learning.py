"""Unit and regression tests for M69: Budgeted Autonomous Learning and Verified Continual Improvement."""

import copy
import hashlib
from typing import Any
import torch
import torch.nn as nn
import pytest

from adaptive_mini_gpt.model import MiniGPT, ModelConfig
from adaptive_mini_gpt.modular_skill_learning.repertoire import (
    ModularSkillRepertoire,
    Rank16ResidualAdapter,
)
from adaptive_mini_gpt.data.bpe_scaling import train_scaled_bpe_tokenizer
from adaptive_mini_gpt.utility_calibrated_selection.task_environments import (
    create_known_reusable_dag_env,
    create_known_incomplete_dag_env,
    create_rail_stage1_train_env,
    create_rail_stage2_qual_env,
    create_rail_stage3_eval_env,
)
from adaptive_mini_gpt.budgeted_autonomous_learning import (
    M68ScientificAuditor,
    FairBudgetLedger,
    BUDGET_REGIMES,
    BudgetTier,
    BudgetConstraint,
    SequentialTaskStreamEngine,
    CausalWeightImprovementAuditor,
    FairHistoryLearningTracker,
    HistoryConditioningMode,
    VerifiedExperimentRecord,
    M69SafeguardsAuditor,
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


def test_m68_scientific_audit_and_checkpoint_lineage():
    """Verify Phase 1 audit reconciles M68 empirical numbers and parameter hash formats."""
    report = M68ScientificAuditor.run_full_m68_scientific_audit()
    assert report.audit_passed is True
    assert report.always_fresh_utility_delta < 0.0  # M68 loop lost to ALWAYS_FRESH by ~0.0100
    assert report.history_conditioned_utility_delta < 0.0  # M68 history degraded utility by ~0.0286
    assert report.integrated_loop_retained_adapters == 0  # 0 adapters retained in M68 benchmark
    assert report.seed_42_causal_gain_over_fresh > 0.0  # Positive on Seed 42
    assert report.seed_43_causal_gain_over_fresh < 0.0  # Failed on Seed 43
    assert report.seed_44_causal_gain_over_fresh < 0.0  # Failed on Seed 44
    assert report.checkpoint_lineage_reconciled is True
    assert "ff27a546" in report.m67_hash_formula_explanation
    assert "8736b364" in report.m68_hash_formula_explanation


def test_fair_budget_regimes_and_pareto_ledger():
    """Verify Phase 2 fair budget constraints, Pareto extraction, and sensitivity."""
    assert BudgetTier.SMALL in BUDGET_REGIMES
    assert BudgetTier.MEDIUM in BUDGET_REGIMES
    assert BudgetTier.LARGE in BUDGET_REGIMES

    small_b = BUDGET_REGIMES[BudgetTier.SMALL]
    assert small_b.max_queries_per_task == 12
    assert small_b.max_steps_per_task == 20

    med_b = BUDGET_REGIMES[BudgetTier.MEDIUM]
    assert med_b.max_queries_per_task == 20
    assert med_b.max_steps_per_task == 40

    large_b = BUDGET_REGIMES[BudgetTier.LARGE]
    assert large_b.max_queries_per_task == 32
    assert large_b.max_steps_per_task == 60

    ledger = FairBudgetLedger(expected_life_cycle_episodes=3)
    out1 = ledger.compute_outcome(
        policy_name="ALWAYS_FRESH",
        budget_tier=BudgetTier.SMALL,
        task_name="task_1",
        macro_f1=0.45,
        accuracy=0.50,
        queries_used=12,
        steps_used=20,
        stored_params=0,
        active_params=2371,
        gpu_seconds=0.10,
        false_allocations=0,
        missed_reuses=0,
    )
    assert out1.single_episode_utility > 0.0
    assert "compute_heavy" in out1.sensitivity_utilities
    assert "quality_heavy" in out1.sensitivity_utilities

    out2 = ledger.compute_outcome(
        policy_name="ALWAYS_REUSE",
        budget_tier=BudgetTier.SMALL,
        task_name="task_1",
        macro_f1=0.30,
        accuracy=0.35,
        queries_used=4,
        steps_used=0,
        stored_params=0,
        active_params=0,
        gpu_seconds=0.02,
        false_allocations=0,
        missed_reuses=0,
    )

    frontier = FairBudgetLedger.extract_pareto_frontier([out1, out2])
    assert len(frontier) >= 1


def test_sequential_stream_persistent_accumulation():
    """Verify Phase 3 task stream persistently accumulates and carries retained skills forward."""
    model, _ = _create_mock_model_and_tokenizer(seed=42)
    device = torch.device("cpu")
    repertoire = ModularSkillRepertoire(d_model=128, bottleneck_dim=16, num_classes=3)

    # Initial skill
    dummy_ad = Rank16ResidualAdapter(d_model=128, bottleneck_dim=16, num_classes=3)
    repertoire.register_skill("source_dag", "TRANSITIVE_DAG", "env_0", dummy_ad, 0.40, 0.40, {})

    ledger = FairBudgetLedger(expected_life_cycle_episodes=3)
    stream_engine = SequentialTaskStreamEngine(cost_ledger=ledger)

    task_seq = [
        {
            "name": "ep1_novel_rail",
            "family": "PARITY_CYCLE",
            "is_known": False,
            "train_env": create_rail_stage1_train_env(),
            "qual_env": create_rail_stage2_qual_env(),
            "eval_env": create_rail_stage3_eval_env(),
        },
        {
            "name": "ep2_transfer_rail",
            "family": "PARITY_CYCLE",
            "is_known": False,
            "train_env": create_rail_stage1_train_env(),
            "qual_env": create_rail_stage2_qual_env(),
            "eval_env": create_rail_stage3_eval_env(),
        },
    ]

    res = stream_engine.run_sequential_stream(
        stream_id="test_stream",
        seed=42,
        model=model,
        initial_repertoire=repertoire,
        task_sequence=task_seq,
        sealed_eval_env=create_rail_stage3_eval_env(),
        budget=BUDGET_REGIMES[BudgetTier.SMALL],
        arm_name="RETAINED_LEARNING",
        device=device,
    )

    assert len(res.episodes) == 2
    assert res.final_repertoire_skill_count >= 1
    assert res.sealed_eval_macro_f1 >= 0.0


def test_causal_weight_improvement_six_arms():
    """Verify Phase 4 six-arm causal ablation comparison interface."""
    model, _ = _create_mock_model_and_tokenizer(seed=42)
    device = torch.device("cpu")
    repertoire = ModularSkillRepertoire(d_model=128, bottleneck_dim=16, num_classes=3)
    dummy_ad = Rank16ResidualAdapter(d_model=128, bottleneck_dim=16, num_classes=3)
    repertoire.register_skill("source_dag", "TRANSITIVE_DAG", "env_0", dummy_ad, 0.40, 0.40, {})

    ledger = FairBudgetLedger(expected_life_cycle_episodes=3)
    stream_engine = SequentialTaskStreamEngine(cost_ledger=ledger)

    task_seq = [
        {
            "name": "ep1_known",
            "family": "TRANSITIVE_DAG",
            "is_known": True,
            "train_env": create_known_reusable_dag_env(),
            "qual_env": create_known_reusable_dag_env(),
            "eval_env": create_known_reusable_dag_env(),
        }
    ]

    report = CausalWeightImprovementAuditor.run_six_arm_causal_ablation(
        stream_engine=stream_engine,
        model=model,
        initial_repertoire=repertoire,
        task_sequence=task_seq,
        sealed_eval_env=create_known_reusable_dag_env(),
        budget=BUDGET_REGIMES[BudgetTier.SMALL],
        seed=42,
        device=device,
    )

    assert report.arm_a_retained_result is not None
    assert report.arm_b_reset_result is not None
    assert report.arm_c_fresh_result is not None
    assert report.arm_d_irrelevant_result is not None
    assert report.arm_e_shuffled_result is not None
    assert report.arm_f_unchanged_result is not None
    assert "RESET_WEIGHTS" in report.comparisons
    assert "SHUFFLED" in report.comparisons


def test_fair_history_learning_tracker_modes():
    """Verify Phase 5 history tracker computes bounded adjustments across all 5 modes."""
    tracker = FairHistoryLearningTracker()

    rec = VerifiedExperimentRecord(
        record_id="rec_1",
        task_name="rail_grid",
        rule_family="PARITY_CYCLE",
        pre_decision_density=0.25,
        pre_decision_probe_f1=0.20,
        sprt_llr=3.5,
        action_taken="TRAIN_NEW_SKILL",
        measured_macro_f1=0.55,
        measured_utility=0.32,
        validation_passed=True,
    )
    tracker.log_verified_experiment(rec)

    for mode in [
        HistoryConditioningMode.NO_HISTORY,
        HistoryConditioningMode.RELEVANT,
        HistoryConditioningMode.SHUFFLED,
        HistoryConditioningMode.IRRELEVANT,
        HistoryConditioningMode.CONSERVATIVE,
    ]:
        bonus = tracker.compute_prior_utility_bonus(
            mode=mode,
            candidate_action="TRAIN_NEW_SKILL",
            current_density=0.25,
            current_probe_f1=0.20,
            current_sprt_llr=3.5,
        )
        assert isinstance(bonus, float)
        if mode == HistoryConditioningMode.NO_HISTORY:
            assert bonus == 0.0


def test_m69_retention_safeguards_and_hash_audit():
    """Verify Phase 8 & 9 retention safeguards, double hash invariance, and bit-exact rollback."""
    model, _ = _create_mock_model_and_tokenizer(seed=42)
    device = torch.device("cpu")
    repertoire = ModularSkillRepertoire(d_model=128, bottleneck_dim=16, num_classes=3)
    dummy_ad = Rank16ResidualAdapter(d_model=128, bottleneck_dim=16, num_classes=3)
    repertoire.register_skill("source_dag", "TRANSITIVE_DAG", "env_0", dummy_ad, 0.40, 0.40, {})

    h_unnamed = M69SafeguardsAuditor.compute_unnamed_model_hash(model)
    h_named = M69SafeguardsAuditor.compute_sorted_named_model_hash(model)
    rep_hashes = M69SafeguardsAuditor.compute_repertoire_hashes(repertoire)

    report = M69SafeguardsAuditor.run_safeguards_audit(
        model=model,
        initial_unnamed_hash=h_unnamed,
        initial_sorted_named_hash=h_named,
        repertoire=repertoire,
        initial_repertoire_hashes=rep_hashes,
        device=device,
    )

    assert report.base_parameter_hash_invariant is True
    assert report.repertoire_checksums_invariant is True
    assert report.arithmetic_carry_retention == 1.0
    assert report.chained_logic_retention == 1.0
    assert report.expanded_logic_prompts_tested == 8
    assert report.rollback_verified is True
    assert report.rank_16_compactness_verified is True
    assert report.all_safeguards_passed is True
