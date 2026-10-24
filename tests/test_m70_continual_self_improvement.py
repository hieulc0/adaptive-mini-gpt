"""Unit and regression tests for M70 Continual Self-Improvement & Causal Verification."""

import copy
import hashlib
from typing import Any
import pytest
import torch
import torch.nn as nn

from adaptive_mini_gpt.data.bpe_scaling import train_scaled_bpe_tokenizer
from adaptive_mini_gpt.model import MiniGPT, ModelConfig
from adaptive_mini_gpt.modular_skill_learning.repertoire import (
    ModularSkillRepertoire,
    Rank16ResidualAdapter,
)
from adaptive_mini_gpt.budgeted_autonomous_learning.budget_regimes import (
    BUDGET_REGIMES,
    BudgetTier,
    FairBudgetLedger,
)
from adaptive_mini_gpt.continual_self_improvement import (
    M69ScientificCorrectionAuditor,
    M70SafeguardsAuditor,
    RationalActionType,
    RationalExperimentController,
    VerifiedContinualLearningStreamEngine,
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


def test_m69_scientific_audit_and_corrections():
    """Verify Phase 1 formally corrects all 6 M69 anomalies and sets revised verdicts."""
    report = M69ScientificCorrectionAuditor.audit_and_correct_m69()
    assert report.audit_passed is True
    assert len(report.corrections) == 6

    # 1. ALWAYS_REUSE beats AUTONOMOUS_LOOP across all 3 budgets in M69
    assert report.always_reuse_beats_autonomous_small is True
    assert report.always_reuse_beats_autonomous_medium is True
    assert report.always_reuse_beats_autonomous_large is True

    # 2. RESET_WEIGHTS beats RETAINED_LEARNING in M69
    assert report.reset_weights_beats_retained_f1 is True
    assert report.reset_weights_beats_retained_utility is True

    # 3. History conditioning gave 0.0000 benefit
    assert report.history_benefit_is_zero is True

    # 4. Revised verdicts
    assert "FAIL" in report.revised_policy_verdict
    assert "FAIL" in report.revised_causal_verdict
    assert "FAIL" in report.revised_history_verdict
    assert report.revised_self_improvement_verdict == "NOT ESTABLISHED"
    assert "NO-GO" in report.production_verdict


def test_rational_experiment_controller():
    """Verify Phase 3 rational controller avoids wasted compute via ENVE and proficiency gating."""
    controller = RationalExperimentController(proficient_f1_threshold=0.40)
    small_budget = BUDGET_REGIMES[BudgetTier.SMALL]
    med_budget = BUDGET_REGIMES[BudgetTier.MEDIUM]

    # Case 1: Existing skill is already proficient (F1 = 0.55 >= 0.40)
    # Controller must choose REUSE with 0 steps
    res_reuse = controller.evaluate_action_for_task(
        task_name="test_known_task",
        best_skill_id="skill_known",
        best_skill_probe_f1=0.55,
        budget=med_budget,
    )
    assert res_reuse.action == RationalActionType.REUSE_QUALIFIED_SKILL
    assert res_reuse.selected_skill_id == "skill_known"
    assert res_reuse.estimated_experiment_cost == 0.0
    assert "proficient" in res_reuse.reason

    # Case 2: Novel task with low probe F1 (0.22 < 0.40) under MEDIUM budget
    # Controller estimates positive ENVE and selects EXPERIMENT_AND_ADAPT
    res_exp = controller.evaluate_action_for_task(
        task_name="test_novel_task",
        best_skill_id="skill_weak",
        best_skill_probe_f1=0.22,
        budget=med_budget,
    )
    assert res_exp.action == RationalActionType.EXPERIMENT_AND_ADAPT
    assert res_exp.expected_net_value > 0.0
    assert "positive" in res_exp.reason

    # Case 3: Candidate acceptance gate
    accepted, gain, reason = controller.decide_candidate_acceptance(
        candidate_id="cand_1",
        held_out_val_f1=0.62,
        baseline_f1=0.22,
    )
    assert accepted is True
    assert gain == pytest.approx(0.40, rel=1e-3)

    # Case 4: Candidate rejection and rollback when gain <= min_held_out_gain
    rejected, bad_gain, rej_reason = controller.decide_candidate_acceptance(
        candidate_id="cand_bad",
        held_out_val_f1=0.21,
        baseline_f1=0.22,
    )
    assert rejected is False
    assert bad_gain < 0.0
    assert "rejected & rolled back" in rej_reason


def test_sequential_stream_and_realized_transfer():
    """Verify Phase 2 & 4 sequential stream executes 5 episodes and accumulates realized transfers."""
    model, tokenizer = _create_mock_model_and_tokenizer(seed=42)
    repertoire = ModularSkillRepertoire(d_model=128, bottleneck_dim=16, num_classes=3)

    # Mock an initial source skill
    ad = Rank16ResidualAdapter(d_model=128, bottleneck_dim=16, num_classes=3)
    repertoire.register_skill(
        skill_id="source_transitive",
        rule_family="TRANSITIVE_DAG",
        source_env_id="env_transitive",
        adapter=ad,
        macro_f1=0.7,
        accuracy=0.7,
        classwise_recall={"NO": 0.7, "YES": 0.7, "UNKNOWN": 0.7},
    )

    class MockEnv:
        def __init__(self, name: str, expected_ans: str = "YES"):
            self.env_id = name
            self.examples = [
                {"question_text": f"Question for {name} sample 1", "expected_answer": expected_ans},
                {"question_text": f"Question for {name} sample 2", "expected_answer": "NO"},
            ]

    # Create 5-episode sequence: Ep 0 (Novel), Ep 1 (Novel), Ep 2 (Known), Ep 3 (Repeated Novel), Ep 4 (Repeated Novel)
    seq = [
        {
            "task_name": "ep0_rail_alpha",
            "rule_family": "PARITY_CYCLE",
            "world_id": "alpha",
            "is_repeated_encounter": False,
            "train_env": MockEnv("rail_train"),
            "qual_env": MockEnv("rail_qual"),
            "eval_env": MockEnv("rail_eval"),
        },
        {
            "task_name": "ep1_chromatin_alpha",
            "rule_family": "CHROMATIN_LATTICE",
            "world_id": "alpha",
            "is_repeated_encounter": False,
            "train_env": MockEnv("chrom_train"),
            "qual_env": MockEnv("chrom_qual"),
            "eval_env": MockEnv("chrom_eval"),
        },
        {
            "task_name": "ep2_dag_alpha",
            "rule_family": "TRANSITIVE_DAG",
            "world_id": "alpha",
            "is_repeated_encounter": False,
            "train_env": MockEnv("dag_train"),
            "qual_env": MockEnv("dag_qual"),
            "eval_env": MockEnv("dag_eval"),
        },
        {
            "task_name": "ep3_rail_beta",
            "rule_family": "PARITY_CYCLE",
            "world_id": "beta",
            "is_repeated_encounter": True,
            "train_env": MockEnv("rail_train"),
            "qual_env": MockEnv("rail_qual"),
            "eval_env": MockEnv("rail_eval"),
        },
        {
            "task_name": "ep4_chromatin_gamma",
            "rule_family": "CHROMATIN_LATTICE",
            "world_id": "gamma",
            "is_repeated_encounter": True,
            "train_env": MockEnv("chrom_train"),
            "qual_env": MockEnv("chrom_qual"),
            "eval_env": MockEnv("chrom_eval"),
        },
    ]

    sealed_env = MockEnv("sealed_eval")
    budget = BUDGET_REGIMES[BudgetTier.MEDIUM]
    engine = VerifiedContinualLearningStreamEngine()

    result = engine.run_sequential_stream(
        stream_id="test_stream",
        seed=42,
        model=model,
        initial_repertoire=repertoire,
        task_sequence=seq,
        sealed_eval_env=sealed_env,
        budget=budget,
        arm_name="RETAINED_LEARNING",
    )

    assert len(result.episodes) == 5
    assert result.final_repertoire_skill_count >= 1
    # Check that realized transfers occurred on repeated encounters
    assert result.realized_future_transfers >= 1
    assert result.accumulated_macro_f1 > 0.0
    assert result.accumulated_net_utility > 0.0


def test_m70_retention_safeguards_and_rollback():
    """Verify Phase 5 retention safeguards, bit-exact rollback, and governance gates."""
    model, tokenizer = _create_mock_model_and_tokenizer(seed=42)
    repertoire = ModularSkillRepertoire(d_model=128, bottleneck_dim=16, num_classes=3)
    ad = Rank16ResidualAdapter(d_model=128, bottleneck_dim=16, num_classes=3)
    repertoire.register_skill(
        skill_id="s1",
        rule_family="FAMILY_1",
        source_env_id="env_1",
        adapter=ad,
        macro_f1=0.8,
        accuracy=0.8,
        classwise_recall={},
    )

    cand_ad = Rank16ResidualAdapter(d_model=128, bottleneck_dim=16, num_classes=3)
    rollback_ok = M70SafeguardsAuditor.verify_bit_exact_rollback(
        repertoire=repertoire,
        candidate_id="test_candidate",
        candidate_adapter=cand_ad,
    )
    assert rollback_ok is True

    safeguards_report = M70SafeguardsAuditor.run_m70_safeguards_audit(
        model=model,
        repertoire=repertoire,
        seed=42,
    )
    assert safeguards_report.bit_exact_rollback_passed is True
    assert safeguards_report.carry_arithmetic_retained is True
    assert safeguards_report.expanded_logic_retained is True
    assert safeguards_report.topology_expansion_blocked is True
    assert "NO-GO" in safeguards_report.production_promotion_status
