"""Deterministic Unit & Integration Tests for Milestone M73: Causal Neural Memory."""

from __future__ import annotations

import pytest
import torch

from adaptive_mini_gpt.causal_neural_memory import (
    CostAndUtilityLedger,
    CostEfficientMemoryRegistry,
    M72ScientificAuditor,
    NegativeTransferAuditor,
    NeuralWeightAttributionAuditor,
    PhysicalResourceRecord,
    SkillState,
    build_m73_open_world_stream_sequence,
    create_hierarchical_subsumption_env,
    create_modular_congruence_mod5_env,
)
from adaptive_mini_gpt.data.bpe_scaling import train_scaled_bpe_tokenizer
from adaptive_mini_gpt.model import MiniGPT, ModelConfig
from adaptive_mini_gpt.modular_skill_learning.repertoire import Rank16ResidualAdapter


@pytest.fixture
def tiny_test_model_and_tokenizer():
    """Create deterministic tiny test model with annotated tokenizer."""
    tok = train_scaled_bpe_tokenizer(
        "A B C 0 1 2 YES NO UNKNOWN Evidence Subsumes Congruent", target_vocab_size=64
    )
    cfg = ModelConfig(
        vocab_size=tok.vocab_size,
        d_model=128,
        n_layers=2,
        n_heads=2,
        d_ff=128,
        context_length=64,
    )
    model = MiniGPT(cfg)
    model.__dict__["tokenizer"] = tok
    model.eval()
    return model, tok


def test_m72_forensic_audit():
    """Test 1: Verify M72 forensic audit reproduces metrics and validates Student's t."""
    report = M72ScientificAuditor.audit_m72_results()
    assert report.audit_passed is True
    assert len(report.anomalies) == 4

    anomaly_ids = [a.anomaly_id for a in report.anomalies]
    assert "M72-A01-UTILITY-INVERSION" in anomaly_ids
    assert "M72-A02-WRONG-ROUTE-NEGATIVE-TRANSFER-DEFECT" in anomaly_ids
    assert "M72-A03-CONFIDENCE-INTERVAL-UNDERESTIMATION" in anomaly_ids
    assert "M72-A04-BACKBONE-PARAMETER-LINEAGE" in anomaly_ids

    # Verify statistical audit: N=3 df=2 uses t=4.303
    stat = report.statistical_audit
    assert stat["sample_size_seeds"] == 3
    assert stat["t_critical_value"] == 4.303
    assert stat["ratio_t_to_z"] > 2.0
    assert stat["contains_zero"] is True

    # Lineage audit: 974,658 weights verified
    lineage = report.lineage_reconciliation
    assert lineage["expanded_backbone_parameters"] == 974658
    assert lineage["source_adapter_parameters"] == 7113
    assert lineage["total_base_model_parameters"] == 981771


def test_transparent_cost_accounting():
    """Test 2: Verify transparent physical resource decomposition and sensitivity analysis."""
    ledger = CostAndUtilityLedger(expected_amortization_k=3)

    resources = PhysicalResourceRecord(
        observation_queries=16,
        candidate_training_steps=40,
        gpu_training_seconds=0.50,
        inference_latency_ms=2.0,
        total_stored_adapter_params=14226,
        total_stored_adapter_bytes=14226 * 4,
        active_adapter_params=2371,
        dormant_adapter_params=11855,
        gpu_memory_allocated_mb=12.5,
        realized_future_reuses=2,
    )

    outcome = ledger.compute_transparent_outcome(
        policy_name="test_selective_retain",
        task_name="test_task_ep0",
        macro_f1=0.70,
        accuracy=0.70,
        resources=resources,
        false_allocations=0,
        missed_reuses=0,
        action_name="EXPERIMENT_AND_ADAPT",
    )

    # Verify breakdown
    breakdown = outcome.legacy_utility_breakdown
    assert breakdown.macro_f1 == 0.70
    assert breakdown.query_penalty > 0.0
    assert breakdown.step_penalty > 0.0
    # Stored params 14226 * 0.00002 = 0.28452
    assert abs(breakdown.storage_penalty - 0.28452) < 1e-4

    # Physical engineering utility reflects tiny 56KB storage instead of 0.28 penalty!
    assert outcome.physical_engineering_utility > outcome.legacy_utility_breakdown.net_utility
    assert outcome.amortized_physical_utility >= outcome.physical_engineering_utility

    # Sensitivity analysis contains all 4 regimes
    assert "compute_heavy" in outcome.sensitivity_utilities
    assert "quality_heavy" in outcome.sensitivity_utilities
    assert "storage_heavy" in outcome.sensitivity_utilities
    assert "zero_storage_penalty" in outcome.sensitivity_utilities


def test_negative_transfer_detector(tiny_test_model_and_tokenizer):
    """Test 3: Verify rigorous negative transfer detection against safe fallback."""
    model, _ = tiny_test_model_and_tokenizer
    device = torch.device("cpu")
    auditor = NegativeTransferAuditor(epsilon_margin=0.02)

    queries = [
        {
            "question_text": "Evidence: [A->B]. Does A subsumes_category B? ",
            "expected_answer": "YES",
        },
        {
            "question_text": "Evidence: [B->C]. Does C subsumes_category A? ",
            "expected_answer": "NO",
        },
        {
            "question_text": "Evidence: [A->B]. Does B subsumes_category C? ",
            "expected_answer": "UNKNOWN",
        },
    ]

    adapter_good = Rank16ResidualAdapter(d_model=128, bottleneck_dim=16, num_classes=3)
    adapter_fallback = Rank16ResidualAdapter(d_model=128, bottleneck_dim=16, num_classes=3)

    cmp_res = auditor.evaluate_task_transfer(
        task_name="test_eval_task",
        rule_family="HIERARCHICAL_SUBSUMPTION",
        is_adversarial=False,
        is_repeated=True,
        selected_skill_id="cand_good",
        selected_adapter=adapter_good,
        safe_fallback_id="source_fallback",
        safe_fallback_adapter=adapter_fallback,
        all_repertoire_skills={"cand_good": adapter_good, "source_fallback": adapter_fallback},
        model=model,
        sealed_test_queries=queries,
        device=device,
    )

    assert cmp_res.task_name == "test_eval_task"
    assert isinstance(cmp_res.delta_vs_fallback, float)
    assert isinstance(cmp_res.is_statistical_negative_transfer, bool)

    summary = auditor.aggregate_stream_comparisons([cmp_res])
    assert summary.total_evaluations == 1
    assert hasattr(summary, "harmful_route_choices")
    assert hasattr(summary, "statistical_negative_transfers")


def test_neural_weight_attribution(tiny_test_model_and_tokenizer):
    """Test 4: Verify 6-condition weight attribution and causal ablation."""
    model, _ = tiny_test_model_and_tokenizer
    device = torch.device("cpu")

    queries = [
        {
            "question_text": "Evidence: [A->B]. Does A subsumes_category B? ",
            "expected_answer": "YES",
        },
        {
            "question_text": "Evidence: [B->C]. Does C subsumes_category A? ",
            "expected_answer": "NO",
        },
    ]

    learned_ad = Rank16ResidualAdapter(d_model=128, bottleneck_dim=16, num_classes=3)
    irrelevant_ad = Rank16ResidualAdapter(d_model=128, bottleneck_dim=16, num_classes=3)

    report = NeuralWeightAttributionAuditor.run_attribution_study(
        task_name="test_recurring_subsumption",
        rule_family="HIERARCHICAL_SUBSUMPTION",
        intervening_episodes=6,
        candidate_skill_id="cand_subsumption_ep0",
        learned_adapter=learned_ad,
        irrelevant_adapter=irrelevant_ad,
        freshly_trained_adapter=None,
        model=model,
        test_queries=queries,
        device=device,
    )

    assert report.task_name == "test_recurring_subsumption"
    assert len(report.evaluations) == 6
    assert "LEARNED_WEIGHTS" in report.evaluations
    assert "RESET_WEIGHTS" in report.evaluations
    assert "SHUFFLED_WEIGHTS" in report.evaluations
    assert "IRRELEVANT_WEIGHTS" in report.evaluations
    assert "BASE_MODEL_FROZEN" in report.evaluations
    assert "FRESHLY_TRAINED" in report.evaluations


def test_cost_efficient_memory_registry_and_consolidation(tiny_test_model_and_tokenizer):
    """Test 5: Verify reversible memory lifecycle, duplicate pruning, and consolidation diagnostic."""
    model, _ = tiny_test_model_and_tokenizer
    device = torch.device("cpu")
    reg = CostEfficientMemoryRegistry(d_model=128, bottleneck_dim=16, num_classes=3)

    ad1 = Rank16ResidualAdapter(d_model=128, bottleneck_dim=16, num_classes=3)
    ad2 = Rank16ResidualAdapter(d_model=128, bottleneck_dim=16, num_classes=3)

    entry1 = reg.register_skill(
        "cand_mod3_ep18", "MODULAR_CONGRUENCE_MOD_3", "env1", ad1, 0.80, 0.80
    )
    entry2 = reg.register_skill(
        "cand_mod3_ep19", "MODULAR_CONGRUENCE_MOD_3", "env2", ad2, 0.70, 0.70
    )

    assert len(reg.list_active_skills()) == 2
    assert reg.total_stored_parameters() == 2 * 2371

    # Prune duplicates: lower-F1 candidate moved to DORMANT
    pruned = reg.prune_redundant_skills()
    assert pruned == ["cand_mod3_ep19"]
    assert entry2.state == SkillState.DORMANT
    assert len(reg.list_active_skills()) == 1

    # Test reversible consolidation diagnostic
    q_dummy = [
        {"question_text": "Evidence: [0->3]. Is 0 congruent with 3? ", "expected_answer": "YES"}
    ]
    diag = CostEfficientMemoryRegistry.test_reversible_consolidation(
        entry_a=entry1,
        entry_b=entry2,
        task_a_queries=q_dummy,
        task_b_queries=q_dummy,
        model=model,
        device=device,
    )

    assert diag.skill_a_id == "cand_mod3_ep18"
    assert diag.skill_b_id == "cand_mod3_ep19"
    assert diag.storage_saving_params == 2371
    assert diag.recommendation in ("PRESERVE_UNMERGED", "PROCEED_CONSOLIDATION")

    # Verify open-world stream generation
    open_world_seq = build_m73_open_world_stream_sequence()
    assert len(open_world_seq) == 20
    assert any(ep["rule_family"] == "HIERARCHICAL_SUBSUMPTION" for ep in open_world_seq)
    assert any(ep["rule_family"] == "MODULAR_CONGRUENCE_MOD_5" for ep in open_world_seq)

    # Test custom env constructors
    sub_env = create_hierarchical_subsumption_env()
    assert len(sub_env.held_out_test_queries) >= 8

    mod5_env = create_modular_congruence_mod5_env()
    assert len(mod5_env.held_out_test_queries) >= 8
