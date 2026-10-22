from __future__ import annotations
from typing import Any
"""Unit tests for M48: Abstract Rule Induction and Cross-Environment Transfer."""

import torch

from adaptive_mini_gpt.data.bpe_scaling import train_scaled_bpe_tokenizer
from adaptive_mini_gpt.model.config import ModelConfig
from adaptive_mini_gpt.model.transformer import MiniGPT
from adaptive_mini_gpt.rule_induction.audit import M47ForensicAuditor
from adaptive_mini_gpt.rule_induction.continual_induction import ContinualRuleEvaluator
from adaptive_mini_gpt.rule_induction.disentangled_benchmark import (
    DisentangledWorldDataset,
    get_m48_disentangled_suite,
)
from adaptive_mini_gpt.rule_induction.learning_efficiency import RuleLearningEfficiencyRunner
from adaptive_mini_gpt.rule_induction.rule_arms import RuleInductionArmsRunner
from adaptive_mini_gpt.rule_induction.transfer_evaluator import PositiveNegativeTransferEvaluator


def _make_dummy_model_and_tok() -> tuple[MiniGPT, Any]:
    text = (
        "ProtoIndoEuropean derived_from ProtoItalic. ProtoGermanic LatinArchaic. "
        "MagistrateCourt appeals_to CountyTribunal. DistrictBench AppellateDivision. "
        "SubstrateEnzyme catalyzes_synthesis_of IntermediateKinaseA. ClientMicroservice calls_dynamically. "
        "HegemonRepublic sponsors_proxy_treaty_with SovereignStateA. YES NO UNKNOWN."
    )
    tok = train_scaled_bpe_tokenizer(text, target_vocab_size=512)
    cfg = ModelConfig(
        vocab_size=512,
        d_model=32,
        n_layers=2,
        n_heads=2,
        d_ff=64,
        context_length=128,
    )
    model = MiniGPT(cfg)
    return model, tok


def test_m47_forensic_audit() -> None:
    auditor = M47ForensicAuditor()
    report = auditor.generate_audit_report()
    assert report.m47_reported_accuracy == 0.6250
    assert report.m47_reported_unk_recall == 0.6250
    assert report.m47_reported_topology_accuracy == 0.8333
    assert report.m47_reported_backward_transfer == 0.1597
    assert report.m47_meta_auc == 0.3811
    assert report.m47_fresh_auc == 0.3928
    assert "DOMAIN-TOPOLOGY CORRELATION CONFOUND" in report.domain_topology_confound_diagnosis
    assert "TRANSFER EFFICIENCY AUDIT" in report.negative_transfer_diagnosis
    assert "M48 REMEDIATION SAFEGUARDS" in report.remediation_safeguard

    d = report.to_dict()
    assert d["m47_reported_accuracy"] == 0.6250


def test_disentangled_benchmark_suite() -> None:
    suite = get_m48_disentangled_suite()
    assert len(suite) == 7

    # Check domains and topologies are factorially decoupled
    topologies = {w.topology for w in suite.values()}
    assert topologies == {"TREE", "CHAIN", "FORK", "DIAMOND"}

    ling_topos = {w.topology for w in suite.values() if w.domain == "linguistics"}
    assert ling_topos == {"TREE", "CHAIN"}

    legal_topos = {w.topology for w in suite.values() if w.domain == "legal"}
    assert legal_topos == {"CHAIN", "FORK"}

    # Check balanced test pools: exactly 8 YES, 8 NO, 8 UNK = 24 queries
    for name, w in suite.items():
        assert len(w.held_out_test_pool) == 24
        counts = {c: sum(1 for t in w.held_out_test_pool if t.expected_verdict == c) for c in ["YES", "NO", "UNKNOWN"]}
        assert counts["YES"] == 8
        assert counts["NO"] == 8
        assert counts["UNKNOWN"] == 8

        # Demonstrations and minimal pair probes
        assert len(w.demonstrations) >= 2
        assert len(w.minimal_pair_probes) >= 2

        d = w.to_dict()
        assert d["world_id"] == name
        assert d["test_size"] == 24


def test_rule_induction_arms_evaluation() -> None:
    model, tok = _make_dummy_model_and_tok()
    suite = get_m48_disentangled_suite()
    w = suite["ling_tree_transitive"]

    runner = RuleInductionArmsRunner(model, tok, torch.device("cpu"), seed=42)

    # Fast evaluation with 2 epochs
    arm_a = runner.evaluate_arm_a_m47_baseline(w, practice_budget=6, epochs=2)
    arm_b = runner.evaluate_arm_b_neural_rule_induction(w, practice_budget=6, epochs=2)
    arm_c = runner.evaluate_arm_c_rule_induction_with_replay(w, practice_budget=6, epochs=2, past_demonstrations=[suite["legal_chain_transitive"]])
    arm_d = runner.evaluate_arm_d_deterministic_solver(w)

    assert 0.0 <= arm_a.overall_accuracy <= 1.0
    assert 0.0 <= arm_b.overall_accuracy <= 1.0
    assert 0.0 <= arm_c.overall_accuracy <= 1.0
    assert arm_d.overall_accuracy == 1.00  # Deterministic symbolic solver ceiling is 100%
    assert arm_d.macro_f1 == 1.00


def test_positive_negative_transfer_evaluation() -> None:
    model, tok = _make_dummy_model_and_tok()
    suite = get_m48_disentangled_suite()
    src = suite["ling_tree_transitive"]
    tgt_pos = suite["legal_chain_transitive"]
    tgt_neg = suite["ling_chain_nontransitive"]

    evaluator = PositiveNegativeTransferEvaluator(model, tok, torch.device("cpu"), seed=42)
    metrics = evaluator.evaluate_transfer_dynamics(src, tgt_pos, tgt_neg)

    assert -1.0 <= metrics.positive_transfer_gain <= 1.0
    assert 0.0 <= metrics.negative_transfer_error_rate <= 1.0
    assert 0.0 <= metrics.asymmetric_reversal_accuracy <= 1.0
    assert 0.0 <= metrics.bridge_deletion_sensitivity <= 1.0
    assert 0.0 <= metrics.minimal_pair_sensitivity <= 1.0

    d = metrics.to_dict()
    assert "positive_transfer_gain" in d


def test_rule_learning_efficiency_runner() -> None:
    model, tok = _make_dummy_model_and_tok()
    suite = get_m48_disentangled_suite()
    target_world = suite["geopolitics_fork_nontransitive"]

    runner = RuleLearningEfficiencyRunner(model, tok, torch.device("cpu"), seed=42)
    # Temporarily set budgets to small subset for fast unit testing
    runner.BUDGETS = [0, 2]
    res_fresh = runner.evaluate_efficiency("FRESH_LEARNER", target_world, epochs_per_budget=2)
    res_rule = runner.evaluate_efficiency("RULE_INDUCTION", target_world, epochs_per_budget=2)

    assert len(res_fresh.curve) == 2
    assert len(res_rule.curve) == 2
    assert 0.0 <= res_fresh.area_under_curve <= 1.0
    assert 0.0 <= res_rule.area_under_curve <= 1.0
    assert res_fresh.arithmetic_carry_retention == 1.00
    assert res_rule.arithmetic_carry_retention == 1.00


def test_continual_rule_evaluator() -> None:
    model, tok = _make_dummy_model_and_tok()
    suite = get_m48_disentangled_suite()
    worlds = [suite["ling_tree_transitive"], suite["ling_chain_nontransitive"]]

    evaluator = ContinualRuleEvaluator(model, tok, torch.device("cpu"), seed=42)
    res = evaluator.run_continual_rule_experiment(worlds, practice_budget_per_world=6, epochs_per_world=2)

    assert set(res.keys()) == {"SHARED_ADAPTER", "REPLAY_ADAPTER", "MODULAR_ADAPTERS"}
    for strat, ep_results in res.items():
        assert len(ep_results) == 2
        for ep in ep_results:
            assert ep.arithmetic_carry_retention == 1.00
            assert ep.chained_logic_retention == 1.00
            assert len(ep.per_world_accuracy) > 0
