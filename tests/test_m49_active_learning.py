from __future__ import annotations
from typing import Any
"""Unit tests for M49: Active Learning and Efficient Experiment Selection."""

import torch

from adaptive_mini_gpt.active_learning.audit import M48ForensicAuditor
from adaptive_mini_gpt.active_learning.continual_active import ContinualActiveEvaluator
from adaptive_mini_gpt.active_learning.environment import (
    ActiveLearningEnvironment,
    CandidateQuery,
    HYPOTHESES,
    build_m49_environment_suite,
)
from adaptive_mini_gpt.active_learning.hypothesis_tracker import BayesianHypothesisTracker
from adaptive_mini_gpt.active_learning.learning_evaluator import ActiveObservationLearner
from adaptive_mini_gpt.active_learning.selection_arms import ActiveLearningArmsRunner
from adaptive_mini_gpt.data.bpe_scaling import train_scaled_bpe_tokenizer
from adaptive_mini_gpt.model.config import ModelConfig
from adaptive_mini_gpt.model.transformer import MiniGPT


def _make_dummy_model_and_tok() -> tuple[MiniGPT, Any]:
    text = (
        "ProtoIndoEuropean derived_from ProtoItalic. ProtoGermanic LatinArchaic. "
        "FederalAuthority treaty_partner_with StateTribunalA. SubstrateEnzyme inhibits_immediately. "
        "RootAuthorityKey authorizes_session_for IntermediateCertA. LeafSignerX HardwareEnclave. "
        "YES NO UNKNOWN."
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


def test_m48_forensic_audit_and_paired_correctness() -> None:
    auditor = M48ForensicAuditor()
    report = auditor.generate_audit_report()

    assert report.m48_reported_fresh_auc == 0.4410
    assert report.m48_reported_rule_auc == 0.4410
    assert report.m48_reported_unk_recall == 0.7917
    assert report.m48_reported_overall_acc == 0.3889
    assert report.m48_reported_yes_recall == 0.0417
    assert report.m48_reported_minimal_pair_sens == 0.6667
    assert report.m48_reported_positive_transfer == 0.0000
    assert "INCOMPLETE FACTORIAL COVERAGE" in report.incomplete_factorial_diagnosis
    assert "CONSERVATIVE COLLAPSE" in report.conservative_collapse_diagnosis
    assert "PAIRED CORRECTNESS CONTROL" in report.paired_correctness_diagnosis
    assert "PASSIVE DEMONSTRATION INEFFICIENCY" in report.passive_demonstration_cost_diagnosis

    # Test paired correctness helper
    # 4 queries: in 2 both are correct; in 1 only original is correct; in 1 neither
    orig = [True, True, True, False]
    pert = [True, True, False, False]
    paired_score = auditor.compute_paired_correctness(orig, pert)
    assert paired_score == 0.50  # 2 out of 4


def test_active_environment_and_oracle() -> None:
    suite = build_m49_environment_suite()
    assert len(suite) == 6
    assert "env_ling_dir_trans" in suite
    assert "env_legal_sym_nontrans" in suite
    assert "env_crypto_sealed_meta_test" in suite

    env = suite["env_ling_dir_trans"]
    assert env.oracle_call_count == 0
    q_id = env.candidate_queries[0].query_id
    ans = env.ask_oracle(q_id)
    assert ans in ("YES", "NO", "UNKNOWN")
    assert env.oracle_call_count == 1

    # Check held-out test pool is balanced: exactly 8 YES, 8 NO, 8 UNK = 24 queries
    for name, e in suite.items():
        assert len(e.held_out_test_queries) == 24
        counts = {c: sum(1 for q in e.held_out_test_queries if q.expected_answer == c) for c in ["YES", "NO", "UNKNOWN"]}
        assert counts["YES"] == 8
        assert counts["NO"] == 8
        assert counts["UNKNOWN"] == 8
        assert len(e.minimal_pairs) >= 2


def test_bayesian_hypothesis_tracker_eig() -> None:
    tracker = BayesianHypothesisTracker()
    assert tracker.get_entropy() == 2.0  # log2(4) = 2.0 bits

    # Candidate reverse query
    q_rev = CandidateQuery("q_rev", "A", "B", "REVERSE_TEST", "Does B rel A?", "NO")
    eig_rev = tracker.compute_expected_information_gain(q_rev)
    assert eig_rev > 0.50  # Informative query

    # Candidate direct query (non-informative because all hypotheses agree on direct edge)
    q_dir = CandidateQuery("q_dir", "A", "B", "DIRECT_EDGE", "Does A rel B?", "YES")
    eig_dir = tracker.compute_expected_information_gain(q_dir)
    assert eig_dir < 0.10  # Zero or negligible information gain

    # Update with observed answer
    gain = tracker.update_belief(q_rev, "NO")
    assert gain > 0.0
    assert tracker.get_entropy() < 2.0
    top_h, conf = tracker.get_top_hypothesis()
    assert not top_h.is_symmetric  # Should rule out symmetric hypotheses


def test_active_learning_arms_runner() -> None:
    model, tok = _make_dummy_model_and_tok()
    suite = build_m49_environment_suite()
    env = suite["env_ling_dir_trans"]

    runner = ActiveLearningArmsRunner(model, tok, torch.device("cpu"), seed=42)

    res_a = runner.run_arm_a_random(env, query_budget=2)
    res_b = runner.run_arm_b_info_gain(env, query_budget=2)
    res_c = runner.run_arm_c_neural_policy(env, query_budget=2)
    res_d = runner.run_arm_d_neural_with_memory(env, query_budget=2)

    assert res_a.query_budget == 2
    assert res_b.query_budget == 2
    assert res_c.query_budget == 2
    assert res_d.query_budget == 2
    assert res_b.cumulative_information_gain >= 0.0
    assert res_b.oracle_call_count == 2
    assert len(res_b.selected_queries) == 2


def test_active_observation_learner_curve() -> None:
    model, tok = _make_dummy_model_and_tok()
    suite = build_m49_environment_suite()
    env = suite["env_crypto_sealed_meta_test"]

    learner = ActiveObservationLearner(model, tok, torch.device("cpu"), seed=42)
    learner.BUDGETS = [0, 2]  # Small subset for fast test

    selected_obs = env.candidate_queries[:2]
    res = learner.evaluate_curve("ACTIVE_TEST", env, selected_obs, epochs_per_budget=2)

    assert len(res.curve) == 2
    assert 0.0 <= res.area_under_curve <= 1.0
    assert 0.0 <= res.final_paired_correctness <= 1.0
    assert res.arithmetic_carry_retention == 1.00
    assert res.chained_logic_retention == 1.00


def test_continual_active_evaluator() -> None:
    model, tok = _make_dummy_model_and_tok()
    suite = build_m49_environment_suite()
    envs = [suite["env_ling_dir_trans"], suite["env_legal_sym_nontrans"]]

    evaluator = ContinualActiveEvaluator(model, tok, torch.device("cpu"), seed=42)
    results = evaluator.run_continual_experiment(envs, practice_budget_per_env=4, epochs_per_env=2)

    assert set(results.keys()) == {"SHARED_ADAPTER", "REPLAY_ADAPTER", "MODULAR_ADAPTERS"}
    for strat, ep_results in results.items():
        assert len(ep_results) == 2
        for ep in ep_results:
            assert ep.arithmetic_carry_retention == 1.00
            assert ep.chained_logic_retention == 1.00
            assert len(ep.per_world_accuracy) > 0
