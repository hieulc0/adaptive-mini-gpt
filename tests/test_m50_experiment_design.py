"""Unit tests for M50: Autonomous Experiment Design and Verified Rule Learning."""
from __future__ import annotations

from typing import Any
import pytest
import torch

from adaptive_mini_gpt.data.bpe_scaling import train_scaled_bpe_tokenizer
from adaptive_mini_gpt.experiment_design.audit import M49ForensicAuditor
from adaptive_mini_gpt.experiment_design.continual_study import ContinualAutonomousEvaluator
from adaptive_mini_gpt.experiment_design.expanded_hypothesis import (
    EXPANDED_HYPOTHESES,
    UNKNOWN_RULE,
    AutonomousEnvironment,
    build_m50_environment_suite,
)
from adaptive_mini_gpt.experiment_design.experiment_generator import (
    ExperimentGenerationRunner,
    GeneratedExperimentResult,
)
from adaptive_mini_gpt.experiment_design.hypothesis_revision import HypothesisRevisionTracker
from adaptive_mini_gpt.experiment_design.learning_evaluator import AutonomousObservationLearner
from adaptive_mini_gpt.experiment_design.rule_memory import ControlledRuleMemory
from adaptive_mini_gpt.experiment_design.syntax_validator import ExperimentSyntaxValidator
from adaptive_mini_gpt.model.config import ModelConfig
from adaptive_mini_gpt.model.transformer import MiniGPT


def _make_dummy_model_and_tok() -> tuple[MiniGPT, Any]:
    text = (
        "ProtoIndoEuropean derived_from ProtoItalic. ProtoGermanic LatinArchaic. "
        "FederalAuthority treaty_partner_with StateTribunalA. "
        "ServiceRegistry in_same_security_zone AuthGatewayNode. "
        "DomainEukarya taxonomically_subsumes KingdomMetazoa. "
        "SubstrateAlpha catalyzes_synthesis_of IntermediateBeta. "
        "RootAuthorityKey authorizes_session_for IntermediateCertA. LeafSignerX. "
        "EntangledPhotonPairA maintains_entanglement_with QuantumRepeaterNode1. "
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


def test_m49_forensic_audit() -> None:
    auditor = M49ForensicAuditor()
    report = auditor.generate_audit_report()

    assert report.m49_reported_arm_b_info_gain == 1.9894
    assert report.m49_reported_arm_a_info_gain == 1.5217
    assert report.m49_reported_paired_correctness == 0.5000
    assert report.m49_reported_final_acc == 0.3472
    assert "STATISTICAL SIGNIFICANCE AUDIT" in report.statistical_significance_audit
    assert "NEURAL-PLUS-MEMORY DEGRADATION" in report.neural_memory_degradation_diagnosis
    assert "SELECTION LATENCY" in report.selection_latency_audit
    assert "NEGATIVE BACKWARD TRANSFER" in report.negative_backward_transfer_diagnosis

    res = auditor.audit_statistical_significance([1.5, 1.6, 1.45], [1.99, 1.99, 1.99])
    assert not res["seed_level_t_test_valid"]
    assert res["query_level_wilcoxon_p"] == 0.003


def test_expanded_hypothesis_suite() -> None:
    suite = build_m50_environment_suite()
    assert len(suite) == 8
    assert "env_ling_dir_trans" in suite
    assert "env_legal_sym_nontrans" in suite
    assert "env_bio_hierarchical" in suite
    assert "env_chem_refutation" in suite
    assert "env_crypto_sealed_meta_test" in suite
    assert "env_quantum_sealed_meta_test" in suite

    # Verify held-out test sets are balanced
    for env in suite.values():
        assert len(env.held_out_test_queries) == 24
        counts = {c: sum(1 for q in env.held_out_test_queries if q.expected_answer == c) for c in ["YES", "NO", "UNKNOWN"]}
        assert counts["YES"] == 8
        assert counts["NO"] == 8
        assert counts["UNKNOWN"] == 8
        assert len(env.minimal_pairs) == 2


def test_syntax_validator() -> None:
    entities = ["NodeA", "NodeB", "NodeC"]
    rel = "connects_to"
    validator = ExperimentSyntaxValidator(entities, rel)

    # Valid
    val1 = validator.validate_experiment("NodeA", rel, "NodeB")
    assert val1.is_valid

    # Redundant
    val2 = validator.validate_experiment("NodeA", rel, "NodeB")
    assert not val2.is_valid
    assert val2.rejection_reason == "REDUNDANT_REPEATED_QUERY"

    # Self loop
    val3 = validator.validate_experiment("NodeC", rel, "NodeC")
    assert not val3.is_valid
    assert val3.rejection_reason == "SELF_LOOP_REFLEXIVE_QUERY"

    # Unknown entity
    val4 = validator.validate_experiment("NodeA", rel, "AlienNode")
    assert not val4.is_valid
    assert "UNKNOWN_TARGET_ENTITY" in val4.rejection_reason

    assert validator.stats.total_proposed == 4
    assert validator.stats.valid_accepted == 1


def test_hypothesis_revision_tracker_and_unknown_rule() -> None:
    edges = [("A", "B"), ("B", "C")]
    tracker = HypothesisRevisionTracker(edges)
    assert tracker.get_entropy() > 2.5

    # Reverse query
    eig = tracker.compute_expected_information_gain("B", "A")
    assert eig > 0.0

    # Observation update
    gain = tracker.update_belief("B", "A", "NO")
    assert gain > 0.0
    top_h, conf = tracker.get_top_hypothesis()
    assert not top_h.is_symmetric

    # Force all hypotheses to be contradicted to trigger UNKNOWN_RULE
    tracker.update_belief("A", "A", "NO")  # impossible under all 8 hypotheses
    top_h2, conf2 = tracker.get_top_hypothesis()
    assert top_h2.hypothesis_id == "UNKNOWN_RULE"
    assert conf2 > 0.90


def test_experiment_generation_arms() -> None:
    model, tok = _make_dummy_model_and_tok()
    suite = build_m50_environment_suite()
    env = suite["env_ling_dir_trans"]

    runner = ExperimentGenerationRunner(model, tok, torch.device("cpu"), seed=42)

    res_a = runner.run_arm_a_random_generation(env, query_budget=2)
    res_b = runner.run_arm_b_info_gain_selection(env, query_budget=2)
    res_c = runner.run_arm_c_neural_selection(env, query_budget=2)
    res_d = runner.run_arm_d_autonomous_generation(env, query_budget=2)

    assert res_a.query_budget == 2
    assert res_b.query_budget == 2
    assert res_c.query_budget == 2
    assert res_d.query_budget == 2
    assert res_b.cumulative_information_gain >= 0.0
    assert len(res_b.selected_queries) == 2
    assert len(res_d.selected_queries) == 2


def test_autonomous_observation_learner_curve() -> None:
    model, tok = _make_dummy_model_and_tok()
    suite = build_m50_environment_suite()
    env = suite["env_crypto_sealed_meta_test"]

    learner = AutonomousObservationLearner(model, tok, torch.device("cpu"), seed=42)
    learner.BUDGETS = [0, 2]

    obs = [("RootAuthorityKey", "IntermediateCertA", "YES"), ("IntermediateCertA", "RootAuthorityKey", "NO")]
    res = learner.evaluate_curve("TEST_STRATEGY", env, obs, epochs_per_budget=2)

    assert len(res.curve) == 2
    assert 0.0 <= res.area_under_curve <= 1.0
    assert 0.0 <= res.final_paired_correctness <= 1.0
    assert res.arithmetic_carry_retention == 1.00
    assert res.chained_logic_retention == 1.00


def test_controlled_rule_memory_qualification() -> None:
    model, tok = _make_dummy_model_and_tok()
    suite = build_m50_environment_suite()
    env = suite["env_ling_dir_trans"]

    memory = ControlledRuleMemory(torch.device("cpu"))
    obs = [("ProtoIndoEuropean", "ProtoItalic", "YES")]

    # Perfect rule -> should qualify
    qualified = memory.qualify_and_store_rule(
        env=env,
        identified_hypothesis=EXPANDED_HYPOTHESES["DIR_TRANS"],
        self_confidence=0.99,
        supporting_obs=obs,
        counterexamples=["SYM_TRANS"],
    )
    assert qualified
    assert len(memory.entries) == 1

    # Incompatible rule -> should fail qualification
    failed = memory.qualify_and_store_rule(
        env=env,
        identified_hypothesis=EXPANDED_HYPOTHESES["SYM_NONTRANS"],
        self_confidence=0.99,
        supporting_obs=obs,
        counterexamples=[],
    )
    assert not failed

    # Test retrieval
    compat = memory.retrieve_compatible_rule(env)
    assert compat is not None
    assert compat.rule_name == "Directed Transitive"


def test_continual_autonomous_evaluator() -> None:
    model, tok = _make_dummy_model_and_tok()
    suite = build_m50_environment_suite()
    envs = [suite["env_ling_dir_trans"], suite["env_legal_sym_nontrans"]]

    evaluator = ContinualAutonomousEvaluator(model, tok, torch.device("cpu"), seed=42)
    results = evaluator.run_continual_experiment(envs, practice_budget_per_env=4, epochs_per_env=2)

    assert set(results.keys()) == {"SHARED_ADAPTER", "REPLAY_ADAPTER", "MODULAR_ADAPTERS"}
    for strat, ep_list in results.items():
        assert len(ep_list) == 2
        for ep in ep_list:
            assert ep.arithmetic_carry_retention == 1.00
            assert ep.chained_logic_retention == 1.00
