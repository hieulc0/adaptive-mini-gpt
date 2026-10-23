"""Unit tests for M51: Guided Experiment Design and Adaptive Learning."""
from __future__ import annotations

from typing import Any
import pytest
import torch
import numpy as np

from adaptive_mini_gpt.data.bpe_scaling import train_scaled_bpe_tokenizer
from adaptive_mini_gpt.guided_experiment import (
    M50ForensicAuditor,
    AutonomousEnvironment,
    EXPANDED_HYPOTHESES,
    UNKNOWN_RULE,
    OUT_OF_HYPOTHESIS_PARITY,
    build_m51_environment_suite,
    StructuralProbeGenerator,
    GuidedHypothesisTracker,
    NeuralProbeRankingHead,
    train_neural_ranking_head,
    GuidedArmsRunner,
    GuidedObservationLearner,
    CandidateAdapterHead,
    CompatibleRuleMemory,
    SemanticCompatibilityChecker,
    ContinualGuidedEvaluator,
)
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
        "HubLeader1 RelayMemberA RelayMemberB BridgeNodeC TargetFollowerD. "
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


def test_m50_audit_reconciliation():
    auditor = M50ForensicAuditor()
    report = auditor.generate_audit_report()
    assert report.m50_reported_arm_b_infogain_ig == 3.0000
    assert report.m50_reported_arm_d_autogen_ig == 1.1187
    assert report.m50_reported_arm_a_random_ig == 1.2767
    assert report.m50_reported_infogain_auc == 0.4635
    assert report.m50_reported_random_auc == 0.3759
    assert report.m50_reported_incompatible_memory_auc == 0.3099
    assert "UNGUIDED NEURAL GENERATION FAILURE" in report.unguided_generation_failure_diagnosis
    assert "MEMORY QUALIFICATION VS REUSE PARADOX" in report.memory_qualification_paradox_diagnosis


def test_structural_probe_generator_validity():
    entities = ["Alpha", "Beta", "Gamma", "Delta", "Epsilon"]
    edges = [("Alpha", "Beta"), ("Beta", "Gamma"), ("Gamma", "Delta")]
    generator = StructuralProbeGenerator(entities, edges, relation_name="connects_to")
    probes, stats = generator.generate_candidate_probes(max_probes=20)

    assert len(probes) > 0
    assert stats.validity_rate == 1.00

    # Ensure no self-loops and no duplicate queries
    pairs = set()
    for p in probes:
        assert p.source != p.target, f"Reflexive query detected: {p}"
        assert p.source in entities and p.target in entities
        pair = (p.source, p.target)
        assert pair not in pairs, f"Duplicate query detected: {pair}"
        pairs.add(pair)

    # Check that probe types include directional reversals and bridges
    types = {p.probe_type for p in probes}
    assert "DIRECTION_REVERSAL" in types
    assert "TWO_HOP_BRIDGE" in types


def test_hypothesis_tracker_shannon_entropy_and_eig():
    edges = [("A", "B"), ("B", "C")]
    tracker = GuidedHypothesisTracker(base_edges=edges, observation_noise=0.01)

    initial_entropy = tracker.compute_entropy()
    assert initial_entropy > 0.0

    # Test EIG computation for a reversal query
    eig = tracker.compute_expected_information_gain(("B", "A"))
    assert eig >= 0.0

    # Update with an observation and verify entropy decreases
    step_ig = tracker.update("B", "A", "NO")
    assert step_ig >= 0.0
    post_entropy = tracker.compute_entropy()
    assert post_entropy <= initial_entropy + 1e-6


def test_out_of_hypothesis_detection():
    # Parity cycle environment: odd length = NO, even length = YES
    edges = [("A", "B"), ("B", "C"), ("C", "D")]
    tracker = GuidedHypothesisTracker(base_edges=edges, observation_noise=0.01)

    # Contradictory evidence: 2-hop bridge (A, C) -> YES (even), 1-hop (A, B) -> NO (odd)
    # Feed observations that violate standard DAG and symmetric transitive rules
    tracker.update("A", "B", "NO")
    tracker.update("B", "C", "NO")
    tracker.update("A", "C", "YES")

    # Inadequate known hypotheses: belief should shift to UNKNOWN_RULE
    assert tracker.beliefs[UNKNOWN_RULE.hypothesis_id] > 0.10


def test_ranking_head_and_arms_runner():
    suite = build_m51_environment_suite()
    env = suite["env_ling_dir_trans"]

    ranking_head = NeuralProbeRankingHead(in_features=8, hidden_dim=16)
    train_neural_ranking_head(ranking_head, [env], num_epochs=2, lr=1e-3, seed=42)

    runner = GuidedArmsRunner(env, ranking_head, query_budget=4, seed=42)

    res_a = runner.run_arm("ARM_A_RANDOM")
    assert res_a.validity_rate == 1.00
    assert len(res_a.queries_executed) == 4

    res_b = runner.run_arm("ARM_B_INFOGAIN")
    assert res_b.total_information_gain >= 0.0

    res_c = runner.run_arm("ARM_C_NEURAL")
    assert res_c.validity_rate == 1.00

    res_d = runner.run_arm("ARM_D_NEURAL_FILTERED")
    assert res_d.validity_rate == 1.00

    res_ablation = runner.run_arm("ARM_ABLATION_FILTER_ONLY")
    assert res_ablation.validity_rate == 1.00


def test_rule_memory_compatibility_verification():
    mock_minigpt, char_tok = _make_dummy_model_and_tok()
    device = torch.device("cpu")
    memory = CompatibleRuleMemory(device=device)

    suite = build_m51_environment_suite()
    source_env = suite["env_ling_dir_trans"]
    target_env = suite["env_crypto_blocked_bridge"]

    # Qualify and store standard transitive rule from source environment
    stored = memory.qualify_and_store_rule(
        env=source_env,
        identified_hypothesis=EXPANDED_HYPOTHESES["DIR_TRANS"],
        self_confidence=0.99,
        supporting_obs=[("ProtoIndoEuropean", "ProtoItalic", "YES")],
        counterexamples=["DIR_NONTRANS"],
    )
    assert stored is True
    assert len(memory.entries) == 1

    # Check compatibility with target environment containing blocked bridges
    # The standard transitive rule should FAIL compatibility check because target has blocked bridges
    retrieved_entry, report = memory.retrieve_compatible_rule(target_env, observed_queries=[])
    assert retrieved_entry is None
    assert report is None  # Blocked by compatibility checker!

    # Benchmark memory conditions
    learner = GuidedObservationLearner(mock_minigpt, char_tok, device, seed=42)
    candidate_obs = [("RootAuthorityKey", "IntermediateCertA", "YES")]

    incompat_entry = list(memory.entries.values())[0]
    results = memory.benchmark_compatibility_arms(
        target_env=target_env,
        learner=learner,
        candidate_obs=candidate_obs,
        incompatible_entry=incompat_entry,
    )
    assert "FRESH_ADAPTATION" in results
    assert "COMPATIBLE_REUSE" in results
    assert "INCOMPATIBLE_REUSE" in results
    assert results["INCOMPATIBLE_REUSE"].compatibility_passed is False


def test_continual_guided_evaluator():
    mock_minigpt, char_tok = _make_dummy_model_and_tok()
    device = torch.device("cpu")
    evaluator = ContinualGuidedEvaluator(mock_minigpt, char_tok, device, seed=42)

    suite = build_m51_environment_suite()
    envs = [suite["env_ling_dir_trans"], suite["env_soft_sym_trans"]]

    results = evaluator.run_continual_experiment(
        environments=envs,
        practice_budget_per_env=4,
        epochs_per_env=2,
    )

    assert "SHARED_ADAPTER" in results
    assert "MODULAR_ADAPTERS" in results
    assert len(results["MODULAR_ADAPTERS"]) == 2
    assert results["MODULAR_ADAPTERS"][-1].arithmetic_carry_retention == 1.00
