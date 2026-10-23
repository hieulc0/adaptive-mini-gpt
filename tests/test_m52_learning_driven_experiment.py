"""Unit tests for M52: Learning-Outcome-Driven Experiment Selection."""
from __future__ import annotations

import math
from typing import Any
import pytest
import torch
import numpy as np

from adaptive_mini_gpt.data.bpe_scaling import train_scaled_bpe_tokenizer
from adaptive_mini_gpt.guided_experiment.environment import (
    EXPANDED_HYPOTHESES,
    UNKNOWN_RULE,
    OUT_OF_HYPOTHESIS_PARITY,
)
from adaptive_mini_gpt.learning_driven_experiment import (
    M51ScientificAuditor,
    NormalizedEpistemicTracker,
    build_m52_environment_suite,
    MinimalPairEngine,
    LearningOutcomePolicyHead,
    LearningOutcomeEvaluator,
    LearningDrivenArmsRunner,
    train_learning_outcome_policy,
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


def test_m51_audit_reconciliation():
    auditor = M51ScientificAuditor()
    report = auditor.generate_audit_report()

    # Verify mathematical entropy reconciliation
    assert math.isclose(report.calculated_prior_entropy_bits, 3.1223, abs_tol=1e-3)
    assert math.isclose(report.reconciled_eig_information_gain, 3.1098, abs_tol=1e-3)
    assert "RECONCILED" in report.entropy_reconciliation_verdict
    assert "DIAGNOSIS OF UNKNOWN_RULE" in report.unknown_rule_failure_diagnosis
    assert "DIAGNOSIS OF 0% STRUCTURAL PAIRED CORRECTNESS" in report.paired_correctness_failure_diagnosis


def test_normalized_epistemic_tracker():
    edges = [("A", "B"), ("B", "C")]
    tracker = NormalizedEpistemicTracker(base_edges=edges, unknown_rule_prior=0.15)

    # 1. Check strict normalization
    total_belief = sum(tracker.beliefs.values())
    assert math.isclose(total_belief, 1.0, abs_tol=1e-6)

    # 2. Check initial entropy
    initial_entropy = tracker.compute_entropy()
    assert initial_entropy > 0.0

    # 3. Check bidirectional belief updates (can increase or decrease)
    # Give confirming evidence for transitive rule
    prior_unk = tracker.beliefs[UNKNOWN_RULE.hypothesis_id]
    tracker.update("A", "B", "YES")
    post_unk_1 = tracker.beliefs[UNKNOWN_RULE.hypothesis_id]
    # UNKNOWN_RULE belief should DECREASE when known rule is confirmed
    assert post_unk_1 <= prior_unk

    # Give sudden contradictory evidence
    tracker.update("B", "A", "UNKNOWN")
    post_unk_2 = tracker.beliefs[UNKNOWN_RULE.hypothesis_id]
    # Total probability remains strictly normalized
    assert math.isclose(sum(tracker.beliefs.values()), 1.0, abs_tol=1e-6)


def test_unknown_rule_detection_calibration():
    """Test known rules, noisy observations, and truly out-of-hypothesis rules separately."""
    edges = [("A", "B"), ("B", "C"), ("C", "D")]

    # Case A: Known rule (DIR_TRANS)
    tracker_known = NormalizedEpistemicTracker(base_edges=edges, unknown_rule_prior=0.15)
    tracker_known.update("A", "B", "YES")
    tracker_known.update("B", "C", "YES")
    tracker_known.update("A", "C", "YES")
    assert not tracker_known.is_unknown_rule_detected(threshold=0.50)
    top_h, _ = tracker_known.get_top_hypothesis()
    assert top_h == "DIR_TRANS"

    # Case B: Out-of-hypothesis rule (Alternating parity cycle)
    # In parity: length 1 -> NO, length 2 -> YES, length 3 -> NO
    tracker_unseen = NormalizedEpistemicTracker(base_edges=edges, unknown_rule_prior=0.15)
    tracker_unseen.update("A", "B", "NO")
    tracker_unseen.update("B", "C", "NO")
    tracker_unseen.update("A", "C", "YES")
    tracker_unseen.update("A", "D", "NO")

    # Belief should shift toward UNKNOWN_RULE as all standard DAG hypotheses fail
    assert tracker_unseen.beliefs[UNKNOWN_RULE.hypothesis_id] > 0.20


def test_segregated_environment_suite():
    suite = build_m52_environment_suite()

    # Verify sets are disjoint
    train_keys = set(suite.train_environments.keys())
    dev_keys = set(suite.dev_environments.keys())
    sealed_keys = set(suite.sealed_test_environments.keys())

    assert len(train_keys & dev_keys) == 0
    assert len(train_keys & sealed_keys) == 0
    assert len(dev_keys & sealed_keys) == 0

    assert len(suite.train_environments) == 4
    assert len(suite.dev_environments) == 2
    assert len(suite.sealed_test_environments) == 3


def test_minimal_pairs_engine():
    suite = build_m52_environment_suite()
    env = suite.sealed_test_environments["env_sealed_crypto_blocked_bridge"]

    pairs = MinimalPairEngine.generate_controlled_minimal_pairs(env, max_pairs=6)
    assert len(pairs) > 0

    for pair in pairs:
        assert pair.base_query.expected_answer != pair.perturbed_query.expected_answer
        assert pair.perturbation_type in ["BRIDGE_DELETION", "DIRECTIONAL_REVERSAL", "TWO_HOP_BRIDGE_CUT"]

    # Test perfect predictor
    def oracle_pred(q_text: str) -> str:
        for p in pairs:
            if p.base_query.question_text == q_text:
                return p.base_query.expected_answer
            if p.perturbed_query.question_text == q_text:
                return p.perturbed_query.expected_answer
        return "UNKNOWN"

    res = MinimalPairEngine.evaluate_paired_correctness(oracle_pred, pairs)
    assert res.paired_correctness_rate == 1.00
    assert res.both_correct_count == len(pairs)


def test_learning_outcome_policy_and_arms():
    model, tok = _make_dummy_model_and_tok()
    device = torch.device("cpu")
    suite = build_m52_environment_suite()

    evaluator = LearningOutcomeEvaluator(model, tok, device, seed=42)
    policy_head = LearningOutcomePolicyHead(in_features=8, hidden_dim=16).to(device)

    # Quick smoke train
    loss = train_learning_outcome_policy(
        policy_head=policy_head,
        evaluator=evaluator,
        train_environments=list(suite.train_environments.values())[:2],
        dev_environments=list(suite.dev_environments.values())[:1],
        num_episodes=2,
        steps_per_episode=2,
        seed=42,
    )
    assert isinstance(loss, float)

    # Test all 5 exploration arms
    test_env = suite.sealed_test_environments["env_sealed_crypto_blocked_bridge"]
    runner = LearningDrivenArmsRunner(
        env=test_env,
        m52_policy=policy_head,
        query_budget=3,
        seed=42,
    )

    for arm in [
        "ARM_A_RANDOM",
        "ARM_B_INFOGAIN",
        "ARM_C_M51_NEURAL",
        "ARM_D_M52_LEARNING_OUTCOME",
        "ARM_E_POLICY_ABLATION",
    ]:
        res = runner.run_arm(arm)
        assert res.validity_rate == 1.00
        assert len(res.queries_executed) == 3
        assert res.total_information_gain >= 0.0
