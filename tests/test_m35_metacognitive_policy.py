"""Unit tests for M35: Learning How to Learn (Metacognitive Policy & Self-Directed Learning)."""

import pytest
import torch

from adaptive_mini_gpt.metacognition.action_space import (
    ACTION_LIST,
    LearningAction,
    compute_action_utility,
)
from adaptive_mini_gpt.metacognition.curriculum import (
    MetacognitiveSample,
    build_metacognitive_curriculum,
)
from adaptive_mini_gpt.metacognition.language_diagnosis import (
    diagnose_language_representation_gap,
)
from adaptive_mini_gpt.metacognition.learning_controller import (
    ControlledSelfDirectedLearner,
)
from adaptive_mini_gpt.metacognition.policy_head import (
    ACTION_TO_IDX,
    NeuralMetacognitivePolicy,
    train_outcome_aware_policy,
    train_supervised_policy,
)
from adaptive_mini_gpt.model.transformer import MiniGPT, ModelConfig


def test_action_space_and_utility_scoring() -> None:
    """Test the 7 metacognitive actions and empirical utility scoring."""
    assert len(ACTION_LIST) == 7
    assert LearningAction.ANSWER in ACTION_LIST
    assert LearningAction.TRAIN_CANDIDATE in ACTION_LIST
    assert LearningAction.ABSTAIN in ACTION_LIST

    # 1. Successful answer -> high positive utility
    u_success = compute_action_utility(LearningAction.ANSWER, task_solved=True)
    assert u_success > 0.9

    # 2. Safe abstention on unanswerable problem -> positive utility
    u_safe_abstain = compute_action_utility(
        LearningAction.ABSTAIN,
        task_solved=True,
        is_ground_truth_unanswerable=True,
    )
    assert u_safe_abstain > 0.9

    # 3. False abstention on answerable problem -> penalty
    u_false_abstain = compute_action_utility(
        LearningAction.ABSTAIN,
        task_solved=False,
        is_ground_truth_unanswerable=False,
    )
    assert u_false_abstain < 0.0

    # 4. Unnecessary training action -> penalty
    u_unneeded_train = compute_action_utility(
        LearningAction.TRAIN_CANDIDATE,
        task_solved=True,
        is_repeated_procedural_failure=False,
    )
    assert u_unneeded_train < 0.0

    # 5. Regression detected -> severe penalty
    u_regression = compute_action_utility(
        LearningAction.TRAIN_CANDIDATE,
        task_solved=True,
        skill_regression_detected=True,
    )
    assert u_regression <= -2.0


def test_neural_metacognitive_policy_mechanics() -> None:
    """Test policy head forward, output shapes, confidence, and normalized entropy uncertainty."""
    torch.manual_seed(42)
    cfg = ModelConfig(
        vocab_size=64,
        context_length=32,
        d_model=32,
        n_layers=2,
        n_heads=2,
        d_ff=64,
    )
    model = MiniGPT(cfg)
    policy = NeuralMetacognitivePolicy(d_model=32, d_hidden=16, num_actions=7)

    dummy_input = torch.randint(0, 64, (1, 8))
    pred_act, conf, uncert, probs = policy.predict(model, dummy_input)

    assert isinstance(pred_act, LearningAction)
    assert 0.0 <= conf <= 1.0
    assert 0.0 <= uncert <= 1.0
    assert torch.isclose(probs.sum(), torch.tensor(1.0), atol=1e-4)


def test_policy_supervised_and_outcome_aware_training() -> None:
    """Test supervised and outcome-aware policy head optimization."""
    torch.manual_seed(42)
    policy = NeuralMetacognitivePolicy(d_model=16, d_hidden=16, num_actions=7)
    features = torch.randn(10, 16)
    targets = torch.randint(0, 7, (10,))
    utilities = torch.tensor([0.9, -0.8, 0.5, -0.2, 0.8, -0.9, 0.4, 0.9, -0.5, 0.7])

    # Supervised training
    losses_sup = train_supervised_policy(policy, features, targets, epochs=5, lr=0.01)
    assert len(losses_sup) == 5
    assert losses_sup[-1] < losses_sup[0]

    # Outcome-aware training
    policy_outcome = NeuralMetacognitivePolicy(d_model=16, d_hidden=16, num_actions=7)
    losses_out = train_outcome_aware_policy(
        policy_outcome, features, targets, utilities, epochs=5, lr=0.01, utility_weight=0.5
    )
    assert len(losses_out) == 5
    assert losses_out[-1] < losses_out[0]


def test_metacognitive_curriculum_disjointness() -> None:
    """Verify that curriculum partitions across all 7 scenarios are 100% disjoint."""
    train_set, eval_set = build_metacognitive_curriculum(
        seed=42,
        train_samples_per_scenario=6,
        eval_samples_per_scenario=4,
    )
    assert len(train_set) == 42  # 7 scenarios * 6
    assert len(eval_set) == 28   # 7 scenarios * 4

    train_queries = {s.query for s in train_set}
    eval_queries = {s.query for s in eval_set}

    # Intersections must be empty
    overlap = train_queries.intersection(eval_queries)
    assert len(overlap) == 0, f"Found overlapping queries: {overlap}"


def test_controlled_self_directed_learner_execution() -> None:
    """Test closed-loop interactive controller gating, budget enforcement, and execution."""
    torch.manual_seed(42)
    cfg = ModelConfig(
        vocab_size=64,
        context_length=32,
        d_model=32,
        n_layers=2,
        n_heads=2,
        d_ff=64,
    )
    model = MiniGPT(cfg)
    policy = NeuralMetacognitivePolicy(d_model=32, d_hidden=16, num_actions=7)

    class DummyTok:
        vocab_size = 64
        eos_id = 0
        def encode(self, s: str) -> list[int]:
            return [1, 2, 3]
        def decode(self, ids: list[int]) -> str:
            return "42"

    learner = ControlledSelfDirectedLearner(
        model=model,
        tokenizer=DummyTok(),
        policy=policy,
        device=torch.device("cpu"),
        memory_bank={"mitochondria": "Mitochondria generate cellular ATP."},
        max_training_budget=1,
    )

    # Test budget override: if budget exhausted, TRAIN_CANDIDATE -> PRACTICE
    learner.training_budget_spent = 1
    action_gated, overrode, reason = learner.validate_and_gate_action(
        LearningAction.TRAIN_CANDIDATE, uncertainty=0.2
    )
    assert overrode is True
    assert action_gated == LearningAction.PRACTICE

    # Test extreme uncertainty override (>0.95 -> ABSTAIN)
    action_uncert, overrode_uncert, _ = learner.validate_and_gate_action(
        LearningAction.ANSWER, uncertainty=0.98
    )
    assert overrode_uncert is True
    assert action_uncert == LearningAction.ABSTAIN
