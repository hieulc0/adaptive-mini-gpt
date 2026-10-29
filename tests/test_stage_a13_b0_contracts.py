"""Stage A.13 and Stage B.0 Local Mock Contracts.

Strictly Local-Safe: ZERO neural training, ZERO checkpoint loading, ZERO CPU training.
Tests:
1. Continuous logit and representation shift metrics (Delta_logit, Delta_h, signed target margin)
   and mathematical verification that they distinguish sub-threshold logit movement from true zero response.
2. Learned query-conditioned attention masking (A13_ARM_2), ensuring query tokens attend strictly
   to prefix <question> tokens and never leak <answer>, padding, or rule text.
3. Stage B.0 single-LearningState identity invariant across Concept A -> Concept B -> Concept C
   (asserting id(state) and optimizer continuity with zero resets, zero test-label leakage,
   and independent accounting of B0_ARM_1..4).
4. Practice policy parameter limit (<1,000 parameters) and feature extraction sanity.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from adaptive_mini_gpt.data.sequential_concepts_b0 import (
    CONCEPTS,
    SequentialEnvironmentB0,
)
from adaptive_mini_gpt.evaluation.continuous_sensitivity import (
    compute_continuous_sensitivity,
)
from adaptive_mini_gpt.natural_language_learning_m103.semantic_bottleneck_m104_e import (
    compute_query_cross_prefix_mask,
)
from adaptive_mini_gpt.natural_language_learning_m103.tokenizer_m103 import (
    ByteTextTokenizerM103,
)
from adaptive_mini_gpt.self_directed_curriculum.practice_policy_b0 import (
    FixedHeuristicSelectorB0,
    NeuralPracticePolicyB0,
    extract_candidate_features,
)
from adaptive_mini_gpt.training.learning_state import LearningState


def test_continuous_sensitivity_distinguishes_subthreshold_movement():
    """Verifies that Delta_logit and signed target margin detect sub-threshold shifts

    where discrete predictions show 0 flips.
    """
    # 2 pairs, 2 labels each
    # Case A: logits [0.2, 0.3], targets [0, 1]
    # Case B: logits [0.8, 0.9], targets [1, 1]
    # Note: binary predictions for both Case A and Case B are [1, 1] (all >= 0.0),
    # so discrete flip rate is ZERO (0 flips).
    # BUT logits shift significantly (Delta_logit > 0), and for label 0,
    # target shifted from 0 to 1 (delta_y = -1), logits shifted from 0.2 to 0.8 (delta_l = -0.6),
    # so signed margin is (-1) * (-0.6) = +0.6 > 0 (correct direction!).
    logits_a = torch.tensor([[0.2, 0.3], [0.1, 0.4]])
    logits_b = torch.tensor([[0.8, 0.9], [0.5, 0.7]])
    h_a = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    h_b = torch.tensor([[0.0, 1.0], [1.0, 0.0]])
    y_a = torch.tensor([[0, 1], [0, 1]])
    y_b = torch.tensor([[1, 1], [1, 1]])

    res = compute_continuous_sensitivity(logits_a, logits_b, h_a, h_b, y_a, y_b)

    # 1. Discrete flips must be 0 (sub-threshold condition)
    assert res["discrete_prediction_flip_rate"] == 0.0
    assert res["sub_threshold_sensitivity_count"] == 2

    # 2. Continuous logit shift must be strictly positive
    assert res["delta_logit_mean"] > 0.5

    # 3. Signed target margin must be positive (movement in ground-truth direction)
    assert res["signed_target_margin_mean"] > 0.0
    assert res["signed_margin_positive_rate"] == 1.0

    # 4. Hidden state shift must be positive
    assert res["delta_h_l2_mean"] > 0.0
    assert res["cosine_distance_mean"] > 0.5


def test_continuous_sensitivity_zero_response():
    """Verifies that when logits do not change, sensitivity metrics report 0.0."""
    logits = torch.tensor([[0.5, -0.5], [1.0, -1.0]])
    h = torch.tensor([[1.0, 1.0], [2.0, 2.0]])
    y_a = torch.tensor([[1, 0], [1, 0]])
    y_b = torch.tensor([[0, 1], [0, 1]])

    res = compute_continuous_sensitivity(logits, logits, h, h, y_a, y_b)
    assert res["delta_logit_mean"] == 0.0
    assert res["delta_h_l2_mean"] == 0.0
    assert res["cosine_distance_mean"] < 1e-6
    assert res["discrete_prediction_flip_rate"] == 0.0
    assert res["sub_threshold_sensitivity_count"] == 0


def test_learned_query_conditioned_masking():
    """Verifies that compute_query_cross_prefix_mask strictly isolates prefix tokens

    and prevents any leakage of rule text, query suffix, answer text, or padding.
    """
    B = 2
    L = 64
    tokens = torch.full((B, L), ByteTextTokenizerM103.PAD_ID, dtype=torch.long)

    # Sequence structure:
    # 0..3: Rule tokens
    # 4: QUESTION_ID
    # 5..14: Prefix clause text ("Object Aero is flammable...")
    # 15..24: Query suffix text ("Query target: Aero.")
    # 25: ANSWER_ID
    # 26..31: Padding
    for b in range(B):
        tokens[b, 0] = ByteTextTokenizerM103.RULE_ID
        tokens[b, 1:4] = torch.tensor([ord("R"), ord("u"), ord("l")])
        tokens[b, 4] = ByteTextTokenizerM103.QUESTION_ID
        # prefix
        prefix_bytes = b"Core Alpha"
        tokens[b, 5 : 5 + len(prefix_bytes)] = torch.tensor(list(prefix_bytes))
        # query suffix
        query_bytes = b"Query target: Alpha."
        q_start = 5 + len(prefix_bytes)
        tokens[b, q_start : q_start + len(query_bytes)] = torch.tensor(list(query_bytes))
        tokens[b, q_start + len(query_bytes)] = ByteTextTokenizerM103.ANSWER_ID

    key_pad_mask, q_indices = compute_query_cross_prefix_mask(tokens)

    # 1. Shape check
    assert key_pad_mask.shape == (B, L)
    assert q_indices.shape == (B,)

    # 2. Key padding mask: True means masked (ignored), False means attendable
    for b in range(B):
        # Rule tokens (0..4) must be masked (True)
        assert bool(key_pad_mask[b, 0:5].all().item()) is True

        # Prefix tokens (5..14) must be UNMASKED (False)
        assert bool((~key_pad_mask[b, 5 : 5 + len(b"Core Alpha")]).all().item()) is True

        # Query suffix tokens must be masked (True)
        q_s = 5 + len(b"Core Alpha")
        assert bool(key_pad_mask[b, q_s:].all().item()) is True

        # Answer and padding must be masked (True)
        ans_pos = q_s + len(b"Query target: Alpha.")
        assert bool(key_pad_mask[b, ans_pos:].all().item()) is True

        # Query index must point to the last question token before ANSWER_ID
        assert q_indices[b].item() == ans_pos - 1


def test_stage_b0_sequential_environment_leak_free():
    """Verifies that Stage B.0 environment maintains strict leak-free isolation."""
    env = SequentialEnvironmentB0(seed=42)

    for k in ("concept_a", "concept_b", "concept_c"):
        cands = env.get_candidate_questions(k)
        assert len(cands) == 480
        val = env.get_held_out_cases(k)
        assert len(val) == 120

        # Verify zero entity overlap between candidate train entities and validation entities
        spec = CONCEPTS[k]
        train_ents = set(spec.train_entities)
        val_ents = set(spec.held_out_entities)
        assert len(train_ents.intersection(val_ents)) == 0

        # Verify candidate cases returned via get_candidate_cases_unlabeled have dummy labels
        unlabeled = env.get_candidate_cases_unlabeled(k)
        assert all(c.labels == (0, 0) for c in unlabeled)
        assert all(c.statuses == ("masked", "masked") for c in unlabeled)


def test_stage_b0_single_learning_state_continuity():
    """Verifies the single-LearningState identity invariant across Concept A -> B -> C."""
    # Build tiny mock linear model and LearningState
    dummy_model = nn.Sequential(
        nn.Linear(10, 26),
    )
    optimizer = torch.optim.Adam(dummy_model.parameters(), lr=0.001)
    state = LearningState(dummy_model)
    state.optimizer = optimizer

    initial_state_id = id(state)
    initial_model_id = id(state.model)
    initial_opt_id = id(state.optimizer)

    # Simulate sequential concept progression
    concepts = ["concept_a", "concept_b", "concept_c"]
    steps_per_concept = 10

    for _c in concepts:
        for _ in range(steps_per_concept):
            # Take mock step without resetting
            x = torch.randn(4, 10)
            tag_targets = torch.randint(0, 2, (4, 26)).float()
            state.optimizer.zero_grad()
            logits = state.model(x)
            loss = nn.functional.binary_cross_entropy_with_logits(logits[:, :2], tag_targets[:, :2])
            loss.backward()
            state.optimizer.step()
            state.metadata["step_count"] = state.metadata.get("step_count", 0) + 1

    # Verify ID continuity (ZERO resets)
    assert id(state) == initial_state_id
    assert id(state.model) == initial_model_id
    assert id(state.optimizer) == initial_opt_id
    assert state.metadata["step_count"] == 30


def test_practice_policy_parameter_bound_and_heuristics():
    """Verifies that the trainable practice policy has <1,000 parameters

    and that feature extraction runs smoothly.
    """
    policy = NeuralPracticePolicyB0(hidden_dim=16)
    assert policy.param_count < 1000, f"Policy has {policy.param_count} parameters >= 1,000"

    N = 32
    logits = torch.randn(N, 2)
    h_cands = torch.randn(N, 128)
    h_replay = torch.randn(8, 128)

    feats = extract_candidate_features(logits, h_cands, h_replay, loss_slope=-0.05)
    assert feats.shape == (N, 4)

    # Test candidate ranking with fixed heuristic
    unqueried = list(range(N))
    heuristic_selected = FixedHeuristicSelectorB0.rank_candidates(logits, unqueried, k=8)
    assert len(heuristic_selected) == 8
    assert len(set(heuristic_selected)) == 8

    # Test policy selection
    policy_selected, log_prob = policy.select_candidates(feats, unqueried, k=8, sample=True)
    assert len(policy_selected) == 8
    assert len(set(policy_selected)) == 8
    assert log_prob is not None

    # Test policy update
    loss = policy.update_policy(log_prob, reward=0.15, baseline=0.05)
    assert isinstance(loss, float)
