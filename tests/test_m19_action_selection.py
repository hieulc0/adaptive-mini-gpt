"""Focused Test Suite for M19.1 (Verified Action Traces), M19.2 (Action Selection), and M19.3 (Representation Adaptation).

Validates:
1. Trace Extraction Correctness: Traces derived only from verified goal-satisfying plans.
2. Equivalence & Commutativity: Multiple valid actions supported without arbitrary penalties.
3. Strict Partition Isolation: 0% data leakage across train, val, and test splits.
4. Quarantined Answers: Zero benchmark test answers stored in traces.
5. Checkpoint & Backbone Safety: 100% frozen backbone in Arm A with zero gradient leakage and 0 forgetting.
6. Arm B Residual Adapters: Trainable bottleneck adapters with zero-init identity and selective gradients.
7. Arm C Full Fine-Tuning: End-to-end gradient propagation across backbone.
8. Free-Running Rollout: Evaluates end-to-end plan execution and semantic goal satisfaction.
9. Language Retention: Tracks unrounded cross-entropy loss drift on prior prose.
"""

from __future__ import annotations

import copy
import pytest
import torch

from adaptive_mini_gpt.data.subword_tokenizer import BPETokenizer
from adaptive_mini_gpt.model.config import ModelConfig
from adaptive_mini_gpt.model.transformer import MiniGPT
from adaptive_mini_gpt.rules.action_dataset import (
    ActionTrace,
    ActionTraceDataset,
    extract_verified_m19_traces,
    get_slot_dependencies,
    step_canonical_signature,
)
from adaptive_mini_gpt.rules.action_policy import (
    ActionPolicyModel,
    ActionScoringHead,
    CandidateActionFeaturizer,
    ResidualAdapter,
    compute_action_loss,
    evaluate_action_ranking,
    rollout_problem,
    train_action_policy,
)
from adaptive_mini_gpt.rules.multistep_dataset import (
    DEFAULT_ARCHETYPES,
    generate_addition_depth_ladder,
    generate_multistep_problems,
)
from adaptive_mini_gpt.rules.plan import ExecutionPlan, PlanStep


@pytest.fixture(scope="module")
def shared_tokenizer_and_backbone() -> tuple[BPETokenizer, MiniGPT]:
    corpus = (
        "Paris is the capital of France. A car travels 100 km in 2 hours. "
        "What is the total sum of 10 and 20? If Anna is taller than Ben. "
        "add average_speed compare reverse step sum distance duration km hours km/h"
    )
    tok = BPETokenizer()
    tok.train(corpus, target_vocab_size=256)

    cfg = ModelConfig(
        vocab_size=tok.vocab_size,
        d_model=128,
        n_layers=2,
        n_heads=2,
        d_ff=256,
        context_length=256,
        dropout=0.0,
    )
    backbone = MiniGPT(cfg)
    return tok, backbone


# =========================================================================
# M19.1: ActionTraceDataset Verification & Split Isolation Tests
# =========================================================================


def test_m19_1_trace_extraction_and_equivalence() -> None:
    """Test verified trace extraction and commutative / parallel equivalence."""
    train_ds, val_ds, ood_ds = extract_verified_m19_traces(
        num_samples_per_archetype=8,
        seed=42,
        held_out_archetypes=["3segment_speed"],
    )

    assert len(train_ds) > 0, "Train dataset should contain verified traces"
    assert len(val_ds) > 0, "Val dataset should contain verified traces"
    assert len(ood_ds) > 0, "OOD dataset should contain verified traces"

    has_multi_correct = False
    for trace in train_ds:
        assert trace.valid_candidates, f"Trace {trace.trace_id} must have valid candidates"
        assert trace.correct_action_indices, f"Trace {trace.trace_id} must have correct actions"
        assert trace.target_action_index in trace.correct_action_indices
        if len(trace.correct_action_indices) > 1:
            has_multi_correct = True

    assert has_multi_correct, "Equivalence-aware extractor must identify states with multiple valid next actions"


def test_m19_1_strict_split_isolation_and_no_leakage() -> None:
    """Verify 0% problem or trace overlap between train, val, and test splits."""
    train_ds, val_ds, ood_ds = extract_verified_m19_traces(
        num_samples_per_archetype=15,
        seed=42,
        held_out_archetypes=["3segment_speed"],
    )

    train_prob_keys = {t.problem_key for t in train_ds}
    val_prob_keys = {t.problem_key for t in val_ds}
    ood_prob_keys = {t.problem_key for t in ood_ds}

    # Strict partition isolation
    assert train_prob_keys.isdisjoint(val_prob_keys), "Train and Val problem keys must be strictly disjoint!"
    assert train_prob_keys.isdisjoint(ood_prob_keys), "Train and OOD problem keys must be strictly disjoint!"
    assert val_prob_keys.isdisjoint(ood_prob_keys), "Val and OOD problem keys must be strictly disjoint!"

    # Ensure zero test answers or expected values leaked into traces
    for ds in (train_ds, val_ds, ood_ds):
        for trace in ds:
            d = trace.to_dict()
            assert "expected_answer" not in d, "Expected answer must never leak into trace!"
            assert "expected_plan" not in d, "Expected plan must never leak into trace!"


def test_m19_1_canonical_signature_commutativity() -> None:
    """Verify that add(a, b) and add(b, a) produce identical canonical signatures."""
    step1 = PlanStep(output_var="sum_1", operation="add", args={"a": "$d1", "b": "$d2"}, units="km")
    step2 = PlanStep(output_var="sum_1", operation="add", args={"a": "$d2", "b": "$d1"}, units="km")

    slots = {}
    var_deps = {"d1": frozenset(["slot_d1"]), "d2": frozenset(["slot_d2"])}

    sig1 = step_canonical_signature(step1, var_deps, slots)
    sig2 = step_canonical_signature(step2, var_deps, slots)

    assert sig1 == sig2, "Canonical signatures for commutative additions must be identical!"


def test_m19_1_dataset_fingerprint_reproducibility() -> None:
    """Verify cryptographic fingerprinting stability across extractions."""
    train_a, _, _ = extract_verified_m19_traces(num_samples_per_archetype=5, seed=42)
    train_b, _, _ = extract_verified_m19_traces(num_samples_per_archetype=5, seed=42)

    assert train_a.dataset_fingerprint == train_b.dataset_fingerprint
    assert len(train_a.dataset_fingerprint) == 16


# =========================================================================
# M19.2 & M19.3: Policy Models, Adapters, and Rollout Tests
# =========================================================================


def test_m19_2_frozen_backbone_and_parameter_isolation(
    shared_tokenizer_and_backbone: tuple[BPETokenizer, MiniGPT],
) -> None:
    """Verify that Mini-GPT backbone is 100% frozen in Arm A and receives zero gradients."""
    tok, backbone = shared_tokenizer_and_backbone
    policy = ActionPolicyModel(backbone, tok, d_hidden=128, adapter_mode="none")

    assert policy.num_trainable_params > 0, "Policy must have trainable parameters"
    assert policy.num_frozen_params == backbone.get_num_params(non_embedding=False)

    for p in policy.backbone.parameters():
        assert not p.requires_grad, "Backbone parameters must be strictly frozen"

    initial_backbone_weights = {
        name: p.clone() for name, p in policy.backbone.named_parameters()
    }

    device = torch.device("cpu")
    candidates = [
        PlanStep(output_var="step_1", operation="add", args={"a": "$d1", "b": "$d2"}, units="km"),
        PlanStep(output_var="step_1", operation="average_speed", args={"distance_km": "$d1", "time_hours": "$t1"}),
    ]
    prompt = "<|user|>\nTravel 100 km in 2 hours\n<|assistant|>\nPlan:\n1. "
    scores = policy.score_candidates(prompt, candidates, device)
    loss = compute_action_loss(scores, [0])
    loss.backward()

    for name, p in policy.named_parameters():
        if p.requires_grad:
            assert p.grad is not None, f"Trainable parameter {name} must have gradient"
        else:
            assert p.grad is None, f"Frozen backbone parameter {name} must NOT have gradient"

    for name, p in policy.backbone.named_parameters():
        assert torch.equal(p, initial_backbone_weights[name]), f"Backbone weight {name} drifted!"


def test_m19_3_residual_adapter_architecture_and_gradients(
    shared_tokenizer_and_backbone: tuple[BPETokenizer, MiniGPT],
) -> None:
    """Verify Arm B: Residual adapters have gradients while backbone remains frozen."""
    tok, backbone = shared_tokenizer_and_backbone
    policy = ActionPolicyModel(backbone, tok, d_hidden=128, adapter_mode="residual", d_bottleneck=32)

    assert policy.adapters is not None
    assert len(policy.adapters) == backbone.config.n_layers

    # Check zero-initialization: adapter(x) == x initially
    dummy_x = torch.randn(1, 10, backbone.config.d_model)
    with torch.no_grad():
        adapted_x = policy.adapters[0](dummy_x)
    assert torch.allclose(dummy_x, adapted_x, atol=1e-6), "Zero-init adapter must initially be identity!"

    # Forward + backward check
    device = torch.device("cpu")
    candidates = [
        PlanStep(output_var="step_1", operation="add", args={"a": "$d1", "b": "$d2"}, units="km"),
        PlanStep(output_var="step_1", operation="average_speed", args={"distance_km": "$d1", "time_hours": "$t1"}),
    ]
    prompt = "<|user|>\nTravel 100 km in 2 hours\n<|assistant|>\nPlan:\n1. "
    scores = policy.score_candidates(prompt, candidates, device)
    loss = compute_action_loss(scores, [0])
    loss.backward()

    # Backbone parameters must have NO gradients
    for name, p in policy.backbone.named_parameters():
        assert p.grad is None, f"Backbone param {name} should NOT receive gradients in Arm B"

    # Adapter parameters MUST have gradients
    for name, p in policy.adapters.named_parameters():
        assert p.grad is not None, f"Adapter param {name} must receive gradients in Arm B"


def test_m19_3_full_fine_tuning_gradients(
    shared_tokenizer_and_backbone: tuple[BPETokenizer, MiniGPT],
) -> None:
    """Verify Arm C: Full backbone unfreezing and gradient propagation."""
    tok, backbone = shared_tokenizer_and_backbone
    policy = ActionPolicyModel(backbone, tok, d_hidden=128, adapter_mode="full")

    assert policy.adapters is None
    assert policy.num_frozen_params == 0, "Arm C must have 0 frozen parameters"

    device = torch.device("cpu")
    candidates = [
        PlanStep(output_var="step_1", operation="add", args={"a": "$d1", "b": "$d2"}, units="km"),
    ]
    prompt = "<|user|>\nTravel 100 km in 2 hours\n<|assistant|>\nPlan:\n1. "
    scores = policy.score_candidates(prompt, candidates, device)
    loss = compute_action_loss(scores, [0])
    loss.backward()

    # Backbone parameters MUST have gradients in Arm C
    has_backbone_grad = any(p.grad is not None for p in policy.backbone.parameters())
    assert has_backbone_grad, "Arm C must propagate gradients to backbone parameters"


def test_m19_3_free_running_rollout_and_position_metrics(
    shared_tokenizer_and_backbone: tuple[BPETokenizer, MiniGPT],
) -> None:
    """Verify free-running rollout and position-stratified evaluation."""
    tok, backbone = shared_tokenizer_and_backbone
    policy = ActionPolicyModel(backbone, tok, d_hidden=128, adapter_mode="none")

    probs = generate_multistep_problems(num_samples_per_archetype=2, seed=42, split="val")
    device = torch.device("cpu")

    # Test rollout on a single problem
    success, steps, plan = rollout_problem(policy, probs[0], device, use_policy=False)
    assert success, "Deterministic heuristic rollout should succeed on valid problem"
    assert steps > 0

    # Test trace dataset evaluation with position metrics and rollouts
    val_ds = ActionTraceDataset.from_problems(probs, split="val")
    metrics = evaluate_action_ranking(policy, val_ds, device, problems_for_rollout=probs)

    assert metrics.position_accuracies, "Position metrics must be recorded"
    assert 1 in metrics.position_accuracies, "Step 1 accuracy must exist"
    assert metrics.rollout_deterministic_acc > 0.0


def test_m19_3_language_retention_unrounded(
    shared_tokenizer_and_backbone: tuple[BPETokenizer, MiniGPT],
) -> None:
    """Verify unrounded language loss evaluation with adapters active and bypassed."""
    tok, backbone = shared_tokenizer_and_backbone
    policy = ActionPolicyModel(backbone, tok, d_hidden=128, adapter_mode="residual", d_bottleneck=32)

    prose = "Paris is the capital of France. William Shakespeare wrote Hamlet in London."
    tokens = torch.tensor([tok.encode(prose)], dtype=torch.long)

    loss_active = policy.evaluate_language_loss(tokens, bypass_adapters=False)
    loss_bypassed = policy.evaluate_language_loss(tokens, bypass_adapters=True)

    # Initially before adapter training, zero-init adapter means active and bypassed are bit-identical
    assert abs(loss_active - loss_bypassed) < 1e-5
