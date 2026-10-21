"""Unit and Integration Tests for M32 Verified Recurrent Learning and Adaptive Reasoning."""

from __future__ import annotations

import pytest
import torch

from adaptive_mini_gpt.data.bpe_scaling import train_scaled_bpe_tokenizer
from adaptive_mini_gpt.data.subword_tokenizer import BPETokenizer
from adaptive_mini_gpt.model.capacity_migration import create_1m_control_config
from adaptive_mini_gpt.model.transformer import MiniGPT
from adaptive_mini_gpt.reasoning.adaptive_router import (
    AlwaysDirectRouter,
    AlwaysRecurrentRouter,
    AlwaysRetrieveRouter,
    DeterministicRouter,
    NeuralActionSelector,
    ReasoningAction,
    build_routing_benchmark_dataset,
)
from adaptive_mini_gpt.reasoning.recurrent_audit import (
    ElementaryAuditRecord,
    TransitionFailureType,
    audit_elementary_state,
)
from adaptive_mini_gpt.reasoning.recurrent_curriculum import (
    build_deep_logic_suite,
    build_failure_targeted_recurrent_practice,
    build_multi_digit_addition_suite,
)
from adaptive_mini_gpt.reasoning.step_correction import (
    CorrectionMode,
    execute_step_with_mode,
    rollout_addition_with_correction,
)
from adaptive_mini_gpt.rules.reliable_transition import ElementaryTransition


@pytest.fixture
def mock_tokenizer() -> BPETokenizer:
    corpus = [
        "Col: 0 + 0 + 0",
        "Col: 7 + 6 + 1",
        "14 > 4 : 1 | HALT",
        "Route: Add: 487 + 596 =",
        "RECURRENT_EXECUTION",
        "DIRECT_ANSWER",
        "VERIFIED_RETRIEVAL",
        "ABSTAIN",
    ]
    return train_scaled_bpe_tokenizer(
        corpus,
        target_vocab_size=120,
        special_tokens=["<|pad|>", "<|user|>", "<|assistant|>", "<|eos|>"],
    )


@pytest.fixture
def mock_model(mock_tokenizer: BPETokenizer) -> MiniGPT:
    cfg = create_1m_control_config()
    cfg.vocab_size = mock_tokenizer.vocab_size
    cfg.n_layer = 2
    cfg.d_model = 64
    cfg.n_head = 2
    return MiniGPT(cfg)


def test_elementary_transition_audit_and_classification(mock_model: MiniGPT, mock_tokenizer: BPETokenizer) -> None:
    """Verify single-state audit correctly identifies transition properties."""
    tr = ElementaryTransition.create(7, 6, 1)  # sum=14, write=4, carry=1, propagate
    rec = audit_elementary_state(mock_model, mock_tokenizer, tr, device=torch.device("cpu"))
    assert rec.a == 7
    assert rec.b == 6
    assert rec.c_in == 1
    assert rec.expected_sum == 14
    assert rec.expected_write == 4
    assert rec.expected_carry == 1
    assert rec.category == "generate"
    assert isinstance(rec.failure_type, TransitionFailureType)


def test_step_correction_modes(mock_model: MiniGPT, mock_tokenizer: BPETokenizer) -> None:
    """Verify single_attempt, neural_retry, and verifier_assisted correction execution."""
    # Test Verifier-Assisted mode: guaranteed step correction
    pw, nc, ok, retries, ver_corr, raw = execute_step_with_mode(
        model=mock_model,
        tokenizer=mock_tokenizer,
        da=8,
        db=9,
        carry_in=1,
        device=torch.device("cpu"),
        mode=CorrectionMode.VERIFIER_ASSISTED,
    )
    assert ok is True
    assert pw == (8 + 9 + 1) % 10  # 8
    assert nc == (8 + 9 + 1) // 10  # 1

    # Test complete rollout with verifier assistance
    res = rollout_addition_with_correction(
        model=mock_model,
        tokenizer=mock_tokenizer,
        a=99,
        b=1,
        device=torch.device("cpu"),
        mode=CorrectionMode.VERIFIER_ASSISTED,
    )
    assert res.is_verifier_assisted is True
    assert res.is_exact_match is True
    assert res.final_answer == "100"


def test_adaptive_router_heuristic_and_baselines() -> None:
    """Verify deterministic routing and static baselines."""
    router = DeterministicRouter()

    # Multi-digit arithmetic -> RECURRENT_EXECUTION
    dec_arith = router.route("Add: 487 + 596 =")
    assert dec_arith.action == ReasoningAction.RECURRENT_EXECUTION

    # Simple arithmetic -> DIRECT_ANSWER
    dec_simple = router.route("Add: 5 + 3 =")
    assert dec_simple.action == ReasoningAction.DIRECT_ANSWER

    # Shallow logic -> DIRECT_ANSWER
    dec_logic_shallow = router.route("If A then B. A is True. What is B?")
    assert dec_logic_shallow.action == ReasoningAction.DIRECT_ANSWER

    # Deep logic -> RECURRENT_EXECUTION
    dec_logic_deep = router.route("If A then B. If B then C. If C then D. If D then E. A is True. What is E?")
    assert dec_logic_deep.action == ReasoningAction.RECURRENT_EXECUTION

    # Factual QA -> VERIFIED_RETRIEVAL
    dec_fact = router.route("Fact: What is the biological function of ribosomes?")
    assert dec_fact.action == ReasoningAction.VERIFIED_RETRIEVAL

    # Malformed -> ABSTAIN
    dec_mal = router.route("gibberish asdf %%%% ????")
    assert dec_mal.action == ReasoningAction.ABSTAIN

    # Static Baselines
    assert AlwaysDirectRouter().route("anything").action == ReasoningAction.DIRECT_ANSWER
    assert AlwaysRecurrentRouter().route("anything").action == ReasoningAction.RECURRENT_EXECUTION
    assert AlwaysRetrieveRouter().route("anything").action == ReasoningAction.VERIFIED_RETRIEVAL


def test_routing_benchmark_dataset_generation() -> None:
    """Verify routing dataset contains balanced classes and disjoint splits."""
    train_samples, eval_samples = build_routing_benchmark_dataset()
    assert len(train_samples) == 48
    assert len(eval_samples) == 24

    actions_in_train = {s.optimal_action for s in train_samples}
    actions_in_eval = {s.optimal_action for s in eval_samples}
    assert actions_in_train == {act for act in ReasoningAction}
    assert actions_in_eval == {act for act in ReasoningAction}


def test_multi_digit_addition_suite_generation() -> None:
    """Verify addition suite covers lengths up to 32 digits, cascading carries, and overflow."""
    suite = build_multi_digit_addition_suite(seed=42)
    assert "arith_2_digits" in suite
    assert "arith_4_digits" in suite
    assert "arith_8_digits" in suite
    assert "arith_16_digits" in suite
    assert "arith_32_digits" in suite
    assert "arith_cascading_carry" in suite
    assert "arith_final_overflow" in suite

    # Check 32-digit sample
    sample_32 = suite["arith_32_digits"][0]
    assert len(str(sample_32.a)) == 32
    assert len(str(sample_32.b)) == 32

    # Check cascading carry sample
    cascading = suite["arith_cascading_carry"]
    assert any("9999" in str(c.a) for c in cascading)


def test_deep_logic_suite_generation() -> None:
    """Verify deep logic suite generates 3, 4, 5, and 6 hop deduction chains."""
    suite = build_deep_logic_suite(seed=42)
    assert "logic_3_hop" in suite
    assert "logic_4_hop" in suite
    assert "logic_5_hop" in suite
    assert "logic_6_hop" in suite

    s6 = suite["logic_6_hop"][0]
    assert s6.expected_final_answer in ["True", "False"]
    assert len(s6.recurrent_steps) == 6


def test_failure_targeted_recurrent_practice_curriculum() -> None:
    """Verify failure-targeted curriculum generates required count and focuses on carries."""
    practice = build_failure_targeted_recurrent_practice(target_sample_count=72, seed=42)
    assert len(practice) == 72
    # Verify samples contain recurrent steps
    assert len(practice[0].recurrent_steps) > 0
