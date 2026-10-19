"""Unit and integration tests for M13 Procedural Reasoning Curriculum and Diagnostics."""

from __future__ import annotations

import re
from pathlib import Path

import pytest
import torch

from adaptive_mini_gpt.autonomous import (
    GrowthAction,
    LearningController,
    LearningPlan,
)
from adaptive_mini_gpt.autonomous.tasks import (
    TASK_REGISTRY,
    evaluate_symbolic_diagnostics,
    extract_reversed_answer,
    generate_instruction_symbolic_char_data,
    generate_instruction_symbolic_curriculum_data,
    generate_instruction_symbolic_data,
    generate_instruction_symbolic_stepwise_data,
    get_string_split,
    score_instruction_symbolic_stepwise,
    verify_instruction_symbolic_char,
    verify_instruction_symbolic_curriculum,
    verify_instruction_symbolic_stepwise,
)
from adaptive_mini_gpt.data.subword_tokenizer import BPETokenizer
from adaptive_mini_gpt.model.config import ModelConfig
from adaptive_mini_gpt.model.transformer import MiniGPT


def test_m13_task_registry_registration() -> None:
    """Verify that all M13 procedural curriculum tasks are properly registered with split support."""
    expected_tasks = [
        "instruction_symbolic",
        "instruction_symbolic_char",
        "instruction_symbolic_stepwise",
        "instruction_symbolic_curriculum",
    ]
    for tid in expected_tasks:
        assert tid in TASK_REGISTRY, f"Task {tid} missing from TASK_REGISTRY"
        spec = TASK_REGISTRY[tid]
        assert spec.split_support is True, f"Task {tid} must support split partitioning"
        assert callable(spec.generator_fn)
        assert callable(spec.verifier_fn)


def test_m13_char_arm_split_isolation() -> None:
    """Verify that Arm B (character-delimited) maintains strictly disjoint splits."""
    train_text, train_fp = generate_instruction_symbolic_char_data(
        num_samples=100, seed=42, split="train"
    )
    val_text, val_fp = generate_instruction_symbolic_char_data(
        num_samples=50, seed=43, split="val"
    )
    test_text, test_fp = generate_instruction_symbolic_char_data(
        num_samples=50, seed=44, split="test"
    )

    assert train_fp != val_fp
    assert train_fp != test_fp

    # Extract target characters
    train_targets = set(
        "".join(c.split())
        for c in re.findall(r"Reverse the sequence of characters: ([a-zA-Z0-9\s]+)\n", train_text)
    )
    val_targets = set(
        "".join(c.split())
        for c in re.findall(r"Reverse the sequence of characters: ([a-zA-Z0-9\s]+)\n", val_text)
    )
    test_targets = set(
        "".join(c.split())
        for c in re.findall(r"Reverse the sequence of characters: ([a-zA-Z0-9\s]+)\n", test_text)
    )

    assert len(train_targets) > 0
    assert len(val_targets) > 0
    assert len(test_targets) > 0
    assert len(train_targets.intersection(val_targets)) == 0
    assert len(train_targets.intersection(test_targets)) == 0


def test_m13_stepwise_arm_split_isolation() -> None:
    """Verify that Arm C (stepwise procedure) maintains strictly disjoint splits."""
    train_text, train_fp = generate_instruction_symbolic_stepwise_data(
        num_samples=100, seed=42, split="train"
    )
    test_text, test_fp = generate_instruction_symbolic_stepwise_data(
        num_samples=50, seed=44, split="test"
    )

    assert train_fp != test_fp

    train_words = set(re.findall(r"word '([a-zA-Z0-9]+)'", train_text))
    test_words = set(re.findall(r"word '([a-zA-Z0-9]+)'", test_text))

    assert len(train_words) > 0
    assert len(test_words) > 0
    assert len(train_words.intersection(test_words)) == 0


def test_m13_curriculum_arm_stages_and_isolation() -> None:
    """Verify that Arm D includes progressive stages in train split and proper holdout for test."""
    train_text, train_fp = generate_instruction_symbolic_curriculum_data(
        num_samples=100, seed=42, split="train"
    )
    test_text, test_fp = generate_instruction_symbolic_curriculum_data(
        num_samples=50, seed=44, split="test"
    )

    assert "Copy the sequence" in train_text
    assert "Swap the two characters" in train_text
    assert "Reverse the sequence" in train_text

    # Holdout in test evaluates on target reversal
    assert "Reverse the sequence" in test_text


def test_m13_char_verifier_strictness() -> None:
    """Verify strictness of Arm B character-level verifier."""
    prompt = "<|user|>\nReverse the sequence of characters: c a t\n<|assistant|>\n"

    # Valid completions
    assert verify_instruction_symbolic_char(prompt, "The reversed sequence is: t a c.<|eos|>")
    assert verify_instruction_symbolic_char(prompt, "Result: t a c.")
    assert verify_instruction_symbolic_char(prompt, "t a c")
    assert verify_instruction_symbolic_char(prompt, "tac.<|eos|>")

    # Invalid completions
    assert not verify_instruction_symbolic_char(prompt, "The reversed sequence is: c a t.<|eos|>")
    assert not verify_instruction_symbolic_char(prompt, "t a b")
    assert not verify_instruction_symbolic_char(prompt, "t a c. Actually no, cat.")
    assert not verify_instruction_symbolic_char(prompt, "")
    assert not verify_instruction_symbolic_char(prompt, "vlv")


def test_m13_stepwise_verifier_and_step_scoring() -> None:
    """Verify independent step-level scoring for Arm C explicit procedure."""
    prompt = "<|user|>\nReverse the sequence of characters step-by-step for the word 'photon'.\n<|assistant|>\n"
    good_completion = (
        "Step 1: Read characters: p, h, o, t, o, n\n"
        "Step 2: Reverse index:\n"
        "0: n\n"
        "1: o\n"
        "2: t\n"
        "3: o\n"
        "4: h\n"
        "5: p\n"
        "Result: notohp.<|eos|>"
    )

    scores = score_instruction_symbolic_stepwise(prompt, good_completion)
    assert scores["step1_valid"] is True
    assert scores["step2_valid"] is True
    assert scores["result_valid"] is True
    assert scores["all_valid"] is True
    assert verify_instruction_symbolic_stepwise(prompt, good_completion) is True

    # Corrupt Step 1 (character omission)
    bad_step1 = good_completion.replace("p, h, o, t, o, n", "p, h, o, t")
    s1 = score_instruction_symbolic_stepwise(prompt, bad_step1)
    assert s1["step1_valid"] is False
    assert s1["step2_valid"] is True
    assert s1["result_valid"] is True
    assert s1["all_valid"] is False

    # Corrupt Step 2 (wrong index mapping)
    bad_step2 = good_completion.replace("2: t", "2: z")
    s2 = score_instruction_symbolic_stepwise(prompt, bad_step2)
    assert s2["step1_valid"] is True
    assert s2["step2_valid"] is False
    assert s2["result_valid"] is True
    assert s2["all_valid"] is False

    # Corrupt Result
    bad_result = good_completion.replace("Result: notohp.", "Result: photon.")
    s3 = score_instruction_symbolic_stepwise(prompt, bad_result)
    assert s3["step1_valid"] is True
    assert s3["step2_valid"] is True
    assert s3["result_valid"] is False
    assert s3["all_valid"] is False


def test_m13_curriculum_verifier_cases() -> None:
    """Verify that the curriculum verifier correctly handles copy, swap, and reversal."""
    # Copy
    p_copy = "<|user|>\nCopy the sequence of characters: apple\n<|assistant|>\n"
    assert verify_instruction_symbolic_curriculum(p_copy, "The copied sequence is: apple.<|eos|>")
    assert not verify_instruction_symbolic_curriculum(p_copy, "The copied sequence is: elppa.<|eos|>")

    # Swap
    p_swap = "<|user|>\nSwap the two characters: x y\n<|assistant|>\n"
    assert verify_instruction_symbolic_curriculum(p_swap, "The swapped sequence is: y x.<|eos|>")
    assert not verify_instruction_symbolic_curriculum(p_swap, "The swapped sequence is: x y.<|eos|>")

    # Reverse
    p_rev = "<|user|>\nReverse the sequence of characters in the word 'guitar'.\n<|assistant|>\n"
    assert verify_instruction_symbolic_curriculum(p_rev, "The reversed sequence is: ratiug.<|eos|>")
    assert not verify_instruction_symbolic_curriculum(p_rev, "The reversed sequence is: guitar.<|eos|>")


def test_m13_answer_extraction_and_diagnostics() -> None:
    """Verify answer extractors and multi-dimensional diagnostic metrics."""
    assert extract_reversed_answer("The reversed sequence is: notohp.<|eos|>", "direct") == "notohp"
    assert extract_reversed_answer("The reversed sequence is: n o t o h p.<|eos|>", "char") == "notohp"
    assert (
        extract_reversed_answer("Step 1...\nStep 2...\nResult: notohp.<|eos|>", "stepwise") == "notohp"
    )

    # Smoke test evaluate_symbolic_diagnostics with a small MiniGPT model
    tok = BPETokenizer()
    tok.train(
        "abcdefghijklmnopqrstuvwxyz 0123456789 <|user|> <|assistant|> <|eos|> : ' . \n",
        target_vocab_size=120,
    )
    cfg = ModelConfig(
        vocab_size=tok.vocab_size,
        d_model=32,
        n_layers=2,
        n_heads=2,
        d_ff=64,
        context_length=128,
    )
    model = MiniGPT(cfg)

    device = torch.device("cpu")
    diags = evaluate_symbolic_diagnostics(
        model=model,
        tokenizer=tok,
        device=device,
        arm="direct",
        num_samples_per_cat=3,
        seed=123,
    )

    assert "unseen_in_distribution_len_3_6" in diags
    assert "ood_extrapolation_len_7_8" in diags
    d3_6 = diags["unseen_in_distribution_len_3_6"]
    assert "exact_match_acc" in d3_6
    assert "per_char_acc" in d3_6
    assert "length_match_acc" in d3_6
    assert "termination_acc" in d3_6
    assert "output_diversity" in d3_6


def test_m13_controller_plan_and_replay_support(tmp_path: Path) -> None:
    """Verify that LearningPlan and controller support strategy and experience replay."""
    plan = LearningPlan(
        task_id="instruction_symbolic",
        growth_action="continue",
        budget_steps=100,
        learning_rate=1e-3,
        batch_size=8,
        data_samples=50,
        seed=42,
        strategy="curriculum",
        replay_ratio=0.25,
    )
    assert plan.strategy == "curriculum"
    assert plan.replay_ratio == 0.25
