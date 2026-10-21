"""Unit tests for M21 Digit-Level Neural Arithmetic and Procedural Scratchpad."""

from __future__ import annotations

import pytest
import torch

from adaptive_mini_gpt.data.subword_tokenizer import BPETokenizer
from adaptive_mini_gpt.model.config import ModelConfig
from adaptive_mini_gpt.model.transformer import MiniGPT
from adaptive_mini_gpt.rules.arithmetic_dataset import (
    ArithmeticDataset,
    ArithmeticProblem,
    ColumnCalculation,
    collate_arithmetic_batch,
    compute_dataset_fingerprint,
    format_problem,
    generate_controlled_addition_dataset,
    parse_arithmetic_completion,
)
from adaptive_mini_gpt.rules.arithmetic_evaluator import (
    evaluate_arithmetic_unaided,
    train_arithmetic_model,
)


def test_arithmetic_problem_creation_and_column_math() -> None:
    """Verify column-by-column carry decomposition and math properties."""
    # 127 + 386 = 513
    # Col 0 (units): 7 + 6 + 0 = 13 -> write 3, carry 1
    # Col 1 (tens): 2 + 8 + 1 = 11 -> write 1, carry 1
    # Col 2 (hundreds): 1 + 3 + 1 = 5 -> write 5, carry 0
    p = ArithmeticProblem.create(127, 386)
    assert p.result == 513
    assert p.num_carries == 2
    assert len(p.columns) == 3

    c0 = p.columns[0]
    assert c0.digit_a == 7 and c0.digit_b == 6 and c0.carry_in == 0
    assert c0.total_sum == 13 and c0.write_digit == 3 and c0.carry_out == 1

    c1 = p.columns[1]
    assert c1.digit_a == 2 and c1.digit_b == 8 and c1.carry_in == 1
    assert c1.total_sum == 11 and c1.write_digit == 1 and c1.carry_out == 1

    c2 = p.columns[2]
    assert c2.digit_a == 1 and c2.digit_b == 3 and c2.carry_in == 1
    assert c2.total_sum == 5 and c2.write_digit == 5 and c2.carry_out == 0

    # Overflow carry test: 99 + 1 = 100
    p_ov = ArithmeticProblem.create(99, 1)
    assert p_ov.result == 100
    assert p_ov.num_carries == 2
    assert len(p_ov.columns) == 3
    assert p_ov.columns[-1].write_digit == 1


def test_format_problem_across_four_arms() -> None:
    """Verify prompt and target generation for Arms A, B, C, and D."""
    p = ArithmeticProblem.create(127, 386)

    # Arm A: Standard
    p_a, t_a = format_problem(p, "standard")
    assert "Add: 127 + 386" in p_a
    assert "Answer: " in p_a
    assert t_a.strip().startswith("513")

    # Arm B: Digit-Level MSD
    p_b, t_b = format_problem(p, "digit_msd")
    assert "Add: 1 2 7 + 3 8 6" in p_b
    assert t_b.strip().startswith("5 1 3")

    # Arm C: Scratchpad
    p_c, t_c = format_problem(p, "scratchpad")
    assert "Scratchpad:" in p_c
    assert "col 0 (units): 7 + 6 + 0 = 13 | write 3 | carry 1" in t_c
    assert "col 1 (tens): 2 + 8 + 1 = 11 | write 1 | carry 1" in t_c
    assert "col 2 (hundreds): 1 + 3 + 1 = 5 | write 5 | carry 0" in t_c
    assert "Answer: 5 1 3" in t_c

    # Arm D: Digit-Level LSD
    p_d, t_d = format_problem(p, "digit_lsd")
    assert "Add (LSD): 7 2 1 + 6 8 3" in p_d
    assert t_d.strip().startswith("3 1 5")


def test_parse_arithmetic_completion() -> None:
    """Test parsing completions for all arms."""
    # Standard
    res_a = parse_arithmetic_completion("Answer: 513\n<|eos|>", "standard")
    assert res_a["is_valid"] is True
    assert res_a["predicted_answer"] == "513"

    # Digit MSD
    res_b = parse_arithmetic_completion("Answer: 5 1 3\n<|eos|>", "digit_msd")
    assert res_b["is_valid"] is True
    assert res_b["predicted_answer"] == "513"

    # Scratchpad
    raw_c = (
        "col 0 (units): 7 + 6 + 0 = 13 | write 3 | carry 1\n"
        "col 1 (tens): 2 + 8 + 1 = 11 | write 1 | carry 1\n"
        "col 2 (hundreds): 1 + 3 + 1 = 5 | write 5 | carry 0\n"
        "Answer: 5 1 3\n<|eos|>"
    )
    res_c = parse_arithmetic_completion(raw_c, "scratchpad")
    assert res_c["is_valid"] is True
    assert res_c["predicted_answer"] == "513"
    assert res_c["predicted_carries"] == [1, 1, 0]

    # Digit LSD
    res_d = parse_arithmetic_completion("Answer: 3 1 5\n<|eos|>", "digit_lsd")
    assert res_d["is_valid"] is True
    assert res_d["predicted_answer"] == "513"


def test_controlled_addition_dataset_generation_and_uniqueness() -> None:
    """Verify unique sample generation, carry distribution, and fingerprinting."""
    probs = generate_controlled_addition_dataset(seed=42, num_samples=40, lengths=(2, 3))
    assert len(probs) == 40

    # Ensure zero duplicates
    pairs = set((p.a, p.b) for p in probs)
    assert len(pairs) == 40

    # Ensure carry distribution contains carries 0, 1, 2, 3
    carries = set(p.num_carries for p in probs)
    assert len(carries) >= 3

    fp1 = compute_dataset_fingerprint(probs)
    fp2 = compute_dataset_fingerprint(probs)
    assert fp1 == fp2
    assert len(fp1) == 16


def test_target_only_loss_masking_arithmetic() -> None:
    """Verify that prompt tokens are masked with -100 in target labels."""
    tokenizer = BPETokenizer()
    tokenizer.train("Add: 1 2 7 + 3 8 6 Answer: 5 1 3", target_vocab_size=100)

    p = ArithmeticProblem.create(12, 34)
    ds = ArithmeticDataset([p], tokenizer, arm="digit_msd", context_length=128)
    inp, lab = ds[0]

    prompt_str, _ = format_problem(p, "digit_msd")
    p_len = len(tokenizer.encode(prompt_str))

    assert (lab[:p_len] == -100).all()
    assert (lab[p_len:] != -100).all()
    assert (inp[p_len:] == lab[p_len:]).all()


def test_arithmetic_model_forward_and_gradient() -> None:
    """Verify that MiniGPT computes valid cross entropy loss and finite gradients on arithmetic batch."""
    cfg = ModelConfig(vocab_size=100, d_model=32, n_layers=2, n_heads=2, d_ff=64, context_length=64)
    model = MiniGPT(cfg)

    inp = torch.randint(0, 100, (2, 32))
    lab = inp.clone()
    lab[:, :10] = -100  # mask prompt

    logits, _ = model(inp)
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = lab[:, 1:].contiguous()
    loss = torch.nn.functional.cross_entropy(shift_logits.view(-1, cfg.vocab_size), shift_labels.view(-1), ignore_index=-100)

    assert torch.isfinite(loss)
    loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters() if p.requires_grad)
