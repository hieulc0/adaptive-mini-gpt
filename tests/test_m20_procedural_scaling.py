"""Unit tests for M20 Neural Capacity and Procedural Scaling."""

from __future__ import annotations

import pytest
import torch

from adaptive_mini_gpt.data.subword_tokenizer import BPETokenizer
from adaptive_mini_gpt.model.config import ModelConfig
from adaptive_mini_gpt.model.transformer import MiniGPT
from adaptive_mini_gpt.rules.memory import GroundedSlot
from adaptive_mini_gpt.rules.multistep_dataset import (
    MultiStepProblem,
    generate_addition_depth_ladder,
    generate_multistep_problems,
)
from adaptive_mini_gpt.rules.plan import ExecutionPlan, PlanStep
from adaptive_mini_gpt.rules.procedural_dataset import (
    ProceduralDataset,
    ProceduralExample,
    ProceduralStep,
    collate_procedural_batch,
    generate_unseen_carry_dataset,
    parse_procedural_completion,
    problem_to_procedural_example,
)
from adaptive_mini_gpt.rules.procedural_evaluator import (
    evaluate_neural_procedure_unaided,
    train_procedural_model,
)


def test_m20_model_capacity_configurations() -> None:
    """Verify exact parameter counts and scaling ratios for Configurations A, B, and C."""
    # Config A: 1M parameter control (matches M18 base checkpoints)
    cfg_a = ModelConfig(
        vocab_size=400,
        d_model=128,
        n_layers=4,
        n_heads=4,
        d_ff=512,
        context_length=256,
    )
    model_a = MiniGPT(cfg_a)
    params_a = sum(p.numel() for p in model_a.parameters())
    assert params_a == 928512, f"Expected 928,512 parameters for 1M control, got {params_a}"

    # Config B: 3M parameter primary scaling experiment
    cfg_b = ModelConfig(
        vocab_size=400,
        d_model=256,
        n_layers=4,
        n_heads=8,
        d_ff=1024,
        context_length=256,
    )
    model_b = MiniGPT(cfg_b)
    params_b = sum(p.numel() for p in model_b.parameters())
    assert params_b == 3429888, f"Expected 3,429,888 parameters for 3M model, got {params_b}"

    # Config C: 10M parameter follow-up configuration
    cfg_c = ModelConfig(
        vocab_size=400,
        d_model=384,
        n_layers=6,
        n_heads=6,
        d_ff=1536,
        context_length=256,
    )
    model_c = MiniGPT(cfg_c)
    params_c = sum(p.numel() for p in model_c.parameters())
    assert params_c == 11053056, f"Expected 11,053,056 parameters for 10M model, got {params_c}"

    # Scaling ratio check
    ratio_b_a = params_b / params_a
    assert 3.5 < ratio_b_a < 3.8, f"Expected scaling ratio around 3.7x, got {ratio_b_a:.2f}"


def test_problem_to_procedural_example_conversion() -> None:
    """Test converting MultiStepProblem into structured procedural format with intermediate calculations."""
    slots = {
        "operand_1": GroundedSlot("operand_1", 120, None, "120", 0, 3, type_name="int"),
        "operand_2": GroundedSlot("operand_2", 80, None, "80", 0, 2, type_name="int"),
        "operand_3": GroundedSlot("operand_3", 45, None, "45", 0, 2, type_name="int"),
    }
    steps = [
        PlanStep(output_var="$step1", operation="add", args={"a": "$operand_1", "b": "$operand_2"}, units=None, step_index=1),
        PlanStep(output_var="$step2", operation="add", args={"a": "$step1", "b": "$operand_3"}, units=None, step_index=2),
    ]
    prob = MultiStepProblem(
        question="Calculate the sum of 120, 80, and 45.",
        expected_plan=ExecutionPlan(steps=steps, final_var="$step2"),
        expected_answer="245",
        num_steps=2,
        archetype="chained_addition_3",
        split="test",
        slots=slots,
    )

    proc_ex = problem_to_procedural_example(prob)
    assert proc_ex.final_answer == "245"
    assert len(proc_ex.steps) == 2
    assert proc_ex.steps[0].intermediate_calc == "120 + 80 = 200"
    assert proc_ex.steps[1].intermediate_calc == "200 + 45 = 245"

    prompt = proc_ex.format_prompt()
    target = proc_ex.format_target()
    assert "<|user|>" in prompt
    assert "<|assistant|>\nProcedure:" in prompt
    assert "1. $step1 = add(a='$operand_1', b='$operand_2') -> 120 + 80 = 200" in target
    assert "Final Answer: 245" in target


def test_parse_procedural_completion() -> None:
    """Test robust extraction of steps and final answer from raw model completions."""
    sample_completion = (
        "1. $step1 = add(a='$operand_1', b='$operand_2') -> 120 + 80 = 200 [km]\n"
        "2. $step2 = add(a='$step1', b='$operand_3') -> 200 + 45 = 245 [km]\n"
        "Final Answer: 245\n<|eos|>"
    )
    parsed = parse_procedural_completion(sample_completion)
    assert parsed["has_final_answer"] is True
    assert parsed["final_answer"] == "245"
    assert parsed["is_valid"] is True
    assert len(parsed["parsed_steps"]) == 2
    assert parsed["parsed_steps"][0]["calc"] == "120 + 80 = 200"
    assert parsed["parsed_steps"][0]["units"] == "km"

    # Malformed completion without final answer
    malformed = "1. $step1 = add(a='$operand_1', b='$operand_2')"
    parsed_mal = parse_procedural_completion(malformed)
    assert parsed_mal["has_final_answer"] is False
    assert parsed_mal["is_valid"] is False


def test_target_only_loss_masking() -> None:
    """Verify that ProceduralDataset masks prompt tokens with -100."""
    tokenizer = BPETokenizer()
    tokenizer.train("Calculate the sum of 120, 80, and 45. $operand_1 = 120 Procedure: 1. $step1 = add Final Answer: 245", target_vocab_size=100)

    slots = {
        "operand_1": GroundedSlot("operand_1", 10, None, "10", 0, 2, type_name="int"),
        "operand_2": GroundedSlot("operand_2", 20, None, "20", 0, 2, type_name="int"),
    }
    steps = [PlanStep(output_var="$step1", operation="add", args={"a": "$operand_1", "b": "$operand_2"}, units=None, step_index=1)]
    ex = ProceduralExample(
        question="Sum 10 and 20",
        slots=slots,
        steps=[ProceduralStep(1, "$step1", "add", {"a": "$operand_1", "b": "$operand_2"}, "10 + 20 = 30", 30)],
        final_answer="30",
        archetype="test",
        depth=1,
    )

    ds = ProceduralDataset([ex], tokenizer, context_length=128)
    inp, lab = ds[0]
    prompt_len = len(tokenizer.encode(ex.format_prompt()))

    # Prompt portion must be exactly -100
    assert (lab[:prompt_len] == -100).all()
    # Target portion must NOT be -100
    assert (lab[prompt_len:] != -100).all()
    assert (inp[prompt_len:] == lab[prompt_len:]).all()


def test_generate_unseen_carry_dataset() -> None:
    """Verify generation of multi-digit cascading carry arithmetic problems."""
    carry_examples = generate_unseen_carry_dataset(num_samples=5, seed=42)
    assert len(carry_examples) == 5
    for ex in carry_examples:
        assert ex.unseen_operands is True
        assert ex.has_complex_carry is True
        assert len(ex.steps) == 2
        # Sum must be mathematically exact
        n1 = ex.slots["operand_1"].value
        n2 = ex.slots["operand_2"].value
        n3 = ex.slots["operand_3"].value
        assert int(ex.final_answer) == n1 + n2 + n3


def test_matched_compute_budget_formula() -> None:
    """Verify that matched compute FLOPs ratio scales proportionally to parameter ratio."""
    params_1m = 928512
    params_3m = 3429888
    flops_per_token_1m = 6 * params_1m
    flops_per_token_3m = 6 * params_3m

    ratio = flops_per_token_3m / flops_per_token_1m
    base_epochs = 10
    matched_epochs_1m = round(base_epochs * ratio)

    # Compute of 1M at matched_epochs must approximately equal compute of 3M at base_epochs
    tokens_per_epoch = 5000
    total_flops_1m = matched_epochs_1m * tokens_per_epoch * flops_per_token_1m
    total_flops_3m = base_epochs * tokens_per_epoch * flops_per_token_3m

    diff_rel = abs(total_flops_1m - total_flops_3m) / total_flops_3m
    assert diff_rel < 0.05, f"Expected matched compute within 5%, got relative difference {diff_rel:.4f}"
