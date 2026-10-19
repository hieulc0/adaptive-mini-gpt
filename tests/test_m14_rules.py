"""Unit and integration tests for M14 Verified Rules and Reasoning Loop."""

from __future__ import annotations

import pytest
import torch

from adaptive_mini_gpt.data.subword_tokenizer import BPETokenizer
from adaptive_mini_gpt.model.config import ModelConfig
from adaptive_mini_gpt.model.transformer import MiniGPT
from adaptive_mini_gpt.rules import (
    RULE_REGISTRY,
    ReasoningController,
    RuleExecutionResult,
    RuleExecutor,
    SchemaValidationError,
    VerificationError,
    execute_add,
    execute_average_speed,
    execute_compare,
    execute_reverse,
    format_rule_call,
    generate_rule_problems,
    get_rule,
    get_rule_split,
    list_allowed_rules,
    parse_rule_call,
    verify_add,
    verify_average_speed,
    verify_compare,
    verify_reverse,
)


def test_rule_registry_and_specs() -> None:
    """Verify registration and specifications of all 4 M14 operations."""
    allowed = list_allowed_rules()
    for op in ("reverse", "add", "compare", "average_speed"):
        assert op in allowed
        spec = get_rule(op)
        assert spec.name == op
        assert callable(spec.execute_fn)
        assert callable(spec.verify_fn)
        assert len(spec.args_schema) > 0

    with pytest.raises(SchemaValidationError, match="Unknown operation 'eval'"):
        get_rule("eval")

    with pytest.raises(SchemaValidationError, match="Unknown operation 'bash'"):
        get_rule("bash")


def test_rule_execution_reverse() -> None:
    """Verify string reversal execution and verifier."""
    executor = RuleExecutor()
    res = executor.execute("reverse", {"s": "photon"})
    assert res.success is True
    assert res.verified is True
    assert res.result == "notohp"
    assert res.error is None

    # Invalid empty string
    res_bad = executor.execute("reverse", {"s": ""})
    assert res_bad.success is False
    assert res_bad.verified is False
    assert "SchemaValidationError" in res_bad.error


def test_rule_execution_add() -> None:
    """Verify addition execution and inverse subtraction verification."""
    executor = RuleExecutor()
    res = executor.execute("add", {"a": 127, "b": 386})
    assert res.success is True
    assert res.verified is True
    assert res.result == 513

    # String coercion check
    res_coerced = executor.execute("add", {"a": "45", "b": "55"})
    assert res_coerced.success is True
    assert res_coerced.result == 100

    # Non-numeric input rejection
    res_bad = executor.execute("add", {"a": "invalid", "b": 10})
    assert res_bad.success is False
    assert "must be numeric" in res_bad.error


def test_rule_execution_compare() -> None:
    """Verify transitive comparison execution and cycle rejection."""
    executor = RuleExecutor()
    res = executor.execute(
        "compare",
        {
            "entity_a": "Anna",
            "entity_b": "Chris",
            "relations": ["Anna > Ben", "Ben > Chris"],
        },
    )
    assert res.success is True
    assert res.verified is True
    assert res.result == "Anna"

    # Reverse direction
    res2 = executor.execute(
        "compare",
        {
            "entity_a": "Chris",
            "entity_b": "Anna",
            "relations": ["Anna > Ben", "Ben > Chris"],
        },
    )
    assert res2.success is True
    assert res2.result == "Anna"

    # Cyclic contradiction detection
    res_cycle = executor.execute(
        "compare",
        {
            "entity_a": "A",
            "entity_b": "B",
            "relations": ["A > B", "B > A"],
        },
    )
    assert res_cycle.success is False
    assert "VerificationError" in res_cycle.error or "Contradictory" in res_cycle.error

    # Indeterminate path
    res_indet = executor.execute(
        "compare",
        {
            "entity_a": "A",
            "entity_b": "C",
            "relations": ["A > B", "D > C"],
        },
    )
    assert res_indet.success is False


def test_rule_execution_average_speed() -> None:
    """Verify kinematics speed calculation and constraint enforcement."""
    executor = RuleExecutor()
    res = executor.execute("average_speed", {"distance_km": 180, "time_hours": 3})
    assert res.success is True
    assert res.verified is True
    assert res.result == "60 km/h"

    # Zero/negative time rejection
    res_zero = executor.execute("average_speed", {"distance_km": 100, "time_hours": 0})
    assert res_zero.success is False
    assert "must be strictly positive" in res_zero.error

    res_neg = executor.execute("average_speed", {"distance_km": -50, "time_hours": 2})
    assert res_neg.success is False
    assert "must be non-negative" in res_neg.error


def test_format_and_parse_rule_call() -> None:
    """Verify formatting and robust parsing of model tool calls."""
    call_str = format_rule_call("reverse", {"s": "k8f2a"})
    assert "Action: reverse(s='k8f2a')" in call_str
    r, a = parse_rule_call(call_str)
    assert r == "reverse"
    assert a == {"s": "k8f2a"}

    # JSON format
    json_call = '```json\n{"rule": "add", "args": {"a": 12, "b": 34}}\n```'
    r2, a2 = parse_rule_call(json_call)
    assert r2 == "add"
    assert a2 == {"a": 12, "b": 34}

    # Malformed call rejection
    with pytest.raises(ValueError, match="Could not parse valid rule call"):
        parse_rule_call("The answer is 42.")


def test_dataset_partitioning_disjoint() -> None:
    """Verify that rule dataset generation maintains strictly disjoint train/val/test splits."""
    train_probs = generate_rule_problems(num_samples_per_rule=20, seed=42, split="train")
    val_probs = generate_rule_problems(num_samples_per_rule=10, seed=43, split="val")
    test_probs = generate_rule_problems(num_samples_per_rule=10, seed=44, split="test")

    train_qs = {p.question for p in train_probs}
    val_qs = {p.question for p in val_probs}
    test_qs = {p.question for p in test_probs}

    assert len(train_qs) > 0
    assert len(val_qs) > 0
    assert len(test_qs) > 0

    assert len(train_qs.intersection(val_qs)) == 0
    assert len(train_qs.intersection(test_qs)) == 0
    assert len(val_qs.intersection(test_qs)) == 0


def test_reasoning_controller_oracle_mode() -> None:
    """Verify that Oracle mode executes with 100% precision on benchmark questions."""
    cfg = ModelConfig(vocab_size=100, d_model=32, n_layers=2, n_heads=2, d_ff=64, context_length=64)
    model = MiniGPT(cfg)
    tok = BPETokenizer()
    tok.train("abcdefghijklmnopqrstuvwxyz 0123456789 <|user|> <|assistant|> <|eos|>", target_vocab_size=80)

    controller = ReasoningController(model=model, tokenizer=tok, device=torch.device("cpu"))
    test_probs = generate_rule_problems(num_samples_per_rule=5, seed=999, split="test")

    for prob in test_probs:
        res = controller.solve(
            question=prob.question,
            expected_rule=prob.rule_name,
            expected_answer=prob.expected_answer,
            ground_truth_args=prob.args,
            mode="oracle",
        )
        assert res.status == "success", f"Oracle failed on {prob.question}: {res.final_answer}"
        assert res.rule_selection_correct is True
        assert res.arg_extraction_correct is True
        assert res.answer_correct is True
