"""Unit and integration tests for M8 Autonomous Learning and Adaptive Growth Controller."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from adaptive_mini_gpt.autonomous import (
    ControllerState,
    GrowthAction,
    LearningController,
)
from adaptive_mini_gpt.autonomous.tasks import (
    generate_addition_data,
    generate_multiplication_data,
    generate_parity_data,
    generate_reverse_data,
    generate_subtraction_data,
    verify_addition,
    verify_multiplication,
    verify_parity,
    verify_reverse,
    verify_subtraction,
)
from adaptive_mini_gpt.data.tokenizer import CharacterTokenizer
from adaptive_mini_gpt.model.config import ModelConfig
from adaptive_mini_gpt.model.transformer import MiniGPT


def test_independent_verifiers() -> None:
    """Verify independent verifiers reject incorrect answers and accept ground truth."""
    # Addition
    assert verify_addition("12+34=", "46")
    assert verify_addition("12+34=", "46\n")
    assert not verify_addition("12+34=", "45")

    # Subtraction
    assert verify_subtraction("50-12=", "38")
    assert not verify_subtraction("50-12=", "39")

    # Multiplication
    assert verify_multiplication("7*8=", "56")
    assert not verify_multiplication("7*8=", "54")

    # Parity
    assert verify_parity("parity 42=", "even")
    assert verify_parity("parity 103=", "odd")
    assert not verify_parity("parity 42=", "odd")

    # Reverse
    assert verify_reverse("rev(hello)=", "olleh")
    assert not verify_reverse("rev(hello)=", "hello")


def test_procedural_generators() -> None:
    """Verify procedural task generators produce non-empty, reproducible datasets."""
    for _tid, gen_fn in [
        ("addition", generate_addition_data),
        ("subtraction", generate_subtraction_data),
        ("multiplication", generate_multiplication_data),
        ("parity", generate_parity_data),
        ("reverse", generate_reverse_data),
    ]:
        text, fp = gen_fn(10, 42)
        assert len(text) > 0
        assert len(fp) == 16
        text2, fp2 = gen_fn(10, 42)
        assert text == text2
        assert fp == fp2


@pytest.fixture
def base_model_checkpoint(tmp_path: Path) -> Path:
    """Create a minimal valid base checkpoint for controller testing."""
    ckpt_dir = tmp_path / "base_model"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = CharacterTokenizer(use_base_ascii=True).fit(
        "Shakespeare text sample\n1234567890=+-*"
    )
    cfg = ModelConfig(
        vocab_size=tokenizer.vocab_size,
        d_model=32,
        n_layers=2,
        n_heads=2,
        d_ff=64,
        context_length=32,
        dropout=0.0,
    )
    model = MiniGPT(cfg)

    ckpt_data = {
        "model_state_dict": model.state_dict(),
        "model_config": cfg.to_dict(),
        "tokenizer": tokenizer.to_dict(),
        "step": 50,
        "tokens_seen": 500,
        "best_val_loss": 2.5,
        "run_id": "test_pretrain",
    }
    ckpt_file = ckpt_dir / "latest.pt"
    torch.save(ckpt_data, ckpt_file)

    manifest_data = {
        "run_id": "test_pretrain",
        "step": 50,
        "tokens_seen": 500,
        "best_val_loss": 2.5,
        "latest_checkpoint": "latest.pt",
        "history": [],
    }
    with open(ckpt_dir / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest_data, f)

    return ckpt_dir


def test_controller_state_machine_cycle(base_model_checkpoint: Path, tmp_path: Path) -> None:
    """Test full autonomous learning controller cycle."""
    out_dir = tmp_path / "auto_learn_test"
    controller = LearningController(
        base_checkpoint_dir=base_model_checkpoint,
        output_dir=out_dir,
        budget_steps=5,
        min_improvement=-1.0,  # Allow promotion for testing cycle
        max_regression_tolerance=10.0,
        force_task="parity",
        growth_action=GrowthAction.GROW_EXPERT,
        device="cpu",
        precision="fp32",
        seed=42,
    )

    result = controller.run_cycle()

    assert result["decision"] == ControllerState.PROMOTE.value
    assert (out_dir / "controller_state.json").is_file()
    assert (out_dir / "auto_learn_history.json").is_file()
    assert (out_dir / "active_model" / "latest.pt").is_file()
    assert (out_dir / "active_model" / "manifest.json").is_file()

    with open(out_dir / "auto_learn_history.json", encoding="utf-8") as f:
        history = json.load(f)
    assert len(history) == 1
    assert history[0]["decision"] == "PROMOTE"
    assert history[0]["selected_task"] == "parity"


def test_controller_rollback_on_regression(base_model_checkpoint: Path, tmp_path: Path) -> None:
    """Test controller correctly rejects candidate and rolls back if criteria fail."""
    out_dir = tmp_path / "auto_learn_rollback"
    controller = LearningController(
        base_checkpoint_dir=base_model_checkpoint,
        output_dir=out_dir,
        budget_steps=2,
        min_improvement=0.99,  # Unattainable threshold in 2 steps
        max_regression_tolerance=0.0000000001,
        force_task="addition",
        growth_action=GrowthAction.CONTINUE,
        device="cpu",
        precision="fp32",
        seed=42,
    )

    result = controller.run_cycle()

    assert result["decision"] == ControllerState.ROLLBACK.value
    assert "ROLLBACK" in result["state"] or result["state"] == "STOP"
    assert not (out_dir / "active_model").exists()

    with open(out_dir / "auto_learn_history.json", encoding="utf-8") as f:
        history = json.load(f)
    assert len(history) == 1
    assert history[0]["decision"] == "ROLLBACK"
    assert "Target accuracy gain" in history[0]["reason"]


def test_controller_crash_recovery(base_model_checkpoint: Path, tmp_path: Path) -> None:
    """Test controller can resume from an interrupted state."""
    out_dir = tmp_path / "auto_learn_crash"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Simulate an interrupted controller in PLAN state
    interrupted_journal = {
        "experiment_id": "exp_crash_123",
        "state": ControllerState.PLAN.value,
        "current_step": 0,
        "selected_task": "subtraction",
        "plan": None,
        "baseline_metrics": {"tasks": {"subtraction": {"accuracy": 0.0}}},
        "candidate_metrics": {},
        "decision": None,
        "decision_reason": None,
        "history": [],
    }
    with open(out_dir / "controller_state.json", "w", encoding="utf-8") as f:
        json.dump(interrupted_journal, f)

    # Resume controller
    resumed_controller = LearningController(
        base_checkpoint_dir=base_model_checkpoint,
        output_dir=out_dir,
        budget_steps=2,
        min_improvement=-1.0,
        device="cpu",
        precision="fp32",
        seed=42,
    )

    assert resumed_controller.journal.state == ControllerState.PLAN.value
    assert resumed_controller.journal.selected_task == "subtraction"

    result = resumed_controller.run_cycle()
    assert result["experiment_id"] == "exp_crash_123"
    assert result["selected_task"] == "subtraction"
    assert result["decision"] in ("PROMOTE", "ROLLBACK")
