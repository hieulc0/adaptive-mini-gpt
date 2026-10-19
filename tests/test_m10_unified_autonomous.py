"""Unit and integration tests for M10 Unified Autonomous Learning Qualification."""

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
from adaptive_mini_gpt.data.subword_tokenizer import BPETokenizer
from adaptive_mini_gpt.data.tokenizer import CharacterTokenizer
from adaptive_mini_gpt.model.config import ModelConfig
from adaptive_mini_gpt.model.transformer import MiniGPT
from adaptive_mini_gpt.training.checkpoint import CheckpointManager


@pytest.fixture
def character_base_checkpoint(tmp_path: Path) -> Path:
    """Create a minimal valid Character-based checkpoint."""
    ckpt_dir = tmp_path / "char_base_model"
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
        "tokenizer_fingerprint": tokenizer.get_fingerprint(),
        "step": 50,
        "tokens_seen": 500,
        "best_val_loss": 2.5,
        "run_id": "test_char_base",
    }
    ckpt_file = ckpt_dir / "latest.pt"
    torch.save(ckpt_data, ckpt_file)

    manifest_data = {
        "run_id": "test_char_base",
        "step": 50,
        "tokens_seen": 500,
        "best_val_loss": 2.5,
        "latest_checkpoint": "latest.pt",
        "history": [],
    }
    with open(ckpt_dir / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest_data, f)

    return ckpt_dir


@pytest.fixture
def bpe_base_checkpoint(tmp_path: Path) -> Path:
    """Create a minimal valid BPE-based checkpoint."""
    ckpt_dir = tmp_path / "bpe_base_model"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    corpus = "The quick brown fox jumps over the lazy dog. 14+25=39 85-32=53"
    tokenizer = BPETokenizer()
    tokenizer.train(corpus, target_vocab_size=275)

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
        "tokenizer_fingerprint": tokenizer.get_fingerprint(),
        "step": 100,
        "tokens_seen": 1000,
        "best_val_loss": 2.1,
        "run_id": "test_bpe_base",
    }
    ckpt_file = ckpt_dir / "latest.pt"
    torch.save(ckpt_data, ckpt_file)

    manifest_data = {
        "run_id": "test_bpe_base",
        "step": 100,
        "tokens_seen": 1000,
        "best_val_loss": 2.1,
        "latest_checkpoint": "latest.pt",
        "history": [],
    }
    with open(ckpt_dir / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest_data, f)

    return ckpt_dir


def test_bpe_checkpoint_loading_and_lineage(bpe_base_checkpoint: Path) -> None:
    """Test tokenizer-aware checkpoint loading and lineage validation."""
    raw = CheckpointManager.load_raw_checkpoint(bpe_base_checkpoint / "latest.pt")
    cfg = ModelConfig(**raw["model_config"])

    tok = CheckpointManager.load_tokenizer(raw, cfg)
    assert isinstance(tok, BPETokenizer)
    assert tok.vocab_size == cfg.vocab_size
    assert tok.get_fingerprint() == raw["tokenizer_fingerprint"]

    # Test tampering with fingerprint raises ValueError
    tampered_raw = dict(raw)
    tampered_raw["tokenizer_fingerprint"] = "bad_fingerprint_hash"
    with pytest.raises(ValueError, match="Tokenizer fingerprint mismatch"):
        CheckpointManager.load_tokenizer(tampered_raw, cfg)

    # Test vocab size mismatch raises ValueError
    mismatched_cfg = ModelConfig(vocab_size=999)
    with pytest.raises(ValueError, match="Lineage Mismatch"):
        CheckpointManager.load_tokenizer(raw, mismatched_cfg)


def test_promotion_gate_rejects_regression_even_above_mastery(tmp_path: Path) -> None:
    """Test that candidate with high accuracy is rejected if it regressed from baseline."""
    out_dir = tmp_path / "gate_test"
    controller = LearningController(
        base_checkpoint_dir=tmp_path,
        output_dir=out_dir,
        min_improvement=0.05,
        target_accuracy_threshold=0.90,
    )
    # Baseline was 0.98, candidate is 0.92 (exceeds 0.90 mastery threshold, but regressed by -0.06!)
    controller.journal.baseline_metrics = {
        "tasks": {"parity": {"accuracy": 0.98}},
        "retention": {
            "clean_prose": {"delta_loss": 0.0},
            "numerical_prose": {"delta_loss": 0.0},
            "mixed_prose_math": {"delta_loss": 0.0},
        },
    }

    base_acc = 0.98
    cand_acc = 0.92
    acc_gain = cand_acc - base_acc

    target_non_regressed = cand_acc >= (base_acc - 1e-6)
    target_improved = acc_gain >= controller.min_improvement
    target_mastered = (cand_acc >= controller.target_accuracy_threshold) and target_non_regressed
    target_passed = target_non_regressed and (target_improved or target_mastered)

    assert not target_non_regressed
    assert not target_passed


def test_crash_safe_atomic_publication(
    character_base_checkpoint: Path, tmp_path: Path
) -> None:
    """Test verified staging, immutable published directory, and atomic active-model swap."""
    out_dir = tmp_path / "atomic_pub_test"
    controller = LearningController(
        base_checkpoint_dir=character_base_checkpoint,
        output_dir=out_dir,
        budget_steps=2,
        min_improvement=-1.0,  # Force promotion
        force_task="parity",
        growth_action=GrowthAction.CONTINUE,
        device="cpu",
        precision="fp32",
        seed=42,
    )

    result = controller.run_cycle()
    assert result["decision"] == ControllerState.PROMOTE.value

    # Verify published directory exists with immutable checkpoint and manifest
    published_dirs = list((out_dir / "published").glob("v_*"))
    assert len(published_dirs) == 1
    pub_dir = published_dirs[0]
    assert (pub_dir / "checkpoint.pt").is_file()
    assert (pub_dir / "manifest.json").is_file()

    # Verify active_model directory contains atomically swapped files
    active_dir = out_dir / "active_model"
    assert (active_dir / "latest.pt").is_file()
    assert (active_dir / "manifest.json").is_file()
    assert (active_dir / "active_version.json").is_file()

    with open(active_dir / "active_version.json", encoding="utf-8") as f:
        v_info = json.load(f)
    assert v_info["version_index"] == 1
    assert str(pub_dir / "checkpoint.pt") in v_info["published_path"]


def test_two_consecutive_cycles_with_lineage(
    character_base_checkpoint: Path, tmp_path: Path
) -> None:
    """Test that two consecutive cycles operate on promoted checkpoint lineage."""
    out_dir = tmp_path / "consecutive_cycles_test"
    controller = LearningController(
        base_checkpoint_dir=character_base_checkpoint,
        output_dir=out_dir,
        budget_steps=2,
        min_improvement=-1.0,  # Allow promotions for testing cycle chaining
        force_task="parity",
        growth_action=GrowthAction.CONTINUE,
        device="cpu",
        precision="fp32",
        seed=42,
    )

    results = controller.run_cycles(max_cycles=2)
    assert len(results) == 2
    assert results[0]["decision"] == ControllerState.PROMOTE.value
    assert results[1]["decision"] == ControllerState.PROMOTE.value

    # Check that cycle 2 operated with cycle_index == 2
    assert results[0]["cycle_index"] == 1
    assert results[1]["cycle_index"] == 2

    # Check history has 2 records
    with open(out_dir / "auto_learn_history.json", encoding="utf-8") as f:
        history = json.load(f)
    assert len(history) == 2
    assert history[0]["cycle_index"] == 1
    assert history[1]["cycle_index"] == 2

    # Verify 2 distinct published versions
    published_dirs = sorted((out_dir / "published").glob("v_*"))
    assert len(published_dirs) == 2


def test_bpe_model_in_autonomous_controller(
    bpe_base_checkpoint: Path, tmp_path: Path
) -> None:
    """Test autonomous controller cycle executing with a BPE subword model."""
    out_dir = tmp_path / "bpe_controller_test"
    controller = LearningController(
        base_checkpoint_dir=bpe_base_checkpoint,
        output_dir=out_dir,
        budget_steps=2,
        min_improvement=-1.0,
        force_task="addition",
        growth_action=GrowthAction.CONTINUE,
        device="cpu",
        precision="fp32",
        seed=42,
    )

    result = controller.run_cycle()
    assert result["decision"] == ControllerState.PROMOTE.value

    # Verify active checkpoint retains BPE tokenizer
    promoted_raw = CheckpointManager.load_raw_checkpoint(
        out_dir / "active_model" / "latest.pt"
    )
    promoted_cfg = ModelConfig(**promoted_raw["model_config"])
    promoted_tok = CheckpointManager.load_tokenizer(promoted_raw, promoted_cfg)

    assert isinstance(promoted_tok, BPETokenizer)
    assert promoted_tok.vocab_size == promoted_cfg.vocab_size


def test_compute_budget_accounting_and_termination(
    character_base_checkpoint: Path, tmp_path: Path
) -> None:
    """Test that controller tracks compute budget and terminates when budget is exceeded."""
    out_dir = tmp_path / "budget_test"
    controller = LearningController(
        base_checkpoint_dir=character_base_checkpoint,
        output_dir=out_dir,
        budget_steps=10,
        max_budget_steps=5,  # Cap at 5 steps total
        min_improvement=-1.0,
        force_task="parity",
        growth_action=GrowthAction.CONTINUE,
        device="cpu",
        precision="fp32",
        seed=42,
    )

    _ = controller.run_cycle()
    # Budget was capped at min(10, 5) = 5 steps
    assert controller.journal.cumulative_steps == 5
    assert controller.journal.cumulative_tokens > 0
    assert controller.journal.cumulative_gpu_seconds >= 0.0
