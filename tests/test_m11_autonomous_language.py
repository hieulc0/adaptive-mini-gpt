"""Unit and integration tests for M11 Autonomous Language Learning Qualification."""

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
    generate_instruction_qa_data,
    generate_instruction_symbolic_data,
    verify_instruction_qa,
    verify_instruction_symbolic,
)
from adaptive_mini_gpt.data.subword_tokenizer import BPETokenizer
from adaptive_mini_gpt.model.config import ModelConfig
from adaptive_mini_gpt.model.transformer import MiniGPT
from adaptive_mini_gpt.training.checkpoint import CheckpointManager


def test_instruction_generators_and_verifiers() -> None:
    """Verify newly introduced M11 instruction generators and independent verifiers."""
    # 1. Symbolic reversal
    sym_text, sym_fp = generate_instruction_symbolic_data(num_samples=10, seed=42)
    assert len(sym_text) > 0
    assert len(sym_fp) == 16
    assert "<|user|>" in sym_text
    assert "<|assistant|>" in sym_text

    # Verify symbolic verifier
    prompt_sym = (
        "<|user|>\nReverse the sequence of characters in the word 'tensor'.\n<|assistant|>\n"
    )
    assert verify_instruction_symbolic(prompt_sym, "The reversed sequence is: rosnet.<|eos|>")
    assert not verify_instruction_symbolic(prompt_sym, "The reversed sequence is: tensor.<|eos|>")

    # 2. Factual QA
    qa_text, qa_fp = generate_instruction_qa_data(num_samples=10, seed=42)
    assert len(qa_text) > 0
    assert len(qa_fp) == 16

    prompt_qa = "<|user|>\nWhat is the capital of France?\n<|assistant|>\n"
    assert verify_instruction_qa(prompt_qa, "The capital of France is Paris.<|eos|>")
    assert not verify_instruction_qa(prompt_qa, "The capital of France is London.<|eos|>")


@pytest.fixture
def bpe_language_base_checkpoint(tmp_path: Path) -> tuple[Path, BPETokenizer, ModelConfig]:
    """Create a minimal pretrained BPE language model checkpoint."""
    ckpt_dir = tmp_path / "bpe_lang_base"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    corpus = (
        "The quick brown fox jumps over the lazy dog.\n"
        "William Shakespeare wrote Hamlet, Macbeth, and King Lear.\n"
        "Paris is the capital of France, and Tokyo is the capital of Japan.\n"
        "In quantum physics, photons and bosons mediate fundamental interactions.\n"
        "<|user|>\nWhat is the capital of France?\n<|assistant|>\nParis.<|eos|>\n"
        "<|user|>\nReverse the sequence of characters in the word 'tensor'.\n"
        "<|assistant|>\nrosnet.<|eos|>\n"
    ) * 10

    tokenizer = BPETokenizer()
    tokenizer.train(corpus, target_vocab_size=280)

    cfg = ModelConfig(
        vocab_size=tokenizer.vocab_size,
        d_model=64,
        n_layers=2,
        n_heads=2,
        d_ff=128,
        context_length=64,
        dropout=0.0,
    )
    model = MiniGPT(cfg)

    ckpt_data = {
        "model_state_dict": model.state_dict(),
        "model_config": cfg.to_dict(),
        "tokenizer": tokenizer.to_dict(),
        "tokenizer_fingerprint": tokenizer.get_fingerprint(),
        "step": 100,
        "tokens_seen": 2000,
        "best_val_loss": 3.2,
        "run_id": "test_bpe_lang_base",
    }
    ckpt_file = ckpt_dir / "latest.pt"
    torch.save(ckpt_data, ckpt_file)

    manifest_data = {
        "run_id": "test_bpe_lang_base",
        "step": 100,
        "tokens_seen": 2000,
        "best_val_loss": 3.2,
        "latest_checkpoint": "latest.pt",
        "history": [],
    }
    with open(ckpt_dir / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest_data, f)

    return ckpt_dir, tokenizer, cfg


def test_atomic_pointer_fault_injection(
    bpe_language_base_checkpoint: tuple[Path, BPETokenizer, ModelConfig], tmp_path: Path
) -> None:
    """Fault-injection test: verify crash resilience during atomic pointer switches."""
    base_dir, _, _ = bpe_language_base_checkpoint
    out_dir = tmp_path / "fault_injection_test"

    controller = LearningController(
        base_checkpoint_dir=base_dir,
        output_dir=out_dir,
        budget_steps=2,
        min_improvement=-1.0,  # Force promotion
        force_task="instruction_symbolic",
        growth_action=GrowthAction.CONTINUE,
        device="cpu",
        precision="fp32",
        seed=42,
    )

    _ = controller.run_cycle()
    active_dir = out_dir / "active_model"
    assert (active_dir / "active_pointer.json").is_file()

    # Fault 1: Simulate leftover uncommitted .tmp files in active_model
    corrupt_tmp = active_dir / "active_pointer.json.tmp"
    with open(corrupt_tmp, "w", encoding="utf-8") as f:
        f.write("{partial corrupted json...")

    # Loader must ignore .tmp and correctly resolve through the valid active_pointer.json
    resolved_ckpt = CheckpointManager.find_latest_valid_checkpoint(active_dir)
    assert resolved_ckpt.is_file()
    assert "published" in str(resolved_ckpt)

    # Fault 2: Simulate pointer referencing a deleted target file
    bad_pointer_file = active_dir / "active_pointer.json"
    with open(bad_pointer_file, "w", encoding="utf-8") as f:
        json.dump({"checkpoint_path": "/tmp/non_existent_path_xyz.pt"}, f)

    # Loader must gracefully fall back to backup/manifest/latest.pt without crashing
    fallback_ckpt = CheckpointManager.find_latest_valid_checkpoint(active_dir)
    assert fallback_ckpt.is_file()
    assert fallback_ckpt.name == "latest.pt"


def test_three_sequential_autonomous_language_cycles(
    bpe_language_base_checkpoint: tuple[Path, BPETokenizer, ModelConfig], tmp_path: Path
) -> None:
    """Test 3 sequential autonomous cycles with BPE language model: 2 promotions + 1 rollback."""
    base_dir, _, _ = bpe_language_base_checkpoint
    out_dir = tmp_path / "three_cycles_test"

    controller = LearningController(
        base_checkpoint_dir=base_dir,
        output_dir=out_dir,
        budget_steps=2,
        min_improvement=-1.0,  # Promote cycles 1 and 2
        force_task="instruction_symbolic",
        growth_action=GrowthAction.CONTINUE,
        device="cpu",
        precision="fp32",
        seed=42,
    )

    # --- Cycle 1: Learn symbolic reversal ---
    c1 = controller.run_cycle()
    assert c1["decision"] == ControllerState.PROMOTE.value
    assert c1["cycle_index"] == 1

    # --- Cycle 2: Learn factual QA from promoted checkpoint ---
    controller.force_task = "instruction_qa"
    controller.prepare_next_cycle()
    c2 = controller.run_cycle()
    assert c2["decision"] == ControllerState.PROMOTE.value
    assert c2["cycle_index"] == 2

    # --- Cycle 3: Test rollback on unattainable threshold ---
    controller.force_task = "instruction_symbolic"
    controller.min_improvement = 0.99  # Unattainable threshold in 2 steps
    controller.prepare_next_cycle()
    c3 = controller.run_cycle()
    assert c3["decision"] == ControllerState.ROLLBACK.value
    assert c3["cycle_index"] == 3

    # Check history contains all 3 cycles
    with open(out_dir / "auto_learn_history.json", encoding="utf-8") as f:
        history = json.load(f)
    assert len(history) == 3
    assert history[0]["decision"] == "PROMOTE"
    assert history[1]["decision"] == "PROMOTE"
    assert history[2]["decision"] == "ROLLBACK"

    # Check that published versions are immutable and intact
    published_versions = sorted((out_dir / "published").glob("v_*"))
    assert len(published_versions) == 2  # Exactly 2 promoted versions


def test_dense_control_vs_growth_decision(
    bpe_language_base_checkpoint: tuple[Path, BPETokenizer, ModelConfig], tmp_path: Path
) -> None:
    """Verify controller uses dense fine-tuning by default unless growth is supported/requested."""
    base_dir, _, _ = bpe_language_base_checkpoint
    out_dir = tmp_path / "dense_control_test"

    controller = LearningController(
        base_checkpoint_dir=base_dir,
        output_dir=out_dir,
        growth_action=GrowthAction.AUTO,
    )

    model, _, _ = controller.load_active_model()
    plan = controller.step_plan(model, "instruction_symbolic")

    # For AUTO on a dense model with 0 rollbacks, action must be CONTINUE (matched dense control)
    assert plan.growth_action == GrowthAction.CONTINUE.value
