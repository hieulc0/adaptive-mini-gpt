"""Tests for checkpointing, atomic writes, manifest, and compatibility verification."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from torch.optim import AdamW

from adaptive_mini_gpt.data.tokenizer import CharacterTokenizer
from adaptive_mini_gpt.model.config import ModelConfig
from adaptive_mini_gpt.model.transformer import MiniGPT
from adaptive_mini_gpt.training.checkpoint import CheckpointManager
from adaptive_mini_gpt.training.config import TrainingConfig


def test_atomic_checkpoint_and_manifest(tmp_path: Path) -> None:
    model_cfg = ModelConfig(
        vocab_size=30, d_model=16, n_layers=1, n_heads=2, d_ff=32, context_length=8
    )
    train_cfg = TrainingConfig(max_steps=10)
    tok = CharacterTokenizer().fit("abcdefghijklmnopqrstuvwxyz")
    model = MiniGPT(model_cfg)
    opt = AdamW(model.parameters(), lr=1e-3)

    mgr = CheckpointManager(tmp_path)
    ckpt_path = mgr.save_checkpoint(
        checkpoint_name="latest.pt",
        model=model,
        optimizer=opt,
        model_config=model_cfg,
        training_config=train_cfg,
        tokenizer=tok,
        dataset_fingerprint="dummy_fp",
        step=5,
        tokens_seen=640,
        data_position=128,
        best_val_loss=2.34,
        run_id="test_run",
    )

    assert ckpt_path.is_file()
    # Ensure temporary file is gone
    assert not (tmp_path / "latest.pt.tmp").exists()

    # Verify manifest
    manifest_file = tmp_path / "manifest.json"
    assert manifest_file.is_file()

    # Restore state
    restored_model = MiniGPT(model_cfg)
    restored_opt = AdamW(restored_model.parameters(), lr=1e-3)
    meta = mgr.restore_checkpoint(
        checkpoint_path=ckpt_path,
        model=restored_model,
        optimizer=restored_opt,
        tokenizer=tok,
        dataset_fingerprint="dummy_fp",
    )
    assert meta["step"] == 5
    assert meta["tokens_seen"] == 640
    assert meta["data_position"] == 128
    assert meta["best_val_loss"] == 2.34

    # Verify weights are identical
    for p1, p2 in zip(model.parameters(), restored_model.parameters(), strict=True):
        assert torch.equal(p1, p2)


def test_corrupted_checkpoint_detection(tmp_path: Path) -> None:
    corrupt_file = tmp_path / "corrupt.pt"
    corrupt_file.write_bytes(b"PK\x03\x04truncated_garbage_data")

    mgr = CheckpointManager(tmp_path)
    model_cfg = ModelConfig(
        vocab_size=30, d_model=16, n_layers=1, n_heads=2, d_ff=32, context_length=8
    )
    model = MiniGPT(model_cfg)

    # Without auto-recovery, raises "Failed to read checkpoint"
    with pytest.raises(RuntimeError, match="Failed to read checkpoint"):
        mgr.restore_checkpoint(
            checkpoint_path=corrupt_file,
            model=model,
            auto_recover=False,
        )

    # With auto-recovery but no valid fallback in directory, raises "No valid checkpoint found"
    with pytest.raises(RuntimeError, match="No valid checkpoint found"):
        mgr.restore_checkpoint(
            checkpoint_path=corrupt_file,
            model=model,
            auto_recover=True,
        )


def test_incompatible_architecture_rejection(tmp_path: Path) -> None:
    model_cfg1 = ModelConfig(
        vocab_size=30, d_model=16, n_layers=1, n_heads=2, d_ff=32, context_length=8
    )
    train_cfg = TrainingConfig()
    tok = CharacterTokenizer().fit("abcdefghijklmnopqrstuvwxyz")
    model1 = MiniGPT(model_cfg1)
    opt = AdamW(model1.parameters())

    mgr = CheckpointManager(tmp_path)
    ckpt_path = mgr.save_checkpoint(
        checkpoint_name="latest.pt",
        model=model1,
        optimizer=opt,
        model_config=model_cfg1,
        training_config=train_cfg,
        tokenizer=tok,
        dataset_fingerprint="fp1",
        step=1,
        tokens_seen=10,
        data_position=0,
        best_val_loss=1.0,
        run_id="run1",
    )

    # Incompatible d_model
    model_cfg2 = ModelConfig(
        vocab_size=30, d_model=32, n_layers=1, n_heads=2, d_ff=32, context_length=8
    )
    model2 = MiniGPT(model_cfg2)

    with pytest.raises(ValueError, match="Architecture mismatch"):
        mgr.restore_checkpoint(ckpt_path, model2, tokenizer=tok, dataset_fingerprint="fp1")
