"""Tests for M2: persistence, backup mirroring, corrupted checkpoint recovery, and AMP state."""

from __future__ import annotations

from pathlib import Path

import torch
from torch.optim import AdamW

from adaptive_mini_gpt.data.dataset import TokenDataset
from adaptive_mini_gpt.data.tokenizer import CharacterTokenizer
from adaptive_mini_gpt.model.config import ModelConfig
from adaptive_mini_gpt.model.transformer import MiniGPT
from adaptive_mini_gpt.training.checkpoint import CheckpointManager
from adaptive_mini_gpt.training.config import TrainingConfig
from adaptive_mini_gpt.training.trainer import Trainer


def test_backup_mirroring_and_pruning(tmp_path: Path) -> None:
    local_dir = tmp_path / "local_runs"
    backup_dir = tmp_path / "drive_backup"

    model_cfg = ModelConfig(
        vocab_size=30, d_model=16, n_layers=1, n_heads=2, d_ff=32, context_length=8
    )
    train_cfg = TrainingConfig(max_steps=100, backup_dir=str(backup_dir), keep_last_n_checkpoints=2)
    tok = CharacterTokenizer().fit("abcdefghijklmnopqrstuvwxyz")
    model = MiniGPT(model_cfg)
    opt = AdamW(model.parameters())

    mgr = CheckpointManager(save_dir=local_dir, backup_dir=backup_dir, keep_last_n=2)

    # Save step 10
    mgr.save_checkpoint(
        checkpoint_name="latest.pt",
        model=model,
        optimizer=opt,
        model_config=model_cfg,
        training_config=train_cfg,
        tokenizer=tok,
        dataset_fingerprint="fp1",
        step=10,
        tokens_seen=100,
        data_position=50,
        best_val_loss=3.0,
        run_id="run_m2",
    )

    # Verify both local and backup have step 10 and latest.pt
    assert (local_dir / "step_000010.pt").is_file()
    assert (backup_dir / "step_000010.pt").is_file()
    assert (backup_dir / "latest.pt").is_file()
    assert (backup_dir / "manifest.json").is_file()

    # Save step 20
    mgr.save_checkpoint(
        checkpoint_name="latest.pt",
        model=model,
        optimizer=opt,
        model_config=model_cfg,
        training_config=train_cfg,
        tokenizer=tok,
        dataset_fingerprint="fp1",
        step=20,
        tokens_seen=200,
        data_position=100,
        best_val_loss=2.5,
        run_id="run_m2",
    )

    # Save step 30 (should prune step 10 because keep_last_n=2)
    mgr.save_checkpoint(
        checkpoint_name="latest.pt",
        model=model,
        optimizer=opt,
        model_config=model_cfg,
        training_config=train_cfg,
        tokenizer=tok,
        dataset_fingerprint="fp1",
        step=30,
        tokens_seen=300,
        data_position=150,
        best_val_loss=2.0,
        run_id="run_m2",
    )

    assert not (backup_dir / "step_000010.pt").exists(), "Old checkpoint should have been pruned."
    assert (backup_dir / "step_000020.pt").is_file()
    assert (backup_dir / "step_000030.pt").is_file()


def test_corrupted_checkpoint_auto_recovery(tmp_path: Path) -> None:
    save_dir = tmp_path / "runs"
    model_cfg = ModelConfig(
        vocab_size=30, d_model=16, n_layers=1, n_heads=2, d_ff=32, context_length=8
    )
    train_cfg = TrainingConfig()
    tok = CharacterTokenizer().fit("abcdefghijklmnopqrstuvwxyz")
    model = MiniGPT(model_cfg)
    opt = AdamW(model.parameters())

    mgr = CheckpointManager(save_dir=save_dir)

    # 1. Save valid checkpoint at step 10
    mgr.save_checkpoint(
        checkpoint_name="latest.pt",
        model=model,
        optimizer=opt,
        model_config=model_cfg,
        training_config=train_cfg,
        tokenizer=tok,
        dataset_fingerprint="fp1",
        step=10,
        tokens_seen=100,
        data_position=50,
        best_val_loss=3.0,
        run_id="run_recover",
    )

    # 2. Simulate corruption of latest.pt (truncate to corrupt bytes)
    corrupted_latest = save_dir / "latest.pt"
    corrupted_latest.write_bytes(b"PK\x03\x04truncated_corrupt_data")

    # 3. Request restoration of the corrupted latest.pt with auto_recover=True
    restored_model = MiniGPT(model_cfg)
    restored_opt = AdamW(restored_model.parameters())
    meta = mgr.restore_checkpoint(
        checkpoint_path=corrupted_latest,
        model=restored_model,
        optimizer=restored_opt,
        tokenizer=tok,
        dataset_fingerprint="fp1",
        auto_recover=True,
    )

    # Assert recovery automatically recovered valid step 10 checkpoint
    assert meta["step"] == 10
    assert "step_000010.pt" in meta["recovered_from"]


def test_cpu_mixed_precision_graceful_fallback(tmp_path: Path) -> None:
    """Verifies that requesting fp16 precision on CPU falls back cleanly without crashing."""
    raw_text = "The quick brown fox jumps over the lazy dog. " * 10
    tok = CharacterTokenizer().fit(raw_text)
    tokens = torch.tensor(tok.encode(raw_text), dtype=torch.long)

    model_cfg = ModelConfig(
        vocab_size=tok.vocab_size,
        d_model=32,
        n_layers=1,
        n_heads=2,
        d_ff=64,
        context_length=8,
    )
    train_cfg = TrainingConfig(
        max_steps=5,
        device="cpu",
        precision="fp16",  # Requests fp16 on CPU
        save_dir=str(tmp_path / "cpu_run"),
    )

    model = MiniGPT(model_cfg)
    ds = TokenDataset(tokens, model_cfg.context_length)
    trainer = Trainer(
        model=model,
        model_config=model_cfg,
        training_config=train_cfg,
        train_dataset=ds,
        val_dataset=ds,
        tokenizer=tok,
        dataset_fingerprint="fp_cpu",
    )

    res = trainer.train()
    assert res["step"] == 5

    # Check metrics file recorded step_time_sec and gpu_type
    metrics_file = tmp_path / "cpu_run" / "metrics.jsonl"
    assert metrics_file.is_file()
    lines = metrics_file.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) >= 5
    import json
    rec = json.loads(lines[0])
    assert "step_time_sec" in rec
    assert "gpu_type" in rec
    assert rec["gpu_type"] == "CPU"
