"""Tests verifying continuous vs interrupted and resumed training trajectories."""

from __future__ import annotations

from pathlib import Path

import torch

from adaptive_mini_gpt.data.dataset import TokenDataset
from adaptive_mini_gpt.data.tokenizer import CharacterTokenizer
from adaptive_mini_gpt.model.config import ModelConfig
from adaptive_mini_gpt.model.transformer import MiniGPT
from adaptive_mini_gpt.training.config import TrainingConfig
from adaptive_mini_gpt.training.trainer import Trainer


def test_continuous_vs_interrupted_resumption(tmp_path: Path) -> None:
    """Verifies that interrupted and resumed training matches continuous training."""
    raw_text = "The quick brown fox jumps over the lazy dog. " * 30
    tokenizer = CharacterTokenizer().fit(raw_text)
    tokens = torch.tensor(tokenizer.encode(raw_text), dtype=torch.long)

    model_config = ModelConfig(
        vocab_size=tokenizer.vocab_size,
        d_model=32,
        n_layers=2,
        n_heads=2,
        d_ff=64,
        context_length=16,
        dropout=0.0,
    )

    # 1. Continuous Run for 20 steps
    torch.manual_seed(1337)
    dir_continuous = tmp_path / "continuous"
    train_cfg_continuous = TrainingConfig(
        seed=1337,
        batch_size=4,
        gradient_accumulation=1,
        learning_rate=1e-3,
        max_steps=20,
        eval_every_steps=100,
        save_dir=str(dir_continuous),
        device="cpu",
    )
    ds_train_1 = TokenDataset(tokens, model_config.context_length)
    ds_val_1 = TokenDataset(tokens, model_config.context_length)

    model_cont = MiniGPT(model_config)
    trainer_cont = Trainer(
        model=model_cont,
        model_config=model_config,
        training_config=train_cfg_continuous,
        train_dataset=ds_train_1,
        val_dataset=ds_val_1,
        tokenizer=tokenizer,
        dataset_fingerprint="fp_test",
    )
    trainer_cont.train()

    # 2. Interrupted Run: Train 10 steps, save checkpoint
    torch.manual_seed(1337)
    dir_interrupted = tmp_path / "interrupted"
    train_cfg_part1 = TrainingConfig(
        seed=1337,
        batch_size=4,
        gradient_accumulation=1,
        learning_rate=1e-3,
        max_steps=10,
        eval_every_steps=100,
        checkpoint_every_steps=10,
        save_dir=str(dir_interrupted),
        device="cpu",
    )
    ds_train_2 = TokenDataset(tokens, model_config.context_length)
    ds_val_2 = TokenDataset(tokens, model_config.context_length)

    model_int1 = MiniGPT(model_config)
    trainer_int1 = Trainer(
        model=model_int1,
        model_config=model_config,
        training_config=train_cfg_part1,
        train_dataset=ds_train_2,
        val_dataset=ds_val_2,
        tokenizer=tokenizer,
        dataset_fingerprint="fp_test",
    )
    trainer_int1.train()

    checkpoint_step10 = dir_interrupted / "latest.pt"
    assert checkpoint_step10.is_file()

    # 3. Resume in a fresh instance for 10 more steps (up to 20)
    torch.manual_seed(9999)  # Deliberately different initial seed to ensure restoration sets state
    train_cfg_part2 = TrainingConfig(
        seed=9999,
        batch_size=4,
        gradient_accumulation=1,
        learning_rate=1e-3,
        max_steps=20,
        eval_every_steps=100,
        save_dir=str(dir_interrupted),
        device="cpu",
    )
    ds_train_3 = TokenDataset(tokens, model_config.context_length)
    ds_val_3 = TokenDataset(tokens, model_config.context_length)

    model_int2 = MiniGPT(model_config)
    trainer_int2 = Trainer(
        model=model_int2,
        model_config=model_config,
        training_config=train_cfg_part2,
        train_dataset=ds_train_3,
        val_dataset=ds_val_3,
        tokenizer=tokenizer,
        dataset_fingerprint="fp_test",
    )
    trainer_int2.resume_from_checkpoint(checkpoint_step10)
    assert trainer_int2.step == 10

    trainer_int2.train()
    assert trainer_int2.step == 20

    # 4. Compare final model weights between continuous and interrupted runs
    for (name_c, param_c), (name_i, param_i) in zip(
        model_cont.named_parameters(), model_int2.named_parameters(), strict=True
    ):
        assert name_c == name_i
        assert torch.allclose(param_c, param_i, atol=1e-5), (
            f"Parameter {name_c} diverged between continuous and resumed runs! "
            f"Max diff: {(param_c - param_i).abs().max().item()}"
        )
