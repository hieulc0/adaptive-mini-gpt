"""Unit and integration tests for M9 Natural Language Learning and Subword Tokenizer."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from adaptive_mini_gpt.data.instruction_data import (
    InstructionDataset,
    format_instruction_pair,
    generate_curated_instruction_corpus,
)
from adaptive_mini_gpt.data.subword_tokenizer import BPETokenizer
from adaptive_mini_gpt.evaluation.language_evaluator import (
    evaluate_instruction_following,
    validate_checkpoint_tokenizer_lineage,
)
from adaptive_mini_gpt.model.config import ModelConfig
from adaptive_mini_gpt.model.transformer import MiniGPT
from adaptive_mini_gpt.training.config import TrainingConfig
from adaptive_mini_gpt.training.trainer import Trainer


def test_bpe_tokenizer_training_and_roundtrip(tmp_path: Path) -> None:
    """Verify BPETokenizer trains merges, preserves special tokens, and decodes losslessly."""
    text = (
        "The quick brown fox jumps over the lazy dog. "
        "Natural language processing with Byte-Pair Encoding subword tokenizers!\n"
    ) * 10
    tokenizer = BPETokenizer()
    tokenizer.train(text, target_vocab_size=300)

    assert tokenizer.vocab_size <= 300
    assert len(tokenizer.merges) > 0
    assert tokenizer.pad_id >= 0
    assert tokenizer.user_id >= 0
    assert tokenizer.assistant_id >= 0
    assert tokenizer.eos_id >= 0

    # Test roundtrip
    sample_text = "The quick brown fox jumps over the lazy dog."
    encoded = tokenizer.encode(sample_text)
    decoded = tokenizer.decode(encoded)
    assert decoded == sample_text

    # Test special token preservation
    formatted = format_instruction_pair("What is 2+2?", "4")
    enc_formatted = tokenizer.encode(formatted)
    dec_formatted = tokenizer.decode(enc_formatted)
    assert dec_formatted == formatted

    # Test serialization
    save_path = tmp_path / "bpe_tokenizer.json"
    tokenizer.save(save_path)
    loaded = BPETokenizer.load(save_path)
    assert loaded.vocab_size == tokenizer.vocab_size
    assert loaded.merges == tokenizer.merges
    assert loaded.decode(loaded.encode(sample_text)) == sample_text


def test_tokenizer_lineage_validation() -> None:
    """Verify that attempting to load a mismatched vocabulary raises ValueError."""
    tokenizer = BPETokenizer()
    tokenizer.train("Small training sample " * 5, target_vocab_size=280)

    # Model configured with 108 character vocabulary
    char_cfg = ModelConfig(vocab_size=108, d_model=32, n_layers=1, n_heads=1, d_ff=32)
    char_model = MiniGPT(char_cfg)

    with pytest.raises(ValueError, match="Lineage Mismatch"):
        validate_checkpoint_tokenizer_lineage(char_model, tokenizer.to_dict())


def test_instruction_dataset_loss_masking() -> None:
    """Verify InstructionDataset masks user prompt tokens with -1."""
    tokenizer = BPETokenizer()
    tokenizer.train("What is 1+1? It is 2.\n" * 5, target_vocab_size=275)

    pair_text = format_instruction_pair("What is 1+1?", "It is 2.")
    tokens = tokenizer.encode(pair_text)

    dataset = InstructionDataset(
        token_ids=tokens,
        context_length=len(tokens) - 2,
        assistant_token_id=tokenizer.assistant_id,
        eos_token_id=tokenizer.eos_id,
        user_token_id=tokenizer.user_id,
        ignore_index=-1,
    )

    x, y = dataset[0]
    assert x.shape == y.shape
    # Ensure some tokens are masked (-1) and some are unmasked (> -1)
    assert (y == -1).any()
    assert (y != -1).any()


def test_instruction_fine_tuning_and_evaluation(tmp_path: Path) -> None:
    """Verify language model can train on instruction dataset and evaluate format compliance."""
    text, fp = generate_curated_instruction_corpus(num_synthetic_reasoning=20, seed=42)

    tokenizer = BPETokenizer()
    tokenizer.train(text, target_vocab_size=320)

    cfg = ModelConfig(
        vocab_size=tokenizer.vocab_size,
        d_model=32,
        n_layers=2,
        n_heads=2,
        d_ff=64,
        context_length=64,
        dropout=0.0,
    )
    model = MiniGPT(cfg)

    tokens = tokenizer.encode(text)
    train_tokens = tokens[: int(len(tokens) * 0.8)]
    val_tokens = tokens[int(len(tokens) * 0.8) :]

    train_ds = InstructionDataset(
        token_ids=train_tokens,
        context_length=cfg.context_length,
        assistant_token_id=tokenizer.assistant_id,
        eos_token_id=tokenizer.eos_id,
        user_token_id=tokenizer.user_id,
    )
    val_ds = InstructionDataset(
        token_ids=val_tokens,
        context_length=cfg.context_length,
        assistant_token_id=tokenizer.assistant_id,
        eos_token_id=tokenizer.eos_id,
        user_token_id=tokenizer.user_id,
    )

    train_cfg = TrainingConfig(
        seed=42,
        batch_size=8,
        learning_rate=1e-3,
        max_steps=5,
        device="cpu",
        precision="fp32",
        save_dir=str(tmp_path / "m9_run"),
    )

    trainer = Trainer(
        model=model,
        model_config=cfg,
        training_config=train_cfg,
        train_dataset=train_ds,
        val_dataset=val_ds,
        tokenizer=tokenizer,
        dataset_fingerprint=fp,
    )
    summary = trainer.train()
    assert summary["step"] == 5

    # Run instruction evaluation on held-out queries
    test_queries = [
        {"instruction": "What is the capital of France?", "expected_contains": "Paris"},
        {"instruction": "What is 10 + 20?", "expected_contains": "30"},
    ]
    eval_res = evaluate_instruction_following(
        model=model,
        tokenizer=tokenizer,
        test_queries=test_queries,
        device=torch.device("cpu"),
        max_new_tokens=32,
    )

    assert "format_compliance_rate" in eval_res
    assert "exact_answer_accuracy" in eval_res
    assert len(eval_res["sample_outputs"]) == 2
