"""Tests for MiniGPT model architecture, dimensions, and parameter counting."""

from __future__ import annotations

import pytest
import torch

from adaptive_mini_gpt.model.config import ModelConfig
from adaptive_mini_gpt.model.transformer import MiniGPT


def test_model_forward_dimensions() -> None:
    config = ModelConfig(
        vocab_size=64,
        d_model=32,
        n_layers=2,
        n_heads=2,
        d_ff=64,
        context_length=16,
        dropout=0.0,
    )
    model = MiniGPT(config)
    batch_size = 4
    seq_len = 10
    idx = torch.randint(0, config.vocab_size, (batch_size, seq_len))
    targets = torch.randint(0, config.vocab_size, (batch_size, seq_len))

    # Forward without targets
    logits, loss = model(idx)
    assert logits.shape == (batch_size, seq_len, config.vocab_size)
    assert loss is None

    # Forward with targets
    logits, loss = model(idx, targets=targets)
    assert logits.shape == (batch_size, seq_len, config.vocab_size)
    assert loss is not None
    assert loss.dim() == 0
    assert torch.isfinite(loss)


def test_context_length_exceeded() -> None:
    config = ModelConfig(
        vocab_size=32,
        d_model=16,
        n_layers=1,
        n_heads=2,
        d_ff=32,
        context_length=8,
    )
    model = MiniGPT(config)
    idx = torch.randint(0, config.vocab_size, (1, 9))
    with pytest.raises(ValueError, match="Cannot forward sequence of length"):
        model(idx)


def test_parameter_counting() -> None:
    config = ModelConfig(
        vocab_size=32,
        d_model=16,
        n_layers=1,
        n_heads=2,
        d_ff=32,
        context_length=8,
    )
    model = MiniGPT(config)
    info = model.inspect_architecture()

    assert info["d_model"] == 16
    assert info["n_layers"] == 1
    assert info["total_params"] > 0
    assert info["trainable_params"] == info["total_params"]
    assert info["non_embedding_params"] < info["total_params"]
    assert len(info["layers"]) == 1


def test_weight_tying() -> None:
    config_tied = ModelConfig(
        vocab_size=32,
        d_model=16,
        n_layers=1,
        n_heads=2,
        d_ff=32,
        context_length=8,
        tie_word_embeddings=True,
    )
    model_tied = MiniGPT(config_tied)
    assert model_tied.lm_head.weight is model_tied.wte.weight
