"""Tests for tiny-batch memorization and loss decline."""

from __future__ import annotations

import torch
from torch.optim import AdamW

from adaptive_mini_gpt.model.config import ModelConfig
from adaptive_mini_gpt.model.transformer import MiniGPT


def test_tiny_batch_memorization() -> None:
    """Verifies that the model can completely overfit/memorize a tiny batch."""
    torch.manual_seed(42)
    vocab_size = 20
    config = ModelConfig(
        vocab_size=vocab_size,
        d_model=32,
        n_layers=2,
        n_heads=2,
        d_ff=64,
        context_length=8,
        dropout=0.0,
    )
    model = MiniGPT(config)
    optimizer = AdamW(model.parameters(), lr=0.01)

    # Fixed tiny batch
    x = torch.randint(0, vocab_size, (2, 8))
    y = torch.randint(0, vocab_size, (2, 8))

    initial_loss = float("inf")
    final_loss = float("inf")

    model.train()
    for step in range(120):
        optimizer.zero_grad()
        _, loss = model(x, y)
        assert loss is not None
        if step == 0:
            initial_loss = loss.item()
        loss.backward()
        optimizer.step()
        final_loss = loss.item()

    # The loss must decrease significantly and reach near zero (< 0.05)
    assert final_loss < initial_loss
    assert final_loss < 0.05, f"Expected loss < 0.05 on tiny batch, got {final_loss:.4f}"
