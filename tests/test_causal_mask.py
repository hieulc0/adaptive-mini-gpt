"""Tests for causal masking and future token independence."""

from __future__ import annotations

import torch

from adaptive_mini_gpt.model.config import ModelConfig
from adaptive_mini_gpt.model.transformer import MiniGPT


def test_causal_mask_mathematical_independence() -> None:
    """Verifies that changing future tokens at t >= k has zero effect on earlier logits."""
    torch.manual_seed(42)
    config = ModelConfig(
        vocab_size=50,
        d_model=32,
        n_layers=2,
        n_heads=2,
        d_ff=64,
        context_length=16,
        dropout=0.0,  # Ensure deterministic evaluation
    )
    model = MiniGPT(config)
    model.eval()

    seq_len = 12
    # Create baseline sequence
    x1 = torch.randint(0, config.vocab_size, (1, seq_len))

    # Split position k
    k = 7

    # Create modified sequence where tokens from position k onward are different
    x2 = x1.clone()
    # Shift tokens from index k to ensure they differ
    x2[:, k:] = (x2[:, k:] + 1) % config.vocab_size

    with torch.no_grad():
        logits1, _ = model(x1)
        logits2, _ = model(x2)

    # 1. Past positions (0 to k - 1) MUST be identical
    assert torch.allclose(
        logits1[:, :k, :], logits2[:, :k, :], atol=1e-6
    ), "Past logits changed when modifying future tokens! Causal mask violated."

    # 2. Future positions (k onward) MUST differ because the input differs
    assert not torch.allclose(
        logits1[:, k:, :], logits2[:, k:, :], atol=1e-4
    ), "Future logits should reflect changed tokens."
