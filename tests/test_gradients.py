"""Tests for gradient calculation, backward pass, and clipping."""

from __future__ import annotations

import torch

from adaptive_mini_gpt.model.config import ModelConfig
from adaptive_mini_gpt.model.transformer import MiniGPT


def test_finite_gradients() -> None:
    config = ModelConfig(
        vocab_size=32,
        d_model=16,
        n_layers=2,
        n_heads=2,
        d_ff=32,
        context_length=8,
    )
    model = MiniGPT(config)
    x = torch.randint(0, config.vocab_size, (2, 8))
    y = torch.randint(0, config.vocab_size, (2, 8))

    _, loss = model(x, y)
    assert loss is not None
    loss.backward()

    for name, param in model.named_parameters():
        if param.requires_grad:
            assert param.grad is not None, f"Parameter {name} has no gradient."
            assert torch.isfinite(param.grad).all(), f"Parameter {name} has non-finite gradient."


def test_gradient_clipping() -> None:
    config = ModelConfig(
        vocab_size=32,
        d_model=16,
        n_layers=1,
        n_heads=2,
        d_ff=32,
        context_length=8,
    )
    model = MiniGPT(config)
    x = torch.randint(0, config.vocab_size, (2, 8))
    y = torch.randint(0, config.vocab_size, (2, 8))

    _, loss = model(x, y)
    assert loss is not None
    loss.backward()

    max_norm = 0.5
    total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_norm)
    assert torch.isfinite(total_norm)

    # Re-calculate norm after clipping
    clipped_norm = torch.norm(
        torch.stack(
            [torch.norm(p.grad.detach(), 2) for p in model.parameters() if p.grad is not None]
        ),
        2,
    )
    assert clipped_norm <= max_norm + 1e-5
