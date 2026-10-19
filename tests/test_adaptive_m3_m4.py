"""Tests for M3 modular experts and M4 function-preserving growth."""

from __future__ import annotations

import torch

from adaptive_mini_gpt.adaptive.experts import BottleneckExpert
from adaptive_mini_gpt.adaptive.growth import grow_model_depth, grow_model_experts
from adaptive_mini_gpt.data.dataset import generate_arithmetic_corpus
from adaptive_mini_gpt.model.config import ModelConfig
from adaptive_mini_gpt.model.transformer import MiniGPT, TransformerBlock


def test_function_preserving_expert_growth() -> None:
    """Growing a dense model into an MoE model must preserve exact forward logits at step 0."""
    torch.manual_seed(42)
    cfg = ModelConfig(vocab_size=64, d_model=32, n_layers=2, n_heads=4, d_ff=64, context_length=16)
    model = MiniGPT(cfg)
    model.eval()

    x = torch.randint(0, 64, (2, 16))
    with torch.no_grad():
        logits_before, _ = model(x)

    # Grow 4 residual experts per layer
    grow_model_experts(model, add_experts=4, freeze_backbone=True)
    model.eval()

    with torch.no_grad():
        logits_after, _ = model(x)

    assert torch.allclose(logits_before, logits_after, atol=1e-6), (
        "Function-preserving invariant violated during expert growth!"
    )


def test_function_preserving_depth_growth() -> None:
    """Adding a zero-initialized residual block must preserve exact forward logits at step 0."""
    torch.manual_seed(42)
    cfg = ModelConfig(vocab_size=64, d_model=32, n_layers=2, n_heads=4, d_ff=64, context_length=16)
    model = MiniGPT(cfg)
    model.eval()

    x = torch.randint(0, 64, (2, 16))
    with torch.no_grad():
        logits_before, _ = model(x)

    grow_model_depth(model, add_layers=1)
    model.eval()

    with torch.no_grad():
        logits_after, _ = model(x)

    assert model.config.n_layers == 3
    assert torch.allclose(logits_before, logits_after, atol=1e-6), (
        "Function-preserving invariant violated during depth growth!"
    )


def test_frozen_backbone_and_expert_gradients() -> None:
    """Verify backbone weights receive no gradients when frozen, while zero-init experts learn."""
    torch.manual_seed(42)
    cfg = ModelConfig(vocab_size=64, d_model=32, n_layers=2, n_heads=4, d_ff=64, context_length=16)
    model = MiniGPT(cfg)
    grow_model_experts(model, add_experts=3, freeze_backbone=True)
    model.train()

    x = torch.randint(0, 64, (4, 16))
    y = torch.randint(0, 64, (4, 16))
    _, loss = model(x, y)
    assert loss is not None
    loss.backward()

    # Shared embeddings and dense attention must have grad=None
    block0 = model.blocks[0]
    assert isinstance(block0, TransformerBlock)
    assert model.wte.weight.grad is None
    assert block0.attn.c_attn.weight.grad is None

    # At least one expert w_down must have non-zero gradients on step 1
    assert block0.moe is not None
    expert_grads = [
        exp.w_down.weight.grad.abs().sum().item()
        for exp in block0.moe.experts
        if isinstance(exp, BottleneckExpert) and exp.w_down.weight.grad is not None
    ]
    assert sum(expert_grads) > 0.0
    assert "router_entropy" in model.last_aux_info


def test_procedural_arithmetic_corpus() -> None:
    """Verify deterministic generation of arithmetic corpus and fingerprinting."""
    text1, fp1 = generate_arithmetic_corpus(num_samples=100, seed=42)
    text2, fp2 = generate_arithmetic_corpus(num_samples=100, seed=42)
    assert text1 == text2
    assert fp1 == fp2
    assert "ADD" in text1 or "SUB" in text1 or "PAR" in text1


def test_task_conditioned_routing() -> None:
    """Verify deterministic routing by task_id in task mode."""
    cfg = ModelConfig(
        vocab_size=64,
        d_model=32,
        n_layers=1,
        n_heads=2,
        d_ff=64,
        context_length=8,
        adaptive_enabled=True,
        n_experts=4,
        routing_mode="task",
    )
    model = MiniGPT(cfg)
    model.eval()
    x = torch.randint(0, 64, (2, 8))
    _, _ = model(x, task_id=2)
    assert model.last_aux_info["expert_utilization"][2] == 1.0
