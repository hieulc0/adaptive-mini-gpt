"""Unit and integration tests for M30 Fast Neural Adaptation and Generalization."""

from __future__ import annotations

import copy
import pytest
import torch
import torch.nn as nn

from adaptive_mini_gpt.adaptation.curriculum import (
    build_multiscale_episode_a,
    build_multiscale_episode_b,
    build_multiscale_episode_c,
)
from adaptive_mini_gpt.adaptation.trainer import (
    hash_base_parameters,
    run_fast_adaptation_regime,
)
from adaptive_mini_gpt.data.subword_tokenizer import BPETokenizer
from adaptive_mini_gpt.model.capacity_migration import create_1m_control_config
from adaptive_mini_gpt.model.lora import (
    LoRALinear,
    freeze_base_parameters,
    get_lora_parameter_stats,
    inject_lora_adapters,
    toggle_lora_adapters,
)
from adaptive_mini_gpt.model.transformer import MiniGPT
from adaptive_mini_gpt.rules.reliable_transition import get_all_200_transitions


def test_lora_linear_zero_init_identity() -> None:
    """Verify that LoRALinear at initialization identically matches the base linear layer."""
    torch.manual_seed(42)
    base = nn.Linear(32, 64)
    lora = LoRALinear(base_linear=base, r=4, lora_alpha=8.0)

    x = torch.randn(4, 10, 32)
    with torch.no_grad():
        out_base = base(x)
        out_lora = lora(x)

    # Since lora_B is initialized to zeros, base and lora outputs must match exactly
    assert torch.allclose(out_base, out_lora, atol=1e-7), "LoRA did not match base at init!"


def test_lora_adapter_injection_and_parameter_freezing() -> None:
    """Verify adapter injection into MiniGPT attention layers and parameter freezing."""
    cfg = create_1m_control_config(vocab_size=64)
    model = MiniGPT(cfg)

    model, count = inject_lora_adapters(model, r=4, lora_alpha=8.0, target_modules=["c_attn", "c_proj"])
    assert count > 0, "No LoRA adapters were injected!"

    trainable, frozen = freeze_base_parameters(model)
    assert trainable > 0, "No trainable LoRA parameters found!"
    assert frozen > 0, "Base parameters were not frozen!"

    stats = get_lora_parameter_stats(model)
    assert stats["trainable_params"] == trainable
    assert stats["trainable_ratio_pct"] < 5.0, f"Trainable ratio too high: {stats['trainable_ratio_pct']}%"

    # Check that base parameters truly have requires_grad = False
    for name, p in model.named_parameters():
        if "lora_A" in name or "lora_B" in name:
            assert p.requires_grad, f"LoRA parameter {name} should have requires_grad=True"
        else:
            assert not p.requires_grad, f"Base parameter {name} should have requires_grad=False"


def test_lora_toggle_enable_disable() -> None:
    """Verify that toggle_lora_adapters toggles LoRA forward behavior cleanly."""
    base = nn.Linear(16, 16)
    lora = LoRALinear(base_linear=base, r=4, lora_alpha=4.0)

    # Manually set lora_B to non-zero so LoRA modifies the output
    with torch.no_grad():
        lora.lora_B.fill_(1.0)

    x = torch.randn(2, 5, 16)
    out_active = lora(x)
    lora.enabled = False
    out_disabled = lora(x)
    base_out = base(x)

    assert not torch.allclose(out_active, base_out, atol=1e-5), "LoRA did not alter output when active"
    assert torch.allclose(out_disabled, base_out, atol=1e-7), "Disabled LoRA did not match base output"


def test_multiscale_practice_curriculum_disjointness() -> None:
    """Verify multi-scale practice sets (8, 32, 128) have 0.0% overlap with fresh test sets."""
    ep_a = build_multiscale_episode_a(seed=42)
    ep_b = build_multiscale_episode_b(seed=42)
    ep_c = build_multiscale_episode_c(seed=42)

    for ep in [ep_a, ep_b, ep_c]:
        assert len(ep.practice_8) == 8, f"{ep.episode_id} practice_8 has len {len(ep.practice_8)}"
        assert len(ep.practice_32) == 32, f"{ep.episode_id} practice_32 has len {len(ep.practice_32)}"
        assert len(ep.practice_128) == 128, f"{ep.episode_id} practice_128 has len {len(ep.practice_128)}"
        assert len(ep.id_fresh_test) > 0, f"{ep.episode_id} missing ID fresh test"
        assert len(ep.ood_transfer_test) > 0, f"{ep.episode_id} missing OOD test"

        # Check prompt disjointness between practice_128 and test sets
        practice_prompts = {s.prompt for s in ep.practice_128}
        id_prompts = {s.prompt for s in ep.id_fresh_test}
        ood_prompts = {s.prompt for s in ep.ood_transfer_test}

        assert len(practice_prompts & id_prompts) == 0, f"{ep.episode_id} has prompt leakage into ID test!"
        assert len(practice_prompts & ood_prompts) == 0, f"{ep.episode_id} has prompt leakage into OOD test!"
        assert len(id_prompts & ood_prompts) == 0, f"{ep.episode_id} has overlap between ID and OOD tests!"


def test_hash_base_parameters_invariance() -> None:
    """Verify that updating LoRA parameters does NOT change the base parameters hash."""
    cfg = create_1m_control_config(vocab_size=64)
    model = MiniGPT(cfg)
    model, _ = inject_lora_adapters(model, r=4, lora_alpha=8.0)
    freeze_base_parameters(model)

    hash1 = hash_base_parameters(model)

    # Mutate LoRA parameters only
    with torch.no_grad():
        for name, p in model.named_parameters():
            if "lora_B" in name:
                p.add_(0.5)

    hash2 = hash_base_parameters(model)
    assert hash1 == hash2, "Base parameter hash changed after LoRA parameter mutation!"
