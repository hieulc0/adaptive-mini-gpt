"""Function-preserving structural growth operators for M4 and M5 experiments."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from adaptive_mini_gpt.adaptive.experts import BottleneckExpert
from adaptive_mini_gpt.adaptive.mod import MixtureOfDepthsRouter

if TYPE_CHECKING:
    from adaptive_mini_gpt.model.transformer import MiniGPT, TransformerBlock


def nn_init_zeros_residual(block: TransformerBlock) -> None:
    """Zero out output projections of attention, MLP, and MoE in a block."""
    torch.nn.init.zeros_(block.attn.c_proj.weight)
    if block.attn.c_proj.bias is not None:
        torch.nn.init.zeros_(block.attn.c_proj.bias)

    torch.nn.init.zeros_(block.mlp.c_proj.weight)
    if block.mlp.c_proj.bias is not None:
        torch.nn.init.zeros_(block.mlp.c_proj.bias)

    if block.moe is not None:
        for exp in block.moe.experts:
            if isinstance(exp, BottleneckExpert):
                torch.nn.init.zeros_(exp.w_down.weight)
                if exp.w_down.bias is not None:
                    torch.nn.init.zeros_(exp.w_down.bias)


@torch.no_grad()
def grow_model_experts(
    model: MiniGPT,
    add_experts: int = 2,
    freeze_backbone: bool = False,
    freeze_old_experts: bool = False,
    trainable_token_ids: list[int] | None = None,
) -> MiniGPT:
    """Enable or expand modular experts across all Transformer blocks in-place.

    Because newly added experts have W_down = 0, the model's forward logits are
    preserved at the exact step of growth.
    """
    from adaptive_mini_gpt.model.transformer import TransformerBlock

    if not model.config.adaptive_enabled:
        model.config.adaptive_enabled = True
        model.config.n_experts = add_experts
        model.config.top_k = min(model.config.top_k, add_experts)
        device = next(model.parameters()).device
        dtype = next(model.parameters()).dtype
        for block in model.blocks:
            assert isinstance(block, TransformerBlock)
            block.enable_zero_init_moe(model.config, device=device, dtype=dtype)
    else:
        for block in model.blocks:
            assert isinstance(block, TransformerBlock)
            if block.moe is not None:
                block.moe.add_experts(add_experts, freeze_existing=freeze_old_experts)
        model.config.n_experts += add_experts

    if freeze_backbone:
        model.freeze_backbone(
            freeze_existing_experts=freeze_old_experts,
            trainable_token_ids=trainable_token_ids,
        )

    return model


@torch.no_grad()
def enable_model_mod(
    model: MiniGPT,
    capacity_factor: float = 0.5,
) -> MiniGPT:
    """Enable M5 Mixture-of-Depths token-level routing across all Transformer blocks."""
    from adaptive_mini_gpt.model.transformer import TransformerBlock

    if not (0.0 < capacity_factor <= 1.0):
        raise ValueError(f"capacity_factor must be in (0.0, 1.0], got {capacity_factor}")

    model.config.mod_enabled = True
    model.config.mod_capacity_factor = capacity_factor
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype

    for block in model.blocks:
        assert isinstance(block, TransformerBlock)
        block.config.mod_enabled = True
        block.config.mod_capacity_factor = capacity_factor
        block.mod_router = MixtureOfDepthsRouter(
            d_model=model.config.d_model,
            capacity_factor=capacity_factor,
            bias=model.config.bias,
        ).to(device=device, dtype=dtype)

    return model


@torch.no_grad()
def grow_model_depth(
    model: MiniGPT,
    add_layers: int = 1,
) -> MiniGPT:
    """Append identity-initialized (zero residual projection) Transformer blocks."""
    from adaptive_mini_gpt.model.transformer import TransformerBlock

    if add_layers <= 0:
        raise ValueError(f"add_layers must be positive, got {add_layers}")

    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype

    for _ in range(add_layers):
        new_block = TransformerBlock(model.config).to(device=device, dtype=dtype)
        nn_init_zeros_residual(new_block)
        model.blocks.append(new_block)

    model.config.n_layers += add_layers
    return model
