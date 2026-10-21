"""Low-Rank Adaptation (LoRA) for Adaptive Mini-GPT (M30 Phase 3).

Implements parameter-efficient low-rank adaptation on linear projections with:
- Frozen base weights (verified immutable).
- Zero-initialized B matrix for exact function preservation at initialization.
- Configurable rank (r=4, r=8) and alpha scaling factor (alpha/r).
- Adapter enable/disable toggle for zero-cost ablation and retention verification.
"""

from __future__ import annotations

import math
from typing import Any, Iterable

import torch
import torch.nn as nn
from torch import Tensor

from adaptive_mini_gpt.model.transformer import MiniGPT


class LoRALinear(nn.Module):
    """Linear layer augmented with trainable Low-Rank Adaptation (LoRA) bypass."""

    def __init__(
        self,
        base_linear: nn.Linear,
        r: int = 4,
        lora_alpha: float = 8.0,
        lora_dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.in_features = base_linear.in_features
        self.out_features = base_linear.out_features
        self.r = r
        self.lora_alpha = lora_alpha
        self.scaling = lora_alpha / r if r > 0 else 1.0
        self.enabled = True

        # Base layer is frozen and preserved
        self.base_linear = base_linear
        for p in self.base_linear.parameters():
            p.requires_grad = False

        dev = base_linear.weight.device
        dt = base_linear.weight.dtype

        if r > 0:
            # Down-projection A: in_features -> r (initialized with Gaussian)
            self.lora_A = nn.Parameter(torch.empty(r, self.in_features, device=dev, dtype=dt))
            # Up-projection B: r -> out_features (initialized with zeros)
            self.lora_B = nn.Parameter(torch.zeros(self.out_features, r, device=dev, dtype=dt))
            # Initialize A with Kaiming uniform
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            if lora_dropout > 0.0:
                self.dropout = nn.Dropout(p=lora_dropout)
            else:
                self.dropout = nn.Identity()
        else:
            self.register_parameter("lora_A", None)
            self.register_parameter("lora_B", None)
            self.dropout = nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass through frozen base linear plus scaled low-rank adapter."""
        base_out = self.base_linear(x)
        if not self.enabled or self.r == 0 or self.lora_A is None or self.lora_B is None:
            return base_out

        # Compute LoRA forward: (dropout(x) @ A.T) @ B.T * scaling
        lora_out = (self.dropout(x) @ self.lora_A.T) @ self.lora_B.T * self.scaling
        return base_out + lora_out

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"r={self.r}, alpha={self.lora_alpha}, scaling={self.scaling}, enabled={self.enabled}"
        )


def inject_lora_adapters(
    model: MiniGPT,
    r: int = 4,
    lora_alpha: float = 8.0,
    target_modules: list[str] | None = None,
    lora_dropout: float = 0.0,
) -> tuple[MiniGPT, int]:
    """Inject LoRALinear adapters into specified linear modules of a MiniGPT model.

    Default target_modules: ["c_attn", "c_proj"] in self-attention blocks.
    Returns the modified model and the number of injected adapter layers.
    """
    targets = target_modules or ["c_attn", "c_proj"]
    injected_count = 0

    for block_idx, block in enumerate(model.blocks):
        # Attention target modules
        if hasattr(block, "attn"):
            for name in targets:
                if hasattr(block.attn, name):
                    submodule = getattr(block.attn, name)
                    if isinstance(submodule, nn.Linear) and not isinstance(submodule, LoRALinear):
                        adapter = LoRALinear(
                            base_linear=submodule,
                            r=r,
                            lora_alpha=lora_alpha,
                            lora_dropout=lora_dropout,
                        ).to(device=submodule.weight.device, dtype=submodule.weight.dtype)
                        setattr(block.attn, name, adapter)
                        injected_count += 1

        # MLP target modules
        if hasattr(block, "mlp"):
            for name in targets:
                if hasattr(block.mlp, name):
                    submodule = getattr(block.mlp, name)
                    if isinstance(submodule, nn.Linear) and not isinstance(submodule, LoRALinear):
                        adapter = LoRALinear(
                            base_linear=submodule,
                            r=r,
                            lora_alpha=lora_alpha,
                            lora_dropout=lora_dropout,
                        ).to(device=submodule.weight.device, dtype=submodule.weight.dtype)
                        setattr(block.mlp, name, adapter)
                        injected_count += 1

    return model, injected_count


def freeze_base_parameters(model: MiniGPT) -> tuple[int, int]:
    """Freeze all non-LoRA parameters and ensure only LoRA parameters require grad.

    Returns:
        (trainable_parameters, frozen_parameters)
    """
    trainable = 0
    frozen = 0
    for name, param in model.named_parameters():
        if "lora_A" in name or "lora_B" in name:
            param.requires_grad = True
            trainable += param.numel()
        else:
            param.requires_grad = False
            frozen += param.numel()

    return trainable, frozen


def toggle_lora_adapters(model: MiniGPT, enabled: bool) -> int:
    """Enable or disable all LoRA adapters in the model.

    Returns the count of toggled adapter modules.
    """
    count = 0
    for module in model.modules():
        if isinstance(module, LoRALinear):
            module.enabled = enabled
            count += 1
    return count


def get_lora_parameter_stats(model: MiniGPT) -> dict[str, Any]:
    """Calculate parameter allocation stats between base and LoRA components."""
    total_params = 0
    trainable_params = 0
    lora_params = 0
    base_params = 0

    for name, param in model.named_parameters():
        total_params += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
        if "lora_A" in name or "lora_B" in name:
            lora_params += param.numel()
        else:
            base_params += param.numel()

    return {
        "total_params": total_params,
        "base_params": base_params,
        "lora_params": lora_params,
        "trainable_params": trainable_params,
        "trainable_ratio_pct": round((trainable_params / max(1, total_params)) * 100.0, 3),
    }
