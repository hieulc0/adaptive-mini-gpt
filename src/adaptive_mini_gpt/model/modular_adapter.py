"""M34 Phase 3: Modular Parameter Adaptation and Multi-Domain LoRA Layers.

Implements parameter-isolated modular adaptation for MiniGPT:
  - MultiAdapterLinear: Linear layer with frozen base weights and a registry of
    independent skill-specific adapters (e.g. "arithmetic", "logic", "language").
  - Runtime dynamic switching via set_active_domain_adapter(name).
  - Strict parameter isolation: when training adapter X, base weights and all other
    adapters are frozen (requires_grad = False).
  - Exact function preservation: zero-initialized B matrix guarantees identity at init.
  - Mathematical weight invariance: base parameter hash verified invariant across all training epochs.
"""

from __future__ import annotations

import copy
import hashlib
import math
from typing import Any, Iterable

import torch
import torch.nn as nn
from torch import Tensor

from adaptive_mini_gpt.model.transformer import MiniGPT


class LoRAAdapter(nn.Module):
    """Single low-rank adapter module (A and B parameter matrices)."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        r: int = 8,
        lora_alpha: float = 16.0,
        lora_dropout: float = 0.0,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.r = r
        self.lora_alpha = lora_alpha
        self.scaling = lora_alpha / r if r > 0 else 1.0

        if r > 0:
            self.lora_A = nn.Parameter(torch.empty(r, in_features, device=device, dtype=dtype))
            self.lora_B = nn.Parameter(torch.zeros(out_features, r, device=device, dtype=dtype))
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            self.dropout = nn.Dropout(p=lora_dropout) if lora_dropout > 0.0 else nn.Identity()
        else:
            self.register_parameter("lora_A", None)
            self.register_parameter("lora_B", None)
            self.dropout = nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        if self.r == 0 or self.lora_A is None or self.lora_B is None:
            return torch.zeros(
                *x.shape[:-1], self.out_features, device=x.device, dtype=x.dtype
            )
        return (self.dropout(x) @ self.lora_A.T) @ self.lora_B.T * self.scaling


class MultiAdapterLinear(nn.Module):
    """Linear layer augmented with multiple named Low-Rank Adapters (LoRA).

    The base linear layer is frozen and shared across all domains.
    Each domain/skill registers its own adapter in `adapters`.
    The active adapter is selectable at runtime via `set_active_adapter(name)`.
    """

    def __init__(self, base_linear: nn.Linear) -> None:
        super().__init__()
        self.in_features = base_linear.in_features
        self.out_features = base_linear.out_features
        self.base_linear = base_linear

        # Freeze base parameters
        for p in self.base_linear.parameters():
            p.requires_grad = False

        self.adapters = nn.ModuleDict()
        self.active_adapter: str | None = None

    def add_adapter(
        self,
        name: str,
        r: int = 8,
        lora_alpha: float = 16.0,
        lora_dropout: float = 0.0,
    ) -> LoRAAdapter:
        """Add a new named adapter."""
        dev = self.base_linear.weight.device
        dt = self.base_linear.weight.dtype
        adapter = LoRAAdapter(
            in_features=self.in_features,
            out_features=self.out_features,
            r=r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            device=dev,
            dtype=dt,
        )
        self.adapters[name] = adapter
        return adapter

    def set_active_adapter(self, name: str | None, allow_missing: bool = False) -> None:
        """Set active adapter name. None disables adapters and runs pure base layer."""
        if name is not None and name not in self.adapters:
            if allow_missing:
                self.active_adapter = None
                return
            raise KeyError(
                f"Adapter '{name}' not found. Registered adapters: {list(self.adapters.keys())}"
            )
        self.active_adapter = name

    def forward(self, x: Tensor) -> Tensor:
        base_out = self.base_linear(x)
        if self.active_adapter is not None and self.active_adapter in self.adapters:
            return base_out + self.adapters[self.active_adapter](x)
        return base_out

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"active_adapter={self.active_adapter}, registered={list(self.adapters.keys())}"
        )


def inject_multi_domain_adapters(
    model: MiniGPT,
    target_modules: list[str] | None = None,
    r: int = 8,
) -> tuple[MiniGPT, int]:
    """Convert linear layers in transformer blocks to MultiAdapterLinear.

    Targets by default attention projections ('c_attn', 'c_proj') and MLP layers ('c_fc', 'c_proj').
    """
    targets = target_modules or ["c_attn", "c_proj", "c_fc"]
    injected_count = 0

    for block in model.blocks:
        # Attention target modules
        if hasattr(block, "attn"):
            for name in targets:
                if hasattr(block.attn, name):
                    sub = getattr(block.attn, name)
                    if isinstance(sub, nn.Linear) and not isinstance(sub, MultiAdapterLinear):
                        wrapped = MultiAdapterLinear(sub)
                        setattr(block.attn, name, wrapped)
                        injected_count += 1

        # MLP target modules
        if hasattr(block, "mlp"):
            for name in targets:
                if hasattr(block.mlp, name):
                    sub = getattr(block.mlp, name)
                    if isinstance(sub, nn.Linear) and not isinstance(sub, MultiAdapterLinear):
                        wrapped = MultiAdapterLinear(sub)
                        setattr(block.mlp, name, wrapped)
                        injected_count += 1

    return model, injected_count


def add_domain_adapter(
    model: MiniGPT,
    domain_name: str,
    r: int = 8,
    lora_alpha: float = 16.0,
    lora_dropout: float = 0.0,
) -> int:
    """Add a named adapter to all MultiAdapterLinear layers across the model."""
    count = 0
    for module in model.modules():
        if isinstance(module, MultiAdapterLinear):
            module.add_adapter(
                name=domain_name,
                r=r,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
            )
            count += 1
    return count


def set_active_domain_adapter(
    model: MiniGPT,
    domain_name: str | None,
    allow_missing: bool = True,
) -> int:
    """Set the active domain adapter across all MultiAdapterLinear modules.
    
    If allow_missing is True and domain_name has not been registered yet,
    deactivates adapters and evaluates base model parameters.
    """
    count = 0
    for module in model.modules():
        if isinstance(module, MultiAdapterLinear):
            module.set_active_adapter(domain_name, allow_missing=allow_missing)
            count += 1
    return count


def freeze_for_adapter_training(
    model: MiniGPT,
    target_domain: str,
) -> tuple[int, int]:
    """Freeze all base parameters AND all other adapters; only target_domain requires grad.

    Returns:
        (trainable_parameters, frozen_parameters)
    """
    set_active_domain_adapter(model, target_domain)
    trainable = 0
    frozen = 0

    for name, param in model.named_parameters():
        if f"adapters.{target_domain}." in name:
            param.requires_grad = True
            trainable += param.numel()
        else:
            param.requires_grad = False
            frozen += param.numel()

    return trainable, frozen


def get_adapter_state_dict(model: MiniGPT, domain_name: str) -> dict[str, Tensor]:
    """Extract isolated state dict containing only parameters for domain_name."""
    state: dict[str, Tensor] = {}
    prefix = f"adapters.{domain_name}."
    for name, param in model.state_dict().items():
        if prefix in name:
            state[name] = param.detach().cpu().clone()
    return state


def load_adapter_state_dict(
    model: MiniGPT,
    domain_name: str,
    adapter_state: dict[str, Tensor],
) -> None:
    """Load state dict for the specified domain adapter."""
    model_state = model.state_dict()
    for k, v in adapter_state.items():
        if k in model_state:
            model_state[k].copy_(v)
        else:
            raise KeyError(f"Key {k} not found in model state dict.")


def compute_base_model_hash(model: MiniGPT) -> str:
    """Compute SHA-256 hash of ONLY base linear & embedding parameters, ignoring adapters and skip connections."""
    hasher = hashlib.sha256()
    state = model.state_dict()
    for key in sorted(state.keys()):
        # Exclude adapters and structural skip connections
        if "adapters." in key or "lora_" in key or "skip_connections." in key:
            continue
        hasher.update(key.encode("utf-8"))
        tensor = state[key].detach().cpu().contiguous()
        hasher.update(tensor.numpy().tobytes())
    return hasher.hexdigest()[:16]
