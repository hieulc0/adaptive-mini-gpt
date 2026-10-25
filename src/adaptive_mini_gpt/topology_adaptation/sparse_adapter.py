"""Milestone M84 Phase 1: Dynamic Sparse Neural Sequence Adapter.

Implements connection-level adaptive neural topology:
1. `SparseMaskedLinear`: Linear layer with explicit binary connection mask.
   - Computes unmasked gradients during backward pass to support gradient-based link regrowth (RigL).
   - Enforces strict zero-masked weight invariant after optimizer steps.
2. `DynamicSparseSequenceAdapter`: Bottleneck sequence adapter with mutable topology.
   - Supports connection deactivation (magnitude pruning, gradient pruning, random pruning).
   - Supports connection activation (gradient-based regrowth, random regrowth).
   - Bounded active-connection budgets (fixed capacity rewiring or dynamic shrinkage).
   - Deterministic mask/weight serialization and bit-exact rollback.
"""

from __future__ import annotations

import copy
import hashlib
import logging
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class PruningCriterion(StrEnum):
    """Criteria for identifying and deactivating connections."""

    MAGNITUDE = "MAGNITUDE"
    GRADIENT_MAGNITUDE = "GRADIENT_MAGNITUDE"
    RANDOM = "RANDOM"
    UTILITY_SCORE = "UTILITY_SCORE"


class RegrowthCriterion(StrEnum):
    """Criteria for proposing and reactivating new connections."""

    GRADIENT_SALIENT = "GRADIENT_SALIENT"
    RANDOM = "RANDOM"
    STATIC = "STATIC"


@dataclass(frozen=True)
class TopologyStats:
    """Quantitative measurement of neural connection topology."""

    total_connections: int
    active_connections: int
    pruned_connections: int
    sparsity_ratio: float
    active_ratio: float
    active_parameters: int
    stored_parameters: int
    topology_hash: str


class MaskedLinearFunction(torch.autograd.Function):
    """Autograd function maintaining unmasked gradients for RigL-style sparse regrowth."""

    @staticmethod
    def forward(
        ctx: Any,
        x: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor | None,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        ctx.save_for_backward(x, weight, bias, mask)
        eff_weight = weight * mask
        return F.linear(x, eff_weight, bias)

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, None]:
        x, weight, bias, mask = ctx.saved_tensors

        # grad_x = grad_output @ eff_weight
        eff_weight = weight * mask
        grad_x = grad_output.matmul(eff_weight)

        # grad_weight unmasked: grad_output.T @ x
        # Handle arbitrary batch dimensions
        grad_out_flat = grad_output.reshape(-1, grad_output.shape[-1])
        x_flat = x.reshape(-1, x.shape[-1])
        grad_weight = grad_out_flat.t().matmul(x_flat)

        grad_bias = None
        if bias is not None:
            grad_bias = grad_out_flat.sum(dim=0)

        return grad_x, grad_weight, grad_bias, None


class SparseMaskedLinear(nn.Module):
    """Linear projection with an explicit mutable connection mask."""

    def __init__(self, in_features: int, out_features: int, bias: bool = True) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter("bias", None)

        # Binary connection mask: 1.0 = active connection, 0.0 = pruned
        self.register_buffer("mask", torch.ones(out_features, in_features, dtype=torch.float32))

        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weight, a=5 ** 0.5)
        if self.bias is not None:
            fan_in = self.in_features
            bound = 1 / (fan_in ** 0.5) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)
        self.mask.fill_(1.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return MaskedLinearFunction.apply(x, self.weight, self.bias, self.mask)

    def count_active_connections(self) -> int:
        return int((self.mask > 0.5).sum().item())

    def count_total_connections(self) -> int:
        return self.weight.numel()

    def get_sparsity(self) -> float:
        total = self.count_total_connections()
        if total == 0:
            return 0.0
        active = self.count_active_connections()
        return (total - active) / total

    def enforce_mask(self) -> None:
        """Zero out any weights whose mask is deactivated."""
        with torch.no_grad():
            self.weight.mul_(self.mask)


class DynamicSparseSequenceAdapter(nn.Module):
    """Parametric sequence adapter with dynamically mutable neural connection topology."""

    def __init__(
        self,
        d_model: int = 128,
        bottleneck_dim: int = 8,
        num_symbols: int = 16,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.bottleneck_dim = bottleneck_dim
        self.num_symbols = num_symbols

        self.ln = nn.LayerNorm(d_model)
        self.down_proj = SparseMaskedLinear(d_model, bottleneck_dim, bias=True)
        self.act = nn.GELU()
        self.up_proj = SparseMaskedLinear(bottleneck_dim, num_symbols, bias=True)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        normed = self.ln(hidden_states)
        z = self.act(self.down_proj(normed))
        return self.up_proj(z)

    def enforce_masks(self) -> None:
        """Ensure all pruned connections have strictly zero weight values."""
        self.down_proj.enforce_mask()
        self.up_proj.enforce_mask()

    def get_topology_stats(self) -> TopologyStats:
        """Compute comprehensive connection and parameter metrics."""
        down_total = self.down_proj.count_total_connections()
        up_total = self.up_proj.count_total_connections()
        total_connections = down_total + up_total

        down_active = self.down_proj.count_active_connections()
        up_active = self.up_proj.count_active_connections()
        active_connections = down_active + up_active
        pruned_connections = total_connections - active_connections

        sparsity_ratio = pruned_connections / total_connections if total_connections > 0 else 0.0
        active_ratio = active_connections / total_connections if total_connections > 0 else 1.0

        # Parameter accounting:
        # LayerNorm: 2 * d_model (256)
        # Biases: bottleneck_dim (8) + num_symbols (16) = 24
        # Active weights: active_connections
        ln_params = sum(p.numel() for p in self.ln.parameters())
        bias_params = (self.down_proj.bias.numel() if self.down_proj.bias is not None else 0) + \
                      (self.up_proj.bias.numel() if self.up_proj.bias is not None else 0)
        active_params = ln_params + bias_params + active_connections
        stored_params = ln_params + bias_params + active_connections * 2  # (weight + index pointer)

        topo_hash = self.compute_topology_hash()

        return TopologyStats(
            total_connections=total_connections,
            active_connections=active_connections,
            pruned_connections=pruned_connections,
            sparsity_ratio=round(sparsity_ratio, 4),
            active_ratio=round(active_ratio, 4),
            active_parameters=active_params,
            stored_parameters=stored_params,
            topology_hash=topo_hash,
        )

    def compute_topology_hash(self) -> str:
        """Fingerprint connection mask and active weights for bit-exact rollback."""
        hasher = hashlib.sha256()
        # Include masks
        hasher.update(self.down_proj.mask.detach().cpu().numpy().tobytes())
        hasher.update(self.up_proj.mask.detach().cpu().numpy().tobytes())
        # Include active parameters
        for p in self.parameters():
            hasher.update(p.detach().cpu().contiguous().numpy().tobytes())
        return hasher.hexdigest()[:16]

    def prune_connections(
        self,
        prune_ratio: float,
        criterion: PruningCriterion = PruningCriterion.MAGNITUDE,
        seed: int | None = None,
    ) -> int:
        """Deactivate a fraction of currently active connections. Returns count of pruned links."""
        if prune_ratio <= 0.0:
            return 0

        self.enforce_masks()
        total_pruned = 0

        for layer in [self.down_proj, self.up_proj]:
            active_indices = (layer.mask > 0.5).nonzero(as_tuple=True)
            num_active = len(active_indices[0])
            if num_active == 0:
                continue

            num_to_prune = int(round(num_active * prune_ratio))
            num_to_prune = max(0, min(num_to_prune, num_active - 1))  # Keep at least 1 active link per layer
            if num_to_prune == 0:
                continue

            if criterion == PruningCriterion.MAGNITUDE:
                scores = layer.weight[active_indices].abs()
                _, prune_k_idx = torch.topk(scores, k=num_to_prune, largest=False)
            elif criterion == PruningCriterion.GRADIENT_MAGNITUDE:
                if layer.weight.grad is not None:
                    scores = layer.weight.grad[active_indices].abs()
                else:
                    scores = layer.weight[active_indices].abs()
                _, prune_k_idx = torch.topk(scores, k=num_to_prune, largest=False)
            elif criterion == PruningCriterion.RANDOM:
                if seed is not None:
                    g = torch.Generator(device=layer.weight.device).manual_seed(seed)
                    perm = torch.randperm(num_active, generator=g, device=layer.weight.device)
                else:
                    perm = torch.randperm(num_active, device=layer.weight.device)
                prune_k_idx = perm[:num_to_prune]
            else:
                scores = layer.weight[active_indices].abs()
                _, prune_k_idx = torch.topk(scores, k=num_to_prune, largest=False)

            prune_rows = active_indices[0][prune_k_idx]
            prune_cols = active_indices[1][prune_k_idx]

            with torch.no_grad():
                layer.mask[prune_rows, prune_cols] = 0.0
                layer.weight[prune_rows, prune_cols] = 0.0

            total_pruned += num_to_prune

        return total_pruned

    def regrow_connections(
        self,
        regrow_count: int,
        criterion: RegrowthCriterion = RegrowthCriterion.GRADIENT_SALIENT,
        seed: int | None = None,
    ) -> int:
        """Reactivate previously pruned connections using gradient saliency. Returns count of regrown links."""
        if regrow_count <= 0:
            return 0

        total_regrown = 0

        for layer in [self.down_proj, self.up_proj]:
            inactive_indices = (layer.mask < 0.5).nonzero(as_tuple=True)
            num_inactive = len(inactive_indices[0])
            if num_inactive == 0:
                continue

            # Apportion regrowth proportionally to layer size
            layer_share = layer.count_total_connections() / (self.down_proj.count_total_connections() + self.up_proj.count_total_connections())
            k_target = max(1, min(int(round(regrow_count * layer_share)), num_inactive))

            if criterion == RegrowthCriterion.GRADIENT_SALIENT:
                if layer.weight.grad is not None:
                    scores = layer.weight.grad[inactive_indices].abs()
                else:
                    scores = torch.randn(num_inactive, device=layer.weight.device)
                _, regrow_k_idx = torch.topk(scores, k=k_target, largest=True)
            elif criterion == RegrowthCriterion.RANDOM:
                if seed is not None:
                    g = torch.Generator(device=layer.weight.device).manual_seed(seed)
                    perm = torch.randperm(num_inactive, generator=g, device=layer.weight.device)
                else:
                    perm = torch.randperm(num_inactive, device=layer.weight.device)
                regrow_k_idx = perm[:k_target]
            else:
                continue

            regrow_rows = inactive_indices[0][regrow_k_idx]
            regrow_cols = inactive_indices[1][regrow_k_idx]

            with torch.no_grad():
                layer.mask[regrow_rows, regrow_cols] = 1.0
                # Initialize regrown weights to zero so topology growth starts function-preserving
                layer.weight[regrow_rows, regrow_cols] = 0.0

            total_regrown += k_target

        return total_regrown

    def rewire_topology(
        self,
        rewire_fraction: float = 0.15,
        prune_criterion: PruningCriterion = PruningCriterion.MAGNITUDE,
        regrow_criterion: RegrowthCriterion = RegrowthCriterion.GRADIENT_SALIENT,
        seed: int | None = None,
    ) -> tuple[int, int]:
        """Balanced topology rewiring (RigL): prunes lowest magnitude links and regrows highest gradient links."""
        stats = self.get_topology_stats()
        k_rewire = int(round(stats.active_connections * rewire_fraction))
        if k_rewire <= 0:
            return 0, 0

        pruned = self.prune_connections(prune_ratio=rewire_fraction, criterion=prune_criterion, seed=seed)
        regrown = self.regrow_connections(regrow_count=pruned, criterion=regrow_criterion, seed=seed)
        return pruned, regrown

    def save_topology_state(self) -> dict[str, Any]:
        """Export complete state for bit-exact rollback."""
        return {
            "state_dict": copy.deepcopy(self.state_dict()),
            "down_mask": self.down_proj.mask.clone(),
            "up_mask": self.up_proj.mask.clone(),
            "topology_hash": self.compute_topology_hash(),
        }

    def load_topology_state(self, checkpoint: dict[str, Any]) -> None:
        """Restore exact connection topology and weights."""
        self.load_state_dict(checkpoint["state_dict"])
        self.down_proj.mask.copy_(checkpoint["down_mask"])
        self.up_proj.mask.copy_(checkpoint["up_mask"])
        self.enforce_masks()
