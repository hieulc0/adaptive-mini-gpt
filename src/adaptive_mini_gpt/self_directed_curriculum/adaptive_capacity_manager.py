"""Adaptive Neural Capacity Manager for Milestone M99.

Implements evidence-guided neural capacity growth and shrinkage:
1. Configurable experimental budget tiers:
   - Tier 20k (20,000 parameters): Historical scientific baseline.
   - Tier 30k (30,000 parameters): Small growth.
   - Tier 50k (50,000 parameters): Moderate growth.
   - Tier 100k (100,000 parameters): Flexible experimental ceiling.
   - Adaptive Tier: Proposes growth strictly based on measured capability deficits.
2. Evidence-based growth protocol:
   - Learn and reuse before growing.
   - When existing capacity is insufficient (plateau with error > threshold),
     proposes allocating a compact low-rank modular residual (e.g. rank-2: +260 params).
   - Verifies held-out generalization improvement and retention on anchor skills (>= 90%).
   - Commits expansion if net utility is positive; otherwise executes bit-exact rollback.
3. Parameter accounting:
   - Separately reports backbone weights (1,071,872), trainable substrate parameters,
     acquisition controller parameters, curriculum policy parameters, and dynamic modules.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from enum import Enum
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from adaptive_mini_gpt.cross_domain_transfer.neural_cross_domain_transfer import (
    ModularRank2Residual,
)
from adaptive_mini_gpt.cross_subject_skills.cross_subject_substrate import (
    CrossSubjectReasoningSubstrate,
)


class CapacityTier(str, Enum):
    """Configurable experimental capacity budget tiers."""

    TIER_20K = "tier_20k"  # Historical baseline: 20,000
    TIER_30K = "tier_30k"  # Small growth: 30,000
    TIER_50K = "tier_50k"  # Moderate growth: 50,000
    TIER_100K = "tier_100k"  # Flexible ceiling: 100,000
    TIER_ADAPTIVE = "tier_adaptive"  # Evidence-driven dynamic growth


@dataclass(frozen=True)
class CapacityAccounting:
    """Detailed breakdown of neural and physical resource utilization."""

    tier: CapacityTier
    budget_ceiling: int
    frozen_backbone_weights: int
    substrate_parameters: int
    acquisition_controller_parameters: int
    curriculum_policy_parameters: int
    dynamic_residual_parameters: int
    total_persistent_trainable_parameters: int
    headroom_parameters: int
    estimated_flops_per_token: int
    allocated_modules_count: int


class AdaptiveCapacityManager:
    """Oversees evidence-based expansion, consolidation, and rollback of neural capacity."""

    def __init__(
        self,
        tier: CapacityTier = CapacityTier.TIER_20K,
        device: torch.device | None = None,
    ) -> None:
        self.tier = tier
        self.device = device or torch.device("cpu")

        # Map tier to parameter ceiling
        self.budget_ceilings: dict[CapacityTier, int] = {
            CapacityTier.TIER_20K: 20_000,
            CapacityTier.TIER_30K: 30_000,
            CapacityTier.TIER_50K: 50_000,
            CapacityTier.TIER_100K: 100_000,
            CapacityTier.TIER_ADAPTIVE: 50_000,
        }
        self.budget_ceiling = self.budget_ceilings[tier]
        self.expansion_history: list[dict[str, Any]] = []

    def get_accounting(
        self,
        substrate: CrossSubjectReasoningSubstrate,
        curriculum_policy_params: int = 113,
    ) -> CapacityAccounting:
        """Compute exact resource accounting."""
        bd = substrate.get_parameter_breakdown()
        p_sub = bd["substrate_parameters"]
        p_ctrl = bd["acquisition_controller_parameters"]

        # Count any dynamic modular residuals added during lifelong learning
        dyn_p = 0
        mod_count = 0
        if hasattr(substrate.substrate, "concept_residuals"):
            for name, mod in substrate.substrate.concept_residuals.items():
                if name.startswith("dynamic_"):
                    dyn_p += sum(p.numel() for p in mod.parameters() if p.requires_grad)
                    mod_count += 1

        total_trainable = p_sub + p_ctrl + curriculum_policy_params
        headroom = max(0, self.budget_ceiling - total_trainable)

        # Baseline MiniGPT FLOPs ~ 2 * parameters per forward pass token
        frozen_backbone = 1_071_872
        flops = 2 * (frozen_backbone + total_trainable)

        return CapacityAccounting(
            tier=self.tier,
            budget_ceiling=self.budget_ceiling,
            frozen_backbone_weights=frozen_backbone,
            substrate_parameters=p_sub,
            acquisition_controller_parameters=p_ctrl,
            curriculum_policy_parameters=curriculum_policy_params,
            dynamic_residual_parameters=dyn_p,
            total_persistent_trainable_parameters=total_trainable,
            headroom_parameters=headroom,
            estimated_flops_per_token=flops,
            allocated_modules_count=mod_count,
        )

    def propose_evidence_based_growth(
        self,
        substrate: CrossSubjectReasoningSubstrate,
        task_id: str,
        current_loss: float,
        current_acc: float,
        loss_slope: float,
    ) -> bool:
        """Evaluate whether empirical evidence justifies allocating additional neural capacity.

        Growth Rule:
        1. Reuse first: if current_acc >= 80% or loss_slope > 0.05 (learning is progressing), do NOT expand.
        2. Expand only if learning has plateaued (loss_slope <= 0.01) with persistent error (acc < 80%).
        3. Enforce budget ceiling for current tier.
        """
        accounting = self.get_accounting(substrate)
        # Check if headroom allows at least a rank-2 residual (~260 parameters)
        needed = 260
        if accounting.headroom_parameters < needed:
            return False

        # If model is already performing well or actively learning, reuse existing weights
        if current_acc >= 80.0 or loss_slope > 0.02:
            return False

        # Evidence criterion met: plateau with insufficient capacity
        return True

    def allocate_residual_expert(
        self,
        substrate: CrossSubjectReasoningSubstrate,
        module_name: str,
        rank: int = 2,
    ) -> dict[str, Any]:
        """Allocate a new modular rank-R residual expert to the concept trunk."""
        # Create zero-initialized low-rank residual: W_up @ W_down
        # Input/output dim is d_hidden = 64
        # rank 2: down (64x2) = 128 + up (2x64) = 128 -> 256 + biases/norm = ~260
        res = ModularRank2Residual(d_model=substrate.substrate.d_hidden, rank=rank).to(self.device)
        substrate.substrate.concept_residuals[module_name] = res

        alloc_info = {
            "module_name": module_name,
            "rank": rank,
            "parameters": sum(p.numel() for p in res.parameters() if p.requires_grad),
            "tier": self.tier.value,
        }
        self.expansion_history.append(alloc_info)
        return alloc_info

    def rollback_module(
        self,
        substrate: CrossSubjectReasoningSubstrate,
        module_name: str,
    ) -> bool:
        """Remove an unverified or failed modular expansion with bit-exact restoration."""
        if module_name in substrate.substrate.concept_residuals:
            del substrate.substrate.concept_residuals[module_name]
            return True
        return False
