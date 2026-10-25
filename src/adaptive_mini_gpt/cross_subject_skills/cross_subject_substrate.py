"""Cross-Subject Neural Substrate & Integrated Acquisition Architecture for Milestone M98.

Implements:
1. Unified Neural Substrate (17,116 trainable parameters) supporting:
   - Algebra operations (modular arithmetic, linear maps)
   - Graph transitions (adjacency traversal, reachability)
   - Logic predicates (Boolean rules, threshold guards)
   - Cross-subject compositions (cascaded execution)
2. Integrated Learned Acquisition Controller (1,835 trainable parameters):
   - Dynamically predicts initialization, learning rate, rank capacity, and early stopping.
3. Total Persistent Trainable Capacity:
   - 17,116 (Substrate) + 1,835 (Controller) = 18,951 parameters <= 20,000 budget ceiling.
   - Headroom: 1,049 parameters.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from adaptive_mini_gpt.cross_domain_transfer.neural_cross_domain_transfer import (
    CrossDomainReasoningSubstrate,
    MODULUS,
    NUM_SYMBOLS,
    OP_ADD,
    OP_ASSIGN,
    OP_IF_GT,
    OP_SUB,
)
from adaptive_mini_gpt.meta_acquisition.controller import (
    AcquisitionAction,
    AcquisitionState,
    LearnedAcquisitionController,
)


class CrossSubjectReasoningSubstrate(nn.Module):
    """Integrated neural architecture for cross-subject reasoning and adaptive acquisition."""

    def __init__(
        self,
        budget_ceiling: int = 20_000,
        device: torch.device | None = None,
    ) -> None:
        super().__init__()
        self.budget_ceiling = budget_ceiling
        self.substrate = CrossDomainReasoningSubstrate(budget_ceiling=budget_ceiling)
        self.controller = LearnedAcquisitionController(in_features=8, hidden_dim=32)

        if device is not None:
            self.to(device)

    def forward_transition(
        self,
        s: torch.Tensor,
        op: torch.Tensor,
        v: torch.Tensor,
        domain_key: str = "PROGRAM",
        concept_keys: list[str] | None = None,
    ) -> torch.Tensor:
        """Execute neural transition step through the substrate."""
        return self.substrate.forward_transition(
            state_val=s,
            op_code=op,
            op_val=v,
            domain_key=domain_key,
        )

    def evaluate_controller(self, state: AcquisitionState) -> AcquisitionAction:
        """Sample or select action from the learned acquisition controller."""
        return self.controller.select_action(state=state, deterministic=True)

    def count_parameters(self) -> int:
        """Total trainable parameters across substrate and controller."""
        p_sub = sum(p.numel() for p in self.substrate.parameters() if p.requires_grad)
        p_ctrl = sum(p.numel() for p in self.controller.parameters() if p.requires_grad)
        return p_sub + p_ctrl

    def get_parameter_breakdown(self) -> dict[str, int]:
        """Breakdown of trainable parameters."""
        p_sub = sum(p.numel() for p in self.substrate.parameters() if p.requires_grad)
        p_ctrl = sum(p.numel() for p in self.controller.parameters() if p.requires_grad)
        return {
            "substrate_parameters": p_sub,
            "acquisition_controller_parameters": p_ctrl,
            "total_trainable_parameters": p_sub + p_ctrl,
            "budget_ceiling": self.budget_ceiling,
            "budget_headroom": self.budget_ceiling - (p_sub + p_ctrl),
        }
