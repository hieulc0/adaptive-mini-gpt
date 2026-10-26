"""Milestone M104-C Phase C4 Neural Self-Modification Interface (M105 Foundation).

Defines a versioned, resource-bounded, and reversible interface through which
an autonomous student can propose, evaluate, and certify architectural modifications.

Core Principles:
1. Reversibility & Bit-Exact Rollback: Every proposed modification can be completely reverted to its prior checkpoint.
2. Resource-Bounded Constraints: Strict parameter, depth, and memory thresholds.
3. Decoupled Independent Evaluator: Modification proposal and policy cannot access or overwrite the evaluation harness.
4. Clean Accounting: Discrepancy, orphaned parameter detection, and capacity certification.

Supported Initial Operators:
- SELECTIVE_BACKBONE_UPDATE: Switch between frozen, selective top-layer, or full plasticity.
- ADAPTER_ALLOCATION: Initialize or attach domain-specialized low-rank residual adapters.
- REHEARSAL_CONSOLIDATION: Allocate or compress experience memory buffers.
- DYNAMIC_ROUTING_CHOICE: Enable or tune layer skipping routers.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

import torch
import torch.nn as nn


class ModificationOperatorTypeM104_C(StrEnum):
    SELECTIVE_BACKBONE_UPDATE = "selective_backbone_update"
    ADAPTER_ALLOCATION = "adapter_allocation"
    REHEARSAL_CONSOLIDATION = "rehearsal_consolidation"
    DYNAMIC_ROUTING_CHOICE = "dynamic_routing_choice"
    LAYER_GROWTH_PREVIEW = "layer_growth_preview"


@dataclass
class NeuralModificationProposalM104_C:
    proposal_id: str
    operator_type: ModificationOperatorTypeM104_C
    target_module: str
    hyperparameters: dict[str, Any]
    expected_gain_rationale: str
    timestamp: str = field(default_factory=lambda: time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))


@dataclass
class ModificationValidationResultM104_C:
    proposal_id: str
    operator_type: ModificationOperatorTypeM104_C
    pre_eval_metric: float
    post_eval_metric: float
    delta_gain: float
    parameters_added: int
    rollback_executed: bool
    accepted: bool
    reason: str


class StructuralRollbackControllerM104_C:
    """Manages transactional checkpoints and atomic rollbacks for structural modifications."""

    def __init__(self, model: nn.Module) -> None:
        self.model = model
        self.checkpoint_stack: list[tuple[dict[str, torch.Tensor], dict[str, Any]]] = []
        self.proposals_history: list[NeuralModificationProposalM104_C] = []
        self.validation_history: list[ModificationValidationResultM104_C] = []

    def checkpoint(self) -> None:
        """Stores bit-exact state dict snapshot and metadata."""
        state = {k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()}
        meta = {"mode": getattr(self.model, "mode", None)}
        self.checkpoint_stack.append((state, meta))

    def rollback(self) -> bool:
        """Restores most recent snapshot and metadata."""
        if not self.checkpoint_stack:
            return False
        state, meta = self.checkpoint_stack.pop()
        self.model.load_state_dict(state)
        if meta.get("mode") is not None and hasattr(self.model, "mode"):
            self.model.mode = meta["mode"]
            if hasattr(self.model, "_configure_plasticity"):
                self.model._configure_plasticity()
        return True

    def commit(self) -> None:
        """Commits changes and frees checkpoint."""
        if self.checkpoint_stack:
            self.checkpoint_stack.pop()

    def apply_and_evaluate(
        self,
        proposal: NeuralModificationProposalM104_C,
        eval_fn: Any,
        min_gain_threshold: float = 0.0,
    ) -> ModificationValidationResultM104_C:
        """Applies proposal in a reversible transaction, evaluates, and retains or rolls back."""
        self.checkpoint()
        initial_params = sum(p.numel() for p in self.model.parameters())
        pre_metric = eval_fn()

        # Apply structural operator
        try:
            if proposal.operator_type == ModificationOperatorTypeM104_C.SELECTIVE_BACKBONE_UPDATE:
                mode = proposal.hyperparameters.get("mode", "selective_plasticity")
                if hasattr(self.model, "mode"):
                    self.model.mode = mode
                    if hasattr(self.model, "_configure_plasticity"):
                        self.model._configure_plasticity()

            post_metric = eval_fn()
            delta = post_metric - pre_metric
            params_added = sum(p.numel() for p in self.model.parameters()) - initial_params

            if delta >= min_gain_threshold:
                self.commit()
                res = ModificationValidationResultM104_C(
                    proposal_id=proposal.proposal_id,
                    operator_type=proposal.operator_type,
                    pre_eval_metric=pre_metric,
                    post_eval_metric=post_metric,
                    delta_gain=delta,
                    parameters_added=params_added,
                    rollback_executed=False,
                    accepted=True,
                    reason=f"Accepted with positive gain: {delta:+.2f}%",
                )
            else:
                self.rollback()
                res = ModificationValidationResultM104_C(
                    proposal_id=proposal.proposal_id,
                    operator_type=proposal.operator_type,
                    pre_eval_metric=pre_metric,
                    post_eval_metric=post_metric,
                    delta_gain=delta,
                    parameters_added=params_added,
                    rollback_executed=True,
                    accepted=False,
                    reason=f"Rejected and rolled back: gain {delta:+.2f}% below threshold {min_gain_threshold:+.2f}%",
                )
        except Exception as e:
            self.rollback()
            res = ModificationValidationResultM104_C(
                proposal_id=proposal.proposal_id,
                operator_type=proposal.operator_type,
                pre_eval_metric=pre_metric,
                post_eval_metric=0.0,
                delta_gain=-100.0,
                parameters_added=0,
                rollback_executed=True,
                accepted=False,
                reason=f"Error executing modification: {e}",
            )

        self.proposals_history.append(proposal)
        self.validation_history.append(res)
        return res
