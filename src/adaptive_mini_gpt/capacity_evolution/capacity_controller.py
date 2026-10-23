"""Phase 4: Bounded Adaptive Capacity Controller.

Decides whether to KEEP existing compact capacity, EXPAND residual width,
or RESET to a fresh adapter based on development-set dynamics and observable
task complexity, without accessing hidden rule labels or test sets.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Any

from adaptive_mini_gpt.capacity_evolution.task_complexity import ObservableTaskComplexity

logger = logging.getLogger(__name__)


class CapacityAction(str, Enum):
    """The three discrete actions available to the bounded capacity controller."""
    KEEP = "KEEP"      # Continue with compact residual capacity (d_res = 16)
    EXPAND = "EXPAND"  # Allocate small reversible increase in residual width (16 -> 32)
    RESET = "RESET"    # Discard harmful source representation and adapt from fresh initialization


@dataclass
class ControllerDecision:
    """Action decision trace with transparent justification metrics."""
    action: CapacityAction
    step_n: int
    val_loss_initial: float
    val_loss_current: float
    val_loss_delta: float
    is_learning_stalled: bool
    structural_complexity_index: float
    rationale: str


class AdaptiveCapacityController:
    """Bounded, evidence-driven controller for structural residual evolution.

    Governs structural adaptation based strictly on observable practice signals:
    1. Initial jumpstart cross-entropy loss relative to uniform chance.
    2. Progress rate (validation loss delta) across early practice batches.
    3. Observable topological graph complexity (diameter, size, density).
    """

    def __init__(
        self,
        stagnation_loss_delta_threshold: float = 0.05,
        incompatible_loss_threshold: float = 1.15,
        complexity_expansion_threshold: float = 6.0,
    ) -> None:
        self.stagnation_loss_delta_threshold = stagnation_loss_delta_threshold
        self.incompatible_loss_threshold = incompatible_loss_threshold
        self.complexity_expansion_threshold = complexity_expansion_threshold

    def evaluate_action(
        self,
        step_n: int,
        val_loss_initial: float,
        val_loss_current: float,
        complexity: ObservableTaskComplexity,
    ) -> ControllerDecision:
        """Evaluate observable evidence and select optimal bounded capacity action."""
        val_delta = float(val_loss_initial - val_loss_current)
        stalled = (val_delta < self.stagnation_loss_delta_threshold)
        comp_idx = complexity.structural_complexity_index

        # Rule 1: Negative transfer / incompatible source representation
        # If initial loss is severe (worse than uniform 3-class cross-entropy ln(3) ~ 1.098)
        # and has not recovered significantly, source prior is actively harmful.
        if val_loss_current > self.incompatible_loss_threshold and val_delta < 0.02:
            return ControllerDecision(
                action=CapacityAction.RESET,
                step_n=step_n,
                val_loss_initial=val_loss_initial,
                val_loss_current=val_loss_current,
                val_loss_delta=val_delta,
                is_learning_stalled=stalled,
                structural_complexity_index=comp_idx,
                rationale=(
                    f"Severe negative transfer detected (val_loss={val_loss_current:.4f} > "
                    f"{self.incompatible_loss_threshold:.2f}). Source representation is incompatible; "
                    "resetting to fresh adapter."
                ),
            )

        # Rule 2: Capacity Bottleneck on High-Complexity Graph
        # If observable complexity is high (deep diameter d >= 5 or complexity index >= threshold)
        # AND learning has plateaued under current compact capacity (stalled), expand residual.
        if (complexity.graph_diameter >= 5 or comp_idx >= self.complexity_expansion_threshold) and stalled:
            return ControllerDecision(
                action=CapacityAction.EXPAND,
                step_n=step_n,
                val_loss_initial=val_loss_initial,
                val_loss_current=val_loss_current,
                val_loss_delta=val_delta,
                is_learning_stalled=stalled,
                structural_complexity_index=comp_idx,
                rationale=(
                    f"Capacity bottleneck detected on high-complexity task (diameter={complexity.graph_diameter}, "
                    f"complexity_idx={comp_idx:.2f}) with stalled progress (val_delta={val_delta:.4f} < "
                    f"{self.stagnation_loss_delta_threshold:.2f}). Expanding residual bottleneck from 16 to 32."
                ),
            )

        # Rule 3: Isomorphic / manageable complexity or healthy ongoing progress
        return ControllerDecision(
            action=CapacityAction.KEEP,
            step_n=step_n,
            val_loss_initial=val_loss_initial,
            val_loss_current=val_loss_current,
            val_loss_delta=val_delta,
            is_learning_stalled=stalled,
            structural_complexity_index=comp_idx,
            rationale=(
                f"Compact capacity sufficient (val_delta={val_delta:.4f}, complexity_idx={comp_idx:.2f}). "
                "Maintaining compact residual representation."
            ),
        )
