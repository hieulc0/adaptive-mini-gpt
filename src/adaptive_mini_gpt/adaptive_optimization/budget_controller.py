"""Phase 3: Minimal Bounded Adaptive Training-Budget Controller.

Implements autonomous allocation of optimization effort across discrete actions:
  - KEEP: Stop adaptation and retain current qualified candidate (prevents compute waste).
  - TRAIN: Allocate one additional fixed block of optimization steps (e.g. +10 steps).
  - RESET: Discard harmful prior transfer and train fresh residual candidate.
  - EXPAND: Optional M57 structural expansion control strictly as a reference baseline.

All decisions are driven strictly by observable validation loss trajectory,
learning-progress slope, and uncertainty—without hidden rule labels or test peeking.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Any

from adaptive_mini_gpt.adaptive_optimization.task_environments import ObservableTaskComplexity

logger = logging.getLogger(__name__)


class BudgetAction(str, Enum):
    """Discrete actions for adaptive learning-budget management."""
    KEEP = "KEEP"      # Stop training; candidate is adequate or converged
    TRAIN = "TRAIN"    # Spend one additional block of optimization steps
    RESET = "RESET"    # Abandon harmful transfer; reset to fresh adapter
    EXPAND = "EXPAND"  # Reference control: expand residual width


@dataclass
class ControllerBudgetDecision:
    """Audit log of an adaptive budget controller decision."""
    action: BudgetAction
    block_steps_allocated: int
    rationale: str
    val_loss_current: float
    learning_progress_slope: float
    uncertainty_score: float
    total_steps_so_far: int
    fraction_compute_used: float


class AdaptiveBudgetController:
    """Deterministic bounded controller allocating optimization effort evidence-first."""

    def __init__(
        self,
        block_step_size: int = 10,
        max_budget_steps: int = 40,
        convergence_loss_threshold: float = 0.45,
        stagnation_slope_threshold: float = 0.015,
        incompatible_loss_threshold: float = 1.15,
        allow_expansion_reference: bool = False,
    ) -> None:
        self.block_step_size = block_step_size
        self.max_budget_steps = max_budget_steps
        self.convergence_loss_threshold = convergence_loss_threshold
        self.stagnation_slope_threshold = stagnation_slope_threshold
        self.incompatible_loss_threshold = incompatible_loss_threshold
        self.allow_expansion_reference = allow_expansion_reference

    def evaluate_budget_action(
        self,
        current_steps: int,
        val_loss_history: list[float],
        complexity: ObservableTaskComplexity | None = None,
        uncertainty_entropy: float = 0.0,
        has_reset: bool = False,
    ) -> ControllerBudgetDecision:
        """Decide whether to KEEP, TRAIN, or RESET based strictly on observable signals."""
        if not val_loss_history:
            return ControllerBudgetDecision(
                action=BudgetAction.TRAIN,
                block_steps_allocated=self.block_step_size,
                rationale="Initial adaptation block allocation.",
                val_loss_current=1.0,
                learning_progress_slope=0.0,
                uncertainty_score=uncertainty_entropy,
                total_steps_so_far=current_steps,
                fraction_compute_used=float(current_steps / self.max_budget_steps),
            )

        curr_loss = val_loss_history[-1]
        prev_loss = val_loss_history[-2] if len(val_loss_history) >= 2 else curr_loss
        slope = max(0.0, prev_loss - curr_loss)  # positive slope indicates improvement

        # 1. Budget Cap Exceeded -> KEEP
        if current_steps >= self.max_budget_steps:
            return ControllerBudgetDecision(
                action=BudgetAction.KEEP,
                block_steps_allocated=0,
                rationale=f"Max budget reached ({current_steps}/{self.max_budget_steps} steps).",
                val_loss_current=curr_loss,
                learning_progress_slope=slope,
                uncertainty_score=uncertainty_entropy,
                total_steps_so_far=current_steps,
                fraction_compute_used=1.0,
            )

        # 2. Harmful Negative Transfer -> RESET (if not already reset)
        if not has_reset and curr_loss >= self.incompatible_loss_threshold and slope < 0.01:
            return ControllerBudgetDecision(
                action=BudgetAction.RESET,
                block_steps_allocated=self.block_step_size,
                rationale=(
                    f"Severe negative transfer detected (val_loss={curr_loss:.4f} >= {self.incompatible_loss_threshold}); "
                    f"resetting harmful prior to fresh residual."
                ),
                val_loss_current=curr_loss,
                learning_progress_slope=slope,
                uncertainty_score=uncertainty_entropy,
                total_steps_so_far=current_steps,
                fraction_compute_used=float(current_steps / self.max_budget_steps),
            )

        # 3. Mastered / Low Loss -> KEEP (Stop early, save compute!)
        if curr_loss <= self.convergence_loss_threshold:
            return ControllerBudgetDecision(
                action=BudgetAction.KEEP,
                block_steps_allocated=0,
                rationale=(
                    f"Candidate converged to target proficiency (val_loss={curr_loss:.4f} <= {self.convergence_loss_threshold}). "
                    f"Stopping early to preserve compute budget."
                ),
                val_loss_current=curr_loss,
                learning_progress_slope=slope,
                uncertainty_score=uncertainty_entropy,
                total_steps_so_far=current_steps,
                fraction_compute_used=float(current_steps / self.max_budget_steps),
            )

        # 4. Progress Plateau / Stagnation -> KEEP (Avoid wasting steps on flat plateau)
        if current_steps >= self.block_step_size * 2 and slope < self.stagnation_slope_threshold:
            # Check if expansion reference is enabled for high complexity
            if self.allow_expansion_reference and complexity and complexity.graph_diameter >= 6:
                return ControllerBudgetDecision(
                    action=BudgetAction.EXPAND,
                    block_steps_allocated=self.block_step_size,
                    rationale=f"Stalled on deep graph (diameter={complexity.graph_diameter}); testing expansion reference.",
                    val_loss_current=curr_loss,
                    learning_progress_slope=slope,
                    uncertainty_score=uncertainty_entropy,
                    total_steps_so_far=current_steps,
                    fraction_compute_used=float(current_steps / self.max_budget_steps),
                )
            return ControllerBudgetDecision(
                action=BudgetAction.KEEP,
                block_steps_allocated=0,
                rationale=(
                    f"Optimization progress plateaued (recent slope={slope:.4f} < {self.stagnation_slope_threshold}). "
                    f"Halting adaptation to avoid overfitting and wasteful compute."
                ),
                val_loss_current=curr_loss,
                learning_progress_slope=slope,
                uncertainty_score=uncertainty_entropy,
                total_steps_so_far=current_steps,
                fraction_compute_used=float(current_steps / self.max_budget_steps),
            )

        # 5. Healthy Improvement -> TRAIN (Allocate another block)
        steps_to_allocate = min(self.block_step_size, self.max_budget_steps - current_steps)
        return ControllerBudgetDecision(
            action=BudgetAction.TRAIN,
            block_steps_allocated=steps_to_allocate,
            rationale=(
                f"Active learning in progress (val_loss={curr_loss:.4f}, slope={slope:.4f}). "
                f"Allocating +{steps_to_allocate} optimization steps."
            ),
            val_loss_current=curr_loss,
            learning_progress_slope=slope,
            uncertainty_score=uncertainty_entropy,
            total_steps_so_far=current_steps,
            fraction_compute_used=float(current_steps / self.max_budget_steps),
        )


class HeuristicLearnedBudgetController:
    """Comparative heuristic controller incorporating task complexity weighting and entropy."""

    def __init__(
        self,
        base_controller: AdaptiveBudgetController,
        complexity_weight: float = 0.05,
        entropy_weight: float = 0.10,
    ) -> None:
        self.base_controller = base_controller
        self.complexity_weight = complexity_weight
        self.entropy_weight = entropy_weight

    def evaluate_budget_action(
        self,
        current_steps: int,
        val_loss_history: list[float],
        complexity: ObservableTaskComplexity | None = None,
        uncertainty_entropy: float = 0.0,
        has_reset: bool = False,
    ) -> ControllerBudgetDecision:
        """Adjusts stagnation threshold dynamically based on structural complexity index."""
        eff_threshold = self.base_controller.stagnation_slope_threshold
        if complexity:
            # Complex tasks (higher diameter) tolerate slightly slower progress before stopping
            eff_threshold = max(0.005, eff_threshold - self.complexity_weight * (complexity.graph_diameter / 10.0))

        # Delegate with adjusted sensitivity
        curr_loss = val_loss_history[-1] if val_loss_history else 1.0
        prev_loss = val_loss_history[-2] if len(val_loss_history) >= 2 else curr_loss
        slope = max(0.0, prev_loss - curr_loss)

        if current_steps >= self.base_controller.max_budget_steps:
            return self.base_controller.evaluate_budget_action(
                current_steps, val_loss_history, complexity, uncertainty_entropy, has_reset
            )

        if current_steps >= self.base_controller.block_step_size * 2 and slope < eff_threshold:
            return ControllerBudgetDecision(
                action=BudgetAction.KEEP,
                block_steps_allocated=0,
                rationale=f"Heuristic controller detected convergence for complexity index {complexity.structural_complexity_index if complexity else 0.0:.2f}.",
                val_loss_current=curr_loss,
                learning_progress_slope=slope,
                uncertainty_score=uncertainty_entropy,
                total_steps_so_far=current_steps,
                fraction_compute_used=float(current_steps / self.base_controller.max_budget_steps),
            )

        return self.base_controller.evaluate_budget_action(
            current_steps, val_loss_history, complexity, uncertainty_entropy, has_reset
        )
