"""Phase 3: Bounded Adaptive Training-Budget Controllers for Generalizable Learning.

Compares:
  A. Fixed 10 steps
  B. Fixed 20 steps
  C. Fixed 40 steps
  D. Existing M58 deterministic controller
  E. Improved controller using validation-learning progress and cost (ProgressCostController)
  F. Experiential controller using historical learning records (ExperientialBudgetController)

Supports discrete decisions:
  - KEEP (STOP): Stop adaptation early to conserve compute.
  - TRAIN: Spend one additional block of optimization steps.
  - RESET: Discard harmful prior transfer and train fresh residual candidate.
  - EXPAND: Reference control only (bounded research arm).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Any

from adaptive_mini_gpt.generalizable_learning.experiential_memory import (
    ExperientialMemoryStore,
    ExperientialRecommendation,
)
from adaptive_mini_gpt.generalizable_learning.task_suite import ObservableTaskComplexity

logger = logging.getLogger(__name__)


class BudgetAction(str, Enum):
    """Discrete actions for adaptive learning-budget management."""
    KEEP = "KEEP"      # Stop training; candidate converged or reached plateau
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


# ---------------------------------------------------------------------------
# Base Fixed-Step Controller
# ---------------------------------------------------------------------------

class FixedBudgetController:
    """Fixed-budget controller allocating an exact pre-determined number of steps."""

    def __init__(self, target_steps: int) -> None:
        self.target_steps = target_steps

    def evaluate_budget_action(
        self,
        current_steps: int,
        val_loss_history: list[float],
        complexity: ObservableTaskComplexity | None = None,
        uncertainty_entropy: float = 0.0,
        has_reset: bool = False,
    ) -> ControllerBudgetDecision:
        curr_loss = val_loss_history[-1] if val_loss_history else 1.0
        if current_steps >= self.target_steps:
            return ControllerBudgetDecision(
                action=BudgetAction.KEEP,
                block_steps_allocated=0,
                rationale=f"Fixed budget cap ({self.target_steps} steps) reached.",
                val_loss_current=curr_loss,
                learning_progress_slope=0.0,
                uncertainty_score=uncertainty_entropy,
                total_steps_so_far=current_steps,
                fraction_compute_used=1.0,
            )
        step_block = min(10, self.target_steps - current_steps)
        return ControllerBudgetDecision(
            action=BudgetAction.TRAIN,
            block_steps_allocated=step_block,
            rationale=f"Fixed budget arm allocating +{step_block} steps towards {self.target_steps}.",
            val_loss_current=curr_loss,
            learning_progress_slope=0.0,
            uncertainty_score=uncertainty_entropy,
            total_steps_so_far=current_steps,
            fraction_compute_used=float(current_steps / self.target_steps),
        )


# ---------------------------------------------------------------------------
# M58 Deterministic Controller
# ---------------------------------------------------------------------------

class M58AdaptiveBudgetController:
    """Deterministic M58 controller allocating optimization effort based on loss and slope."""

    def __init__(
        self,
        block_step_size: int = 10,
        max_budget_steps: int = 40,
        convergence_loss_threshold: float = 0.45,
        stagnation_slope_threshold: float = 0.015,
        incompatible_loss_threshold: float = 1.15,
    ) -> None:
        self.block_step_size = block_step_size
        self.max_budget_steps = max_budget_steps
        self.convergence_loss_threshold = convergence_loss_threshold
        self.stagnation_slope_threshold = stagnation_slope_threshold
        self.incompatible_loss_threshold = incompatible_loss_threshold

    def evaluate_budget_action(
        self,
        current_steps: int,
        val_loss_history: list[float],
        complexity: ObservableTaskComplexity | None = None,
        uncertainty_entropy: float = 0.0,
        has_reset: bool = False,
    ) -> ControllerBudgetDecision:
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
        slope = max(0.0, prev_loss - curr_loss)

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

        if not has_reset and curr_loss >= self.incompatible_loss_threshold and slope <= 0.02:
            return ControllerBudgetDecision(
                action=BudgetAction.RESET,
                block_steps_allocated=self.block_step_size,
                rationale=f"Negative transfer detected (loss={curr_loss:.4f} >= {self.incompatible_loss_threshold}); resetting.",
                val_loss_current=curr_loss,
                learning_progress_slope=slope,
                uncertainty_score=uncertainty_entropy,
                total_steps_so_far=current_steps,
                fraction_compute_used=float(current_steps / self.max_budget_steps),
            )

        if curr_loss <= self.convergence_loss_threshold:
            return ControllerBudgetDecision(
                action=BudgetAction.KEEP,
                block_steps_allocated=0,
                rationale=f"Candidate converged (loss={curr_loss:.4f} <= {self.convergence_loss_threshold}); stopping early.",
                val_loss_current=curr_loss,
                learning_progress_slope=slope,
                uncertainty_score=uncertainty_entropy,
                total_steps_so_far=current_steps,
                fraction_compute_used=float(current_steps / self.max_budget_steps),
            )

        if current_steps >= self.block_step_size * 2 and slope < self.stagnation_slope_threshold:
            return ControllerBudgetDecision(
                action=BudgetAction.KEEP,
                block_steps_allocated=0,
                rationale=f"Optimization plateaued (slope={slope:.4f} < {self.stagnation_slope_threshold}); stopping early.",
                val_loss_current=curr_loss,
                learning_progress_slope=slope,
                uncertainty_score=uncertainty_entropy,
                total_steps_so_far=current_steps,
                fraction_compute_used=float(current_steps / self.max_budget_steps),
            )

        steps_to_allocate = min(self.block_step_size, self.max_budget_steps - current_steps)
        return ControllerBudgetDecision(
            action=BudgetAction.TRAIN,
            block_steps_allocated=steps_to_allocate,
            rationale=f"Active learning in progress (loss={curr_loss:.4f}, slope={slope:.4f}); allocating +{steps_to_allocate} steps.",
            val_loss_current=curr_loss,
            learning_progress_slope=slope,
            uncertainty_score=uncertainty_entropy,
            total_steps_so_far=current_steps,
            fraction_compute_used=float(current_steps / self.max_budget_steps),
        )


# ---------------------------------------------------------------------------
# Improved Validation-Learning Progress and Cost Controller (Arm E)
# ---------------------------------------------------------------------------

class ValidationProgressCostController:
    """Controller using marginal validation-learning return per unit compute cost.

    Evaluates the marginal gain:
      marginal_return = (val_loss_prev - val_loss_curr) / compute_block_cost
    Accounts for validation overhead and terminates training when marginal improvement
    is insufficient to justify further compute expenditure.
    """

    def __init__(
        self,
        block_step_size: int = 10,
        max_budget_steps: int = 40,
        min_marginal_return: float = 0.002,  # min loss drop per step
        convergence_loss_threshold: float = 0.42,
        incompatible_loss_threshold: float = 1.10,
    ) -> None:
        self.block_step_size = block_step_size
        self.max_budget_steps = max_budget_steps
        self.min_marginal_return = min_marginal_return
        self.convergence_loss_threshold = convergence_loss_threshold
        self.incompatible_loss_threshold = incompatible_loss_threshold

    def evaluate_budget_action(
        self,
        current_steps: int,
        val_loss_history: list[float],
        complexity: ObservableTaskComplexity | None = None,
        uncertainty_entropy: float = 0.0,
        has_reset: bool = False,
    ) -> ControllerBudgetDecision:
        if not val_loss_history:
            return ControllerBudgetDecision(
                action=BudgetAction.TRAIN,
                block_steps_allocated=self.block_step_size,
                rationale="ProgressCostController: Allocating initial exploration block.",
                val_loss_current=1.0,
                learning_progress_slope=0.0,
                uncertainty_score=uncertainty_entropy,
                total_steps_so_far=current_steps,
                fraction_compute_used=float(current_steps / self.max_budget_steps),
            )

        curr_loss = val_loss_history[-1]
        prev_loss = val_loss_history[-2] if len(val_loss_history) >= 2 else curr_loss
        loss_drop = max(0.0, prev_loss - curr_loss)
        marginal_return = loss_drop / float(self.block_step_size)

        if current_steps >= self.max_budget_steps:
            return ControllerBudgetDecision(
                action=BudgetAction.KEEP,
                block_steps_allocated=0,
                rationale=f"ProgressCostController: Max budget reached ({current_steps}/{self.max_budget_steps}).",
                val_loss_current=curr_loss,
                learning_progress_slope=loss_drop,
                uncertainty_score=uncertainty_entropy,
                total_steps_so_far=current_steps,
                fraction_compute_used=1.0,
            )

        # Incompatible check
        if not has_reset and curr_loss >= self.incompatible_loss_threshold and marginal_return <= 0.002:
            return ControllerBudgetDecision(
                action=BudgetAction.RESET,
                block_steps_allocated=self.block_step_size,
                rationale=(
                    f"ProgressCostController: Severe negative transfer (loss={curr_loss:.4f} >= {self.incompatible_loss_threshold}); "
                    f"marginal return {marginal_return:.4f} <= 0.002. Resetting harmful prior."
                ),
                val_loss_current=curr_loss,
                learning_progress_slope=loss_drop,
                uncertainty_score=uncertainty_entropy,
                total_steps_so_far=current_steps,
                fraction_compute_used=float(current_steps / self.max_budget_steps),
            )

        # Early convergence check
        if curr_loss <= self.convergence_loss_threshold:
            return ControllerBudgetDecision(
                action=BudgetAction.KEEP,
                block_steps_allocated=0,
                rationale=f"ProgressCostController: High task proficiency achieved (loss={curr_loss:.4f}); stopping early.",
                val_loss_current=curr_loss,
                learning_progress_slope=loss_drop,
                uncertainty_score=uncertainty_entropy,
                total_steps_so_far=current_steps,
                fraction_compute_used=float(current_steps / self.max_budget_steps),
            )

        # Cost-benefit stopping: If after at least 20 steps, the return per step is below threshold
        if current_steps >= self.block_step_size * 2 and marginal_return < self.min_marginal_return:
            return ControllerBudgetDecision(
                action=BudgetAction.KEEP,
                block_steps_allocated=0,
                rationale=(
                    f"ProgressCostController: Marginal return {marginal_return:.5f} / step is below economic cost threshold "
                    f"({self.min_marginal_return}); stopping adaptation to prevent compute waste."
                ),
                val_loss_current=curr_loss,
                learning_progress_slope=loss_drop,
                uncertainty_score=uncertainty_entropy,
                total_steps_so_far=current_steps,
                fraction_compute_used=float(current_steps / self.max_budget_steps),
            )

        steps_to_allocate = min(self.block_step_size, self.max_budget_steps - current_steps)
        return ControllerBudgetDecision(
            action=BudgetAction.TRAIN,
            block_steps_allocated=steps_to_allocate,
            rationale=f"ProgressCostController: Positive marginal return ({marginal_return:.4f}/step); allocating +{steps_to_allocate} steps.",
            val_loss_current=curr_loss,
            learning_progress_slope=loss_drop,
            uncertainty_score=uncertainty_entropy,
            total_steps_so_far=current_steps,
            fraction_compute_used=float(current_steps / self.max_budget_steps),
        )


# ---------------------------------------------------------------------------
# Experiential Learning-to-Learn Controller (Arm F)
# ---------------------------------------------------------------------------

class ExperientialBudgetController:
    """Controller using verified historical episode summaries to set budget and decisions.

    Queries ExperientialMemoryStore with observable task topology and initial sample loss
    to retrieve matched prior experiences, establishing a true learning-to-learn mechanism.
    """

    def __init__(
        self,
        memory_store: ExperientialMemoryStore,
        block_step_size: int = 10,
        max_budget_steps: int = 40,
    ) -> None:
        self.memory_store = memory_store
        self.block_step_size = block_step_size
        self.max_budget_steps = max_budget_steps
        self.recommendation: ExperientialRecommendation | None = None

    def evaluate_budget_action(
        self,
        current_steps: int,
        val_loss_history: list[float],
        complexity: ObservableTaskComplexity | None = None,
        uncertainty_entropy: float = 0.0,
        has_reset: bool = False,
    ) -> ControllerBudgetDecision:
        curr_loss = val_loss_history[-1] if val_loss_history else 1.0
        prev_loss = val_loss_history[-2] if len(val_loss_history) >= 2 else curr_loss
        loss_drop = max(0.0, prev_loss - curr_loss)

        # On initial step, query experiential memory
        if self.recommendation is None and complexity is not None:
            # Estimate sample accuracy from loss roughly or pass 0.5
            sample_acc = max(0.0, min(1.0, 1.0 - (curr_loss / 2.0)))
            self.recommendation = self.memory_store.query_recommendation(
                complexity=complexity,
                sample_loss=curr_loss,
                sample_acc=sample_acc,
            )

        rec = self.recommendation
        target_budget = rec.recommended_budget_steps if rec else self.max_budget_steps

        # 1. Negative transfer detected via experiential retrieval
        if not has_reset and rec and rec.negative_transfer_warning and current_steps <= 10:
            return ControllerBudgetDecision(
                action=BudgetAction.RESET,
                block_steps_allocated=self.block_step_size,
                rationale=f"ExperientialController: {rec.rationale}",
                val_loss_current=curr_loss,
                learning_progress_slope=loss_drop,
                uncertainty_score=uncertainty_entropy,
                total_steps_so_far=current_steps,
                fraction_compute_used=float(current_steps / self.max_budget_steps),
            )

        # 2. Reached experiential target budget or max budget
        if current_steps >= min(target_budget, self.max_budget_steps):
            return ControllerBudgetDecision(
                action=BudgetAction.KEEP,
                block_steps_allocated=0,
                rationale=f"ExperientialController: Reached empirically optimal budget ({current_steps} steps).",
                val_loss_current=curr_loss,
                learning_progress_slope=loss_drop,
                uncertainty_score=uncertainty_entropy,
                total_steps_so_far=current_steps,
                fraction_compute_used=1.0,
            )

        # 3. Early convergence
        if curr_loss <= 0.40:
            return ControllerBudgetDecision(
                action=BudgetAction.KEEP,
                block_steps_allocated=0,
                rationale=f"ExperientialController: Early convergence achieved (loss={curr_loss:.4f} <= 0.40).",
                val_loss_current=curr_loss,
                learning_progress_slope=loss_drop,
                uncertainty_score=uncertainty_entropy,
                total_steps_so_far=current_steps,
                fraction_compute_used=float(current_steps / self.max_budget_steps),
            )

        # 4. Allocate step block
        steps_to_allocate = min(self.block_step_size, target_budget - current_steps)
        return ControllerBudgetDecision(
            action=BudgetAction.TRAIN,
            block_steps_allocated=steps_to_allocate,
            rationale=f"ExperientialController: Allocating +{steps_to_allocate} steps towards target {target_budget}.",
            val_loss_current=curr_loss,
            learning_progress_slope=loss_drop,
            uncertainty_score=uncertainty_entropy,
            total_steps_so_far=current_steps,
            fraction_compute_used=float(current_steps / self.max_budget_steps),
        )
