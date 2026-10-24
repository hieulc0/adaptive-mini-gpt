"""Phase 3: Rational Experiment Controller Avoiding Wasted Experimentation.

Implements Expected Net Value of Experimentation (ENVE):
- Evaluates existing qualified skills before spending training resources.
- If an existing skill is already proficient (F1 >= 0.40), selects REUSE with 0 steps.
- Proposes candidate training only when Expected Net Value of Experimentation is positive
  under the active budget regime.
- Calculates actual held-out validation gain before accepting candidate into repertoire.
- Rolls back strictly if held-out gain <= min_gain_threshold.
"""
from __future__ import annotations

import enum
import logging
from dataclasses import dataclass
from typing import Any

from adaptive_mini_gpt.budgeted_autonomous_learning.budget_regimes import (
    BudgetConstraint,
    BudgetTier,
    CostWeightVector,
)

logger = logging.getLogger(__name__)


class RationalActionType(str, enum.Enum):
    """Action choices determined by the rational experiment controller."""
    REUSE_QUALIFIED_SKILL = "REUSE_QUALIFIED_SKILL"
    EXPERIMENT_AND_ADAPT = "EXPERIMENT_AND_ADAPT"
    BASE_MODEL_INFERENCE = "BASE_MODEL_INFERENCE"


@dataclass
class RationalDecisionResult:
    """Decision output and expected-value accounting for a task."""
    action: RationalActionType
    selected_skill_id: str | None
    probe_f1: float
    expected_gain: float
    estimated_experiment_cost: float
    expected_net_value: float  # ENVE
    reason: str


class RationalExperimentController:
    """Deterministic, value-calibrated controller for experiment decisions."""

    def __init__(
        self,
        proficient_f1_threshold: float = 0.40,
        expected_future_encounters: float = 2.0,
        min_held_out_gain_for_acceptance: float = 0.01,
        cost_weights: CostWeightVector | None = None,
    ) -> None:
        self.proficient_f1_threshold = proficient_f1_threshold
        self.expected_future_encounters = max(1.0, expected_future_encounters)
        self.min_held_out_gain = min_held_out_gain_for_acceptance
        self.weights = cost_weights or CostWeightVector()

    def evaluate_action_for_task(
        self,
        task_name: str,
        best_skill_id: str | None,
        best_skill_probe_f1: float,
        budget: BudgetConstraint,
    ) -> RationalDecisionResult:
        """Determine whether to REUSE or EXPERIMENT based on ENVE and proficiency."""
        # Case 1: Existing skill is already proficient -> strictly REUSE, zero training waste
        if best_skill_id is not None and best_skill_probe_f1 >= self.proficient_f1_threshold:
            return RationalDecisionResult(
                action=RationalActionType.REUSE_QUALIFIED_SKILL,
                selected_skill_id=best_skill_id,
                probe_f1=best_skill_probe_f1,
                expected_gain=0.0,
                estimated_experiment_cost=0.0,
                expected_net_value=0.0,
                reason=(
                    f"Existing skill '{best_skill_id}' is proficient "
                    f"(probe F1={best_skill_probe_f1:.4f} >= {self.proficient_f1_threshold:.4f}); "
                    "declining unnecessary experimentation."
                ),
            )

        # Case 2: Estimate Expected Net Value of Experimentation (ENVE)
        # Expected adapted performance on synthetic relational task with rank-16 adapter ~ 0.65
        projected_adapted_f1 = 0.65
        baseline_f1 = max(0.20, best_skill_probe_f1)
        expected_gain = max(0.0, projected_adapted_f1 - baseline_f1)

        # Training resource requirements bounded by budget constraint
        needed_queries = min(budget.max_training_queries, 16)
        needed_steps = min(budget.max_steps_per_task, 40)

        experiment_cost = (
            self.weights.lambda_query * needed_queries
            + self.weights.lambda_step * needed_steps
        )

        # ENVE discounted over expected future encounters
        enve = (expected_gain * self.expected_future_encounters) - experiment_cost

        # Under SMALL budget, if budget cannot accommodate minimum effective training
        if needed_steps < 10 or needed_queries < 6:
            action = RationalActionType.REUSE_QUALIFIED_SKILL if best_skill_id else RationalActionType.BASE_MODEL_INFERENCE
            return RationalDecisionResult(
                action=action,
                selected_skill_id=best_skill_id,
                probe_f1=best_skill_probe_f1,
                expected_gain=expected_gain,
                estimated_experiment_cost=experiment_cost,
                expected_net_value=enve,
                reason="Budget constraint too tight for effective training; falling back to reuse/base.",
            )

        if enve > 0.05:
            return RationalDecisionResult(
                action=RationalActionType.EXPERIMENT_AND_ADAPT,
                selected_skill_id=None,
                probe_f1=best_skill_probe_f1,
                expected_gain=expected_gain,
                estimated_experiment_cost=experiment_cost,
                expected_net_value=enve,
                reason=(
                    f"ENVE positive ({enve:.4f} > 0.05; expected gain {expected_gain:.4f} "
                    f"amortized across {self.expected_future_encounters:.1f} encounters vs cost {experiment_cost:.4f}); "
                    "training candidate adapter."
                ),
            )

        # If ENVE not positive enough to justify cost, default to best available reuse or base
        action = RationalActionType.REUSE_QUALIFIED_SKILL if best_skill_id else RationalActionType.BASE_MODEL_INFERENCE
        return RationalDecisionResult(
            action=action,
            selected_skill_id=best_skill_id,
            probe_f1=best_skill_probe_f1,
            expected_gain=expected_gain,
            estimated_experiment_cost=experiment_cost,
            expected_net_value=enve,
            reason=f"ENVE insufficient ({enve:.4f} <= 0.05); experimentation cost not justified.",
        )

    def decide_candidate_acceptance(
        self,
        candidate_id: str,
        held_out_val_f1: float,
        baseline_f1: float,
    ) -> tuple[bool, float, str]:
        """Strict empirical gate before retaining candidate weights in repertoire."""
        gain = held_out_val_f1 - baseline_f1
        if gain >= self.min_held_out_gain:
            reason = (
                f"Candidate '{candidate_id}' accepted: held-out validation F1={held_out_val_f1:.4f} "
                f"exceeds baseline {baseline_f1:.4f} by +{gain:.4f} >= {self.min_held_out_gain:.4f}."
            )
            return True, gain, reason

        reason = (
            f"Candidate '{candidate_id}' rejected & rolled back: held-out validation F1={held_out_val_f1:.4f} "
            f"gain +{gain:.4f} below threshold {self.min_held_out_gain:.4f}."
        )
        return False, gain, reason
