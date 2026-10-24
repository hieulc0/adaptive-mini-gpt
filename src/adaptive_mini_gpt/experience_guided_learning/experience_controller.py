"""Phase 3 & 6: Experience-Guided Autonomous Training Controller & Adaptive Step Budgets.

Implements:
- Conservative, bounded learning action selection:
    * Actions: REUSE_EXISTING, FRESH_ADAPT, TRAIN_CANDIDATE
    * Step Budgets: 20, 40, 60 optimizer steps
    * Learning Rates: 1e-3, 3e-3, 5e-3
- Empirical similarity-based calibration on observable pre-training task descriptors.
- Uncertainty-aware utility discounting for conservative policy.
- Seven evaluated policy regimes:
    1. NO_HISTORY: Memory-free baseline (fixed 50 steps, dev reuse threshold).
    2. RELEVANT_HISTORY: Adaptive selection using verified prior experiences.
    3. SHUFFLED_HISTORY: Identical record count with scrambled associations.
    4. IRRELEVANT_HISTORY: History from unrelated synthetic tasks.
    5. CONSERVATIVE_HISTORY: Lower-confidence-bound uncertainty discounting.
    6. FIXED_BUDGET_FRESH: Always fresh adapt with fixed 40-step budget.
    7. OFFLINE_HINDSIGHT: Diagnostic upper-bound oracle.
- Fallback guarantee: seamlessly reverts to memory-free baseline when history is empty or sparse.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from enum import StrEnum

import numpy as np

from adaptive_mini_gpt.experience_guided_learning.experience_records import (
    LearningAction,
    LearningActionType,
    LearningExperienceManifest,
    LearningExperienceRecord,
    ObservableTaskDescriptors,
)

logger = logging.getLogger(__name__)


class ControllerPolicyMode(StrEnum):
    """Evaluated experience-guided policy modes for Phase 4 causal comparison."""

    NO_HISTORY = "NO_HISTORY"
    RELEVANT_HISTORY = "RELEVANT_HISTORY"
    SHUFFLED_HISTORY = "SHUFFLED_HISTORY"
    IRRELEVANT_HISTORY = "IRRELEVANT_HISTORY"
    CONSERVATIVE_HISTORY = "CONSERVATIVE_HISTORY"
    FIXED_BUDGET_FRESH = "FIXED_BUDGET_FRESH"
    OFFLINE_HINDSIGHT = "OFFLINE_HINDSIGHT"


@dataclass
class BudgetRecommendation:
    """Action and budget recommendation with diagnostic attribution."""

    action: LearningAction
    predicted_dev_em: float
    predicted_utility: float
    confidence_penalty: float
    nearest_neighbors_count: int
    decision_reasoning: str


class ExperienceGuidedController:
    """Bounded, conservative controller optimizing learning actions and compute budgets."""

    PERMITTED_BUDGETS = (20, 40, 60)
    PERMITTED_LRS = (0.001, 0.003, 0.005)
    REUSE_ACCURACY_THRESHOLD = 0.80
    COST_WEIGHT = 0.15  # Penalty weight per unit compute (steps / 60)

    def __init__(
        self,
        mode: ControllerPolicyMode = ControllerPolicyMode.RELEVANT_HISTORY,
        temperature: float = 0.5,
        min_records_for_inference: int = 2,
    ) -> None:
        self.mode = mode
        self.temperature = temperature
        self.min_records_for_inference = min_records_for_inference

    def select_action_and_budget(
        self,
        descriptors: ObservableTaskDescriptors,
        repertoire_skill_ids: list[str],
        history: LearningExperienceManifest | None,
        best_repertoire_skill_id: str | None = None,
        hindsight_oracle_budget: int | None = None,
    ) -> BudgetRecommendation:
        """Select bounded learning action, step budget, and learning rate."""
        # 1. FIXED_BUDGET_FRESH: Always fresh adapt with fixed 40 steps, lr=3e-3
        if self.mode == ControllerPolicyMode.FIXED_BUDGET_FRESH:
            return BudgetRecommendation(
                action=LearningAction(
                    action_type=LearningActionType.FRESH_ADAPT,
                    step_budget=40,
                    learning_rate=0.003,
                ),
                predicted_dev_em=0.70,
                predicted_utility=0.70 - self.COST_WEIGHT * (40 / 60),
                confidence_penalty=0.0,
                nearest_neighbors_count=0,
                decision_reasoning="Fixed fresh adaptation policy: 40 steps, lr=0.003",
            )

        # 2. Check if an existing qualified skill already satisfies the dev threshold
        if (
            descriptors.best_repertoire_em >= self.REUSE_ACCURACY_THRESHOLD
            and best_repertoire_skill_id is not None
            and self.mode != ControllerPolicyMode.OFFLINE_HINDSIGHT
        ):
            return BudgetRecommendation(
                action=LearningAction(
                    action_type=LearningActionType.REUSE_EXISTING,
                    step_budget=0,
                    learning_rate=0.0,
                    reused_skill_id=best_repertoire_skill_id,
                ),
                predicted_dev_em=descriptors.best_repertoire_em,
                predicted_utility=descriptors.best_repertoire_em,
                confidence_penalty=0.0,
                nearest_neighbors_count=0,
                decision_reasoning=f"High zero-shot dev accuracy ({descriptors.best_repertoire_em:.2f} >= {self.REUSE_ACCURACY_THRESHOLD}) on skill {best_repertoire_skill_id}",
            )

        # 3. NO_HISTORY: Static fallback policy
        if (
            self.mode == ControllerPolicyMode.NO_HISTORY
            or history is None
            or history.count() < self.min_records_for_inference
        ):
            # Memory-free baseline: Train candidate with standard fixed 40 steps, lr=0.003
            return BudgetRecommendation(
                action=LearningAction(
                    action_type=LearningActionType.TRAIN_CANDIDATE,
                    step_budget=40,
                    learning_rate=0.003,
                ),
                predicted_dev_em=0.70,
                predicted_utility=0.70 - self.COST_WEIGHT * (40 / 60),
                confidence_penalty=0.0,
                nearest_neighbors_count=0,
                decision_reasoning="Memory-free baseline: default 40 steps, lr=0.003",
            )

        # 4. OFFLINE_HINDSIGHT: Diagnostic upper-bound oracle
        if self.mode == ControllerPolicyMode.OFFLINE_HINDSIGHT and hindsight_oracle_budget is not None:
            return BudgetRecommendation(
                action=LearningAction(
                    action_type=LearningActionType.TRAIN_CANDIDATE,
                    step_budget=hindsight_oracle_budget,
                    learning_rate=0.003,
                ),
                predicted_dev_em=1.0,
                predicted_utility=1.0 - self.COST_WEIGHT * (hindsight_oracle_budget / 60),
                confidence_penalty=0.0,
                nearest_neighbors_count=history.count(),
                decision_reasoning=f"Hindsight oracle upper bound: optimal budget {hindsight_oracle_budget} steps",
            )

        # 5. Experience-Guided Decision (RELEVANT_HISTORY, SHUFFLED_HISTORY, IRRELEVANT_HISTORY, CONSERVATIVE_HISTORY)
        curr_vec = np.array(descriptors.feature_vector(), dtype=np.float32)

        # Compute distances and weights to historical experiences
        hist_records: list[LearningExperienceRecord] = history.records
        dists = []
        for r in hist_records:
            h_vec = np.array(r.descriptors.feature_vector(), dtype=np.float32)
            d = float(np.linalg.norm(curr_vec - h_vec))
            dists.append(d)

        # Softmax similarity weights
        dists_arr = np.array(dists, dtype=np.float32)
        weights = np.exp(-dists_arr / max(self.temperature, 0.05))
        w_sum = float(np.sum(weights))
        if w_sum > 0:
            weights = weights / w_sum
        else:
            weights = np.ones_like(weights) / len(weights)

        # Evaluate candidate step budgets: 20, 40, 60 steps
        candidate_evaluations: list[tuple[int, float, float, float, int]] = []
        for budget in self.PERMITTED_BUDGETS:
            # Filter historical records with this budget
            matching_indices = [
                i for i, r in enumerate(hist_records) if r.action.step_budget == budget
            ]
            if not matching_indices:
                # Interpolate if budget unobserved
                matching_indices = list(range(len(hist_records)))

            sub_weights = weights[matching_indices]
            sw_sum = float(np.sum(sub_weights))
            if sw_sum > 0:
                norm_sub_weights = sub_weights / sw_sum
            else:
                norm_sub_weights = np.ones_like(sub_weights) / len(sub_weights)

            sub_ems = [hist_records[i].dev_em for i in matching_indices]
            pred_em = float(np.sum(norm_sub_weights * np.array(sub_ems)))

            # Sample variance
            if len(matching_indices) > 1:
                weighted_var = float(
                    np.sum(norm_sub_weights * (np.array(sub_ems) - pred_em) ** 2)
                )
                pred_std = math.sqrt(weighted_var)
            else:
                pred_std = 0.15

            # Compute penalty
            n_eff = max(1.0, 1.0 / float(np.sum(norm_sub_weights**2)))
            penalty = 0.0
            if self.mode == ControllerPolicyMode.CONSERVATIVE_HISTORY:
                # Lower confidence bound penalty
                penalty = 1.96 * (pred_std / math.sqrt(n_eff))

            compute_cost = self.COST_WEIGHT * (budget / 60.0)
            net_utility = pred_em - penalty - compute_cost

            candidate_evaluations.append(
                (budget, pred_em, net_utility, penalty, len(matching_indices))
            )

        # Select budget with highest estimated net utility
        best_budget, best_em, best_util, best_penalty, best_count = max(
            candidate_evaluations, key=lambda x: x[2]
        )

        return BudgetRecommendation(
            action=LearningAction(
                action_type=LearningActionType.TRAIN_CANDIDATE,
                step_budget=best_budget,
                learning_rate=0.003,
            ),
            predicted_dev_em=round(best_em, 4),
            predicted_utility=round(best_util, 4),
            confidence_penalty=round(best_penalty, 4),
            nearest_neighbors_count=best_count,
            decision_reasoning=(
                f"Selected budget {best_budget} steps with estimated EM {best_em:.2f}, "
                f"net utility {best_util:.4f} (penalty={best_penalty:.4f}) under {self.mode.value}"
            ),
        )
