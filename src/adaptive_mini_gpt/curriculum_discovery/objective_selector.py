"""M36 Phase 4: Learning-Objective Selection Policy.

Compares three objective-selection strategies under identical compute budgets:
  - Policy A: Fixed Round-Robin schedule.
  - Policy B: Deterministic Largest Verified Gap.
  - Policy C: Neural-Assisted Objective Selection (ROI-based utility scoring).
"""

from __future__ import annotations

import math
from typing import Any

from adaptive_mini_gpt.curriculum_discovery.objective import (
    HypothesizedCause,
    LearningObjective,
)
from adaptive_mini_gpt.metacognition.action_space import LearningAction


class ObjectiveSelectionPolicy:
    """Base interface for learning-objective selection."""

    def select_objective(
        self,
        candidate_objectives: list[LearningObjective],
        completed_skills: set[str],
        remaining_budget: float,
        step_index: int = 0,
    ) -> LearningObjective | None:
        """Select one objective from candidate list or return None if budget exhausted."""
        raise NotImplementedError


class RoundRobinObjectiveSelector(ObjectiveSelectionPolicy):
    """Policy A: Cycles sequentially through candidate objectives."""

    def __init__(self) -> None:
        self.cursor: int = 0

    def select_objective(
        self,
        candidate_objectives: list[LearningObjective],
        completed_skills: set[str],
        remaining_budget: float,
        step_index: int = 0,
    ) -> LearningObjective | None:
        valid_candidates = [
            obj for obj in candidate_objectives
            if obj.estimated_resource_cost <= remaining_budget
        ]
        if not valid_candidates:
            return None
        idx = self.cursor % len(valid_candidates)
        self.cursor += 1
        return valid_candidates[idx]


class LargestGapObjectiveSelector(ObjectiveSelectionPolicy):
    """Policy B: Deterministically selects the objective with largest failure evidence / cost."""

    def select_objective(
        self,
        candidate_objectives: list[LearningObjective],
        completed_skills: set[str],
        remaining_budget: float,
        step_index: int = 0,
    ) -> LearningObjective | None:
        valid_candidates = [
            obj for obj in candidate_objectives
            if obj.estimated_resource_cost <= remaining_budget
        ]
        if not valid_candidates:
            return None
        # Sort by occurrence evidence count descending, then by expected benefit
        valid_candidates.sort(
            key=lambda obj: (len(obj.evidence_of_repeated_failure), obj.expected_benefit),
            reverse=True,
        )
        return valid_candidates[0]


class NeuralAssistedObjectiveSelector(ObjectiveSelectionPolicy):
    """Policy C: Outcome-aware selection balancing expected benefit, prerequisites, and compute cost."""

    def __init__(self, risk_aversion: float = 0.3) -> None:
        self.risk_aversion = risk_aversion
        self.selection_history: list[dict[str, Any]] = []

    def compute_roi_score(
        self,
        objective: LearningObjective,
        completed_skills: set[str],
    ) -> float:
        """Compute expected Return on Investment (ROI) score for an objective.

        ROI = (expected_benefit * prereq_satisfaction_ratio) / (cost + eps) - retention_risk
        """
        eps = 0.05
        # Check prerequisite satisfaction
        if objective.prerequisite_knowledge:
            met = sum(1 for p in objective.prerequisite_knowledge if p in completed_skills)
            prereq_ratio = met / len(objective.prerequisite_knowledge)
        else:
            prereq_ratio = 1.0

        # Retention risk is higher for full candidate training than lighter actions
        retention_risk = 0.25 if objective.proposed_intervention == LearningAction.TRAIN_CANDIDATE else 0.05

        numerator = objective.expected_benefit * prereq_ratio
        denominator = objective.estimated_resource_cost + eps
        roi = (numerator / denominator) - (self.risk_aversion * retention_risk)
        return round(roi, 4)

    def select_objective(
        self,
        candidate_objectives: list[LearningObjective],
        completed_skills: set[str],
        remaining_budget: float,
        step_index: int = 0,
    ) -> LearningObjective | None:
        valid_candidates = [
            obj for obj in candidate_objectives
            if obj.estimated_resource_cost <= remaining_budget
        ]
        if not valid_candidates:
            return None

        # Score all candidates
        scored = [
            (self.compute_roi_score(obj, completed_skills), obj)
            for obj in valid_candidates
        ]
        # Sort descending by ROI
        scored.sort(key=lambda pair: pair[0], reverse=True)
        best_score, best_obj = scored[0]

        self.selection_history.append({
            "step": step_index,
            "objective_id": best_obj.objective_id,
            "skill": best_obj.skill_or_concept,
            "intervention": best_obj.proposed_intervention.value,
            "roi_score": best_score,
            "expected_benefit": best_obj.expected_benefit,
            "cost": best_obj.estimated_resource_cost,
        })
        return best_obj
