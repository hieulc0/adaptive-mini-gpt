"""Milestone M102.2 Lifelong Coordinator.

Drives 100-episode continuous learning lifetimes per seed:
1. Complete census of all episodes (promotions, rollbacks, conflict detections).
2. Separate measurement of:
   - Validation Accuracy (used for candidate promotion / early stopping).
   - Never-Trained Held-Out Accuracy (unseen test inputs).
   - Novel Rule Parameterization Generalization (unseen rule parameters).
   - Independent Retention across all qualified families.
   - Autonomous Unlabeled Retention and Domain Routing Accuracy.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from adaptive_mini_gpt.scientific_requalification_m102_2.curriculum_m102_2 import (
    LearningGoalM102_2,
)
from adaptive_mini_gpt.scientific_requalification_m102_2.persistent_student_m102_2 import (
    PersistentStudentM102_2,
)


@dataclass
class EpisodeTelemetryM102_2:
    episode_index: int
    target_goal_id: str
    target_domain: str
    adaptation_steps: int
    validation_accuracy: float
    held_out_accuracy: float
    novel_generalization_accuracy: float
    conflicts_detected: int
    independent_mean_retention: float
    independent_worst_family_retention: float
    unlabeled_retention_accuracy: float
    domain_routing_accuracy: float
    promoted: bool
    rollback_triggered: bool


@dataclass
class LifelongExecutionSummaryM102_2:
    total_episodes: int
    promoted_episodes: int
    rollback_episodes: int
    consolidation_runs: int
    mean_held_out_accuracy: float
    mean_novel_generalization_accuracy: float
    mean_independent_retention: float
    worst_family_independent_retention: float
    worst_family_domain: str = ""
    mean_unlabeled_retention: float = 0.0
    mean_domain_routing_accuracy: float = 0.0
    per_family_retention: dict[str, float] = field(default_factory=dict)
    episode_census: list[EpisodeTelemetryM102_2] = field(default_factory=list)


class LifelongCoordinatorM102_2:
    """Coordinates multi-episode continuous learning without resets."""

    def __init__(
        self,
        student: PersistentStudentM102_2,
        consolidation_interval: int = 2,
    ) -> None:
        self.student = student
        self.consolidation_interval = consolidation_interval
        self.episode_counter = 0
        self.census: list[EpisodeTelemetryM102_2] = []

    def run_episode(self, available_goals: list[LearningGoalM102_2]) -> EpisodeTelemetryM102_2:
        """Execute a single learning episode and record scientific telemetry."""
        self.episode_counter += 1

        # Select next goal in round-robin fashion
        idx = (self.episode_counter - 1) % len(available_goals)
        goal = available_goals[idx]

        # Fast adaptation (stops on validation set, evaluates on held-out test set)
        res_adapt = self.student.adapt_fast_skill(goal, max_steps=30, early_stop_acc=85.0)

        # Slow consolidation every interval
        promoted = True
        rollback = False
        if self.episode_counter % self.consolidation_interval == 0:
            res_cons = self.student.consolidate_slow_backbone(goal, steps=3)
            promoted = res_cons["promoted"]
            rollback = res_cons["rollback"]

        # Audit independent retention
        ret_summary = self.student.retention_manager.evaluate_retention(self.student)

        telemetry = EpisodeTelemetryM102_2(
            episode_index=self.episode_counter,
            target_goal_id=goal.goal_id,
            target_domain=goal.domain.value,
            adaptation_steps=res_adapt["steps_taken"],
            validation_accuracy=res_adapt["validation_acc"],
            held_out_accuracy=res_adapt["held_out_test_acc"],
            novel_generalization_accuracy=res_adapt["novel_generalization_acc"],
            conflicts_detected=res_adapt["conflicts_detected"],
            independent_mean_retention=ret_summary.mean_independent_retention,
            independent_worst_family_retention=ret_summary.worst_family_independent_retention,
            unlabeled_retention_accuracy=ret_summary.mean_unlabeled_retention,
            domain_routing_accuracy=ret_summary.mean_domain_routing_accuracy,
            promoted=promoted,
            rollback_triggered=rollback,
        )
        self.census.append(telemetry)
        return telemetry

    def run_lifetime(
        self,
        goals: list[LearningGoalM102_2],
        num_episodes: int = 100,
    ) -> LifelongExecutionSummaryM102_2:
        """Run complete continuous lifetime."""
        for _ in range(num_episodes):
            self.run_episode(goals)

        final_ret = self.student.retention_manager.evaluate_retention(self.student)
        per_fam = {k: v.oracle_retention_acc for k, v in final_ret.family_audits.items()}

        mean_held_out = sum(e.held_out_accuracy for e in self.census) / len(self.census)
        mean_novel = sum(e.novel_generalization_accuracy for e in self.census) / len(self.census)
        promoted_count = sum(1 for e in self.census if e.promoted)
        rollback_count = sum(1 for e in self.census if e.rollback_triggered)

        return LifelongExecutionSummaryM102_2(
            total_episodes=num_episodes,
            promoted_episodes=promoted_count,
            rollback_episodes=rollback_count,
            consolidation_runs=num_episodes // self.consolidation_interval,
            mean_held_out_accuracy=round(mean_held_out, 2),
            mean_novel_generalization_accuracy=round(mean_novel, 2),
            mean_independent_retention=final_ret.mean_independent_retention,
            worst_family_independent_retention=final_ret.worst_family_independent_retention,
            worst_family_domain=final_ret.worst_family_domain,
            mean_unlabeled_retention=final_ret.mean_unlabeled_retention,
            mean_domain_routing_accuracy=final_ret.mean_domain_routing_accuracy,
            per_family_retention=per_fam,
            episode_census=self.census,
        )
