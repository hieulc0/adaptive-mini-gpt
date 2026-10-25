"""Milestone M102.1 Lifelong Coordinator with Detailed Episode Census.

Runs 100 sequential continuous learning episodes without resets:
1. Evaluates goal diagnostic states.
2. Selects goals via CurriculumDiscoveryPolicy.
3. Fast adaptation with head isolation, PCGrad conflict protection, and rollback guard.
4. Slow consolidation with top-layer backbone unfreezing and retention validation.
5. Tracks complete episode census: attempts, promotions, rollbacks, and per-family retention.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from adaptive_mini_gpt.continual_interference_repair.persistent_student_m102_1 import (
    PersistentStudentM102_1,
)
from adaptive_mini_gpt.continual_interference_repair.rehearsal_buffer_m102_1 import (
    FamilyRetentionReportM102_1,
)
from adaptive_mini_gpt.sustainable_lifelong_learning.lifelong_curriculum_m101 import (
    DomainM101,
    LearningGoalM101,
)


@dataclass
class EpisodeTelemetryM102_1:
    episode_index: int
    selected_goal_id: str
    selected_goal_score: float
    held_out_accuracy: float
    mean_retention_accuracy: float
    worst_family_retention_accuracy: float
    worst_family_domain: str
    skill_acquired: bool
    promoted: bool
    rollback_triggered: bool
    backbone_consolidated: bool
    fast_adaptation_steps: int
    gradient_conflicts_count: int
    memory_facts_count: int


@dataclass
class LifelongExecutionSummaryM102_1:
    total_episodes: int
    episodes_telemetry: list[EpisodeTelemetryM102_1]
    acquired_skills: list[str]
    average_held_out_accuracy: float
    mean_family_retention: float
    worst_family_retention: float
    per_family_retention_breakdown: dict[str, float]
    per_family_reports: dict[str, FamilyRetentionReportM102_1]
    total_adaptation_steps: int
    promoted_episodes_count: int
    rollback_episodes_count: int
    consolidation_promotions: int


class LifelongCoordinatorM102_1:
    """Coordinates 100-episode continuous learning stream for Milestone M102.1."""

    def __init__(
        self,
        student: PersistentStudentM102_1,
        consolidation_interval: int = 5,
    ) -> None:
        self.student = student
        self.consolidation_interval = consolidation_interval
        self.episode_counter: int = 0

    def run_episode(self, candidate_goals: list[LearningGoalM101]) -> EpisodeTelemetryM102_1:
        self.episode_counter += 1

        # 1. Assess Diagnostics
        diagnostics = {g.goal_id: self.student.evaluate_goal_diagnostic(g) for g in candidate_goals}

        # 2. Select Goal via Curriculum Policy
        best_goal: LearningGoalM101 | None = None
        best_score = -1e9
        for g in candidate_goals:
            st = diagnostics[g.goal_id]
            score = self.student.curriculum_policy.score_goal(st, device=self.student.device)
            if g.goal_id in self.student.acquired_skills:
                score -= 100.0
            if g.domain == DomainM101.ALREADY_MASTERED or st.current_accuracy >= 95.0:
                score -= 150.0
            if not g.is_learnable and len(self.student.diagnostic_history.get(g.goal_id, [])) > 0:
                score -= 200.0

            if score > best_score:
                best_score = score
                best_goal = g

        assert best_goal is not None

        # 3. Fast Adaptation with Interference Control
        res_adapt = self.student.adapt_fast_skill(best_goal, max_steps=35, early_stop_acc=85.0)
        held_out_acc = res_adapt["final_test_acc"]
        mean_ret = res_adapt.get("mean_retention_acc", 100.0)
        worst_ret = res_adapt.get("worst_family_retention_acc", 100.0)
        promoted = res_adapt.get("promoted", True)
        skill_acquired = held_out_acc >= 75.0 and promoted and not res_adapt["abstained"]

        # 4. Slow Consolidation with Retention Guard
        attempt_cons = (self.episode_counter % self.consolidation_interval == 0) and skill_acquired
        backbone_cons = False
        if attempt_cons:
            rep = self.student.consolidate_slow_backbone(best_goal, steps=10)
            backbone_cons = rep.promoted

        # 5. Store Factual Trace
        self.student.memory_store.store_fact(
            entity=best_goal.domain.value,
            attribute="competence_acc",
            value=int(held_out_acc),
            source=f"ep_{self.episode_counter}",
            timestamp=self.episode_counter,
        )

        audit_summary = self.student.rehearsal_buffer.evaluate_all_families_retention(self.student)

        return EpisodeTelemetryM102_1(
            episode_index=self.episode_counter,
            selected_goal_id=best_goal.goal_id,
            selected_goal_score=round(best_score, 4),
            held_out_accuracy=held_out_acc,
            mean_retention_accuracy=mean_ret,
            worst_family_retention_accuracy=worst_ret,
            worst_family_domain=audit_summary.worst_family_domain,
            skill_acquired=skill_acquired,
            promoted=promoted,
            rollback_triggered=not promoted,
            backbone_consolidated=backbone_cons,
            fast_adaptation_steps=res_adapt["steps_taken"],
            gradient_conflicts_count=res_adapt.get("gradient_conflicts_count", 0),
            memory_facts_count=len(self.student.memory_store.facts),
        )

    def run_100_episodes(self, candidate_goals: list[LearningGoalM101]) -> LifelongExecutionSummaryM102_1:
        telemetry: list[EpisodeTelemetryM102_1] = []
        for _ in range(100):
            ep_res = self.run_episode(candidate_goals)
            telemetry.append(ep_res)

        # Final Rehearsal Audit
        audit_summary = self.student.rehearsal_buffer.evaluate_all_families_retention(self.student)
        per_fam_scores = {d: rep.retention_accuracy for d, rep in audit_summary.family_reports.items()}

        # Mean held-out accuracy on all valid domain tasks
        domain_goals = [
            g for g in candidate_goals
            if g.domain not in (DomainM101.NOISE_DISTRACTOR, DomainM101.ALREADY_MASTERED)
        ]
        self.student.eval()
        dom_accs = [self.student.evaluate_goal_diagnostic(dg).current_accuracy for dg in domain_goals]
        mean_acc = float(sum(dom_accs) / max(1, len(dom_accs)))

        return LifelongExecutionSummaryM102_1(
            total_episodes=len(telemetry),
            episodes_telemetry=telemetry,
            acquired_skills=list(self.student.acquired_skills),
            average_held_out_accuracy=round(mean_acc, 2),
            mean_family_retention=round(audit_summary.mean_retention_accuracy, 2),
            worst_family_retention=round(audit_summary.worst_family_retention_accuracy, 2),
            per_family_retention_breakdown=per_fam_scores,
            per_family_reports=audit_summary.family_reports,
            total_adaptation_steps=sum(t.fast_adaptation_steps for t in telemetry),
            promoted_episodes_count=sum(1 for t in telemetry if t.promoted),
            rollback_episodes_count=sum(1 for t in telemetry if t.rollback_triggered),
            consolidation_promotions=sum(1 for t in telemetry if t.backbone_consolidated),
        )
