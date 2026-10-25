"""Milestone M102 Lifelong Learning Coordinator.

Drives 100-episode continuous autonomous learning with:
1. Multi-family balanced rehearsal.
2. Dynamic Transformer layer skipping and compute routing.
3. Per-family worst-case retention tracking and disclosure.
4. Autonomous architecture adaptation opportunities.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from adaptive_mini_gpt.sustainable_lifelong_learning.lifelong_curriculum_m101 import (
    DomainM101,
    LearningGoalM101,
)

if TYPE_CHECKING:
    from adaptive_mini_gpt.dynamic_transformer_evolution.persistent_student_m102 import (
        PersistentStudentM102,
    )


@dataclass
class EpisodeTelemetryM102:
    """Record of an M102 lifelong learning episode."""

    episode_index: int
    selected_goal_id: str
    selected_goal_score: float
    held_out_accuracy: float
    mean_retention_accuracy: float
    worst_family_retention_accuracy: float
    skill_acquired: bool
    backbone_consolidated: bool
    fast_adaptation_steps: int
    memory_facts_count: int


@dataclass
class LifelongExecutionSummaryM102:
    """Summary of 100-episode lifelong stream."""

    total_episodes: int
    episodes_telemetry: list[EpisodeTelemetryM102] = field(default_factory=list)
    acquired_skills: list[str] = field(default_factory=list)
    average_held_out_accuracy: float = 0.0
    mean_family_retention: float = 0.0
    worst_family_retention: float = 0.0
    per_family_retention_breakdown: dict[str, float] = field(default_factory=dict)
    total_adaptation_steps: int = 0
    consolidation_promotions: int = 0


class LifelongCoordinatorM102:
    """Coordinates 100-episode continuous learning stream for M102."""

    def __init__(
        self,
        student: PersistentStudentM102,
        consolidation_interval: int = 5,
    ) -> None:
        self.student = student
        self.consolidation_interval = consolidation_interval
        self.episode_counter: int = 0

    def run_episode(self, candidate_goals: list[LearningGoalM101]) -> EpisodeTelemetryM102:
        self.episode_counter += 1

        # 1. Assess Current Capabilities
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

        # 3. Fast Adaptation with Balanced Rehearsal
        res_adapt = self.student.adapt_fast_skill(best_goal, max_steps=35, early_stop_acc=85.0)
        held_out_acc = res_adapt["final_test_acc"]
        mean_ret = res_adapt.get("mean_retention_acc", 100.0)
        worst_ret = res_adapt.get("worst_family_retention_acc", 100.0)
        skill_acquired = held_out_acc >= 75.0 and not res_adapt["abstained"]

        # 4. Slow Consolidation with Per-Family Protection
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

        return EpisodeTelemetryM102(
            episode_index=self.episode_counter,
            selected_goal_id=best_goal.goal_id,
            selected_goal_score=round(best_score, 4),
            held_out_accuracy=held_out_acc,
            mean_retention_accuracy=mean_ret,
            worst_family_retention_accuracy=worst_ret,
            skill_acquired=skill_acquired,
            backbone_consolidated=backbone_cons,
            fast_adaptation_steps=res_adapt["steps_taken"],
            memory_facts_count=len(self.student.memory_store.facts),
        )

    def run_100_episodes(self, candidate_goals: list[LearningGoalM101]) -> LifelongExecutionSummaryM102:
        telemetry: list[EpisodeTelemetryM102] = []
        for _ in range(100):
            ep_res = self.run_episode(candidate_goals)
            telemetry.append(ep_res)

        # Full per-family final audit
        per_fam = self.student.rehearsal_buffer.evaluate_all_families_retention(self.student)
        worst_fam = min(per_fam.values()) if per_fam else 0.0
        mean_fam = float(sum(per_fam.values()) / max(1, len(per_fam))) if per_fam else 0.0

        # Mean held-out accuracy across non-distractor goals
        domain_goals = [
            g for g in candidate_goals
            if g.domain not in (DomainM101.NOISE_DISTRACTOR, DomainM101.ALREADY_MASTERED)
        ]
        self.student.eval()
        dom_accs = [self.student.evaluate_goal_diagnostic(dg).current_accuracy for dg in domain_goals]
        mean_acc = float(sum(dom_accs) / max(1, len(dom_accs)))

        return LifelongExecutionSummaryM102(
            total_episodes=len(telemetry),
            episodes_telemetry=telemetry,
            acquired_skills=list(self.student.acquired_skills),
            average_held_out_accuracy=round(mean_acc, 2),
            mean_family_retention=round(mean_fam, 2),
            worst_family_retention=round(worst_fam, 2),
            per_family_retention_breakdown=per_fam,
            total_adaptation_steps=sum(t.fast_adaptation_steps for t in telemetry),
            consolidation_promotions=sum(1 for t in telemetry if t.backbone_consolidated),
        )
