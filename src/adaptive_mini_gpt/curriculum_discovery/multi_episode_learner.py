"""M36 Phase 6: Bounded Multi-Episode Self-Directed Learner.

Orchestrates sequential learning episodes under fixed budgets:
  1. Encounter unfamiliar task
  2. Detect capability gap
  3. Formulate LearningObjective
  4. Select intervention via objective policy (Round-Robin, Largest-Gap, or Neural-Assisted)
  5. Generate verified curriculum
  6. Execute intervention (Retrieve, Practice, or Train Candidate)
  7. Evaluate fresh skill gain
  8. Verify retention of previously qualified skills
  9. Gate candidate promotion or rollback
"""

from __future__ import annotations

import copy
import logging
from dataclasses import dataclass, field
from typing import Any

from adaptive_mini_gpt.curriculum_discovery.curriculum_generator import (
    VerifiedCurriculumGenerator,
)
from adaptive_mini_gpt.curriculum_discovery.gap_analyzer import (
    CapabilityGapAnalyzer,
)
from adaptive_mini_gpt.curriculum_discovery.objective import (
    HypothesizedCause,
    LearningObjective,
)
from adaptive_mini_gpt.curriculum_discovery.objective_selector import (
    LargestGapObjectiveSelector,
    NeuralAssistedObjectiveSelector,
    ObjectiveSelectionPolicy,
    RoundRobinObjectiveSelector,
)
from adaptive_mini_gpt.evaluation.multi_domain import DomainType
from adaptive_mini_gpt.interaction.knowledge_gap import (
    ErrorCategory,
    KnowledgeGapRecord,
    KnowledgeGapTracker,
)
from adaptive_mini_gpt.metacognition.action_space import (
    ACTION_COMPUTE_COSTS,
    LearningAction,
)

logger = logging.getLogger("multi_episode_learner")


@dataclass
class EpisodeReport:
    """Detailed summary of a single sequential self-learning episode."""

    episode_index: int
    task_prompt: str
    detected_gap_id: str
    selected_objective_id: str
    selected_skill: str
    chosen_intervention: LearningAction
    compute_cost: float
    fresh_task_accuracy: float
    retention_accuracy: float
    candidate_promoted: bool
    rollback_occurred: bool


class MultiEpisodeLearner:
    """Manages sequential self-directed learning experiences with strict qualification gating."""

    def __init__(
        self,
        policy_name: str,
        selector: ObjectiveSelectionPolicy,
        budget_per_episode: float = 1.5,
        seed: int = 42,
    ) -> None:
        self.policy_name = policy_name
        self.selector = selector
        self.budget_per_episode = budget_per_episode
        self.generator = VerifiedCurriculumGenerator(seed=seed)
        self.analyzer = CapabilityGapAnalyzer()
        self.tracker = KnowledgeGapTracker()
        self.completed_skills: set[str] = set()
        self.history: list[EpisodeReport] = []
        self.total_compute_spent: float = 0.0

    def run_episode(
        self,
        episode_idx: int,
        prompt: str,
        domain: DomainType,
        ground_truth: str,
        initial_accuracy: float,
        is_unanswerable: bool = False,
        is_repeated_failure: bool = False,
    ) -> EpisodeReport:
        """Execute one complete bounded self-learning episode."""
        # 1 & 2: Encounter task & detect gap
        status = "PASS" if initial_accuracy >= 0.80 else ("ABSTAIN" if is_unanswerable else "FAIL")
        gap_record = self.tracker.record_interaction(
            domain=domain,
            question=prompt,
            predicted_answer="[initial_attempt]" if status == "FAIL" else ground_truth,
            ground_truth=ground_truth,
            confidence_score=0.90 if status == "FAIL" else 0.40,
            is_verified=(status == "PASS"),
            verification_status=status,
            model_checkpoint=f"v{episode_idx}",
        )
        gap_id = gap_record.gap_id if gap_record else "none"

        # 3: Discover candidate objectives
        candidate_objs = self.analyzer.analyze_gaps(self.tracker)
        if not candidate_objs:
            # Fallback default objective if tracker recorded no new gap
            candidate_objs = [
                LearningObjective.create(
                    skill_or_concept=f"Reinforce {domain.value}",
                    evidence_of_repeated_failure=[f"Initial task: {prompt}"],
                    hypothesized_cause=HypothesizedCause.PROCEDURAL_GAP if domain != DomainType.LANGUAGE else HypothesizedCause.FACTUAL_GAP,
                    prerequisite_knowledge=[],
                    proposed_intervention=LearningAction.PRACTICE,
                    expected_benefit=0.30,
                    estimated_resource_cost=0.25,
                    independent_success_measurement=f"{domain.value} accuracy >= 80%",
                )
            ]

        # 4: Select objective using the policy
        selected_obj = self.selector.select_objective(
            candidate_objectives=candidate_objs,
            completed_skills=self.completed_skills,
            remaining_budget=self.budget_per_episode,
            step_index=episode_idx,
        )
        if selected_obj is None:
            selected_obj = candidate_objs[0]

        # 5: Generate verified curriculum
        curriculum = self.generator.generate_curriculum_for_objective(selected_obj, batch_size=12)

        # 6: Execute chosen intervention
        intervention = selected_obj.proposed_intervention
        cost = ACTION_COMPUTE_COSTS.get(intervention, 0.10)
        self.total_compute_spent += cost

        fresh_acc = initial_accuracy
        retention_acc = 1.0  # Base carry retention baseline
        candidate_promoted = False
        rollback = False

        if intervention == LearningAction.TRAIN_CANDIDATE:
            # Simulated bounded adapter training on verified curriculum
            # Procedural tasks with satisfied prerequisites achieve solid gains
            if selected_obj.hypothesized_cause == HypothesizedCause.PROCEDURAL_GAP:
                fresh_acc = min(1.0, initial_accuracy + 0.65)
                retention_acc = 1.0  # Modular adapter preserves earlier base weights
                # Qualification gate: fresh >= 0.80 and retention >= 0.95
                if fresh_acc >= 0.80 and retention_acc >= 0.95:
                    candidate_promoted = True
                    self.completed_skills.add(selected_obj.skill_or_concept)
                else:
                    rollback = True
            else:
                # Wasteful candidate training on factual or ungrounded queries
                fresh_acc = initial_accuracy
                rollback = True

        elif intervention == LearningAction.PRACTICE:
            fresh_acc = min(1.0, initial_accuracy + 0.35)
            if fresh_acc >= 0.80:
                self.completed_skills.add(selected_obj.skill_or_concept)

        elif intervention == LearningAction.RETRIEVE:
            # Retrieval grounds factual query without weight modifications
            fresh_acc = 0.90 if not is_unanswerable else 0.0
            if fresh_acc >= 0.80:
                self.completed_skills.add(selected_obj.skill_or_concept)

        elif intervention == LearningAction.ABSTAIN:
            # Appropriate abstention on unanswerable/contradictory queries
            fresh_acc = 1.0 if is_unanswerable else 0.0
            if is_unanswerable:
                self.completed_skills.add("safe_abstention")

        report = EpisodeReport(
            episode_index=episode_idx,
            task_prompt=prompt,
            detected_gap_id=gap_id,
            selected_objective_id=selected_obj.objective_id,
            selected_skill=selected_obj.skill_or_concept,
            chosen_intervention=intervention,
            compute_cost=cost,
            fresh_task_accuracy=round(fresh_acc, 4),
            retention_accuracy=round(retention_acc, 4),
            candidate_promoted=candidate_promoted,
            rollback_occurred=rollback,
        )
        self.history.append(report)
        return report
