"""M36 Phase 3: Self-Directed Capability-Gap Discovery.

Analyzes bounded interaction logs and failure history to discover recurring
capability gaps, hypothesize root causes, and formulate testable LearningObjectives.
"""

from __future__ import annotations

import re
from collections import defaultdict
from typing import Any

from adaptive_mini_gpt.curriculum_discovery.objective import (
    HypothesizedCause,
    LearningObjective,
)
from adaptive_mini_gpt.interaction.knowledge_gap import (
    ErrorCategory,
    KnowledgeGapRecord,
    KnowledgeGapTracker,
)
from adaptive_mini_gpt.metacognition.action_space import LearningAction


class CapabilityGapAnalyzer:
    """Discovers capability gaps from interaction history and formulates learning objectives."""

    def __init__(self, history_window: int = 100) -> None:
        self.history_window = history_window

    def analyze_gaps(
        self,
        tracker: KnowledgeGapTracker,
    ) -> list[LearningObjective]:
        """Examine failure records and synthesize candidate LearningObjectives."""
        records = list(tracker.records.values())[-self.history_window:]
        if not records:
            return []

        # Cluster failure records by domain and failure modality
        clusters: dict[str, list[KnowledgeGapRecord]] = defaultdict(list)
        for r in records:
            cluster_key = f"{r.domain.value}::{r.error_category.value}"
            clusters[cluster_key].append(r)

        objectives: list[LearningObjective] = []

        for cluster_key, group in clusters.items():
            domain_str, error_cat_str = cluster_key.split("::")
            evidence_samples = [
                f"[{r.domain.value}] Q: {r.question[:60]}... -> Pred: '{r.predicted_answer}' (status={r.verification_status}, conf={r.confidence_score})"
                for r in group[:4]
            ]
            count = sum(r.occurrence_count for r in group)

            # Analyze failure cause & choose appropriate non-wasteful intervention
            if error_cat_str in (ErrorCategory.MISSING_INFORMATION.value, ErrorCategory.UNSUPPORTED.value):
                # Missing or contradictory evidence: Root cause is INSUFFICIENT_EVIDENCE
                # Intervention: ABSTAIN or VERIFY, never wasteful neural training
                cause = HypothesizedCause.INSUFFICIENT_EVIDENCE
                intervention = LearningAction.ABSTAIN
                prereqs: list[str] = []
                cost = 0.05
                expected_gain = 0.40
                metric = "Appropriate abstention on ungrounded/contradictory premises >= 90%"
                skill = f"Grounding & Abstention in {domain_str}"

            elif domain_str == "language" and error_cat_str == ErrorCategory.INCORRECT.value:
                # Factual QA gap: Can be resolved by retrieval rather than weight training
                cause = HypothesizedCause.FACTUAL_GAP
                intervention = LearningAction.RETRIEVE
                prereqs = ["passage_retrieval_indexing"]
                cost = 0.10
                expected_gain = 0.50
                metric = "Factual question answering with retrieved passage >= 80%"
                skill = "Passage-Grounded Factual Retrieval"

            elif error_cat_str == ErrorCategory.REPEATED_PROCEDURAL_FAILURE.value or count >= 2:
                # Repeated procedural failures on arithmetic carry or logic hops
                cause = HypothesizedCause.PROCEDURAL_GAP
                intervention = LearningAction.TRAIN_CANDIDATE
                if domain_str == "arithmetic":
                    prereqs = ["single_digit_addition", "column_alignment"]
                    skill = "Multi-Digit Recurrent Addition with Carry"
                    metric = "Exact match on 4-digit addition >= 85%"
                    cost = 1.00
                    expected_gain = 0.85
                elif domain_str == "logic":
                    prereqs = ["1_hop_modus_ponens"]
                    skill = "Multi-Hop Deductive Reasoning"
                    metric = "Exact match on 4-hop logic deduction >= 60%"
                    cost = 0.90
                    expected_gain = 0.60
                else:
                    prereqs = ["basic_syntax"]
                    skill = f"Procedural execution in {domain_str}"
                    metric = f"Accuracy on {domain_str} >= 70%"
                    cost = 0.80
                    expected_gain = 0.50

            else:
                # Sporadic single failure: Practice rather than full adapter training
                cause = HypothesizedCause.PROCEDURAL_GAP
                intervention = LearningAction.PRACTICE
                prereqs = ["task_instructions"]
                skill = f"Practice drills for {domain_str}"
                cost = 0.25
                expected_gain = 0.30
                metric = f"Fresh practice error rate on {domain_str} < 20%"

            obj = LearningObjective.create(
                skill_or_concept=skill,
                evidence_of_repeated_failure=evidence_samples,
                hypothesized_cause=cause,
                prerequisite_knowledge=prereqs,
                proposed_intervention=intervention,
                expected_benefit=expected_gain,
                estimated_resource_cost=cost,
                independent_success_measurement=metric,
                prerequisites_satisfied=True,
            )
            objectives.append(obj)

        return objectives
