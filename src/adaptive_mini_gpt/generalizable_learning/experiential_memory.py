"""Phase 7: Compact Experiential Memory for Generalizable Learning Decisions.

Stores compact verified records of previous learning episodes:
- Observable task topological complexity (nodes, diameter, density, complexity index)
- Prior skill candidate
- Initial validation performance on small verified sample
- Steps spent, final learning gain, and negative transfer / reset outcomes

Provides similarity-based experiential retrieval to guide budget decisions
and skill selection without peeking at ground truth rules or test labels.
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any

from adaptive_mini_gpt.generalizable_learning.task_suite import ObservableTaskComplexity

logger = logging.getLogger(__name__)


@dataclass
class LearningEpisodeRecord:
    """Verified historical record of an adaptation episode on a prior task."""
    episode_id: str
    task_id: str
    num_nodes: int
    graph_diameter: int
    graph_density: float
    structural_complexity_index: float
    prior_skill_id: str
    initial_sample_loss: float
    initial_sample_acc: float
    steps_spent: int
    final_loss: float
    final_f1: float
    learning_gain_f1: float
    was_negative_transfer: bool
    recommended_action: str  # "KEEP", "TRAIN", "RESET"
    recommended_budget_steps: int


@dataclass
class ExperientialRecommendation:
    """Actionable recommendation derived from historical learning experience."""
    recommended_action: str
    recommended_budget_steps: int
    nearest_episode_id: str
    similarity_score: float
    predicted_learning_gain: float
    negative_transfer_warning: bool
    rationale: str


class ExperientialMemoryStore:
    """In-memory store of past learning episodes enabling learning-to-learn."""

    def __init__(self) -> None:
        self.records: list[LearningEpisodeRecord] = []

    def add_record(self, record: LearningEpisodeRecord) -> None:
        """Add a verified episode record."""
        self.records.append(record)

    def populate_seed_experiences(self) -> None:
        """Populate initial baseline experiences from meta-training tasks."""
        # Experience 1: Deep hierarchical chain (benefited from 30-40 steps, positive transfer)
        self.add_record(LearningEpisodeRecord(
            episode_id="exp_meta_train_deep_chain",
            task_id="train_genomics_hierarchy",
            num_nodes=6,
            graph_diameter=5,
            graph_density=0.167,
            structural_complexity_index=6.73,
            prior_skill_id="source_hierarchy",
            initial_sample_loss=0.72,
            initial_sample_acc=0.65,
            steps_spent=40,
            final_loss=0.38,
            final_f1=0.72,
            learning_gain_f1=0.25,
            was_negative_transfer=False,
            recommended_action="TRAIN",
            recommended_budget_steps=40,
        ))

        # Experience 2: Symmetric cluster mesh (plateaued early around 20 steps)
        self.add_record(LearningEpisodeRecord(
            episode_id="exp_meta_train_cluster_mesh",
            task_id="train_cluster_mesh",
            num_nodes=6,
            graph_diameter=2,
            graph_density=0.40,
            structural_complexity_index=4.80,
            prior_skill_id="source_hierarchy",
            initial_sample_loss=0.62,
            initial_sample_acc=0.60,
            steps_spent=20,
            final_loss=0.42,
            final_f1=0.68,
            learning_gain_f1=0.18,
            was_negative_transfer=False,
            recommended_action="KEEP",
            recommended_budget_steps=20,
        ))

        # Experience 3: Incompatible parity coalition (severe negative transfer, required reset)
        self.add_record(LearningEpisodeRecord(
            episode_id="exp_meta_train_parity_clash",
            task_id="train_parity_coalition",
            num_nodes=6,
            graph_diameter=3,
            graph_density=0.20,
            structural_complexity_index=5.20,
            prior_skill_id="source_hierarchy",
            initial_sample_loss=1.28,
            initial_sample_acc=0.33,
            steps_spent=10,
            final_loss=1.18,
            final_f1=0.36,
            learning_gain_f1=-0.12,
            was_negative_transfer=True,
            recommended_action="RESET",
            recommended_budget_steps=20,
        ))

    def query_recommendation(
        self,
        complexity: ObservableTaskComplexity,
        sample_loss: float,
        sample_acc: float,
        prior_skill_id: str = "source_hierarchy",
    ) -> ExperientialRecommendation:
        """Derive actionable learning advice by retrieving most similar historical experience."""
        if not self.records:
            return ExperientialRecommendation(
                recommended_action="TRAIN",
                recommended_budget_steps=40,
                nearest_episode_id="none",
                similarity_score=0.0,
                predicted_learning_gain=0.10,
                negative_transfer_warning=False,
                rationale="No previous learning experience available; defaulting to standard exploration.",
            )

        best_score = -float("inf")
        best_rec: LearningEpisodeRecord | None = None

        # Distance metric combining topological features and initial validation behavior
        for rec in self.records:
            # Diameter difference (scaled by 10)
            d_diam = abs(complexity.graph_diameter - rec.graph_diameter) / 10.0
            # Node count difference (scaled by 10)
            d_nodes = abs(complexity.num_nodes - rec.num_nodes) / 10.0
            # Complexity index difference (scaled by 10)
            d_comp = abs(complexity.structural_complexity_index - rec.structural_complexity_index) / 10.0
            # Initial loss difference
            d_loss = abs(sample_loss - rec.initial_sample_loss) / 1.5

            # Total distance
            dist = 0.3 * d_diam + 0.2 * d_nodes + 0.2 * d_comp + 0.3 * d_loss
            similarity = 1.0 / (1.0 + dist)

            if similarity > best_score:
                best_score = similarity
                best_rec = rec

        assert best_rec is not None

        # Check negative transfer risk
        neg_risk = best_rec.was_negative_transfer or sample_loss >= 1.15
        if neg_risk:
            action = "RESET"
            budget = 20
            rationale = (
                f"Experiential match '{best_rec.episode_id}' (sim={best_score:.2f}) indicates severe negative transfer "
                f"or incompatible rule semantics (initial loss={sample_loss:.2f}). Recommending RESET to fresh adapter."
            )
        elif sample_acc >= 0.85 or sample_loss <= 0.40:
            action = "KEEP"
            budget = 10
            rationale = (
                f"Initial validation indicates high transfer proficiency (acc={sample_acc:.2f}, loss={sample_loss:.2f}). "
                f"Recommending KEEP to prevent overfitting and save compute."
            )
        else:
            action = "TRAIN"
            budget = best_rec.recommended_budget_steps
            rationale = (
                f"Experiential match '{best_rec.episode_id}' (sim={best_score:.2f}, diam={best_rec.graph_diameter}) "
                f"recommends {budget} steps for optimal convergence (predicted gain=+{best_rec.learning_gain_f1:.2f})."
            )

        return ExperientialRecommendation(
            recommended_action=action,
            recommended_budget_steps=budget,
            nearest_episode_id=best_rec.episode_id,
            similarity_score=best_score,
            predicted_learning_gain=best_rec.learning_gain_f1,
            negative_transfer_warning=neg_risk,
            rationale=rationale,
        )
