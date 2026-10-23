"""Phase 6: Experience-Memory Dynamic Update & Confidence Calibration for M61.

Tests whether adding new verified learning experience improves decisions on later tasks:
- Measures adaptation on initial tasks, stores verified compact episode records.
- Evaluates later tasks with and without the accumulated experiences.
- Tests conflict resolution: if historical advice conflicts with real-time validation evidence,
  the safe controller prefers current observations or continues bounded training.
- Separately quantifies negative transfer prevention and compute savings.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from adaptive_mini_gpt.generalizable_learning.experiential_memory import (
    ExperientialMemoryStore,
    LearningEpisodeRecord,
)

logger = logging.getLogger(__name__)


@dataclass
class ExperienceUpdateReport:
    """Quantitative evaluation of whether accumulating experience improves future learning decisions."""
    initial_episodes_count: int
    updated_episodes_count: int
    episodes_added: int
    initial_controller_mean_auc: float
    updated_controller_mean_auc: float
    auc_delta: float
    initial_controller_mean_steps: float
    updated_controller_mean_steps: float
    steps_saved_pct: float
    conflict_resolution_accuracy: float
    negative_transfer_events_prevented: int
    experience_accumulation_effective: bool


class DynamicExperienceMemoryUpdater:
    """Manages verified learning episode accumulation and evaluates continuous learning gains."""

    def __init__(self, memory_store: ExperientialMemoryStore) -> None:
        self.memory_store = memory_store

    def record_completed_episode(
        self,
        task_id: str,
        num_nodes: int,
        graph_diameter: int,
        graph_density: float,
        structural_complexity_index: float,
        steps_spent: int,
        initial_loss: float,
        final_loss: float,
        final_f1: float,
        learning_gain_f1: float,
        was_negative_transfer: bool,
        recommended_action: str = "TRAIN",
    ) -> LearningEpisodeRecord:
        """Create and commit a verified episode record to memory."""
        ep_id = f"ep_m61_{task_id}_{len(self.memory_store.records)}"
        rec = LearningEpisodeRecord(
            episode_id=ep_id,
            task_id=task_id,
            num_nodes=num_nodes,
            graph_diameter=graph_diameter,
            graph_density=graph_density,
            structural_complexity_index=structural_complexity_index,
            prior_skill_id="source_hierarchy",
            initial_sample_loss=initial_loss,
            initial_sample_acc=0.50,
            steps_spent=steps_spent,
            final_loss=final_loss,
            final_f1=final_f1,
            learning_gain_f1=learning_gain_f1,
            was_negative_transfer=was_negative_transfer,
            recommended_action=recommended_action,
            recommended_budget_steps=steps_spent,
        )
        self.memory_store.add_record(rec)
        return rec
