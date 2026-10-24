"""Phase 6: Verified Experiment History Tracking & Topology-Conditioned Calibration.

Tracks verified outcomes of completed research experiments and conditions
future action priors on graph topology and empirical task features rather than
applying global scalar priors that cause over-spawning.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class CompletedExperimentRecord:
    """Historical record of an executed research experiment."""
    env_id: str
    graph_topology: str  # PARITY_CYCLE, CHROMATIN_LATTICE, FAULT_CASCADE, TRANSITIVE_DAG
    action_taken: str
    validation_f1_gain: float
    candidate_accepted: bool
    life_cycle_utility_gain: float


class VerifiedExperimentHistoryTracker:
    """Manages history records and evaluates topology-conditioned calibration."""

    def __init__(self) -> None:
        self._history: list[CompletedExperimentRecord] = []

    def record_completed_experiment(
        self,
        env_id: str,
        graph_topology: str,
        action_taken: str,
        validation_f1_gain: float,
        candidate_accepted: bool,
        life_cycle_utility_gain: float,
    ) -> None:
        """Append verified experiment record."""
        rec = CompletedExperimentRecord(
            env_id=env_id,
            graph_topology=graph_topology,
            action_taken=action_taken,
            validation_f1_gain=validation_f1_gain,
            candidate_accepted=candidate_accepted,
            life_cycle_utility_gain=life_cycle_utility_gain,
        )
        self._history.append(rec)

    def get_topology_conditioned_prior_bonus(
        self,
        target_topology: str,
        mode: str = "RELEVANT",  # RELEVANT, SHUFFLED, IRRELEVANT, CONSERVATIVE, NO_HISTORY
    ) -> float:
        """Compute prior utility bonus conditioned on historical experiments."""
        if mode == "NO_HISTORY" or not self._history:
            return 0.0

        if mode == "SHUFFLED":
            # Permute topology labels across records
            all_gains = [r.life_cycle_utility_gain for r in self._history]
            permuted = np.random.permutation(all_gains)
            return float(np.mean(permuted)) if len(permuted) > 0 else 0.0

        if mode == "IRRELEVANT":
            # Use negative or mismatched family history
            return -0.05

        if mode == "CONSERVATIVE":
            # Lower confidence bound (LCB = mean - 2 * std)
            relevant_records = [
                r for r in self._history if r.graph_topology == target_topology
            ]
            if not relevant_records:
                return 0.0
            gains = [r.life_cycle_utility_gain for r in relevant_records]
            lcb = float(np.mean(gains) - 1.96 * (np.std(gains) / np.sqrt(len(gains)) if len(gains) > 1 else 0.05))
            return max(0.0, lcb)

        # Default: RELEVANT (Topology-matched history)
        relevant_records = [
            r for r in self._history if r.graph_topology == target_topology
        ]
        if not relevant_records:
            # Different topology -> DO NOT over-generalize rail parity success to lattice/cascade!
            return 0.0

        gains = [r.life_cycle_utility_gain for r in relevant_records if r.candidate_accepted]
        return float(np.mean(gains)) if gains else 0.0
