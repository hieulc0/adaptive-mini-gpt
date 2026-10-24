"""Phase 5: Fair History Learning & Causal Experience Tracker.

Implements the 5 history modes for autonomous learning decisions:
Mode A: NO_HISTORY (Memory-free control: decisions depend strictly on current evidence)
Mode B: RELEVANT (Uses pre-decision observable graph features and past verified outcomes)
Mode C: SHUFFLED (Randomly permutes outcomes across past records to test causal association)
Mode D: IRRELEVANT (Synthetic unaligned history records)
Mode E: CONSERVATIVE (Uncertainty-discounted prior updates to prevent over-spawning)

Tests whether past verified experiments can improve decision utility on novel tasks
without optimistic over-spawning.
"""
from __future__ import annotations

import enum
import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


class HistoryConditioningMode(str, enum.Enum):
    """The 5 history conditioning modes in M69."""
    NO_HISTORY = "NO_HISTORY"
    RELEVANT = "RELEVANT"
    SHUFFLED = "SHUFFLED"
    IRRELEVANT = "IRRELEVANT"
    CONSERVATIVE = "CONSERVATIVE"


@dataclass
class VerifiedExperimentRecord:
    """A verified, immutable record of a completed learning experiment."""
    record_id: str
    task_name: str
    rule_family: str
    pre_decision_density: float
    pre_decision_probe_f1: float
    sprt_llr: float
    action_taken: str
    measured_macro_f1: float
    measured_utility: float
    validation_passed: bool


@dataclass
class HistoryAblationComparison:
    """Outcome of testing a single history conditioning mode."""
    mode: HistoryConditioningMode
    mean_macro_f1: float
    mean_amortized_utility: float
    oracle_queries: float
    optimizer_steps: float
    false_allocations: int
    candidate_rejections: int
    over_spawning_detected: bool
    advantage_over_no_history: float


class FairHistoryLearningTracker:
    """Manages verified experiment history and computes bounded pre-decision prior adjustments."""

    def __init__(self) -> None:
        self.history: list[VerifiedExperimentRecord] = []

    def log_verified_experiment(self, record: VerifiedExperimentRecord) -> None:
        """Append a newly verified experiment outcome to the research ledger."""
        self.history.append(record)

    def compute_prior_utility_bonus(
        self,
        mode: HistoryConditioningMode,
        candidate_action: str,
        current_density: float,
        current_probe_f1: float,
        current_sprt_llr: float,
    ) -> float:
        """Compute observable-evidence-based prior adjustment under the given mode."""
        if mode == HistoryConditioningMode.NO_HISTORY or not self.history:
            return 0.0

        if mode == HistoryConditioningMode.IRRELEVANT:
            # Synthetic unrelated task bias
            return 0.01 if candidate_action == "REUSE_EXISTING" else -0.01

        history_pool = list(self.history)
        if mode == HistoryConditioningMode.SHUFFLED:
            # Permute actions and utilities randomly
            rng = np.random.default_rng(seed=123)
            utilities = [r.measured_utility for r in history_pool]
            rng.shuffle(utilities)
            for i, r in enumerate(history_pool):
                r = VerifiedExperimentRecord(
                    record_id=r.record_id,
                    task_name=r.task_name,
                    rule_family=r.rule_family,
                    pre_decision_density=r.pre_decision_density,
                    pre_decision_probe_f1=r.pre_decision_probe_f1,
                    sprt_llr=r.sprt_llr,
                    action_taken=r.action_taken,
                    measured_macro_f1=r.measured_macro_f1,
                    measured_utility=utilities[i],
                    validation_passed=r.validation_passed,
                )

        # Compute observable similarity: distance in (density, probe_f1, sprt_llr)
        similarities = []
        for rec in history_pool:
            if rec.action_taken == candidate_action:
                d = (
                    abs(rec.pre_decision_density - current_density)
                    + abs(rec.pre_decision_probe_f1 - current_probe_f1)
                    + 0.1 * abs(rec.sprt_llr - current_sprt_llr)
                )
                sim = 1.0 / (1.0 + d)
                similarities.append((sim, rec.measured_utility, rec.validation_passed))

        if not similarities:
            return 0.0

        # Weighted utility delta
        total_w = sum(s[0] for s in similarities)
        w_util = sum(s[0] * s[1] for s in similarities) / total_w

        if mode == HistoryConditioningMode.CONSERVATIVE:
            # Discount positive prior by 50% and penalize candidate rejections
            return float(max(0.0, w_util * 0.5) if w_util > 0 else w_util)

        return float(w_util * 0.1)  # Bounded scaling
