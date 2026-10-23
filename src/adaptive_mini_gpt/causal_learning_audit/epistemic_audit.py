"""Phase 6: Systematic Diagnostic Audit of Epistemic Dirichlet-Multinomial Belief Tracker."""
from __future__ import annotations

import copy
import logging
import random
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from adaptive_mini_gpt.guided_experiment.environment import (
    AutonomousEnvironment,
    EXPANDED_HYPOTHESES,
    OUT_OF_HYPOTHESIS_PARITY,
    UNKNOWN_RULE,
)
from adaptive_mini_gpt.learning_driven_experiment.epistemic_tracker import NormalizedEpistemicTracker

logger = logging.getLogger(__name__)


@dataclass
class EpistemicRegimeResult:
    """Outcome of tracker diagnostic on a specific environment regime."""
    regime_name: str  # "known", "noisy_known", "out_of_hypothesis"
    num_episodes: int
    mean_final_entropy: float
    mean_unknown_rule_prob: float
    detection_rate_at_threshold: float  # threshold=0.50
    false_positive_rate: float
    posterior_calibration_error: float
    trajectory_unknown_probs: list[float]  # averaged over steps


@dataclass
class EpistemicAuditReport:
    """Comprehensive forensic report on epistemic tracker behavior across regimes."""
    known_regime: EpistemicRegimeResult
    noisy_regime: EpistemicRegimeResult
    out_of_hypothesis_regime: EpistemicRegimeResult
    threshold_used: float
    query_budget: int
    is_epistemic_qualified: bool
    diagnostic_rationale: str


class EpistemicAuditEngine:
    """Audits Dirichlet-multinomial belief tracker calibration and discrimination."""

    CLASS_LABELS = ["YES", "NO", "UNKNOWN"]

    @classmethod
    def evaluate_episode(
        cls,
        env: AutonomousEnvironment,
        query_budget: int = 12,
        noise_rate: float = 0.0,
        seed: int = 42,
    ) -> tuple[float, float, list[float]]:
        """Run a single episode on an environment and track UNKNOWN_RULE posterior."""
        random.seed(seed)
        tracker = NormalizedEpistemicTracker(base_edges=env.base_edges)
        entities = env.entities
        all_pairs = [(u, v) for u in entities for v in entities if u != v]
        random.shuffle(all_pairs)

        probs_history: list[float] = [float(tracker.beliefs.get(UNKNOWN_RULE.hypothesis_id, 0.0))]

        for u, v in all_pairs[:query_budget]:
            true_ans = env.evaluate_ground_truth(u, v)
            observed_ans = true_ans
            if noise_rate > 0.0 and random.random() < noise_rate:
                # Corrupt observation to a different label
                other_labels = [l for l in cls.CLASS_LABELS if l != true_ans]
                observed_ans = random.choice(other_labels)

            tracker.update(u, v, observed_ans)
            probs_history.append(float(tracker.beliefs.get(UNKNOWN_RULE.hypothesis_id, 0.0)))

        final_ent = tracker.compute_entropy()
        final_unk_p = float(tracker.beliefs.get(UNKNOWN_RULE.hypothesis_id, 0.0))
        return final_ent, final_unk_p, probs_history

    @classmethod
    def run_regime_audit(
        cls,
        environments: list[AutonomousEnvironment],
        regime_name: str,
        query_budget: int = 12,
        noise_rate: float = 0.0,
        threshold: float = 0.50,
        num_seeds: int = 5,
    ) -> EpistemicRegimeResult:
        """Run diagnostic across multiple environments and seeds for a regime."""
        final_ents: list[float] = []
        final_unk_probs: list[float] = []
        trajectories: list[list[float]] = []
        detections = 0
        total_episodes = 0

        for env in environments:
            for s in range(num_seeds):
                ent, unk_p, traj = cls.evaluate_episode(
                    env, query_budget=query_budget, noise_rate=noise_rate, seed=100 + s
                )
                final_ents.append(ent)
                final_unk_probs.append(unk_p)
                trajectories.append(traj)
                if unk_p >= threshold:
                    detections += 1
                total_episodes += 1

        # Mean trajectory
        max_len = max(len(t) for t in trajectories)
        mean_traj = [
            float(np.mean([t[i] for t in trajectories if i < len(t)]))
            for i in range(max_len)
        ]

        det_rate = detections / max(1, total_episodes)
        is_ooh = (regime_name == "out_of_hypothesis")
        fp_rate = 0.0 if is_ooh else det_rate
        cal_error = float(np.mean([abs(p - (1.0 if is_ooh else 0.0)) for p in final_unk_probs]))

        return EpistemicRegimeResult(
            regime_name=regime_name,
            num_episodes=total_episodes,
            mean_final_entropy=float(np.mean(final_ents)),
            mean_unknown_rule_prob=float(np.mean(final_unk_probs)),
            detection_rate_at_threshold=det_rate,
            false_positive_rate=fp_rate,
            posterior_calibration_error=cal_error,
            trajectory_unknown_probs=mean_traj,
        )

    @classmethod
    def audit_epistemic_tracker(
        cls,
        known_envs: list[AutonomousEnvironment],
        ooh_envs: list[AutonomousEnvironment],
        query_budget: int = 12,
        threshold: float = 0.50,
    ) -> EpistemicAuditReport:
        """Run complete 3-regime audit of Dirichlet-multinomial belief tracker."""
        res_known = cls.run_regime_audit(
            known_envs, "known", query_budget=query_budget, noise_rate=0.0, threshold=threshold
        )
        res_noisy = cls.run_regime_audit(
            known_envs, "noisy_known", query_budget=query_budget, noise_rate=0.15, threshold=threshold
        )
        res_ooh = cls.run_regime_audit(
            ooh_envs, "out_of_hypothesis", query_budget=query_budget, noise_rate=0.0, threshold=threshold
        )

        # Qualification requires high recall on OOH and low false alarms on known/noisy
        qualified = (res_ooh.detection_rate_at_threshold >= 0.50) and (res_known.false_positive_rate <= 0.10)

        rationale = (
            f"EPISTEMIC AUDIT RESULTS: Known regime false positive rate={res_known.false_positive_rate:.2%} "
            f"(mean unk prob={res_known.mean_unknown_rule_prob:.4f}). "
            f"Noisy known regime false positive rate={res_noisy.false_positive_rate:.2%} "
            f"(mean unk prob={res_noisy.mean_unknown_rule_prob:.4f}). "
            f"Out-of-hypothesis detection rate={res_ooh.detection_rate_at_threshold:.2%} "
            f"(mean unk prob={res_ooh.mean_unknown_rule_prob:.4f}) over budget Q={query_budget}."
        )

        return EpistemicAuditReport(
            known_regime=res_known,
            noisy_regime=res_noisy,
            out_of_hypothesis_regime=res_ooh,
            threshold_used=threshold,
            query_budget=query_budget,
            is_epistemic_qualified=qualified,
            diagnostic_rationale=rationale,
        )
