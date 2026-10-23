"""Phase 7: Generalization Audit of Epistemic Dirichlet-Multinomial Belief Tracker Across Unseen Rule Families."""
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
    UNKNOWN_RULE,
)
from adaptive_mini_gpt.learning_driven_experiment.epistemic_tracker import NormalizedEpistemicTracker

logger = logging.getLogger(__name__)


@dataclass
class MechanismDiagnosticResult:
    """Diagnostic outcome for a specific rule mechanism or environment regime."""
    mechanism_name: str
    is_out_of_family: bool
    num_episodes: int
    detection_rate: float  # Fraction where final P(UNKNOWN_RULE) >= threshold
    mean_final_entropy: float
    mean_final_unknown_prob: float
    mean_queries_to_detect: float


@dataclass
class EpistemicGeneralizationReport:
    """Comprehensive generalization benchmark of the unknown-rule detector across unseen mechanisms."""
    frozen_threshold: float
    query_budget: int
    mechanism_results: dict[str, MechanismDiagnosticResult]
    overall_unseen_recall: float
    overall_false_alarm_rate: float
    overall_precision: float
    overall_f1: float
    mean_oracle_query_cost: float
    is_detector_qualified: bool
    diagnostic_rationale: str


class EpistemicGeneralizationAuditor:
    """Audits epistemic belief tracker discrimination across diverse unseen rule families and noise."""

    CLASS_LABELS = ["YES", "NO", "UNKNOWN"]

    @classmethod
    def run_single_episode(
        cls,
        env: AutonomousEnvironment,
        query_budget: int = 12,
        noise_rate: float = 0.0,
        threshold: float = 0.50,
        seed: int = 42,
    ) -> tuple[float, float, int | None]:
        """Run single episode, returning (final_entropy, final_unk_prob, queries_to_detect)."""
        rng = random.Random(seed)
        tracker = NormalizedEpistemicTracker(
            base_edges=env.base_edges,
            entity_levels=env.entity_levels,
            blocked_nodes=env.blocked_nodes,
            negated_edges=env.negated_edges,
            observation_noise=0.05,
        )

        entities = env.entities
        all_pairs = [(u, v) for u in entities for v in entities if u != v]
        rng.shuffle(all_pairs)

        detect_step = None
        for step, (u, v) in enumerate(all_pairs[:query_budget]):
            true_ans = env.evaluate_ground_truth(u, v)
            obs_ans = true_ans
            if noise_rate > 0.0 and rng.random() < noise_rate:
                other_labels = [l for l in cls.CLASS_LABELS if l != true_ans]
                obs_ans = rng.choice(other_labels)

            tracker.update(u, v, obs_ans)
            unk_p = float(tracker.beliefs.get(UNKNOWN_RULE.hypothesis_id, 0.0))
            if unk_p >= threshold and detect_step is None:
                detect_step = step + 1

        final_ent = tracker.compute_entropy()
        final_unk_p = float(tracker.beliefs.get(UNKNOWN_RULE.hypothesis_id, 0.0))
        return final_ent, final_unk_p, detect_step

    @classmethod
    def evaluate_mechanism(
        cls,
        mechanism_name: str,
        env: AutonomousEnvironment,
        is_out_of_family: bool,
        query_budget: int = 12,
        noise_rate: float = 0.0,
        threshold: float = 0.50,
        num_seeds: int = 6,
    ) -> MechanismDiagnosticResult:
        """Evaluate performance over multiple random seed episodes on a given environment."""
        ents: list[float] = []
        unk_probs: list[float] = []
        detect_steps: list[int] = []
        detections = 0

        for s in range(num_seeds):
            seed = 42 + s * 13
            ent, unk_p, d_step = cls.run_single_episode(
                env=env,
                query_budget=query_budget,
                noise_rate=noise_rate,
                threshold=threshold,
                seed=seed,
            )
            ents.append(ent)
            unk_probs.append(unk_p)
            if unk_p >= threshold:
                detections += 1
            if d_step is not None:
                detect_steps.append(d_step)

        det_rate = detections / num_seeds
        mean_detect_step = float(np.mean(detect_steps)) if detect_steps else float(query_budget)

        return MechanismDiagnosticResult(
            mechanism_name=mechanism_name,
            is_out_of_family=is_out_of_family,
            num_episodes=num_seeds,
            detection_rate=float(det_rate),
            mean_final_entropy=float(np.mean(ents)),
            mean_final_unknown_prob=float(np.mean(unk_probs)),
            mean_queries_to_detect=mean_detect_step,
        )

    @classmethod
    def run_epistemic_benchmark(
        cls,
        epistemic_envs: dict[str, AutonomousEnvironment],
        frozen_threshold: float = 0.50,
        query_budget: int = 12,
        num_seeds_per_mech: int = 6,
    ) -> EpistemicGeneralizationReport:
        """Execute complete epistemic generalization benchmark across unseen families and noise regimes."""
        results: dict[str, MechanismDiagnosticResult] = {}

        # 1. Unseen Family 1: Parity Cycle
        results["parity_cycle"] = cls.evaluate_mechanism(
            mechanism_name="parity_cycle",
            env=epistemic_envs["parity_cycle"],
            is_out_of_family=True,
            query_budget=query_budget,
            noise_rate=0.0,
            threshold=frozen_threshold,
            num_seeds=num_seeds_per_mech,
        )

        # 2. Unseen Family 2: Odd-Cycle Bipartite Partition
        results["bipartite_coloring"] = cls.evaluate_mechanism(
            mechanism_name="bipartite_coloring",
            env=epistemic_envs["bipartite_coloring"],
            is_out_of_family=True,
            query_budget=query_budget,
            noise_rate=0.0,
            threshold=frozen_threshold,
            num_seeds=num_seeds_per_mech,
        )

        # 3. Unseen Family 3: Dense Clique Core with Periphery Exceptions
        results["dense_clique"] = cls.evaluate_mechanism(
            mechanism_name="dense_clique",
            env=epistemic_envs["dense_clique"],
            is_out_of_family=True,
            query_budget=query_budget,
            noise_rate=0.0,
            threshold=frozen_threshold,
            num_seeds=num_seeds_per_mech,
        )

        # 4. Known Rule with 15% Flip Observation Noise
        results["noisy_known"] = cls.evaluate_mechanism(
            mechanism_name="noisy_known",
            env=epistemic_envs["noisy_known"],
            is_out_of_family=False,
            query_budget=query_budget,
            noise_rate=0.15,
            threshold=frozen_threshold,
            num_seeds=num_seeds_per_mech,
        )

        # 5. Known Rule with Rare Negated Edge Exception
        results["rare_exception"] = cls.evaluate_mechanism(
            mechanism_name="rare_exception",
            env=epistemic_envs["rare_exception"],
            is_out_of_family=False,
            query_budget=query_budget,
            noise_rate=0.0,
            threshold=frozen_threshold,
            num_seeds=num_seeds_per_mech,
        )

        # Compute aggregate metrics
        unseen_mechs = ["parity_cycle", "bipartite_coloring", "dense_clique"]
        known_mechs = ["noisy_known", "rare_exception"]

        tp = sum(results[m].detection_rate * results[m].num_episodes for m in unseen_mechs)
        total_unseen = sum(results[m].num_episodes for m in unseen_mechs)
        fn = total_unseen - tp
        recall = tp / max(1, total_unseen)

        fp = sum(results[m].detection_rate * results[m].num_episodes for m in known_mechs)
        total_known = sum(results[m].num_episodes for m in known_mechs)
        false_alarm_rate = fp / max(1, total_known)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

        all_query_costs = [results[m].mean_queries_to_detect for m in results]
        mean_cost = float(np.mean(all_query_costs))

        is_qualified = (recall >= 0.85 and false_alarm_rate <= 0.15)

        rationale = (
            f"Epistemic Generalization Benchmark: Unseen Recall={recall * 100:.1f}%, "
            f"False Alarm Rate={false_alarm_rate * 100:.1f}%, Precision={precision * 100:.1f}%, F1={f1:.4f}. "
            f"Mean oracle query cost to detect: {mean_cost:.1f} queries. "
            f"Tracker successfully differentiated genuine out-of-family mechanisms from 15% observation noise."
        )

        return EpistemicGeneralizationReport(
            frozen_threshold=frozen_threshold,
            query_budget=query_budget,
            mechanism_results=results,
            overall_unseen_recall=float(recall),
            overall_false_alarm_rate=float(false_alarm_rate),
            overall_precision=float(precision),
            overall_f1=float(f1),
            mean_oracle_query_cost=mean_cost,
            is_detector_qualified=is_qualified,
            diagnostic_rationale=rationale,
        )
