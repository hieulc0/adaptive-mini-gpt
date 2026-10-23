"""Phase 6: Independent Epistemic Qualification for M60.

Evaluates Sequential Probability Ratio Test (SPRT) across:
1. Multiple unknown-rule families (Bipartite, Parity, Dense Clique, Non-transitive clusters).
2. Multiple difficult known-rule controls (Deep DAG, Branching Tree, Corporate Hierarchy).
3. Observation noise regimes: 0%, 5%, 15%, 25%.
4. Probing policies: `graph_contrastive_targeted` vs `graph_uniform_random`.
5. Oracle validation verified against an independent graph reachability engine.
"""
from __future__ import annotations

import logging
import math
import random
import time
from dataclasses import dataclass
from typing import Any

import numpy as np

from adaptive_mini_gpt.causal_meta_learning.task_suite import (
    compute_independent_graph_reachability,
    create_dev_balanced_hierarchy_environment,
    create_dev_deep_chain_environment,
    create_dev_dense_cluster_exceptions_environment,
    create_dev_incompatible_parity_environment,
    create_dev_non_transitive_bipartite_environment,
    create_dev_wide_branching_tree_environment,
)
from adaptive_mini_gpt.guided_experiment.environment import AutonomousEnvironment

logger = logging.getLogger(__name__)


@dataclass
class EpistemicValidationRecord:
    """Diagnostic outcome of an epistemic testing session."""
    regime_name: str
    noise_level: float
    probing_policy: str
    episodes_evaluated: int
    unknown_rule_recall: float
    known_world_false_alarm_rate: float
    detection_macro_f1: float
    positive_class_f1: float
    mean_decision_queries: float
    mean_latency_ms: float
    epistemic_pass: bool


class IndependentEpistemicValidator:
    """Rigorous epistemic tester for unknown rule detection under observation noise."""

    def __init__(
        self,
        alpha: float = 0.05,
        beta: float = 0.05,
        budget: int = 12,
        seed: int = 42,
    ) -> None:
        self.alpha = alpha
        self.beta = beta
        self.budget = budget
        self.seed = seed
        self.bound_a = math.log((1.0 - beta) / alpha)
        self.bound_b = math.log(beta / (1.0 - alpha))

    def _sample_query(
        self,
        env: AutonomousEnvironment,
        policy: str,
        rng: random.Random,
    ) -> tuple[str, str]:
        nodes = list(env.entities)
        if policy == "graph_uniform_random":
            u, v = rng.sample(nodes, 2)
            return u, v

        # graph_contrastive_targeted: sample 2-hop reachability candidates
        edges = list(env.base_edges)
        adj: dict[str, list[str]] = {n: [] for n in nodes}
        for u, v in edges:
            adj[u].append(v)

        two_hop_candidates = []
        for u in nodes:
            for mid in adj.get(u, []):
                for w in adj.get(mid, []):
                    if w != u and (u, w) not in edges:
                        two_hop_candidates.append((u, w))

        if two_hop_candidates:
            return rng.choice(two_hop_candidates)
        return rng.sample(nodes, 2)

    def run_epistemic_episode(
        self,
        env: AutonomousEnvironment,
        is_known_world: bool,
        noise_level: float,
        policy: str,
        rng: random.Random,
    ) -> tuple[str, int, float]:
        """Run single SPRT episode with independent reachability oracle."""
        t0 = time.perf_counter()
        log_likelihood_ratio = 0.0
        queries_used = 0

        # Independent graph reachability
        independent_reach = compute_independent_graph_reachability(
            list(env.entities), list(env.base_edges)
        )

        p_noise = max(1e-4, min(0.49, noise_level))
        p_match = 1.0 - p_noise

        for step in range(1, self.budget + 1):
            queries_used = step
            u, v = self._sample_query(env, policy, rng)

            # Hypothesis prediction (transitive reachability)
            hyp_pred = independent_reach.get((u, v), False)

            # Oracle truth from environment
            actual_truth = env.evaluate_ground_truth(u, v)

            # Apply observation noise to oracle measurement
            obs = actual_truth
            if rng.random() < noise_level:
                obs = "NO" if actual_truth == "YES" else "YES"

            hyp_label = "YES" if hyp_pred else "NO"
            is_consistent = (hyp_label == obs)

            # Likelihood update:
            # Under H0 (known transitive rule): P(consistent) = 1 - noise, P(inconsistent) = noise
            # Under H1 (unknown rule): P(consistent) = 0.5, P(inconsistent) = 0.5
            if is_consistent:
                step_llr = math.log(0.5 / p_match)
            else:
                step_llr = math.log(0.5 / p_noise)

            log_likelihood_ratio += step_llr

            # Decision boundaries
            if log_likelihood_ratio >= self.bound_a:
                lat = (time.perf_counter() - t0) * 1000.0
                return "UNKNOWN", queries_used, lat
            if log_likelihood_ratio <= self.bound_b:
                lat = (time.perf_counter() - t0) * 1000.0
                return "KNOWN", queries_used, lat

        decision = "UNKNOWN" if log_likelihood_ratio > 0.0 else "KNOWN"
        lat = (time.perf_counter() - t0) * 1000.0
        return decision, queries_used, lat

    def evaluate_regime(
        self,
        noise_level: float,
        policy: str,
        episodes_per_world: int = 10,
    ) -> EpistemicValidationRecord:
        """Evaluate epistemic performance across known and unknown rule families."""
        rng = random.Random(self.seed + int(noise_level * 1000))

        known_envs = [
            create_dev_deep_chain_environment(seed=self.seed),
            create_dev_wide_branching_tree_environment(seed=self.seed),
            create_dev_balanced_hierarchy_environment(seed=self.seed),
        ]
        unknown_envs = [
            create_dev_dense_cluster_exceptions_environment(seed=self.seed),
            create_dev_incompatible_parity_environment(seed=self.seed),
            create_dev_non_transitive_bipartite_environment(seed=self.seed),
        ]

        total_episodes = 0
        known_false_alarms = 0
        known_total = 0
        unknown_detected = 0
        unknown_total = 0

        queries_list: list[int] = []
        latencies_list: list[float] = []

        # 1. Known world trials
        for env in known_envs:
            for _ in range(episodes_per_world):
                dec, q_used, lat = self.run_epistemic_episode(
                    env, is_known_world=True, noise_level=noise_level, policy=policy, rng=rng
                )
                total_episodes += 1
                known_total += 1
                queries_list.append(q_used)
                latencies_list.append(lat)
                if dec == "UNKNOWN":
                    known_false_alarms += 1

        # 2. Unknown world trials
        for env in unknown_envs:
            for _ in range(episodes_per_world):
                dec, q_used, lat = self.run_epistemic_episode(
                    env, is_known_world=False, noise_level=noise_level, policy=policy, rng=rng
                )
                total_episodes += 1
                unknown_total += 1
                queries_list.append(q_used)
                latencies_list.append(lat)
                if dec == "UNKNOWN":
                    unknown_detected += 1

        recall = float(unknown_detected / unknown_total) if unknown_total > 0 else 0.0
        far = float(known_false_alarms / known_total) if known_total > 0 else 0.0

        # Balanced Macro F1
        # Class 1: UNKNOWN (TP=unknown_detected, FP=known_false_alarms, FN=unknown_total - unknown_detected)
        prec_unk = unknown_detected / (unknown_detected + known_false_alarms) if (unknown_detected + known_false_alarms) > 0 else 0.0
        rec_unk = recall
        f1_unk = (2.0 * prec_unk * rec_unk / (prec_unk + rec_unk)) if (prec_unk + rec_unk) > 0 else 0.0

        # Class 2: KNOWN (TN=known_total - known_false_alarms, FN=known_false_alarms, FP=unknown_total - unknown_detected)
        known_correct = known_total - known_false_alarms
        prec_kn = known_correct / (known_correct + (unknown_total - unknown_detected)) if (known_correct + (unknown_total - unknown_detected)) > 0 else 0.0
        rec_kn = known_correct / known_total if known_total > 0 else 0.0
        f1_kn = (2.0 * prec_kn * rec_kn / (prec_kn + rec_kn)) if (prec_kn + rec_kn) > 0 else 0.0

        macro_f1 = float((f1_unk + f1_kn) / 2.0)

        epistemic_pass = (far <= 0.15 and recall >= 0.70)

        return EpistemicValidationRecord(
            regime_name=f"Noise_{int(noise_level * 100)}%_{policy}",
            noise_level=noise_level,
            probing_policy=policy,
            episodes_evaluated=total_episodes,
            unknown_rule_recall=recall,
            known_world_false_alarm_rate=far,
            detection_macro_f1=macro_f1,
            positive_class_f1=f1_unk,
            mean_decision_queries=float(np.mean(queries_list)),
            mean_latency_ms=float(np.mean(latencies_list)),
            epistemic_pass=epistemic_pass,
        )
