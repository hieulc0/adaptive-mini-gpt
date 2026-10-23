"""Phase 6: Independent Epistemic Qualification for M60.

Evaluates Sequential Probability Ratio Test (SPRT) across:
1. Unknown-rule environments (Non-transitive bipartite partition where 2-hop relations fail).
2. Difficult known-rule controls (Hierarchical corporate structures).
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
    create_dev_non_transitive_bipartite_environment,
)
from adaptive_mini_gpt.guided_experiment.environment import AutonomousEnvironment
from adaptive_mini_gpt.reliable_transfer.noise_aware_belief import (
    SequentialProbabilityRatioTest,
)

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
        self.rng = random.Random(seed)

    def _generate_contrastive_pool(self, env: AutonomousEnvironment) -> list[tuple[str, str]]:
        """Generate ordered query pool prioritizing 2-hop candidate edges followed by remaining pairs."""
        nodes = list(env.entities)
        edges = set(env.base_edges)
        two_hop_candidates = []

        for u in nodes:
            for v in nodes:
                if (u, v) in edges:
                    for w in nodes:
                        if (v, w) in edges and u != w and (u, w) not in edges:
                            two_hop_candidates.append((u, w))

        for u in nodes:
            for v in nodes:
                if u != v and (u, v) not in two_hop_candidates:
                    two_hop_candidates.append((u, v))

        return two_hop_candidates

    def run_epistemic_episode(
        self,
        env: AutonomousEnvironment,
        noise_level: float,
        policy: str,
        sprt: SequentialProbabilityRatioTest,
    ) -> tuple[bool, int, float]:
        """Run single SPRT episode with independent reachability oracle."""
        t0 = time.perf_counter()
        nodes = list(env.entities)
        edges = list(env.base_edges)

        independent_reach = compute_independent_graph_reachability(nodes, edges)

        if policy == "graph_uniform_random":
            pool = [(u, v) for u in nodes for v in nodes if u != v]
        else:
            pool = self._generate_contrastive_pool(env)

        history: set[tuple[str, str]] = set()
        observations: list[tuple[str, str]] = []

        for step in range(self.budget):
            if policy == "graph_uniform_random":
                pair = self.rng.choice(pool)
            else:
                unqueried = [p for p in pool if p not in history]
                pair = unqueried[0] if unqueried else self.rng.choice(pool)

            history.add(pair)
            u, v = pair

            # Oracle Ground Truth (Evaluated via Environment Ground Truth)
            actual_answer = env.evaluate_ground_truth(u, v)

            # Observation noise
            if self.rng.random() < noise_level:
                obs_answer = "NO" if actual_answer == "YES" else "YES"
            else:
                obs_answer = actual_answer

            # Predicted answer under transitive hypothesis
            reach = independent_reach.get((u, v), False)
            pred_answer = "YES" if reach else "NO"

            observations.append((pred_answer, obs_answer))

            decision = sprt.evaluate_episode(observations, budget=len(observations))
            if decision.queries_consumed < len(observations):
                break

        lat = (time.perf_counter() - t0) * 1000.0
        final_decision = sprt.evaluate_episode(observations, budget=self.budget)
        return final_decision.is_unknown_declared, len(observations), lat

    def evaluate_regime(
        self,
        noise_level: float,
        policy: str,
        episodes_per_world: int = 10,
    ) -> EpistemicValidationRecord:
        """Evaluate epistemic performance across known and unknown rule families."""
        self.rng = random.Random(self.seed + int(noise_level * 1000))

        known_env = create_dev_balanced_hierarchy_environment(seed=self.rng.randint(1, 10000))
        unknown_env = create_dev_non_transitive_bipartite_environment(seed=self.rng.randint(1, 10000))

        sprt = SequentialProbabilityRatioTest(
            assumed_noise_floor=max(0.01, noise_level),
        )

        tp, fn, fp, tn = 0, 0, 0, 0
        queries_list: list[int] = []
        latencies_list: list[float] = []

        # 1. Unknown Rule Worlds (Positive class = UNKNOWN)
        for _ in range(episodes_per_world):
            detected, q_used, lat = self.run_epistemic_episode(
                unknown_env, noise_level=noise_level, policy=policy, sprt=sprt
            )
            if detected:
                tp += 1
            else:
                fn += 1
            queries_list.append(q_used)
            latencies_list.append(lat)

        # 2. Known Rule Worlds (Negative class = KNOWN)
        for _ in range(episodes_per_world):
            detected, q_used, lat = self.run_epistemic_episode(
                known_env, noise_level=noise_level, policy=policy, sprt=sprt
            )
            if detected:
                fp += 1
            else:
                tn += 1
            queries_list.append(q_used)
            latencies_list.append(lat)

        recall = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        far = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0

        prec_pos = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
        pos_f1 = (2 * prec_pos * recall / (prec_pos + recall)) if (prec_pos + recall) > 0 else 0.0

        spec = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
        prec_neg = float(tn / (tn + fn)) if (tn + fn) > 0 else 0.0
        neg_f1 = (2 * prec_neg * spec / (prec_neg + spec)) if (prec_neg + spec) > 0 else 0.0

        macro_f1 = float((pos_f1 + neg_f1) / 2.0)
        epistemic_pass = bool(far <= 0.15 and recall >= 0.70)

        return EpistemicValidationRecord(
            regime_name=f"Noise_{int(noise_level * 100)}%_{policy}",
            noise_level=noise_level,
            probing_policy=policy,
            episodes_evaluated=episodes_per_world * 2,
            unknown_rule_recall=recall,
            known_world_false_alarm_rate=far,
            detection_macro_f1=macro_f1,
            positive_class_f1=pos_f1,
            mean_decision_queries=float(np.mean(queries_list)),
            mean_latency_ms=float(np.mean(latencies_list)),
            epistemic_pass=epistemic_pass,
        )
