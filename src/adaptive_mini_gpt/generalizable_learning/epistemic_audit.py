"""Phase 6: Corrected SPRT Epistemic Auditor with Zero-Defect Ground Truth.

Resolves M58's 100% false alarm failure:
1. Oracle truth evaluation uses `env.evaluate_ground_truth(u, v)` (transitive closure)
   instead of `(u, v) in edges` (direct edges only).
2. Proper likelihood ratio updates using calibrated noise model.
3. Tests across noise levels: 0%, 5%, 15%, 25%.
4. Computes true balanced binary Macro F1 across both KNOWN and UNKNOWN classes.
"""
from __future__ import annotations

import logging
import math
import random
import time
from dataclasses import dataclass
from typing import Any

import numpy as np

from adaptive_mini_gpt.guided_experiment.environment import AutonomousEnvironment
from adaptive_mini_gpt.reliable_transfer.noise_aware_belief import SequentialProbabilityRatioTest
from adaptive_mini_gpt.generalizable_learning.task_suite import (
    create_dev_balanced_hierarchy_environment,
    create_dev_incompatible_parity_environment,
    create_dev_dense_cluster_exceptions_environment,
)

logger = logging.getLogger(__name__)


@dataclass
class EpistemicRegimeDiagnostic:
    """Detailed audit of epistemic detection across noise levels and policies."""
    regime_name: str
    noise_level: float
    query_policy: str  # "graph_uniform_random" or "graph_contrastive_targeted"
    total_episodes: int
    unknown_rule_recall: float
    known_world_false_alarm_rate: float
    detection_macro_f1: float  # Balanced macro F1 across KNOWN and UNKNOWN classes
    positive_class_f1: float   # M58 reported metric (shown for direct audit comparison)
    mean_queries_to_decision: float
    mean_latency_ms: float


class CorrectedEpistemicAuditor:
    """Auditor testing corrected SPRT epistemic detection under observation noise."""

    def __init__(self, budget: int = 12, seed: int = 42) -> None:
        self.budget = budget
        self.rng = random.Random(seed)

    def _generate_contrastive_query_pool(self, env: AutonomousEnvironment) -> list[tuple[str, str]]:
        """Identify candidate pairs where multi-hop relational transitions and boundaries are concentrated."""
        nodes = list(env.entities)
        edges = set(env.base_edges)
        candidates = []

        # 1. Direct base edges (tests symmetry, parity, refutations)
        for u, v in edges:
            candidates.append((u, v))

        # 2. Multi-hop candidates (u -> v -> w)
        for u in nodes:
            for v in nodes:
                if (u, v) in edges:
                    for w in nodes:
                        if (v, w) in edges and u != w and (u, w) not in edges:
                            if (u, w) not in candidates:
                                candidates.append((u, w))

        # 3. Peripheral/boundary candidates
        deg = {n: 0 for n in nodes}
        for u, v in edges:
            deg[u] = deg.get(u, 0) + 1
            deg[v] = deg.get(v, 0) + 1
        periphery = [n for n in nodes if deg[n] <= 1]
        core = [n for n in nodes if deg[n] > 1]
        for c in core:
            for p in periphery:
                if c != p and (c, p) not in candidates:
                    candidates.append((c, p))

        # 4. Remaining node pairs
        for u in nodes:
            for v in nodes:
                if u != v and (u, v) not in candidates:
                    candidates.append((u, v))

        return candidates

    def run_graph_episode(
        self,
        env: AutonomousEnvironment,
        query_policy: str,
        noise_level: float,
        sprt: SequentialProbabilityRatioTest,
        use_corrected_oracle: bool = True,
    ) -> tuple[bool, int, float]:
        """Execute a single epistemic query episode.

        If `use_corrected_oracle` is True, evaluates true transitive reachability via
        `env.evaluate_ground_truth(u, v)`. If False, reproduces M58's defective direct edge check.
        """
        nodes = list(env.entities)
        edges = set(env.base_edges)

        if query_policy == "graph_uniform_random":
            pool = [(u, v) for u in nodes for v in nodes if u != v]
        elif query_policy == "graph_contrastive_targeted":
            pool = self._generate_contrastive_query_pool(env)
        else:
            raise ValueError(f"Unknown query policy: {query_policy}")

        history: set[tuple[str, str]] = set()
        observations: list[tuple[str, str]] = []

        t0 = time.perf_counter()
        for step in range(self.budget):
            if query_policy == "graph_uniform_random":
                pair = self.rng.choice(pool)
            else:
                unqueried = [p for p in pool if p not in history]
                pair = unqueried[0] if unqueried else self.rng.choice(pool)

            history.add(pair)
            u, v = pair

            # Oracle Ground Truth: Transitive Reachability vs Defective Direct Edge
            if use_corrected_oracle:
                actual_answer = env.evaluate_ground_truth(u, v)
            else:
                actual_answer = "YES" if (u, v) in edges else "NO"

            # Observation noise
            if self.rng.random() < noise_level:
                obs_answer = "NO" if actual_answer == "YES" else "YES"
            else:
                obs_answer = actual_answer

            # Predicted answer under transitive hypothesis
            reach = False
            visited = set()
            queue = [u]
            while queue:
                curr = queue.pop(0)
                if curr == v:
                    reach = True
                    break
                visited.add(curr)
                for src, dst in edges:
                    if src == curr and dst not in visited:
                        queue.append(dst)

            pred_answer = "YES" if reach else "NO"
            observations.append((pred_answer, obs_answer))

            decision = sprt.evaluate_episode(observations, budget=len(observations))
            if decision.queries_consumed < len(observations):
                break

        latency_ms = (time.perf_counter() - t0) * 1000.0
        final_decision = sprt.evaluate_episode(observations, budget=self.budget)
        return final_decision.is_unknown_declared, len(observations), latency_ms

    def evaluate_regime(
        self,
        noise_level: float,
        query_policy: str,
        episodes_per_world: int = 10,
        use_corrected_oracle: bool = True,
    ) -> EpistemicRegimeDiagnostic:
        """Evaluate epistemic performance across known and unknown rule worlds."""
        known_env = create_dev_balanced_hierarchy_environment(seed=self.rng.randint(1, 10000))
        unknown_env = create_dev_incompatible_parity_environment(seed=self.rng.randint(1, 10000))

        sprt = SequentialProbabilityRatioTest(
            assumed_noise_floor=max(0.01, noise_level),
        )

        tp, fn, fp, tn = 0, 0, 0, 0
        queries_used: list[int] = []
        latencies: list[float] = []

        # 1. Unknown Rule Worlds (Positive class = UNKNOWN)
        for _ in range(episodes_per_world):
            detected, q_used, lat = self.run_graph_episode(
                unknown_env, query_policy, noise_level, sprt, use_corrected_oracle=use_corrected_oracle
            )
            if detected:
                tp += 1
            else:
                fn += 1
            queries_used.append(q_used)
            latencies.append(lat)

        # 2. Known Rule Worlds (Negative class = KNOWN)
        for _ in range(episodes_per_world):
            detected, q_used, lat = self.run_graph_episode(
                known_env, query_policy, noise_level, sprt, use_corrected_oracle=use_corrected_oracle
            )
            if detected:
                fp += 1
            else:
                tn += 1
            queries_used.append(q_used)
            latencies.append(lat)

        recall = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        far = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0

        # Positive-class F1 (M58 metric)
        prec_pos = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
        pos_f1 = (2 * prec_pos * recall / (prec_pos + recall)) if (prec_pos + recall) > 0 else 0.0

        # Known-class F1 (Negative class)
        spec = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
        prec_neg = float(tn / (tn + fn)) if (tn + fn) > 0 else 0.0
        neg_f1 = (2 * prec_neg * spec / (prec_neg + spec)) if (prec_neg + spec) > 0 else 0.0

        # True Balanced Binary Macro F1
        macro_f1 = (pos_f1 + neg_f1) / 2.0

        regime_label = f"Noise_{int(noise_level*100)}%_{query_policy}" + ("_corrected" if use_corrected_oracle else "_m58_flawed")

        return EpistemicRegimeDiagnostic(
            regime_name=regime_label,
            noise_level=noise_level,
            query_policy=query_policy,
            total_episodes=episodes_per_world * 2,
            unknown_rule_recall=recall,
            known_world_false_alarm_rate=far,
            detection_macro_f1=macro_f1,
            positive_class_f1=pos_f1,
            mean_queries_to_decision=float(np.mean(queries_used)),
            mean_latency_ms=float(np.mean(latencies)),
        )

    def run_full_epistemic_audit(self, episodes_per_world: int = 10) -> list[EpistemicRegimeDiagnostic]:
        """Run complete 4-level noise diagnostic for both corrected and flawed regimes."""
        results = []
        # Corrected regime across noise levels
        for noise in [0.0, 0.05, 0.15, 0.25]:
            results.append(self.evaluate_regime(noise, "graph_contrastive_targeted", episodes_per_world, use_corrected_oracle=True))
            results.append(self.evaluate_regime(noise, "graph_uniform_random", episodes_per_world, use_corrected_oracle=True))

        # Flawed M58 regime for comparison
        results.append(self.evaluate_regime(0.0, "graph_contrastive_targeted", episodes_per_world, use_corrected_oracle=False))
        return results
