"""Phase 7: Epistemic Verification Check on Sparse Graphs with Observation Noise.

Evaluates the Sequential Probability Ratio Test (SPRT) under:
- Noise levels: 0%, 5%, 15%, 25% observation noise
- Query exploration policies:
  * graph_uniform_random (arbitrary pair sampling)
  * graph_contrastive_targeted (hypothesis-directed boundary and transitive edge sampling)
- Sparse graphs with rare peripheral exceptions

Preserves independence from the adaptive optimization benchmark to avoid confounded claims.
"""
from __future__ import annotations

import logging
import math
import random
import time
from dataclasses import dataclass
from typing import Any

from adaptive_mini_gpt.guided_experiment.environment import AutonomousEnvironment
from adaptive_mini_gpt.reliable_transfer.noise_aware_belief import SequentialProbabilityRatioTest
from adaptive_mini_gpt.adaptive_optimization.task_environments import (
    create_dev_balanced_hierarchy_environment,
    create_dev_dense_cluster_exceptions_environment,
    create_meta_modular_network_biomedical,
)

logger = logging.getLogger(__name__)


@dataclass
class EpistemicRegimeDiagnostic:
    """Detailed audit of SPRT across noise levels and query exploration policies."""
    regime_name: str
    noise_level: float
    query_policy: str  # "graph_uniform_random" or "graph_contrastive_targeted"
    total_episodes: int
    unknown_rule_recall: float
    known_world_false_alarm_rate: float
    decision_macro_f1: float
    mean_queries_to_decision: float
    mean_latency_ms: float


class M58EpistemicAuditor:
    """Independent auditor evaluating SPRT epistemic robustness on sparse graphs."""

    def __init__(self, budget: int = 12, seed: int = 42) -> None:
        self.budget = budget
        self.rng = random.Random(seed)

    def _generate_contrastive_query_pool(self, env: AutonomousEnvironment) -> list[tuple[str, str]]:
        """Identify candidate pairs where relational contradictions are concentrated.

        Rather than uniform sampling over all N*(N-1) pairs (which is >85% non-edges),
        contrastive probing focuses on 2-hop paths (u -> v -> w) to test transitive closure.
        """
        nodes = list(env.entities)
        edges = set(env.base_edges)
        two_hop_candidates = []

        for u in nodes:
            for v in nodes:
                if (u, v) in edges:
                    for w in nodes:
                        if (v, w) in edges and u != w and (u, w) not in edges:
                            two_hop_candidates.append((u, w))

        # Augment with remaining pairs
        for u in nodes:
            for v in nodes:
                if u != v and (u, v) not in two_hop_candidates:
                    two_hop_candidates.append((u, v))

        return two_hop_candidates

    def run_graph_episode(
        self,
        env: AutonomousEnvironment,
        query_policy: str,
        noise_level: float,
        sprt: SequentialProbabilityRatioTest,
    ) -> tuple[bool, int, float]:
        """Execute a single graph query episode."""
        nodes = list(env.entities)
        all_pairs = [(u, v) for u in nodes for v in nodes if u != v]

        if query_policy == "graph_contrastive_targeted":
            candidate_pool = self._generate_contrastive_query_pool(env)
        else:
            candidate_pool = list(all_pairs)
            self.rng.shuffle(candidate_pool)

        sprt.reset()
        t0 = time.perf_counter()
        query_count = 0
        detected_unknown = False

        for u, v in candidate_pool:
            if query_count >= self.budget:
                break

            query_count += 1
            # Ground truth reachability in true graph
            true_ans = env.oracle_reachability(u, v)

            # Apply observation noise
            if self.rng.random() < noise_level:
                obs_ans = not true_ans
            else:
                obs_ans = true_ans

            # Believed / known hypothesis (transitive DAG assumption)
            believed_ans = env.query_assumed_hypothesis(u, v)

            # Update SPRT with observation consistency
            is_match = (obs_ans == believed_ans)
            decision = sprt.update_evidence(is_consistent=is_match)

            if decision == "REJECT_KNOWN_RULE":
                detected_unknown = True
                break
            elif decision == "ACCEPT_KNOWN_RULE":
                detected_unknown = False
                break

        t1 = time.perf_counter()
        lat_ms = (t1 - t0) * 1000.0
        return detected_unknown, query_count, lat_ms

    def evaluate_regime(
        self,
        noise_level: float,
        query_policy: str,
        episodes_per_world: int = 10,
    ) -> EpistemicRegimeDiagnostic:
        """Evaluate a specific noise level and exploration policy across known and unknown rule worlds."""
        from adaptive_mini_gpt.reliable_transfer.noise_aware_belief import SequentialProbabilityRatioTest

        known_env = create_dev_balanced_hierarchy_environment(seed=self.rng.randint(1, 10000))
        unknown_env = create_dev_dense_cluster_exceptions_environment(seed=self.rng.randint(1, 10000))

        unknown_detected_count = 0
        known_false_alarm_count = 0
        queries_used = []
        latencies = []

        sprt = SequentialProbabilityRatioTest(
            assumed_noise_floor=max(0.01, noise_level),
            log_upper_bound=math.log(19.0),
            log_lower_bound=math.log(1.0 / 19.0),
        )

        # 1. Unknown Rule Worlds (Test Recall)
        for _ in range(episodes_per_world):
            detected, q_used, lat = self.run_graph_episode(unknown_env, query_policy, noise_level, sprt)
            if detected:
                unknown_detected_count += 1
            queries_used.append(q_used)
            latencies.append(lat)

        # 2. Known Rule Worlds (Test False Alarm Rate)
        for _ in range(episodes_per_world):
            detected, q_used, lat = self.run_graph_episode(known_env, query_policy, noise_level, sprt)
            if detected:
                known_false_alarm_count += 1
            queries_used.append(q_used)
            latencies.append(lat)

        recall = float(unknown_detected_count / episodes_per_world)
        far = float(known_false_alarm_count / episodes_per_world)

        prec = float(unknown_detected_count / (unknown_detected_count + known_false_alarm_count)) if (unknown_detected_count + known_false_alarm_count) > 0 else 0.0
        macro_f1 = (2.0 * prec * recall) / (prec + recall) if (prec + recall) > 0 else 0.0

        regime_name = f"Noise_{int(noise_level*100)}pct_{query_policy}"
        return EpistemicRegimeDiagnostic(
            regime_name=regime_name,
            noise_level=noise_level,
            query_policy=query_policy,
            total_episodes=episodes_per_world * 2,
            unknown_rule_recall=recall,
            known_world_false_alarm_rate=far,
            decision_macro_f1=macro_f1,
            mean_queries_to_decision=float(np.mean(queries_used)),
            mean_latency_ms=float(np.mean(latencies)),
        )

    def run_full_epistemic_audit(self) -> list[EpistemicRegimeDiagnostic]:
        """Audit SPRT across 0%, 5%, 15%, 25% noise under uniform vs contrastive probing."""
        noise_levels = [0.0, 0.05, 0.15, 0.25]
        policies = ["graph_uniform_random", "graph_contrastive_targeted"]
        diagnostics = []

        for noise in noise_levels:
            for pol in policies:
                diag = self.evaluate_regime(noise, pol, episodes_per_world=10)
                diagnostics.append(diag)
                logger.info("Epistemic Diagnostic [%s]: Recall=%.2f, FAR=%.2f, Queries=%.2f",
                            diag.regime_name, diag.unknown_rule_recall, diag.known_world_false_alarm_rate, diag.mean_queries_to_decision)

        return diagnostics
