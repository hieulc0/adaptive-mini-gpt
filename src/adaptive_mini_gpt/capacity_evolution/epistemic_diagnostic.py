"""Phase 7: Independent Epistemic Verification Diagnostic.

Reconciles SPRT performance between synthetic high-discrepancy streams and
graph exploration regimes (uniform random vs contrastive edge probing) across
0%, 5%, 15%, and 25% observation noise.
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
from adaptive_mini_gpt.capacity_evolution.task_complexity import (
    create_balanced_hierarchy_environment,
    create_dense_cluster_exceptions_environment,
)

logger = logging.getLogger(__name__)


@dataclass
class EpistemicRegimeDiagnostic:
    """Detailed audit of SPRT across noise levels and query exploration policies."""
    regime_name: str
    noise_level: float
    query_policy: str  # "synthetic_dense", "graph_uniform_random", "graph_contrastive_targeted"
    total_episodes: int
    unknown_rule_recall: float
    known_world_false_alarm_rate: float
    decision_macro_f1: float
    mean_queries_to_decision: float
    mean_latency_ms: float


class CalibratedSPRTDiagnosticAuditor:
    """Audits SPRT under matched budgets (Q=12) to reconcile exploration base-rate impacts."""

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

        # If sparse, augment with cross-partition and direct edges
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
        edges = set(env.base_edges)
        is_unknown = "dense" in env.env_id or "parity" in env.env_id or "exception" in env.env_id

        if query_policy == "graph_uniform_random":
            all_pairs = [(u, v) for u in nodes for v in nodes if u != v]
            pool = all_pairs
        elif query_policy == "graph_contrastive_targeted":
            pool = self._generate_contrastive_query_pool(env)
        else:
            raise ValueError(f"Unknown graph query policy: {query_policy}")

        history: set[tuple[str, str]] = set()
        observations: list[tuple[str, str]] = []

        t0 = time.perf_counter()
        for step in range(self.budget):
            # Select next query according to policy
            if query_policy == "graph_uniform_random":
                pair = self.rng.choice(pool)
            else:
                unqueried = [p for p in pool if p not in history]
                pair = unqueried[0] if unqueried else self.rng.choice(pool)

            history.add(pair)
            u, v = pair

            # Oracle truth
            actual_yes = (u, v) in edges
            actual_answer = "YES" if actual_yes else "NO"

            # Apply observation noise
            if self.rng.random() < noise_level:
                obs_answer = "NO" if actual_answer == "YES" else "YES"
            else:
                obs_answer = actual_answer

            # Predicted known rule under transitive closure
            pred_answer = "YES" if (u, v) in edges or any((u, k) in edges and (k, v) in edges for k in nodes) else "NO"

            observations.append((pred_answer, obs_answer))

            # Early stopping check
            decision = sprt.evaluate_episode(observations, budget=len(observations))
            if decision.queries_consumed < len(observations):
                break

        latency_ms = (time.perf_counter() - t0) * 1000.0
        final_decision = sprt.evaluate_episode(observations, budget=self.budget)
        success = (final_decision.is_unknown_declared == is_unknown)
        return success, len(observations), latency_ms

    def audit_epistemic_regimes(
        self,
        noise_levels: list[float] | None = None,
        episodes_per_condition: int = 12,
    ) -> list[EpistemicRegimeDiagnostic]:
        """Execute comparative diagnostic reconciling synthetic vs uniform vs contrastive probing."""
        if noise_levels is None:
            noise_levels = [0.0, 0.05, 0.15, 0.25]

        diagnostics: list[EpistemicRegimeDiagnostic] = []

        env_known = create_balanced_hierarchy_environment(seed=42)
        env_unknown = create_dense_cluster_exceptions_environment(seed=43)

        for noise in noise_levels:
            for policy in ["graph_uniform_random", "graph_contrastive_targeted"]:
                sprt = SequentialProbabilityRatioTest(assumed_noise_floor=max(0.01, noise))

                known_correct = 0
                known_queries = []
                known_latencies = []
                for _ in range(episodes_per_condition):
                    succ, q_used, lat = self.run_graph_episode(env_known, policy, noise, sprt)
                    if succ:
                        known_correct += 1
                    known_queries.append(q_used)
                    known_latencies.append(lat)

                unk_correct = 0
                unk_queries = []
                unk_latencies = []
                for _ in range(episodes_per_condition):
                    succ, q_used, lat = self.run_graph_episode(env_unknown, policy, noise, sprt)
                    if succ:
                        unk_correct += 1
                    unk_queries.append(q_used)
                    unk_latencies.append(lat)

                rec = unk_correct / episodes_per_condition
                fa = 1.0 - (known_correct / episodes_per_condition)
                f1 = (2 * rec * (1.0 - fa)) / (rec + (1.0 - fa) + 1e-8)
                all_queries = known_queries + unk_queries
                all_lats = known_latencies + unk_latencies

                diag = EpistemicRegimeDiagnostic(
                    regime_name=f"Noise_{int(noise*100)}pct_{policy}",
                    noise_level=noise,
                    query_policy=policy,
                    total_episodes=2 * episodes_per_condition,
                    unknown_rule_recall=float(rec),
                    known_world_false_alarm_rate=float(fa),
                    decision_macro_f1=float(f1),
                    mean_queries_to_decision=float(sum(all_queries) / len(all_queries)),
                    mean_latency_ms=float(sum(all_lats) / len(all_lats)),
                )
                diagnostics.append(diag)
                logger.info(
                    "Epistemic Diagnostic [%s]: Recall=%.1f%%, FAR=%.1f%%, F1=%.4f, Mean Queries=%.1f",
                    diag.regime_name, diag.unknown_rule_recall * 100, diag.known_world_false_alarm_rate * 100,
                    diag.decision_macro_f1, diag.mean_queries_to_decision,
                )

        return diagnostics
