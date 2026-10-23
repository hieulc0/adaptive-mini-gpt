"""Phase 5: Controlled Active Probing Benchmark (Random vs Targeted Peripheral/Bridge Probing)."""
from __future__ import annotations

import time
import random
import logging
from dataclasses import dataclass, field
from typing import Any

from adaptive_mini_gpt.guided_experiment.environment import AutonomousEnvironment
from adaptive_mini_gpt.reliable_transfer.noise_aware_belief import (
    SequentialProbabilityRatioTest,
    EpistemicDecision,
)
from adaptive_mini_gpt.reliable_transfer.target_environments import (
    create_dense_clique_diagnostic_environment,
    create_noisy_dense_control_environment,
)

logger = logging.getLogger(__name__)


@dataclass
class ActiveProbingEpisodeResult:
    """Detailed trace for a single active probing episode."""
    strategy: str  # "random_valid", "targeted_peripheral_bridge"
    is_unknown_world: bool
    is_correct_decision: bool
    queries_consumed: int
    repeated_queries_count: int
    invalid_queries_count: int
    useful_evidence_count: int
    total_latency_ms: float
    decision_log_likelihood_ratio: float


@dataclass
class ActiveProbingBenchmarkSummary:
    """Aggregated comparison between random and targeted probing strategies."""
    strategy: str
    total_episodes: int
    unknown_rule_recall: float
    known_world_false_alarm_rate: float
    decision_macro_f1: float
    mean_queries_to_decision: float
    mean_repeated_queries: float
    mean_invalid_queries: float
    mean_useful_evidence_bits: float
    mean_latency_ms: float


class ControlledActiveProbingAuditor:
    """Evaluates query generation strategies under identical oracle budgets ($Q=12$)."""

    def __init__(self, budget: int = 12, noise_rate: float = 0.15, seed: int = 42) -> None:
        self.budget = budget
        self.noise_rate = noise_rate
        self.rng = random.Random(seed)

    def _generate_candidate_pool(self, nodes: list[str]) -> list[tuple[str, str]]:
        pool = []
        for u in nodes:
            for v in nodes:
                if u != v:
                    pool.append((u, v))
        return pool

    def run_probing_episode(
        self,
        strategy: str,
        env: AutonomousEnvironment,
        sprt_detector: SequentialProbabilityRatioTest,
    ) -> ActiveProbingEpisodeResult:
        nodes = env.entities
        edges = set(env.base_edges)
        candidate_pool = self._generate_candidate_pool(nodes)
        is_unknown = "dense_clique" in env.env_id or "parity" in env.env_id

        history: set[tuple[str, str]] = set()
        observations: list[tuple[str, str]] = []

        repeated = 0
        invalid = 0
        useful = 0

        t0 = time.perf_counter()

        for step in range(self.budget):
            if strategy == "random_valid":
                u, v = self.rng.choice(candidate_pool)
            elif strategy == "targeted_peripheral_bridge":
                # Targeted strategy: prioritize boundary nodes and cross-tier pairs
                degrees = {n: 0 for n in nodes}
                for src, dst in edges:
                    degrees[src] = degrees.get(src, 0) + 1
                    degrees[dst] = degrees.get(dst, 0) + 1
                sorted_pool = sorted(
                    candidate_pool,
                    key=lambda pair: abs(degrees.get(pair[0], 0) - degrees.get(pair[1], 0)) + (1 if pair in history else 0),
                    reverse=True,
                )
                u, v = sorted_pool[step % len(sorted_pool)]
            else:
                raise ValueError(f"Unknown strategy: {strategy}")

            if (u, v) in history:
                repeated += 1
            history.add((u, v))

            # Query oracle
            is_connected = (u, v) in edges
            actual_answer = "YES" if is_connected else "NO"

            # Apply observation noise
            if self.rng.random() < self.noise_rate:
                observed_answer = "NO" if actual_answer == "YES" else "YES"
            else:
                observed_answer = actual_answer

            # Predicted answer under baseline transitive/symmetric hypothesis
            hyp_pred = "YES" if (u, v) in edges or (v, u) in edges else "NO"

            if hyp_pred != observed_answer:
                useful += 1

            observations.append((hyp_pred, observed_answer))

            # Check if SPRT has reached decisive boundary
            decision = sprt_detector.evaluate_episode(observations, budget=len(observations))
            if decision.queries_consumed < len(observations):
                break

        latency_ms = (time.perf_counter() - t0) * 1000.0

        final_decision = sprt_detector.evaluate_episode(observations, budget=self.budget)
        correct = (final_decision.is_unknown_declared == is_unknown)

        return ActiveProbingEpisodeResult(
            strategy=strategy,
            is_unknown_world=is_unknown,
            is_correct_decision=correct,
            queries_consumed=len(observations),
            repeated_queries_count=repeated,
            invalid_queries_count=invalid,
            useful_evidence_count=useful,
            total_latency_ms=latency_ms,
            decision_log_likelihood_ratio=final_decision.log_likelihood_ratio,
        )

    def benchmark_strategies(
        self,
        num_known_episodes: int = 15,
        num_unknown_episodes: int = 15,
    ) -> list[ActiveProbingBenchmarkSummary]:
        summaries = []
        for strat in ["random_valid", "targeted_peripheral_bridge"]:
            sprt = SequentialProbabilityRatioTest(assumed_noise_floor=self.noise_rate)
            episodes: list[ActiveProbingEpisodeResult] = []

            # 1. Known Environments with noise
            for i in range(num_known_episodes):
                env = create_noisy_dense_control_environment(seed=100 + i)
                episodes.append(self.run_probing_episode(strat, env, sprt))

            # 2. Unknown Environments (dense clique with peripheral exceptions)
            for i in range(num_unknown_episodes):
                env = create_dense_clique_diagnostic_environment(seed=200 + i)
                episodes.append(self.run_probing_episode(strat, env, sprt))

            unknown_eps = [e for e in episodes if e.is_unknown_world]
            known_eps = [e for e in episodes if not e.is_unknown_world]

            rec = sum(1 for e in unknown_eps if e.is_correct_decision) / len(unknown_eps)
            fa = sum(1 for e in known_eps if not e.is_correct_decision) / len(known_eps)
            f1 = (2 * rec * (1.0 - fa)) / (rec + (1.0 - fa) + 1e-8)

            summary = ActiveProbingBenchmarkSummary(
                strategy=strat,
                total_episodes=len(episodes),
                unknown_rule_recall=rec,
                known_world_false_alarm_rate=fa,
                decision_macro_f1=f1,
                mean_queries_to_decision=float(sum(e.queries_consumed for e in episodes) / len(episodes)),
                mean_repeated_queries=float(sum(e.repeated_queries_count for e in episodes) / len(episodes)),
                mean_invalid_queries=float(sum(e.invalid_queries_count for e in episodes) / len(episodes)),
                mean_useful_evidence_bits=float(sum(e.useful_evidence_count for e in episodes) / len(episodes)),
                mean_latency_ms=float(sum(e.total_latency_ms for e in episodes) / len(episodes)),
            )
            summaries.append(summary)

        return summaries
