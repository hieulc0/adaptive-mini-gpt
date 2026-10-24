"""Phase 2: Fair Resource-Budgeted Benchmark & Pareto Analysis.

Provides:
- Multiple budget regimes:
    * Small: 12 queries, 20 optimizer steps.
    * Medium: 20 queries, 40 optimizer steps.
    * Large: 32 queries, 60 optimizer steps.
- Fair adaptation: EVERY policy (including ALWAYS_FRESH and ALWAYS_SPAWN) adapts
  under identical budget constraints.
- Multi-dimensional cost accounting: task quality, queries, steps, storage, GPU time.
- Pareto frontier extraction across quality vs compute cost.
- Sensitivity analysis across reasonable alternative utility cost weights.
"""
from __future__ import annotations

import enum
import logging
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


class BudgetTier(str, enum.Enum):
    """The three standardized resource budget regimes in M69."""
    SMALL = "SMALL"      # 12 queries, 20 steps
    MEDIUM = "MEDIUM"    # 20 queries, 40 steps
    LARGE = "LARGE"      # 32 queries, 60 steps


@dataclass
class BudgetConstraint:
    """Explicit resource constraints for a given budget tier."""
    tier: BudgetTier
    max_queries_per_task: int
    max_steps_per_task: int
    max_training_queries: int
    max_probe_queries: int


BUDGET_REGIMES: dict[BudgetTier, BudgetConstraint] = {
    BudgetTier.SMALL: BudgetConstraint(
        tier=BudgetTier.SMALL,
        max_queries_per_task=12,
        max_steps_per_task=20,
        max_training_queries=8,
        max_probe_queries=4,
    ),
    BudgetTier.MEDIUM: BudgetConstraint(
        tier=BudgetTier.MEDIUM,
        max_queries_per_task=20,
        max_steps_per_task=40,
        max_training_queries=16,
        max_probe_queries=4,
    ),
    BudgetTier.LARGE: BudgetConstraint(
        tier=BudgetTier.LARGE,
        max_queries_per_task=32,
        max_steps_per_task=60,
        max_training_queries=24,
        max_probe_queries=8,
    ),
}


@dataclass
class CostWeightVector:
    """Weights for the composite net utility objective."""
    lambda_query: float = 0.002
    lambda_step: float = 0.0005
    lambda_storage: float = 0.00002
    lambda_gpu_sec: float = 0.001
    penalty_false_alloc: float = 0.05
    penalty_missed_reuse: float = 0.02


@dataclass
class EpisodeBudgetOutcome:
    """Empirical metrics measured for one policy on one task under a budget tier."""
    policy_name: str
    budget_tier: BudgetTier
    task_name: str
    macro_f1: float
    accuracy: float
    queries_used: int
    steps_used: int
    stored_params: int
    active_params: int
    gpu_seconds: float
    false_allocations: int
    missed_reuses: int
    realized_future_reuses: int = 0
    single_episode_utility: float = 0.0
    amortized_utility: float = 0.0
    sensitivity_utilities: dict[str, float] = field(default_factory=dict)


class FairBudgetLedger:
    """Computes exact utility, Pareto frontiers, and sensitivity analysis across budget regimes."""

    def __init__(
        self,
        default_weights: CostWeightVector | None = None,
        expected_life_cycle_episodes: int = 3,
    ) -> None:
        self.weights = default_weights or CostWeightVector()
        self.k_episodes = max(1, expected_life_cycle_episodes)

    def compute_outcome(
        self,
        policy_name: str,
        budget_tier: BudgetTier,
        task_name: str,
        macro_f1: float,
        accuracy: float,
        queries_used: int,
        steps_used: int,
        stored_params: int,
        active_params: int,
        gpu_seconds: float,
        false_allocations: int,
        missed_reuses: int,
        realized_future_reuses: int = 0,
        action_name: str = "REUSE",
    ) -> EpisodeBudgetOutcome:
        """Compute transparent multi-dimensional cost and utility."""
        w = self.weights
        direct_compute_cost = (
            w.lambda_query * queries_used
            + w.lambda_step * steps_used
            + w.lambda_gpu_sec * gpu_seconds
            + w.lambda_storage * stored_params
            + w.penalty_false_alloc * false_allocations
            + w.penalty_missed_reuse * missed_reuses
        )
        single_ep_u = float(macro_f1 - direct_compute_cost)

        # Amortized utility: if candidate is retained, acquisition cost is amortized across K encounters
        k = max(1, realized_future_reuses if realized_future_reuses > 0 else self.k_episodes)
        if action_name in ("TRAIN_NEW_SKILL", "ALWAYS_SPAWN") and stored_params > 0:
            acquisition_overhead = (w.lambda_query * queries_used + w.lambda_step * steps_used)
            amortized_cost = (
                (acquisition_overhead / k)
                + w.lambda_gpu_sec * gpu_seconds
                + w.lambda_storage * stored_params
                + w.penalty_false_alloc * false_allocations
                + w.penalty_missed_reuse * missed_reuses
            )
        else:
            amortized_cost = direct_compute_cost

        amort_u = float(macro_f1 - amortized_cost)

        # Sensitivity analysis with alternative weight vectors
        alt_weights = {
            "compute_heavy": CostWeightVector(lambda_query=0.005, lambda_step=0.001, lambda_gpu_sec=0.004),
            "quality_heavy": CostWeightVector(lambda_query=0.0005, lambda_step=0.0001, lambda_gpu_sec=0.0002),
            "storage_heavy": CostWeightVector(lambda_storage=0.0001, penalty_false_alloc=0.10),
        }
        sensitivity = {}
        for w_name, aw in alt_weights.items():
            aw_cost = (
                aw.lambda_query * queries_used
                + aw.lambda_step * steps_used
                + aw.lambda_gpu_sec * gpu_seconds
                + aw.lambda_storage * stored_params
                + aw.penalty_false_alloc * false_allocations
                + aw.penalty_missed_reuse * missed_reuses
            )
            sensitivity[w_name] = float(macro_f1 - aw_cost)

        return EpisodeBudgetOutcome(
            policy_name=policy_name,
            budget_tier=budget_tier,
            task_name=task_name,
            macro_f1=float(macro_f1),
            accuracy=float(accuracy),
            queries_used=int(queries_used),
            steps_used=int(steps_used),
            stored_params=int(stored_params),
            active_params=int(active_params),
            gpu_seconds=float(gpu_seconds),
            false_allocations=int(false_allocations),
            missed_reuses=int(missed_reuses),
            realized_future_reuses=int(realized_future_reuses),
            single_episode_utility=single_ep_u,
            amortized_utility=amort_u,
            sensitivity_utilities=sensitivity,
        )

    @staticmethod
    def extract_pareto_frontier(
        outcomes: list[EpisodeBudgetOutcome],
    ) -> list[EpisodeBudgetOutcome]:
        """Extract the non-dominated Pareto frontier points (maximizing Macro F1, minimizing compute cost)."""
        frontier = []
        for cand in outcomes:
            cand_cost = cand.queries_used + (cand.steps_used * 0.5)
            is_dominated = False
            for other in outcomes:
                other_cost = other.queries_used + (other.steps_used * 0.5)
                # Dominated if other has strictly higher F1 and equal/lower cost, or equal F1 and strictly lower cost
                if (other.macro_f1 >= cand.macro_f1 and other_cost <= cand_cost) and (
                    other.macro_f1 > cand.macro_f1 or other_cost < cand_cost
                ):
                    is_dominated = True
                    break
            if not is_dominated:
                frontier.append(cand)
        return sorted(frontier, key=lambda x: (x.macro_f1, -x.queries_used))
