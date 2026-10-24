"""Phase 2: Correct Real Learning-Cost Accounting & Multi-Episode Amortization Ledger.

Tracks true measured resource expenditures:
- Oracle query count
- GPU optimizer steps and GPU seconds
- Candidate validation cost
- Research adapter storage parameters
- Active trainable parameters
- Inference latency
- Life-cycle multi-episode amortization across expected future encounters (K >= 2)
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class EpisodeCostLedger:
    """Detailed ledger of actual measured learning costs for a single episode."""
    action_name: str
    macro_f1: float
    oracle_queries: int
    optimizer_steps: int
    gpu_seconds: float
    active_trainable_params: int
    stored_adapter_params: int
    inference_latency_ms: float
    false_allocation_count: int
    missed_reuse_count: int
    candidate_rejection_count: int
    single_episode_net_utility: float
    amortized_life_cycle_utility: float
    cost_breakdown: dict[str, float]


class RealLearningCostLedger:
    """Computes transparent cost-adjusted utility with life-cycle reuse amortization."""

    def __init__(
        self,
        lambda_query: float = 0.001,       # Cost per oracle query
        lambda_step: float = 0.0005,       # Cost per optimizer step
        lambda_gpu_sec: float = 0.002,     # Cost per GPU second
        lambda_storage: float = 0.00002,   # Cost per stored parameter
        penalty_false_alloc: float = 0.10, # Penalty for redundant skill creation
        penalty_missed_reuse: float = 0.05,# Penalty for missing an applicable skill
        expected_life_cycle_episodes: int = 3, # K encounters over life cycle
    ) -> None:
        self.lambda_query = lambda_query
        self.lambda_step = lambda_step
        self.lambda_gpu_sec = lambda_gpu_sec
        self.lambda_storage = lambda_storage
        self.penalty_false_alloc = penalty_false_alloc
        self.penalty_missed_reuse = penalty_missed_reuse
        self.expected_k = expected_life_cycle_episodes

    def compute_episode_utility(
        self,
        action_name: str,
        macro_f1: float,
        oracle_queries: int,
        optimizer_steps: int,
        gpu_seconds: float,
        active_trainable_params: int,
        stored_adapter_params: int,
        inference_latency_ms: float,
        false_allocation_count: int = 0,
        missed_reuse_count: int = 0,
        candidate_rejection_count: int = 0,
    ) -> EpisodeCostLedger:
        """Compute both unamortized single-episode utility and amortized life-cycle utility."""
        query_cost = self.lambda_query * oracle_queries
        step_cost = self.lambda_step * optimizer_steps
        gpu_cost = self.lambda_gpu_sec * gpu_seconds
        storage_cost = self.lambda_storage * stored_adapter_params
        penalty_cost = (
            self.penalty_false_alloc * false_allocation_count
            + self.penalty_missed_reuse * missed_reuse_count
        )

        total_direct_cost = query_cost + step_cost + gpu_cost + storage_cost + penalty_cost
        single_net_utility = macro_f1 - total_direct_cost

        # Life-cycle amortization:
        # If an action creates a permanent skill (TRAIN_NEW_SKILL), the upfront training & validation
        # cost is amortized across K expected encounters. For subsequent encounters, training cost is 0.
        # Conversely, FRESH_ADAPT must pay the full training cost on EVERY single encounter!
        if action_name in ("TRAIN_NEW_SKILL", "PROPOSE_NEW"):
            # Upfront acquisition cost amortized over K episodes
            amortized_acquisition_cost = (query_cost + step_cost + gpu_cost) / float(self.expected_k)
            amortized_utility = macro_f1 - (amortized_acquisition_cost + storage_cost + penalty_cost)
        elif action_name in ("FRESH_ADAPT", "ALWAYS_FRESH"):
            # Fresh adaptation pays full training compute on every encounter
            amortized_utility = macro_f1 - (query_cost + step_cost + gpu_cost + penalty_cost)
        else:  # REUSE
            # Reuse only pays small probe cost
            amortized_utility = macro_f1 - (query_cost + penalty_cost)

        cost_breakdown = {
            "query_cost": query_cost,
            "optimizer_step_cost": step_cost,
            "gpu_time_cost": gpu_cost,
            "storage_cost": storage_cost,
            "penalty_cost": penalty_cost,
            "total_direct_cost": total_direct_cost,
        }

        return EpisodeCostLedger(
            action_name=action_name,
            macro_f1=macro_f1,
            oracle_queries=oracle_queries,
            optimizer_steps=optimizer_steps,
            gpu_seconds=gpu_seconds,
            active_trainable_params=active_trainable_params,
            stored_adapter_params=stored_adapter_params,
            inference_latency_ms=inference_latency_ms,
            false_allocation_count=false_allocation_count,
            missed_reuse_count=missed_reuse_count,
            candidate_rejection_count=candidate_rejection_count,
            single_episode_net_utility=single_net_utility,
            amortized_life_cycle_utility=amortized_utility,
            cost_breakdown=cost_breakdown,
        )
