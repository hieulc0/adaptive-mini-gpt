"""Milestone M73 Phase 2: Transparent Cost Analysis and Utility Decomposition.

Decomposes real engineering resource metrics from mathematical utility penalties:
1. Physical Resource Metrics:
   - Verified observation queries
   - Candidate training steps
   - Actual GPU training time (seconds)
   - Model inference latency (milliseconds per forward pass)
   - Router evaluation cost (probe queries + adapter forward passes)
   - Candidate qualification cost (qualification queries + evaluation time)
   - Total stored adapter bytes (disk / host RAM / GPU VRAM)
   - Active adapter parameters vs Dormant adapter parameters
   - CPU and GPU peak memory allocated (MB)
   - Realized future reuse count
2. Multi-Tier Utility Accounting:
   - Physical Engineering Utility: charges measured compute seconds, latency, and true memory pressure.
   - Benchmark Utility: recomputes the legacy M70-M72 utility score and shows the exact contribution of each penalty.
   - Sensitivity Analysis: evaluates rank order robustness under compute-heavy, quality-heavy, and storage-heavy regimes.
   - Amortization Transparency: cleanly separates realized future reuse amortization from unamortized episode costs.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class PhysicalResourceRecord:
    """Exact physical resource consumption measured for an episode."""

    observation_queries: int = 0
    candidate_training_steps: int = 0
    gpu_training_seconds: float = 0.0
    inference_latency_ms: float = 0.0
    router_probe_queries: int = 0
    router_evaluation_seconds: float = 0.0
    qualification_queries: int = 0
    qualification_seconds: float = 0.0
    active_adapter_params: int = 0
    dormant_adapter_params: int = 0
    total_stored_adapter_params: int = 0
    total_stored_adapter_bytes: int = 0
    gpu_memory_allocated_mb: float = 0.0
    realized_future_reuses: int = 0


@dataclass
class CostDecompositionBreakdown:
    """Detailed breakdown of each component's contribution to net utility."""

    macro_f1: float
    query_penalty: float
    step_penalty: float
    gpu_time_penalty: float
    storage_penalty: float
    false_allocation_penalty: float
    missed_reuse_penalty: float
    total_cost_penalty: float
    net_utility: float


@dataclass
class TransparentEpisodeBudgetOutcome:
    """Complete multi-dimensional cost and quality outcome for an episode."""

    policy_name: str
    task_name: str
    macro_f1: float
    accuracy: float
    physical_resources: PhysicalResourceRecord
    legacy_utility_breakdown: CostDecompositionBreakdown
    physical_engineering_utility: float
    amortized_physical_utility: float
    sensitivity_utilities: dict[str, float] = field(default_factory=dict)

    def summary_dict(self) -> dict[str, Any]:
        """Return serializable summary."""
        return {
            "policy": self.policy_name,
            "task": self.task_name,
            "macro_f1": self.macro_f1,
            "accuracy": self.accuracy,
            "physical_resources": {
                "queries": self.physical_resources.observation_queries,
                "steps": self.physical_resources.candidate_training_steps,
                "gpu_train_sec": self.physical_resources.gpu_training_seconds,
                "inference_latency_ms": self.physical_resources.inference_latency_ms,
                "stored_params": self.physical_resources.total_stored_adapter_params,
                "stored_bytes": self.physical_resources.total_stored_adapter_bytes,
                "active_params": self.physical_resources.active_adapter_params,
                "dormant_params": self.physical_resources.dormant_adapter_params,
                "gpu_mem_mb": self.physical_resources.gpu_memory_allocated_mb,
                "realized_reuses": self.physical_resources.realized_future_reuses,
            },
            "legacy_utility": self.legacy_utility_breakdown.net_utility,
            "physical_engineering_utility": self.physical_engineering_utility,
            "amortized_physical_utility": self.amortized_physical_utility,
            "sensitivity": self.sensitivity_utilities,
        }


class CostAndUtilityLedger:
    """Evaluates multi-dimensional physical resource costs and mathematical utility."""

    BYTES_PER_FP32_PARAM: int = 4

    def __init__(
        self,
        lambda_query: float = 0.002,
        lambda_step: float = 0.0005,
        lambda_gpu_sec: float = 0.001,
        lambda_storage: float = 0.00002,
        penalty_false_alloc: float = 0.05,
        penalty_missed_reuse: float = 0.02,
        expected_amortization_k: int = 3,
    ) -> None:
        self.lambda_query = lambda_query
        self.lambda_step = lambda_step
        self.lambda_gpu_sec = lambda_gpu_sec
        self.lambda_storage = lambda_storage
        self.penalty_false_alloc = penalty_false_alloc
        self.penalty_missed_reuse = penalty_missed_reuse
        self.k_episodes = max(1, expected_amortization_k)

    def compute_transparent_outcome(
        self,
        policy_name: str,
        task_name: str,
        macro_f1: float,
        accuracy: float,
        resources: PhysicalResourceRecord,
        false_allocations: int = 0,
        missed_reuses: int = 0,
        action_name: str = "REUSE",
    ) -> TransparentEpisodeBudgetOutcome:
        """Compute transparent decomposition across physical and legacy utility models."""
        total_queries = (
            resources.observation_queries
            + resources.router_probe_queries
            + resources.qualification_queries
        )
        total_steps = resources.candidate_training_steps
        total_gpu_sec = (
            resources.gpu_training_seconds
            + resources.router_evaluation_seconds
            + resources.qualification_seconds
        )
        stored_params = resources.total_stored_adapter_params

        # Calculate bytes if not populated
        if resources.total_stored_adapter_bytes == 0 and stored_params > 0:
            resources.total_stored_adapter_bytes = stored_params * self.BYTES_PER_FP32_PARAM

        # 1. Legacy M72 utility breakdown
        q_pen = self.lambda_query * total_queries
        s_pen = self.lambda_step * total_steps
        gpu_pen = self.lambda_gpu_sec * total_gpu_sec
        stor_pen = self.lambda_storage * stored_params
        fa_pen = self.penalty_false_alloc * false_allocations
        mr_pen = self.penalty_missed_reuse * missed_reuses
        tot_pen = q_pen + s_pen + gpu_pen + stor_pen + fa_pen + mr_pen
        legacy_u = float(macro_f1 - tot_pen)

        breakdown = CostDecompositionBreakdown(
            macro_f1=float(macro_f1),
            query_penalty=float(q_pen),
            step_penalty=float(s_pen),
            gpu_time_penalty=float(gpu_pen),
            storage_penalty=float(stor_pen),
            false_allocation_penalty=float(fa_pen),
            missed_reuse_penalty=float(mr_pen),
            total_cost_penalty=float(tot_pen),
            net_utility=legacy_u,
        )

        # 2. Physical Engineering Cost Model:
        # Measures real physical constraints: GPU training time (sec), inference latency (ms),
        # and physical memory footprint pressure (normalized to MB).
        # Note: 28 KB of storage is 0.028 MB. At standard rate 0.0001 per MB, storage cost is ~0.000003!
        physical_compute_cost = (
            (0.005 * total_gpu_sec)
            + (0.0001 * (resources.inference_latency_ms / 10.0))
            + (0.0001 * (resources.total_stored_adapter_bytes / (1024 * 1024)))
            + (0.05 * false_allocations)
        )
        phys_u = float(macro_f1 - physical_compute_cost)

        # 3. Amortized Physical Utility:
        # Amortizes acquisition cost over observed/expected future reuses K
        k = max(
            1,
            resources.realized_future_reuses
            if resources.realized_future_reuses > 0
            else self.k_episodes,
        )
        if action_name in ("EXPERIMENT_AND_ADAPT", "TRAIN_NEW_SKILL", "ALWAYS_SPAWN"):
            acquisition_cost = 0.005 * total_gpu_sec
            amortized_phys_cost = (
                (acquisition_cost / k)
                + (0.0001 * (resources.inference_latency_ms / 10.0))
                + (0.0001 * (resources.total_stored_adapter_bytes / (1024 * 1024)))
                + (0.05 * false_allocations)
            )
        else:
            amortized_phys_cost = physical_compute_cost
        amort_phys_u = float(macro_f1 - amortized_phys_cost)

        # 4. Sensitivity Analysis
        sensitivity = {
            "compute_heavy": float(
                macro_f1
                - (
                    0.005 * total_queries
                    + 0.001 * total_steps
                    + 0.004 * total_gpu_sec
                    + 0.00001 * stored_params
                )
            ),
            "quality_heavy": float(
                macro_f1
                - (
                    0.0005 * total_queries
                    + 0.0001 * total_steps
                    + 0.0002 * total_gpu_sec
                    + 0.000005 * stored_params
                )
            ),
            "storage_heavy": float(
                macro_f1
                - (
                    0.001 * total_queries
                    + 0.0002 * total_steps
                    + 0.0005 * total_gpu_sec
                    + 0.0001 * stored_params
                )
            ),
            "zero_storage_penalty": float(macro_f1 - (q_pen + s_pen + gpu_pen + fa_pen + mr_pen)),
        }

        return TransparentEpisodeBudgetOutcome(
            policy_name=policy_name,
            task_name=task_name,
            macro_f1=float(macro_f1),
            accuracy=float(accuracy),
            physical_resources=resources,
            legacy_utility_breakdown=breakdown,
            physical_engineering_utility=phys_u,
            amortized_physical_utility=amort_phys_u,
            sensitivity_utilities=sensitivity,
        )
