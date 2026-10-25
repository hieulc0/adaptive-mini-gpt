"""Adaptive Mini-GPT: Milestone M84 Autonomous Neural Connection Pruning and Regrowth.

This package implements dynamic sparse connection topology adaptation:
- Phase 0: Forensic Audit of M83 & Amortized Overhead Analysis.
- Phase 1: Dynamic Sparse Neural Adapters with Autograd Gradient-Salient Regrowth (RigL-style).
- Phase 2: Learned Topology Adaptation Controller (< 1,000 parameters).
- Phase 3 & 4: Causal Training Engine and Multi-Seed Benchmark across 7 Comparative Arms.
- Phase 5: Scientific Qualification Suite, Bit-Exact Rollback, and Base Invariance Verification.
"""

from adaptive_mini_gpt.topology_adaptation.audit_m83 import (
    M83ForensicAuditReport,
    M83ForensicAuditor,
)
from adaptive_mini_gpt.topology_adaptation.benchmark import (
    M84TopologyAdaptationBenchmark,
)
from adaptive_mini_gpt.topology_adaptation.sparse_adapter import (
    DynamicSparseSequenceAdapter,
    PruningCriterion,
    RegrowthCriterion,
    SparseMaskedLinear,
    TopologyStats,
)
from adaptive_mini_gpt.topology_adaptation.sparse_trainer import (
    SparseAdapterTrainer,
    TopologyTrainingStrategy,
    TrainingOutcome,
)
from adaptive_mini_gpt.topology_adaptation.topology_controller import (
    LearnedTopologyController,
    TopologyAction,
    TopologyState,
)

__all__ = [
    "M83ForensicAuditReport",
    "M83ForensicAuditor",
    "M84TopologyAdaptationBenchmark",
    "DynamicSparseSequenceAdapter",
    "PruningCriterion",
    "RegrowthCriterion",
    "SparseMaskedLinear",
    "TopologyStats",
    "SparseAdapterTrainer",
    "TopologyTrainingStrategy",
    "TrainingOutcome",
    "LearnedTopologyController",
    "TopologyAction",
    "TopologyState",
]
