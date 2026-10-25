"""Adaptive Mini-GPT: Milestone M85 Autonomous Neural Dimension Adaptation & Module Evolution.

Package exports:
- Phase 0: Forensic Audit of M84 (physical parameters vs sparsity, resource accounting).
- Phase 1: Physical Dynamic Dimension Adapter (exact tensor resizing across ranks 2, 4, 8, 16).
- Phase 2: Dynamic Module Registry & Lifecycle Manager (allocate, expand, shrink, retire).
- Phase 3: Learned Architecture Policy Controller (< 1,000 parameters, 989 params).
- Phase 4 & 5: Causal Benchmark and Scientific Qualification Suite.
"""

from adaptive_mini_gpt.dimension_evolution.architecture_policy import (
    ArchitectureAction,
    ArchitectureState,
    LearnedArchitecturePolicy,
)
from adaptive_mini_gpt.dimension_evolution.audit_m84 import (
    M84AuditReport,
    M84ForensicAuditor,
)
from adaptive_mini_gpt.dimension_evolution.benchmark import (
    M85DimensionEvolutionBenchmark,
)
from adaptive_mini_gpt.dimension_evolution.dynamic_adapter import (
    DimensionStats,
    DynamicDimensionAdapter,
)
from adaptive_mini_gpt.dimension_evolution.module_registry import (
    DynamicModuleManager,
    ModuleActionType,
    ModuleLifecycleEvent,
)

__all__ = [
    "ArchitectureAction",
    "ArchitectureState",
    "LearnedArchitecturePolicy",
    "M84AuditReport",
    "M84ForensicAuditor",
    "M85DimensionEvolutionBenchmark",
    "DimensionStats",
    "DynamicDimensionAdapter",
    "DynamicModuleManager",
    "ModuleActionType",
    "ModuleLifecycleEvent",
]
