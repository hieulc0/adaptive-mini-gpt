"""Continual-Learning Interference Repair, Stable Neural Knowledge, and Verified Architecture Preservation (M102.1)."""

from adaptive_mini_gpt.continual_interference_repair.audit_m102 import (
    ParameterAuditResult,
    SeedVarianceAnalysis,
    analyze_m102_seed_variance,
    reconcile_m102_parameters,
)
from adaptive_mini_gpt.continual_interference_repair.benchmark_m102_1 import (
    M102_1InterferenceRepairBenchmark,
)
from adaptive_mini_gpt.continual_interference_repair.causal_comparator_m102_1 import (
    CausalComparatorM102_1,
    InterferenceAblationArm,
)
from adaptive_mini_gpt.continual_interference_repair.interference_controllers import (
    DomainRouter,
    OrthogonalGradientProjector,
    TaskConditionedHeadManager,
)
from adaptive_mini_gpt.continual_interference_repair.lifelong_coordinator_m102_1 import (
    EpisodeRecord,
    LifelongCensusResult,
    LifelongCoordinatorM102_1,
)
from adaptive_mini_gpt.continual_interference_repair.persistent_student_m102_1 import (
    M102_1PersistentStudent,
    PersistentStudentConfig,
)
from adaptive_mini_gpt.continual_interference_repair.rehearsal_buffer_m102_1 import (
    Exemplar,
    FullBatchBalancedRehearsalBuffer,
)

__all__ = [
    "reconcile_m102_parameters",
    "ParameterAuditResult",
    "analyze_m102_seed_variance",
    "SeedVarianceAnalysis",
    "DomainRouter",
    "TaskConditionedHeadManager",
    "OrthogonalGradientProjector",
    "Exemplar",
    "FullBatchBalancedRehearsalBuffer",
    "M102_1PersistentStudent",
    "PersistentStudentConfig",
    "LifelongCoordinatorM102_1",
    "LifelongCensusResult",
    "EpisodeRecord",
    "CausalComparatorM102_1",
    "InterferenceAblationArm",
    "M102_1InterferenceRepairBenchmark",
]
