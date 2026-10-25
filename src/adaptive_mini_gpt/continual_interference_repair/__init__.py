"""Continual-Learning Interference Repair, Stable Neural Knowledge, and Verified Architecture Preservation (M102.1)."""

from adaptive_mini_gpt.continual_interference_repair.audit_m102 import (
    CapacityReconciliationReport,
    M102ForensicAuditResult,
    M102ForensicAuditor,
    ParameterAuditResult,
    SeedVarianceAnalysis,
    analyze_m102_seed_variance,
    reconcile_m102_parameters,
)
from adaptive_mini_gpt.continual_interference_repair.benchmark_m102_1 import (
    M102_1InterferenceRepairBenchmark,
)
from adaptive_mini_gpt.continual_interference_repair.causal_comparator_m102_1 import (
    CausalArmResultM102_1,
    CausalComparatorM102_1,
    M102_1CausalComparisonResult,
)
from adaptive_mini_gpt.continual_interference_repair.interference_controllers import (
    OrthogonalGradientProjector,
    TaskConditionedHeadManager,
)
from adaptive_mini_gpt.continual_interference_repair.lifelong_coordinator_m102_1 import (
    EpisodeTelemetryM102_1,
    LifelongCoordinatorM102_1,
    LifelongExecutionSummaryM102_1,
)
from adaptive_mini_gpt.continual_interference_repair.persistent_student_m102_1 import (
    PersistentStudentM102_1,
)
from adaptive_mini_gpt.continual_interference_repair.rehearsal_buffer_m102_1 import (
    BalancedRehearsalBufferM102_1,
    FamilyRetentionReportM102_1,
    RehearsalBufferAuditSummaryM102_1,
)

__all__ = [
    "reconcile_m102_parameters",
    "ParameterAuditResult",
    "CapacityReconciliationReport",
    "analyze_m102_seed_variance",
    "SeedVarianceAnalysis",
    "M102ForensicAuditResult",
    "M102ForensicAuditor",
    "TaskConditionedHeadManager",
    "OrthogonalGradientProjector",
    "BalancedRehearsalBufferM102_1",
    "FamilyRetentionReportM102_1",
    "RehearsalBufferAuditSummaryM102_1",
    "PersistentStudentM102_1",
    "LifelongCoordinatorM102_1",
    "EpisodeTelemetryM102_1",
    "LifelongExecutionSummaryM102_1",
    "CausalComparatorM102_1",
    "CausalArmResultM102_1",
    "M102_1CausalComparisonResult",
    "M102_1InterferenceRepairBenchmark",
]
