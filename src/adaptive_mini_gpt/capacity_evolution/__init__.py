"""Adaptive Mini-GPT Milestone M57: Controlled Neural Capacity Evolution."""
from adaptive_mini_gpt.capacity_evolution.audit import (
    M56AuditSummary,
    M56ScientificAuditor,
)
from adaptive_mini_gpt.capacity_evolution.capacity_controller import (
    AdaptiveCapacityController,
    CapacityAction,
    ControllerDecision,
)
from adaptive_mini_gpt.capacity_evolution.epistemic_diagnostic import (
    CalibratedSPRTDiagnosticAuditor,
    EpistemicRegimeDiagnostic,
)
from adaptive_mini_gpt.capacity_evolution.evolution_engine import (
    AdaptationStepMetrics,
    ArmEvolutionEvaluation,
    ControlledCapacityEvolutionEngine,
    MultiTaskEvolutionBenchmarkSummary,
)
from adaptive_mini_gpt.capacity_evolution.evolvable_adapter import (
    EvolvableResidualSkillAdapter,
)
from adaptive_mini_gpt.capacity_evolution.retention_rollback import (
    CapacityRetentionAuditor,
    CapacityRetentionResult,
)
from adaptive_mini_gpt.capacity_evolution.task_complexity import (
    ObservableTaskComplexity,
    compute_observable_complexity,
    create_balanced_hierarchy_environment,
    create_deep_chain_environment,
    create_dense_cluster_exceptions_environment,
    create_incompatible_parity_environment,
    create_wide_branching_tree_environment,
)

__all__ = [
    "AdaptationStepMetrics",
    "AdaptiveCapacityController",
    "ArmEvolutionEvaluation",
    "CalibratedSPRTDiagnosticAuditor",
    "CapacityAction",
    "CapacityRetentionAuditor",
    "CapacityRetentionResult",
    "ControlledCapacityEvolutionEngine",
    "ControllerDecision",
    "EpistemicRegimeDiagnostic",
    "EvolvableResidualSkillAdapter",
    "M56AuditSummary",
    "M56ScientificAuditor",
    "MultiTaskEvolutionBenchmarkSummary",
    "ObservableTaskComplexity",
    "compute_observable_complexity",
    "create_balanced_hierarchy_environment",
    "create_deep_chain_environment",
    "create_dense_cluster_exceptions_environment",
    "create_incompatible_parity_environment",
    "create_wide_branching_tree_environment",
]
