"""M42 Meta-Learning and Sample Efficiency package."""
from adaptive_mini_gpt.meta.audit import M41Auditor, M41AuditReport
from adaptive_mini_gpt.meta.benchmark_suite import (
    MetaFamilyDataset,
    MetaGraphOracle,
    MetaTask,
    get_benchmark_families,
)
from adaptive_mini_gpt.meta.learning_policy import (
    PolicyDecision,
    PolicyComparisonReport,
    SelfDirectedLearningPolicy,
)
from adaptive_mini_gpt.meta.meta_learner import (
    CumulativeRetentionReport,
    EfficiencyCurvePoint,
    MetaLearner,
    StrategyEfficiencyReport,
)
from adaptive_mini_gpt.meta.tri_state_diagnostic import (
    DiagnosticMethodResult,
    TriStateDiagnostic,
)

__all__ = [
    "M41Auditor",
    "M41AuditReport",
    "MetaFamilyDataset",
    "MetaGraphOracle",
    "MetaTask",
    "get_benchmark_families",
    "PolicyDecision",
    "PolicyComparisonReport",
    "SelfDirectedLearningPolicy",
    "CumulativeRetentionReport",
    "EfficiencyCurvePoint",
    "MetaLearner",
    "StrategyEfficiencyReport",
    "DiagnosticMethodResult",
    "TriStateDiagnostic",
]
