"""Adaptive Mini-GPT: M44 Structural Analogy, Memory Transfer, and Learning Diagnostics."""
from adaptive_mini_gpt.analogy.audit import (
    BudgetConfusionMatrix,
    M43DiagnosticAuditor,
    M43FlatCurveAuditReport,
    PositiveControlResult,
)
from adaptive_mini_gpt.analogy.positive_control import (
    MemoryPositiveControlEvaluator,
    PositiveControlSuiteReport,
    ScenarioRetrievalMetric,
)
from adaptive_mini_gpt.analogy.structural_schema import (
    CanonicalGraphSignature,
    DeterministicStructuralMatcher,
    RelationProperties,
    StructuralAnalogyExperience,
)
from adaptive_mini_gpt.analogy.analogy_experiments import (
    AnalogyApproachReport,
    AnalogyLearningCurveStep,
    StructuralAnalogyExperimentRunner,
    build_analogy_task_family,
    get_m44_benchmark_families,
)
from adaptive_mini_gpt.analogy.sequential_retention import (
    ContinualStrategyReport,
    SequentialRetentionEvaluator,
    SequentialRetentionSuiteReport,
)
from adaptive_mini_gpt.analogy.language_diagnostic import (
    FoundationComparisonReport,
    LanguageDiagnosticEvaluator,
    LanguageDiagnosticSuiteReport,
)

__all__ = [
    "BudgetConfusionMatrix",
    "M43DiagnosticAuditor",
    "M43FlatCurveAuditReport",
    "PositiveControlResult",
    "CanonicalGraphSignature",
    "DeterministicStructuralMatcher",
    "RelationProperties",
    "StructuralAnalogyExperience",
    "MemoryPositiveControlEvaluator",
    "PositiveControlSuiteReport",
    "ScenarioRetrievalMetric",
    "AnalogyApproachReport",
    "AnalogyLearningCurveStep",
    "StructuralAnalogyExperimentRunner",
    "build_analogy_task_family",
    "get_m44_benchmark_families",
    "ContinualStrategyReport",
    "SequentialRetentionEvaluator",
    "SequentialRetentionSuiteReport",
    "FoundationComparisonReport",
    "LanguageDiagnosticEvaluator",
    "LanguageDiagnosticSuiteReport",
]
