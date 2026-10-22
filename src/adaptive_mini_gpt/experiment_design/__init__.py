"""M50: Autonomous Experiment Design and Verified Rule Learning package."""
from __future__ import annotations

from adaptive_mini_gpt.experiment_design.audit import M49AuditReport, M49ForensicAuditor
from adaptive_mini_gpt.experiment_design.expanded_hypothesis import (
    EXPANDED_HYPOTHESES,
    UNKNOWN_RULE,
    AutonomousEnvironment,
    EnvironmentQuery,
    MinimalPairControl,
    RuleHypothesis,
    build_m50_environment_suite,
)
from adaptive_mini_gpt.experiment_design.syntax_validator import (
    ExperimentSyntaxValidator,
    ExperimentValidatorStats,
    ValidationResult,
)
from adaptive_mini_gpt.experiment_design.hypothesis_revision import (
    HypothesisRevisionReport,
    HypothesisRevisionTracker,
    RevisionStep,
)
from adaptive_mini_gpt.experiment_design.experiment_generator import (
    AutonomousExperimentGenerator,
    ExperimentGenerationRunner,
    GeneratedExperimentResult,
)
from adaptive_mini_gpt.experiment_design.learning_evaluator import (
    AutonomousLearningCurveResult,
    AutonomousObservationLearner,
    BudgetEvaluationResult,
)
from adaptive_mini_gpt.experiment_design.rule_memory import (
    ControlledRuleMemory,
    RuleMemoryComparisonResult,
    VerifiedRuleEntry,
)
from adaptive_mini_gpt.experiment_design.continual_study import (
    ContinualAutonomousEvaluator,
    ContinualEpisodeResult,
)

__all__ = [
    "M49AuditReport",
    "M49ForensicAuditor",
    "EXPANDED_HYPOTHESES",
    "UNKNOWN_RULE",
    "AutonomousEnvironment",
    "EnvironmentQuery",
    "MinimalPairControl",
    "RuleHypothesis",
    "build_m50_environment_suite",
    "ExperimentSyntaxValidator",
    "ExperimentValidatorStats",
    "ValidationResult",
    "HypothesisRevisionReport",
    "HypothesisRevisionTracker",
    "RevisionStep",
    "AutonomousExperimentGenerator",
    "ExperimentGenerationRunner",
    "GeneratedExperimentResult",
    "AutonomousLearningCurveResult",
    "AutonomousObservationLearner",
    "BudgetEvaluationResult",
    "ControlledRuleMemory",
    "RuleMemoryComparisonResult",
    "VerifiedRuleEntry",
    "ContinualAutonomousEvaluator",
    "ContinualEpisodeResult",
]
