"""Adaptive Mini-GPT: M51 Guided Experiment Design and Adaptive Learning Package."""

from adaptive_mini_gpt.guided_experiment.audit import (
    M50AuditReport,
    M50ForensicAuditor,
)
from adaptive_mini_gpt.guided_experiment.environment import (
    AutonomousEnvironment,
    EnvironmentQuery,
    EXPANDED_HYPOTHESES,
    MinimalPairControl,
    OUT_OF_HYPOTHESIS_PARITY,
    RuleHypothesis,
    UNKNOWN_RULE,
    build_m51_environment_suite,
)
from adaptive_mini_gpt.guided_experiment.structural_probe_generator import (
    ProbeGenerationStats,
    StructuralProbe,
    StructuralProbeGenerator,
)
from adaptive_mini_gpt.guided_experiment.hypothesis_tracker import (
    GuidedHypothesisTracker,
)
from adaptive_mini_gpt.guided_experiment.ranking_policy import (
    ArmExecutionResult,
    GuidedArmsRunner,
    NeuralProbeRankingHead,
    train_neural_ranking_head,
)
from adaptive_mini_gpt.guided_experiment.learning_evaluator import (
    BudgetEvaluationResult,
    CandidateAdapterHead,
    GuidedLearningCurveResult,
    GuidedObservationLearner,
)
from adaptive_mini_gpt.guided_experiment.rule_memory import (
    CompatibilityReport,
    CompatibleRuleMemory,
    RuleMemoryComparisonResult,
    SemanticCompatibilityChecker,
    VerifiedRuleEntry,
)
from adaptive_mini_gpt.guided_experiment.continual_study import (
    ContinualGuidedEpisodeResult,
    ContinualGuidedEvaluator,
)

__all__ = [
    "M50AuditReport",
    "M50ForensicAuditor",
    "AutonomousEnvironment",
    "EnvironmentQuery",
    "EXPANDED_HYPOTHESES",
    "MinimalPairControl",
    "OUT_OF_HYPOTHESIS_PARITY",
    "RuleHypothesis",
    "UNKNOWN_RULE",
    "build_m51_environment_suite",
    "ProbeGenerationStats",
    "StructuralProbe",
    "StructuralProbeGenerator",
    "GuidedHypothesisTracker",
    "ArmExecutionResult",
    "GuidedArmsRunner",
    "NeuralProbeRankingHead",
    "train_neural_ranking_head",
    "BudgetEvaluationResult",
    "CandidateAdapterHead",
    "GuidedLearningCurveResult",
    "GuidedObservationLearner",
    "CompatibilityReport",
    "CompatibleRuleMemory",
    "RuleMemoryComparisonResult",
    "SemanticCompatibilityChecker",
    "VerifiedRuleEntry",
    "ContinualGuidedEpisodeResult",
    "ContinualGuidedEvaluator",
]
