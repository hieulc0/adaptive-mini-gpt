"""M52: Learning-Outcome-Driven Experiment Selection Package."""
from __future__ import annotations

from adaptive_mini_gpt.learning_driven_experiment.audit import (
    M51AuditReport,
    M51ScientificAuditor,
)
from adaptive_mini_gpt.learning_driven_experiment.epistemic_tracker import (
    EpistemicState,
    NormalizedEpistemicTracker,
)
from adaptive_mini_gpt.learning_driven_experiment.environment import (
    SegregatedEnvironmentSuite,
    build_m52_environment_suite,
)
from adaptive_mini_gpt.learning_driven_experiment.evaluator import (
    LearningBenefitReward,
    LearningOutcomeEvaluator,
)
from adaptive_mini_gpt.learning_driven_experiment.minimal_pairs import (
    MinimalPairEngine,
    MinimalPairEvaluationResult,
)
from adaptive_mini_gpt.learning_driven_experiment.policy import (
    LearningDrivenArmsRunner,
    LearningOutcomePolicyHead,
    train_learning_outcome_policy,
)

__all__ = [
    "M51AuditReport",
    "M51ScientificAuditor",
    "EpistemicState",
    "NormalizedEpistemicTracker",
    "SegregatedEnvironmentSuite",
    "build_m52_environment_suite",
    "LearningBenefitReward",
    "LearningOutcomeEvaluator",
    "MinimalPairEngine",
    "MinimalPairEvaluationResult",
    "LearningDrivenArmsRunner",
    "LearningOutcomePolicyHead",
    "train_learning_outcome_policy",
]
