"""M35 Metacognition Package: Action Space, Policy Head, Curriculum, Controller, and Language Diagnosis."""

from adaptive_mini_gpt.metacognition.action_space import (
    ACTION_COMPUTE_COSTS,
    ActionOutcomeRecord,
    LearningAction,
    compute_action_utility,
)
from adaptive_mini_gpt.metacognition.curriculum import (
    MetacognitiveSample,
    build_metacognitive_curriculum,
)
from adaptive_mini_gpt.metacognition.language_diagnosis import (
    LanguageDiagnosticRecord,
    diagnose_language_representation_gap,
)
from adaptive_mini_gpt.metacognition.learning_controller import (
    ControlledSelfDirectedLearner,
    EpisodeExecutionResult,
)
from adaptive_mini_gpt.metacognition.policy_head import (
    ACTION_LIST,
    ACTION_TO_IDX,
    IDX_TO_ACTION,
    NeuralMetacognitivePolicy,
    train_outcome_aware_policy,
    train_supervised_policy,
)

__all__ = [
    "LearningAction",
    "ActionOutcomeRecord",
    "compute_action_utility",
    "ACTION_COMPUTE_COSTS",
    "MetacognitiveSample",
    "build_metacognitive_curriculum",
    "NeuralMetacognitivePolicy",
    "train_supervised_policy",
    "train_outcome_aware_policy",
    "ACTION_LIST",
    "ACTION_TO_IDX",
    "IDX_TO_ACTION",
    "ControlledSelfDirectedLearner",
    "EpisodeExecutionResult",
    "LanguageDiagnosticRecord",
    "diagnose_language_representation_gap",
]
