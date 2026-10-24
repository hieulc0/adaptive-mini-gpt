"""Milestone M81: Learning-to-Learn Through Transfer and Experience."""

from adaptive_mini_gpt.learning_to_learn.audit_m80 import (
    M80AuditReport,
    M80ForensicAuditor,
)
from adaptive_mini_gpt.learning_to_learn.benchmark import (
    M81LearningToLearnBenchmark,
)
from adaptive_mini_gpt.learning_to_learn.capacity_adaptation import (
    BottleneckSequenceAdapter,
    CapacityAdaptationManager,
    CapacityDecision,
    CapacityTrialResult,
)
from adaptive_mini_gpt.learning_to_learn.composition_evaluator import (
    CompositionAblationReport,
    CompositionEvaluator,
)
from adaptive_mini_gpt.learning_to_learn.learning_to_learn_engine import (
    LearningArmTrialResult,
    LearningArmType,
    LearningToLearnEngine,
    TrajectoryCheckpoint,
)
from adaptive_mini_gpt.learning_to_learn.retention_auditor import (
    RetentionAuditReport,
    RetentionAuditor,
)
from adaptive_mini_gpt.learning_to_learn.statistics import (
    PairedTTestResult,
    compute_paired_t_test,
    student_t_two_tailed_p_value,
)
from adaptive_mini_gpt.learning_to_learn.task_curriculum import (
    CurriculumExample,
    CurriculumFamily,
    CurriculumTaskDataset,
    CurriculumTaskGenerator,
    apply_curriculum_rule,
)

__all__ = [
    "BottleneckSequenceAdapter",
    "CapacityAdaptationManager",
    "CapacityDecision",
    "CapacityTrialResult",
    "CompositionAblationReport",
    "CompositionEvaluator",
    "CurriculumExample",
    "CurriculumFamily",
    "CurriculumTaskDataset",
    "CurriculumTaskGenerator",
    "LearningArmTrialResult",
    "LearningArmType",
    "LearningToLearnEngine",
    "M80AuditReport",
    "M80ForensicAuditor",
    "M81LearningToLearnBenchmark",
    "PairedTTestResult",
    "RetentionAuditReport",
    "RetentionAuditor",
    "TrajectoryCheckpoint",
    "apply_curriculum_rule",
    "compute_paired_t_test",
    "student_t_two_tailed_p_value",
]
