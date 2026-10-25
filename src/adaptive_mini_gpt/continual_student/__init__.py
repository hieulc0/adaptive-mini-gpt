"""Milestone M86: Integrated Autonomous Continual Learning Package."""

from adaptive_mini_gpt.continual_student.audit_m85 import (
    M85AuditReport,
    M85ForensicAuditor,
)
from adaptive_mini_gpt.continual_student.benchmark import (
    ContinualBenchmarkStrategy,
    M86ContinualLearningBenchmark,
)
from adaptive_mini_gpt.continual_student.continual_learner import (
    UnifiedContinualStudent,
)
from adaptive_mini_gpt.continual_student.learning_policy import (
    ContinualStudentStateSignals,
    StudentAction,
    UnifiedLearningPolicy,
)
from adaptive_mini_gpt.continual_student.student_state import (
    GlobalResourceBudget,
    LearningEpisodeLog,
    PersistentStudentState,
    SkillDescriptor,
)
from adaptive_mini_gpt.continual_student.task_stream import (
    ContinualTaskEpisode,
    ContinualTaskStreamGenerator,
)

__all__ = [
    "M85AuditReport",
    "M85ForensicAuditor",
    "ContinualBenchmarkStrategy",
    "M86ContinualLearningBenchmark",
    "UnifiedContinualStudent",
    "ContinualStudentStateSignals",
    "StudentAction",
    "UnifiedLearningPolicy",
    "GlobalResourceBudget",
    "LearningEpisodeLog",
    "PersistentStudentState",
    "SkillDescriptor",
    "ContinualTaskEpisode",
    "ContinualTaskStreamGenerator",
]
