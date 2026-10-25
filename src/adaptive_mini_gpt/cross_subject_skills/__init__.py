"""Cross-Subject Skill Acquisition, Composition, and Continual Learning for Milestone M98."""

from adaptive_mini_gpt.cross_subject_skills.audit_m97 import M97ForensicAuditor, M97AuditResult
from adaptive_mini_gpt.cross_subject_skills.subject_curriculum import (
    Subject,
    CrossSubjectTask,
    CrossSubjectCurriculumGenerator,
)
from adaptive_mini_gpt.cross_subject_skills.cross_subject_substrate import CrossSubjectReasoningSubstrate
from adaptive_mini_gpt.cross_subject_skills.autonomous_skill_learner import (
    AutonomousSkillLearner,
    StudentType,
    AdaptationOutcome,
)
from adaptive_mini_gpt.cross_subject_skills.persistent_student_m98 import PersistentAdaptiveStudentM98
from adaptive_mini_gpt.cross_subject_skills.benchmark_m98 import M98CrossSubjectBenchmark, M98BenchmarkReport

__all__ = [
    "M97ForensicAuditor",
    "M97AuditResult",
    "Subject",
    "CrossSubjectTask",
    "CrossSubjectCurriculumGenerator",
    "CrossSubjectReasoningSubstrate",
    "AutonomousSkillLearner",
    "StudentType",
    "AdaptationOutcome",
    "PersistentAdaptiveStudentM98",
    "M98CrossSubjectBenchmark",
    "M98BenchmarkReport",
]
