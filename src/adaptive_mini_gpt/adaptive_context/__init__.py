"""Milestone M97: Adaptive Context Length, Working Memory, and Long-Context Reasoning."""

from __future__ import annotations

from adaptive_mini_gpt.adaptive_context.audit_m96 import M96ForensicAuditor, M96AuditResult
from adaptive_mini_gpt.adaptive_context.context_extension import (
    ContextExtensionEngine,
    ExtendedContextConfig,
    NeedleEvaluationResult,
)
from adaptive_mini_gpt.adaptive_context.working_context_builder import (
    WorkingContextBuilder,
    WorkingContextDocument,
)
from adaptive_mini_gpt.adaptive_context.adaptive_context_manager import (
    AdaptiveContextPolicy,
    ContextMode,
    ContextDecision,
)
from adaptive_mini_gpt.adaptive_context.persistent_student_m97 import (
    PersistentAdaptiveStudentM97,
    StudentFullCheckpoint,
)
from adaptive_mini_gpt.adaptive_context.long_context_curriculum import (
    LongContextTask,
    LongContextCurriculumGenerator,
    TaskType,
)
from adaptive_mini_gpt.adaptive_context.benchmark_m97 import (
    M97AdaptiveContextBenchmark,
    M97BenchmarkReport,
    SeedM97Result,
)

__all__ = [
    "M96ForensicAuditor",
    "M96AuditResult",
    "ContextExtensionEngine",
    "ExtendedContextConfig",
    "NeedleEvaluationResult",
    "WorkingContextBuilder",
    "WorkingContextDocument",
    "AdaptiveContextPolicy",
    "ContextMode",
    "ContextDecision",
    "PersistentAdaptiveStudentM97",
    "StudentFullCheckpoint",
    "LongContextTask",
    "LongContextCurriculumGenerator",
    "TaskType",
    "M97AdaptiveContextBenchmark",
    "M97BenchmarkReport",
    "SeedM97Result",
]
