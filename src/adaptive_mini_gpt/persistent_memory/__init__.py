"""Milestone M96: Persistent Factual Memory & Long-Horizon Learning."""

from adaptive_mini_gpt.persistent_memory.audit_m95 import (
    M95AuditResult,
    M95ForensicAuditor,
)
from adaptive_mini_gpt.persistent_memory.benchmark_m96 import (
    M96BenchmarkReport,
    M96PersistentMemoryBenchmark,
    SeedM96Result,
)
from adaptive_mini_gpt.persistent_memory.integrated_student import (
    IntegratedAdaptiveStudent,
    StudentInferenceResult,
)
from adaptive_mini_gpt.persistent_memory.knowledge_store import (
    CorrectionRecord,
    EpisodicRecord,
    FactualRecord,
    PersistentKnowledgeStore,
    SkillReference,
)
from adaptive_mini_gpt.persistent_memory.long_horizon_curriculum import (
    EpisodeType,
    LongHorizonCurriculumGenerator,
    LongHorizonEpisode,
)
from adaptive_mini_gpt.persistent_memory.selective_memory_manager import (
    MemoryAction,
    MemoryDecision,
    SelectiveMemoryManager,
)

__all__ = [
    "M95AuditResult",
    "M95ForensicAuditor",
    "M96BenchmarkReport",
    "M96PersistentMemoryBenchmark",
    "SeedM96Result",
    "IntegratedAdaptiveStudent",
    "StudentInferenceResult",
    "CorrectionRecord",
    "EpisodicRecord",
    "FactualRecord",
    "PersistentKnowledgeStore",
    "SkillReference",
    "EpisodeType",
    "LongHorizonCurriculumGenerator",
    "LongHorizonEpisode",
    "MemoryAction",
    "MemoryDecision",
    "SelectiveMemoryManager",
]
