"""Milestone M83: Autonomous Neural Knowledge Consolidation and Self-Distillation Package."""

from adaptive_mini_gpt.knowledge_consolidation.audit_m82 import (
    M82ForensicAuditReport,
    M82ForensicAuditor,
)
from adaptive_mini_gpt.knowledge_consolidation.benchmark import (
    M83KnowledgeConsolidationBenchmark,
)
from adaptive_mini_gpt.knowledge_consolidation.candidate_analyzer import (
    ConsolidationCandidateAnalyzer,
    SkillPairCompatibility,
)
from adaptive_mini_gpt.knowledge_consolidation.consolidation_policy import (
    ConsolidationAction,
    ConsolidationState,
    LearnedConsolidationController,
)
from adaptive_mini_gpt.knowledge_consolidation.consolidated_adapter import (
    ConsolidatedSharedAdapter,
    SkillResidualHead,
)
from adaptive_mini_gpt.knowledge_consolidation.distillation_engine import (
    CompressionMethod,
    DistillationResult,
    NeuralSelfDistillationEngine,
)

__all__ = [
    "M82ForensicAuditReport",
    "M82ForensicAuditor",
    "M83KnowledgeConsolidationBenchmark",
    "ConsolidationCandidateAnalyzer",
    "SkillPairCompatibility",
    "ConsolidationAction",
    "ConsolidationState",
    "LearnedConsolidationController",
    "ConsolidatedSharedAdapter",
    "SkillResidualHead",
    "CompressionMethod",
    "DistillationResult",
    "NeuralSelfDistillationEngine",
]
