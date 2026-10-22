"""M37: Independent Research and Source-Grounded Knowledge Acquisition."""

from adaptive_mini_gpt.research.corpus import (
    APPROVED_DOCUMENTS,
    RESEARCH_BENCHMARK_TASKS,
    DocumentType,
    ResearchDocument,
    ResearchTask,
)
from adaptive_mini_gpt.research.evidence_validator import (
    EvidenceValidationResult,
    EvidenceValidator,
)
from adaptive_mini_gpt.research.knowledge_acquisition import (
    AcquisitionMethodResult,
    KnowledgeAcquisitionEvaluator,
)
from adaptive_mini_gpt.research.search_engine import (
    LocalSearchEngine,
    SearchQueryFormulator,
    SearchResult,
)
from adaptive_mini_gpt.research.continual_researcher import (
    ContinualEpisodeReport,
    ContinualResearcher,
)

__all__ = [
    "APPROVED_DOCUMENTS",
    "AcquisitionMethodResult",
    "ContinualEpisodeReport",
    "ContinualResearcher",
    "DocumentType",
    "EvidenceValidationResult",
    "EvidenceValidator",
    "KnowledgeAcquisitionEvaluator",
    "LocalSearchEngine",
    "RESEARCH_BENCHMARK_TASKS",
    "ResearchDocument",
    "ResearchTask",
    "SearchQueryFormulator",
    "SearchResult",
]
