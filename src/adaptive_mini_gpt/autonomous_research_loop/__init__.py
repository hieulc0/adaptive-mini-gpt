"""M68 Autonomous Research Loop & Verified Self-Improvement Package."""
from __future__ import annotations

from adaptive_mini_gpt.autonomous_research_loop.audit_correction import (
    M67CorrectionAuditReport,
    M67QualificationCorrectionAuditor,
)
from adaptive_mini_gpt.autonomous_research_loop.causal_neural_attribution import (
    CausalNeuralAttributionAuditor,
    CausalNeuralAttributionReport,
)
from adaptive_mini_gpt.autonomous_research_loop.cost_accounting import (
    EpisodeCostLedger,
    RealLearningCostLedger,
)
from adaptive_mini_gpt.autonomous_research_loop.experiment_proposals import (
    ExperimentProposal,
    ExperimentProposalEngine,
    ResearchExperimentAction,
)
from adaptive_mini_gpt.autonomous_research_loop.history_calibration import (
    CompletedExperimentRecord,
    VerifiedExperimentHistoryTracker,
)
from adaptive_mini_gpt.autonomous_research_loop.research_loop import (
    AutonomousResearchLoopEngine,
    ExperimentExecutionResult,
)
from adaptive_mini_gpt.autonomous_research_loop.safeguards import (
    M68SafeguardsAuditor,
    M68SafeguardsReport,
)

__all__ = [
    "M67CorrectionAuditReport",
    "M67QualificationCorrectionAuditor",
    "CausalNeuralAttributionAuditor",
    "CausalNeuralAttributionReport",
    "EpisodeCostLedger",
    "RealLearningCostLedger",
    "ExperimentProposal",
    "ExperimentProposalEngine",
    "ResearchExperimentAction",
    "CompletedExperimentRecord",
    "VerifiedExperimentHistoryTracker",
    "AutonomousResearchLoopEngine",
    "ExperimentExecutionResult",
    "M68SafeguardsAuditor",
    "M68SafeguardsReport",
]
