"""M53: Causal Learning Audit and Transferable Skill Acquisition."""
from __future__ import annotations

from adaptive_mini_gpt.causal_learning_audit.adapter import PersistentCandidateAdapter
from adaptive_mini_gpt.causal_learning_audit.audit import (
    M52AuditReport,
    M52ScientificAuditor,
)
from adaptive_mini_gpt.causal_learning_audit.environment import (
    CausalAuditEnvironmentSuite,
    build_m53_environment_suite,
)
from adaptive_mini_gpt.causal_learning_audit.epistemic_audit import (
    EpistemicAuditEngine,
    EpistemicAuditReport,
    EpistemicRegimeResult,
)
from adaptive_mini_gpt.causal_learning_audit.factorial import (
    FactorialAnalysisReport,
    FactorialCellResult,
    FactorialStudyEngine,
)
from adaptive_mini_gpt.causal_learning_audit.transfer import (
    SkillTransferEngine,
    TransferConditionResult,
    TransferEvaluationReport,
)

__all__ = [
    "PersistentCandidateAdapter",
    "M52AuditReport",
    "M52ScientificAuditor",
    "CausalAuditEnvironmentSuite",
    "build_m53_environment_suite",
    "FactorialCellResult",
    "FactorialAnalysisReport",
    "FactorialStudyEngine",
    "PersistentCandidateAdapter",
    "SkillTransferEngine",
    "TransferConditionResult",
    "TransferEvaluationReport",
    "EpistemicRegimeResult",
    "EpistemicAuditReport",
    "EpistemicAuditEngine",
]
