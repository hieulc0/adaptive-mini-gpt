"""Milestone M54: Verified Skill Reuse and Adaptive Transfer Package."""
from __future__ import annotations

from adaptive_mini_gpt.skill_reuse.adaptive_transfer import (
    AdaptiveTransferEngine,
    ArmTransferTrajectory,
    BudgetEvaluationStep,
    MultiArmTransferReport,
)
from adaptive_mini_gpt.skill_reuse.audit import (
    HashLineageReconciliation,
    M53ScientificAuditReport,
    M53ScientificAuditor,
    RetentionMetricConvention,
)
from adaptive_mini_gpt.skill_reuse.compatibility import (
    NeuralSkillSelectorHead,
    SelectionDecision,
    SkillCompatibilityEvaluator,
    SkillCompatibilityScore,
    TargetObservation,
)
from adaptive_mini_gpt.skill_reuse.environment import (
    OUT_OF_HYPOTHESIS_BIPARTITE,
    OUT_OF_HYPOTHESIS_DENSE_CLIQUE,
    SkillBenchmarkEnvironments,
    build_m54_environment_suite,
)
from adaptive_mini_gpt.skill_reuse.epistemic_generalization import (
    EpistemicGeneralizationAuditor,
    EpistemicGeneralizationReport,
    MechanismDiagnosticResult,
)
from adaptive_mini_gpt.skill_reuse.registry import (
    RegisteredSkill,
    SkillRepertoireRegistry,
    compute_adapter_hash,
)
from adaptive_mini_gpt.skill_reuse.retention import (
    ContinualRetentionReport,
    RetentionEvaluationEngine,
    TaskRetentionMatrix,
)

__all__ = [
    "HashLineageReconciliation",
    "RetentionMetricConvention",
    "M53ScientificAuditReport",
    "M53ScientificAuditor",
    "RegisteredSkill",
    "SkillRepertoireRegistry",
    "compute_adapter_hash",
    "OUT_OF_HYPOTHESIS_BIPARTITE",
    "OUT_OF_HYPOTHESIS_DENSE_CLIQUE",
    "SkillBenchmarkEnvironments",
    "build_m54_environment_suite",
    "TargetObservation",
    "SkillCompatibilityScore",
    "SelectionDecision",
    "NeuralSkillSelectorHead",
    "SkillCompatibilityEvaluator",
    "BudgetEvaluationStep",
    "ArmTransferTrajectory",
    "MultiArmTransferReport",
    "AdaptiveTransferEngine",
    "TaskRetentionMatrix",
    "ContinualRetentionReport",
    "RetentionEvaluationEngine",
    "MechanismDiagnosticResult",
    "EpistemicGeneralizationReport",
    "EpistemicGeneralizationAuditor",
]
