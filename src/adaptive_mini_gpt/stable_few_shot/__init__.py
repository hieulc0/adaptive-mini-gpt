"""Milestone M63: Few-Shot Adaptation Integrity and Stable Neural Learning."""
from __future__ import annotations

from adaptive_mini_gpt.stable_few_shot.ablation import (
    AblationArmSummary,
    DiagnosticAblationReport,
    FewShotAdaptationAblation,
)
from adaptive_mini_gpt.stable_few_shot.audit import (
    M62ForensicAuditReport,
    M62ForensicAuditor,
)
from adaptive_mini_gpt.stable_few_shot.meta_evaluation import (
    BalancedReptileMetaLearner,
    FairMetaInitEvaluator,
    GeneralizationAuditReport,
    M63InitializationArm,
    MetaInitArmMetrics,
    create_minimal_hybrid_init,
    train_balanced_multitask_adapter,
)
from adaptive_mini_gpt.stable_few_shot.safeguards import (
    M63SafeguardAuditor,
    M63SafeguardReport,
)
from adaptive_mini_gpt.stable_few_shot.stable_optimizer import (
    ClassBalancedFewShotSampler,
    FewShotSupervisionObjective,
    StableAdaptationTrajectory,
    StableFewShotEngine,
    StableFewShotPoint,
    ThreeClassMetrics,
    compute_standard_horizon_auc,
)

__all__ = [
    "M62ForensicAuditor",
    "M62ForensicAuditReport",
    "ClassBalancedFewShotSampler",
    "ThreeClassMetrics",
    "StableFewShotPoint",
    "StableAdaptationTrajectory",
    "StableFewShotEngine",
    "FewShotSupervisionObjective",
    "compute_standard_horizon_auc",
    "AblationArmSummary",
    "DiagnosticAblationReport",
    "FewShotAdaptationAblation",
    "M63InitializationArm",
    "MetaInitArmMetrics",
    "GeneralizationAuditReport",
    "BalancedReptileMetaLearner",
    "train_balanced_multitask_adapter",
    "create_minimal_hybrid_init",
    "FairMetaInitEvaluator",
    "M63SafeguardAuditor",
    "M63SafeguardReport",
]
