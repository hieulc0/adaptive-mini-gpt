"""Milestone M64: Three-Class Learning and Cross-Family Neural Meta-Generalization."""
from __future__ import annotations

from adaptive_mini_gpt.three_class_meta.audit import (
    M63ForensicAuditReport,
    M63ForensicAuditor,
)
from adaptive_mini_gpt.three_class_meta.broad_meta_learning import (
    M64InitializationArm,
    ThreeClassReptileLearner,
    create_hybrid_init,
    train_three_class_multitask_adapter,
)
from adaptive_mini_gpt.three_class_meta.evaluator import (
    CrossFamilyMetaEvaluator,
    FamilyGeneralizationReport,
)
from adaptive_mini_gpt.three_class_meta.safeguards import (
    M64SafeguardAuditor,
    M64SafeguardReport,
)
from adaptive_mini_gpt.three_class_meta.three_class_engine import (
    ThreeClassAdaptationTrajectory,
    ThreeClassFewShotPoint,
    ThreeClassSupervisionEngine,
    ThreeClassSupervisionMode,
    TriStateMetrics,
    compute_standard_horizon_auc,
)
from adaptive_mini_gpt.three_class_meta.three_class_sampler import (
    PracticeBudgetManifest,
    ThreeClassFewShotSampler,
)

__all__ = [
    "M63ForensicAuditor",
    "M63ForensicAuditReport",
    "ThreeClassFewShotSampler",
    "PracticeBudgetManifest",
    "TriStateMetrics",
    "ThreeClassFewShotPoint",
    "ThreeClassAdaptationTrajectory",
    "ThreeClassSupervisionEngine",
    "ThreeClassSupervisionMode",
    "compute_standard_horizon_auc",
    "M64InitializationArm",
    "ThreeClassReptileLearner",
    "train_three_class_multitask_adapter",
    "create_hybrid_init",
    "CrossFamilyMetaEvaluator",
    "FamilyGeneralizationReport",
    "M64SafeguardAuditor",
    "M64SafeguardReport",
]
