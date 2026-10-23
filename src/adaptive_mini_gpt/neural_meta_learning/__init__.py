"""Package exports for neural meta-learning of initializations (M62)."""
from __future__ import annotations

from adaptive_mini_gpt.neural_meta_learning.audit import (
    M61MeasurementAuditReport,
    M61MeasurementAuditor,
    compute_horizon_quality_auc,
)
from adaptive_mini_gpt.neural_meta_learning.reptile import (
    MetaTrainingProgressRecord,
    ReptileMetaLearner,
    train_multitask_pretrained_adapter,
)
from adaptive_mini_gpt.neural_meta_learning.evaluator import (
    M62AdaptationEvaluator,
    M62InitializationArm,
    M62TaskAdaptationOutcome,
    SampleEfficiencyPoint,
)
from adaptive_mini_gpt.neural_meta_learning.generalization import (
    CrossFamilyGeneralizationAuditor,
    FamilyGeneralizationReport,
)
from adaptive_mini_gpt.neural_meta_learning.safeguards import (
    M62SafeguardAuditor,
    M62SafeguardReport,
)

__all__ = [
    "M61MeasurementAuditReport",
    "M61MeasurementAuditor",
    "compute_horizon_quality_auc",
    "MetaTrainingProgressRecord",
    "ReptileMetaLearner",
    "train_multitask_pretrained_adapter",
    "M62AdaptationEvaluator",
    "M62InitializationArm",
    "M62TaskAdaptationOutcome",
    "SampleEfficiencyPoint",
    "CrossFamilyGeneralizationAuditor",
    "FamilyGeneralizationReport",
    "M62SafeguardAuditor",
    "M62SafeguardReport",
]
