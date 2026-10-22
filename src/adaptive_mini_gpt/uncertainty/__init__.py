"""Adaptive Mini-GPT: M45 Epistemic Uncertainty and Transferable Learning."""
from adaptive_mini_gpt.uncertainty.audit import (
    AttributionSourceBreakdown,
    M44AttributionAuditor,
    M44AuditPerSeedResult,
    M44ForensicAuditReport,
)
from adaptive_mini_gpt.uncertainty.epistemic_benchmark import (
    EpistemicTask,
    EpistemicWorldDataset,
    RelationSemantics,
    build_epistemic_world,
    get_m45_epistemic_worlds,
)
from adaptive_mini_gpt.uncertainty.tri_state_classifier import (
    EpistemicEvaluationMetrics,
    EpistemicUncertaintyEvaluator,
    TriStateConfusionMatrix,
    TriStateEpistemicHead,
)
from adaptive_mini_gpt.uncertainty.learned_transfer import (
    LearnedStructuralTransferEvaluator,
    LearnedTransferReport,
    NeuralSchemaPredictorHead,
    SchemaAblationResult,
)
from adaptive_mini_gpt.uncertainty.corrective_learning import (
    CorrectiveEpisodeReport,
    CorrectiveLearningComparisonReport,
    CorrectiveSelfLearningRunner,
)
from adaptive_mini_gpt.uncertainty.learning_efficiency import (
    EpistemicCurveStep,
    EpistemicEfficiencyBenchmarkRunner,
    EpistemicEfficiencyReport,
)

__all__ = [
    "AttributionSourceBreakdown",
    "M44AttributionAuditor",
    "M44AuditPerSeedResult",
    "M44ForensicAuditReport",
    "EpistemicTask",
    "EpistemicWorldDataset",
    "RelationSemantics",
    "build_epistemic_world",
    "get_m45_epistemic_worlds",
    "EpistemicEvaluationMetrics",
    "EpistemicUncertaintyEvaluator",
    "TriStateConfusionMatrix",
    "TriStateEpistemicHead",
    "LearnedStructuralTransferEvaluator",
    "LearnedTransferReport",
    "NeuralSchemaPredictorHead",
    "SchemaAblationResult",
    "CorrectiveEpisodeReport",
    "CorrectiveLearningComparisonReport",
    "CorrectiveSelfLearningRunner",
    "EpistemicCurveStep",
    "EpistemicEfficiencyBenchmarkRunner",
    "EpistemicEfficiencyReport",
]
