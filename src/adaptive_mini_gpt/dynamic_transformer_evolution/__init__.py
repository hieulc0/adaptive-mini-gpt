"""Milestone M102: Dynamic Transformer Architecture, Worst-Family Forgetting Repair, and Resource-Aware Self-Evolution."""

from adaptive_mini_gpt.dynamic_transformer_evolution.audit_m101 import (
    M101ForensicAuditResult,
    M101ForensicAuditor,
)
from adaptive_mini_gpt.dynamic_transformer_evolution.benchmark_m102 import (
    M102DynamicTransformerBenchmark,
)
from adaptive_mini_gpt.dynamic_transformer_evolution.dynamic_transformer_substrate import (
    DynamicEvolvableBackbone,
    DynamicLayerSkippingRouter,
)
from adaptive_mini_gpt.dynamic_transformer_evolution.future_learning_evaluator_m102 import (
    CausalArmOutcomeM102,
    FutureLearningEvaluatorM102,
    M102CausalEvaluationResult,
)
from adaptive_mini_gpt.dynamic_transformer_evolution.lifelong_coordinator_m102 import (
    EpisodeTelemetryM102,
    LifelongCoordinatorM102,
    LifelongExecutionSummaryM102,
)
from adaptive_mini_gpt.dynamic_transformer_evolution.per_family_rehearsal_buffer import (
    BalancedPerFamilyRehearsalBuffer,
    DomainExemplarBatch,
)
from adaptive_mini_gpt.dynamic_transformer_evolution.persistent_student_m102 import (
    PersistentStudentM102,
)

__all__ = [
    "BalancedPerFamilyRehearsalBuffer",
    "CausalArmOutcomeM102",
    "DomainExemplarBatch",
    "DynamicEvolvableBackbone",
    "DynamicLayerSkippingRouter",
    "EpisodeTelemetryM102",
    "FutureLearningEvaluatorM102",
    "LifelongCoordinatorM102",
    "LifelongExecutionSummaryM102",
    "M101ForensicAuditResult",
    "M101ForensicAuditor",
    "M102CausalEvaluationResult",
    "M102DynamicTransformerBenchmark",
    "PersistentStudentM102",
]
