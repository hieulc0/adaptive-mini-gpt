"""M39 Knowledge Integration and Generalization Package."""

from adaptive_mini_gpt.integration.benchmark_suite import (
    FACTUAL_LEARNING_DATASET,
    PROCEDURAL_LOGIC_TASKS,
    SEQUENTIAL_INTEGRATION_EPISODES,
    IntegrationTask,
)
from adaptive_mini_gpt.integration.carry_audit import (
    CarryAuditResult,
    CarryRetentionAuditor,
)
from adaptive_mini_gpt.integration.evaluator import (
    MultiEpisodeRetentionReport,
    SequentialEpisodeSnapshot,
    SequentialMultiEpisodeEvaluator,
)
from adaptive_mini_gpt.integration.knowledge_integrator import (
    ArmResult,
    KnowledgeIntegrator,
)
from adaptive_mini_gpt.integration.learned_router import (
    MetacognitiveRoutingSystem,
    NeuralTaskRouter,
    RoutingEvaluationResult,
)

__all__ = [
    "CarryAuditResult",
    "CarryRetentionAuditor",
    "FACTUAL_LEARNING_DATASET",
    "PROCEDURAL_LOGIC_TASKS",
    "SEQUENTIAL_INTEGRATION_EPISODES",
    "IntegrationTask",
    "ArmResult",
    "KnowledgeIntegrator",
    "NeuralTaskRouter",
    "MetacognitiveRoutingSystem",
    "RoutingEvaluationResult",
    "SequentialEpisodeSnapshot",
    "MultiEpisodeRetentionReport",
    "SequentialMultiEpisodeEvaluator",
]
