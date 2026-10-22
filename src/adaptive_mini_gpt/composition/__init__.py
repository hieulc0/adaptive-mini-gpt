"""M40 Compositional Knowledge, Transfer, and Foundation Readiness Package."""

from adaptive_mini_gpt.composition.audit import M39AuditRecord, M39BaselineAuditor
from adaptive_mini_gpt.composition.benchmark_suite import (
    M40_COMPOSITIONAL_BENCHMARK,
    M40_CROSS_MODULE_BENCHMARK,
    SEQUENTIAL_RELATIONAL_EPISODES,
    CompositionalTask,
)
from adaptive_mini_gpt.composition.composition_trainer import (
    CompositionTrainer,
    CompositionalArmResult,
)
from adaptive_mini_gpt.composition.evaluator import (
    SequentialRelationalEvaluator,
    SequentialRelationalReport,
)
from adaptive_mini_gpt.composition.hybrid_router import (
    HybridRoutingEngine,
    HybridRoutingEvaluation,
    SmallNeuralRouter,
)
from adaptive_mini_gpt.composition.relational_world import (
    DeductionResult,
    DeterministicGraphSolver,
    RelationalFact,
    RelationSemantics,
    SyntheticRelationalUniverse,
)

__all__ = [
    "RelationSemantics",
    "RelationalFact",
    "DeductionResult",
    "DeterministicGraphSolver",
    "SyntheticRelationalUniverse",
    "CompositionalTask",
    "M40_COMPOSITIONAL_BENCHMARK",
    "M40_CROSS_MODULE_BENCHMARK",
    "SEQUENTIAL_RELATIONAL_EPISODES",
    "M39AuditRecord",
    "M39BaselineAuditor",
    "CompositionalArmResult",
    "CompositionTrainer",
    "SmallNeuralRouter",
    "HybridRoutingEvaluation",
    "HybridRoutingEngine",
    "SequentialRelationalReport",
    "SequentialRelationalEvaluator",
]
