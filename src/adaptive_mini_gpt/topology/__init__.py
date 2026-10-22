"""M38: Controlled Neural Structural Evolution Package."""

from adaptive_mini_gpt.topology.benchmark_suite import (
    CONCEPT_TRANSFER_TASKS,
    PARAPHRASED_FACTUAL_TASKS,
    SEQUENTIAL_EPISODES,
    SOURCE_FACTUAL_TASKS,
    UNSEEN_ARITHMETIC_TASKS,
    UNSEEN_LOGIC_TASKS,
    TopologyBenchmarkTask,
    generate_carry_retention_suite,
)
from adaptive_mini_gpt.topology.gated_skip import (
    ArchitectureSpec,
    GatedSkipConnection,
    TopologyEvolutionManager,
)
from adaptive_mini_gpt.topology.structural_evaluator import (
    ArmEvaluationResult,
    ControlledStructuralEvaluator,
    RetentionMatrixReport,
)

__all__ = [
    "ArchitectureSpec",
    "GatedSkipConnection",
    "TopologyEvolutionManager",
    "TopologyBenchmarkTask",
    "UNSEEN_ARITHMETIC_TASKS",
    "UNSEEN_LOGIC_TASKS",
    "SOURCE_FACTUAL_TASKS",
    "PARAPHRASED_FACTUAL_TASKS",
    "CONCEPT_TRANSFER_TASKS",
    "SEQUENTIAL_EPISODES",
    "generate_carry_retention_suite",
    "ArmEvaluationResult",
    "RetentionMatrixReport",
    "ControlledStructuralEvaluator",
]
