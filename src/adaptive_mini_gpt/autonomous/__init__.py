"""Autonomous learning, adaptive growth, and curriculum selection package (M8-M26)."""

from adaptive_mini_gpt.autonomous.controller import (
    ControllerJournal,
    ControllerState,
    GrowthAction,
    LearningController,
    LearningPlan,
)
from adaptive_mini_gpt.autonomous.curriculum_selector import (
    CurriculumProposal,
    CurriculumSelector,
    CurriculumStrategy,
    CycleResult,
    InterventionType,
)
from adaptive_mini_gpt.autonomous.lineage_registry import (
    CANONICAL_LINEAGE_REGISTRY,
    LineageRecord,
    LineageRole,
    get_candidate_root_lineage,
    verify_checkpoint_integrity,
)
from adaptive_mini_gpt.autonomous.tasks import (
    TASK_REGISTRY,
    TaskSpecification,
    evaluate_symbolic_diagnostics,
    evaluate_symbolic_reversal_breakdown,
    evaluate_task_holdout,
)

__all__ = [
    "CANONICAL_LINEAGE_REGISTRY",
    "ControllerJournal",
    "ControllerState",
    "CurriculumProposal",
    "CurriculumSelector",
    "CurriculumStrategy",
    "CycleResult",
    "GrowthAction",
    "InterventionType",
    "LearningController",
    "LearningPlan",
    "LineageRecord",
    "LineageRole",
    "TASK_REGISTRY",
    "TaskSpecification",
    "evaluate_symbolic_diagnostics",
    "evaluate_symbolic_reversal_breakdown",
    "evaluate_task_holdout",
    "get_candidate_root_lineage",
    "verify_checkpoint_integrity",
]
