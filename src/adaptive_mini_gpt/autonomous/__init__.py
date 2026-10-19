"""Autonomous learning and adaptive growth controller package (M8-M13)."""

from adaptive_mini_gpt.autonomous.controller import (
    ControllerJournal,
    ControllerState,
    GrowthAction,
    LearningController,
    LearningPlan,
)
from adaptive_mini_gpt.autonomous.tasks import (
    TASK_REGISTRY,
    TaskSpecification,
    evaluate_symbolic_diagnostics,
    evaluate_symbolic_reversal_breakdown,
    evaluate_task_holdout,
)

__all__ = [
    "ControllerJournal",
    "ControllerState",
    "GrowthAction",
    "LearningController",
    "LearningPlan",
    "TASK_REGISTRY",
    "TaskSpecification",
    "evaluate_symbolic_diagnostics",
    "evaluate_symbolic_reversal_breakdown",
    "evaluate_task_holdout",
]
