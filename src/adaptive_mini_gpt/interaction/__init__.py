"""Closed-loop interaction and knowledge-gap-driven learning package (M29).

Implements the 6-step continual self-learning loop:
  INTERACT -> ATTEMPT -> VERIFY -> IDENTIFY GAP -> PRACTICE -> TRAIN CANDIDATE -> REEVALUATE -> PROMOTE/ROLLBACK
"""

from __future__ import annotations

from adaptive_mini_gpt.interaction.action_selector import (
    ActionDecision,
    InteractionActionSelector,
    LearningAction,
)
from adaptive_mini_gpt.interaction.knowledge_gap import (
    ErrorCategory,
    KnowledgeGapRecord,
    KnowledgeGapTracker,
)
from adaptive_mini_gpt.interaction.loop import ClosedLoopInteractiveLearner
from adaptive_mini_gpt.interaction.verified_memory import (
    VerifiedMemoryBank,
    VerifiedMemoryEntry,
)

__all__ = [
    "ActionDecision",
    "ClosedLoopInteractiveLearner",
    "ErrorCategory",
    "InteractionActionSelector",
    "KnowledgeGapRecord",
    "KnowledgeGapTracker",
    "LearningAction",
    "VerifiedMemoryBank",
    "VerifiedMemoryEntry",
]
