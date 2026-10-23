"""Package exports for safe meta-learning and evidence-based learning decisions (M61)."""
from __future__ import annotations

from adaptive_mini_gpt.safe_meta_learning.audit import (
    M60ForensicAuditReport,
    M60ScientificAuditor,
)
from adaptive_mini_gpt.safe_meta_learning.controller import (
    ControllerAction,
    SafeControlDecision,
    SafeLearningProgressController,
)
from adaptive_mini_gpt.safe_meta_learning.ablation import (
    M61AblationArm,
    M61AdaptationResult,
    SafeCausalAdaptationEngine,
    build_m61_memory_store,
)
from adaptive_mini_gpt.safe_meta_learning.routing import (
    CounterfactualQueryOutcome,
    CounterfactualRoutingReport,
    CounterfactualSkillRouter,
)
from adaptive_mini_gpt.safe_meta_learning.memory_update import (
    DynamicExperienceMemoryUpdater,
    ExperienceUpdateReport,
)
from adaptive_mini_gpt.safe_meta_learning.safeguards import (
    M61RetentionAndSafeguardReport,
    M61SafeguardAuditor,
)

__all__ = [
    "M60ForensicAuditReport",
    "M60ScientificAuditor",
    "ControllerAction",
    "SafeControlDecision",
    "SafeLearningProgressController",
    "M61AblationArm",
    "M61AdaptationResult",
    "SafeCausalAdaptationEngine",
    "build_m61_memory_store",
    "CounterfactualQueryOutcome",
    "CounterfactualRoutingReport",
    "CounterfactualSkillRouter",
    "DynamicExperienceMemoryUpdater",
    "ExperienceUpdateReport",
    "M61RetentionAndSafeguardReport",
    "M61SafeguardAuditor",
]
