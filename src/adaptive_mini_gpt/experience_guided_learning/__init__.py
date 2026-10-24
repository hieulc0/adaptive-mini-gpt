"""Milestone M75: Experience-Guided Autonomous Learning and Adaptive Training Budgets.

Exports:
- M74ScientificAuditor: Forensic scientific audit, shuffled-target analysis, paired Student's t intervals.
- LearningExperienceRecord, LearningExperienceManifest: Verified learning-history records and manifests.
- ExperienceGuidedController, ControllerPolicyMode: Conservative experience-aware training controller.
- ProceduralRuleFamily, ProceduralTaskGenerator, GeneralizationCategory: 4-tier procedural tasks.
- CausalExperienceBenchmark: 7-arm causal lifelong evaluation engine.
- M75SafeguardsAuditor: Double SHA-256 base invariance, arithmetic/logic retention, rollback, capacity controls.
"""

from __future__ import annotations

from adaptive_mini_gpt.experience_guided_learning.audit_m74 import (
    M74AuditRecord,
    M74ScientificAuditor,
    PairedStatisticalInterval,
)
from adaptive_mini_gpt.experience_guided_learning.causal_experience_benchmark import (
    CausalEpisodeResult,
    CausalExperienceBenchmark,
    PolicyCausalSummary,
)
from adaptive_mini_gpt.experience_guided_learning.experience_controller import (
    BudgetRecommendation,
    ControllerPolicyMode,
    ExperienceGuidedController,
)
from adaptive_mini_gpt.experience_guided_learning.experience_records import (
    LearningAction,
    LearningActionType,
    LearningExperienceManifest,
    LearningExperienceRecord,
    ObservableTaskDescriptors,
    RetentionDecision,
)
from adaptive_mini_gpt.experience_guided_learning.safeguards import M75SafeguardsAuditor
from adaptive_mini_gpt.experience_guided_learning.task_stream import (
    GeneralizationCategory,
    ProceduralExample,
    ProceduralRuleFamily,
    ProceduralTaskDataset,
    ProceduralTaskGenerator,
    apply_procedural_rule,
)

__all__ = [
    "BudgetRecommendation",
    "CausalEpisodeResult",
    "CausalExperienceBenchmark",
    "ControllerPolicyMode",
    "ExperienceGuidedController",
    "GeneralizationCategory",
    "LearningAction",
    "LearningActionType",
    "LearningExperienceManifest",
    "LearningExperienceRecord",
    "M74AuditRecord",
    "M74ScientificAuditor",
    "M75SafeguardsAuditor",
    "ObservableTaskDescriptors",
    "PairedStatisticalInterval",
    "PolicyCausalSummary",
    "ProceduralExample",
    "ProceduralRuleFamily",
    "ProceduralTaskDataset",
    "ProceduralTaskGenerator",
    "RetentionDecision",
    "apply_procedural_rule",
]
