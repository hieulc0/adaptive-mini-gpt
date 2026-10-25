"""Milestone M88: Foundational Representation Learning and Lifelong Knowledge Retention.

Exports:
- M87ForensicAuditor, M87AuditReport
- FoundationalDomain, FoundationalCurriculumGenerator, apply_foundational_rule
- SharedKnowledgeTrunk, SelectiveResidualAdapter, SharedFoundationalSubstrate
- ExemplarReplayBuffer, SharedBoundedReplayLearner, SharedSelectiveResidualsLearner,
  SharedNoReplayLearner, BudgetMatchedIndependentLearner, UnconstrainedOracleLearner
- FoundationalContinualStudent
- M88FoundationalBenchmark
"""

from __future__ import annotations

from adaptive_mini_gpt.foundational.audit_m87 import (
    M87AuditReport,
    M87ForensicAuditor,
)
from adaptive_mini_gpt.foundational.benchmark import (
    ArmBenchmarkResult,
    M88FoundationalBenchmark,
)
from adaptive_mini_gpt.foundational.continual_mechanisms import (
    BudgetMatchedIndependentLearner,
    ExemplarReplayBuffer,
    SharedBoundedReplayLearner,
    SharedNoReplayLearner,
    SharedSelectiveResidualsLearner,
    UnconstrainedOracleLearner,
)
from adaptive_mini_gpt.foundational.curriculum import (
    FoundationalCurriculumGenerator,
    FoundationalDomain,
    apply_foundational_rule,
)
from adaptive_mini_gpt.foundational.shared_representation import (
    SelectiveResidualAdapter,
    SharedFoundationalSubstrate,
    SharedKnowledgeTrunk,
)
from adaptive_mini_gpt.foundational.student import (
    FoundationalContinualStudent,
)

__all__ = [
    "M87ForensicAuditor",
    "M87AuditReport",
    "FoundationalDomain",
    "FoundationalCurriculumGenerator",
    "apply_foundational_rule",
    "SharedKnowledgeTrunk",
    "SelectiveResidualAdapter",
    "SharedFoundationalSubstrate",
    "ExemplarReplayBuffer",
    "SharedBoundedReplayLearner",
    "SharedSelectiveResidualsLearner",
    "SharedNoReplayLearner",
    "BudgetMatchedIndependentLearner",
    "UnconstrainedOracleLearner",
    "FoundationalContinualStudent",
    "M88FoundationalBenchmark",
    "ArmBenchmarkResult",
]
