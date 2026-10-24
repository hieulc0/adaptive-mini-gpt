"""Milestone M79: Autonomous Active Experimentation and Neural Skill-Gap Recognition."""

from adaptive_mini_gpt.active_experimentation.active_experiment_engine import (
    ActiveExperimentEngine,
    ActiveExperimentResult,
    ActiveProbeRecord,
    DiscriminativeProbeGenerator,
    EpistemicState,
    ProbeSelectionStrategy,
    VerifiedTeacherInterface,
)
from adaptive_mini_gpt.active_experimentation.audit_m78 import (
    M78AuditReport,
    M78ForensicAuditor,
)
from adaptive_mini_gpt.active_experimentation.causal_active_benchmark import (
    ActiveArmSummary,
    CausalActiveBenchmark,
    M79BenchmarkReport,
)

__all__ = [
    "M78AuditReport",
    "M78ForensicAuditor",
    "EpistemicState",
    "ProbeSelectionStrategy",
    "ActiveProbeRecord",
    "ActiveExperimentResult",
    "VerifiedTeacherInterface",
    "DiscriminativeProbeGenerator",
    "ActiveExperimentEngine",
    "ActiveArmSummary",
    "CausalActiveBenchmark",
    "M79BenchmarkReport",
]
