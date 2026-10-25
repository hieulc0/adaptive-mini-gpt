"""Milestone M95: Cross-Domain Neural Knowledge Transfer and Reusable Abstraction Learning."""

from adaptive_mini_gpt.cross_domain_transfer.audit_m94 import (
    M94AuditResult,
    M94ForensicAuditor,
)
from adaptive_mini_gpt.cross_domain_transfer.benchmark_m95 import (
    M95BenchmarkReport,
    M95CrossDomainBenchmark,
    SeedM95Result,
)
from adaptive_mini_gpt.cross_domain_transfer.cross_domain_curriculum import (
    ComputationalConcept,
    CrossDomainCurriculumGenerator,
    CrossDomainExample,
    CrossDomainTokenizer,
    Domain,
)
from adaptive_mini_gpt.cross_domain_transfer.cross_domain_learner import (
    AdaptationTrajectory,
    CrossDomainLearner,
)
from adaptive_mini_gpt.cross_domain_transfer.neural_cross_domain_transfer import (
    CrossDomainExecutionResult,
    CrossDomainReasoningSubstrate,
    CrossDomainTransferEngine,
    ModularRank2Residual,
)

__all__ = [
    "M94AuditResult",
    "M94ForensicAuditor",
    "M95BenchmarkReport",
    "M95CrossDomainBenchmark",
    "SeedM95Result",
    "ComputationalConcept",
    "CrossDomainCurriculumGenerator",
    "CrossDomainExample",
    "CrossDomainTokenizer",
    "Domain",
    "AdaptationTrajectory",
    "CrossDomainLearner",
    "CrossDomainExecutionResult",
    "CrossDomainReasoningSubstrate",
    "CrossDomainTransferEngine",
    "ModularRank2Residual",
]
