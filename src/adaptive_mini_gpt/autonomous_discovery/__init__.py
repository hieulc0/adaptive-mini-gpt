"""Milestone M92: Autonomous Rule Discovery, Active Experimentation, and Persistent Neural Knowledge."""

from __future__ import annotations

from adaptive_mini_gpt.autonomous_discovery.active_discovery_engine import (
    ActiveDiscoveryEngine,
    DiscoveryOutcome,
    DiscoveryStepLog,
    HypothesisSpace,
)
from adaptive_mini_gpt.autonomous_discovery.audit_m91 import (
    M91ForensicAuditReport,
    M91ForensicAuditor,
)
from adaptive_mini_gpt.autonomous_discovery.benchmark_m92 import (
    ArmBenchmarkResult,
    M92BenchmarkReport,
    M92DiscoveryBenchmark,
    SeedM92Result,
)
from adaptive_mini_gpt.autonomous_discovery.neural_rule_learner import (
    NeuralAcquisitionResult,
    NeuralRuleLearner,
)
from adaptive_mini_gpt.autonomous_discovery.persistent_knowledge_manager import (
    CompositionTransferResult,
    PersistentKnowledgeManager,
    RetentionEvaluationReport,
)
from adaptive_mini_gpt.autonomous_discovery.unknown_rule_environment import (
    ProceduralRule,
    RuleObservation,
    UnknownRuleEnvironment,
    UnknownRuleTask,
)

__all__ = [
    "ActiveDiscoveryEngine",
    "ArmBenchmarkResult",
    "CompositionTransferResult",
    "DiscoveryOutcome",
    "DiscoveryStepLog",
    "HypothesisSpace",
    "M91ForensicAuditReport",
    "M91ForensicAuditor",
    "M92BenchmarkReport",
    "M92DiscoveryBenchmark",
    "NeuralAcquisitionResult",
    "NeuralRuleLearner",
    "PersistentKnowledgeManager",
    "ProceduralRule",
    "RetentionEvaluationReport",
    "RuleObservation",
    "SeedM92Result",
    "UnknownRuleEnvironment",
    "UnknownRuleTask",
]
