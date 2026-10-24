"""Milestone M82: Learned Meta-Acquisition and Self-Improving Training Strategies Package."""

from adaptive_mini_gpt.meta_acquisition.audit_m81 import (
    M81ForensicAuditReport,
    M81ForensicAuditor,
)
from adaptive_mini_gpt.meta_acquisition.benchmark import (
    BenchmarkArm,
    M82MetaAcquisitionBenchmark,
)
from adaptive_mini_gpt.meta_acquisition.controller import (
    AcquisitionAction,
    AcquisitionState,
    InitChoice,
    LearnedAcquisitionController,
    LRChoice,
    RankChoice,
    StoppingChoice,
)
from adaptive_mini_gpt.meta_acquisition.environment import (
    MetaEnvironmentGenerator,
    MetaTaskDataset,
    SplitType,
    TaskExample,
    TaskFamily,
    apply_procedural_rule,
)
from adaptive_mini_gpt.meta_acquisition.meta_trainer import (
    AdaptationEpisodeEngine,
    EpisodeOutcome,
    MetaPolicyTrainer,
)

__all__ = [
    "M81ForensicAuditReport",
    "M81ForensicAuditor",
    "BenchmarkArm",
    "M82MetaAcquisitionBenchmark",
    "AcquisitionAction",
    "AcquisitionState",
    "InitChoice",
    "LearnedAcquisitionController",
    "LRChoice",
    "RankChoice",
    "StoppingChoice",
    "MetaEnvironmentGenerator",
    "MetaTaskDataset",
    "SplitType",
    "TaskExample",
    "TaskFamily",
    "apply_procedural_rule",
    "AdaptationEpisodeEngine",
    "EpisodeOutcome",
    "MetaPolicyTrainer",
]
