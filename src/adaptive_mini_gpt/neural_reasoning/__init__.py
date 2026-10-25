"""Milestone M91: Autonomous Multi-Step Neural Reasoning and Persistent Concept Reuse."""

from __future__ import annotations

from adaptive_mini_gpt.neural_reasoning.audit_m90 import (
    M90ForensicAuditReport,
    M90ForensicAuditor,
)
from adaptive_mini_gpt.neural_reasoning.benchmark_m91 import (
    M91BenchmarkReport,
    M91ReasoningBenchmark,
    SeedReasoningResult,
)
from adaptive_mini_gpt.neural_reasoning.persistent_substrate import (
    CompressedResidual,
    PersistentReasoningSubstrate,
)
from adaptive_mini_gpt.neural_reasoning.reasoning_curriculum import (
    M91ReasoningCurriculumGenerator,
    M91ReasoningSuite,
    ReasoningPartition,
)
from adaptive_mini_gpt.neural_reasoning.reasoning_engine import (
    ChainExecutionResult,
    EvaluationSummary,
    ReasoningEngine,
)
from adaptive_mini_gpt.neural_reasoning.reasoning_grammar import (
    MultiStepEvaluator,
    MultiStepProblem,
    StepRecord,
)

__all__ = [
    "ChainExecutionResult",
    "CompressedResidual",
    "EvaluationSummary",
    "M90ForensicAuditReport",
    "M90ForensicAuditor",
    "M91BenchmarkReport",
    "M91ReasoningBenchmark",
    "M91ReasoningCurriculumGenerator",
    "M91ReasoningSuite",
    "MultiStepEvaluator",
    "MultiStepProblem",
    "PersistentReasoningSubstrate",
    "ReasoningEngine",
    "ReasoningPartition",
    "SeedReasoningResult",
    "StepRecord",
]
