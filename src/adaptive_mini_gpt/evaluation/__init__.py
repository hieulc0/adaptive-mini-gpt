"""Evaluation modules for Adaptive Mini-GPT."""

from adaptive_mini_gpt.evaluation.evaluator import (
    evaluate_arithmetic_accuracy,
    evaluate_dataset,
    generate_sample,
)
from adaptive_mini_gpt.evaluation.multi_domain import (
    BenchmarkSample,
    BenchmarkSplit,
    DomainResult,
    DomainType,
    MultiDomainBenchmark,
    MultiDomainReport,
)

__all__ = [
    "evaluate_arithmetic_accuracy",
    "evaluate_dataset",
    "generate_sample",
    "BenchmarkSample",
    "BenchmarkSplit",
    "DomainResult",
    "DomainType",
    "MultiDomainBenchmark",
    "MultiDomainReport",
]
