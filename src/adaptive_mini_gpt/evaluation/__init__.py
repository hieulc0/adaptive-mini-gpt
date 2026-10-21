"""Evaluation modules for Adaptive Mini-GPT."""

from __future__ import annotations

try:
    from adaptive_mini_gpt.evaluation.evaluator import (
        evaluate_arithmetic_accuracy,
        evaluate_dataset,
        generate_sample,
    )
except ImportError:
    evaluate_arithmetic_accuracy = None  # type: ignore
    evaluate_dataset = None  # type: ignore
    generate_sample = None  # type: ignore

try:
    from adaptive_mini_gpt.evaluation.multi_domain import (
        BenchmarkSample,
        BenchmarkSplit,
        DomainResult,
        DomainType,
        MultiDomainBenchmark,
        MultiDomainReport,
    )
except ImportError:
    BenchmarkSample = None  # type: ignore
    BenchmarkSplit = None  # type: ignore
    DomainResult = None  # type: ignore
    DomainType = None  # type: ignore
    MultiDomainBenchmark = None  # type: ignore
    MultiDomainReport = None  # type: ignore

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
