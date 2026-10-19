"""Evaluation modules for Adaptive Mini-GPT."""

from adaptive_mini_gpt.evaluation.evaluator import (
    evaluate_arithmetic_accuracy,
    evaluate_dataset,
    generate_sample,
)

__all__ = ["evaluate_arithmetic_accuracy", "evaluate_dataset", "generate_sample"]
