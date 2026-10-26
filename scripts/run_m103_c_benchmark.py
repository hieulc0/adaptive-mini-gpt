#!/usr/bin/env python3
"""Milestone M103-C CLI Benchmark Runner: Generalizable Concept Learning.

Usage:
    python scripts/run_m103_c_benchmark.py --output-dir runs/m103_c_output --steps 150
"""

import argparse
from pathlib import Path
import sys
import torch

from adaptive_mini_gpt.natural_language_learning_m103.benchmark_m103_c import (
    BenchmarkRunnerM103_C,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Milestone M103-C Generalizable Concept Learning Benchmark"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="runs/m103_c_output",
        help="Directory to save report artifacts",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=150,
        help="Training step budget per concept",
    )
    parser.add_argument(
        "--eval-interval",
        type=int,
        default=30,
        help="Interval for logging intermediate evaluation metrics",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Executing M103-C Benchmark on device: {device}")

    runner = BenchmarkRunnerM103_C(
        device=device,
        output_dir=args.output_dir,
        train_steps=args.steps,
        eval_interval=args.eval_interval,
    )
    report = runner.run_benchmark()

    print(f"\n[+] M103-C Benchmark Completed with Status: {report.overall_status}")
    print(f"[+] Gates Passed: {report.gates_passed_count}/{report.total_gates}")
    print(f"[+] Output directory: {args.output_dir}")


if __name__ == "__main__":
    main()
