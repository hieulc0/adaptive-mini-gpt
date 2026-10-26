#!/usr/bin/env python3
"""Milestone M103-F / M103-E.1 CLI Benchmark Runner: Reusable Concepts & Meta-Transfer.

Usage:
    python scripts/run_m103_f_benchmark.py --output-dir runs/m103_f_output --steps 300
"""

import argparse
from pathlib import Path
import sys
import torch

from adaptive_mini_gpt.natural_language_learning_m103.benchmark_m103_f import (
    BenchmarkRunnerM103_F,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Milestone M103-F / M103-E.1 Reusable Concepts Benchmark"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="runs/m103_f_output",
        help="Directory to save report artifacts",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=300,
        help="Training step budget for single-rule optimization",
    )
    return parser.parse_args()


def main() -> None:
    from adaptive_mini_gpt.training.legacy_entrypoints import require_historical_reproduction

    require_historical_reproduction("run_m103_f_benchmark.py")
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Executing M103-F Benchmark on device: {device}")

    runner = BenchmarkRunnerM103_F(
        device=device,
        output_dir=args.output_dir,
        train_steps=args.steps,
    )
    report = runner.run_benchmark()

    print(f"\n[+] M103-F Benchmark Completed with Status: {report.overall_status}")
    print(f"[+] Gates Passed: {report.gates_passed_count}/{report.total_gates}")
    print(f"[+] Output directory: {args.output_dir}")


if __name__ == "__main__":
    main()
