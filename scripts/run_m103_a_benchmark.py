#!/usr/bin/env python3
"""Milestone M103-A CLI Benchmark Runner.

Usage:
    python scripts/run_m103_a_benchmark.py --seeds 42 43 44 45 46 --tier tier_20k --output-dir runs/m103_a_output
"""

import argparse
from pathlib import Path
import sys
import torch

from adaptive_mini_gpt.integrated_autonomous_student.structural_evolution_engine import (
    CapacityTierM100,
)
from adaptive_mini_gpt.natural_language_learning_m103.benchmark_m103 import (
    M103NaturalLanguageBenchmark,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Milestone M103-A Natural-Language Learning Benchmark"
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[42, 43, 44, 45, 46],
        help="Random seeds to evaluate",
    )
    parser.add_argument(
        "--tier",
        type=str,
        default="tier_20k",
        choices=["tier_20k", "tier_30k", "tier_50k", "tier_100k", "tier_adaptive"],
        help="Capacity tier",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="runs/m103_a_output",
        help="Directory to save report artifacts",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tier = CapacityTierM100(args.tier)

    report = M103NaturalLanguageBenchmark.run_benchmark(
        device=device,
        seeds=args.seeds,
        tier=tier,
        output_dir=args.output_dir,
    )

    print(f"\n[+] M103-A Benchmark Completed with Status: {report.status}")
    out_path = Path(args.output_dir)
    print(f"[+] Artifacts written to {out_path.resolve()}")
    if report.status == "M103_FAIL":
        sys.exit(1)


if __name__ == "__main__":
    main()
