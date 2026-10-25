#!/usr/bin/env python3
"""CLI Entrypoint for Milestone M100 Master Benchmark.

Usage:
    python3 scripts/run_m100_benchmark.py --seeds 42 43 44 45 46 --device cuda --output-dir runs/m100_output --tier tier_20k
"""

from __future__ import annotations

import argparse
import sys

import torch

from adaptive_mini_gpt.integrated_autonomous_student.benchmark_m100 import (
    M100IntegratedAutonomousStudentBenchmark,
)
from adaptive_mini_gpt.integrated_autonomous_student.structural_evolution_engine import (
    CapacityTierM100,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Milestone M100 Benchmark Runner")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44, 45, 46])
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output-dir", type=str, default="runs/m100_output")
    parser.add_argument(
        "--tier",
        type=str,
        default="tier_20k",
        choices=["tier_20k", "tier_30k", "tier_50k", "tier_100k", "tier_adaptive"],
    )

    args = parser.parse_args()

    device = torch.device(args.device)
    tier = CapacityTierM100(args.tier)
    print(f"=== Milestone M100 Benchmark Runner (Device: {device} | Tier: {tier.value}) ===")

    report = M100IntegratedAutonomousStudentBenchmark.run_benchmark(
        device=device,
        seeds=args.seeds,
        output_dir=args.output_dir,
        capacity_tier=tier,
    )

    if report["status"] != "M100_QUALIFIED_PASS":
        print(f"\n[ERROR] Benchmark status: {report['status']}")
        sys.exit(1)
    else:
        print("\n[SUCCESS] Milestone M100 Qualified Pass certified.")
        sys.exit(0)


if __name__ == "__main__":
    main()
