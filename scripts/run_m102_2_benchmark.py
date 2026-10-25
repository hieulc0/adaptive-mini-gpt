"""Milestone M102.2 CLI Benchmark Runner."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

from adaptive_mini_gpt.integrated_autonomous_student.structural_evolution_engine import (
    CapacityTierM100,
)
from adaptive_mini_gpt.scientific_requalification_m102_2.benchmark_m102_2 import (
    M102_2ScientificRequalificationBenchmark,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run Milestone M102.2 Scientific Requalification Benchmark"
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[42, 43, 44, 45, 46],
        help="Random seeds for lifelong multi-seed evaluation",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="runs/m102_2_output",
        help="Directory to save report artifacts",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=100,
        help="Number of continuous learning episodes per seed",
    )
    parser.add_argument(
        "--tier",
        type=str,
        default="tier_20k",
        choices=["tier_20k", "tier_30k", "tier_50k", "tier_100k", "tier_adaptive"],
        help="Baseline capacity tier",
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[+] Initializing M102.2 Scientific Requalification Benchmark on {device}...")

    report = M102_2ScientificRequalificationBenchmark.run_benchmark(
        device=device,
        seeds=args.seeds,
        output_dir=args.output_dir,
        capacity_tier=CapacityTierM100(args.tier),
        episodes_per_seed=args.episodes,
    )

    status = report.get("status", "FAILED")
    print(f"\n[+] M102.2 Benchmark Completed with Status: {status}")
    if "PASS" in status:
        sys.exit(0)
    else:
        sys.exit(1)


if __name__ == "__main__":
    main()
