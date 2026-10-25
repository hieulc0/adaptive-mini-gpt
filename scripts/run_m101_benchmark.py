#!/usr/bin/env python3
"""CLI entrypoint for Milestone M101 Master Scientific Benchmark Runner."""

from __future__ import annotations

import argparse
import sys

import torch

from adaptive_mini_gpt.integrated_autonomous_student.structural_evolution_engine import (
    CapacityTierM100,
)
from adaptive_mini_gpt.sustainable_lifelong_learning.benchmark_m101 import (
    M101SustainableLifelongBenchmark,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Milestone M101 Scientific Benchmark.")
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Compute device (e.g. cuda, cpu)",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[42, 43, 44, 45, 46],
        help="List of random seeds to evaluate",
    )
    parser.add_argument(
        "--tier",
        type=str,
        default="tier_20k",
        choices=["tier_20k", "tier_30k", "tier_50k", "tier_100k", "tier_adaptive"],
        help="Capacity tier",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=100,
        help="Lifelong learning episodes per seed",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="runs/m101_output",
        help="Output directory for json and markdown reports",
    )

    args = parser.parse_args()
    device = torch.device(args.device)
    tier = CapacityTierM100(args.tier)

    print(f"=== Milestone M101 Benchmark Runner (Device: {device} | Tier: {tier.value} | Episodes: {args.episodes}) ===")
    report = M101SustainableLifelongBenchmark.run_benchmark(
        device=device,
        seeds=args.seeds,
        output_dir=args.output_dir,
        capacity_tier=tier,
        episodes_per_seed=args.episodes,
    )

    if report["status"] != "M101_QUALIFIED_PASS":
        print(f"\n[ERROR] Benchmark status: {report['status']}")
        sys.exit(1)

    print(f"\n[SUCCESS] Milestone M101 Certified: {report['status']}")
    sys.exit(0)


if __name__ == "__main__":
    main()
